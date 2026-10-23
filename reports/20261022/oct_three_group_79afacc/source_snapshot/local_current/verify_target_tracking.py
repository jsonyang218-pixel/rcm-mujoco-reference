"""Fast regressions for bounded measured-target memory; no renderer/navigation.

Run with ``.venv/bin/python verify_target_tracking.py``. The JSON report records
executed unittest outcomes, including failures, and the process preserves their
exit status. Synthetic observations stand in for sensor output, never scene
coordinates; raw RGB/depth/segmentation are checked for mutation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from rcm_target_tracking import TargetMemory


def observation(target=(1., 2., 3.), tip=(1., 2., 2.8), *,
                target_quality=.001, tip_quality=.001, visible=True):
    geometry = {}
    if target is not None:
        geometry['target_world_mm'] = list(target)
        geometry['target_quality'] = {'fit_quality_mm': target_quality,
                                      'source': 'measured_target_plane'}
    else:
        geometry['target_error'] = 'TARGET_NOT_VISIBLE'
    if tip is not None:
        geometry['tip_world_mm'] = list(tip)
        geometry['tip_quality'] = {'fit_quality_mm': tip_quality,
                                   'source': 'measured_shaft_end_plane'}
    else:
        geometry['tip_error'] = 'INSUFFICIENT_END_FACE_PIXELS'
    return {'geometry': geometry, 'target_pixel': [30., 40.] if visible else None,
            'rgb': np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            'depth': np.full((2, 3), .02, dtype=np.float32),
            'segmentation': np.arange(12, dtype=np.int32).reshape(2, 3, 2)}


class TargetTrackingChecks(unittest.TestCase):
    def register(self, memory=None, **kwargs):
        memory = TargetMemory() if memory is None else memory
        for time in (0., .05, .1):
            memory.resolve(observation(**kwargs), time)
        self.assertTrue(memory.cache_ready)
        return memory

    def rejected(self, memory, raw, time, reason):
        with self.assertRaisesRegex(RuntimeError, '^'+reason+'$'):
            memory.resolve(raw, time)
        self.assertEqual(memory.last_result['rejection_reason'], reason)
        self.assertIsNone(memory.last_result['used_target_world_mm'])

    def test_three_distinct_stable_observations_mature_cache(self):
        memory = TargetMemory()
        for time in (0., .05):
            result = memory.resolve(observation(), time)
            self.assertEqual(result['target_source'], 'observed')
            self.assertFalse(memory.cache_ready)
        result = memory.resolve(observation(), .1)
        self.assertTrue(result['cache_ready'])
        self.assertEqual(result['registration_observations'], 3)

    def test_duplicate_timestamps_do_not_register_or_allow_tip_motion(self):
        memory = TargetMemory()
        for _ in range(5):
            result = memory.resolve(observation(), 0.)
        self.assertFalse(memory.cache_ready)
        self.assertEqual(result['registration_observations'], 1)
        self.rejected(memory, observation(tip=(1., 2., 2.81)), 0.,
                      'TARGET_TRACK_DUPLICATE_TIME_WITH_MOTION')
        self.rejected(memory, observation(target=None, visible=False), .05,
                      'TARGET_MEMORY_NOT_READY')

    def test_unstable_samples_and_missing_frame_break_registration(self):
        memory = TargetMemory()
        for time, point in zip((0., .05, .1),
                               ((1., 2., 3.), (1.02, 2., 3.), (1., 2., 3.))):
            memory.resolve(observation(target=point), time)
        self.assertFalse(memory.cache_ready)
        self.rejected(memory, observation(target=None, visible=False), .15,
                      'TARGET_MEMORY_NOT_READY')
        for time in (.2, .25):
            memory.resolve(observation(), time)
            self.assertFalse(memory.cache_ready)
        memory.resolve(observation(), .3)
        self.assertTrue(memory.cache_ready)

    def test_short_missing_target_uses_cache_and_fresh_tip(self):
        memory = self.register()
        raw = observation(target=None, tip=(1., 2., 2.9), visible=False)
        result = memory.resolve(raw, .3)
        self.assertEqual(result['target_source'], 'registered_memory')
        self.assertFalse(result['target_observed'])
        self.assertAlmostEqual(result['cached_target_age_s'], .2)
        self.assertAlmostEqual(result['target_memory_travel_mm'], .1)
        self.assertEqual(result['used_target_world_mm'], [1., 2., 3.])
        self.assertEqual(result['tip_world_mm'], raw['geometry']['tip_world_mm'])
        self.assertAlmostEqual(result['estimated_position_error_um'], 100.)
        self.assertGreater(result['target_quality']['fit_quality_mm'], .001)

    def test_memory_expires_after_one_second_and_accepts_boundary(self):
        memory = self.register()
        result = memory.resolve(observation(target=None, visible=False), 1.1)
        self.assertAlmostEqual(result['cached_target_age_s'], 1.)
        self.rejected(memory, observation(target=None, visible=False), 1.1001,
                      'TARGET_MEMORY_EXPIRED')

    def test_memory_travel_limit_uses_cumulative_distance(self):
        memory = self.register()
        first = memory.resolve(observation(target=None, tip=(1.3, 2., 2.8),
                                           visible=False), .15)
        self.assertAlmostEqual(first['target_memory_travel_mm'], .3)
        # The tip returns to its start, but its travelled distance is .6 mm.
        self.rejected(memory, observation(target=None, visible=False), .2,
                      'TARGET_MEMORY_TRAVEL_EXCEEDED')
        self.assertAlmostEqual(memory.last_result['target_memory_travel_mm'], .6)

    def test_reset_clears_previous_goal_and_identity_anchor(self):
        memory = self.register()
        memory.reset('P4')
        self.assertFalse(memory.cache_ready)
        self.rejected(memory, observation(target=None, visible=False), 0.,
                      'TARGET_MEMORY_UNAVAILABLE')
        for time in (.05, .1, .15):
            result = memory.resolve(observation(target=(2., 3., 4.)), time)
        self.assertTrue(memory.cache_ready)
        self.assertEqual(result['goal_id'], 'P4')
        self.assertEqual(result['used_target_world_mm'], [2., 3., 4.])

    def test_moved_target_invalidates_cache_until_goal_reset(self):
        memory = self.register()
        memory.resolve(observation(target=None, visible=False), .15)
        self.rejected(memory, observation(target=(1.041, 2., 3.)), .2,
                      'TARGET_MOVED_REPLAN_REQUIRED')
        self.assertAlmostEqual(memory.last_result['reacquisition_shift_mm'], .041)
        self.assertFalse(memory.cache_ready)
        memory.reacquire()
        self.rejected(memory, observation(), .25, 'TARGET_MOVED_REPLAN_REQUIRED')
        memory.reset('new-goal-after-replan')
        self.assertEqual(memory.resolve(observation(target=(1.041, 2., 3.)), .3)
                         ['target_source'], 'observed')

    def test_exact_movement_threshold_and_memory_travel_boundary(self):
        memory = self.register()
        self.assertEqual(memory.resolve(observation(target=(1.04, 2., 3.)), .15)
                         ['target_source'], 'observed')
        self.rejected(memory, observation(target=(1.04001, 2., 3.)), .2,
                      'TARGET_MOVED_REPLAN_REQUIRED')
        memory = self.register()
        result = memory.resolve(observation(target=None, visible=False,
                                             tip=(1.5, 2., 2.8)), .15)
        self.assertEqual(result['target_source'], 'registered_memory')
        self.assertAlmostEqual(result['target_memory_travel_mm'], .5)

    def test_reacquire_preserves_movement_anchor_and_needs_new_samples(self):
        memory = self.register()
        memory.reacquire()
        self.rejected(memory, observation(target=None, visible=False), .15,
                      'TARGET_MEMORY_NOT_READY')
        for time in (.2, .25):
            memory.resolve(observation(target=(1.005, 2., 3.)), time)
            self.assertFalse(memory.cache_ready)
        memory.resolve(observation(target=(1.005, 2., 3.)), .3)
        self.assertTrue(memory.cache_ready)
        # Movement is measured against the original registered target.
        self.rejected(memory, observation(target=(1.043, 2., 3.)), .35,
                      'TARGET_MOVED_REPLAN_REQUIRED')

    def test_low_quality_target_cannot_register_but_mature_cache_can_bridge(self):
        memory = TargetMemory()
        self.rejected(memory, observation(target_quality=.0101), 0.,
                      'TARGET_MEMORY_UNAVAILABLE')
        for time in (.05, .1, .15):
            memory.resolve(observation(), time)
        self.assertTrue(memory.cache_ready)
        result = memory.resolve(observation(target=(99., 99., 99.),
                                             target_quality=.0101), .2)
        self.assertEqual(result['target_source'], 'registered_memory')
        self.assertFalse(result['target_observed'])
        self.assertEqual(result['used_target_world_mm'], [1., 2., 3.])

    def test_memory_quality_budget_accounts_for_age_and_registration_spread(self):
        memory = self.register(target_quality=.0098)
        result = memory.resolve(observation(target=None, visible=False), .2)
        self.assertAlmostEqual(result['target_quality']['fit_quality_mm'], .0099)
        self.rejected(memory, observation(target=None, visible=False), .4,
                      'TARGET_MEMORY_LOW_QUALITY')
        spread_memory = TargetMemory()
        for time, x in zip((0., .05, .1), (1., 1.009, 1.)):
            spread_memory.resolve(observation(target=(x, 2., 3.), target_quality=.005), time)
        self.assertTrue(spread_memory.cache_ready)
        self.rejected(spread_memory, observation(target=None, visible=False), .15,
                      'TARGET_MEMORY_LOW_QUALITY')

    def test_current_tip_missing_or_low_quality_is_never_filled_from_cache(self):
        for kwargs in ({'tip': None}, {'tip_quality': .0101},
                       {'tip': (float('nan'), 0., 0.)}):
            with self.subTest(kwargs=kwargs):
                memory = self.register()
                self.rejected(memory, observation(target=None, visible=False, **kwargs),
                              .15, 'TIP_NOT_OBSERVABLE')
                self.assertNotIn('tip_world_mm', memory.last_result)

    def test_later_observation_disagreement_consumes_fixed_cache_budget(self):
        memory = self.register()
        # This stays below the movement/replan threshold, so a current good
        # measurement is usable. The older registered point cannot then be
        # treated as a 1 um measurement when that current target disappears.
        result = memory.resolve(observation(target=(1.02, 2., 3.)), .15)
        self.assertEqual(result['target_source'], 'observed')
        self.rejected(memory, observation(target=None, visible=False), .2,
                      'TARGET_MEMORY_LOW_QUALITY')
        memory = self.register()
        memory.resolve(observation(target=(1.004, 2., 3.)), .15)
        result = memory.resolve(observation(target=None, visible=False), .2)
        self.assertAlmostEqual(result['target_quality']['last_observation_disagreement_mm'], .004)
        self.assertAlmostEqual(result['target_quality']['fit_quality_mm'], .00505)

    def test_raw_arrays_geometry_and_returned_metadata_are_independent(self):
        memory = self.register()
        for raw, time in ((observation(), .15),
                          (observation(target=None, visible=False), .2)):
            original = deepcopy(raw)
            result = memory.resolve(raw, time)
            self.assertEqual(raw['geometry'], original['geometry'])
            self.assertEqual(raw['target_pixel'], original['target_pixel'])
            for key in ('rgb', 'depth', 'segmentation'):
                np.testing.assert_array_equal(raw[key], original[key])
            result['target_world_mm'][0] = -100.
            result['tip_world_mm'][0] = -100.
            result['target_quality']['fit_quality_mm'] = 999.
            self.assertNotEqual(memory.last_result['tip_world_mm'][0], -100.)
            self.assertNotEqual(memory.last_result['target_world_mm'][0], -100.)
        self.assertEqual(memory.resolve(observation(target=None, visible=False), .25)
                         ['used_target_world_mm'], [1., 2., 3.])

    def test_scene_like_fields_cannot_supply_missing_geometry(self):
        memory = TargetMemory()
        raw = observation(target=None, visible=False)
        raw.update(target_world_mm=[1., 2., 3.],
                   target_world_mm_privileged=[1., 2., 3.],
                   scene_target=[1., 2., 3.])
        self.rejected(memory, raw, 0., 'TARGET_MEMORY_UNAVAILABLE')
        memory = self.register()
        raw = observation(tip=None)
        raw['tip_world_mm'] = [1., 2., 2.8]
        self.rejected(memory, raw, .15, 'TIP_NOT_OBSERVABLE')

    def test_invalid_or_reversed_time_rejects(self):
        for invalid in (-1., float('nan'), float('inf'), None, 'bad-clock'):
            with self.subTest(time=invalid):
                self.rejected(TargetMemory(), observation(), invalid, 'TARGET_TRACK_TIME_INVALID')
        memory = self.register()
        self.rejected(memory, observation(), .09, 'TARGET_TRACK_TIME_REVERSED')


class ReportResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({'test': test.id(), 'status': 'PASS'})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({'test': test.id(), 'status': 'FAIL',
                              'detail': self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({'test': test.id(), 'status': 'ERROR',
                              'detail': self._exc_info_to_string(err, test)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('runs/multitarget_occlusion_20261022/occlusion_checks/target_tracking_checks.json'))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TargetTrackingChecks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReportResult).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
              'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped),
              'outcomes': result.outcomes,
              'measurement_contract': 'Only observation.geometry current measured target and tip; current tip required on every resolve.',
              'memory_limits': {'stable_observations': 3, 'age_s': 1., 'travel_mm': .5,
                                'reacquired_shift_mm': .04, 'fit_quality_mm': .01}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Report: {args.output}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
