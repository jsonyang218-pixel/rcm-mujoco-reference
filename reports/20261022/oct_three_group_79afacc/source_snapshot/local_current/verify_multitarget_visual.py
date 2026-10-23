"""Protocol/action-alignment regressions for the strict visual phase manager.

These deterministic controller doubles test orchestration, not visual accuracy.
Real navigation is exercised by the normal waypoint CLI separately.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

import rcm_multitarget_visual as navigation
from rcm_navigation import task_user_units
from rcm_waypoint_dataset import EpisodeWriter


def poison(*args, **kwargs):
    raise AssertionError('Privileged inverse/target-task must never generate strict motion')


class StubReference:
    """A deliberately simple measurement fixture; not a MuJoCo robot model."""
    def __init__(self):
        self.config = {'navigation': {'visual_servo': {'sample_period_s': .05}}}
        self.rcm_mm, self.q = np.zeros(3), np.zeros(3)
        self.tip, self.target = np.array([0., 0., 4.]), np.array([1., 1., 22.])
        self.placements = []
        self.nominal = type('NoInverse', (), {'inverse': staticmethod(poison)})()
        self.target_task = poison

    def current_task(self):
        return self.q.copy()

    def set_task(self, q):
        self.q = np.asarray(q, dtype=float).copy()

    def set_target_world(self, point):
        self.target = np.asarray(point, dtype=float).copy()
        self.placements.append(self.target.copy())

    def metrics(self):
        return {'tip_target_error_um': float(np.linalg.norm(self.tip-self.target)*1000.)}


class StubSensor:
    def __init__(self, reference, settings=None):
        self.reference = reference

    def observe(self):
        r = self.reference
        axis = r.tip/np.linalg.norm(r.tip)
        return {'rgb': np.full((3, 4, 3), 128, dtype=np.uint8),
                'segmentation': np.zeros((3, 4, 2), dtype=int),
                'target_pixel': [1., 1.], 'target_pixels': 4,
                'geometry': {'target_world_mm': r.target.tolist(),
                             'tip_world_mm': r.tip.tolist(),
                             'target_quality': {'fit_quality_mm': 0.},
                             'tip_quality': {'source': 'stub_measured_shaft', 'axis_world': axis.tolist(),
                                             'fit_quality_mm': 0.}}}

    def visibility_metrics(self, segmentation=None):
        return {'target_pixels_visible': 4, 'target_pixels_unoccluded': 4,
                'target_pixels_occluded_by_needle': 0,
                'target_occlusion_fraction_by_needle': 0., 'target_occlusion_state': 'clear',
                'tip_pixels_visible': 2, 'needle_pixels_visible': 4,
                'tip_fully_occluded': False, 'counterfactual_no_needle_rendered': True}

    def close(self):
        pass


class StubController:
    def __init__(self, reference, *, settings, output_dir, initial_velocity,
                 command_validator, control_axes, completion_axes,
                 measurement_goal=None, target_transform=None,
                 independent_endpoint_acceptance=True, fail=False, registry=None):
        self.reference, self.settings, self.output_dir = reference, settings, Path(output_dir)
        self.control_axes = np.asarray(control_axes, dtype=bool)
        self.measurement_goal, self.target_transform = measurement_goal, target_transform
        self.independent_endpoint_acceptance = independent_endpoint_acceptance
        self.validator, self.fail = command_validator, fail
        self.sensor = StubSensor(reference)
        self.rows = []
        self.final_velocity = np.zeros(3)
        self.max_velocity = np.zeros(3)
        self.max_acceleration = np.zeros(3)
        self.iteration_callback = self.observation_hook = None
        if registry is not None:
            registry.append(self)

    def run(self):
        r = self.reference
        start_q, start_tip = r.current_task(), r.tip.copy()
        if self.measurement_goal is not None:
            goal = np.asarray(self.measurement_goal)
        elif self.target_transform is not None:
            goal = self.target_transform(r.target.copy(), r.tip.copy(), self.sensor.observe())
        else:
            goal = r.target.copy()
        # Tiny test motions only encode phase continuity. No IK or scene solution.
        delta = np.where(self.control_axes, [1e-4, 1e-4, 1e-5], 0.)
        for i in range(3):
            q = start_q+delta*(i/2.)
            assert self.validator(r.current_task(), q)
            r.set_task(q)
            r.tip = start_tip+(goal-start_tip)*(i/2.)
            if self.observation_hook:
                self.observation_hook(i, {'time_s': i*.05})
            obs = self.sensor.observe()
            self.rows.append({'iteration': i, 'time_s': i*.05,
                              **dict(zip(('A_deg', 'B_deg', 'C_mm'), task_user_units(q)))})
            if self.iteration_callback:
                self.iteration_callback(i, obs)
        report = {'result': 'FAIL' if self.fail else 'PASS',
                  'reason': 'STUB_FAILURE' if self.fail else 'STUB_MEASURED_AND_SETTLED',
                  'terminal_stop_validated': not self.fail,
                  'iterations': len(self.rows),
                  'final_geometry_measurement': obs['geometry']}
        return report

    def write(self, report, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir)/'visual_servo_report.json').write_text(json.dumps(report))


class StubFixture:
    def __init__(self):
        self.calls = []

    def update(self, marker, phase, frame):
        self.calls.append((marker, phase, frame))
        return {'event_id': None, 'active': False}

    def hide(self, reason=''):
        pass

    @property
    def audit(self):
        return {'calls': self.calls}


class StrictVisualProtocolTests(unittest.TestCase):
    def run_manager(self, temporary, *, fail_at=None):
        ref, registry, fixture = StubReference(), [], StubFixture()
        writer = EpisodeWriter(Path(temporary)/'dataset', 0, 'strict visual test', 'visual_servo', None,
                               ['P1', 'P2', 'P3'])
        scenes = [{'id': f'P{i+1}', 'requested_world_mm': point}
                  for i, point in enumerate(([1., 1., 22.], [-1., 2., 22.], [2., -1., 22.]))]
        def factory(*args, **kwargs):
            kwargs.update(registry=registry, fail=len(registry) == fail_at)
            return StubController(*args, **kwargs)
        with patch.object(navigation, 'no_surface_slide_edge', return_value=True):
            output = navigation.run_strict_visual(ref, scenes, Path(temporary)/'run', writer,
                                                  fixture=fixture, controller_factory=factory,
                                                  sensor_factory=StubSensor)
        return ref, registry, fixture, writer, output

    def test_measured_virtual_goals_do_not_use_target_inverse(self):
        goal = navigation.alignment_goal([3., 0., 4.], [0., 0., 2.], [0., 0., 0.])
        np.testing.assert_allclose(goal, [1.2, 0., 1.6])
        retract = navigation.measured_retraction_goal(
            {'tip_world_mm': [0., 0., 20.], 'tip_quality': {'axis_world': [0., 0., 2.]}}, 2.)
        np.testing.assert_allclose(retract, [0., 0., 18.])
        with self.assertRaises(RuntimeError):
            navigation.measured_retraction_goal({'tip_world_mm': [0., 0., 20.]}, 2.)

    def test_unknown_control_clock_is_nullable_audit_metadata(self):
        reference = StubReference()
        observation = StubSensor(reference).observe()
        perception = navigation._perception(reference, observation, 'visual_failure_hold')
        self.assertIsNone(perception['control_frame_index'])
        with tempfile.TemporaryDirectory(prefix='rcm_unknown_clock_') as temporary:
            writer = EpisodeWriter(Path(temporary), 0, 'test', 'visual_servo', None, ['P1'])
            writer.add(0., [0., 0., 0.], [0., 0., 0.], 'P1', observation['rgb'],
                       terminal=True, perception=perception)
            self.assertIsNone(writer.records[0]['perception']['control_frame_index'])
            self.assertEqual(writer.records[0]['perception']['occlusion_event_id'], 'unknown')

    def test_three_markers_phase_masks_truth_isolation_and_action_alignment(self):
        with tempfile.TemporaryDirectory(prefix='rcm_strict_visual_') as temporary:
            ref, controllers, fixture, writer, (frames, results, info) = self.run_manager(temporary)
            self.assertEqual([r['result'] for r in results], ['PASS']*3)
            self.assertEqual([r['waypoint_id'] for r in results], ['P1', 'P2', 'P3'])
            expected = [('P1', 'visual_align'), ('P1', 'visual_approach'), ('P1', 'visual_retract'),
                        ('P2', 'visual_align'), ('P2', 'visual_approach'), ('P2', 'visual_retract'),
                        ('P3', 'visual_align'), ('P3', 'visual_approach')]
            self.assertEqual([(d['waypoint_id'], d['phase']) for d in info['control_details']], expected)
            for controller, (_, phase) in zip(controllers, expected):
                self.assertEqual(controller.independent_endpoint_acceptance, phase == 'visual_approach')
                np.testing.assert_array_equal(controller.control_axes,
                                              [True, True, False] if phase == 'visual_align' else [False, False, True])
                self.assertFalse(controller.settings['use_target_pose_calibration'])
            self.assertEqual(len(ref.placements), 3)
            self.assertEqual(len(writer.records), 24)
            for prior, following in zip(writer.records, writer.records[1:]):
                np.testing.assert_array_equal(prior['action'], following['observation.state'])
                self.assertLess(prior['timestamp'], following['timestamp'])
            self.assertTrue(writer.records[-1]['is_terminal'])
            np.testing.assert_array_equal(writer.records[-1]['action'], writer.records[-1]['observation.state'])
            for marker in ['P1', 'P2', 'P3']:
                indexes = [frame for m, _, frame in fixture.calls if m == marker]
                self.assertEqual(indexes, list(range(len(indexes))))
                recorded = [r['perception']['control_frame_index'] for r in writer.records
                            if r['waypoint_id'] == marker]
                self.assertEqual(recorded, indexes)
            self.assertTrue(all(r['perception']['occlusion_event_id'] == 'unknown'
                                for r in writer.records))
            self.assertEqual([r['tip_error_um'] for r in results], [0., 0., 0.])
            self.assertFalse(info['ground_truth_used_for_fundus_transfer'])
            self.assertTrue(info['terminal_stop_validated'])

    def test_failure_stops_later_targets_and_preserves_arrival_before_retract(self):
        with tempfile.TemporaryDirectory(prefix='rcm_strict_fail_') as temporary:
            ref, controllers, fixture, writer, (_, results, info) = self.run_manager(temporary, fail_at=2)
            self.assertEqual([r['result'] for r in results], ['PASS', 'NOT_RUN', 'NOT_RUN'])
            self.assertEqual(results[0]['tip_error_um'], 0.)
            self.assertEqual(results[0]['transition_to_next_result'], 'FAIL')
            self.assertGreater(ref.metrics()['tip_target_error_um'], 1000.)
            self.assertEqual(len(ref.placements), 1)
            self.assertEqual(info['failure']['phase'], 'visual_retract')
            self.assertFalse(info['terminal_stop_validated'])
            for prior, following in zip(writer.records, writer.records[1:]):
                np.testing.assert_array_equal(prior['action'], following['observation.state'])


def run_alignment_smoke():
    """Optional <=200-frame real P3 alignment, started from the zero pose.

    Invoke with MUJOCO_GL=egl and ``--alignment-smoke``. The ordinary unit
    suite does not start renderers. This checks alignment, not final arrival.
    """
    from datetime import datetime
    from rcm_reference import Reference
    from rcm_visual_servo import VisualServoController, visual_servo_settings
    from rcm_fundus import no_surface_slide_edge
    reference = Reference()
    output = Path(__file__).resolve().parent/'runs'/(
        'strict_alignment_smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    settings = visual_servo_settings(reference)
    settings.update(max_iterations=200, position_tolerance_um=1.,
                    save_diagnostics=False, use_target_pose_calibration=False)
    controller = VisualServoController(
        reference, settings=settings, output_dir=output,
        control_axes=(True, True, False), completion_axes=(True, True, False),
        independent_endpoint_acceptance=False,
        target_transform=lambda target, tip, observation: navigation.alignment_goal(
            target, tip, reference.rcm_mm),
        command_validator=lambda a, b: no_surface_slide_edge(
            reference, task_user_units(a), task_user_units(b), 1.))
    initial = reference.current_task().copy()
    started = time.perf_counter()
    with patch.object(reference, 'target_task', side_effect=poison), \
            patch.object(reference.nominal, 'inverse', side_effect=poison):
        report = controller.run()
    controller.write(report, output)
    result = {'result': report['result'], 'reason': report['reason'],
              'observations': report['iterations'],
              'terminal_stop_validated': report['terminal_stop_validated'],
              'initial_q': initial.tolist(), 'final_q': reference.current_task().tolist(),
              'C_unchanged': bool(abs(initial[2]-reference.current_task()[2]) < 1e-12),
              'completion_error_um': controller.rows[-1]['completion_error_um'] if controller.rows else None,
              'known_target_task_and_inverse_poisoned': True,
              'scope': 'Alignment only, not eye-surface arrival; independent endpoint acceptance intentionally disabled.',
              'wall_s': time.perf_counter()-started, 'output': str(output)}
    (output/'alignment_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if report['result'] == 'PASS' and result['C_unchanged'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--alignment-smoke']:
        raise SystemExit(run_alignment_smoke())
    unittest.main()
