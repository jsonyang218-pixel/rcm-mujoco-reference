"""Fast measured-geometry control checks; no renderer or scene-target inverse.

The real controller drives the independent coupled plant from
``verify_visual_motion``. Interventions remove or change sensor measurements;
the only model supplied to the controller is the calibrated actuation matrix.
Per-case reports, CSV rows and observation audit trails are saved for review.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

import rcm_visual_servo as servo
from verify_visual_motion import COUPLED, OFFSET, LinearPlant, MeasuredSensor


OUTPUT = Path('runs/multitarget_occlusion_20261022/control_checks')


def hide_target(observation):
    observation['geometry']['target_world_mm'] = None
    observation['geometry']['target_error'] = 'TARGET_NOT_VISIBLE'
    observation.update(target_pixel=None, target_depth_m=None, target_pixels=0)


def hide_tip(observation):
    observation['geometry']['tip_world_mm'] = None
    observation['geometry']['tip_error'] = 'TIP_NOT_OBSERVABLE'
    observation.update(tip_pixel=None, tip_depth_m=None, tip_pixels=0)


class IntervenedSensor(MeasuredSensor):
    def __init__(self, reference, settings, intervene, events):
        super().__init__(reference, settings)
        self.intervene = intervene
        self.events = events
        self.raw = []

    def observe(self):
        index = self.calls
        if not self.events or self.events[-1] != {'kind': 'hook', 'iteration': index}:
            raise AssertionError('Every sensor observation must follow its matching hook')
        self.events.append({'kind': 'observe', 'iteration': index})
        observation = super().observe()
        if self.intervene is not None:
            self.intervene(index, observation)
        # Deliberately expose plausible privileged lookalikes: neither target
        # nor tip loss may be filled from these non-measurement fields.
        observation['tip_world_mm'] = self.reference.tip().tolist()
        observation['target_world_mm_privileged'] = self.reference.truth_target.tolist()
        self.raw.append(copy.deepcopy({k: observation[k] for k in (
            'geometry', 'target_pixel', 'tip_pixel', 'target_depth_m', 'tip_depth_m',
            'target_pixels', 'tip_pixels')}))
        return observation


class OcclusionRecoveryChecks(unittest.TestCase):
    def run_case(self, label, *, goal=(.005, -.004, .0004), intervene=None,
                 settings=None, navigation=None, mode='static_memory_reobserve',
                 measurement_goal=None, target_transform=None,
                 independent_endpoint_acceptance=True, control_axes=None):
        target = OFFSET+COUPLED@np.asarray(goal)
        plant = LinearPlant(COUPLED, target)
        plant.config['navigation'].update(navigation or {})
        options = {**servo.DEFAULT_SERVO, 'position_feedback': 'geometry',
                   'occlusion_handling': mode, 'position_tolerance_um': 8.,
                   'endpoint_tolerance_um': 20., 'require_settled': True,
                   'settle_frames': 3, 'save_diagnostics': False,
                   'max_iterations': 600, 'sample_period_s': .05,
                   'geometry_gain_per_s': .8}
        options.update(settings or {})
        events, callbacks = [], []
        factory = lambda reference, configuration: IntervenedSensor(
            reference, configuration, intervene, events)
        with patch.object(servo, 'VisualServoSensor', factory):
            controller = servo.VisualServoController(
                plant, settings=options, control_axes=control_axes,
                measurement_goal=measurement_goal, target_transform=target_transform,
                independent_endpoint_acceptance=independent_endpoint_acceptance)
        controller._model_jacobian = lambda q: COUPLED.copy()

        def hook(iteration, state):
            events.append({'kind': 'hook', 'iteration': iteration})

        def callback(iteration, observation):
            events.append({'kind': 'callback', 'iteration': iteration})
            current = {k: observation[k] for k in controller.sensor.raw[-1]}
            self.assertEqual(current, controller.sensor.raw[-1], 'Raw sensor data was edited')
            callbacks.append({'iteration': iteration,
                              'tracking': copy.deepcopy(observation.get('tracking', {})),
                              'raw': copy.deepcopy(current)})
            return True

        controller.observation_hook, controller.iteration_callback = hook, callback
        report = controller.run()
        directory = OUTPUT/label
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (directory/'observations.json').write_text(json.dumps(callbacks, indent=2)+'\n')
        (directory/'events.json').write_text(json.dumps(events, indent=2)+'\n')
        if controller.rows:
            columns = list(controller.rows[0])
            with (directory/'controller.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(controller.rows)
            self.assertTrue(all(set(row) == set(columns) for row in controller.rows),
                            'Every normal, missing, recovery and terminal row needs the same columns')
            with (directory/'controller.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(controller.rows))
            self.assertTrue(all(set(row) == set(columns) and None not in row for row in rows))
        self.assertTrue(controller.sensor.closed)
        # A loss frame needs an audit row and a callback just like a good frame.
        self.assertEqual(len(callbacks), controller.sensor.calls)
        self.assertEqual(len(controller.rows), controller.sensor.calls)
        self.assertEqual(events, [event for i in range(controller.sensor.calls) for event in (
            {'kind': 'hook', 'iteration': i}, {'kind': 'observe', 'iteration': i},
            {'kind': 'callback', 'iteration': i})])
        # Verify applied movement independently of the report's aggregate
        # claims. Failure's final simulator hold is reported separately from
        # these executed, acceleration-limited control intervals.
        dt = options['sample_period_s']
        previous_velocity = np.zeros(3)
        navigation = plant.config['navigation']
        for row, following in zip(controller.rows, controller.rows[1:]):
            velocity = np.asarray(row['command_velocity_A_deg_B_deg_C_mm_per_s'])
            acceleration = (velocity-previous_velocity)/dt
            current_q = np.array([row[key] for key in ('A_deg', 'B_deg', 'C_mm')])
            next_q = np.array([following[key] for key in ('A_deg', 'B_deg', 'C_mm')])
            np.testing.assert_allclose((next_q-current_q)/dt, velocity, atol=1e-10)
            np.testing.assert_allclose(
                row['command_acceleration_A_deg_B_deg_C_mm_per_s2'], acceleration, atol=1e-10)
            self.assertTrue(np.all(np.abs(velocity) <= np.asarray(
                navigation['max_velocity_A_deg_B_deg_C_mm_per_s'])+1e-10))
            self.assertTrue(np.all(np.abs(acceleration) <= np.asarray(
                navigation['max_acceleration_A_deg_B_deg_C_mm_per_s2'])+1e-10))
            previous_velocity = velocity
        return controller, plant, report, callbacks

    def assert_stopped_pass(self, report):
        self.assertEqual(report['result'], 'PASS', report)
        self.assertEqual(report['controller_reason'], 'POSITION_ERROR_AND_SETTLED')
        self.assertTrue(report['terminal_stop_validated'])
        self.assertGreaterEqual(report['settled_frames'], 3)
        self.assertTrue(report['trajectory_rate_limits_pass'])
        self.assertFalse(report['simulation_abort_hold'])
        self.assertFalse(report['physical_braking_validated'])
        np.testing.assert_array_equal(
            report['final_command_velocity_A_deg_B_deg_C_mm_per_s'], np.zeros(3))

    def assert_abort(self, report):
        self.assertEqual(report['result'], 'FAIL', report)
        self.assertEqual(report['controller_result'], 'FAIL')
        self.assertTrue(report['simulation_abort_hold'])
        self.assertFalse(report['physical_braking_validated'])
        self.assertFalse(report['terminal_stop_validated'])
        np.testing.assert_array_equal(
            report['final_command_velocity_A_deg_B_deg_C_mm_per_s'], np.zeros(3))

    def test_short_missing_target_reaches_and_settles_from_registered_memory(self):
        def intervene(index, observation):
            if index >= 4:
                hide_target(observation)
        controller, _, report, callbacks = self.run_case(
            'short_memory_arrival', goal=(.005, -.004, .0001), intervene=intervene,
            settings={'geometry_gain_per_s': 3.})
        self.assert_stopped_pass(report)
        self.assertEqual(report['recovery_attempt_count'], 0)
        self.assertEqual(report['final_target_source'], 'registered_memory')
        self.assertFalse(report['final_target_observed'])
        self.assertGreater(report['memory_frame_count'], 0)
        lost = callbacks[4:]
        self.assertTrue(all(item['tracking']['target_source'] == 'registered_memory' for item in lost))
        self.assertTrue(all(item['raw']['geometry']['target_world_mm'] is None for item in lost))
        self.assertTrue(all(item['tracking']['cached_target_age_s'] <= 1.+1e-12 for item in lost))
        self.assertLess(report['final_tip_target_error_um'], 20.)
        self.assertEqual(len(controller.rows), len(callbacks))

    def test_expired_memory_brakes_retracts_and_requires_three_reobservations(self):
        def intervene(index, observation):
            if 4 <= index < 35:
                hide_target(observation)
        controller, _, report, callbacks = self.run_case('long_loss_recovery', intervene=intervene)
        self.assert_stopped_pass(report)
        self.assertEqual(report['recovery_attempt_count'], 1)
        self.assertEqual(report['recovery_success_count'], 1)
        self.assertEqual(report['memory_frame_count'], 20)
        expired = controller.rows[24]
        self.assertEqual(expired['error_code'], 'TARGET_MEMORY_EXPIRED')
        self.assertEqual(expired['controller_state'], 'braking')
        self.assertIn('retracting', {row['controller_state'] for row in controller.rows})
        resumed = next(row['iteration'] for row in controller.rows[35:]
                       if row['controller_state'] == 'tracking')
        self.assertGreaterEqual(resumed, 37)
        for frame in (35, 36):
            self.assertNotEqual(callbacks[frame]['tracking']['controller_state'], 'tracking')
        self.assertTrue(all(callbacks[frame]['tracking']['target_observed']
                            for frame in range(resumed-2, resumed+1)))
        recovery = [row for row in controller.rows if row['controller_state'] != 'tracking']
        self.assertTrue(all(row['completion_error_um'] is None for row in recovery))

    def test_permanent_loss_times_out_after_four_seconds_and_holds_zero(self):
        def intervene(index, observation):
            if index >= 4:
                hide_target(observation)
        controller, _, report, _ = self.run_case(
            'permanent_loss_timeout', intervene=intervene,
            navigation={'max_velocity_A_deg_B_deg_C_mm_per_s': [4., 2., .05]})
        self.assert_abort(report)
        self.assertEqual(report['controller_reason'], 'OCCLUSION_RECOVERY_TIMEOUT')
        first = next(row['time_s'] for row in controller.rows if row['controller_state'] != 'tracking')
        self.assertAlmostEqual(controller.rows[-1]['time_s']-first, 4.)
        self.assertEqual(report['recovery_success_count'], 0)
        self.assertTrue(all(row['used_target_world_mm'] is None for row in controller.rows[24:]))

    def test_retraction_boundary_exits_fail_without_crossing_bound(self):
        def intervene(index, observation):
            if index >= 4:
                hide_target(observation)
        controller, _, report, _ = self.run_case('permanent_loss_retract_bound', intervene=intervene)
        self.assert_abort(report)
        self.assertEqual(report['controller_reason'], 'OCCLUSION_RETRACTION_LIMIT')
        first = next(row for row in controller.rows if row['controller_state'] == 'retracting')
        self.assertGreaterEqual(controller.rows[-1]['C_mm'], first['C_mm']-.5-1e-9)

    def test_reappearing_target_shift_over_40um_requires_replan(self):
        def intervene(index, observation):
            if 4 <= index < 8:
                hide_target(observation)
            elif index >= 8:
                target = np.asarray(observation['geometry']['target_world_mm'])+[.041, 0., 0.]
                observation['geometry']['target_world_mm'] = target.tolist()
                observation['target_pixel'] = target[:2].tolist()
        controller, _, report, callbacks = self.run_case('moved_target_rejected', intervene=intervene)
        self.assert_abort(report)
        self.assertEqual(report['controller_reason'], 'TARGET_MOVED_REPLAN_REQUIRED')
        self.assertEqual(report['iterations'], 9)
        self.assertEqual(controller.rows[-1]['target_source'], 'unavailable')
        self.assertIsNone(callbacks[-1]['tracking']['used_target_world_mm'])

    def test_missing_tip_cannot_be_filled_by_target_memory_or_privileged_lookalike(self):
        def intervene(index, observation):
            if 4 <= index < 8:
                hide_tip(observation)
        controller, _, report, callbacks = self.run_case('missing_tip_recovery', intervene=intervene)
        self.assert_stopped_pass(report)
        self.assertEqual(report['recovery_attempt_count'], 1)
        self.assertEqual(report['recovery_success_count'], 1)
        for item, row in zip(callbacks[4:8], controller.rows[4:8]):
            self.assertEqual(item['tracking']['error_code'], 'TIP_NOT_OBSERVABLE')
            self.assertEqual(item['tracking']['target_source'], 'unavailable')
            self.assertIsNone(item['raw']['geometry']['tip_world_mm'])
            self.assertIsNone(item['tracking']['used_target_world_mm'])
            self.assertIsNone(row['estimated_position_error_um'])

    def test_stop_mode_never_bridges_a_missing_target(self):
        def intervene(index, observation):
            if index >= 4:
                hide_target(observation)
        _, _, report, callbacks = self.run_case('stop_no_bridge', intervene=intervene, mode='stop')
        self.assert_abort(report)
        self.assertEqual(report['controller_reason'], 'TARGET_NOT_VISIBLE')
        self.assertIsNone(report['final_estimated_position_error_um'])
        self.assertEqual(report['iterations'], 5)
        self.assertEqual(report.get('memory_frame_count', 0), 0)
        self.assertTrue(all(item['tracking'].get('target_source') != 'registered_memory' for item in callbacks))

    def test_visual_retraction_goal_can_complete_while_far_from_true_marker(self):
        waypoint = OFFSET-COUPLED[:, 2]*.0002
        def intervene(index, observation):
            hide_target(observation)
        _, _, report, callbacks = self.run_case(
            'virtual_retraction_goal', goal=(0., 0., .001), intervene=intervene,
            measurement_goal=waypoint, independent_endpoint_acceptance=False,
            control_axes=[False, False, True])
        self.assert_stopped_pass(report)
        self.assertGreater(report['final_tip_target_error_um'], 1000.)
        self.assertEqual(report['measurement_goal_type'], 'measured_tip_retraction_waypoint')
        self.assertFalse(report['ground_truth_used_for_post_run_acceptance'])
        self.assertIsNone(report['independent_accuracy_acceptance_pass'])
        self.assertTrue(all(item['tracking']['target_source'] == 'visual_retraction_waypoint'
                            for item in callbacks))

    def test_virtual_retraction_recovers_tip_without_observing_marker_or_changing_goal(self):
        waypoint = OFFSET-COUPLED[:, 2]*.0002

        def intervene(index, observation):
            # A measured retraction waypoint does not require the scene marker.
            # Its only missing control input is the briefly hidden current tip.
            hide_target(observation)
            if 4 <= index < 8:
                hide_tip(observation)

        controller, plant, report, callbacks = self.run_case(
            'virtual_retraction_tip_recovery', goal=(0., 0., .001), intervene=intervene,
            measurement_goal=waypoint, independent_endpoint_acceptance=False,
            control_axes=[False, False, True])
        self.assert_stopped_pass(report)
        self.assertEqual(report['recovery_attempt_count'], 1)
        self.assertEqual(report['recovery_success_count'], 1)
        self.assertEqual(report['memory_frame_count'], 0)
        self.assertEqual(report['measurement_goal_type'], 'measured_tip_retraction_waypoint')
        self.assertFalse(report['ground_truth_used_for_post_run_acceptance'])
        self.assertGreater(report['final_tip_target_error_um'], 1000.)
        self.assertLess(np.linalg.norm(plant.tip()-waypoint)*1000., 8.)
        np.testing.assert_array_equal(controller.measurement_goal, waypoint)
        np.testing.assert_array_equal(plant.q[:2], np.zeros(2))

        # Reacquisition needs three fresh tip observations plus a stopped
        # command, while target_observed must retain its marker-only meaning.
        resumed = next(row['iteration'] for row in controller.rows[8:]
                       if row['controller_state'] == 'tracking')
        self.assertGreaterEqual(resumed, 10)
        for frame in (8, 9):
            self.assertNotEqual(callbacks[frame]['tracking']['controller_state'], 'tracking')
        self.assertTrue(all(item['raw']['geometry']['tip_world_mm'] is not None
                            for item in callbacks[resumed-2:resumed+1]))
        self.assertTrue(all(not item['tracking']['target_observed'] for item in callbacks))
        self.assertTrue(all(item['raw']['geometry']['target_world_mm'] is None for item in callbacks))
        for index, item in enumerate(callbacks):
            tracking = item['tracking']
            if 4 <= index < 8:
                self.assertEqual(tracking['target_source'], 'unavailable')
                self.assertIsNone(tracking['used_target_world_mm'])
            else:
                self.assertEqual(tracking['target_source'], 'visual_retraction_waypoint')
                np.testing.assert_array_equal(tracking['used_target_world_mm'], waypoint)

    def test_transformed_measured_goal_intermediate_pass_and_final_20um_gate(self):
        transformed = []
        def transform(target, tip, observation):
            np.testing.assert_allclose(target, observation['geometry']['target_world_mm'])
            np.testing.assert_allclose(tip, observation['geometry']['tip_world_mm'])
            transformed.append(target.tolist())
            return target-COUPLED[:, 2]*.0005
        intermediate, plant, report, _ = self.run_case(
            'transformed_goal_intermediate', goal=(0., 0., .001),
            target_transform=transform, independent_endpoint_acceptance=False)
        self.assert_stopped_pass(report)
        self.assertGreater(report['final_tip_target_error_um'], 500.)
        self.assertEqual(report['measurement_goal_type'], 'transformed_measured_target')
        final, other, rejected, _ = self.run_case(
            'transformed_goal_final_gate', goal=(0., 0., .001),
            target_transform=transform, independent_endpoint_acceptance=True)
        self.assertEqual(rejected['controller_result'], 'PASS')
        self.assertEqual(rejected['result'], 'FAIL')
        self.assertEqual(rejected['reason'], 'INDEPENDENT_ACCURACY_REJECTED')
        self.assertTrue(rejected['terminal_stop_validated'])
        self.assertFalse(rejected['independent_accuracy_acceptance_pass'])
        self.assertEqual(rejected['independent_accuracy_tolerance_um'], 20.)
        self.assertEqual(len(intermediate.rows), len(final.rows))
        np.testing.assert_allclose(plant.q, other.q, atol=1e-14)
        self.assertEqual(len(transformed), len(intermediate.rows)+len(final.rows))


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
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output_dir
    OUTPUT.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OcclusionRecoveryChecks)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReportResult).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
              'exit_code': 0 if result.wasSuccessful() else 1,
              'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'outcomes': result.outcomes,
              'feedback': 'current measured tip; current measured or labelled bounded registered static target',
              'plant': 'independent coupled linear plant; only calibrated actuation matrix supplied to controller',
              'renderer_used': False, 'physical_braking_validated': False}
    (OUTPUT/'control_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Report: {OUTPUT / "control_checks.json"}')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
