"""Check accumulated policy intent against the real integer-handle backend."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from rcm.control_target import ContinuousControlTarget, TASK_TO_DEG_DEG_MM
from rcm.reference import Reference


class ControlTargetTests(unittest.TestCase):
    def setUp(self):
        self.ref = Reference()
        self.ref.enable_motor_execution()
        self.motor = self.ref.motor_execution
        self.control = ContinuousControlTarget(self.ref)

    def snapshot(self):
        return copy.deepcopy({key: value for key, value in vars(self.motor.backend).items()
                              if key != 'kin'}), self.motor.time_s, self.motor.revision

    def tick(self, action, scale=1.):
        self.control.synchronize()
        candidate = self.control.preview(action, scale)
        return self.control.commit(candidate)

    def test_small_a_actions_accumulate_where_feedback_actions_are_lost(self):
        for _ in range(8):
            self.ref.set_task(self.ref.current_task() + np.array([.001, 0, 0]) / TASK_TO_DEG_DEG_MM,
                              motor_dt=.05)
        self.assertEqual(self.motor.backend.robot[0], 0)
        for _ in range(30):
            self.tick([.001, 0, 0])
        self.assertGreater(self.motor.backend.robot[0], 0)
        self.assertAlmostEqual(self.control.intent_deg_deg_mm[0], .03)

    def test_small_b_and_c_actions_cross_handle_thresholds(self):
        initial = self.motor.backend.robot
        for _ in range(40):
            self.tick([0, .0005, .0001])
        self.assertNotEqual(self.motor.backend.robot[1], initial[1])
        self.assertNotEqual(self.motor.backend.robot[2], initial[2])
        np.testing.assert_allclose(self.control.intent_deg_deg_mm, [0, .02, .004])

    def test_reverse_and_zero_scale_preserve_fractional_intent(self):
        for _ in range(10):
            self.tick([.001, 0, 0])
        self.tick([1, 1, 1], scale=0)
        self.assertAlmostEqual(self.control.intent_deg_deg_mm[0], .01)
        for _ in range(10):
            self.tick([-.001, 0, 0])
        self.assertAlmostEqual(self.control.intent_deg_deg_mm[0], 0)

    def test_previews_and_rejected_scales_have_no_side_effects(self):
        before = self.snapshot()
        intent = self.control.intent_deg_deg_mm.copy()
        self.control.preview([.005, 0, 0], .5)
        for action in ([float('nan'), 0, 0], [0, 0, float('inf')], [0, 0, 1]):
            with self.assertRaises(ValueError):
                self.control.preview(action)
        self.assertEqual(self.snapshot(), before)
        np.testing.assert_array_equal(self.control.intent_deg_deg_mm, intent)
        self.assertEqual(self.motor.records, [])

    def test_post_quantization_rejection_leaves_backend_and_clock_untouched(self):
        before = self.snapshot()
        with patch.object(self.ref, 'task_from_encoders', side_effect=ValueError('quantized limit')):
            with self.assertRaises(ValueError):
                self.motor.advance([.01, 0, 0])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.motor.records, [])

    def test_intermediate_state_rejection_is_atomic(self):
        before = self.snapshot()
        original = self.ref.task_from_encoders
        calls = 0
        def decode(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ValueError('invalid intermediate linkage state')
            return original(*args, **kwargs)
        with patch.object(self.ref, 'task_from_encoders', side_effect=decode):
            with self.assertRaises(ValueError):
                self.motor.advance([.01, 0, 0])
        self.assertEqual(self.snapshot(), before)

    def test_only_one_prepared_candidate_can_commit(self):
        one = self.control.preview([.01, 0, 0])
        two = self.control.preview([.02, 0, 0])
        accepted = self.control.commit(two)
        np.testing.assert_allclose(accepted, [.02, 0, 0])
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.control.commit(one)
        with self.assertRaises(ValueError):
            self.control.commit(two)
        self.assertEqual(self.snapshot(), before)
        self.assertAlmostEqual(self.motor.time_s, .05)

    def test_duplicate_absolute_handles_do_not_accumulate_motor_targets(self):
        self.tick([.01, 0, .001])
        target, handles = self.motor.backend.target, self.motor.backend.handles
        for _ in range(20):
            self.tick([0, 0, 0])
        self.assertEqual(self.motor.backend.handles, handles)
        self.assertEqual(self.motor.backend.target, target)
        self.assertTrue(self.motor.settled())

    def test_anti_windup_bounds_and_packet_caps(self):
        self.control = ContinuousControlTarget(self.ref, max_handle_delta=(1000, 1000, 1000))
        candidate = self.control.preview([10, 10, 1])
        np.testing.assert_allclose(candidate.intent_deg_deg_mm, [.5, .5, .1])
        self.assertTrue(all(candidate.diagnostics['lead_limited_axes']))
        self.control.commit(candidate)
        actual = self.ref.current_task() * TASK_TO_DEG_DEG_MM
        next_candidate = self.control.preview([10, 10, 1])
        np.testing.assert_allclose(next_candidate.intent_deg_deg_mm - actual, [.5, .5, .1])

    def test_reset_hold_and_watchdog_clear_pending_intent(self):
        self.tick([.005, 0, 0])
        self.motor.hold()
        self.control.synchronize()
        np.testing.assert_array_equal(self.control.intent_deg_deg_mm,
                                      self.ref.current_task() * TASK_TO_DEG_DEG_MM)
        self.tick([.005, 0, 0])
        stale = self.control.preview([.001, 0, 0])
        self.ref.reset_motor_time()
        with self.assertRaises(ValueError):
            self.control.commit(stale)
        self.control.synchronize()
        self.tick([.001, 0, 0])
        self.motor.backend.step(.6)
        with self.assertRaises(RuntimeError):
            self.control.synchronize()
        np.testing.assert_array_equal(self.control.intent_deg_deg_mm,
                                      self.ref.current_task() * TASK_TO_DEG_DEG_MM)

    def test_e_channel_and_noninteger_robot_targets_rejected(self):
        before = self.snapshot()
        for robot in ((0, 1466018, 893518, 1), (.1, 1466018, 893518, 0)):
            with self.assertRaises(ValueError):
                self.motor.advance_robot(robot)
        self.assertEqual(self.snapshot(), before)

    def test_policy_evaluator_uses_accumulator_and_records_accepted_action(self):
        from types import SimpleNamespace
        from lerobot_training.eval_policy import PolicyEvaluator
        evaluator = PolicyEvaluator.__new__(PolicyEvaluator)
        evaluator.controller_variant = 'continuous'
        evaluator.use_oct = False
        evaluator.measure_inference = False
        evaluator.control_mode_counts = {}
        evaluator.last_terminal_solution = None
        evaluator.terminal_locked_target_world_mm = None
        evaluator.reset_buffer = lambda seed: None
        evaluator.get_action = lambda *args: np.array([.001, 0., 0.])
        accepted = []
        observed = []
        evaluator.step_observer = observed.append
        evaluator.record_applied_action = lambda action: accepted.append(action.copy())
        evaluator.invalidate_action_chunk = lambda: None
        sensor = SimpleNamespace(observe=lambda: {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)})
        target = self.ref.site_mm('needle_tip') + np.array([1., 0., 0.])
        result = evaluator.evaluate_episode(self.ref, target, max_steps=30,
                                            error_threshold_mm=.00035, sensor=sensor)
        self.assertIsNone(result['early_stop_reason'])
        self.assertEqual(result['evaluation_contract_version'], 3)
        self.assertEqual(len(accepted), 30)
        self.assertEqual(len(observed), 30)
        self.assertEqual(observed[-1]['motor_record']['step_index'], 29)
        self.assertGreater(self.motor.backend.robot[0], 0)
        self.assertAlmostEqual(self.motor.time_s, 1.5)
        np.testing.assert_allclose(accepted[-1], [.001, 0., 0.])
        self.assertEqual(len(result['trajectory']['control_targets']), 30)

    def test_preview_commit_preserves_original_backend_dynamics(self):
        expected = copy.copy(self.motor.backend)
        expected.history = []
        for goal in ([.01, -.005, .0001], [.02, .003, .0002], [0., 0., 0.]):
            desired = tuple(round(float(v)) for v in self.ref.encoders(goal)) + (0,)
            command = expected.command_for_robot_target(desired)
            start = self.motor.time_s
            expected.consume(command, start)
            for index in range(25):
                expected.step(.002, start + (index + 1) * .002)
            self.ref.set_task(goal, motor_dt=.05)
            self.assertEqual(self.motor.backend.robot, expected.robot)
            self.assertEqual(self.motor.backend.target, expected.target)
            self.assertEqual(self.motor.backend.handles, expected.handles)
            np.testing.assert_allclose(self.motor.backend._native, expected._native, rtol=0, atol=1e-13)

    def test_policy_scaled_rejection_preserves_measured_error_and_commits_once(self):
        from types import SimpleNamespace
        from lerobot_training.eval_policy import PolicyEvaluator
        evaluator = PolicyEvaluator.__new__(PolicyEvaluator)
        evaluator.controller_variant = 'continuous'
        evaluator.use_oct = False
        evaluator.measure_inference = False
        evaluator.control_mode_counts = {}
        evaluator.last_terminal_solution = None
        evaluator.terminal_locked_target_world_mm = None
        evaluator.reset_buffer = lambda seed: None
        evaluator.get_action = lambda *args: np.array([.05, 0., 0.])
        observed, accepted = [], []
        evaluator.step_observer = observed.append
        evaluator.record_applied_action = lambda action: accepted.append(action.copy())
        evaluator.invalidate_action_chunk = lambda: None
        sensor = SimpleNamespace(observe=lambda: {'rgb': np.zeros((2, 2, 3), dtype=np.uint8)})
        target = self.ref.site_mm('needle_tip') + np.array([1., 0., 0.])
        def accumulator(reference):
            # A full 0.05-degree action exceeds one real handle in both gears.
            return ContinuousControlTarget(reference, max_handle_delta=(1, 1, 1))
        with patch('lerobot_training.eval_policy.ContinuousControlTarget', side_effect=accumulator):
            result = evaluator.evaluate_episode(self.ref, target, max_steps=3,
                                                error_threshold_mm=.00035, sensor=sensor)
        self.assertIsNone(result['early_stop_reason'])
        self.assertIn('per-packet handle delta limit', result['last_safety_error'])
        self.assertAlmostEqual(observed[0]['error_before_mm'], 1.)
        self.assertEqual(len(observed), 3)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(self.motor.records), 3)
        self.assertEqual(self.motor.backend.accepted_commands, 3)
        self.assertAlmostEqual(self.motor.time_s, .15)
        self.assertTrue(all(row['scale'] < 1 for row in result['trajectory']['control_targets']))
        np.testing.assert_allclose(accepted, result['trajectory']['actions'])

    def test_v9_accumulator_updates_display_from_feedback(self):
        from rcm.scene_v9 import RCMSceneV9
        scene = RCMSceneV9()
        try:
            scene.enable_motor_execution()
            control = ContinuousControlTarget(scene)
            for _ in range(20):
                control.commit(control.preview([.001, 0, .0001]))
            self.assertIsNone(scene._ref.motor_execution)
            self.assertEqual(tuple(scene.scene.last_robot), scene.motor_robot())
            self.assertGreater(scene.motor_robot()[0], 0)
            np.testing.assert_allclose(scene.current_task(),
                scene.task_from_encoders(scene.motor_robot()[:3]), atol=1e-12)
        finally:
            scene.close()

    def test_pairing_rejects_seed_and_target_mismatches(self):
        from scripts.research.control_g15_diagnostics import paired, aggregate
        row = {'episode_index': 0, 'target_id': 't0', 'seed': 42,
               'target_robot_mm': [0, 0, 21], 'final_error_um': 8., 'strict_success': False}
        better = dict(row, final_error_um=5.)
        self.assertEqual(paired([better], [row])[0]['delta_final_um'], -3.)
        self.assertEqual(aggregate([better])['hit_5um_count'], 1)
        with self.assertRaises(ValueError):
            paired([dict(better, seed=43)], [row])
        with self.assertRaises(ValueError):
            paired([dict(better, target_robot_mm=[0, 0, 22])], [row])
        with self.assertRaises(ValueError):
            aggregate([])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ControlTargetTests))
    output = stream.getvalue()
    print(output, end='')
    (args.output_dir / 'tests.log').write_text(output)
    from mujoco_model.driver_loader import driver_identity
    import hashlib
    sources = ('rcm/control_target.py', 'rcm/motor_execution.py', 'rcm/reference.py',
               'rcm/scene_v9.py', 'lerobot_training/eval_policy.py', __file__)
    payload = {'timestamp': datetime.now().astimezone().isoformat(),
               'status': 'PASS' if result.wasSuccessful() else 'FAIL',
               'tests_run': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'driver': driver_identity(),
               'device': 'CPU: small scalar protocol/kinematics checks',
               'checkpoint_dataset': 'not applicable', 'output_dir': str(args.output_dir),
               'source_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in sources},
               'command': 'python -B -m scripts.verification.verify_control_target --output-dir ' + str(args.output_dir)}
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    with Path('TEST_LOG.md').open('ab') as log:
        log.write(('\n- {timestamp}: P2 control target {status}; {tests_run} tests, '
                   '{failures} failures, {errors} errors; command `{command}`; driver '
                   '{driver}; {device}; checkpoint/dataset N/A; evidence `{output_dir}/summary.json`. '
                   'Software simulation only.\n').format(**payload).encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
