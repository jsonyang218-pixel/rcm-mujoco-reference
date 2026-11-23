"""Regression checks and real RGB-D/CUDA recovery smoke for fractional teacher intent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.research.homog_replay_common import (
    CAPS, HOLD_START, OLD, QUALITY, ROOT, WORKFLOW, identity, make_teacher, read, record, write_json,
)
from scripts.research.homog_replay_target import TeacherIntent, UNIT


def fake_scene():
    return SimpleNamespace(validate_task=lambda q: np.asarray(q, dtype=float),
        motor_execution=SimpleNamespace(control_period_s=.05,
            backend=SimpleNamespace(speed_limits=(10., 10., 2.), fault=None)))


class TeacherIntentTests(unittest.TestCase):
    def test_fractional_steps_survive_stationary_feedback(self):
        intent = TeacherIntent(fake_scene(), CAPS)
        q = np.array([.2, .1, .02])
        increment = np.array([.00519945, -.00346595, -.0005388])
        first, _ = intent.preview(q, increment)
        intent.commit(first)
        second, _ = intent.preview(q, increment)
        np.testing.assert_allclose(first, increment, atol=1e-12)
        np.testing.assert_allclose(second, increment * 2, atol=1e-12)
        self.assertGreater(abs(second[0]), .015 / 2)

    def test_preview_does_not_commit_and_accepts_only_scaled_target(self):
        intent = TeacherIntent(fake_scene(), CAPS)
        q = np.zeros(3)
        first, _ = intent.preview(q, [.01, .005, .001])
        self.assertIsNone(intent.intent_task_si)
        intent.preview(q, [.02, .006, .002])
        intent.commit(first * .5)
        np.testing.assert_allclose(intent.intent_task_si / UNIT, first * .5)
        with self.assertRaises(ValueError):
            intent.commit(first)

    def test_hold_discards_residual_and_uses_actual_pose(self):
        intent = TeacherIntent(fake_scene(), CAPS)
        q = np.array([.2, .1, .02])
        command, _ = intent.preview(q, [.005, .003, .0005])
        intent.commit(command)
        command, diagnostics = intent.preview(q, np.zeros(3), hold=True)
        np.testing.assert_array_equal(command, np.zeros(3))
        self.assertTrue(diagnostics['hold_residual_reset'])
        intent.commit(command)
        np.testing.assert_array_equal(intent.intent_task_si, q)

    def test_blocked_axis_intent_is_bounded_without_windup(self):
        intent = TeacherIntent(fake_scene(), CAPS)
        q = np.zeros(3)
        for tick in range(100):
            command, _ = intent.preview(q, [.2, .1, .03])
            intent.commit(command)
            self.assertTrue(np.all(np.abs(command) <= CAPS + 1e-12))
        np.testing.assert_allclose(intent.intent_task_si / UNIT, CAPS)

    def test_fault_and_nonfinite_inputs_rejected(self):
        scene = fake_scene()
        intent = TeacherIntent(scene, CAPS)
        with self.assertRaises(ValueError):
            intent.preview(np.zeros(3), [np.nan, 0, 0])
        scene.motor_execution.backend.fault = 'fault'
        with self.assertRaises(ValueError):
            intent.preview(np.zeros(3), np.ones(3))


class RecoverySmokeWorker:
    def __init__(self, spec):
        from scripts.research.homog_replay_common import check_identity, gpu_check
        gpu_check()
        check_identity(spec['identity'])
        from lerobot_training.eval_policy import PolicyEvaluator
        self.evaluator = PolicyEvaluator(str(OLD), device='cuda', num_action_samples=1,
            n_action_execute_steps=1, flow_integration_steps=4, measure_inference=False,
            controller_variant='feedback', rgbd_render_quality=QUALITY.copy())

    def run(self, task, output):
        from scripts.research.a_axis_flip_audit import static_scene
        from scripts.research import homog_replay_collect as collection
        from scripts.research.clean_oct_workers import egl_identity
        scene = static_scene()
        recorder = None
        original_hold = collection.HOLD_START
        try:
            collection.HOLD_START = task['hold_start']
            scene.set_target_world(np.asarray(task['target_world_mm']))
            scene.set_task(np.asarray(task['initial_state_si']))
            teacher = make_teacher(scene)
            recorder = collection.RecordingSensor(teacher, scene, self.evaluator, task, output)
            self.evaluator.get_action = lambda *_: recorder.teacher_command.copy()
            self.evaluator.step_observer = recorder.after_step
            ep = self.evaluator.evaluate_episode(scene, np.asarray(task['target_world_mm']),
                max_steps=task['steps'], control_period_s=.05, seed=task['seed'],
                sensor=recorder, progress_every=40, stop_on_success=False,
                max_consecutive_safety_limited=None)
            write_json(output / 'policy_trajectory.json', ep)
            rows = recorder.rows
            true = np.asarray([np.linalg.norm(r['truth_navigation_mm']) * 1000 for r in rows])
            measured = np.asarray([r['measured_error_um'] for r in rows])
            motors = ep['motor_records']
            held = [r for r in rows if r['hold']]
            first_below = np.flatnonzero(true < 5)
            passed = (len(rows) == task['steps'] and ep['early_stop_reason'] is None
                and ep['final_error_mm'] * 1000 < 5 and measured[-1] < 5
                and ep['safety_limited_steps'] == 0
                and any(m['tip_displacement_um'] > 0 for m in motors[:40])
                and (task['regime'] != 'T3' or len(held) == 200)
                and all(r['measured_error_um'] < 5 and np.all(np.asarray(r['action']) == 0)
                        and np.all(np.asarray(r['executed_action']) == 0) for r in held))
            result = {'status': 'passed' if passed else 'failed', 'episode_index': task['episode_index'],
                'regime': task['regime'], 'actions': len(motors), 'final_error_um': ep['final_error_mm'] * 1000,
                'final_measured_error_um': float(measured[-1]), 'initial_error_um': float(true[0]),
                'min_error_um': float(true.min()), 'first_below_5um_step': int(first_below[0]) if len(first_below) else None,
                'hold_frames': len(held), 'hold_true_error_max_um': max((np.linalg.norm(r['truth_navigation_mm']) * 1000 for r in held), default=None),
                'safety_limited_steps': ep['safety_limited_steps'], 'egl': egl_identity(),
                'stored_vs_policy_image_tensor_max_abs': recorder.image_parity,
                'gate_passed': bool(passed), 'scope': 'Three training targets, short verification only; not final model accuracy',
                'from_frozen_plateau': task.get('from_frozen_plateau', False)}
            write_json(output / 'summary.json', result)
            record('teacher_intent_recovery_smoke_episode', result['status'], output, result)
            return result
        finally:
            collection.HOLD_START = original_hold
            self.evaluator.step_observer = None
            if recorder is not None:
                recorder.close()
            scene.close()

    def close(self):
        pass


def smoke(output):
    from scripts.research.gpu_episode_pool import run_pool
    # This old manifest pins the same training targets and all initial states.
    old = ROOT / 'runs/20261123/homog_clean_workflow_v2'
    manifest = read(old / 'execution_manifest.json')
    tasks = []
    for index in (22, 40, 41):
        task = manifest['collection_tasks'][index].copy()
        task.update(steps=120 if index == 22 else 280, hold_start=80)
        if index == 22:
            ep = read(ROOT / 'data/homog_clean_replay_20261123/shards/tasks/collection_E22/policy_trajectory.json')
            task['initial_state_si'] = ep['motor_records'][-1]['actual_after_task_rad_rad_m']
            task['from_frozen_plateau'] = True
        tasks.append(task)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'card.json', {'tasks': tasks, 'quality': QUALITY,
        'workers': 3, 'all_timeouts': None, 'identity': identity(),
        'changed_hold_start_for_smoke': 80,
        'production_hold_start': HOLD_START})
    results = run_pool({'smoke': {'factory': 'scripts.verification.verify_homog_teacher_intent:RecoverySmokeWorker',
        'spec': {'identity': identity()}, 'tasks': tasks}}, output / 'shards',
        workers=3, max_workers=3, timeout_s=None, stage_timeout_s=None,
        initializer_timeout_s=None, execution_timeout_s=None)
    passed = len(results) == 3 and all(r['gate_passed'] for r in results.values())
    result = {'status': 'passed' if passed else 'failed', 'episodes': list(results.values()),
        'scope': 'Isolated verification of teacher recovery and 200 explicit hold frames, not full-plan acceptance'}
    write_json(output / 'summary.json', result)
    record('teacher_intent_recovery_smoke', result['status'], output, result)
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.smoke:
        return smoke(output)
    output.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TeacherIntentTests))
    summary = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'CPU_reason': 'Small deterministic sender-side intent regression fixtures'}
    write_json(output / 'summary.json', summary)
    record('teacher_intent_regressions', summary['status'], output, summary)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
