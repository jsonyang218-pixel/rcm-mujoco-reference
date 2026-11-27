"""T3 hold transition regressions, including the frozen E42 failure protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.research.homog_replay_common import (
    CAPS, HOLD_START, QUALITY, ROOT, identity, read, record, write_json,
)
from scripts.research.homog_replay_hold import T3HoldError, T3HoldIntent
from scripts.research.homog_replay_target import UNIT

OUTPUT = None
FAILURE = ROOT / 'runs/20261123/verification/homog_t3_hold_diagnosis_v1'


def fixture():
    motor = SimpleNamespace(control_period_s=.05,
        backend=SimpleNamespace(speed_limits=(10., 10., 2.), fault=None, robot=(1, 2, 3, 0)),
        last_record=None, is_settled=False)
    motor.settled = lambda: motor.is_settled
    motor.preview = lambda q: SimpleNamespace(actual=np.round(np.asarray(q), 12),
        record={'quantized_target_robot_ABCE': list(motor.backend.robot)})
    motor.summary = lambda: {'encoder_settled': motor.settled()}
    scene = SimpleNamespace(validate_task=lambda q: np.asarray(q, dtype=float), motor_execution=motor)
    return scene, T3HoldIntent(scene, CAPS, 400)


class T3HoldTests(unittest.TestCase):
    def test_only_unreachable_T3_changes_are_compatible_for_T1_T2(self):
        from scripts.tools.prepare_homog_hold_repair import ORIGINAL, compatible_t1_t2
        source = (ROOT/'scripts/research/homog_replay_collect.py').read_text()
        self.assertTrue(all(compatible_t1_t2(ORIGINAL.read_text(), source).values()))
        for modified in (
                source.replace("max_steps=steps,", "max_steps=600,"),
                source.replace("previous_velocity=self.scene.motor_execution.actual_velocity", "previous_velocity=np.zeros(3)"),
                source.replace("if regime == 'T3':", "if regime == 'T2':"),
                source.replace("hold = regime == 'T3' and", "hold = regime == 'T2' and")):
            with self.assertRaises(ValueError):
                compatible_t1_t2(ORIGINAL.read_text(), modified)

    def test_collector_preserves_failure_observation_without_extra_tick(self):
        from scripts.research.homog_replay_collect import RecordingSensor
        scene, _ = fixture()
        scene.current_task = lambda: np.array([.1, .2, .02])
        scene.motor_execution.actual_velocity = np.zeros(3)
        observation = {'rgb': np.zeros((2, 2, 3), dtype=np.uint8),
            'oct_navigation_depth': {'delta_xyz_m': [0., 0., .000003]}}
        sensor = SimpleNamespace(observation_time_s=20., observe=lambda: observation, close=lambda: None)
        teacher = SimpleNamespace(sensor=sensor,
            geometry_action_from_observation=lambda *a, **kw: {'valid': True, 'action_deg_deg_mm': [0., 0., 0.]})
        output = OUTPUT/'failure_capture_fixture'
        output.mkdir()
        recorder = RecordingSensor(teacher, scene, None, {'regime': 'T3', 'id': 'fixture'}, output)
        recorder.index = 400
        try:
            with patch('scripts.research.homog_replay_collect.expected_label', return_value={'measured_error_um': 3.}):
                with self.assertRaises(T3HoldError):
                    recorder.observe()
            details = read(output/'hold_failure.json')
            self.assertEqual(details['code'], 'HOLD_ENTRY_NOT_SETTLED')
            self.assertEqual(details['frame'], 400)
            self.assertEqual(details['observation_calls'], 1)
            self.assertTrue((output/'hold_failure_rgb.png').is_file())
            self.assertEqual(len(recorder.rows), 0)
            self.assertEqual(recorder.index, 400)
        finally:
            recorder.close()

    def test_candidate_is_committed_only_after_an_accepted_tick(self):
        scene, intent = fixture()
        q = np.array([.1, .2, .02])
        command, diag = intent.preview_observed(q, [.01, .01, .001], frame=10, measured_error_um=2.)
        self.assertEqual(diag['phase'], 'settling')
        self.assertIsNone(intent.candidate_task_si)
        np.testing.assert_array_equal(command, np.zeros(3))
        intent.commit(command)
        np.testing.assert_array_equal(intent.candidate_task_si, q)
        moved = q + [.00001, -.00002, .0000001]
        command, _ = intent.preview_observed(moved, [.02, .02, .002], frame=11, measured_error_um=3.)
        np.testing.assert_allclose(moved + command * UNIT, q, atol=1e-16)
        np.testing.assert_array_equal(intent.candidate_task_si, q)
        intent.commit(command * .5)
        np.testing.assert_allclose(intent.candidate_task_si, moved + command * .5 * UNIT)

    def test_unsettled_entry_is_rejected_even_below_5um(self):
        _, intent = fixture()
        with self.assertRaises(T3HoldError) as cm:
            intent.preview_observed(np.zeros(3), [0, 0, 0], frame=400, measured_error_um=3.)
        self.assertEqual(cm.exception.details['code'], 'HOLD_ENTRY_NOT_SETTLED')

    def test_settled_bad_candidate_returns_to_measured_tracking(self):
        scene, intent = fixture()
        q = np.zeros(3)
        command, _ = intent.preview_observed(q, [0, 0, 0], frame=1, measured_error_um=2.)
        intent.commit(command)
        scene.motor_execution.is_settled = True
        q = np.array([.00001, 0, 0])
        command, diag = intent.preview_observed(q, [.01, 0, 0], frame=20, measured_error_um=6.)
        self.assertEqual(diag['phase'], 'tracking')
        np.testing.assert_allclose(command, [.01, 0, 0])
        intent.commit(command)
        self.assertIsNone(intent.candidate_task_si)

    def test_all_200_hold_actions_are_exactly_zero_and_gate_remains_strict(self):
        scene, intent = fixture()
        q = np.array([.1, .2, .02])
        command, _ = intent.preview_observed(q, [0, 0, 0], frame=1, measured_error_um=2.)
        intent.commit(command)
        scene.motor_execution.is_settled = True
        for frame in range(400, 600):
            command, diag = intent.preview_observed(q, [.1, .1, .01], frame=frame, measured_error_um=4.999)
            self.assertEqual(diag['phase'], 'hold')
            np.testing.assert_array_equal(command, np.zeros(3))
            intent.commit(command)
        with self.assertRaises(T3HoldError) as cm:
            intent.preview_observed(q, [0, 0, 0], frame=401, measured_error_um=5.)
        self.assertEqual(cm.exception.details['code'], 'HOLD_ERROR_EXCEEDED')

    def test_zero_command_that_retargets_motor_is_not_accepted_as_ready(self):
        scene, intent = fixture()
        q = np.zeros(3)
        command, _ = intent.preview_observed(q, [0, 0, 0], frame=1, measured_error_um=2.)
        intent.commit(command)
        scene.motor_execution.is_settled = True
        scene.motor_execution.preview = lambda q: SimpleNamespace(actual=q,
            record={'quantized_target_robot_ABCE': [2, 2, 3, 0]})
        with self.assertRaises(T3HoldError):
            intent.preview_observed(q, [0, 0, 0], frame=400, measured_error_um=2.)

    def test_nonfinite_observation_and_motor_fault_fail_closed(self):
        scene, intent = fixture()
        for error in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                intent.preview_observed(np.zeros(3), [0, 0, 0], frame=1, measured_error_um=error)
        scene.motor_execution.backend.fault = 'fault'
        with self.assertRaises(ValueError):
            intent.preview_observed(np.zeros(3), [0, 0, 0], frame=1, measured_error_um=2.)

    def test_frozen_E42_failure_and_settlement_through_real_protocol(self):
        from scripts.research.a_axis_flip_audit import static_scene
        rows = [json.loads(line) for line in (FAILURE/'E42_frames.jsonl').read_text().splitlines()]
        task = read(FAILURE/'execution_manifest.json')['collection_tasks'][42]
        evidence = {}
        for variant in ('original', 'settled'):
            scene = static_scene()
            try:
                scene.set_target_world(np.asarray(task['target_world_mm']))
                scene.set_task(np.asarray(task['initial_state_si']))
                scene.enable_motor_execution(); scene.reset_motor_time(0.)
                motor = scene.motor_execution
                intent = T3HoldIntent(scene, CAPS, HOLD_START)
                first = next(r['source_frame_index'] for r in rows if r['measured_error_um'] < 5)
                max_parity, held, first_tip, measured_vector, camera = 0., [], None, None, None
                for frame in range(600 if variant == 'settled' else 403):
                    q = scene.current_task().copy()
                    if variant == 'original' or frame < first:
                        if frame < len(rows):
                            max_parity = max(max_parity, float(np.abs(q-rows[frame]['state_si']).max()))
                        command = np.asarray(rows[frame]['executed_action']) if frame < len(rows) else np.zeros(3)
                    else:
                        if first_tip is None:
                            first_tip = scene.site_mm('needle_tip').copy()
                            measured_vector = np.asarray(rows[frame]['oct_navigation'])
                            camera = scene.data.cam_xmat[scene.camera_id('eye_internal')].reshape(3, 3).copy()
                        # Protocol fixture only: propagate a frozen measured
                        # vector with model displacement, never production labels.
                        error = np.linalg.norm(measured_vector - (camera.T @
                            (scene.site_mm('needle_tip')-first_tip)) * [1, -1, -1]) * 1000
                        command, _ = intent.preview_observed(q, [0, 0, 0],
                            frame=frame, measured_error_um=float(error))
                        if frame >= HOLD_START:
                            held.append({'frame': frame, 'measured_fixture_error_um': float(error),
                                'command': command.tolist(), 'settled': motor.settled()})
                    scene.set_task((q/UNIT+command)*UNIT, motor_dt=.05)
                    if variant == 'settled' and frame >= first:
                        intent.commit(command)
                evidence[variant] = {'final_true_error_um': float(np.linalg.norm(
                    scene.site_mm('needle_tip')-task['target_world_mm'])*1000),
                    'max_frozen_prefix_parity_abs_si': max_parity,
                    'hold_frames': len(held), 'hold_encoder_settled': all(r['settled'] for r in held),
                    'hold_zero_actions': all(r['command'] == [0., 0., 0.] for r in held),
                    'scope': 'Deterministic protocol regression with synthetic displacement observation; not RGB-D accuracy acceptance'}
                if variant == 'settled':
                    evidence[variant]['max_hold_fixture_error_um'] = max(r['measured_fixture_error_um'] for r in held)
                    self.assertEqual(len(held), 200)
                    self.assertTrue(evidence[variant]['hold_encoder_settled'])
                    self.assertTrue(evidence[variant]['hold_zero_actions'])
                    self.assertLess(evidence[variant]['max_hold_fixture_error_um'], 5.)
                else:
                    self.assertEqual(max_parity, 0.)
                    self.assertGreater(evidence[variant]['final_true_error_um'], 5.)
            finally:
                scene.close()
        write_json(OUTPUT/'E42_protocol_regression.json', evidence)


def smoke(output):
    from scripts.research.gpu_episode_pool import run_pool
    manifest = read(FAILURE/'execution_manifest.json')
    tasks = []
    for index in (40, 41, 42):
        task = manifest['collection_tasks'][index].copy()
        task.update(steps=600, hold_start=400)
        tasks.append(task)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'card.json', {'tasks': tasks, 'quality': QUALITY, 'workers': 3,
        'identity': identity(), 'all_timeouts': None,
        'scope': 'Real RGB-D/CUDA T3 hold transition smoke at production600/400 budget; no training'})
    try:
        results = run_pool({'smoke': {
            'factory': 'scripts.verification.verify_homog_teacher_intent:RecoverySmokeWorker',
            'spec': {'identity': identity()}, 'tasks': tasks}}, output/'shards', workers=3, max_workers=3,
            timeout_s=None, stage_timeout_s=None, initializer_timeout_s=None, execution_timeout_s=None)
        passed = len(results) == 3 and all(r['gate_passed'] for r in results.values())
        summary = {'status': 'passed' if passed else 'failed', 'episodes': list(results.values()),
            'scope': 'Three training targets, real RGB-D/CUDA hold-transition smoke; not dev20 or model accuracy evidence'}
        write_json(output/'summary.json', summary)
        record('T3_hold_transition_RGBD_smoke', summary['status'], output, summary)
        if not passed:
            raise SystemExit(1)
    except BaseException as error:
        if not (output/'summary.json').exists():
            summary = {'status': 'failed', 'reason': repr(error), 'scope': 'Partial smoke; no formal retry'}
            write_json(output/'summary.json', summary)
            record('T3_hold_transition_RGBD_smoke', 'failed', output, summary)
        raise


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    if args.smoke:
        return smoke(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T3HoldTests))
    summary = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'CPU_reason': 'Small deterministic controller and protocol fixtures; no training or rendering'}
    write_json(OUTPUT/'summary.json', summary)
    record('T3_hold_transition_regressions', summary['status'], OUTPUT, summary)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
