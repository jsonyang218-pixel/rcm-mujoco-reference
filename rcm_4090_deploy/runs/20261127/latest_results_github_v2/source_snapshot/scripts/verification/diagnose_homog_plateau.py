"""Read frozen collection traces and diagnose feedback/protocol quantization stalls."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_audit import static_scene
from scripts.research.homog_replay_common import ROOT, DATA, UNIT, record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'timestamp': datetime.now().astimezone().isoformat(),
        'purpose': 'Offline causal diagnosis only; does not change or restart the running collection',
        'CPU_reason': 'Small JSON analysis and non-rendered kinematic/protocol replay; no training or GPU tensor work',
        'script_sha256': file_sha256(__file__), 'episodes': []}
    scene = static_scene()
    from motor_backend import MotorBackend
    from driver_loader import driver_identity
    from rcm.visual_servo import visual_servo_settings
    gain = visual_servo_settings(scene)['geometry_gain_per_s']
    traces = []
    try:
        for index in args.episodes:
            folder = DATA / f'shards/tasks/collection_E{index:02d}'
            receipt = json.loads((folder / 'receipt.json').read_text())
            if receipt['status'] != 'completed':
                raise ValueError('Only complete frozen trajectories may be diagnosed')
            paths = [folder / 'frames.jsonl', folder / 'policy_trajectory.json']
            for path in paths:
                if file_sha256(path) != receipt['files'][path.name]:
                    raise ValueError(f'Frozen trace hash mismatch: {path}')
            frames = [json.loads(line) for line in paths[0].read_text().splitlines()]
            ep = json.loads(paths[1].read_text())
            motors = ep['motor_records']
            truth = np.asarray([r['truth_navigation_mm'] for r in frames])
            oct_vector = np.asarray([r['oct_navigation'] for r in frames])
            states = np.asarray([r['state_si'] for r in frames])
            final_state = states[-1]
            equal_final = np.all(states == final_state, axis=1)
            differing = np.flatnonzero(~equal_final)
            start = int(differing[-1] + 1) if len(differing) else 0
            tail = frames[start:]
            last = motors[-1]
            robot = tuple(last['actual_after_robot_ABCE'])
            handles = tuple(last['handles_ABCE'])
            anchor = tuple(motors[0]['actual_before_robot_ABCE'])
            backend = MotorBackend(initial_robot=anchor, needle_profile='legacy')
            hold_packet = backend.kin.make_control_message(handles, 'hold')
            backend.consume(hold_packet, 0.)
            for tick in range(1000):
                if tick % 25 == 0:
                    backend.consume(hold_packet, tick * .002)
                backend.step(.002, (tick + 1) * .002)
            if backend.robot != robot or backend.target != robot or backend.fault is not None:
                raise ValueError('Offline frozen motor reconstruction failed')
            repeat = backend.command_for_robot_target(tuple(last['requested_robot_ABCE']))
            repeat_receipt = backend.consume(repeat, backend.now_s)
            if not repeat_receipt['duplicate'] or repeat_receipt['target'] != robot:
                raise ValueError('Recorded plateau command is not a duplicate target')
            target = np.asarray(ep['target_world_mm'])
            scene.set_target_world(target)
            scene.set_task(final_state)
            origin = scene.site_mm('needle_tip').copy()
            true_final = float(np.linalg.norm(origin - target) * 1000)
            if not np.isclose(true_final, ep['final_error_mm'] * 1000, atol=1e-7, rtol=0):
                raise ValueError('Frozen scene endpoint does not reproduce recorded error')
            probes = []
            for axis, name in enumerate('ABC'):
                for sign in (-1, 1):
                    desired = list(handles)
                    desired[axis] += sign
                    trial = copy.copy(backend)
                    trial.history = []
                    trial.consume(trial.kin.make_control_message(desired, 'hold'), trial.now_s)
                    endpoint_q = scene.task_from_encoders(trial.target[:3], reference_task=final_state)
                    scene.set_task(endpoint_q)
                    endpoint = scene.site_mm('needle_tip').copy()
                    start_time = trial.now_s
                    for tick in range(25):
                        trial.step(.002, start_time + (tick + 1) * .002)
                    q = scene.task_from_encoders(trial.robot[:3], reference_task=final_state)
                    scene.set_task(q)
                    after = scene.site_mm('needle_tip').copy()
                    probes.append({'axis': name, 'handle_delta': sign,
                        'endpoint_tip_step_um': float(np.linalg.norm(endpoint - origin) * 1000),
                        'endpoint_error_um': float(np.linalg.norm(endpoint - target) * 1000),
                        'one_packet_tip_step_um': float(np.linalg.norm(after - origin) * 1000),
                        'one_packet_error_um': float(np.linalg.norm(after - target) * 1000)})
            # Oracle truth is used solely to prove existence of better nearby legal endpoints.
            # This search is not fed to the running teacher or treated as an evaluation result.
            best = {'error_um': true_final, 'handle_delta': [0, 0, 0]}
            for delta in itertools.product(range(-4, 5), repeat=3):
                values = tuple(handles[j] + delta[j] for j in range(3)) + (handles[3],)
                robot_target = backend.kin.predicted_robot_encoders(anchor, values)
                q = scene.task_from_encoders(robot_target[:3], reference_task=final_state)
                scene.set_task(q)
                error_um = float(np.linalg.norm(scene.site_mm('needle_tip') - target) * 1000)
                if error_um < best['error_um']:
                    best = {'error_um': error_um, 'handle_delta': list(delta),
                            'robot_ABCE': list(robot_target)}
            kin = backend.kin
            step = np.array([2 * kin.CONTROL_GEAR * 3 / 4000,
                kin.CONTROL_GEAR * 3 / 4000,
                kin.C_ENCODER_COUNTS_PER_HANDLE / kin.ENCODER_COUNTS_PER_MM])
            requested = np.asarray(last['requested_robot_ABCE'])
            cumulative_float = [
                float((requested[0] - anchor[0]) / (step[0] * 200000 / 360)),
                float((kin.b_alpha_from_robot_encoder(int(requested[1])) -
                       kin.b_alpha_from_robot_encoder(anchor[1])) / step[1]),
                float((requested[2] - anchor[2]) / kin.C_ENCODER_COUNTS_PER_HANDLE)]
            row = {'episode_index': index, 'frames': len(frames), 'final_error_um': true_final,
                'frozen_state_suffix_start_step_0based': start, 'identical_suffix_frames': len(tail),
                'identical_suffix_fraction': len(tail) / len(frames),
                'suffix_unique_raw_rgb_hashes': len(set(r['raw_rgb_sha256'] for r in tail)),
                'suffix_unique_actions': len(set(tuple(r['action']) for r in tail)),
                'suffix_max_actual_tip_step_um': max(m['tip_displacement_um'] for m in motors[start:]),
                'suffix_all_targets_equal_actual': all(m['quantized_target_robot_ABCE'] == m['actual_after_robot_ABCE'] for m in motors[start:]),
                'suffix_all_teacher_commands_nonzero': all(np.any(np.asarray(r['executed_action']) != 0) for r in tail),
                'suffix_oct_truth_vector_discrepancy_um_max': float(np.linalg.norm(oct_vector[start:] - truth[start:], axis=1).max() * 1000),
                'teacher_action_deg_deg_mm': frames[-1]['executed_action'],
                'handle_step_deg_deg_mm': step.tolist(),
                'teacher_action_divided_by_handle_step': (np.asarray(frames[-1]['executed_action']) / step).tolist(),
                'requested_robot_ABCE': last['requested_robot_ABCE'], 'actual_robot_ABCE': list(robot),
                'requested_cumulative_handle_floats': cumulative_float, 'encoded_handles_ABCE': list(handles),
                'reproduced_duplicate': repeat_receipt['duplicate'], 'single_handle_probes': probes,
                'best_local_oracle_endpoint': best,
                'safety_limited_steps': ep['safety_limited_steps'], 'motor_fault': last['fault'],
                'files_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}}
            result['episodes'].append(row)
            traces.append((index, np.linalg.norm(truth, axis=1) * 1000,
                np.linalg.norm(oct_vector, axis=1) * 1000,
                np.asarray([m['tip_displacement_um'] for m in motors])))
            print(json.dumps({k: row[k] for k in ('episode_index', 'final_error_um',
                'frozen_state_suffix_start_step_0based', 'identical_suffix_fraction',
                'teacher_action_divided_by_handle_step', 'best_local_oracle_endpoint')}, ensure_ascii=False), flush=True)
        result.update(status='completed', driver=driver_identity(), geometry_gain_per_s=gain,
            dt_s=.05, unsaturated_proportional_fraction_per_tick=gain * .05,
            conclusion='Sub-handle proportional actions are quantized to the same cumulative target; feedback mode discards sub-handle residuals every tick',
            counterfactual_scope='One-packet probes and local endpoint enumeration use frozen software simulation and truth only for offline diagnosis; not new achieved closed-loop accuracy')
    except BaseException as error:
        result.update(status='failed', reason=repr(error))
        write_json(output / 'summary.json', result)
        record('plateau_diagnosis', 'failed', output, repr(error))
        raise
    finally:
        scene.close()
    write_json(output / 'summary.json', result)
    with (output / 'episodes.csv').open('w') as stream:
        keys = ['episode_index', 'final_error_um', 'frozen_state_suffix_start_step_0based',
                'identical_suffix_frames', 'identical_suffix_fraction', 'suffix_unique_raw_rgb_hashes',
                'suffix_oct_truth_vector_discrepancy_um_max']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['episodes'])
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for index, truth, oct_error, motor_step in traces:
        axes[0].plot(truth[:101], label=f'E{index:02d}')
        axes[1].plot(motor_step[:101])
    axes[0].set(ylabel='True tip error (um)')
    axes[0].legend(ncol=5)
    axes[1].set(xlabel='Control step (50 ms)', ylabel='Actual tip step (um)')
    fig.tight_layout()
    fig.savefig(output / 'plateau_first100.png', dpi=150)
    plt.close(fig)
    (output / 'diagnostic_source.md').write_text('```python\n' + Path(__file__).read_text() + '\n```\n')
    brief = {'episodes': len(result['episodes']),
        'all_duplicate_commands_reproduced': all(r['reproduced_duplicate'] for r in result['episodes']),
        'all_have_better_one_packet_neighbor': all(min(p['one_packet_error_um'] for p in r['single_handle_probes']) < r['final_error_um'] for r in result['episodes']),
        'best_oracle_endpoints_um': [r['best_local_oracle_endpoint']['error_um'] for r in result['episodes']]}
    record('plateau_diagnosis', 'completed', output, brief)
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
