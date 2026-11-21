"""Measure single G15 packets at home without refresh, residuals or resets in flight."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mujoco_model'))

from driver_loader import driver_identity
from model_sync import DriverModel, HOME
from motor_backend import MotorBackend


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def append_log(text):
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write(('\n- ' + datetime.now().astimezone().isoformat() + '：' + text + '\n').encode())


def native(kin, robot):
    return (robot[0] * 360. / 200000., kin.b_alpha_from_robot_encoder(robot[1]),
            robot[2] / kin.ENCODER_COUNTS_PER_MM)


def ideal_tip(phi_deg, alpha_deg, radius_mm):
    phi, theta = math.radians(phi_deg), math.radians(alpha_deg + 22.)
    return (radius_mm * math.sin(theta) * math.sin(phi),
            radius_mm * math.cos(theta),
            radius_mm * math.sin(theta) * math.cos(phi))


def measure_case(model, axis, count, directory):
    plant = MotorBackend(initial_robot=HOME, needle_profile='legacy')
    kin = plant.kin
    assert kin.CONTROL_GEAR == 15
    initial = plant.robot
    initial_native = native(kin, initial)
    initial_tip = plant.tip_mm()
    model.sync(initial)
    world_tip_before = model.data.site_xpos[model.sites['needle_tip']].copy()
    handles = [0, 0, 0, 0]
    if axis != 'hold':
        handles['ABC'.index(axis)] = count
    command = kin.make_control_message(handles, 'hold')
    receipt = plant.consume(command, 0.)
    target = plant.target
    goal_native = list(initial_native)
    if axis == 'A':
        goal_native[0] += count * .0225
    elif axis == 'B':
        goal_native[1] += count * .01125
    elif axis == 'C':
        goal_native[2] += count * 112.5 / kin.ENCODER_COUNTS_PER_MM
    ideal_goal = ideal_tip(goal_native[0], goal_native[1], goal_native[2] - 14. + 4.)
    case_name = axis + ('_plus1' if count > 0 else '_minus1' if count < 0 else '_zero')
    trace_path = directory / (case_name + '_trace.jsonl')
    frame_rows = []
    previous_frame_robot = initial
    settled_streak = 0
    first_three_frames_s = None
    first_encoder_arrival_s = None
    max_fk_error_m = 0.
    with trace_path.open('w') as trace:
        for step in range(1, 251):
            now_s = step * .002
            plant.step(.002, now_s)
            diagnostics = model.sync(plant.robot)
            max_fk_error_m = max(max_fk_error_m, diagnostics['max_fk_error_m'])
            assert plant.fault is None, (case_name, plant.fault)
            assert plant.target == target, 'Target changed without a new packet'
            if plant.robot == target and first_encoder_arrival_s is None:
                first_encoder_arrival_s = now_s
            row = {'time_s': now_s, 'robot_ABCE': list(plant.robot),
                   'target_ABCE': list(target), 'status': plant.status,
                   'tip_driver_mm': list(plant.tip_mm()),
                   'tip_world_mm': (model.data.site_xpos[model.sites['needle_tip']] * 1000.).tolist()}
            trace.write(json.dumps(row, allow_nan=False) + '\n')
            if step % 25 == 0:
                stationary = plant.robot == target and plant.robot == previous_frame_robot
                settled_streak = settled_streak + 1 if stationary else 0
                if settled_streak >= 3 and first_three_frames_s is None:
                    first_three_frames_s = now_s
                frame_rows.append({**row, 'encoder_settled': stationary,
                                   'consecutive_settled_frames': settled_streak,
                                   'tip_displacement_um': math.dist(plant.tip_mm(), initial_tip) * 1000.})
                previous_frame_robot = plant.robot
    assert plant.accepted_commands == 1, 'More than one packet was accepted'
    assert len([r for r in plant.history if r['event'] == 'command']) == 1
    assert plant.robot == target and first_three_frames_s is not None, 'Not settled before watchdog'
    assert max_fk_error_m < 1e-9
    final_native = native(kin, plant.robot)
    final_world_tip = model.data.site_xpos[model.sites['needle_tip']]
    result = {'case': case_name, 'axis': axis, 'handle_count': count,
              'status': 'PASS', 'accepted_packets': plant.accepted_commands,
              'initial_robot_ABCE': list(initial), 'cumulative_handle_target_ABCE': handles,
              'quantized_target_ABCE': list(target), 'final_robot_ABCE': list(plant.robot),
              'robot_count_delta_ABCE': [b-a for a,b in zip(initial, plant.robot)],
              'actual_axis_delta_A_deg_B_deg_C_mm': [b-a for a,b in zip(initial_native, final_native)],
              'tip_rcm_radius_before_mm': math.sqrt(sum(v*v for v in initial_tip)),
              'tip_displacement_driver_um': math.dist(plant.tip_mm(), initial_tip) * 1000.,
              'tip_displacement_mujoco_um': math.dist(final_world_tip, world_tip_before) * 1e6,
              'nominal_tip_displacement_um': math.dist(ideal_goal, initial_tip) * 1000.,
              'endpoint_error_from_nominal_um': math.dist(plant.tip_mm(), ideal_goal) * 1000.,
              'error_from_quantized_target_um': math.dist(plant.tip_mm(), kin.model_tip_from_robot_encoders(target, 4., 1.)) * 1000.,
              'displacement_after_50ms_um': frame_rows[0]['tip_displacement_um'],
              'first_encoder_arrival_s': first_encoder_arrival_s,
              'first_three_50ms_settled_frames_s': first_three_frames_s,
              'max_fk_error_m': max_fk_error_m, 'final_fault': plant.fault,
              'trace_path': str(trace_path.relative_to(ROOT)), 'frames_50ms': frame_rows}
    write_json(directory / (case_name + '.json'), result)
    append_log('G15 single-packet ' + case_name + ' **PASS**；one packet, initial ABCE=' +
               str(initial) + ', delta encoder=' + str(result['robot_count_delta_ABCE']) +
               '；tip travel=' + str(result['tip_displacement_mujoco_um']) + 'um；three settled frames at ' +
               str(first_three_frames_s) + 's；command `.venv/bin/python -E -B ' +
               str(Path(__file__).relative_to(ROOT)) + ' --output-dir ' + str(directory.relative_to(ROOT)) +
               '`；dataset/checkpoint N/A, deterministic diagnostic, CPU small scalar/MuJoCo FK; evidence `' +
               str((directory/(case_name+'.json')).relative_to(ROOT)) + '`；no ROS/hardware or policy rollout.')
    print(json.dumps({k: result[k] for k in ('case', 'status', 'robot_count_delta_ABCE',
                     'actual_axis_delta_A_deg_B_deg_C_mm', 'tip_displacement_mujoco_um',
                     'displacement_after_50ms_um', 'first_three_50ms_settled_frames_s')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory/'preflight.json').exists(), 'Output already used'
    identity = driver_identity()
    assert identity['variant'] == 'g15'
    assert identity['kinematics_sha256'] == '3465daf19fd3269466f90fd77c6626149ab4da8ddde612c8f3d9a12c77d06088'
    config = json.loads((ROOT/'config.json').read_text())
    assert config['encoder_reference']['robot_counts_ABC'] == list(HOME[:3])
    assert config['encoder_reference']['needle_tip_distance_at_reference_mm'] == 4.
    case_specs = [('hold', 0), ('A', 1), ('A', -1), ('B', 1), ('B', -1), ('C', 1), ('C', -1)]
    sources = ['config.json', 'mujoco_model/driver_loader.py', 'mujoco_model/motor_backend.py',
               'mujoco_model/model_sync.py', 'mujoco_model/source_manifest_g15.json',
               str(Path(__file__).relative_to(ROOT))]
    preflight = {'timestamp': datetime.now().astimezone().isoformat(), 'driver': identity,
                 'purpose': 'minimum nonzero single integer handle packet motion; not hardware positioning accuracy',
                 'hypothesis': 'one G15 handle is executable and reproducible before watchdog; rounding changes nominal motion',
                 'main_change': 'none to control sources; one consume at t=0 followed by plant steps only',
                 'baseline': 'zero-handle packet at identical HOME; nominal G15 angle/axial formulas',
                 'tier': 'deterministic three-axis smoke diagnostic, not navigation screen10/dev20',
                 'cases': case_specs, 'dataset': None, 'checkpoint': None, 'training_rounds': 0,
                 'seed': None, 'seed_reason': 'no stochastic operation',
                 'initial_robot_ABCE': HOME, 'needle_profile': 'legacy', 'tau_s': .04,
                 'speed_limits_A_deg_B_deg_C_mm_s': [10., 10., 2.], 'timeout_s': .5,
                 'integration_step_s': .002, 'observation_period_s': .05,
                 'max_simulation_time_s_per_case': .5, 'max_integration_steps_per_case': 250,
                 'accepted_packets_per_case': 1, 'worker_count': 1, 'wall_budget_s': 30,
                 'gates': ['one accepted packet', 'no watchdog/fault before0.5s',
                           'encoder at target plus unchanged feedback for3consecutive50msframes',
                           'MuJoCo/driver FK agreement <1e-9m', 'fractional handle is rejected'],
                 'failure_policy': 'record failure and stop; no parameter changes or retries',
                 'output_dir': str(directory), 'device': 'CPU; tiny scalar kinematics and MuJoCo FK, no rendering/tensors',
                 'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    write_json(directory/'preflight.json', preflight)
    results = []
    model = DriverModel(needle_profile='legacy')
    try:
        for axis, count in case_specs:
            results.append(measure_case(model, axis, count, directory))
        fractional = MotorBackend(needle_profile='legacy')
        message = fractional.kin.make_control_message((0, 0, 0, 0), 'hold')
        message.handleEncoderA = .1
        try:
            fractional.consume(message, 0.)
        except ValueError as exc:
            fraction_rejection = str(exc)
        else:
            raise AssertionError('Fractional handle command was accepted')
        assert fractional.accepted_commands == 0 and fractional.robot == HOME
        positive = {r['axis']: r for r in results if r['handle_count'] == 1}
        nonzero = [r['tip_displacement_mujoco_um'] for r in results if r['axis'] != 'hold']
        assert results[0]['tip_displacement_mujoco_um'] == 0.
        assert all(value > 0 for value in nonzero)
        summary = {'status': 'PASS', 'driver': identity, 'completed_cases': len(results),
                   'positive_one_handle': positive, 'all_cases': results,
                   'fractional_handle_rejected': fraction_rejection,
                   'minimum_tested_single_axis_displacement_um': min(nonzero),
                   'scope': 'HOME, radius~4mm, one packet per trial, simulated first-order motor, not hardware accuracy or arbitrary XYZ resolution',
                   'strict_0p35um_navigation_success': 'not evaluated',
                   'command': sys.argv, 'output_dir': str(directory)}
        write_json(directory/'summary.json', summary)
        append_log('G15 single-packet diagnostic **COMPLETED / PASS 7/7**；three axes ±1 and zero control, single packet each, G15 pinned hash per preflight; no training/dataset/checkpoint or inference. Fractional handle rejected; baseline zero packet motion=0; acceptance and measured encoder/MuJoCo endpoint results in `' + str((directory/'summary.json').relative_to(ROOT)) + '`；CPU small scalar/MuJoCo FK, no rendering; does not establish hardware accuracy, global XYZ resolution or strict0.35um navigation success.')
        print('RESULT: ' + str(directory/'summary.json'), flush=True)
    except BaseException as exc:
        write_json(directory/'failure.json', {'status': 'FAIL', 'completed_cases': len(results),
                   'error': str(exc), 'traceback': traceback.format_exc(), 'results': results})
        append_log('G15 single-packet diagnostic **FAIL/PARTIAL**；completed=' + str(len(results)) +
                   '；reason=' + str(exc) + '；command=' + str(sys.argv) + '；dataset/checkpoint N/A；evidence `' +
                   str((directory/'failure.json').relative_to(ROOT)) + '`；no retry, control parameters unchanged.')
        raise


if __name__ == '__main__':
    main()
