"""Registered P2 smoke, paired teacher diagnostics and zero-training fast5.

Five targets are diagnostic only. This module never trains, retries, changes
targets, or launches a dependent experiment. SOP controls stage dependencies.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path('configs/v12_reachable_development_targets_0p35um.json')
CHECKPOINT = Path('outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth')
MANIFEST_SHA256 = 'fcc4872a9d5b58bb7158b36d3c0ab6d7452e4d8d747317dfb077b6e7bbc815c7'
FAST5 = (0, 2, 6, 8, 10)
QUALITY = {'offsamples': 0, 'numslices': 112, 'numstacks': 64}
VARIANTS = {'V0': ('g10', 'feedback'), 'V3': ('g10', 'continuous'),
            'V2': ('g15', 'continuous')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def record(purpose, status, path, detail):
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    command = ' '.join(sys.argv)
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write((f'\n- {stamp}: P2 {purpose} **{status}**; command `{command}`; '
                      f'{detail}; evidence `{path}`. Software/MuJoCo only.\n').encode())


def identities():
    from scripts.tools.experiment_workflow import scene_and_code
    return {**scene_and_code(), 'module_sha256': digest(__file__),
            'manifest_sha256': digest(MANIFEST), 'checkpoint_sha256': digest(CHECKPOINT)}


def gpu_check():
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or os.environ.get('MUJOCO_GL') != 'egl'
            or not torch.cuda.is_available()
            or '4090' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Physical GPU 0 / RTX 4090 / NVIDIA EGL required')
    return torch.cuda.get_device_name(0)


def aggregate(rows):
    values = np.asarray([row['final_error_um'] for row in rows], dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError('Cannot summarize missing or nonfinite endpoints')
    return {'completed_targets': len(rows), 'final_mean_um': float(values.mean()),
            'final_median_um': float(np.median(values)), 'final_max_um': float(values.max()),
            'hit_5um_count': int(np.count_nonzero(values <= 5)),
            'hit_10um_count': int(np.count_nonzero(values <= 10)),
            'strict_success_count': sum(bool(row['strict_success']) for row in rows)}


def paired(a, b):
    """Candidate minus baseline; IDs/seeds/pose and protocol must match."""
    keys = ('episode_index', 'target_id', 'seed', 'target_robot_mm')
    if len(a) != len(b) or len({row['episode_index'] for row in a}) != len(a):
        raise ValueError('Incomplete or duplicate paired targets')
    baseline = {row['episode_index']: row for row in b}
    if len(baseline) != len(b):
        raise ValueError('Duplicate baseline targets')
    result = []
    for row in a:
        other = baseline.get(row['episode_index'], {})
        if any(row.get(key) != other.get(key) for key in keys):
            raise ValueError('Target identity or seed mismatch')
        result.append({'episode_index': row['episode_index'], 'seed': row['seed'],
                       'delta_final_um': row['final_error_um'] - other['final_error_um']})
    return result


def policy_episode(evaluator, targets, index, steps, folder):
    from scripts.research.overnight_cfix_eval import validate_episode
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / 'motor_steps.jsonl').open('x') as trace:
        def observe(row):
            trace.write(json.dumps(row, allow_nan=False) + '\n')
            trace.flush()
        evaluator.step_observer = observe
        try:
            payload = evaluator.evaluate(num_episodes=1, output_dir=folder, seed=42,
                max_steps=steps, error_threshold_mm=.00035, target_episodes=targets,
                target_start=index, target_end=index + 1, progress_every=100)
        finally:
            evaluator.step_observer = None
    contract = payload['evaluation_contract']
    expected = {'version': 3 if evaluator.controller_variant == 'continuous' else 2,
                'controller_variant': evaluator.controller_variant,
                'flow_integration_steps': 4, 'num_action_samples': 1,
                'rgbd_render_quality': QUALITY,
                'rgb_preprocessing': '1920x1440 -> 224x224 bilinear',
                'success_error_threshold_mm': .00035, 'required_consecutive_success_frames': 3,
                'control_period_s': .05, 'oct_navigation_used': True,
                'ground_truth_used_for_control': False}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError('Policy contract differs from registered I0/P2/render protocol')
    episode = payload['episodes'][0]
    validate_episode(episode, targets[index], index, require_latency=True, max_steps=steps)
    records = episode['motor_records']
    if episode['motor_summary']['driver']['variant'] != os.environ['RCM_DRIVER_VARIANT']:
        raise ValueError('Policy executed with the wrong driver')
    actions = np.asarray(episode['trajectory']['actions'])
    movement = np.asarray([np.asarray(row['actual_after_task_rad_rad_m'])
                           - row['actual_before_task_rad_rad_m'] for row in records])
    active = np.abs(actions[:, 0]) > 0
    no_motion = np.abs(movement[:, 0] * 180 / np.pi) < 1e-6
    return {'episode_index': index, 'target_id': episode['target_id'], 'seed': episode['seed'],
            'target_robot_mm': episode['target_robot_mm'], 'steps': episode['steps'],
            'final_error_um': episode['final_error_mm'] * 1000,
            'strict_success': episode['success'],
            'a_nonzero_command_frames': int(active.sum()),
            'a_nonzero_command_no_motion_frames': int(np.count_nonzero(active & no_motion)),
            'a_nonzero_command_no_motion_fraction': float(no_motion[active].mean()) if active.any() else None,
            'driver': episode['motor_summary']['driver'], 'result_path': str(folder / 'evaluation_results.json')}


def teacher_episode(targets, index, teacher, folder):
    from rcm.scene_v9 import RCMSceneV9
    from rcm.strict_rgbd import StrictRGBDSensor
    from rcm.visual_servo import visual_servo_settings
    from rcm.control_target import ContinuousControlTarget, TASK_TO_DEG_DEG_MM
    from scripts.research.motor_perception_gates import oracle_command

    scene, sensor = RCMSceneV9(), None
    target = targets[index]
    world = scene.robot_to_world(target['reachable_target_robot_mm'])
    episode = {'episode_index': index, 'target_id': target['id'], 'seed': 42 + index,
               'target_robot_mm': target['reachable_target_robot_mm'], 'teacher': teacher,
               'status': 'running', 'trajectory': []}
    try:
        np.random.seed(42 + index)
        scene.set_task(np.zeros(3))
        scene.set_target_world(world)
        scene.enable_motor_execution()
        motor = scene.motor_execution
        control = ContinuousControlTarget(scene)
        settings = visual_servo_settings(scene)
        settings['rgbd_render_quality'] = QUALITY.copy()
        if teacher == 'rgbd':
            sensor = StrictRGBDSensor(scene, settings)
        velocity = np.zeros(3)
        streak, success = 0, False
        for step in range(1600):
            q = scene.current_task()
            true_tip = scene.site_mm('needle_tip')
            error_um = float(np.linalg.norm(world - true_tip) * 1000)
            settled = bool(motor.settled())
            streak = streak + 1 if error_um <= .35 and settled else 0
            row = {'step': step, 'error_um': error_um, 'encoder_settled': settled,
                   'state_deg_deg_mm': (q * TASK_TO_DEG_DEG_MM).tolist()}
            episode['trajectory'].append(row)
            if streak >= 3:
                success = True
                break
            if sensor is None:
                measured_tip, measured_target = true_tip, world
            else:
                sensor.observation_time_s = motor.time_s
                observation = sensor.observe()
                geometry = observation.get('geometry', {})
                measured_tip, measured_target = geometry.get('tip_world_mm'), geometry.get('target_world_mm')
                if measured_tip is None or measured_target is None:
                    raise ValueError(f'Strict RGB-D geometry missing at step {step}: {geometry.get("target_error")}')
                quality_um = sum(float(geometry.get(key, {}).get('fit_quality_mm', float('inf')))
                                 for key in ('tip_quality', 'target_quality')) * 1000
                if not np.isfinite(quality_um) or quality_um > settings['max_geometry_fit_quality_um']:
                    raise ValueError(f'Strict RGB-D geometry quality rejected at step {step}: {quality_um} um')
                if not np.all(np.isfinite([measured_tip, measured_target])):
                    raise ValueError('Nonfinite RGB-D geometry')
                row['geometry_quality_um'] = quality_um
            control.synchronize()
            intent_task = control.intent_deg_deg_mm / TASK_TO_DEG_DEG_MM
            candidate, _, _ = oracle_command(scene, q, measured_tip, measured_target, velocity, intent_task)
            increment = (candidate - intent_task) * TASK_TO_DEG_DEG_MM
            selected = None
            for scale in (1., .5, .25, .125, .0625, 0.):
                try:
                    selected = control.preview(increment, scale)
                except ValueError:
                    continue
                break
            if selected is None:
                raise ValueError('Teacher rejected all action scales')
            accepted = control.commit(selected)
            velocity = accepted / TASK_TO_DEG_DEG_MM / motor.control_period_s
            row['accepted_delta_deg_deg_mm'] = accepted.tolist()
            if step == 0 or (step + 1) % 100 == 0:
                print(f'{teacher}/E{index:02d} {step + 1}/1600 error={error_um:.6f} um', flush=True)
        episode.update(status='completed', strict_success=success, steps=len(episode['trajectory']),
                       final_error_um=float(np.linalg.norm(world - scene.site_mm('needle_tip')) * 1000),
                       driver=motor.driver_identity, motor_summary=motor.summary(), motor_records=motor.records)
    except BaseException as error:
        episode.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure=repr(error))
        raise
    finally:
        if sensor is not None:
            sensor.close()
        scene.close()
        write(folder / f'E{index:02d}.json', episode)
        record(f'{teacher}/E{index:02d}', episode['status'], folder / f'E{index:02d}.json',
               f'max1600 x50ms; seed={42 + index}; no checkpoint/dataset; final_um={episode.get("final_error_um")}; '
               f'strict={episode.get("strict_success")}; failure={episode.get("failure")}')
    return {key: value for key, value in episode.items() if key not in ('trajectory', 'motor_records')}


def compare_teachers(workflow):
    result = {}
    for driver in ('g10', 'g15'):
        oracle, rgbd = [read(workflow / f's2_{driver}_{teacher}' / 'summary.json')
                        for teacher in ('oracle', 'rgbd')]
        if any(row['completed_targets'] != 5 or row['status'] != 'completed' for row in (oracle, rgbd)):
            raise ValueError('Teacher evaluation incomplete')
        if oracle['identity'] != rgbd['identity'] or oracle['protocol'] != rgbd['protocol']:
            raise ValueError('Teacher provenance/protocol differs')
        pairs = paired(rgbd['episodes'], oracle['episodes'])
        excess = np.asarray([row['delta_final_um'] for row in pairs])
        result[driver] = {'paired': pairs, 'mean_excess_um': float(excess.mean()),
                          'max_excess_um': float(excess.max()),
                          'gate2b_passed': bool(excess.mean() <= 1. and excess.max() <= 3.),
                          'gate': 'mean RGB-D-minus-Oracle <=1 um AND max per-target excess <=3 um'}
    return {'status': 'completed', 'drivers': result,
            'gate2b_passed': all(item['gate2b_passed'] for item in result.values()),
            'step3_allowed': True, 'scope': 'diagnostic fast5; no global physical lower bound established'}


def compare_policy(workflow):
    runs = {variant: read(workflow / f's3_{variant.lower()}' / 'summary.json') for variant in VARIANTS}
    base = runs['V0']
    for run in runs.values():
        if run['status'] != 'completed' or run['completed_targets'] != 5:
            raise ValueError('Policy evaluation incomplete')
        if run['identity'] != base['identity'] or run['protocol'] != base['protocol']:
            raise ValueError('Policy provenance/protocol differs')
    pairs = {key: paired(run['episodes'], base['episodes']) for key, run in runs.items()}
    winner = min(runs, key=lambda key: (runs[key]['final_mean_um'], runs[key]['final_max_um'], key))
    improvements = {key: {'mean_delta_vs_v0_um': float(np.mean([row['delta_final_um'] for row in rows])),
                          'paired': rows} for key, rows in pairs.items()}
    # Compare to the Oracle at the same 1000-step budget, not its 1600-step
    # endpoint. A teacher that strictly finished earlier retains that endpoint.
    excess = {}
    for variant, run in runs.items():
        driver = VARIANTS[variant][0]
        oracle_summary = read(workflow / f's2_{driver}_oracle' / 'summary.json')
        if oracle_summary['identity'] != run['identity']:
            raise ValueError('Oracle and policy input identities differ')
        baseline_rows = []
        for row in oracle_summary['episodes']:
            episode = read(workflow / f's2_{driver}_oracle' / f'E{row["episode_index"]:02d}.json')
            if len(episode['trajectory']) > 1000:
                error_um = episode['trajectory'][1000]['error_um']
            elif episode['strict_success']:
                error_um = episode['final_error_um']
            else:
                raise ValueError('Oracle lacks a complete 1000-step prefix')
            baseline_rows.append(dict(row, final_error_um=error_um))
        rows = paired(run['episodes'], baseline_rows)
        excess[variant] = {'oracle_budget_steps': 1000, 'paired': rows,
                           'mean_policy_minus_oracle_um': float(np.mean([row['delta_final_um'] for row in rows]))}
    return {'status': 'completed', 'best_variant': winner, 'runs': runs, 'comparisons': improvements,
            'same_budget_oracle_comparison': excess,
            'winner_rule': 'lowest mean final error; tie: lowest maximum then variant name',
            'strict_success_not_regressed': runs[winner]['strict_success_count'] >= base['strict_success_count'],
            'phase3_status': 'pending complete training card; not launched by this diagnostic workflow',
            'scope': 'fast5 1000-step diagnostic; not screen10/dev20 or independent final_test'}


def plot_policy(result, folder):
    from matplotlib import pyplot as plt
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for column, (variant, run) in enumerate(result['runs'].items()):
        for row in run['episodes']:
            episode = read(row['result_path'])['episodes'][0]
            error = np.asarray(episode['trajectory']['errors']) * 1000
            axes[column].plot(np.arange(len(error)), error, label=f'E{row["episode_index"]:02d}')
            rows.append({'variant': variant, 'episode_index': row['episode_index'], 'seed': row['seed'],
                         'final_error_um': row['final_error_um'], 'minimum_error_um': float(error.min()),
                         'strict_success': row['strict_success']})
        axes[column].set(title=variant, xlabel='Control step', yscale='log')
        axes[column].legend()
    axes[0].set_ylabel('Tip error (um)')
    figure.tight_layout()
    figure.savefig(folder / 'error_curves_all.png', dpi=150)
    plt.close(figure)
    with (folder / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(FAST5))
    for column, variant in enumerate(VARIANTS):
        selected = [row for row in rows if row['variant'] == variant]
        axis.bar(x + (column - 1) * .25, [row['final_error_um'] for row in selected],
                 width=.25, label=variant)
    axis.set(xticks=x, xticklabels=[f'E{i:02d}' for i in FAST5], ylabel='Final error (um)')
    axis.legend()
    figure.tight_layout()
    figure.savefig(folder / 'best_and_final_error.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'teacher', 'policy', 'compare-teachers', 'compare-policy'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='V0')
    parser.add_argument('--driver', choices=('g10', 'g15'), default='g15')
    parser.add_argument('--teacher', choices=('oracle', 'rgbd'), default='oracle')
    parser.add_argument('--workflow-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    os.environ['RCM_DRIVER_VARIANT'] = (args.driver if args.mode == 'teacher' else VARIANTS[args.variant][0])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'mode': args.mode, 'episodes': []}
    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f'received {signal.Signals(signum).name}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    try:
        before = identities()
        if before['manifest_sha256'] != MANIFEST_SHA256:
            raise ValueError('Frozen manifest changed')
        result['identity'] = before
        if args.mode.startswith('compare-'):
            if args.workflow_dir is None:
                raise ValueError('Comparison requires workflow-dir')
            result.update(compare_teachers(args.workflow_dir) if args.mode == 'compare-teachers'
                          else compare_policy(args.workflow_dir))
            if args.mode == 'compare-policy':
                plot_policy(result, args.output_dir)
        else:
            targets = read(MANIFEST)['targets']
            indices = (0, 1, 2) if args.mode == 'smoke' else FAST5
            steps = 120 if args.mode == 'smoke' else (1600 if args.mode == 'teacher' else 1000)
            result['protocol'] = {'indices': list(indices), 'seeds': [42 + i for i in indices],
                                  'max_steps': steps, 'control_period_s': .05,
                                  'error_threshold_um': .35, 'settled_frames': 3,
                                  'render_quality': QUALITY, 'tier': 'smoke' if args.mode == 'smoke' else 'diagnostic_fast5'}
            result['device'] = gpu_check()
            evaluator = None
            if args.mode != 'teacher':
                from lerobot_training.eval_policy import PolicyEvaluator
                evaluator = PolicyEvaluator(str(CHECKPOINT), device='cuda', num_action_samples=1,
                    flow_integration_steps=4, measure_inference=True,
                    controller_variant=VARIANTS[args.variant][1], rgbd_render_quality=QUALITY.copy())
            for index in indices:
                if args.mode == 'teacher':
                    row = teacher_episode(targets, index, args.teacher, args.output_dir)
                else:
                    folder = args.output_dir / f'E{index:02d}'
                    row = policy_episode(evaluator, targets, index, steps, folder)
                    record(f'{args.mode}/{args.variant}/E{index:02d}', 'completed', row['result_path'],
                           f'checkpoint={CHECKPOINT}; sha256={before["checkpoint_sha256"]}; '
                           f'seed={42 + index}; max_steps={steps}; final_um={row["final_error_um"]}; strict={row["strict_success"]}')
                result['episodes'].append(row)
                write(args.output_dir / 'partial_summary.json', result)
                print(f'Result: {args.output_dir}; completed={len(result["episodes"])}/{len(indices)}', flush=True)
            result.update(aggregate(result['episodes']), status='completed')
        if identities() != before:
            raise ValueError('Code, scene, checkpoint or manifest changed during evaluation')
        write(args.output_dir / 'summary.json', result)
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        write(args.output_dir / 'partial_summary.json', result)
        raise
    finally:
        write(args.output_dir / 'status.json', result)
        record(args.mode, result['status'], args.output_dir / 'status.json',
               f'variant={args.variant}; teacher={args.teacher}; driver={args.driver}; '
               f'completed={len(result["episodes"])}; checkpoint={CHECKPOINT}; no training; failure={result.get("error")}')
        print(f'Stage {result["status"]}: {args.output_dir / "status.json"}', flush=True)


if __name__ == '__main__':
    main()
