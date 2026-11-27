"""Recorded command diagnostics and gated fixed-observation CUDA sampling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.clean_oct_pipeline import identity
from scripts.research.gpu_episode_pool import record, run_pool, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/20261121/clean_oct_dev20_authorized_v1/dev20'
SNAPSHOTS = (800, 1000, 1200)  # Actions already executed; zero-based trace index.
INDICES = (0, 8, 10)
SEEDS = tuple(range(2026112200, 2026112232))
UNIT = np.array([np.pi / 180, np.pi / 180, .001])
TO_UNITS = 1 / UNIT


def source_inputs():
    summary = json.loads((SOURCE / 'summary.json').read_text())
    contract = json.loads((SOURCE / 'episodes/contract.json').read_text())
    group = next(iter(contract['groups'].values()))
    if (summary['status'] != 'completed' or summary['completed_targets'] != 20
            or summary['variant'] != 'V0'
            or group['spec']['checkpoint_sha256'] != file_sha256(summary['checkpoint'])
            or group['spec']['identity'] != identity()):
        raise ValueError('source completion/checkpoint/code/scene identity mismatch')
    rows = {row['episode_index']: row for row in summary['episodes']}
    if sorted(rows) != list(range(20)) or any(rows[i]['seed'] != 42 + i for i in rows):
        raise ValueError('source target/seed coverage differs')
    return summary, group, rows


def segment_statistics(episode, start, end):
    """Keep each conversion stage distinct; quantities are degrees/degrees/mm."""
    trajectory = episode['trajectory']
    records = episode['motor_records'][start:end]
    raw = np.asarray(trajectory['raw_actions'][start:end], dtype=float)
    accepted = np.asarray(trajectory['actions'][start:end], dtype=float)
    before = np.asarray([r['actual_before_task_rad_rad_m'] for r in records]) * TO_UNITS
    after = np.asarray([r['actual_after_task_rad_rad_m'] for r in records]) * TO_UNITS
    requested = np.asarray([r['requested_task_rad_rad_m'] for r in records]) * TO_UNITS
    commanded = np.asarray([r['commanded_task_rad_rad_m'] for r in records]) * TO_UNITS
    velocities = np.asarray(trajectory['velocities'][start:end], dtype=float)
    stages = {'raw': raw, 'accepted': accepted, 'requested_delta': requested - before,
              'quantized_target_delta': commanded - before, 'actual_delta': after - before,
              'target_quantization_error': commanded - requested}
    if (len(raw) != end - start or velocities.shape != raw.shape
            or not np.isfinite(velocities).all()
            or not all(np.isfinite(v).all() for v in stages.values())):
        raise ValueError('incomplete/nonfinite command segment')
    stats = {name: {'sum': values.sum(0).tolist(), 'mean': values.mean(0).tolist(),
                    'std': values.std(0).tolist()} for name, values in stages.items()}
    conditioned = {}
    # Fixed bins; no outcome-driven bin selection. A/B in degrees, C in mm.
    for axis, label in enumerate(('A_deg', 'B_deg', 'C_mm')):
        groups = {}
        for sign, sign_mask in [('negative', accepted[:, axis] < 0),
                                ('zero', accepted[:, axis] == 0),
                                ('positive', accepted[:, axis] > 0)]:
            for low, high in ((0., .001), (.001, .003), (.003, .01), (.01, float('inf'))):
                mask = sign_mask & (np.abs(accepted[:, axis]) >= low) & (np.abs(accepted[:, axis]) < high)
                if not mask.any():
                    continue
                groups[f'{sign}_abs_{low:g}_{high:g}'] = {
                    'count': int(mask.sum()),
                    **{name + '_mean': float(values[mask, axis].mean()) for name, values in stages.items()},
                    'actual_zero_fraction': float(np.mean(np.abs(stages['actual_delta'][mask, axis]) < 1e-6)),
                    'mean_state': float(before[mask, axis].mean()),
                    'mean_velocity_per_s': float(velocities[mask, axis].mean())}
        active = np.abs(accepted[:, axis]) > 0
        conditioned[label] = {'bins': groups,
            'active_no_motion_fraction': float(np.mean(np.abs(stages['actual_delta'][active, axis]) < 1e-6)) if active.any() else None,
            'accepted_actual_sign_disagreement_fraction': float(np.mean(
                accepted[active, axis] * stages['actual_delta'][active, axis] < -1e-12)) if active.any() else None}
    return {'start': start, 'end_exclusive': end, 'units': 'A_deg,B_deg,C_mm',
            'statistics': stats, 'conditioned': conditioned,
            'state_mean_deg_deg_mm': before.mean(0).tolist(),
            'velocity_mean_deg_deg_mm_per_s': velocities.mean(0).tolist(),
            'velocity_std_deg_deg_mm_per_s': velocities.std(0).tolist(),
            'requested_minus_accepted_max_abs': np.max(np.abs(stages['requested_delta'] - accepted), axis=0).tolist()}


def command_analysis(args):
    summary, _, rows = source_inputs()
    evidence, hashes = {}, {}
    for index, row in rows.items():
        path = Path(row['result_path'])
        before_hash = file_sha256(path)
        episode = json.loads(path.read_text())['episodes'][0]
        evidence[f'E{index:02d}'] = {f'{start}_{end}': segment_statistics(episode, start, end)
                                     for start, end in ((600, 1000), (1000, 1600))}
        hashes[str(path)] = file_sha256(path)
        if hashes[str(path)] != before_hash:
            raise ValueError(f'source trace changed during analysis: {path}')
        print(f'Command chain E{index:02d}: analyzed both registered segments', flush=True)
    write_json(args.artifact_dir / 'command_execution.json', evidence)
    write_json(args.artifact_dir / 'source_hashes.json', hashes)
    return {'status': 'completed', 'analyzed_targets': 20, 'segments': 40,
            'checkpoint': summary['checkpoint'], 'variant': 'V0/G10',
            'cpu_reason': 'Small file/array statistics; no model inference or rollout',
            'evidence': str(args.artifact_dir / 'command_execution.json'),
            'interpretation': 'Associations only: command sum and actual net motion need not match; no motor bug established.'}


def reconstruction_check(observed, recorded, reproduced, raw):
    if (np.asarray(observed).shape != (2, 3) or np.asarray(recorded).shape != (2, 3)
            or np.asarray(reproduced).shape != (3,) or np.asarray(raw).shape != (3,)):
        raise ValueError('fixed-input reconstruction requires two OCT frames and one ABC action')
    oct_error = float(np.max(np.abs(np.asarray(observed) - recorded)))
    action_error = float(np.max(np.abs(np.asarray(reproduced) - raw)))
    if not np.isfinite(oct_error + action_error) or oct_error > 1e-6 or action_error > 1e-6:
        raise ValueError(f'fixed-input reconstruction failed: OCT max={oct_error}mm, action max={action_error}deg/deg/mm')
    return {'oct_max_abs_difference_mm': oct_error,
            'original_action_max_abs_difference_deg_deg_mm': action_error,
            'tolerance': 1e-6, 'passed': True,
            'rgb_limitation': 'Original RGB was not saved. Re-rendered image checked indirectly by OCT and original seeded action; pixel equality is not claimed.'}


class FixedInputWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        import torch
        from lerobot_training.eval_policy import PolicyEvaluator
        from rcm.scene_v9 import RCMSceneV9
        from rcm.visual_servo import VisualServoController, visual_servo_settings
        from scripts.research.clean_oct_workers import egl_identity
        torch.set_num_threads(1)
        self.torch, self.spec = torch, spec
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('MUJOCO_GL') != 'egl'
                or not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0)):
            raise RuntimeError('fixed-input diagnostic requires physical GPU0 RTX4090')
        if file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('checkpoint changed before diagnostic worker initialization')
        self.evaluator = PolicyEvaluator(spec['checkpoint'], device='cuda', num_action_samples=1,
            flow_integration_steps=4, controller_variant='feedback', measure_inference=True,
            rgbd_render_quality={'offsamples': 0, 'numslices': 112, 'numstacks': 64})
        if (not self.evaluator.direct_224_preprocessing or self.evaluator.n_obs_steps != 2
                or self.evaluator.n_action_execute_steps != 1
                or not self.evaluator.include_velocity or not self.evaluator.include_previous_action):
            raise ValueError('fixed-input reconstruction requires original two-frame/execute1/direct224 contract')
        config = json.loads((ROOT / 'config.json').read_text())
        config['motor_execution'] = {**config.get('motor_execution', {}), 'enabled': False}
        self.scene = RCMSceneV9(config=config)
        settings = visual_servo_settings(self.scene)
        settings.update(perception_backend='strict_rgbd', position_feedback='geometry', sample_period_s=.05,
            rgbd_render_quality={'offsamples': 0, 'numslices': 112, 'numstacks': 64}, save_diagnostics=False)
        self.teacher_factory = VisualServoController
        self.settings = settings
        self.teacher = None
        self.egl_identity = egl_identity

    def run(self, task, output):
        torch, evaluator, scene = self.torch, self.evaluator, self.scene
        source = Path(task['result_path'])
        if file_sha256(source) != task['source_sha256']:
            raise ValueError('source trace changed')
        episode = json.loads(source.read_text())['episodes'][0]
        trace = episode['trajectory']
        # A persistent pool worker may receive another episode, whose clock
        # starts earlier. Give every task fresh sensor/provider timestamp state.
        if self.teacher is not None:
            self.teacher.sensor.close()
        self.teacher = self.teacher_factory(scene, settings=self.settings)
        self.egl = self.egl_identity()
        print('Fixed-input CUDA/EGL worker', self.egl, flush=True)
        scene.set_target_world(np.asarray(episode['target_world_mm']))
        records, actions_by_step = [], {}
        for step in SNAPSHOTS:
            images, observed_oct, observations = [], [], []
            for j in (step - 1, step):
                scene.set_task(np.asarray(trace['states'][j]) * UNIT)
                self.teacher.sensor.observation_time_s = float(trace['simulation_time_s'][j])
                observation = self.teacher.sensor.observe()
                navigation = observation.get('oct_navigation_depth')
                delta = None if navigation is None else navigation.get('delta_xyz_m')
                if delta is None:
                    raise ValueError('reconstructed observation has no valid OCT')
                observed_oct.append((np.asarray(delta) * 1000).tolist())
                observations.append(observation)
                images.append(evaluator.preprocess_image(observation['rgb']))
            image_tensor = torch.stack(images).unsqueeze(0).to('cuda')
            state_tensor = torch.tensor(trace['states'][step - 1:step + 1], dtype=torch.float32, device='cuda').unsqueeze(0)
            oct_values = np.column_stack((np.asarray(trace['oct_navigation_mm'][step - 1:step + 1]),
                                         np.asarray(trace['oct_navigation_valid'][step - 1:step + 1], dtype=float)))
            if not np.all(oct_values[:, 3] == 1):
                raise ValueError('registered snapshot has invalid original OCT')
            oct_tensor = torch.tensor(oct_values, dtype=torch.float32, device='cuda').unsqueeze(0)
            velocity = (state_tensor[:, -1] - state_tensor[:, -2]) / .05
            previous_action = torch.tensor(trace['actions'][step - 1], dtype=torch.float32, device='cuda').unsqueeze(0)

            def sample(seed, skip=0):
                generator = torch.Generator(device='cuda').manual_seed(int(seed))
                # Original step0 fills history without sampling. Each subsequent
                # Flow call draws one [1,16,3] tensor; preserve draw boundaries.
                for _ in range(skip):
                    torch.randn(1, evaluator.n_action_steps, 3, device='cuda', dtype=torch.float32, generator=generator)
                with torch.no_grad():
                    chunk = evaluator.model.sample(image_tensor, state_tensor, oct_tensor, velocity,
                                                   previous_action, steps=4, generator=generator)
                return chunk[0, 0].cpu().numpy().astype(float)

            original_action = sample(episode['seed'], step - 1)
            np.savez_compressed(output / f'state_{step}_reconstruction.npz',
                images=image_tensor.cpu().numpy(), states=state_tensor.cpu().numpy(),
                oct=oct_tensor.cpu().numpy(), previous_action=previous_action.cpu().numpy(),
                velocity=velocity.cpu().numpy(), reproduced_action=original_action,
                recorded_action=np.asarray(trace['raw_actions'][step]))
            try:
                validation = reconstruction_check(observed_oct,
                    np.asarray(trace['oct_navigation_mm'][step - 1:step + 1]),
                    original_action, trace['raw_actions'][step])
            except ValueError as error:
                write_json(output / 'reconstruction_failure.json', {'step': step, 'reason': str(error),
                    'observed_oct_mm': observed_oct,
                    'recorded_oct_mm': trace['oct_navigation_mm'][step - 1:step + 1],
                    'reproduced_action': original_action.tolist(), 'recorded_action': trace['raw_actions'][step]})
                raise
            samples = np.stack([sample(seed) for seed in SEEDS])
            actions_by_step[str(step)] = samples
            teacher = self.teacher.geometry_action_from_observation(
                np.asarray(trace['states'][step]) * UNIT, observations[-1],
                previous_velocity=np.asarray(trace['velocities'][step]) * UNIT,
                command_q=np.asarray(trace['states'][step]) * UNIT)
            if not teacher['valid']:
                raise ValueError(f'teacher direction unavailable: {teacher.get("reason")}')
            wanted = np.asarray(teacher['action_deg_deg_mm'])
            sensor_geometry = observations[-1]['geometry']
            measured_error = np.asarray(sensor_geometry['target_world_mm']) - sensor_geometry['tip_world_mm']
            jacobian = self.teacher._model_jacobian(np.asarray(trace['states'][step]) * UNIT)
            projected = samples @ (jacobian * UNIT).T
            toward = projected @ measured_error
            records.append({'step': step, 'reconstruction': validation, 'sampling_seeds': list(SEEDS),
                'mean_action_deg_deg_mm': samples.mean(0).tolist(),
                'std_action_deg_deg_mm': samples.std(0).tolist(),
                'positive_axis_fraction': (samples > 0).mean(0).tolist(),
                'teacher_action_deg_deg_mm': wanted.tolist(),
                'teacher_scope': 'diagnostic only, uses reproduced sensor geometry; not policy control',
                'same_sign_as_teacher_fraction': (samples * wanted > 0).mean(0).tolist(),
                'first_order_toward_measured_target_fraction': float(np.mean(toward > 0)),
                'measured_error_um': float(np.linalg.norm(measured_error) * 1000),
                'original_error_um': trace['errors'][step] * 1000})
            write_json(output / 'partial_diagnostic.json', {'completed_snapshots': len(records), 'records': records})
            print(f'E{task["episode_index"]:02d} step{step}: reconstruction passed,32 samples complete', flush=True)
        if file_sha256(self.spec['checkpoint']) != self.spec['checkpoint_sha256'] or file_sha256(source) != task['source_sha256']:
            raise ValueError('checkpoint/source changed during fixed-input diagnostic')
        np.savez_compressed(output / 'actions.npz', **actions_by_step)
        report = {'episode_index': task['episode_index'], 'snapshots': len(records),
                  'samples': len(records) * len(SEEDS), 'records': records, 'egl': self.egl}
        write_json(output / 'fixed_input.json', report)
        return report

    def close(self):
        if self.teacher is not None:
            self.teacher.sensor.close()
        self.scene.close()


def fixed_input(args):
    summary, group, rows = source_inputs()
    tasks = [{'id': f'E{i:02d}', 'episode_index': i, 'result_path': rows[i]['result_path'],
              'source_sha256': file_sha256(rows[i]['result_path'])} for i in INDICES]
    spec = {**group['spec'], 'diagnosis_source_sha256': file_sha256(__file__),
            'snapshots': list(SNAPSHOTS), 'sampling_seeds': list(SEEDS)}
    groups = {'fixed': {'factory': 'scripts.research.clean_oct_drift_diagnosis:FixedInputWorker',
                        'spec': spec, 'tasks': tasks}}
    results = run_pool(groups, args.artifact_dir / 'shards', workers=3, max_workers=3,
                       timeout_s=600, stage_timeout_s=1740)
    return {'status': 'completed', 'reconstruction_passed': 1,
            'targets': len(results), 'snapshots': sum(v['snapshots'] for v in results.values()),
            'samples': sum(v['samples'] for v in results.values()),
            'checkpoint': summary['checkpoint'], 'sampling_seeds': list(SEEDS),
            'results': results, 'interpretation': 'Fixed-observation action distribution and execution records are diagnostic; no closed-loop causal claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('command', 'fixed'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.artifact_dir = args.artifact_dir.resolve()
    started = time.monotonic()
    before = identity()
    created = []
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        created.append(args.output_dir)
        args.artifact_dir.mkdir(parents=True, exist_ok=False)
        created.append(args.artifact_dir)
        report = command_analysis(args) if args.mode == 'command' else fixed_input(args)
        if identity() != before:
            raise ValueError('code/scene identity changed during diagnostic')
        report.update(elapsed_s=time.monotonic() - started, artifact_dir=str(args.artifact_dir))
        write_json(args.artifact_dir / 'summary.json', report)
        write_json(args.output_dir / 'summary.json', report)
        write_json(args.output_dir / 'status.json', {'status': 'completed'})
        record('drift_diagnosis_' + args.mode, 'completed', args.output_dir,
            f'checkpoint={report.get("checkpoint")}; training data=plan2_plus_original_train_224_20261119; '
            f'baseline=completed step2000 dev20,targets0-19 seeds42-61; no training/closed-loop; artifacts={args.artifact_dir}')
        print(f'Diagnosis {args.mode} result: {args.artifact_dir / "summary.json"}', flush=True)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report = {'status': status, 'reason': repr(error), 'mode': args.mode,
                  'artifact_dir': str(args.artifact_dir), 'elapsed_s': time.monotonic() - started}
        for directory in created:
            write_json(directory / 'status.json', report)
        record('drift_diagnosis_' + args.mode, status, args.output_dir, repr(error))
        raise


if __name__ == '__main__':
    main()
