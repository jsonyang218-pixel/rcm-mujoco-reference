"""Frozen contracts shared by the explicitly authorized 2026-11-23 plan."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.clean_oct_workers import QUALITY, UNIT
from scripts.research.gpu_episode_pool import run_pool, signature, write_json

ROOT = Path(os.environ.get("RCM_RUNTIME_ROOT", Path(__file__).resolve().parents[2])).resolve()
DATA = ROOT / 'data/plan2_plus_original_train_224_20261119'
VAL = ROOT / 'data/lerobot_rcm_val_acthist_20261117_224'
OLD = ROOT / 'outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth'
NEW = ROOT / 'outputs/clean_oct_cbias_20261121_resume_v2/step_002000.pth'
CLEAN = ROOT / 'data/clean_oct_cbias_20261121_resume_v2'
TARGETS = ROOT / 'configs/v12_reachable_development_targets_0p35um.json'
FROZEN = ROOT / 'runs/20261122/diagnosis/clean_oct_fixed_input_v1'
SOURCE = ROOT / 'runs/20261121/clean_oct_dev20_authorized_v1/dev20/summary.json'
AUDIT = ROOT / 'github_reports/audit_20261123_a_axis_flip'
PHASE2 = ROOT / 'github_reports/phase2_20261123_a_axis_fix_train'
PHASE3 = ROOT / 'github_reports/phase3_20261123_fast5_1600'
PHASE4 = ROOT / 'github_reports/phase4_20261123_dev20'
FAST5 = (0, 2, 6, 8, 10)
SNAP_EPISODES = (0, 8, 10)
SNAP_STEPS = (800, 1000, 1200)
SEEDS = tuple(range(2026112200, 2026112232))
EPS = 1e-10


def read(path):
    return json.loads(Path(path).read_text())


def arrays():
    with np.load(DATA / 'data/dataset.npz', allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def near_mask(a):
    return (a['oct_navigation_valid'].astype(bool)
            & np.isfinite(a['oct_navigation']).all(1)
            & (np.linalg.norm(a['oct_navigation'].astype(float), axis=1) < .2))


def record(purpose, status, output, details):
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write((f'\n- {stamp}: a_axis_flip_20261123 {purpose} **{status}**; '
                      f'command `{" ".join(sys.argv)}`; {details}; output `{output}`. '
                      'Software/MuJoCo simulation only.\n').encode())


CODE_ROOT = Path(__file__).resolve().parents[2]


def code_identity():
    from scripts.tools.experiment_workflow import scene_and_code
    files = list((CODE_ROOT / 'scripts/research').glob('a_axis_flip_*.py'))
    files += [CODE_ROOT / 'scripts/research/gpu_episode_pool.py',
              CODE_ROOT / 'scripts/research/clean_oct_workers.py',
              CODE_ROOT / 'lerobot_training/navigation_overlay.py',
              CODE_ROOT / 'lerobot_training/eval_policy.py',
              CODE_ROOT / 'lerobot_training/dit_experience_replay.py',
              CODE_ROOT / 'lerobot_training/dataset_contract.py']
    return {"targets_sha256": file_sha256(TARGETS), **scene_and_code(), **{str(p.relative_to(CODE_ROOT)): file_sha256(p) for p in files}}


def gpu_check():
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0'
            or os.environ.get('MUJOCO_GL') != 'egl' or not torch.cuda.is_available()
            or '4090' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Requires deployment venv, physical GPU0 RTX4090, NVIDIA EGL')
    torch.set_num_threads(1)
    return {'device': torch.cuda.get_device_name(0),
            'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'python': sys.version, 'executable': sys.executable}


def pool(groups, destination, *, snapshot_every=None):
    return run_pool(groups, destination, workers=3, max_workers=4,
                    timeout_s=None, stage_timeout_s=None, initializer_timeout_s=None,
                    execution_timeout_s=None, resume=(Path(destination) / 'contract.json').exists(),
                    snapshot_every=snapshot_every)


def atomic_npz(path, **values):
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **values)
    path.with_suffix('.tmp').replace(path)


def apply_abc_overlay(actions, path, dataset_sha256):
    from lerobot_training.navigation_overlay import _load
    v, rows = _load(path, dataset_sha256, len(actions))
    before, after = v['old_action'], v['new_action']
    # Conservative 50 ms teacher replay caps, with negative directions allowed.
    caps = np.array([.2, .1, .03])
    changed = before != after
    if (before.shape != (len(rows), 3) or after.shape != before.shape
            or not np.array_equal(actions[rows], before)
            or not np.isfinite(after).all() or np.any(changed & (np.abs(after) > caps + 1e-8))):
        raise ValueError('ABC overlay source/value/cap mismatch')
    result = actions.copy()
    result[rows] = after
    return result


def episode_metrics(ep, expected_index):
    tr = ep['trajectory']
    n = len(tr['actions'])
    err = np.asarray(tr['errors'], float) * 1000
    final = float(ep['final_error_mm']) * 1000
    values = np.r_[err, final]
    if (n != 1600 or len(err) != 1600 or len(ep['motor_records']) != 1600
            or ep['early_stop_reason'] is not None
            or not np.isfinite(values).all()
            or not np.isfinite(np.asarray(tr['states'])).all()
            or not np.isfinite(np.asarray(tr['actions'])).all()
            or not np.allclose(np.diff(tr['simulation_time_s']), .05, atol=1e-8, rtol=0)):
        raise ValueError(f'E{expected_index:02d}: illegal/nonfinite/incomplete 1600-step rollout')
    # Final post-action encoder state is included by evaluate_episode(False).
    terminal = ep['terminal_success']
    strict = bool(ep['success'] and terminal['encoder_settled_at_exit']
                  and terminal['consecutive_frames_at_exit'] >= 3 and final < .35)
    return {'episode_index': expected_index, 'seed': 42 + expected_index, 'actions': n,
            'final_error_um': final, 'min_error_um': float(values.min()),
            'error_1000_um': float(err[1000]),
            'drift_1000_to_1600_um': float(final - err[1000]),
            'absolute_drift_1000_to_1600_um': float(abs(final - err[1000])),
            'strict_success': strict,
            **{f'final_hit_le_{t}_um': final <= t for t in (5, 10, 30)}}


def aggregate(rows):
    if not rows:
        raise ValueError('empty evaluation')
    return {'completed_targets': len(rows),
            'final_mean_um': float(np.mean([r['final_error_um'] for r in rows])),
            'min_mean_um': float(np.mean([r['min_error_um'] for r in rows])),
            'drift_1000_to_1600_um': float(np.mean([r['drift_1000_to_1600_um'] for r in rows])),
            'absolute_drift_1000_to_1600_um': float(np.mean([r['absolute_drift_1000_to_1600_um'] for r in rows])),
            'strict_success_count': sum(r['strict_success'] for r in rows),
            **{f'final_hits_le_{t}_um': sum(r[f'final_hit_le_{t}_um'] for r in rows) for t in (5, 10, 30)},
            'episodes': rows}


class EvaluationWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        gpu_check()
        from lerobot_training.eval_policy import PolicyEvaluator
        self.spec = spec
        if file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('checkpoint changed')
        self.evaluator = PolicyEvaluator(spec['checkpoint'], device='cuda',
            num_action_samples=spec['samples'], n_action_execute_steps=1,
            flow_integration_steps=4, measure_inference=True,
            action_aggregation=spec.get('action_aggregation', 'median'),
            controller_variant=spec['controller'], rgbd_render_quality=QUALITY.copy())
        self.targets = read(TARGETS)['targets']

    def run(self, task, output):
        i = task['episode_index']
        folder = output / 'raw'
        self.evaluator.evaluate(num_episodes=1, output_dir=folder,
            target_episodes=self.targets, target_start=i, target_end=i + 1,
            seed=42, max_steps=1600, control_period_s=.05,
            error_threshold_mm=.00035, progress_every=100, stop_on_success=False,
            max_consecutive_safety_limited=None)
        path = folder / 'evaluation_results.json'
        payload = read(path)
        ep = payload['episodes'][0]
        if ep['episode_index'] != i or ep['seed'] != 42 + i:
            raise ValueError('target/seed mismatch')
        contract = payload['evaluation_contract']
        if contract['stop_on_success'] is not False:
            raise ValueError('success stopping not disabled')
        row = episode_metrics(ep, i)
        row.update(result_path=str(path), result_sha256=file_sha256(path),
                   checkpoint=self.spec['checkpoint'], samples=self.spec['samples'],
                   controller=self.spec['controller'])
        write_json(output / f'E{i:02d}.json', row)
        record(f'E{i:02d}', 'completed', output, row)
        return row

    def close(self):
        pass
