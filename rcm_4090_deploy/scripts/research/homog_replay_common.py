"""Frozen contracts for the authorized homogeneous-render replay experiment."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_common import (
    DATA as SOURCE_DATA, OLD, ROOT, VAL, TARGETS, FAST5, UNIT, QUALITY,
    read, atomic_npz, gpu_check, code_identity as previous_identity,
)
from scripts.research.gpu_episode_pool import run_pool, signature, write_json

BASE = '71040dd33a48644da662dc92ad28053a2a326871'
BRANCH = 'feature/clean-render-homog-replay-20261123'
TASK = ROOT.parent / '规划5.txt'
DATA = ROOT / 'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
WEIGHTS = ROOT / 'outputs/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
REPORTS = tuple(ROOT / f'github_reports/homog_mixed_1600_600_12_hold_v2_20261123_s{i}_{s}' for i, s in (
    (1, 'collection'), (2, 'train'), (3, 'fast5_1600'), (4, 'dev20')))
PUB = ROOT / 'runs/20261123/homog_clean_git_v1'
WORKFLOW = ROOT / 'runs/20261123/homog_clean_workflow_mixed_1600_600_12_hold_v6'
CAPS = np.array([.2, .1, .03])
STEPS = 1600
COLLECTION_STEPS = {'T1': 1600, 'T2': 600, 'T3': 600}
HOLD_START = COLLECTION_STEPS['T3'] - 200
WORKERS = 12
LATE_DRIFT_START = 1000
ABSOLUTE_DRIFT_KEY = f'absolute_drift_{LATE_DRIFT_START}_to_{STEPS}_um'
TEACHER_TOLERANCE_UM = .1


def collection_steps(regime):
    return COLLECTION_STEPS[regime]


def validate_collection_budgets(tasks):
    """Reject a long-approach budget that cannot even reach the recording shell."""
    minimum = {}
    for task in tasks:
        steps = collection_steps(task['regime'])
        if task['regime'] == 'T1':
            gap_m = abs(task['goal_state_si'][2] - task['initial_state_si'][2])
            approach = int(np.ceil(max(0., gap_m - .0002) / (CAPS[2] * .001)))
            # One two-observation/sixteen-action contiguous training window.
            required = approach + 17
            if steps < required:
                raise ValueError(f"{task['id']}: {steps}steps cannot cover20mm approach and near-target window; minimum{required}")
            minimum[task['id']] = required
        if task['regime'] == 'T3' and (HOLD_START < 1 or steps - HOLD_START != 200):
            raise ValueError('T3 must retain exactly200 final explicit hold actions')
    return minimum


def record(purpose, status, output, details):
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write((f'\n- {stamp}: homog_replay_20261123 {purpose} **{status}**; '
                      f'command `{" ".join(sys.argv)}`; {details}; output `{output}`. '
                      'Software/MuJoCo simulation only.\n').encode())


def identity():
    files = list((ROOT / 'scripts/research').glob('homog_replay_*.py'))
    files += [ROOT / 'scripts/verification/verify_homog_replay.py',
              ROOT / 'scripts/verification/verify_homog_teacher_intent.py',
              ROOT / 'scripts/verification/verify_homog_t3_hold.py',
              ROOT / 'scripts/tools/prepare_homog_hold_repair.py',
              ROOT / 'scripts/tools/prepare_teacher_intent_repair.py',
              ROOT / 'scripts/tools/prepare_homog_mixed_budget.py',
              ROOT / 'scripts/research/a_axis_flip_audit.py',
              ROOT / 'scripts/research/a_axis_flip_pipeline.py',
              ROOT / 'rcm/fundus.py', ROOT / 'rcm/visual_servo.py',
              ROOT / 'lerobot_training/train_diffusion_policy.py']
    return {**previous_identity(), **{str(p.relative_to(ROOT)): file_sha256(p) for p in files}}


def pool(groups, output, *, snapshot_every=5):
    return run_pool(groups, output, workers=WORKERS, max_workers=WORKERS,
        timeout_s=None, stage_timeout_s=None, initializer_timeout_s=None,
        execution_timeout_s=None, snapshot_every=snapshot_every,
        admission_min_ram_mib=4096,
        resume=(Path(output) / 'contract.json').exists())


def make_teacher(scene):
    from rcm.visual_servo import VisualServoController, visual_servo_settings
    settings = visual_servo_settings(scene)
    settings.update(perception_backend='strict_rgbd', position_feedback='geometry',
        sample_period_s=.05, rgbd_render_quality=QUALITY.copy(),
        position_tolerance_um=TEACHER_TOLERANCE_UM, save_diagnostics=False)
    return VisualServoController(scene, settings=settings)


def original_indices(dataset):
    """Only full original trajectories; no Plan-2 windows or overlays."""
    rows = np.asarray(dataset.valid_indices, dtype=np.int64)
    episode = np.asarray(dataset.episode_indices)
    return np.flatnonzero(episode[rows] < 839)


def direction_gate(snapshot, fast):
    return (snapshot['axis_reverse_snapshot_counts']['A'] <= 2
        and snapshot['first_order_toward_fraction'] > .70
        and fast[ABSOLUTE_DRIFT_KEY] < 15)


def dev20_gate(best):
    return best['final_mean_um'] < 35 and best[ABSOLUTE_DRIFT_KEY] < 15


def quality_gate(audit):
    return (audit['sampled_frames'] == 200
        and audit['oct_truth_median_error_um'] < .2
        and all(v < .02 for v in audit['axis_reverse_fractions'].values())
        and audit['RGB_OCT_render_provenance_mismatch'] is False
        and audit['valid_window_contract_pass_fraction'] == 1.
        and audit['all_regimes_complete']
        and audit['T3_hold_frames_below_5um'] >= 4000)


def check_identity(expected):
    if identity() != expected:
        raise ValueError('Frozen source/scene identity changed; refusing mixed-provenance execution')


def env_command():
    return ['env', '-u', 'PYTHONHOME', '-u', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES=0',
        'RCM_DRIVER_VARIANT=g10', 'MUJOCO_GL=egl', 'MUJOCO_EGL_DEVICE_ID=0',
        'MPLBACKEND=Agg', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        str(ROOT / '.venv/bin/python'), '-E', '-B', '-u']
