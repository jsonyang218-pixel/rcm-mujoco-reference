"""Frozen paths and shared execution helpers for planning file 6."""
from __future__ import annotations

from datetime import datetime
import json
import sys
from pathlib import Path

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.gpu_episode_pool import run_pool, signature, write_json
from scripts.research.homog_replay_common import (
    OLD, QUALITY, ROOT, TARGETS, VAL, identity as replay_identity,
)

BASE = 'cd22e10471f76807e2b1ce307db5d38903b40d30'
BRANCH = 'feature/homog-label-fix-20261123'
SOURCE_DATA = ROOT / 'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
FIX_DATA = ROOT / 'data/homog_clean_replay_fixed_labels'
WEIGHTS = ROOT / 'outputs/homog_label_fix_20261123'
WORKFLOW = ROOT / 'runs/20261123/homog_label_fix_workflow_v1'
SOURCE_REPO = ROOT / 'runs/20261123/homog_clean_git_v1/repository'
REPORTS = (
    ROOT / 'github_reports/homog_label_fix_20261123_s1_audit',
    ROOT / 'github_reports/homog_label_fix_20261123_s2_train',
    ROOT / 'github_reports/homog_label_fix_20261123_s3_fast5_1600',
    ROOT / 'github_reports/homog_label_fix_20261123_s4_dev20',
)
TASK = ROOT.parent / '规划6.txt'
FAST5 = (0, 2, 6, 8, 10)
STEPS = 1600
LATE_DRIFT_START = 1000
CAPS = {'A': .2, 'B': .1, 'C': .03}


def read(path):
    return json.loads(Path(path).read_text())


def record(purpose, status, output, details):
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write((f'\n- {stamp}: homog_label_fix_20261123 {purpose} **{status}**; '
                      f'command `{" ".join(sys.argv)}`; {details}; output `{output}`. '
                      'Software/MuJoCo simulation only.\n').encode())


def identity():
    files = list((ROOT / 'scripts/research').glob('homog_label_fix_*.py'))
    files += [ROOT / 'scripts/verification/verify_homog_label_fix.py',
              ROOT / 'scripts/tools/prepare_homog_label_fix.py']
    return {**replay_identity(), **{
        str(path.relative_to(ROOT)): file_sha256(path) for path in files},
        'plan6': str(TASK), 'plan6_sha256': file_sha256(TASK),
        'publication_base': BASE, 'publication_branch': BRANCH}


def asset_fingerprints():
    return {
        'source_dataset': file_sha256(SOURCE_DATA / 'data/dataset.npz'),
        'source_image_paths': file_sha256(SOURCE_DATA / 'data/image_paths.json'),
        'source_metadata': file_sha256(SOURCE_DATA / 'metadata.json'),
        'source_audit': file_sha256(SOURCE_DATA / 'dataset_audit.json'),
        'source_receipt': file_sha256(SOURCE_DATA / 'dataset_receipt.json'),
        'initial_checkpoint': file_sha256(OLD),
        'validation_dataset': file_sha256(VAL / 'data/dataset.npz'),
        'targets': file_sha256(TARGETS),
        'plan6': file_sha256(TASK),
    }


def pool(groups, destination, *, workers=12, snapshot_every=5):
    return run_pool(groups, destination, workers=workers, max_workers=workers,
        timeout_s=None, stage_timeout_s=None, initializer_timeout_s=None,
        execution_timeout_s=None, snapshot_every=snapshot_every,
        admission_min_ram_mib=4096,
        resume=(Path(destination) / 'contract.json').exists())


def env_command():
    return ['env', '-u', 'PYTHONHOME', '-u', 'PYTHONPATH',
        'CUDA_VISIBLE_DEVICES=0', 'RCM_DRIVER_VARIANT=g10', 'MUJOCO_GL=egl',
        'MUJOCO_EGL_DEVICE_ID=0', 'MPLBACKEND=Agg', 'PYTHONDONTWRITEBYTECODE=1',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        str(ROOT / '.venv/bin/python'), '-E', '-B', '-u']
