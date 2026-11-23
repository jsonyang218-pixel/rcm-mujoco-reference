"""Matched fixed-budget evaluation for the user-authorized 600-step workflow."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_common import EvaluationWorker as BaseEvaluationWorker
from scripts.research.homog_replay_common import (
    ABSOLUTE_DRIFT_KEY, LATE_DRIFT_START, QUALITY, STEPS, TARGETS,
    check_identity, pool, read, record, write_json,
)

DRIFT_KEY = f'drift_{LATE_DRIFT_START}_to_{STEPS}_um'


def episode_metrics(ep, expected_index):
    tr = ep['trajectory']
    err = np.asarray(tr['errors'], dtype=float) * 1000
    final = float(ep['final_error_mm']) * 1000
    times = np.asarray(tr['simulation_time_s'], dtype=float)
    values = np.r_[err, final]
    if (len(tr['actions']) != STEPS or len(err) != STEPS
            or len(ep['motor_records']) != STEPS or len(times) != STEPS
            or len(tr['states']) != STEPS or ep['early_stop_reason'] is not None
            or not np.isfinite(values).all() or not np.isfinite(times).all()
            or not np.isfinite(np.asarray(tr['states'])).all()
            or not np.isfinite(np.asarray(tr['actions'])).all()
            or not np.allclose(np.diff(times), .05, atol=1e-8, rtol=0)):
        raise ValueError(f'E{expected_index:02d}: illegal/nonfinite/incomplete {STEPS}-step rollout')
    terminal = ep['terminal_success']
    strict = bool(ep['success'] and terminal['encoder_settled_at_exit']
                  and terminal['consecutive_frames_at_exit'] >= 3 and final < .35)
    return {'episode_index': expected_index, 'seed': 42 + expected_index, 'actions': STEPS,
            'final_error_um': final, 'min_error_um': float(values.min()),
            f'error_{LATE_DRIFT_START}_um': float(err[LATE_DRIFT_START]),
            DRIFT_KEY: float(final - err[LATE_DRIFT_START]),
            ABSOLUTE_DRIFT_KEY: float(abs(final - err[LATE_DRIFT_START])),
            'strict_success': strict,
            **{f'final_hit_le_{t}_um': final <= t for t in (5, 10, 30)}}


def aggregate(rows):
    if not rows:
        raise ValueError('empty evaluation')
    return {'completed_targets': len(rows),
            **{key: float(np.mean([r[key] for r in rows]))
               for key in (DRIFT_KEY, ABSOLUTE_DRIFT_KEY)},
            'final_mean_um': float(np.mean([r['final_error_um'] for r in rows])),
            'min_mean_um': float(np.mean([r['min_error_um'] for r in rows])),
            'strict_success_count': sum(r['strict_success'] for r in rows),
            **{f'final_hits_le_{t}_um': sum(r[f'final_hit_le_{t}_um'] for r in rows)
               for t in (5, 10, 30)}, 'episodes': rows}


class EvaluationWorker(BaseEvaluationWorker):
    def __init__(self, spec):
        if spec['steps'] != STEPS or spec['late_drift_start'] != LATE_DRIFT_START:
            raise ValueError('Evaluation step/drift contract mismatch')
        check_identity(spec['identity'])
        super().__init__(spec)

    def run(self, task, output):
        i = task['episode_index']
        if task['seed'] != 42 + i:
            raise ValueError('Requested seed mismatch')
        self.evaluator.evaluate(num_episodes=1, output_dir=output / 'raw',
            target_episodes=self.targets, target_start=i, target_end=i + 1,
            seed=42, max_steps=STEPS, control_period_s=.05,
            error_threshold_mm=.00035, progress_every=100, stop_on_success=False,
            max_consecutive_safety_limited=None)
        path = output / 'raw/evaluation_results.json'
        payload = read(path)
        ep = payload['episodes'][0]
        if (ep['episode_index'] != i or ep['seed'] != 42 + i
                or payload['evaluation_contract']['stop_on_success'] is not False):
            raise ValueError('Target/seed/stopping mismatch')
        row = episode_metrics(ep, i)
        row.update(result_path=str(path), result_sha256=file_sha256(path),
                   checkpoint=self.spec['checkpoint'], samples=self.spec['samples'],
                   controller=self.spec['controller'])
        write_json(output / f'E{i:02d}.json', row)
        record(f'evaluation_E{i:02d}', 'completed', output, row)
        return row


def evaluate(checkpoint, samples, controller, indices, destination, manifest):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
            'samples': samples, 'controller': controller, 'quality': QUALITY,
            'identity': manifest['identity'], 'steps': STEPS,
            'late_drift_start': LATE_DRIFT_START, 'stop_on_success': False}
    groups = {'eval': {'factory': __name__ + ':EvaluationWorker', 'spec': spec,
        'tasks': [{'id': f'E{i:02d}', 'episode_index': i, 'seed': 42 + i} for i in indices]}}
    rows = sorted(pool(groups, destination / 'shards').values(), key=lambda r: r['episode_index'])
    if [r['episode_index'] for r in rows] != list(indices):
        raise ValueError('Evaluation target coverage mismatch')
    result = {'status': 'completed', 'protocol': spec, **aggregate(rows),
              'metric_scope': 'Matched600-step development fast5/dev20; not comparable to1600-step endpoints'}
    for row in rows:
        write_json(destination / f"E{row['episode_index']:02d}.json", row)
    write_json(destination / 'summary.json', result)
    return result
