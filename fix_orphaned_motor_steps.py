"""Retroactively trim orphaned trailing motor steps from collected v9 episodes.

_run_geometry() can break (LOW_QUALITY_VISUAL_GEOMETRY, SINGULAR_ACTUATION_JACOBIAN)
right after the motor advance at the top of the loop appended a record, but
before iteration_callback captured a frame for that step. The episode then has
len(motor_steps) == len(frames) instead of len(frames) - 1, which fails
verify_v9_dataset line 45. This trims the trailing orphaned step and updates
the manifest motor_execution summary (control_steps, simulation_time_s) so the
verifier passes without re-collecting.

Only episodes with exactly one extra trailing step are touched; everything else
is left alone and reported as an anomaly for manual review.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def fix_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / 'dataset' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    period = float(manifest['nominal_sample_period_s'])
    trimmed = ok = anomalies = 0
    for episode in manifest['episodes']:
        steps_path = (run_dir / episode['motor_steps']).resolve()
        audits_path = (run_dir / episode['frame_audit']).resolve()
        steps = read_jsonl(steps_path)
        audits = read_jsonl(audits_path)
        if not audits:
            anomalies += 1
            print(f'  episode {episode["episode_index"]}: no frames')
            continue
        if len(steps) == len(audits) - 1:
            ok += 1
            continue
        if len(steps) != len(audits):  # only the one-orphan case is auto-fixed
            anomalies += 1
            print(f'  episode {episode["episode_index"]}: anomaly '
                  f'len(steps)={len(steps)} len(audits)={len(audits)}')
            continue
        steps = steps[:len(audits) - 1]
        write_jsonl(steps_path, steps)
        episode['motor_execution']['control_steps'] = len(steps)
        episode['motor_execution']['simulation_time_s'] = (len(audits) - 1) * period
        trimmed += 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'{run_dir.name}: trimmed={trimmed} already_ok={ok} anomalies={anomalies}')
    return trimmed, ok, anomalies


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        fix_run(arg)
