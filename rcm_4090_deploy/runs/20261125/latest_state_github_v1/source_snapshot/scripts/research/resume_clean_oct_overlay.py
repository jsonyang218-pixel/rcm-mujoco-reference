"""Continue an interrupted OCT render using verified, immutable source shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research import clean_oct_pipeline as pipeline
from scripts.research import gpu_episode_pool as pool


def verified_source(source, groups):
    """Validate the original full contract and every completed task artifact."""
    source = Path(source).resolve()
    contract_path = source / 'contract.json'
    contract = json.loads(contract_path.read_text())
    if (contract['groups'] != groups
            or contract['pool_sha256'] != file_sha256(pool.__file__)):
        raise ValueError('source groups/code/scene/data/protocol differ; refusing reuse')
    checksum = pool.signature(contract)
    results, remaining, receipts = {}, {}, []
    for group_id, group in groups.items():
        tasks = []
        for task in group['tasks']:
            key = f"{group_id}_{task['id']}"
            folder = source / 'tasks' / key
            if (folder / 'receipt.json').is_file():
                expected = pool.signature({'contract': checksum, 'group': group_id, 'task': task})
                result = pool.reuse(folder, expected)
                if (result['episode_index'] != task['episode_index']
                        or result['frames'] != len(task['rows'])):
                    raise ValueError(f'source result coverage differs: {key}')
                # Assemblers must consume the files covered by this receipt.
                for field, filename in [('oct_path', 'oct.npz'), ('audit_path', 'audit.json')]:
                    if Path(result[field]).resolve() != folder / filename:
                        raise ValueError(f'source artifact location differs: {key}')
                results[key] = result
                receipts.append({'task': key, 'source': str(folder), 'signature': expected,
                                 'receipt_sha256': file_sha256(folder / 'receipt.json')})
            else:
                tasks.append(task)
        if tasks:
            remaining[group_id] = {**group, 'tasks': tasks}
    if not results:
        raise ValueError('no complete source tasks to reuse')
    frames = sum(len(task['rows']) for group in groups.values() for task in group['tasks'])
    completed_frames = sum(row['frames'] for row in results.values())
    report = {'source_shards': str(source), 'source_contract_sha256': file_sha256(contract_path),
              'source_contract_signature': checksum, 'reused_tasks': len(results),
              'reused_frames': completed_frames, 'total_frames': frames,
              'remaining_tasks': sum(len(group['tasks']) for group in remaining.values()),
              'remaining_frames': frames - completed_frames, 'receipts': receipts}
    return results, remaining, report


def execute(args):
    started = time.monotonic()
    before = pipeline.identity()
    groups = pipeline.render_groups()
    reused, remaining, evidence = verified_source(args.source_shards, groups)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pool.write_json(args.output_dir / 'source_reuse.json', evidence)
    print(f"Validated immutable source: {evidence['reused_tasks']} tasks / "
          f"{evidence['reused_frames']} frames; remaining "
          f"{evidence['remaining_tasks']} tasks / {evidence['remaining_frames']} frames", flush=True)
    if args.mode == 'preflight':
        report = {**evidence, 'status': 'completed', 'mode': 'preflight', 'identity': before}
        report.pop('receipts')
    else:
        if args.overlay_dir is None or args.overlay_dir.exists():
            raise ValueError('render requires a new --overlay-dir')
        results = dict(reused)
        if remaining:
            results.update(pool.run_pool(
                remaining, args.output_dir / 'shards', workers=args.workers,
                max_workers=args.max_workers, timeout_s=args.task_timeout_s,
                stage_timeout_s=args.stage_timeout_s))
        # Check source artifacts again after the long render, before assembly.
        checked, _, checked_evidence = verified_source(args.source_shards, groups)
        if checked != reused or checked_evidence != evidence:
            raise ValueError('source evidence changed during continuation')
        spec = groups['render']['spec']
        if (file_sha256(pipeline.DATA / 'data/dataset.npz') != spec['dataset_sha256']
                or file_sha256(pipeline.BASE_C) != spec['base_c_sha256']):
            raise ValueError('source dataset/base labels changed during continuation')
        expected = np.sort(np.concatenate([
            np.asarray(task['rows'], dtype=np.int64)
            for group in groups.values() for task in group['tasks']]))
        report = pipeline.assemble_overlay(results, args.overlay_dir, expected)
        report.update(reused_tasks=len(reused), reused_frames=evidence['reused_frames'],
                      newly_rendered_tasks=len(results) - len(reused),
                      newly_rendered_frames=evidence['remaining_frames'],
                      source_reuse=str(args.output_dir / 'source_reuse.json'))
        report['elapsed_s'] = time.monotonic() - started
        report['new_frames_per_second'] = evidence['remaining_frames'] / report['elapsed_s']
    pool.write_json(args.output_dir / 'summary.json', report)
    if pipeline.identity() != before:
        raise ValueError('code/scene changed during continuation')
    if args.mode == 'render' and not report['audit_passed']:
        raise ValueError('full OCT audit failed; dependent training must stop')
    pool.write_json(args.output_dir / 'status.json', {'status': 'completed', 'mode': args.mode})
    pool.record('continuation_' + args.mode, 'completed', args.output_dir,
                f"source={args.source_shards}; reused={len(reused)}; "
                f"remaining_frames={evidence['remaining_frames']}; dataset={pipeline.DATA}; "
                f"initial_checkpoint={pipeline.INITIAL}; summary={args.output_dir / 'summary.json'}")
    print(f"Continuation {args.mode} result: {args.output_dir / 'summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'render'), required=True)
    parser.add_argument('--source-shards', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overlay-dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--task-timeout-s', type=float, default=1200)
    parser.add_argument('--stage-timeout-s', type=float, default=10800)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.source_shards = args.source_shards.resolve()
    if args.overlay_dir is not None:
        args.overlay_dir = args.overlay_dir.resolve()
    try:
        execute(args)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        if args.output_dir.is_dir():
            pool.write_json(args.output_dir / 'status.json', {'status': status, 'failure': repr(error)})
        pool.record('continuation_' + args.mode, status, args.output_dir, repr(error))
        raise


if __name__ == '__main__':
    main()
