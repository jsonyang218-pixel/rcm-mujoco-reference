"""Explicitly authorized eight-worker resume without changing frozen science code."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import subprocess

from lerobot_training.navigation_overlay import file_sha256
from scripts.research import gpu_episode_pool as scheduler
from scripts.research import homog_replay_common as common

MODULE = 'scripts.tools.scale_homog_replay'
WORKERS = 8


def pool8(groups, output, *, snapshot_every=5):
    return scheduler.run_pool(groups, output, workers=WORKERS, max_workers=WORKERS,
        timeout_s=None, stage_timeout_s=None, initializer_timeout_s=None,
        execution_timeout_s=None, snapshot_every=snapshot_every,
        resume=(Path(output) / 'contract.json').exists())


def evaluation_pool8(groups, output, *, snapshot_every=None):
    return pool8(groups, output, snapshot_every=snapshot_every)


def migrate_pool(output, evidence, expected_groups):
    """Verify every old byte before re-signing a scheduling-only contract change."""
    old = common.read(output / 'contract.json')
    if old['workers'] != 3 or old['max_workers'] != 4:
        raise ValueError('Expected original 3-to-4 worker contract')
    if old['groups'] != expected_groups:
        raise ValueError('Scientific task/specification differs from frozen manifest')
    if old['pool_sha256'] != file_sha256(scheduler.__file__):
        raise ValueError('Frozen scheduler source changed')
    if any(old.get(k) is not None for k in ('timeout_s', 'stage_timeout_s',
                                          'initializer_timeout_s', 'execution_timeout_s')):
        raise ValueError('All original deadlines must remain None')
    new = {**old, 'workers': WORKERS, 'max_workers': WORKERS}
    old_sha, new_sha = scheduler.signature(old), scheduler.signature(new)
    verified = []
    for group_id, group in old['groups'].items():
        for task in group['tasks']:
            key = f"{group_id}_{task['id']}"
            folder = output / 'tasks' / key
            receipt_path = folder / 'receipt.json'
            if not receipt_path.exists():
                continue
            expected = scheduler.signature({'contract': old_sha, 'group': group_id, 'task': task})
            scheduler.reuse(folder, expected)  # Includes every PNG and all frame evidence.
            receipt = common.read(receipt_path)
            replacement = {**receipt, 'signature': scheduler.signature(
                {'contract': new_sha, 'group': group_id, 'task': task})}
            verified.append((key, receipt_path, receipt_path.read_bytes(), replacement))
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence / 'contract_original.json').write_bytes((output / 'contract.json').read_bytes())
    originals = evidence / 'original_receipts'
    originals.mkdir()
    for key, receipt_path, raw, replacement in verified:
        (originals / f'{key}.json').write_bytes(raw)
    # Only receipt signatures change. Artifacts, paths, results and source identity do not.
    scheduler.write_json(output / 'contract.json', new)
    for key, receipt_path, raw, replacement in verified:
        scheduler.write_json(receipt_path, replacement)
        scheduler.reuse(receipt_path.parent, replacement['signature'])
    result = {'status': 'prepared', 'workers': WORKERS, 'max_workers': WORKERS,
        'completed_preserved': len(verified), 'reused_tasks': [v[0] for v in verified],
        'old_contract_sha256': old_sha, 'new_contract_sha256': new_sha,
        'changed_contract_keys': [k for k in old if old[k] != new[k]],
        'artifact_bytes_changed': False, 'scientific_specification_changed': False,
        'original_receipts': str(originals)}
    scheduler.write_json(evidence / 'migration.json', result)
    return result


def workflow_from_amendment(output):
    request = common.read(output / 'interrupt_request.json')
    if request['requested_workers'] != WORKERS:
        raise ValueError('Explicit eight-worker authorization receipt required')
    workflow = Path(request['workflow'])
    if workflow != common.WORKFLOW or output.parent != workflow:
        raise ValueError('Only the explicitly authorized current workflow may resume')
    return workflow, request


def unlocked(workflow):
    lock = (workflow / 'workflow.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise RuntimeError('Original workflow still owns the lock')
    return lock


def prepare(output):
    workflow, request = workflow_from_amendment(output)
    proc = Path(f"/proc/{request['pid']}")
    if proc.exists():
        start = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        if start != request['process_start_ticks']:
            raise RuntimeError('Original PID was reused; refusing to signal or wait')
        descriptor = os.pidfd_open(request['pid'])
        try:
            select.select([descriptor], [], [])  # Wait for authorized shutdown, no run deadline.
        finally:
            os.close(descriptor)
    with unlocked(workflow):
        manifest = common.read(workflow / 'execution_manifest.json')
        common.check_identity(manifest['identity'])
        state = common.read(workflow / 'workflow_status.json')
        if state['status'] != 'interrupted':
            raise ValueError(f'Expected controlled interruption, got {state}')
        for name in ('failure.json', 'workflow_status.json', 'SUMMARY.md',
                     'publication_status.json', 'publication_failure.json'):
            path = workflow / name
            if path.exists():
                (output / ('before_resume_' + name)).write_bytes(path.read_bytes())
        groups = {'collection': {
            'factory': 'scripts.research.homog_replay_collect:CollectionWorker',
            'spec': {'identity': manifest['identity'],
                     'checkpoint_sha256': file_sha256(common.OLD),
                     'quality': common.QUALITY, 'steps': common.STEPS},
            'tasks': manifest['collection_tasks']}}
        result = migrate_pool(common.DATA / 'shards', output / 'migration', groups)
        result.update(timestamp=datetime.now().astimezone().isoformat(),
            wrapper_sha256=file_sha256(__file__),
            execution_manifest_sha256=file_sha256(workflow / 'execution_manifest.json'),
            actual_gpu_preflight=common.gpu_check(), resources=scheduler.resources(),
            training='Original batch16, AMP, 8% replay and 4000 batches unchanged',
            scope='Collection, audit and closed-loop worker pools use eight workers; training remains one CUDA trainer')
        scheduler.write_json(output / 'prepared.json', result)
        common.record('parallel8_resume_preflight', 'passed', output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def install_policy():
    from scripts.research import a_axis_flip_audit, a_axis_flip_common, a_axis_flip_pipeline
    from scripts.research import homog_replay_collect, homog_replay_publish
    common.pool = homog_replay_collect.pool = pool8
    a_axis_flip_common.pool = a_axis_flip_audit.pool = a_axis_flip_pipeline.pool = evaluation_pool8
    original_files = homog_replay_publish.publication_files

    def files(output):
        result = original_files(output)
        for path in (Path(__file__), common.ROOT / 'scripts/verification/verify_parallel8_resume.py'):
            result['rcm_4090_deploy/' + path.relative_to(common.ROOT).as_posix()] = path
        return result

    homog_replay_publish.publication_files = files


def run(output):
    workflow, _ = workflow_from_amendment(output)
    prepared = common.read(output / 'prepared.json')
    if prepared['wrapper_sha256'] != file_sha256(__file__):
        raise ValueError('Authorized scheduling wrapper changed after preflight')
    if prepared['execution_manifest_sha256'] != file_sha256(workflow / 'execution_manifest.json'):
        raise ValueError('Frozen execution manifest changed')
    install_policy()
    scheduler.write_json(output / 'runtime_policy.json', {
        'workers': WORKERS, 'max_workers': WORKERS, 'all_timeouts': None,
        'wrapper_sha256': file_sha256(__file__),
        'timestamp': datetime.now().astimezone().isoformat()})
    from scripts.research.homog_replay_pipeline import run as original_run
    original_run(workflow)


def launch(output):
    from scripts.research.homog_replay_pipeline import preflight
    workflow, _ = workflow_from_amendment(output)
    receipt_path = output / 'launch_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Eight-worker resume already launched; no duplicate or automatic retry')
    prepared = common.read(output / 'prepared.json')
    if prepared['wrapper_sha256'] != file_sha256(__file__):
        raise ValueError('Scheduling wrapper changed after preparation')
    with unlocked(workflow):
        preflight(workflow)
        contract = common.read(common.DATA / 'shards/contract.json')
        if scheduler.signature(contract) != prepared['new_contract_sha256']:
            raise ValueError('Prepared scheduling contract changed')
        for name in ('failure.json', 'publication_failure.json'):
            path = workflow / name
            if path.exists():
                path.rename(output / ('archived_' + name))
    command = common.env_command() + ['-m', MODULE, 'run', '--output-dir', str(output)]
    log = workflow / 'terminal.log'
    shell = ('set -o pipefail\ncd ' + shlex.quote(str(common.ROOT)) + '\n'
        + shlex.join(command) + ' 2>&1 | tee -a ' + shlex.quote(str(log))
        + '\ncode=${PIPESTATUS[0]}\nprintf "\\nExperiment exited: %s\\n" "$code"\n'
        + 'read -r -p "Press Enter to close this terminal..."\nexit "$code"\n')
    result = subprocess.run(['gnome-terminal', '--title=Homogeneous replay - 8 GPU workers',
        '--working-directory=' + str(common.ROOT), '--', 'bash', '-c', shell],
        capture_output=True, text=True)
    receipt = {'accepted': result.returncode == 0, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'command': command,
        'workers': WORKERS, 'max_workers': WORKERS, 'log': str(log),
        'timestamp': datetime.now().astimezone().isoformat()}
    scheduler.write_json(receipt_path, receipt)
    common.record('parallel8_resume_launch', 'accepted' if receipt['accepted'] else 'failed', output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    if not receipt['accepted']:
        raise RuntimeError('GNOME Terminal failed; no background fallback or automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.output_dir.resolve())


if __name__ == '__main__':
    main()
