"""Launch one gated T3 smoke and continuation, preserving failed-run evidence."""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime
import fcntl
from pathlib import Path
import shlex
import shutil
import subprocess

from lerobot_training.navigation_overlay import file_sha256
from scripts.research import gpu_episode_pool as scheduler
from scripts.research import homog_replay_common as common
from scripts.tools.prepare_homog_mixed_budget import UnreachableT3

SOURCE_WORKFLOW = common.ROOT/'runs/20261123/homog_clean_workflow_mixed_1600_600_12_v5'
SOURCE_DATA = common.ROOT/'data/homog_clean_replay_20261123_mixed_1600_600_12_v1'
ORIGINAL = common.ROOT/'runs/20261123/verification/homog_t3_hold_diagnosis_v1/original_sources/scripts/research/homog_replay_collect.py'
MODULE = 'scripts.tools.prepare_homog_hold_repair'
ALLOWED_CHANGES = {
    'scripts/research/homog_replay_common.py', 'scripts/research/homog_replay_collect.py',
    'scripts/research/homog_replay_pipeline.py', 'scripts/research/homog_replay_publish.py',
    'scripts/research/homog_replay_hold.py', 'scripts/tools/prepare_homog_mixed_budget.py',
    'scripts/tools/prepare_homog_hold_repair.py', 'scripts/verification/verify_homog_t3_hold.py',
    'scripts/verification/verify_homog_teacher_intent.py',
}


def compatible_t1_t2(original, current):
    old, new = ast.parse(original), ast.parse(current)
    checks = {}
    for name in ('collection_tasks', 'expected_label', 'RecordingSensor',
                 'CollectionWorker', 'AuditWorker', 'axis_audit'):
        before = next(n for n in old.body if getattr(n, 'name', None) == name)
        after = next(n for n in new.body if getattr(n, 'name', None) == name)
        checks[name] = ast.dump(UnreachableT3().visit(copy.deepcopy(before)), include_attributes=False) == ast.dump(
            UnreachableT3().visit(copy.deepcopy(after)), include_attributes=False)
    if not all(checks.values()):
        raise ValueError(f'Reachable T1/T2 execution changed: {checks}')
    return checks


def verified_sources(manifest):
    previous = common.read(SOURCE_WORKFLOW/'execution_manifest.json')
    if common.read(SOURCE_WORKFLOW/'workflow_status.json')['status'] != 'failed':
        raise ValueError('Only stopped failed v5 can supply completed evidence')
    if file_sha256(ORIGINAL) != previous['identity']['scripts/research/homog_replay_collect.py']:
        raise ValueError('Original collector does not match frozen failed-run identity')
    checks = compatible_t1_t2(ORIGINAL.read_text(), (common.ROOT/'scripts/research/homog_replay_collect.py').read_text())
    if (manifest['assets_sha256'] != previous['assets_sha256']
            or manifest['collection_tasks'] != previous['collection_tasks']
            or manifest['card']['collection']['steps_by_regime'] != previous['card']['collection']['steps_by_regime']):
        raise ValueError('Assets, targets, seeds, initial states or budgets changed')
    changed = {key for key in previous['identity'].keys() | manifest['identity'].keys()
               if previous['identity'].get(key) != manifest['identity'].get(key)}
    if changed - ALLOWED_CHANGES:
        raise ValueError(f'Unapproved model/physics/sensor/execution change: {changed-ALLOWED_CHANGES}')
    contract_sha = scheduler.signature(common.read(SOURCE_DATA/'shards/contract.json'))
    sources = []
    for task in manifest['collection_tasks']:
        folder = SOURCE_DATA/'shards/tasks'/f"collection_{task['id']}"
        if not (folder/'receipt.json').exists():
            continue
        if task['regime'] not in ('T1', 'T2'):
            raise ValueError('T3 execution changed; no T3 receipt can be migrated')
        expected = scheduler.signature({'contract': contract_sha, 'group': 'collection', 'task': task})
        result = scheduler.reuse(folder, expected)
        if (result['actions'] != common.collection_steps(task['regime'])
                or result['regime'] != task['regime'] or result['task'] != task
                or result['dataset_sha256'] != file_sha256(folder/'dataset.npz')):
            raise ValueError(f'Incomplete or incompatible source: {folder}')
        sources.append((task, folder, result))
    counts = {regime: sum(t['regime'] == regime for t, _, _ in sources) for regime in ('T1', 'T2')}
    if counts != {'T1': 20, 'T2': 15}:
        raise ValueError(f'Frozen failure had20T1+15T2 complete; found {counts}')
    return sources, {'reachable_T1_T2_AST': checks, 'source_counts': counts,
        'changed_source_keys': sorted(changed), 'source_identity': previous['identity']}


def preflight(output):
    from scripts.research.homog_replay_pipeline import preflight as pipeline_preflight
    if output.exists():
        raise FileExistsError(f'Use a new workflow output: {output}')
    output.mkdir(parents=True)
    try:
        manifest = pipeline_preflight(output)
        sources, report = verified_sources(manifest)
        report.update(status='passed', reuse_count=len(sources), remaining_count=60-len(sources),
            source_workflow=str(SOURCE_WORKFLOW), source_data=str(SOURCE_DATA),
            source_receipts={str(p/'receipt.json'): file_sha256(p/'receipt.json') for _, p, _ in sources},
            smoke='E40/E41/E42;3workers;600actions;hold400..599;must pass before copying or formal continuation',
            original_checkpoint=str(common.OLD), destination_data=str(common.DATA),
            destination_weights=str(common.WEIGHTS), identity=manifest['identity'])
        common.write_json(output/'hold_repair_preflight.json', report)
        common.record('T3_hold_repair_preflight', 'passed', output,
            {k: v for k, v in report.items() if k not in ('source_identity', 'source_receipts', 'identity')})
        return report
    except BaseException as error:
        common.write_json(output/'preflight_failure.json', {'status': 'failed', 'reason': repr(error)})
        common.record('T3_hold_repair_preflight', 'failed', output, repr(error))
        raise


def prepare(output):
    smoke = common.read(output/'hold_smoke/summary.json')
    episodes = smoke.get('episodes', [])
    if (smoke['status'] != 'passed' or {r['episode_index'] for r in episodes} != {40, 41, 42}
            or not all(r['actions'] == 600 and r['hold_frames'] == 200
                and r['hold_all_motor_records_motionless'] and r['gate_passed'] for r in episodes)
            or common.read(output/'hold_smoke/card.json')['identity'] != common.identity()):
        raise ValueError('Matching real CUDA/RGB-D 600-step hold smoke has not passed')
    manifest = common.read(output/'execution_manifest.json')
    common.check_identity(manifest['identity'])
    with (SOURCE_WORKFLOW/'workflow.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources, compatibility = verified_sources(manifest)
        frozen = common.read(output/'hold_repair_preflight.json')['source_receipts']
        if any(file_sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError('Historical source receipt changed after preflight')
        groups = {'collection': {'factory': 'scripts.research.homog_replay_collect:CollectionWorker',
            'spec': {'identity': manifest['identity'], 'checkpoint_sha256': file_sha256(common.OLD),
                'quality': common.QUALITY, 'steps_by_regime': common.COLLECTION_STEPS},
            'tasks': manifest['collection_tasks']}}
        contract = {'groups': groups, 'workers': common.WORKERS, 'max_workers': common.WORKERS,
            'timeout_s': None, 'stage_timeout_s': None, 'pool_sha256': file_sha256(scheduler.__file__),
            'initializer_timeout_s': None, 'execution_timeout_s': None, 'snapshot_every': 5,
            'admission_min_ram_mib': 4096}
        shards = common.DATA/'shards'
        shards.mkdir(parents=True, exist_ok=False)
        common.write_json(shards/'contract.json', contract)
        contract_sha = scheduler.signature(contract)
        lineage = []
        for task, source, old_result in sources:
            destination = shards/'tasks'/source.name
            receipt_sha = file_sha256(source/'receipt.json')
            shutil.copytree(source, destination)
            result = {**old_result, 'dataset_path': str(destination/'dataset.npz'),
                'source_reuse': {'source_receipt': str(source/'receipt.json'),
                    'source_receipt_sha256': receipt_sha, **compatibility,
                    'compatibility': 'Only unreachable T3 branches changed; equal reachable T1/T2 AST, assets, budgets, targets, seeds, teacher, render and protocol; independent byte copy'}}
            common.write_json(destination/f"E{task['episode_index']:02d}.json", result)
            expected = scheduler.signature({'contract': contract_sha, 'group': 'collection', 'task': task})
            common.write_json(destination/'receipt.json', {'status': 'completed', 'signature': expected,
                'files': scheduler.artifact_hashes(destination), 'result': result})
            scheduler.reuse(destination, expected)
            parity = {name: file_sha256(source/name) == file_sha256(destination/name)
                and (source/name).stat().st_ino != (destination/name).stat().st_ino
                for name in ('dataset.npz', 'frames.jsonl', 'policy_trajectory.json')}
            if not all(parity.values()) or file_sha256(source/'receipt.json') != receipt_sha:
                raise ValueError('Independent copy parity failed or historical receipt changed')
            lineage.append({'task': source.name, 'source_receipt_sha256': receipt_sha,
                'destination': str(destination), 'byte_parity_and_distinct_inode': parity})
        report = {'status': 'prepared', 'reused': len(lineage), 'remaining': 60-len(lineage),
            'source_workflow': str(SOURCE_WORKFLOW), 'data': str(common.DATA),
            'weights': str(common.WEIGHTS), 'contract_sha256': contract_sha,
            **compatibility, 'lineage': lineage}
        common.write_json(output/'hold_repair_preparation.json', report)
        common.record('T3_hold_repair_preparation', 'passed', output,
            {k: v for k, v in report.items() if k not in ('source_identity', 'lineage')})
        print(f'REUSE VERIFIED: {len(lineage)}/60; remaining25; data: {common.DATA}', flush=True)


def run(output):
    from scripts.verification.verify_homog_t3_hold import smoke
    from scripts.research.homog_replay_pipeline import run as pipeline_run
    entered_pipeline = False
    try:
        with (output/'hold_repair.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            common.check_identity(common.read(output/'hold_repair_preflight.json')['identity'])
            common.write_json(output/'workflow_status.json', {'status': 'running', 'stage': 'hold_transition_smoke'})
            smoke(output/'hold_smoke')
            print(f'HOLD SMOKE PASSED: {output/"hold_smoke/summary.json"}', flush=True)
            common.write_json(output/'workflow_status.json', {'status': 'running', 'stage': 'compatible_receipt_migration'})
            prepare(output)
            entered_pipeline = True
            pipeline_run(output)
    except BaseException as error:
        if not entered_pipeline:
            result = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                'stage': 'hold_transition_smoke_or_preparation', 'reason': repr(error),
                'training_started': False, 'formal_collection_started': False,
                'retry': False, 'completed_stages': []}
            common.write_json(output/'failure.json', result)
            common.write_json(output/'workflow_status.json', result)
            common.record('T3_hold_repair_workflow', result['status'], output, result)
            from scripts.research.homog_replay_publish import publish
            try:
                publish(output, push=True)
            except BaseException as publication_error:
                common.write_json(output/'publication_failure.json', {'reason': repr(publication_error)})
                common.record('publication', 'failed', output, repr(publication_error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output != common.WORKFLOW:
        raise ValueError('Use the versioned hold-repair workflow destination')
    if args.action == 'run':
        return run(output)
    preflight(output)
    command = common.env_command()+['-m', MODULE, 'run', '--output-dir', str(output)]
    log = output/'terminal.log'
    shell = ('set -o pipefail\ncd '+shlex.quote(str(common.ROOT))+'\n'+shlex.join(command)
        +' 2>&1 | tee -a '+shlex.quote(str(log))
        +'\ncode=${PIPESTATUS[0]}\nprintf "\\nExperiment exited: %s\\n" "$code"\n'
        +'read -r -p "Press Enter to close this terminal..."\nexit "$code"\n')
    result = subprocess.run(['gnome-terminal', '--title=T3 hold repair: smoke then12worker continuation',
        '--working-directory='+str(common.ROOT), '--', 'bash', '-c', shell], capture_output=True, text=True)
    receipt = {'accepted': result.returncode == 0, 'returncode': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr, 'command': command, 'log': str(log),
        'timestamp': datetime.now().astimezone().isoformat(),
        'stages': ['real_E40_E41_E42_RGBD_smoke', 'verify_and_copy35', 'original_gated_workflow'],
        'retry': False, 'no_wall_clock_limits': True}
    common.write_json(output/'launch_receipt.json', receipt)
    common.record('T3_hold_repair_launch', 'accepted' if receipt['accepted'] else 'failed', output, receipt)
    if not receipt['accepted']:
        raise RuntimeError(f'GNOME Terminal failed: {result.stderr}; no background fallback')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
