"""Verify and copy unchanged full1600-step T1 evidence into the mixed-budget run."""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
from pathlib import Path
import shutil

from lerobot_training.navigation_overlay import file_sha256
from scripts.research import gpu_episode_pool as scheduler
from scripts.research import homog_replay_common as common

OLD_WORKFLOW = common.ROOT / 'runs/20261123/homog_clean_workflow_intent_v3'
OLD_DATA = common.ROOT / 'data/homog_clean_replay_20261123_intent_v2'
ORIGINAL = common.ROOT / 'runs/20261123/homog_mixed_1600_600_12_amendment_v1/original_sources/scripts/research/homog_replay_collect.py'


class T1Budget(ast.NodeTransformer):
    """Specialize only the new per-regime budget selection to the unchanged T1 budget."""

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'steps'
                and ast.unparse(node.value) == "collection_steps(task['regime'])"):
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == 'steps':
            return ast.copy_location(ast.Name(id='STEPS', ctx=node.ctx), node)
        return node


def equivalent_t1(original, current):
    old, new = ast.parse(original), ast.parse(current)
    checks = {}
    for name in ('collection_tasks', 'expected_label', 'RecordingSensor',
                 'CollectionWorker', 'AuditWorker', 'axis_audit'):
        before = next(n for n in old.body if getattr(n, 'name', None) == name)
        after = next(n for n in new.body if getattr(n, 'name', None) == name)
        checks[name] = (ast.dump(before, include_attributes=False)
            == ast.dump(T1Budget().visit(copy.deepcopy(after)), include_attributes=False))
    if not all(checks.values()):
        raise ValueError(f'T1 execution compatibility failed: {checks}')
    return checks


def prepare(output):
    from scripts.research.homog_replay_pipeline import preflight
    with (OLD_WORKFLOW / 'workflow.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if common.read(OLD_WORKFLOW / 'workflow_status.json')['status'] != 'interrupted':
            raise ValueError('Source workflow must be stopped and interrupted')
        previous = common.read(OLD_WORKFLOW / 'execution_manifest.json')
        if previous['card']['collection']['steps_per_trajectory'] != 1600:
            raise ValueError('Source T1 must have the full1600-step budget')
        if common.collection_steps('T1') != 1600:
            raise ValueError('Destination T1 must preserve1600steps')
        if file_sha256(ORIGINAL) != previous['identity']['scripts/research/homog_replay_collect.py']:
            raise ValueError('Original collector snapshot hash does not match source manifest')
        checks = equivalent_t1(ORIGINAL.read_text(),
            (common.ROOT / 'scripts/research/homog_replay_collect.py').read_text())
        output.mkdir(parents=True, exist_ok=False)
        manifest = preflight(output)
        if (previous['assets_sha256'] != manifest['assets_sha256']
                or previous['collection_tasks'] != manifest['collection_tasks']):
            raise ValueError('Data/checkpoint/targets/seeds/initial states changed')
        allowed = {'scripts/research/homog_replay_common.py',
            'scripts/research/homog_replay_collect.py', 'scripts/research/homog_replay_pipeline.py',
            'scripts/research/homog_replay_publish.py', 'scripts/research/gpu_episode_pool.py',
            'scripts/verification/verify_homog_replay.py',
            'scripts/verification/verify_homog_teacher_intent.py'}
        changed = {k for k, v in previous['identity'].items() if manifest['identity'].get(k) != v}
        if changed - allowed:
            raise ValueError(f'Physics/sensor/model source changed: {changed - allowed}')
        old_sha = scheduler.signature(common.read(OLD_DATA / 'shards/contract.json'))
        groups = {'collection': {'factory': 'scripts.research.homog_replay_collect:CollectionWorker',
            'spec': {'identity': manifest['identity'], 'checkpoint_sha256': file_sha256(common.OLD),
                     'quality': common.QUALITY, 'steps_by_regime': common.COLLECTION_STEPS},
            'tasks': manifest['collection_tasks']}}
        contract = {'groups': groups, 'workers': common.WORKERS, 'max_workers': common.WORKERS,
            'timeout_s': None, 'stage_timeout_s': None, 'pool_sha256': file_sha256(scheduler.__file__),
            'initializer_timeout_s': None, 'execution_timeout_s': None, 'snapshot_every': 5,
            'admission_min_ram_mib': 4096}
        new_sha = scheduler.signature(contract)
        sources = []
        for task in manifest['collection_tasks'][:20]:
            key = f"collection_{task['id']}"
            folder = OLD_DATA / 'shards/tasks' / key
            result = scheduler.reuse(folder, scheduler.signature(
                {'contract': old_sha, 'group': 'collection', 'task': task}))
            if result['regime'] != 'T1' or result['actions'] != 1600:
                raise ValueError('Only complete1600-step T1 can be reused')
            sources.append((task, key, folder, result))
        shards = common.DATA / 'shards'
        shards.mkdir(parents=True, exist_ok=False)
        scheduler.write_json(shards / 'contract.json', contract)
        lineage = []
        for task, key, folder, old_result in sources:
            destination = shards / 'tasks' / key
            source_receipt_sha = file_sha256(folder / 'receipt.json')
            shutil.copytree(folder, destination)
            result = {**old_result, 'dataset_path': str(destination / 'dataset.npz'),
                'source_reuse': {'source_receipt': str(folder / 'receipt.json'),
                    'source_receipt_sha256': source_receipt_sha,
                    'source_identity': previous['identity'],
                    'compatibility': 'Equivalent T1execution AST, same1600step budget, model/assets/targets/seeds/render/physics/labels;12workers scheduling only'}}
            scheduler.write_json(destination / f"E{task['episode_index']:02d}.json", result)
            expected = scheduler.signature({'contract': new_sha, 'group': 'collection', 'task': task})
            scheduler.write_json(destination / 'receipt.json', {'status': 'completed',
                'signature': expected, 'files': scheduler.artifact_hashes(destination), 'result': result})
            scheduler.reuse(destination, expected)
            if file_sha256(folder / 'receipt.json') != source_receipt_sha:
                raise ValueError('Historical receipt changed')
            parity = {name: file_sha256(folder / name) == file_sha256(destination / name)
                      for name in ('dataset.npz', 'frames.jsonl', 'policy_trajectory.json')}
            if not all(parity.values()):
                raise ValueError('Source T1 data changed during independent copy')
            lineage.append({'task': key, 'source_receipt_sha256': source_receipt_sha,
                            'destination': str(destination), 'byte_parity': parity})
        report = {'status': 'prepared', 'reused_T1': len(lineage), 'recollect_T2_T3': 40,
            'steps_by_regime': common.COLLECTION_STEPS, 'evaluation_steps': common.STEPS,
            'workers': common.WORKERS, 'T1_execution_AST': checks,
            'changed_old_source_keys': sorted(changed), 'lineage': lineage,
            'contract_sha256': new_sha, 'old_workflow': str(OLD_WORKFLOW),
            'data': str(common.DATA), 'weights': str(common.WEIGHTS)}
        scheduler.write_json(output / 'mixed_budget_preparation.json', report)
        shutil.copy2(common.ROOT / 'runs/20261123/homog_mixed_1600_600_12_amendment_v1/amendment.json',
                     output / 'budget_amendment.json')
        common.record('mixed_budget_preparation', 'passed', output,
                      {k: v for k, v in report.items() if k != 'lineage'})
        print({k: v for k, v in report.items() if k != 'lineage'}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.output_dir.resolve())
    except BaseException as error:
        common.record('mixed_budget_preparation', 'failed', args.output_dir, repr(error))
        raise


if __name__ == '__main__':
    main()
