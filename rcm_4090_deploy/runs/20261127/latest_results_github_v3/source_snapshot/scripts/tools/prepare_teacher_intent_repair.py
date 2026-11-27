"""Preflight a new repair version and import only byte-verified compatible T1 data."""
from __future__ import annotations

import argparse
import ast
import copy
import shutil
from pathlib import Path

from lerobot_training.navigation_overlay import file_sha256
from scripts.research import gpu_episode_pool as scheduler
from scripts.research import homog_replay_common as common

PREPARATION = common.ROOT / 'runs/20261123/homog_plateau_repair_preparation_v1'
OLD_WORKFLOW = common.ROOT / 'runs/20261123/homog_clean_workflow_v2'
OLD_DATA = common.ROOT / 'data/homog_clean_replay_20261123'


class T1Only(ast.NodeTransformer):
    """Erase only the new T2/T3-only execution path, then compare the T1 AST."""

    def visit_If(self, node):
        condition = ast.unparse(node.test)
        if condition in ("regime != 'T1'", 'self.teacher_intent is not None'):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        names = [ast.unparse(target) for target in node.targets]
        if names in (['self.teacher_intent'], ['self.execution_diagnostics'],
                     ["result['expert_execution']"]):
            return None
        return self.generic_visit(node)


def compatible_t1(original, current):
    old, new = ast.parse(original), ast.parse(current)
    checks = {}
    for before in old.body:
        if not isinstance(before, (ast.FunctionDef, ast.ClassDef)):
            continue
        after = next(n for n in new.body if getattr(n, 'name', None) == before.name)
        a = T1Only().visit(copy.deepcopy(before))
        b = T1Only().visit(copy.deepcopy(after))
        checks[before.name] = ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    if not all(checks.values()):
        raise ValueError(f'T1 behavioral code changed: {checks}')
    return checks


def prepare(output):
    from scripts.research.homog_replay_pipeline import preflight
    output.mkdir(parents=True, exist_ok=False)
    smoke = common.ROOT / 'runs/20261123/verification/homog_replay_teacher_intent_smoke_v1/summary.json'
    if common.read(smoke)['status'] != 'passed':
        raise ValueError('Real CUDA/NVIDIA EGL recovery and200-frame hold verification must pass')
    smoke_identity = common.read(smoke.parent / 'card.json')['identity']
    current_identity = common.identity()
    for key in ('code_sha256', 'scene_sha256', 'scripts/research/homog_replay_collect.py',
                'scripts/research/homog_replay_target.py', 'rcm/visual_servo.py'):
        if smoke_identity[key] != current_identity[key]:
            raise ValueError(f'Recovery smoke no longer matches current execution: {key}')
    previous = common.read(OLD_WORKFLOW / 'execution_manifest.json')
    if common.read(OLD_WORKFLOW / 'workflow_status.json')['status'] != 'failed':
        raise ValueError('Do not modify or copy a live source run')
    source = common.ROOT / 'scripts/research/homog_replay_collect.py'
    original = PREPARATION / 'original_sources/scripts/research/homog_replay_collect.py'
    if file_sha256(original) != previous['identity']['scripts/research/homog_replay_collect.py']:
        raise ValueError('Original collection source snapshot does not match execution identity')
    checks = compatible_t1(original.read_text(), source.read_text())
    manifest = preflight(output)
    if previous['assets_sha256'] != manifest['assets_sha256']:
        raise ValueError('Original datasets, checkpoint or target manifest changed')
    if previous['collection_tasks'] != manifest['collection_tasks']:
        raise ValueError('Training targets or initial states changed')
    allowed_changes = {'scripts/research/homog_replay_collect.py',
        'scripts/research/homog_replay_common.py', 'scripts/research/homog_replay_pipeline.py',
        'scripts/research/homog_replay_publish.py'}
    changed = {key for key, value in previous['identity'].items()
               if manifest['identity'].get(key) != value}
    if changed - allowed_changes:
        raise ValueError(f'Non-collection physics/model/sensor source changed: {changed - allowed_changes}')
    old_contract = common.read(OLD_DATA / 'shards/contract.json')
    old_sha = scheduler.signature(old_contract)
    groups = {'collection': {'factory': 'scripts.research.homog_replay_collect:CollectionWorker',
        'spec': {'identity': manifest['identity'], 'checkpoint_sha256': file_sha256(common.OLD),
                 'quality': common.QUALITY, 'steps': common.STEPS},
        'tasks': manifest['collection_tasks']}}
    contract = {'groups': groups, 'workers': 8, 'max_workers': 8,
        'timeout_s': None, 'stage_timeout_s': None, 'pool_sha256': file_sha256(scheduler.__file__),
        'initializer_timeout_s': None, 'execution_timeout_s': None, 'snapshot_every': 5}
    contract_sha = scheduler.signature(contract)
    # Verify all source bytes before copying or issuing any new provenance receipt.
    originals = []
    for task in manifest['collection_tasks'][:20]:
        key = f"collection_{task['id']}"
        folder = OLD_DATA / 'shards/tasks' / key
        result = scheduler.reuse(folder, scheduler.signature(
            {'contract': old_sha, 'group': 'collection', 'task': task}))
        if result['regime'] != 'T1' or result['actions'] != common.STEPS:
            raise ValueError('Only complete unchanged T1 student trajectories may be imported')
        originals.append((task, key, folder, result))
    shards = common.DATA / 'shards'
    shards.mkdir(parents=True, exist_ok=False)
    scheduler.write_json(shards / 'contract.json', contract)
    lineage = []
    for task, key, folder, old_result in originals:
        destination = shards / 'tasks' / key
        # Independent copies keep historical evidence immutable even if a new
        # version is accidentally edited later. Never hardlink writable data.
        shutil.copytree(folder, destination)
        before_receipt_sha = file_sha256(folder / 'receipt.json')
        result = {**old_result, 'dataset_path': str(destination / 'dataset.npz'),
            'expert_execution': 'old_student_feedback',
            'source_reuse': {'source_receipt': str(folder / 'receipt.json'),
                'source_receipt_sha256': before_receipt_sha,
                'source_collection_identity': previous['identity'],
                'compatibility': 'T1 effective AST, assets, physics, scene, render, seeds and labels unchanged; no T2/T3 imported'}}
        scheduler.write_json(destination / f"E{task['episode_index']:02d}.json", result)
        expected = scheduler.signature({'contract': contract_sha, 'group': 'collection', 'task': task})
        scheduler.write_json(destination / 'receipt.json', {'status': 'completed', 'signature': expected,
            'files': scheduler.artifact_hashes(destination), 'result': result})
        scheduler.reuse(destination, expected)
        if file_sha256(folder / 'receipt.json') != before_receipt_sha:
            raise ValueError('Historical source receipt changed during import')
        lineage.append({'task': key, 'source_receipt': str(folder / 'receipt.json'),
            'source_receipt_sha256': before_receipt_sha, 'destination': str(destination),
            'dataset_bytes_identical': file_sha256(folder / 'dataset.npz') == file_sha256(destination / 'dataset.npz'),
            'frame_evidence_bytes_identical': file_sha256(folder / 'frames.jsonl') == file_sha256(destination / 'frames.jsonl')})
    summary = {'status': 'prepared', 'reused_T1': len(lineage), 'T2_T3_to_recollect': 40,
        'workers': 8, 'T1_effective_AST_comparison': checks, 'changed_frozen_source_keys': sorted(changed),
        'source_workflow': str(OLD_WORKFLOW), 'source_dataset_unchanged': str(OLD_DATA),
        'new_data': str(common.DATA), 'new_weights': str(common.WEIGHTS),
        'new_workflow': str(output), 'smoke_evidence': str(smoke),
        'contract_sha256': contract_sha, 'lineage': lineage}
    scheduler.write_json(output / 'repair_preparation.json', summary)
    common.record('teacher_intent_repair_preparation', 'passed', output, {
        'reused_T1': 20, 'recollect_T2_T3': 40, 'workers': 8,
        'T1_AST_unchanged': True, 'all_original_images_and_receipts_hash_verified': True,
        'data': str(common.DATA), 'weights': str(common.WEIGHTS)})
    print({k: v for k, v in summary.items() if k != 'lineage'}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir.resolve())


if __name__ == '__main__':
    main()
