"""Check scheduling-only contract migration before reusing collected trajectories."""
from __future__ import annotations

import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.tools import scale_homog_replay as scale
from scripts.research import gpu_episode_pool as scheduler
from scripts.research.homog_replay_common import record

OUTPUT = None


class Parallel8ResumeTests(unittest.TestCase):
    def fixture(self, name):
        root = OUTPUT / name
        task = {'id': 'E00', 'seed': 1000}
        groups = {'collection': {'factory': 'frozen:Worker', 'spec': {'quality': 112}, 'tasks': [task]}}
        contract = {'groups': groups, 'workers': 3, 'max_workers': 4,
            'pool_sha256': scale.file_sha256(scheduler.__file__), 'timeout_s': None,
            'stage_timeout_s': None, 'initializer_timeout_s': None,
            'execution_timeout_s': None, 'snapshot_every': 5}
        folder = root / 'tasks/collection_E00'
        folder.mkdir(parents=True)
        (folder / 'frame.png').write_bytes(b'frozen original frame')
        scheduler.write_json(root / 'contract.json', contract)
        signature = scheduler.signature({'contract': scheduler.signature(contract), 'group': 'collection', 'task': task})
        scheduler.write_json(folder / 'receipt.json', {'status': 'completed', 'signature': signature,
            'files': scheduler.artifact_hashes(folder), 'result': {'error_um': 17.28}})
        return root, folder, contract

    def test_pool_requests_eight_and_preserves_unlimited_budgets(self):
        with patch.object(scheduler, 'run_pool', return_value={}) as mock:
            scale.pool8({}, OUTPUT / 'not_started')
        kw = mock.call_args.kwargs
        self.assertEqual((kw['workers'], kw['max_workers']), (8, 8))
        for key in ('timeout_s', 'stage_timeout_s', 'initializer_timeout_s', 'execution_timeout_s'):
            self.assertIsNone(kw[key])

    def test_migration_preserves_artifacts_results_and_original_receipts(self):
        root, folder, contract = self.fixture('legal')
        old_receipt = (folder / 'receipt.json').read_bytes()
        result = scale.migrate_pool(root, root / 'migration', contract['groups'])
        self.assertEqual(result['changed_contract_keys'], ['workers', 'max_workers'])
        self.assertEqual(result['completed_preserved'], 1)
        self.assertEqual((root / 'migration/original_receipts/collection_E00.json').read_bytes(), old_receipt)
        self.assertEqual((folder / 'frame.png').read_bytes(), b'frozen original frame')
        receipt = scale.common.read(folder / 'receipt.json')
        self.assertEqual(scheduler.reuse(folder, receipt['signature']), {'error_um': 17.28})

    def test_changed_image_rejected_before_any_contract_write(self):
        root, folder, contract = self.fixture('tampered')
        before = (root / 'contract.json').read_bytes()
        (folder / 'frame.png').write_bytes(b'changed frame')
        with self.assertRaises(ValueError):
            scale.migrate_pool(root, root / 'migration', contract['groups'])
        self.assertEqual((root / 'contract.json').read_bytes(), before)
        self.assertFalse((root / 'migration').exists())

    def test_changed_scientific_groups_rejected(self):
        root, folder, contract = self.fixture('changed_science')
        with self.assertRaises(ValueError):
            scale.migrate_pool(root, root / 'migration', {'altered': {}})
        self.assertEqual(scale.common.read(root / 'contract.json'), contract)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Parallel8ResumeTests))
    summary = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors),
        'CPU_reason': 'Small deterministic contract and artifact-integrity fixtures; actual collection uses CUDA/EGL'}
    scheduler.write_json(OUTPUT / 'summary.json', summary)
    record('parallel8_resume_verification', summary['status'], OUTPUT, summary)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
