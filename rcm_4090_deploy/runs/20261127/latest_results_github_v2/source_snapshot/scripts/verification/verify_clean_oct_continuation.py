"""Small CPU checks for immutable OCT shard continuation, without GPU rollouts."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research import gpu_episode_pool as pool
from scripts.research.resume_clean_oct_overlay import verified_source


OUTPUT = None


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='source_', dir=OUTPUT)
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)
        self.groups = {'render': {'factory': 'unchanged:Worker', 'spec': {'data_sha': 'fixed'},
                                 'tasks': [{'id': 'E00', 'episode_index': 0, 'rows': [1, 2]},
                                           {'id': 'E01', 'episode_index': 1, 'rows': [3]}]}}
        self.contract = {'groups': self.groups, 'workers': 4, 'max_workers': 8,
                         'timeout_s': 180, 'stage_timeout_s': 2400,
                         'pool_sha256': pool.file_sha256(pool.__file__)}
        pool.write_json(self.source / 'contract.json', self.contract)
        self.folder = self.source / 'tasks/render_E00'
        self.folder.mkdir(parents=True)
        (self.folder / 'oct.npz').write_bytes(b'fixed OCT evidence')
        pool.write_json(self.folder / 'audit.json', {'rows': [1, 2]})
        self.result = {'episode_index': 0, 'frames': 2, 'valid_frames': 2,
                       'oct_path': str(self.folder / 'oct.npz'),
                       'audit_path': str(self.folder / 'audit.json')}
        expected = pool.signature({'contract': pool.signature(self.contract), 'group': 'render',
                                   'task': self.groups['render']['tasks'][0]})
        self.receipt = {'status': 'completed', 'signature': expected,
                        'files': pool.artifact_hashes(self.folder), 'result': self.result}
        pool.write_json(self.folder / 'receipt.json', self.receipt)

    def test_only_incomplete_tasks_are_queued_source_untouched(self):
        before = pool.artifact_hashes(self.source)
        results, remaining, report = verified_source(self.source, self.groups)
        self.assertEqual(results, {'render_E00': self.result})
        self.assertEqual(remaining['render']['tasks'], [self.groups['render']['tasks'][1]])
        self.assertEqual(report['reused_frames'], 2)
        self.assertEqual(report['remaining_frames'], 1)
        self.assertEqual(pool.artifact_hashes(self.source), before)

    def test_changed_task_protocol_rejected(self):
        changed = copy.deepcopy(self.groups)
        changed['render']['tasks'][1]['rows'] = [99]
        with self.assertRaisesRegex(ValueError, 'differ'):
            verified_source(self.source, changed)

    def test_changed_pool_code_identity_rejected(self):
        self.contract['pool_sha256'] = 'changed'
        pool.write_json(self.source / 'contract.json', self.contract)
        with self.assertRaisesRegex(ValueError, 'differ'):
            verified_source(self.source, self.groups)

    def test_corrupt_completed_artifact_rejected(self):
        (self.folder / 'oct.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            verified_source(self.source, self.groups)

    def test_result_must_point_at_verified_artifacts(self):
        self.receipt['result']['oct_path'] = '/tmp/unverified.npz'
        pool.write_json(self.folder / 'receipt.json', self.receipt)
        with self.assertRaisesRegex(ValueError, 'location'):
            verified_source(self.source, self.groups)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    OUTPUT = str(args.output_dir.resolve())
    with (args.output_dir / 'tests.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ContinuationTests))
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
              'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'scope': 'CPU receipt/hash/coverage checks; no training or GPU experiment'}
    pool.write_json(args.output_dir / 'summary.json', report)
    pool.record('continuation_contract_verification', report['status'], args.output_dir, json.dumps(report))
    print(json.dumps(report))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
