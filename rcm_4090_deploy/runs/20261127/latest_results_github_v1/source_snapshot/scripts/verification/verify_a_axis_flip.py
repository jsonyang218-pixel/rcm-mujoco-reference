"""Targeted lifecycle, full-horizon, quota, isolation and gate regression checks."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.research.a_axis_flip_common import apply_abc_overlay, episode_metrics, record, write_json
from scripts.research.a_axis_flip_audit import select_branch
from scripts.research.a_axis_flip_pipeline import direction_gate, dev20_gate
from scripts.research.gpu_episode_pool import run_pool


OUTPUT = None


class EvidenceWorker:
    def __init__(self, spec):
        # Deliberately spend time in initialization; nullable deadline is real.
        time.sleep(.1)

    def run(self, task, output):
        if task.get('fail_once_marker'):
            marker = Path(task['fail_once_marker'])
            if not marker.exists():
                marker.write_text('external interruption fixture')
                write_json(output/'incomplete.json', {'partial': True})
                raise ValueError('injected one-time interruption')
        if task.get('fail'):
            raise ValueError('injected illegal state')
        time.sleep(.1)
        write_json(output/f"{task['id']}.json", task)
        return task

    def close(self):
        pass


class AxisFlipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='rcm_a_axis_', dir=OUTPUT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_pool(self, tasks, resume=False):
        with patch('scripts.research.gpu_episode_pool.resources', return_value={
                'gpu_free_mib': 48000, 'gpu_utilization_percent': 0, 'ram_available_mib': 24000}):
            return run_pool({'test': {'factory': 'scripts.verification.verify_a_axis_flip:EvidenceWorker',
                'spec': {'scope': 'CPU lifecycle fixture'}, 'tasks': tasks}}, self.root/'pool',
                workers=1, max_workers=1, timeout_s=None, stage_timeout_s=None,
                initializer_timeout_s=None, execution_timeout_s=None, snapshot_every=1, resume=resume)

    def test_nullable_budgets_save_evidence_and_reuse(self):
        tasks = [{'id': 'E00'}, {'id': 'E01'}]
        first = self.run_pool(tasks)
        self.assertEqual(len(first), 2)
        folder = self.root/'pool'
        for name in ('partial_summary.json', 'reuse_receipt.json', 'snapshot_001.json', 'snapshot_002.json'):
            self.assertTrue((folder/name).is_file())
        contract = json.loads((folder/'contract.json').read_text())
        self.assertTrue(all(contract[k] is None for k in ('timeout_s', 'stage_timeout_s',
                                                         'initializer_timeout_s', 'execution_timeout_s')))
        with patch('scripts.research.gpu_episode_pool.worker_main', side_effect=AssertionError('completed item relaunched')):
            self.assertEqual(self.run_pool(tasks, resume=True), first)
        self.assertTrue(all(x['reused'] for x in json.loads((folder/'reuse_receipt.json').read_text())))

    def test_changed_completed_evidence_cannot_be_reused(self):
        tasks = [{'id': 'E00'}]
        self.run_pool(tasks)
        (self.root/'pool/tasks/test_E00/E00.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'incomplete, changed'):
            self.run_pool(tasks, resume=True)

    def test_failed_worker_stops_without_retry(self):
        with self.assertRaisesRegex(RuntimeError, 'injected illegal state'):
            self.run_pool([{'id': 'E00', 'fail': True}, {'id': 'E01'}])
        state = json.loads((self.root/'pool/partial_summary.json').read_text())
        self.assertEqual(state['status'], 'failed')
        self.assertFalse((self.root/'pool/tasks/test_E01').exists())

    def test_explicit_resume_skips_complete_and_archives_partial(self):
        tasks = [{'id': 'E00'}, {'id': 'E01', 'fail_once_marker': str(self.root/'marker')}]
        with self.assertRaisesRegex(RuntimeError, 'one-time interruption'):
            self.run_pool(tasks)
        first = json.loads((self.root/'pool/tasks/test_E00/receipt.json').read_text())
        result = self.run_pool(tasks, resume=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(first, json.loads((self.root/'pool/tasks/test_E00/receipt.json').read_text()))
        receipts = json.loads((self.root/'pool/reuse_receipt.json').read_text())
        self.assertTrue(next(r for r in receipts if r['task']=='test_E00')['reused'])
        self.assertFalse(next(r for r in receipts if r['task']=='test_E01')['reused'])
        self.assertEqual(len(list((self.root/'pool/interrupted').glob('test_E01_*'))), 1)

    def test_abc_overlay_isolated_and_checked(self):
        source = np.zeros((20, 3), np.float32)
        source[4] = [-.01, .3, .1]  # Unchanged legacy axes may exceed conservative repair caps.
        after = source[[4]].copy()
        after[0, 0] = .001
        path = self.root/'abc.npz'
        def save(**extra):
            values = dict(rows=np.array([4], np.int64), old_action=source[[4]],
                          new_action=after, dataset_sha256='source')
            values.update(extra)
            np.savez(path, **values)
        save()
        result = apply_abc_overlay(source, path, 'source')
        np.testing.assert_array_equal(result[[4]], after)
        self.assertEqual(float(source[4, 0]), float(np.float32(-.01)))
        np.testing.assert_array_equal(result[:, 1:], source[:, 1:])
        with self.assertRaises(ValueError):
            apply_abc_overlay(source, path, 'other')
        save(new_action=np.full((1, 3), np.nan))
        with self.assertRaises(ValueError):
            apply_abc_overlay(source, path, 'source')
        save(new_action=np.full((1, 3), 1.))
        with self.assertRaises(ValueError):
            apply_abc_overlay(source, path, 'source')

    def test_exact_8_percent_four_epochs_and_seed_determinism(self):
        from lerobot_training.dit_experience_replay import RatioReplacementBatchSampler
        totals = np.zeros(2, dtype=int)
        for epoch in range(4):
            a = RatioReplacementBatchSampler(100, 40, .08, 1000, 16, 20261118+epoch)
            batches = list(a)
            self.assertEqual(batches, list(RatioReplacementBatchSampler(100, 40, .08, 1000, 16, 20261118+epoch)))
            ids = np.array(batches)
            counts = np.array([(ids < 100).sum(), (ids >= 100).sum()])
            np.testing.assert_array_equal(counts, [14720, 1280])
            totals += counts
        np.testing.assert_array_equal(totals, [58880, 5120])

    def test_branch_B_priority_and_no_branch(self):
        label = {'branch_A_hit': True}
        snap = {'A_old': {'axis_reverse_snapshot_counts': {'A': 3}},
                'A_new': {'axis_reverse_snapshot_counts': {'A': 7}}}
        rgb = {'RGB_OCT_render_provenance_mismatch': True, 'boundary_jump_increased': False}
        self.assertEqual(select_branch(label, snap, rgb)['selected_branch'], 'B')
        snap['A_old']['axis_reverse_snapshot_counts']['A'] = 4
        self.assertEqual(select_branch(label, snap, rgb)['selected_branch'], 'A')
        label['branch_A_hit'] = False
        self.assertIsNone(select_branch(label, snap, rgb)['selected_branch'])

    def test_direction_and_dev20_gates_are_strict(self):
        snap = {'axis_reverse_snapshot_counts': {'A': 1}, 'first_order_toward_fraction': .76}
        fast = {'final_mean_um': 34.9, 'drift_1000_to_1600_um': 14.9}
        self.assertTrue(direction_gate(snap, fast))
        self.assertTrue(dev20_gate(fast))
        snap['first_order_toward_fraction'] = .75
        self.assertFalse(direction_gate(snap, fast))
        fast['final_mean_um'] = 35.
        self.assertFalse(dev20_gate(fast))
        fast.update(final_mean_um=1., drift_1000_to_1600_um=15.)
        self.assertFalse(dev20_gate(fast))

    def test_no_branch_summary_has_explicit_skips(self):
        from scripts.research.a_axis_flip_pipeline import summary
        stages = {'audit': {'status': 'completed', 'selected_branch': None},
                  **{name: {'status': 'skipped', 'reason': 'No branch'} for name in ('phase2', 'phase3', 'phase4')}}
        with patch('scripts.research.a_axis_flip_pipeline.PUB', self.root):
            summary(self.root, stages, 'completed_with_conditional_skips')
        self.assertIn('skipped', (self.root/'SUMMARY.md').read_text())

    def test_publication_preserves_both_divergent_log_versions(self):
        from scripts.research.a_axis_flip_publish import preserve_log_history
        base = b'old published history\x00\n- older complete record\n'
        local = b'local divergent history\x00\n- date: a_axis_flip first\n- next record\n'
        combined, receipt = preserve_log_history(base, local)
        self.assertTrue(combined.startswith(base))
        self.assertTrue(combined.endswith(b'- date: a_axis_flip first\n- next record\n'))
        self.assertTrue(receipt['merged'])
        self.assertEqual(local, b'local divergent history\x00\n- date: a_axis_flip first\n- next record\n')
        extended = local+b'- latest appended\n'
        second, _ = preserve_log_history(base, extended)
        self.assertTrue(second.startswith(combined))
        ordinary, receipt = preserve_log_history(base, base+b'new record\n')
        self.assertEqual(ordinary, base+b'new record\n')
        self.assertFalse(receipt['merged'])

    def evaluator(self, action=None):
        from lerobot_training.eval_policy import PolicyEvaluator
        e = PolicyEvaluator.__new__(PolicyEvaluator)
        e.controller_variant = 'feedback'
        e.use_oct = False
        e.measure_inference = False
        e.control_mode_counts = {}
        e.last_terminal_solution = None
        e.terminal_locked_target_world_mm = None
        e.reset_buffer = lambda seed: None
        e.get_action = action or (lambda *args: np.zeros(3))
        e.record_applied_action = lambda action: None
        e.invalidate_action_chunk = lambda: None
        return e

    def test_success_never_stops_fixed1600_and_default_still_stops(self):
        from rcm.reference import Reference
        sensor = SimpleNamespace(observe=lambda: {'rgb': np.zeros((2, 2, 3), np.uint8)})
        ref = Reference()
        ref.enable_motor_execution()
        ep = self.evaluator().evaluate_episode(ref, ref.site_mm('needle_tip'),
            max_steps=1600, error_threshold_mm=.00035, sensor=sensor, stop_on_success=False,
            max_consecutive_safety_limited=None)
        self.assertEqual(len(ep['trajectory']['actions']), 1600)
        self.assertTrue(ep['success'])
        metrics = episode_metrics(ep, 0)
        self.assertTrue(metrics['strict_success'])
        self.assertEqual(metrics['drift_1000_to_1600_um'], 0.)
        ref = Reference()
        ref.enable_motor_execution()
        default = self.evaluator().evaluate_episode(ref, ref.site_mm('needle_tip'),
            max_steps=1600, error_threshold_mm=.00035, sensor=sensor)
        self.assertLess(default['steps'], 1600)
        self.assertTrue(default['success'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            episode_metrics(default, 0)

    def test_nonfinite_action_is_fatal(self):
        from rcm.reference import Reference
        ref = Reference()
        ref.enable_motor_execution()
        sensor = SimpleNamespace(observe=lambda: {'rgb': np.zeros((2, 2, 3), np.uint8)})
        with self.assertRaises(FloatingPointError):
            self.evaluator(lambda *args: np.array([np.nan, 0., 0.])).evaluate_episode(
                ref, ref.site_mm('needle_tip')+np.array([1., 0., 0.]),
                max_steps=1600, sensor=sensor, stop_on_success=False)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    OUTPUT = args.output_dir
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AxisFlipTests))
    report = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'CPU_reason': 'Small deterministic contract fixtures and real integer-handle backend regression; no model training/inference or rendering. Production uses CUDA/NVIDIA EGL.'}
    write_json(args.output_dir/'summary.json', report)
    record('verification', report['status'], args.output_dir, report)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
