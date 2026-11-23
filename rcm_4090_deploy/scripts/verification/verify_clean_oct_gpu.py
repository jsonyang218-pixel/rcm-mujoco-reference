"""Verify overlay alignment, task reuse, failure isolation and deadlines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lerobot_training.dataset_contract import valid_window_indices
from lerobot_training.navigation_overlay import (
    apply_c_overlay, apply_dataset_overlays, apply_oct_overlay, file_sha256,
)
from scripts.research.gpu_episode_pool import record, run_pool, write_json

OUTPUT = None


class FakeWorker:
    def __init__(self, spec):
        self.spec = spec

    def run(self, task, output):
        if task.get('fail'):
            raise ValueError('injected task failure')
        time.sleep(task.get('sleep', 0))
        write_json(output / 'result.json', task)
        return {'id': task['id']}

    def close(self):
        pass


class CleanOctGpuTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rcm_clean_oct_', dir=OUTPUT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.oct = np.zeros((24,3), dtype=np.float32)
        self.valid = np.ones(24, dtype=bool)
        self.actions = np.full((24,3), .001, dtype=np.float32)
        self.source = self.root / 'data/dataset.npz'
        self.source.parent.mkdir()
        np.savez(self.source, oct_navigation=self.oct, action=self.actions)
        self.sha = file_sha256(self.source)

    def overlay(self, **changes):
        values = dict(rows=np.array([5], dtype=np.int64), old_oct_mm=self.oct[[5]],
                      new_oct_mm=np.ones((1,3), dtype=np.float32)*.002,
                      old_valid=np.array([True]), new_valid=np.array([True]), dataset_sha256=self.sha)
        values.update(changes)
        path = self.root / 'overlay.npz'
        np.savez(path, **values)
        return path

    def test_source_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            apply_oct_overlay(self.oct, self.valid, self.overlay(dataset_sha256='wrong'), self.sha)

    def gate_inputs(self):
        checkpoint = self.root / 'checkpoint.pth'
        checkpoint.write_bytes(b'frozen checkpoint')
        selection = {'best_final_mean_um': 45.84530045175171,
                     'best_checkpoint': str(checkpoint), 'best_variant': 'V0'}
        source = self.root / 'selection.json'
        write_json(source, selection)
        authorization = self.root / 'authorization.json'
        write_json(authorization, {'scope': 'dev20_only', 'user_instruction': '差不多,继续',
                   'reason': 'User authorized development confirmation after missed gate',
                   'fast5_summary_sha256': file_sha256(source),
                   'checkpoint_sha256': file_sha256(checkpoint), 'variant': 'V0',
                   'observed_fast5_final_mean_um': selection['best_final_mean_um']})
        return selection, source, authorization

    def test_missed_dev20_gate_still_stops_without_authorization(self):
        from scripts.research.clean_oct_pipeline import dev20_admission
        selection, source, _ = self.gate_inputs()
        with self.assertRaisesRegex(ValueError, 'not selected'):
            dev20_admission(selection, source)

    def test_authorized_dev20_preserves_failed_gate_and_original_selection(self):
        from scripts.research.clean_oct_pipeline import dev20_admission
        selection, source, authorization = self.gate_inputs()
        before = source.read_bytes()
        result = dev20_admission(selection, source, authorization)
        self.assertFalse(result['original_gate_passed'])
        self.assertTrue(result['user_authorized_exception'])
        self.assertEqual(result['fast5_final_mean_um'], 45.84530045175171)
        self.assertEqual(source.read_bytes(), before)

    def test_dev20_authorization_cannot_follow_changed_checkpoint(self):
        from scripts.research.clean_oct_pipeline import dev20_admission
        selection, source, authorization = self.gate_inputs()
        Path(selection['best_checkpoint']).write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            dev20_admission(selection, source, authorization)

    def test_dev20_authorization_cannot_follow_changed_selection(self):
        from scripts.research.clean_oct_pipeline import dev20_admission
        selection, source, authorization = self.gate_inputs()
        selection['best_variant'] = 'V2'
        write_json(source, selection)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            dev20_admission(selection, source, authorization)

    def test_assembler_rejects_mixed_source_shards(self):
        from scripts.research.clean_oct_pipeline import assemble_overlay
        path=self.overlay(dataset_sha256='wrong')
        audit=self.root/'audit.json'
        write_json(audit,{'rows':[5],'error_um':[0.0],'c_audit':[],'c_proposals':[]})
        with patch('scripts.research.clean_oct_pipeline.DATA',self.root):
            with self.assertRaisesRegex(ValueError,'mixed overlay'):
                assemble_overlay({'one':{'oct_path':str(path),'audit_path':str(audit)}},
                                 self.root/'assembled',np.array([5]),smoke=True)

    def test_scanned_world_audit_is_not_reinterpreted_as_legacy_frame(self):
        from scripts.research.clean_oct_workers import target_world
        frozen = np.array([.346, -.015, .372])
        scene = SimpleNamespace(task_from_encoders=lambda values, reference_task: np.zeros(3),
                                set_task=lambda value: None,
                                site_mm=lambda name: frozen * 1000)
        target = {'kind':'scanned_world_m','value':frozen.tolist(), 'terminal_encoders':[1,2,3,0]}
        np.testing.assert_array_equal(target_world(scene,target,np.zeros(3)),frozen*1000)
        scene.site_mm=lambda name: frozen*1000 + np.array([.001,0,0])
        with self.assertRaisesRegex(ValueError,'calibration'):
            target_world(scene,target,np.zeros(3))

    def test_duplicate_and_nan_rows_rejected(self):
        for changes in [dict(rows=np.array([5,5], dtype=np.int64)),
                        dict(new_oct_mm=np.full((1,3), np.nan))]:
            with self.assertRaises(ValueError):
                apply_oct_overlay(self.oct, self.valid, self.overlay(**changes), self.sha)

    def test_observation_overlay_cannot_change_unselected_rows(self):
        result, valid = apply_oct_overlay(self.oct, self.valid, self.overlay(), self.sha)
        np.testing.assert_array_equal(result[np.arange(24)!=5], self.oct[np.arange(24)!=5])
        np.testing.assert_array_equal(valid, self.valid)
        np.testing.assert_array_equal(self.oct, np.zeros((24,3)))

    def test_invalid_new_oct_excludes_history_windows(self):
        dataset = SimpleNamespace(data_dir=self.root, oct_navigation=self.oct.copy(),
                                  oct_navigation_valid=self.valid.copy(), actions=self.actions.copy())
        def build():
            return valid_window_indices(np.zeros(24), np.arange(24), np.arange(24)*.05,
                n_obs_steps=2,n_action_steps=16, oct_navigation=dataset.oct_navigation,
                oct_navigation_valid=dataset.oct_navigation_valid).tolist()
        dataset._build_valid_indices = build
        dataset.valid_indices = build()
        receipt = apply_dataset_overlays(dataset, oct_overlay=self.overlay(new_valid=np.array([False])))
        self.assertGreater(receipt['excluded_windows'], 0)
        self.assertEqual(receipt['added_windows'], 0)

    def test_positive_C_composition_preserves_AB_and_cap(self):
        path = self.root / 'c.npz'
        def save(value):
            np.savez(path, rows=np.array([5],dtype=np.int64), old_dC_mm=self.actions[[5],2],
                     new_dC_mm=np.array([value],dtype=np.float32), dataset_sha256=self.sha)
        save(.0075)
        result = apply_c_overlay(self.actions, path, self.sha)
        np.testing.assert_array_equal(result[:,:2], self.actions[:,:2])
        self.assertAlmostEqual(float(result[5,2]),.0075)
        save(.031)
        with self.assertRaises(ValueError):
            apply_c_overlay(self.actions,path,self.sha)

    def groups(self, tasks):
        return {'fake': {'factory':'scripts.verification.verify_clean_oct_gpu:FakeWorker',
                         'spec':{'test':'CPU lifecycle only, no CUDA rollout'}, 'tasks':tasks}}

    def run_pool(self, tasks, **kwargs):
        with patch('scripts.research.gpu_episode_pool.resources', return_value={
            'gpu_free_mib':48000, 'gpu_utilization_percent':0, 'ram_available_mib':24000}):
            return run_pool(self.groups(tasks), self.root/'pool', workers=1,max_workers=1,
                            timeout_s=kwargs.pop('timeout_s',2), stage_timeout_s=10, **kwargs)

    def test_complete_task_reused_without_worker_restart(self):
        tasks=[{'id':'E00'},{'id':'E01'}]
        first=self.run_pool(tasks)
        with patch('scripts.research.gpu_episode_pool.worker_main', side_effect=AssertionError('no repeat')):
            self.assertEqual(self.run_pool(tasks,resume=True),first)
        receipts=json.loads((self.root/'pool/reuse_receipt.json').read_text())
        self.assertTrue(all(row['reused'] for row in receipts))

    def test_tampered_task_reuse_rejected(self):
        tasks=[{'id':'E00'}]
        self.run_pool(tasks)
        (self.root/'pool/tasks/fake_E00/result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'receipt'):
            self.run_pool(tasks,resume=True)

    def test_failed_task_stops_pending_without_retry(self):
        with self.assertRaisesRegex(RuntimeError,'injected task failure'):
            self.run_pool([{'id':'E00','fail':True},{'id':'E01'}])
        status=json.loads((self.root/'pool/partial_summary.json').read_text())
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['completed'],0)
        self.assertFalse((self.root/'pool/tasks/fake_E01').exists())

    def test_deadline_keeps_partial_and_no_success_receipt(self):
        with self.assertRaises(TimeoutError):
            self.run_pool([{'id':'E00','sleep':1}],timeout_s=.1)
        self.assertFalse((self.root/'pool/tasks/fake_E00/receipt.json').exists())
        self.assertEqual(json.loads((self.root/'pool/partial_summary.json').read_text())['completed'],0)


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--oct-overlay',type=Path)
    parser.add_argument('--c-overlay',type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    OUTPUT=str(args.output_dir.resolve())
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CleanOctGpuTests)
    with (args.output_dir/'tests.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),
            'scope':'CPU small contract/mock worker lifecycle; no training or policy rollout'}
    if args.oct_overlay is not None or args.c_overlay is not None:
        try:
            from lerobot_training.train_diffusion_policy import RCMDataset
            from lerobot_training.dit_experience_replay import split_experience_indices, RatioReplacementBatchSampler
            from scripts.research.clean_oct_pipeline import DATA
            dataset=RCMDataset(DATA,n_obs_steps=2,n_action_steps=16,
                               include_velocity=True,include_previous_action=True)
            history=dataset.executed_actions.copy()
            ab=dataset.actions[:,:2].copy()
            receipt=apply_dataset_overlays(dataset,oct_overlay=args.oct_overlay,c_overlay=args.c_overlay)
            np.testing.assert_array_equal(dataset.executed_actions,history)
            np.testing.assert_array_equal(dataset.actions[:,:2],ab)
            original,correction=split_experience_indices(dataset,'teacher')
            sampler=RatioReplacementBatchSampler(len(original),len(correction),.08,1000,16,20261118)
            seen=sum(int(np.count_nonzero(np.asarray(batch)>=len(original))) for batch in sampler)
            if seen != 1280:
                raise AssertionError('8% teacher replay quota mismatch')
            report['real_dataset_integration']={**receipt,'original_windows':len(original),
                'teacher_windows':len(correction),'teacher_samples_per_1000_batches':seen,
                'executed_history_unchanged':True,'AB_labels_unchanged':True}
        except Exception as error:
            report.update(status='failed',integration_error=repr(error))
    write_json(args.output_dir/'summary.json',report)
    print((args.output_dir/'tests.log').read_text(),flush=True)
    print(json.dumps(report),flush=True)
    record('implementation verification',report['status'],args.output_dir,str(report))
    raise SystemExit(0 if report['status']=='passed' else 1)


if __name__=='__main__':
    main()
