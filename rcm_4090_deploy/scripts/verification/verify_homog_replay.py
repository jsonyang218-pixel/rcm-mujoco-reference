"""Small deterministic regression checks for homogeneous replay contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.research.homog_replay_common import (
    ABSOLUTE_DRIFT_KEY, COLLECTION_STEPS, HOLD_START, LATE_DRIFT_START, STEPS, WORKERS,
    collection_steps, validate_collection_budgets,
    direction_gate,dev20_gate,quality_gate,original_indices,pool,record,write_json,
)
from scripts.research.homog_replay_collect import axis_audit
from scripts.research.homog_replay_publish import preserve_log_history
from scripts.research.gpu_episode_pool import artifact_hashes,reuse,signature

OUTPUT=None


class HomogReplayTests(unittest.TestCase):
    def test_quality_gates_fail_closed_at_boundaries(self):
        base={'sampled_frames':200,'oct_truth_median_error_um':.1999,
            'axis_reverse_fractions':dict.fromkeys('ABC',.0199),
            'RGB_OCT_render_provenance_mismatch':False,
            'valid_window_contract_pass_fraction':1.,'all_regimes_complete':True,
            'T3_hold_frames_below_5um':4000}
        self.assertTrue(quality_gate(base))
        for change in ({'oct_truth_median_error_um':.2},
                       {'axis_reverse_fractions':{'A':0,'B':0,'C':.02}},
                       {'RGB_OCT_render_provenance_mismatch':True},
                       {'valid_window_contract_pass_fraction':.999},
                       {'sampled_frames':199},{'T3_hold_frames_below_5um':3999}):
            self.assertFalse(quality_gate({**base,**change}))

    def test_absolute_not_signed_drift_controls_gates(self):
        snap={'axis_reverse_snapshot_counts':{'A':2},'first_order_toward_fraction':.701}
        fast={'final_mean_um':34.99,'drift_1000_to_1600_um':-100,
              ABSOLUTE_DRIFT_KEY:100}
        self.assertFalse(direction_gate(snap,fast));self.assertFalse(dev20_gate(fast))
        fast[ABSOLUTE_DRIFT_KEY]=14.99
        self.assertTrue(direction_gate(snap,fast));self.assertTrue(dev20_gate(fast))
        self.assertFalse(direction_gate({**snap,'first_order_toward_fraction':.70},fast))
        self.assertFalse(dev20_gate({**fast,'final_mean_um':35}))
        self.assertFalse(dev20_gate({**fast,ABSOLUTE_DRIFT_KEY:15}))

    def test_original_pool_excludes_all_plan2(self):
        d=SimpleNamespace(valid_indices=[1,3,5,7],episode_indices=np.array([0,0,838,838,839,839,1038,1038]))
        np.testing.assert_array_equal(original_indices(d),[0,1])

    def test_all_axes_audited_and_zero_holds_not_reverse(self):
        a={'zero':np.ones((3,3)),'pre':np.ones((3,3)),
           'action':np.array([[-1,1,1],[0,0,0],[1,1,-1]])}
        result=axis_audit(a)
        self.assertEqual(result['A']['reverse_frames'],1)
        self.assertEqual(result['B']['reverse_frames'],0)
        self.assertEqual(result['C']['reverse_frames'],1)

    def test_sampling_has_exact_eight_percent_each_epoch(self):
        # Exercise the same sampler constructed by the native loader.
        from lerobot_training.dit_experience_replay import RatioReplacementBatchSampler
        for epoch in range(4):
            sampler=RatioReplacementBatchSampler(123,77,.08,1000,16,20261118+epoch)
            picks=np.asarray(list(sampler))
            self.assertEqual(picks.shape,(1000,16))
            self.assertEqual(int((picks>=123).sum()),1280)
            self.assertEqual(int((picks<123).sum()),14720)

    def test_pool_has_no_wall_clock_deadlines(self):
        p=OUTPUT/'deadline_fixture'
        with patch('scripts.research.homog_replay_common.run_pool',return_value={}) as fake:
            pool({'x':{}},p)
            kw=fake.call_args.kwargs
            for key in ('timeout_s','stage_timeout_s','initializer_timeout_s','execution_timeout_s'):
                self.assertIsNone(kw[key])
            self.assertEqual(kw['snapshot_every'],5)
            self.assertEqual((kw['workers'],kw['max_workers']),(12,12))
            self.assertEqual(kw['admission_min_ram_mib'],4096)

    def test_authorized_mixed_step_card_and_hold(self):
        from scripts.research.homog_replay_pipeline import card
        c=card()
        self.assertEqual((STEPS,HOLD_START,LATE_DRIFT_START,WORKERS),(1600,400,1000,12))
        self.assertEqual(c['collection']['steps_by_regime'],{'T1':1600,'T2':600,'T3':600})
        self.assertEqual(c['collection']['T3_hold_frames'],200)
        self.assertEqual(c['evaluation']['steps'],1600)
        self.assertEqual(c['training']['epochs']*c['training']['batches_per_epoch'],4000)
        self.assertEqual(c['gpu_parallelism']['workers'],12)

    def test_1600_metrics_use_actual_post_action_final_and_full_drift(self):
        from scripts.research.homog_replay_evaluate import episode_metrics,aggregate
        errors=np.full(1600,.020);errors[1000]=.010
        ep={'trajectory':{'actions':np.zeros((1600,3)),'states':np.zeros((1600,3)),
            'errors':errors,'simulation_time_s':np.arange(1600)*.05},
            'motor_records':[{}]*1600,'final_error_mm':.004,
            'early_stop_reason':None,'success':False,
            'terminal_success':{'encoder_settled_at_exit':True,'consecutive_frames_at_exit':3}}
        row=episode_metrics(ep,2)
        self.assertEqual(row['final_error_um'],4)
        self.assertEqual(row[ABSOLUTE_DRIFT_KEY],6)
        self.assertEqual(row['drift_1000_to_1600_um'],-6)
        self.assertNotIn('absolute_drift_400_to_600_um',row)
        self.assertEqual(aggregate([row])[ABSOLUTE_DRIFT_KEY],6)
        ep['trajectory']['actions']=np.zeros((600,3))
        with self.assertRaises(ValueError):episode_metrics(ep,2)

    def test_long_approach_600_step_regression_rejected_before_launch(self):
        tasks=[{'id':'E00','regime':'T1','initial_state_si':[0,0,0],
                'goal_state_si':[0,0,.020]}, {'id':'E40','regime':'T3'}]
        self.assertGreater(validate_collection_budgets(tasks)['E00'],600)
        self.assertEqual([collection_steps(r) for r in ('T1','T2','T3')],[1600,600,600])
        with patch.dict(COLLECTION_STEPS,{'T1':600}):
            with self.assertRaises(ValueError):validate_collection_budgets(tasks)
        with patch.dict(COLLECTION_STEPS,{'T3':599}):
            with self.assertRaises(ValueError):validate_collection_budgets(tasks)

    def test_reused_T1_execution_ast_matches_historical_code(self):
        from scripts.tools.prepare_homog_mixed_budget import ORIGINAL,equivalent_t1
        source=Path('scripts/research/homog_replay_collect.py').read_text()
        self.assertTrue(all(equivalent_t1(ORIGINAL.read_text(),source).values()))
        with self.assertRaises(ValueError):
            equivalent_t1(ORIGINAL.read_text(),source.replace('max_steps=steps,','max_steps=600,'))

    def test_scheduler_accepts_12_and_rejects_13(self):
        from scripts.research.gpu_episode_pool import run_pool
        groups={'x':{'factory':'unused:Fixture','spec':{},'tasks':[]}}
        run_pool(groups,OUTPUT/'twelve_worker_fixture',workers=12,max_workers=12,
                 timeout_s=None,stage_timeout_s=None,initializer_timeout_s=None)
        with self.assertRaises(ValueError):
            run_pool(groups,OUTPUT/'thirteen_worker_fixture',workers=13,max_workers=13)

    def test_evidence_tamper_rejected_before_reuse(self):
        p=OUTPUT/'receipt_fixture';p.mkdir()
        (p/'frame.json').write_text('{"action":[0,0,0]}')
        write_json(p/'receipt.json',{'status':'completed','signature':'frozen',
            'files':artifact_hashes(p),'result':{'ok':True}})
        self.assertEqual(reuse(p,'frozen'),{'ok':True})
        (p/'frame.json').write_text('{"action":[1,0,0]}')
        with self.assertRaises(ValueError):reuse(p,'frozen')

    def test_divergent_log_preserves_parent_prefix(self):
        base=b'published historical bytes\x00\n'
        local=b'different old bytes\x00\n- date: homog_replay_20261123 verification complete\n'
        first,meta=preserve_log_history(base,local)
        self.assertTrue(first.startswith(base));self.assertTrue(meta['merged'])
        second,_=preserve_log_history(base,local+b'\n- next experiment record\n')
        self.assertTrue(second.startswith(first))

    def test_dataset_gate_failure_skips_all_dependent_stages(self):
        from scripts.research import homog_replay_pipeline as pipeline
        output=OUTPUT/'gate_failure_fixture';output.mkdir()
        result={'status':'gate_failed','audit':{'gate_passed':False}}
        with patch.object(pipeline,'preflight',return_value={'identity':{}}), \
             patch.object(pipeline,'check_identity'),patch.object(pipeline,'record'), \
             patch.object(pipeline,'summary'), \
             patch.object(pipeline,'completed_stage',side_effect=lambda out,name,fn:fn()), \
             patch('scripts.research.homog_replay_collect.collect',return_value=result), \
             patch('scripts.research.homog_replay_publish.publish'), \
             patch.object(pipeline,'stage2') as train,patch.object(pipeline,'stage3') as evaluate:
            pipeline.run(output)
            train.assert_not_called();evaluate.assert_not_called()
        status=json.loads((output/'workflow_status.json').read_text())
        self.assertEqual(status['stages'],{'stage1':'gate_failed','stage2':'skipped','stage3':'skipped','stage4':'skipped'})

    def test_completed_collection_receipt_detects_changed_pixels(self):
        from scripts.research import homog_replay_pipeline as pipeline
        out=OUTPUT/'image_reuse_fixture';out.mkdir()
        data=out/'data';data.mkdir();image=data/'frame.png';image.write_bytes(b'original pixels')
        report=out/'report';report.mkdir()
        with patch.object(pipeline,'DATA',data),patch.object(pipeline,'REPORTS',(report,)*4),patch.object(pipeline,'record'):
            pipeline.completed_stage(out,'stage1',lambda:{'status':'completed'})
            image.write_bytes(b'changed pixels')
            with self.assertRaises(ValueError):
                pipeline.completed_stage(out,'stage1',lambda:self.fail('Must not rerun mutated evidence'))

    def test_completed_training_receipt_detects_changed_weights(self):
        from scripts.research import homog_replay_pipeline as pipeline
        out=OUTPUT/'weight_reuse_fixture';out.mkdir()
        weights=out/'weights';weights.mkdir();checkpoint=weights/'step_002000.pth';checkpoint.write_bytes(b'frozen checkpoint fixture')
        report=out/'report';report.mkdir()
        with patch.object(pipeline,'WEIGHTS',weights),patch.object(pipeline,'REPORTS',(report,)*4),patch.object(pipeline,'record'):
            pipeline.completed_stage(out,'stage2',lambda:{'status':'completed'})
            checkpoint.write_bytes(b'changed checkpoint fixture')
            with self.assertRaises(ValueError):
                pipeline.completed_stage(out,'stage2',lambda:self.fail('Must not reuse changed weights'))


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();OUTPUT=args.output_dir.resolve();OUTPUT.mkdir(parents=True,exist_ok=False)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HomogReplayTests))
    report={'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),
        'CPU_reason':'Small deterministic I/O/contract fixtures; model work and real rendering preflight use CUDA/NVIDIA EGL'}
    write_json(OUTPUT/'summary.json',report);record('verification',report['status'],OUTPUT,report)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
