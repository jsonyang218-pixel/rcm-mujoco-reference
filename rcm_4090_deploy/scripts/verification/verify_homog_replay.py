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
              'absolute_drift_1000_to_1600_um':100}
        self.assertFalse(direction_gate(snap,fast));self.assertFalse(dev20_gate(fast))
        fast['absolute_drift_1000_to_1600_um']=14.99
        self.assertTrue(direction_gate(snap,fast));self.assertTrue(dev20_gate(fast))
        self.assertFalse(direction_gate({**snap,'first_order_toward_fraction':.70},fast))
        self.assertFalse(dev20_gate({**fast,'final_mean_um':35}))
        self.assertFalse(dev20_gate({**fast,'absolute_drift_1000_to_1600_um':15}))

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
