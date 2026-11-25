"""Real-data regression for Plan8 cached features and two-batch head training."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import traceback

import numpy as np
import torch
from . import workflow as w
from .head import geom_condition, RegressionHead, make_sample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    result={'status':'running','baseline':str(w.BASELINE),'baseline_sha256':w.BASE_SHA,'dataset':str(w.DATA),
        'command':'python -m scripts.research.output_head_sanity.verify_cache --output-dir '+str(out),
        'rtol':1e-5,'atol':1e-6,'smoke_updates_per_arm':2}
    precision=torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
        ev=w.evaluator();model=ev.model.eval()
        for p in model.parameters():p.requires_grad_(False)
        ds=w.dataset();result['units']=w.unit_checks(model,ds)
        from lerobot_training.dit_experience_replay import split_experience_indices
        original,correction=split_experience_indices(ds);order=np.r_[original,correction]
        starts=np.asarray(ds.valid_indices)[order]
        comparisons=[]
        # Match every production cache block: 73 real ~4k-row GEMMs versus
        # independent RCMDataset singleton reads, including every block tail.
        with torch.no_grad():
            for part in np.array_split(starts,max(1,len(starts)//4096)):
                hist=part[:,None]+[-1,0]
                velocity=((ds.states[part]-ds.states[part-1])/(ds.timestamps_s[part]-ds.timestamps_s[part-1])[:,None]).astype(np.float32)
                arrays=[ds.states[hist],np.concatenate([ds.oct_navigation[hist],ds.oct_navigation_valid[hist][...,None]],axis=-1).astype(np.float32),velocity,ds.executed_actions[part-1]]
                inputs=[torch.as_tensor(a,device='cuda') for a in arrays]
                batch=geom_condition(model,*inputs)
                assert torch.get_float32_matmul_precision()=='high'
                for j in (0,len(part)//2,len(part)-1):
                    row=int(part[j]);sample=ds[int(np.flatnonzero(np.asarray(ds.valid_indices)==row)[0])]
                    direct=[sample[k][None].cuda() for k in ('observation.state','observation.oct','observation.velocity','observation.previous_action')]
                    expected=geom_condition(model,*direct)
                    torch.testing.assert_close(batch[j:j+1],expected,rtol=1e-5,atol=1e-6)
                    comparisons.append({'row':row,'batch_size':len(part),'max_abs':float((batch[j:j+1]-expected).abs().max())})
                    # Outer AMP must not change the frozen feature map.
                    with torch.autocast('cuda',dtype=torch.float16):
                        amp_feature=geom_condition(model,*direct)
                    torch.testing.assert_close(expected,amp_feature,rtol=0,atol=0)
                del batch
        result['cache_singleton_checks']=len(comparisons)
        result['cache_singleton_max_abs']=max(v['max_abs'] for v in comparisons)
        result['comparisons']=comparisons
        result['caller_precision_restored']=torch.get_float32_matmul_precision()=='high'
        result['outer_amp_feature_parity']=True
        # Exercise the exact production cache builder and actual backward paths.
        distance,_=w.target_errors(ds)
        cache=w.training_cache(ds,model,distance)
        result['full_cache_shape']=list(cache[0].shape)
        before={name:p.detach().clone() for name,p in model.global_condition.named_parameters()}
        result['smokes']={}
        for arm in ('R1','R2'):
            head,receipt=w.train(arm,model,cache,smoke=True)
            result['smokes'][arm]=receipt
            sample=ds[len(ds)//2]
            inputs=[sample[k][None].cuda() for k in ('observation.image','observation.state','observation.oct','observation.velocity','observation.previous_action')]
            fn=make_sample(head,model,arm)
            torch.testing.assert_close(fn(*inputs),fn(*inputs),rtol=0,atol=0)
        for name,p in model.global_condition.named_parameters():
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        assert not any(p.grad is not None for p in model.parameters())
        result['frozen_encoder_unchanged']=True
        result['smoke_initialization_matched']=result['smokes']['R1']['initial_head_sha256']==result['smokes']['R2']['initial_head_sha256']
        assert result['smoke_initialization_matched']
        result['source_sha256']={str(p.relative_to(w.CODE)):w.sha(p) for p in Path(__file__).parent.glob('*.py')}
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        torch.set_float32_matmul_precision(precision)
        w.save(out/'verification.json',result)
        with (w.RUNTIME/'TEST_LOG.md').open('a') as f:
            f.write(f'\n- {datetime.now().astimezone().isoformat()}: Plan8 cache repair regression {result["status"]}; command={result["command"]}; data={w.DATA}; checkpoint={w.BASELINE} sha256={w.BASE_SHA}; original tolerances rtol1e-5/atol1e-6; checked={result.get("cache_singleton_checks")},max_abs={result.get("cache_singleton_max_abs")}; R1/R2 smoke2updates={result.get("smokes")}; output={out}/verification.json; no retained training checkpoint.\n')
        print({k:v for k,v in result.items() if k not in ('comparisons','source_sha256')},flush=True)


if __name__=='__main__':main()
