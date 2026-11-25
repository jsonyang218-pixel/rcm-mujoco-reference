"""Identify cached-feature drift without training or relaxing parity tolerances."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from . import workflow as w
from .head import geom_condition


def original_geometry(encoder, state, oct_feature, velocity, previous_action):
    normalized=encoder._normalize_observation(state,oct_feature,velocity,previous_action)
    return encoder.global_condition(torch.cat(normalized,dim=-1).flatten(1))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    result={'status':'running','checkpoint':str(w.BASELINE),'checkpoint_sha256':w.BASE_SHA,
        'dataset':str(w.DATA),'command':'python -m scripts.research.output_head_sanity.diagnose_cache --output-dir '+str(out)}
    original_precision=torch.get_float32_matmul_precision()
    try:
        torch.set_num_threads(1)
        model=w.evaluator().model.eval();ds=w.dataset()
        from lerobot_training.dit_experience_replay import split_experience_indices
        original,correction=split_experience_indices(ds)
        order=np.r_[original,correction]
        starts=np.asarray(ds.valid_indices)[order]
        parts=np.array_split(starts,max(1,len(starts)//4096))
        result['measurements']=[]
        # Match the actual first, middle and final cache chunks and singleton paths.
        for chunk_index in (0,len(parts)//2,len(parts)-1):
            part=parts[chunk_index];hist=part[:,None]+[-1,0]
            velocity=((ds.states[part]-ds.states[part-1])/(ds.timestamps_s[part]-ds.timestamps_s[part-1])[:,None]).astype(np.float32)
            arrays=[ds.states[hist],np.concatenate([ds.oct_navigation[hist],ds.oct_navigation_valid[hist][...,None]],axis=-1).astype(np.float32),velocity,ds.executed_actions[part-1]]
            inputs=[torch.as_tensor(a,device='cuda') for a in arrays]
            start=int(np.sum([len(p) for p in parts[:chunk_index]]))
            sample=ds[int(order[start])]
            exact=[sample[k][None].cuda() for k in ('observation.state','observation.oct','observation.velocity','observation.previous_action')]
            field_drift={k:float((a[:1]-b).abs().max()) for k,a,b in zip(('state','oct','velocity','previous'),inputs,exact)}
            for precision in ('high','highest'):
                torch.set_float32_matmul_precision(precision)
                with torch.no_grad():
                    batched=original_geometry(model,*inputs)[:1]
                    singleton=original_geometry(model,*exact)
                    identical_input=original_geometry(model,*[a[:1] for a in inputs])
                result['measurements'].append({'chunk':chunk_index,'row':int(part[0]),'precision':precision,
                    'input_max_abs_by_field':field_drift,
                    'batch_vs_dataset_singleton_max_abs':float((batched-singleton).abs().max()),
                    'same_input_batch_vs_singleton_max_abs':float((batched-identical_input).abs().max()),
                    'passes_original_tolerance':bool(torch.isclose(batched,singleton,rtol=1e-5,atol=1e-6).all()),
                    'path':'unprotected original geometry; protected helper tested by verify_cache' })
        result['status']='completed'
    except BaseException as exc:
        import traceback
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        torch.set_float32_matmul_precision(original_precision)
        w.save(out/'diagnosis.json',result)
        with (w.RUNTIME/'TEST_LOG.md').open('a') as f:
            f.write(f'\n- {datetime.now().astimezone().isoformat()}: Plan8 cache numerical diagnosis {result["status"]}; command={result["command"]}; data={w.DATA}; checkpoint={w.BASELINE} sha256={w.BASE_SHA}; no training; measurements={result.get("measurements")}; evidence={out}/diagnosis.json.\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
