"""Prelaunch smoke of the exact real recorder and window parser, no fitting."""
from pathlib import Path
import os
import sys
import time
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import campaign as c
for k,v in c.RUNTIME.items():os.environ[k]=v
import data_pipeline as d

def main():
    started=time.monotonic();results=[]
    try:
        c.torch_setup()
        helper=c.imp('paired_smoke',c.PRIOR/'pilot.py');helper.OUT=OUT
        local=c.imp('collection_smoke',OUT/'collection_helper.py')
        local.__dict__.update({k:v for k,v in helper.__dict__.items() if not k.startswith('__')})
        local.OUT=OUT;local.record=c.record;local.save=lambda p,v,update=False:c.save(p,v)
        for target in ('E40','E42'):
            spec=dict(next(s for s in d.plans() if s['target']==target and s['side']=='below' and s['level_um']==70 and s['requested_incoming']=='receding'))
            spec.update(id=f'preflight_{target}_recorder_smoke',teacher_steps=20)
            local.campaign_collect(spec)
            windows=list(d.episode_windows(OUT/'data'/spec['id'],spec,OUT,'train'))
            assert len(windows)==5
            data={k:__import__('torch').stack([values[i] for values,meta in windows]) for i,k in enumerate((*c.KEYS,'action'))}
            metas=[dict(meta,window_index=i) for i,(values,meta) in enumerate(windows)]
            pools=d.pools_for(metas);indices=d.draw_batch(pools,__import__('torch').Generator().manual_seed(12),8)
            assert all(0<=x<len(windows) for x in indices)
            results.append(dict(target=target,real_teacher_steps=20,real16step_windows=len(windows),
                shapes={k:list(v.shape) for k,v in data.items()},sample_indices=indices,
                excluded_from_campaign_training=True,all_window_checks_passed=True))
        # Exercise the same parser on the existing genuine E40/E41 records which will be reused.
        prior=c.imp('parser_existing',c.PRIOR/'pilot.py')
        for spec in prior.plans():
            windows=list(d.episode_windows(c.PRIOR/'data'/spec['id'],spec,c.PRIOR,'train' if spec['target']=='E40' else 'development_E41'))
            assert windows,spec['id']
        c.save('preflight_collection_parser.json',dict(status='passed',time=c.now(),new_recorders=results,
            existing_paired_episodes_validated=24,training_targets=['E40','E42'],E41_development_only=True,
            optimizer_updates=0,elapsed_s=time.monotonic()-started))
        c.record('real_recorder_and_window_parser_smoke',c.read(OUT/'preflight_collection_parser.json'))
    except BaseException as err:
        import traceback
        c.save(f'preflight_collection_failure_{time.time_ns()}.json',dict(error=repr(err),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started))
        c.record('real_recorder_and_window_parser_smoke_failed',dict(error=repr(err)));raise
    finally:
        budget=c.read(OUT/'preparation_budget.json');budget['validation_s']+=time.monotonic()-started;c.save('preparation_budget.json',budget)

if __name__=='__main__':main()
