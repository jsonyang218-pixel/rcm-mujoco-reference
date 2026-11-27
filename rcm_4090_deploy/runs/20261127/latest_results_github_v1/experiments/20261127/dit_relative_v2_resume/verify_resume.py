"""Bounded prelaunch verification; no experiment collection, fitting, or old diagnostics."""
import ast
import json
from pathlib import Path
import tempfile
import torch
import campaign as c
import resume as r

OUT=Path(__file__).resolve().parent


def main():
    checks={}
    for name in ('campaign.py','resume.py','data_pipeline.py','model.py','collection_helper.py'):
        ast.parse((OUT/name).read_text())
    checks['syntax']=True
    original=c.read(r.OLD/'experiment_card.json')
    current=c.read(OUT/'experiment_card.json')
    assert {k:v for k,v in current.items() if k not in ('output_scope','resume')}=={k:v for k,v in original.items() if k!='output_scope'}
    checks['science_card_unchanged']=True
    reused=c.read(OUT/'reuse_manifest.json')
    plans=c.expansion_plans()
    all_ids={s['id'] for s in plans}
    reused_ids={s['id'] for s in reused['reused']}
    pending=set(reused['pending'])
    assert len(plans)==146 and reused_ids.isdisjoint(pending) and reused_ids|pending==all_ids
    assert all((OUT/'data'/name).is_symlink() for name in reused_ids)
    assert all(not (OUT/'data'/name).exists() for name in pending)
    checks['exact_complete_pending_partition']={'reused':len(reused_ids),'pending':len(pending),'smoke_excluded':len(reused['excluded_smoke_summaries'])}
    # A tiny synthetic fixture checks that corrupt action provenance is rejected.
    base=OUT/'verification';base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temp:
        root=Path(temp);folder=root/'data'/'fixture';folder.mkdir(parents=True)
        spec={'id':'fixture','target':'E40','role':'training','teacher_steps':0}
        image=folder/'rgb.png';image.write_bytes(b'fixture-only-no-simulation')
        observations=[dict(index=i,actual_sensor_observe_call=i+1,target_id='E40',role='training',time_s=i*.05,
            rgb_path='data/fixture/rgb.png',rgb_sha256=c.sha(image),signed_OCT_mm=[0,0,0,1],previous_accepted_deg_deg_mm=[0,0,0]) for i in range(3)]
        actions=[dict(pre_observation=i,post_observation=i+1,global_action=i,phase='prefix',accepted_deg_deg_mm=[0,0,0]) for i in range(2)]
        def write(name,value):
            (folder/name).write_text(json.dumps(value))
        write('summary.json',dict(status='completed',branch=spec,teacher_actions=0,observations=3,real_sensor_calls=3,
                                 final_post_RGB_OCT_saved=True,no_midbranch_resets=True,neural_action_calls=0))
        write('teacher_trajectory.json',{'early_stop_reason':None})
        (folder/'observations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in observations))
        (folder/'actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in actions))
        r.branch_check(folder,spec,root)
        actions[1]['pre_observation']=0
        (folder/'actions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in actions))
        try:
            r.branch_check(folder,spec,root)
        except AssertionError:
            checks['corrupt_provenance_rejected']=True
        else:
            raise AssertionError('Corrupted branch was accepted')
    # Exercise the actual checkpoint serializer and restoration with a tiny model.
    # This is a serializer verification only, with zero optimizer/model updates.
    with tempfile.TemporaryDirectory(dir=base) as temp:
        folder=Path(temp)
        model=torch.nn.Linear(3,3).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5)
        c.TRAIN_RNG=torch.Generator().manual_seed(c.SEED)
        c.TRAIN_GRADIENTS={'fixture':{'block':1.0}}
        mean=torch.zeros(3);std=torch.ones(3)
        c.checkpoint_save(folder,model,optimizer,0,mean,std)
        cp=torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=False)
        expected_sampler=torch.randint(0,100,(16,),generator=c.TRAIN_RNG)
        expected_cpu=torch.rand(7)
        expected_cuda=torch.rand(7,device='cuda')
        c.TRAIN_RNG.set_state(cp['sampler_rng']);torch.set_rng_state(cp['torch_rng'])
        torch.cuda.set_rng_state_all(cp['cuda_rng'])
        torch.testing.assert_close(expected_sampler,torch.randint(0,100,(16,),generator=c.TRAIN_RNG),atol=0,rtol=0)
        torch.testing.assert_close(expected_cpu,torch.rand(7),atol=0,rtol=0)
        torch.testing.assert_close(expected_cuda,torch.rand(7,device='cuda'),atol=0,rtol=0)
        restored=torch.nn.Linear(3,3).cuda();restored.load_state_dict(cp['model'])
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=1.0);restored_opt.load_state_dict(cp['optimizer'])
        assert restored_opt.param_groups[0]['lr']==3e-5
        assert cp['actual_updates']==0 and cp['gradient_evidence']==c.TRAIN_GRADIENTS
        assert (folder/'step_000000.pth').exists()
        checks['actual_serializer_model_optimizer_cpu_cuda_sampler_rng_roundtrip']=True
    # The final point is included; a single final-point miss blocks the downstream five.
    assert c.score([0.0]*240,240)['strict20um_pass']
    assert not c.score([0.0]*239+[20.001],240)['strict20um_pass']
    assert not c.score([0.0]*239,240)['strict20um_pass']
    checks['fixed_count_and_final_post_gate']=True
    result=dict(status='passed',time=c.now(),checks=checks,simulation_steps=0,experiment_optimizer_updates=0)
    c.save('verification.json',result)
    c.record('resume_engineering_verification',result)
    # The verification program is also pinned for this launch.
    fingerprints=c.read(OUT/'input_fingerprints.json')
    for name in ('campaign.py','verify_resume.py'):
        fingerprints[str(OUT/name)]=c.sha(OUT/name)
    c.save('input_fingerprints.json',fingerprints)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
