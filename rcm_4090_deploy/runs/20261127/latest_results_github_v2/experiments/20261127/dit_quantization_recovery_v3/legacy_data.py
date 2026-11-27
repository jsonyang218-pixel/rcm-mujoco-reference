"""Task-local real-teacher selection, relative-input checks, and bias diagnostics."""
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
import torch
import campaign as c

OUT=c.OUT
OLD=c.FROZEN_DAY/'dit_terminal_precision_campaign'
KEYS=c.KEYS

def plans():
    result=[]
    # Full Cartesian signs x motion, never bind a lateral sign to motion class.
    for target,side,level,sa,sb,motion in itertools.product(
            (40,42),('below','above'),(15,70,140),(-1,1),(-1,1),('approaching','receding','hold')):
        sign=-1 if side=='below' else 1
        speed=.0015 if sa==sb else .003
        direction=(-sign if motion=='approaching' else sign) if motion!='hold' else 0
        # Setup only: two real ticks establish measured nonzero entry velocity.
        extra=4.4*(speed/.003) if motion=='approaching' else -4.4*(speed/.003) if motion=='receding' else 0.
        lateral_deg=.02 if level<140 else .01
        lateral=[sa*lateral_deg*math.pi/180,sb*lateral_deg*math.pi/180]
        prefix=[[0.,0.,direction*speed]]*2
        name=f'E{target}_{side}_{level}_A{sa:+d}_B{sb:+d}_{motion}'
        result.append(dict(id=name,target=f'E{target}',role='training',side=side,level_um=level,
            requested_incoming=motion,initial_C_offset_m=sign*(level+extra)*1e-6,
            initial_AB_offset_rad=lateral,prefix_actions_deg_deg_mm=prefix,teacher_steps=64,
            designated_anchor_teacher_step=0,AB_sign=[sa,sb],prefix_speed_mm=speed))
    for target in (40,42):
        result.append(dict(id=f'E{target}_zero_hold',target=f'E{target}',role='training',side='zero',level_um=0,
            requested_incoming='hold',initial_C_offset_m=0.,initial_AB_offset_rad=[0.,0.],
            prefix_actions_deg_deg_mm=[[0.,0.,0.]]*2,teacher_steps=224,
            designated_anchor_teacher_step=0,AB_sign=[0,0],prefix_speed_mm=0.))
    return result

def read_rows(path):
    return [json.loads(v) for v in path.read_text().splitlines() if v.strip()]

def metadata(row,prev,episode,target,split,source):
    import numpy as np
    oct4=np.asarray(row['signed_OCT_mm'],float); old=np.asarray(prev['signed_OCT_mm'],float)
    valid=bool(oct4[3]>.5 and old[3]>.5)
    distance=float(np.linalg.norm(oct4[:3])*1000)
    radial=float((np.linalg.norm(oct4[:3])-np.linalg.norm(old[:3]))*1000/.05)
    state=np.asarray(row['state_deg_deg_mm']);pstate=np.asarray(prev['state_deg_deg_mm'])
    vel=(state-pstate)/.05
    settled=bool(np.max(np.abs(vel))<1e-5)
    cls='settled' if settled else 'approaching' if radial<0 else 'receding'
    return dict(episode=episode,target=target,split=split,role='training' if split=='train' else 'development',
        source=source,input_class=cls,side='below' if oct4[2]>0 else 'above' if oct4[2]<0 else 'zero',
        C_sign=int(np.sign(oct4[2])),OCT_sign_xyz=np.sign(oct4[:3]).astype(int).tolist(),
        distance_um=distance,distance_band='zero_0to5' if distance<=5 else 'near_5to20' if distance<=20 else '20to100' if distance<=100 else '100to150',
        speed_band='hold' if settled else 'slow' if abs(vel[2])<.035 else 'fast',
        actual_velocity_deg_deg_mm_s=vel.tolist(),radial_OCT_um_s=radial,OCT_valid=valid,
        true_error_um=row.get('true_error_um'),label='same-tick real teacher accepted ABC',padding=0)

def episode_windows(folder,spec,root,split):
    obs=read_rows(folder/'observations.jsonl');actions=read_rows(folder/'actions.jsonl')
    assert len(obs)==len(actions)+1
    for j in range(2,len(actions)-15):
        cur,past=obs[j],obs[j-1]
        meta=metadata(cur,past,spec['id'],spec['target'],split,str(folder))
        if not meta['OCT_valid'] or meta['distance_um']>150.:
            continue
        assert actions[j]['pre_observation']==j and actions[j]['phase']=='teacher'
        assert all(actions[x]['pre_observation']==x and actions[x]['post_observation']==x+1 for x in range(j,j+16))
        for x in (j-1,j):
            assert c.sha(root/obs[x]['rgb_path'])==obs[x]['rgb_sha256']
        state=torch.tensor([past['state_deg_deg_mm'],cur['state_deg_deg_mm']],dtype=torch.float32)
        oct4=torch.tensor([past['signed_OCT_mm'],cur['signed_OCT_mm']],dtype=torch.float32)
        velocity=(state[-1]-state[-2])/.05
        measured=torch.tensor(cur['actual_velocity_si'],dtype=torch.float64)*torch.tensor([180/math.pi,180/math.pi,1000.])
        torch.testing.assert_close(velocity.double(),measured,atol=7e-5,rtol=1e-3)
        previous=torch.tensor(cur['previous_accepted_deg_deg_mm'],dtype=torch.float32)
        torch.testing.assert_close(previous,torch.tensor(actions[j-1]['accepted_deg_deg_mm'],dtype=torch.float32),atol=0,rtol=0)
        values=[torch.zeros(2,1),state,oct4,velocity,previous,
            torch.tensor([a['accepted_deg_deg_mm'] for a in actions[j:j+16]],dtype=torch.float32)]
        meta.update(pre_observation=j,teacher_step=actions[j]['teacher_step'],
            image_paths=[str(root/past['rgb_path']),str(root/cur['rgb_path'])])
        yield values,meta

def build():
    import numpy as np
    c.torch_setup()
    data={split:{k:[] for k in (*KEYS,'action')} for split in ('train','development_E41')}
    metas={split:[] for split in data}
    coverage=[]
    def add_obs(folder,spec,root,split):
        for values,meta in episode_windows(folder,spec,root,split):
            meta['window_index']=len(metas[split])
            for k,v in zip((*KEYS,'action'),values):data[split][k].append(v)
            metas[split].append(meta)
            if split=='train':coverage.append(meta)
    for spec in plans():
        assert spec['target'] in ('E40','E42')
        add_obs(OUT/'data'/spec['id'],spec,OUT,'train')
    # Reuse the genuine E40 RGB/teacher records, not the prior optimizer or its E41 samples.
    prior=c.imp('relative_prior_plans',c.PRIOR/'pilot.py')
    for spec in prior.plans():
        split='train' if spec['target']=='E40' else 'development_E41'
        add_obs(c.PRIOR/'data'/spec['id'],spec,c.PRIOR,split)
    data={s:{k:torch.stack(v) for k,v in b.items()} for s,b in data.items()}
    assert set(x['target'] for x in metas['train'])=={'E40','E42'}
    assert set(x['target'] for x in metas['development_E41'])=={'E41'}
    target_coverage={}
    for t in ('E40','E42'):
        rows=[r for r in coverage if r['target']==t]
        target_coverage[t]=dict(windows=len(rows),range_um=[min(r['distance_um'] for r in rows),max(r['distance_um'] for r in rows)],
            valid90to150=sum(90<=r['distance_um']<=150 for r in rows),near5=sum(r['distance_um']<=5 for r in rows),
            classes=dict(Counter(r['input_class'] for r in rows)),
            OCT_sign_xyz=[sorted(set(r['OCT_sign_xyz'][i] for r in rows)) for i in range(3)])
        assert all(-1 in signs and 1 in signs for signs in target_coverage[t]['OCT_sign_xyz'])
        assert all(any(r['side']==side and r['input_class']==cls for r in rows) for side in ('below','above') for cls in ('approaching','receding'))
        assert target_coverage[t]['valid90to150']>=4 and target_coverage[t]['near5']>0
    torch.save(data,OUT/'relative_windows.pt')
    c.save('relative_window_manifest.json',dict(**metas,training_targets=['E40','E42'],development_only=['E41'],
        standard_targets_excluded=True,all_labels_real_teacher=True,all_rgb_real_and_hashed=True))
    c.save('coverage.json',dict(status='passed',training_targets=['E40','E42'],development_only='E41',
        switch100um_supported=True,per_target=target_coverage,training_windows=len(metas['train']),
        development_windows=len(metas['development_E41']),scale_source='fixed physical input units + train-only action moments',
        joint_data_input_change_no_single_factor_attribution=True))
    pools=pools_for(metas['train'])
    c.save('sampling_strata.json',dict(hierarchy=['target','C_side','measured_motion','distance_band','speed_band','OCT_sign_xyz','episode','window'],
        strata=[dict(key=list(k),windows=len(v)) for k,v in sorted(pools.items())]))
    c.record('relative_data_ready',c.read(OUT/'coverage.json'))

def pools_for(meta):
    out=defaultdict(list)
    for i,r in enumerate(meta):
        assert r['target'] in ('E40','E42') and r['split']=='train'
        key=(r['target'],r['side'],r['input_class'],r['distance_band'],r['speed_band'],tuple(r['OCT_sign_xyz']),r['episode'])
        out[key].append(i)
    assert out
    return dict(out)

def draw_batch(pools,generator,size=8):
    def choose(items):return items[int(torch.randint(len(items),(1,),generator=generator))]
    keys=list(pools);result=[]
    for _ in range(size):
        eligible=keys
        # Uniform among observed levels at each hierarchy; no E41/XYZ truth used.
        for depth in range(7):
            value=choose(sorted(set(k[depth] for k in eligible)))
            eligible=[k for k in eligible if k[depth]==value]
        result.append(choose(pools[eligible[0]]))
    return result

def verify_inputs(model,data,path):
    from model import integrate,NOISE_SEED,install
    prior_mode=model.training;model.eval();rows=[]
    with torch.no_grad():
        for split in data:
            b=data[split];i=0
            args=tuple(b[k][i:i+1].cuda() for k in KEYS)
            base=integrate(model,args)
            for mode in ('RGB_only','absolute_ABC_only','RGB_and_absolute_ABC'):
                alt=list(args)
                if 'RGB' in mode:alt[0]=torch.randn(1,2,3,224,224,device='cuda')*17+21
                if 'ABC' in mode:alt[1]=torch.randn_like(args[1])*10000+3000
                prediction=integrate(model,tuple(alt))
                assert torch.equal(base,prediction),(split,mode,float((base-prediction).abs().max()))
                rows.append(dict(split=split,ablation=mode,max_abs_action_delta=0.,bit_exact=True))
            # Live relative signals must reach the DiT condition, not a separate control output.
            original_condition=model._condition(*args)
            sensitivities={}
            for key in ('OCT_frame0','OCT_frame1','OCT_valid','velocity','previous_accepted'):
                alt=[x.clone() for x in args]
                if key.startswith('OCT_frame'):alt[2][:,int(key[-1]),:3]+=.025
                elif key=='OCT_valid':alt[2][:,:,3]=0.
                elif key=='velocity':alt[3]+=.05
                else:alt[4]+=.003
                other=model._condition(*alt)
                delta=max(float((a-b).abs().max()) for a,b in zip(original_condition,other))
                assert delta>0,(key,'inactive condition')
                sensitivities[key]=dict(condition_max_abs_delta=delta,
                    action_max_abs_delta=float((base-integrate(model,tuple(alt))).abs().max()))
            # Original class sampler and differentiable sampler must agree across all 16 actions.
            native=type(model).sample(model,*args,steps=4,generator=torch.Generator(device='cuda').manual_seed(NOISE_SEED))
            torch.testing.assert_close(base,native,atol=1e-7,rtol=1e-6)
            rows.append(dict(split=split,live_input_sensitivity=sensitivities,
                original_four_step_sampler_max_abs=float((base-native).abs().max())))
    model.train(prior_mode);model.vision.eval()
    result=dict(status='passed',architecture=dict(blocks=len(model.blocks),dim=model.dit_dim,horizon=model.n_action_steps),
        image_and_absolute_ABC_invariant=True,output_path='original six-block DiT + original four midpoint RF steps + original action denormalization',
        fixed_input_scales=True,checks=rows)
    c.save(path,result)
    return result

def diagnostic():
    import numpy as np
    c.torch_setup();e=c.evaluator();m=e.model
    old_model=c.imp('old_fixed_template',OLD/'model.py')
    cp=torch.load(OLD/'candidate1/checkpoint.pth',map_location='cuda',weights_only=False)
    m.load_state_dict(cp['model'],strict=True);old_model.install(m,cp['action_mean'],cp['action_std']);old_model.allow_visual_cache(m);m.eval()
    data=torch.load(OLD/'original_visual_cache.pt',map_location='cpu',weights_only=False)
    manifest=c.read(c.PRIOR/'window_manifest.json');rows=[]
    with torch.no_grad():
        for split in ('train','development_E41'):
            b=data[split];selected=[i for i,r in enumerate(manifest[split]) if r['level_um']==50 or r['input_class']=='settled' or r.get('strict_hold')]
            for start in range(0,len(selected),8):
                js=selected[start:start+8];pred=old_model.integrate(m,tuple(b[k][js].cuda() for k in KEYS))[:,0].cpu()
                for j,prediction in zip(js,pred):
                    meta=manifest[split][j]
                    rows.append(dict(split=split,target=meta['target'],episode=meta['episode'],window_index=j,
                        teacher_step=meta['teacher_step'],level_um=meta['level_um'],side=meta['side'],input_class=meta['input_class'],
                        strict_hold=meta.get('strict_hold',False),measured_error_um=meta['measured_error_um'],
                        signed_OCT_mm=meta['signed_OCT_mm'],teacher_ABC=b['action'][j,0].tolist(),old_model_ABC=prediction.tolist(),
                        teacher_C_um=float(b['action'][j,0,2]*1000),old_C_um=float(prediction[2]*1000),
                        residual_C_um=float((prediction[2]-b['action'][j,0,2])*1000)))
    c.save('diagnostic_old_same_window.json',dict(status='completed',old_checkpoint=str(OLD/'candidate1/checkpoint.pth'),
        old_checkpoint_sha256=c.sha(OLD/'candidate1/checkpoint.pth'),used_as_stop_gate=False,
        selection='all existing real E40/E41 50um-pair windows plus settled/strict-hold windows',rows=rows))
    with (OUT/'diagnostic_old_same_window_C.csv').open('x',newline='') as f:
        columns=['target','episode','window_index','teacher_step','level_um','side','input_class','strict_hold','measured_error_um','teacher_C_um','old_C_um','residual_C_um']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:groups.setdefault((r['target'],r['side'],r['input_class']),[]).append(r)
    result=dict(status='completed',rows=len(rows),used_as_stop_gate=False,groups=[dict(target=k[0],side=k[1],motion=k[2],windows=len(v),
        mean_teacher_C_um=float(np.mean([r['teacher_C_um'] for r in v])),mean_old_C_um=float(np.mean([r['old_C_um'] for r in v])),
        mean_residual_C_um=float(np.mean([r['residual_C_um'] for r in v]))) for k,v in sorted(groups.items())])
    c.save('diagnostic_old_summary.json',result);c.record('old_policy_same_window_diagnostic',result)

def bias_report():
    import numpy as np
    result={}
    for kind in c.KINDS:
        rows=read_rows(OUT/'candidate1/near'/kind/'cycles.jsonl')
        old=read_rows(OLD/'candidate1/near'/kind/'cycles.jsonl')
        def metrics(rs):
            errors=np.asarray([r['post']['needle_site_world_mm'] for r in rs])-np.asarray([r['post']['target_world_mm'] for r in rs])
            return dict(last200_mean_error_xyz_um=(errors[-200:].mean(0)*1000).tolist(),
                last200_mean_OCT_xyz_um=(np.asarray([r['post']['signed_OCT_mm'][:3] for r in rs[-200:]]).mean(0)*1000).tolist(),
                last200_mean_accepted_C_um=float(np.mean([r['event']['accepted_action_deg_deg_mm'][2] for r in rs[-200:]])*1000),
                last200_max_postXYZ_um=float(np.linalg.norm(errors[-200:],axis=1).max()*1000))
        new=metrics(rows);baseline=metrics(old)
        zero_crossings=[]
        for i in range(1,len(rows)):
            prev,cur=rows[i-1]['pre']['signed_OCT_mm'],rows[i]['pre']['signed_OCT_mm']
            if prev[3]>.5 and cur[3]>.5 and prev[2]*cur[2]<0:
                zero_crossings.append(dict(cycle=i+1,OCT_C_before_um=prev[2]*1000,OCT_C_after_um=cur[2]*1000,
                    accepted_C_before_um=rows[i-1]['event']['accepted_action_deg_deg_mm'][2]*1000,
                    accepted_C_after_um=rows[i]['event']['accepted_action_deg_deg_mm'][2]*1000))
        result[kind]=dict(new=new,old=baseline,measured_C_zero_crossings=zero_crossings)
    old_same_side=len(set(np.sign(v['old']['last200_mean_OCT_xyz_um'][2]) for v in result.values()))==1
    new_same_side=len(set(np.sign(v['new']['last200_mean_OCT_xyz_um'][2]) for v in result.values()))==1
    value=dict(old_four_same_side=old_same_side,new_four_same_side=new_same_side,
        disappearance_of_74um_C_bias=all(v['new']['last200_max_postXYZ_um']<=20 for v in result.values()),
        interpretation='Bias disappearance requires bounded independent postXYZ, not merely changed mean command; joint data/input feasibility only.',arms=result)
    c.save('C_bias_report.json',value);return value
