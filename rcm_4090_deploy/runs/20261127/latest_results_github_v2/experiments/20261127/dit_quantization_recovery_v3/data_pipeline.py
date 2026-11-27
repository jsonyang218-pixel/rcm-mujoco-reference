"""Continuous real G10 teacher data; only E40/E42 enter training/scales."""
import itertools
import math
from collections import Counter,defaultdict
import numpy as np
import torch
import campaign as c
import legacy_data as legacy

OUT=c.OUT
OLD=c.ROOT/'runs/20261127/dit_relative_v2_resume'
KEYS=c.KEYS

def plans():
    result=[]
    dirs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for target,level,d in itertools.product((40,42),(20,35,50,75,100),dirs):
        motion=('hold','approaching','receding')[(dirs.index(d)+int(level)+target)%3]
        direction=0 if motion=='hold' else -1 if motion=='approaching' else 1
        result.append(dict(id=f'E{target}_lateral{level}_A{d[0]:+d}_B{d[1]:+d}',target=f'E{target}',role='training',
            side='lateral',level_um=level,lateral_um=level,AB_sign=list(d),requested_incoming=motion,
            initial_C_offset_m=0.,initial_AB_offset_rad=[0.,0.],
            prefix_actions_deg_deg_mm=[[direction*d[0]*.015,direction*d[1]*.0075,0.]]*2,
            teacher_steps=64,teacher='measured_geometry_discrete_G10',history='continuous_no_reset'))
    for target,side,level in itertools.product((40,42),(-1,1),(20,50,100)):
        result.append(dict(id=f'E{target}_axial{side*level:+d}',target=f'E{target}',role='training',side='below' if side<0 else 'above',
            level_um=level,lateral_um=0,AB_sign=[0,0],requested_incoming='hold',initial_C_offset_m=side*level*1e-6,
            initial_AB_offset_rad=[0.,0.],prefix_actions_deg_deg_mm=[[0.,0.,0.]]*2,teacher_steps=64,
            teacher='measured_geometry_discrete_G10',history='continuous_no_reset'))
    return result

def build():
    data={k:[] for k in (*KEYS,'action')};meta=[];branch_stats=[]
    for spec in plans():
        folder=OUT/'data'/spec['id'];s=c.read(folder/'summary.json')
        assert s['status']=='completed' and s['branch']==spec
        obs=legacy.read_rows(folder/'observations.jsonl');actions=legacy.read_rows(folder/'actions.jsonl')
        assert len(obs)==len(actions)+1==spec['teacher_steps']+3
        assert all(abs(row['time_s']-i*.05)<1e-9 and row['index']==i for i,row in enumerate(obs))
        for i,a in enumerate(actions):
            assert a['pre_observation']==i and a['post_observation']==i+1
            assert a['motor_record']['driver']['variant']=='g10'
            np.testing.assert_array_equal(obs[i+1]['previous_accepted_deg_deg_mm'],a['accepted_deg_deg_mm'])
            if a['phase']=='teacher':
                assert a['teacher_diagnostic']['teacher_only'] and not a['teacher_diagnostic']['truth_used']
                assert a['teacher_diagnostic']['quantized_target_ABCE']==a['motor_record']['quantized_target_robot_ABCE']
        for values,m in legacy.episode_windows(folder,spec,OUT,'train'):
            j=m['pre_observation'];motor=actions[j]['motor_record']
            m.update(source_kind='v3_quantized_teacher',stationary_input=bool(np.max(np.abs(values[3].numpy()))<1e-5),
                crosses_quantization=motor['quantized_target_robot_ABCE']!=motor['actual_before_robot_ABCE'],
                lateral_um=float(np.linalg.norm(obs[j]['signed_OCT_mm'][:2])*1000),
                requested_lateral_um=spec['lateral_um'],AB_sign=spec['AB_sign'])
            for k,v in zip((*KEYS,'action'),values):data[k].append(v)
            meta.append(m)
        branch_stats.append(dict(id=spec['id'],target=spec['target'],setup=s['branch'],initial_um=obs[2]['true_error_um'],
                                 final_um=obs[-1]['true_error_um']))
    nnew=len(meta)
    # Reuse original continuous E40/E42 teacher windows only for axial/near-hold
    # retention. V2's lateral dead-zone labels do not supervise lateral recovery.
    original=torch.load(OLD/'relative_windows.pt',map_location='cpu',weights_only=False)
    manifest=c.read(OLD/'relative_window_manifest.json')
    for i,m in enumerate(manifest['train']):
        oct4=original['train']['observation.oct'][i,-1]
        if float(torch.linalg.vector_norm(oct4[:2]))>.005:continue
        assert m['target'] in ('E40','E42')
        for k in data:data[k].append(original['train'][k][i])
        meta.append(dict(m,source_kind='v2_axial_retention',stationary_input=m['input_class']=='settled',crosses_quantization=None))
    tensors={k:torch.stack(v) for k,v in data.items()}
    assert {m['target'] for m in meta}=={'E40','E42'}
    assert all(m['target'] not in ('E41','E00','E02','E06','E08','E10') for m in meta)
    recovery=[i for i,m in enumerate(meta[:nnew]) if m['distance_um']>15]
    stalled=[i for i in recovery if meta[i]['stationary_input'] and meta[i]['crosses_quantization']]
    assert stalled,'No stationary-input executable recovery labels'
    for target in ('E40','E42'):
        assert any(meta[i]['target']==target for i in stalled)
    pools=dict(recovery=recovery,stalled_recovery=stalled,new_hold=[i for i in range(nnew) if i not in recovery],
               retention=list(range(nnew,len(meta))))
    assert all(pools.values())
    torch.save(dict(train=tensors,pools=pools),OUT/'relative_windows.pt')
    c.save('relative_window_manifest.json',dict(train=meta,training_targets=['E40','E42'],
        excluded_targets=['E41','E00','E02','E06','E08','E10'],all_labels_real_closed_loop=True,exact_future16_no_padding=True,
        no_interpolation_or_synthetic_teacher_labels=True,development_used_for_training=False))
    coverage=dict(status='passed',switch100um_supported=True,training_windows=len(meta),new_windows=nnew,
        stationary_executable_recovery_windows=len(stalled),pools={k:len(v) for k,v in pools.items()},
        branches=branch_stats,by_target={t:dict(windows=sum(m['target']==t for m in meta),
            stationary_executable=sum(meta[i]['target']==t for i in stalled),
            max_sensor_lateral_um=max(m.get('lateral_um',0) for m in meta if m['target']==t)) for t in ('E40','E42')},
        excluded=['E41','E00','E02','E06','E08','E10'])
    c.save('coverage.json',coverage);c.record('v3_contiguous_teacher_data_ready',coverage)

def draw(pools,rng,size=12,repair=False):
    names=('stalled_recovery','recovery','new_hold','retention')
    pattern=(0,1,1,2,2,3,0,1,2,3,1,2) if not repair else (0,1,2,3,2,3,0,1,2,3,2,3)
    result=[]
    for k in range(size):
        pool=pools[names[pattern[k%len(pattern)]]]
        result.append(pool[int(torch.randint(len(pool),(1,),generator=rng))])
    return result

def verify_inputs(model,data,path):
    # No held-out observations are loaded for fitting, scales or validation.
    return legacy.verify_inputs(model,{'train':data['train']},path)
