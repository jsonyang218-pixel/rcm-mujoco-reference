"""Deterministic existing-file CPU audit; no project runtime/model imports."""
from __future__ import annotations

import ast
from collections import Counter
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

START = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT/'runs/20261126/original_neural_terminal_pilot_v1'
REC = BASE/'recovery_report_v1'
STAGES = []
IDENTITY = {}


def now():
    return datetime.now().astimezone().isoformat()


def save(name, value, replace=False):
    path = OUT/name
    assert path.parent == OUT
    with path.open('w' if replace else 'x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def status(stage, state='running', **extra):
    row = dict(time=now(), status=state, stage=stage, model_inference_calls=0, policy_samples=0,
        gradient_updates=0, new_simulation_cycles=0, compute_elapsed_seconds=time.monotonic()-START, **extra)
    save('task_status.json',row,replace=True)
    with (OUT/'status_history.jsonl').open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n')


def complete(stage, result):
    row = dict(time=now(),stage=stage,result=result,command=sys.argv,command_reference=str(OUT/'command.json'),identity=IDENTITY,output=str(OUT))
    encoded = ('\n- '+json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode()
    for path in (OUT/'TEST_LOG.md',ROOT/'TEST_LOG.md'):
        with path.open('ab') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
    STAGES.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)


def csv_save(name, rows):
    with (OUT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def intervals(values):
    result=[]
    for i,v in enumerate(values):
        if not result or result[-1]['class']!=v:
            result.append(dict(start_step=i,end_step=i,count=1,**{'class':v}))
        else:
            result[-1]['end_step']=i
            result[-1]['count']+=1
    return result


def checks_add(checks, name, observed, expected, unit, atol=0.):
    import numpy as np
    a=np.asarray(observed)
    b=np.asarray(expected)
    assert a.shape==b.shape, (name,a.shape,b.shape)
    difference=np.abs(a.astype(float)-b.astype(float))
    bad=difference>atol
    row=dict(name=name,unit=unit,elements=int(a.size),exact_unequal_elements=int(np.count_nonzero(a!=b)),
        tolerance=atol,mismatch_elements=int(bad.sum()),max_abs_difference=float(difference.max()) if a.size else 0.,
        passed=bool(not bad.any()),first_mismatch=np.argwhere(bad)[0].tolist() if bad.any() else None)
    checks.append(row)
    return row


def source_block(path, name, parent=None):
    source=path.read_text()
    tree=ast.parse(source)
    nodes=tree.body
    if parent:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==parent).body
    node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return dict(path=str(path),line=node.lineno,end_line=node.end_lineno,text=ast.get_source_segment(source,node))


def main():
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    assert not (OUT/'summary.json').exists(), 'Completed measurements already exist; inspect instead of overwrite'
    status('input_identity')
    originals=sorted(p for p in BASE.rglob('*') if p.is_file())
    codepaths=[ROOT/'lerobot_training'/n for n in ('eval_policy.py','dit_rectified_flow.py','dit_state_attention.py','dataset_contract.py','train_diffusion_policy.py')]
    codepaths += [ROOT/'rcm/motor_execution.py',ROOT/'rcm/strict_rgbd.py',ROOT/'scripts/research/homog_replay_common.py',
        ROOT/'runs/20261125/train_direction_pair_code_v1/paired_select.py',ROOT.parent/'AGENTS.md']
    card=read(BASE/'experiment_card.json')
    frozen=read(BASE/'input_fingerprints.json')['files']
    files=sorted(set(originals+codepaths+[Path(p) for p in frozen]+[Path(card['init_checkpoint'])]))
    hashes={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    save('input_hashes.json',dict(time=now(),files=hashes,read_only_original_tree=str(BASE)))
    for p,h in frozen.items():
        assert hashes[p]['sha256']==h if p in hashes else sha(p)==h
    for p,row in read(REC/'original_hashes_before.json')['files'].items():
        assert hashes[str(BASE/p)]['sha256']==row['sha256']
    IDENTITY.update(before_sha256=hashes[str(BASE/'before.pth')]['sha256'],after_sha256=hashes[str(BASE/'after.pth')]['sha256'],
        train_target='E40',heldout_target='E41',seed=20261126,source=str(BASE),model_inference_calls=0,gradient_updates=0,new_simulation_cycles=0)
    complete('check1_input_hashes',dict(status='passed',files=len(files),frozen_input_entries=len(frozen),original_62_file_manifest_matches=True))
    status('raw_episode_classification')
    unit=np.asarray([np.pi/180,np.pi/180,.001])
    physical_scale=np.asarray([1.,1.,1000.])
    episodes={}
    raw_rows=[]
    prefix_rows=[]
    raw_checks=[]
    episode_summaries=[]
    candidates=[]
    for target in (40,41):
        for kind in ('below_recede','above_recede'):
            name=f'E{target}_{kind}'
            folder=BASE/'data'/name
            ep=read(folder/'teacher_trajectory.json')
            align=read(folder/'alignment.json')
            tr=ep['trajectory']
            with np.load(folder/'observations.npz',allow_pickle=False) as archive:
                arrays={k:archive[k].copy() for k in archive.files if k!='rgb'}
                # Read the NPY header rather than allocating the 348MB raw RGB stream at this stage.
                import numpy.lib.format as fmt
                with archive.zip.open('rgb.npy') as f:
                    version=fmt.read_magic(f)
                    rgb_shape,fortran,rgb_dtype=fmt._read_array_header(f,version)
            assert rgb_shape==(42,1440,1920,3) and rgb_dtype==np.dtype('uint8')
            assert len(align)==len(ep['motor_records'])==len(tr['actions'])==40
            states=np.asarray(tr['states'])
            actions=np.asarray(tr['actions'])
            before=np.asarray([m['actual_before_task_rad_rad_m'] for m in ep['motor_records']])
            after=np.asarray([m['actual_after_task_rad_rad_m'] for m in ep['motor_records']])
            starts=np.asarray([m['start_time_s'] for m in ep['motor_records']])
            ends=np.asarray([m['end_time_s'] for m in ep['motor_records']])
            requested=np.asarray([m['requested_task_rad_rad_m'] for m in ep['motor_records']])
            velocities=arrays['actual_encoder_velocity_si']/unit
            error=np.asarray(tr['errors'])*1000
            post=arrays['post_true_error_um']
            octs=arrays['oct']
            octnorm=np.linalg.norm(octs[:,:3].astype(np.float64),axis=1)*1000
            current_oct_delta=np.diff(octnorm)/.05
            labels=arrays['accepted_teacher_action']
            checks_add(raw_checks,name+'.ABC_source',states,arrays['state'][2:],'deg,deg,mm')
            checks_add(raw_checks,name+'.pre_motor_state',before,states*unit,'rad,rad,m',1e-11)
            checks_add(raw_checks,name+'.label_accepted',labels,actions,'deg,deg,mm')
            checks_add(raw_checks,name+'.accepted_diagnostics',labels,np.asarray([r['accepted_delta_deg_deg_mm'] for r in tr['control_targets']]),'deg,deg,mm')
            checks_add(raw_checks,name+'.raw_command',np.asarray(tr['raw_actions']),np.asarray([a['raw_teacher'] for a in align]),'deg,deg,mm')
            checks_add(raw_checks,name+'.raw_clipped_accepted',np.clip(np.asarray(tr['raw_actions']),-np.asarray([.2,.1,.03]),np.asarray([.2,.1,.03])),labels,'deg,deg,mm')
            for key,expected,units in [('state',arrays['state'][2:],'deg,deg,mm'),('oct',octs[2:],'mm+valid'),
                ('actual_velocity_si',arrays['actual_encoder_velocity_si'],'rad,rad,m per s'),
                ('previous_action',arrays['previous_action'][2:],'deg,deg,mm'),('time',arrays['timestamp_s'][2:],'s'),
                ('accepted_action',labels,'deg,deg,mm'),('next_state_si',arrays['actual_next_state_si'],'rad,rad,m'),
                ('encoder_after',arrays['encoder_after'],'encoder counts'),('post_true_error_um',post,'um')]:
                checks_add(raw_checks,name+'.alignment_'+key,np.asarray([r[key] for r in align]),expected,units)
            checks_add(raw_checks,name+'.requested_motor_SI',requested,(states+labels)*unit,'rad,rad,m',1e-11)
            checks_add(raw_checks,name+'.recovered_accepted_command',(requested-states*unit)/unit,labels,'deg,deg,mm',1e-11)
            checks_add(raw_checks,name+'.actual_next_state',after,arrays['actual_next_state_si'],'rad,rad,m')
            checks_add(raw_checks,name+'.next_state_continuity',after[:-1],states[1:]*unit,'rad,rad,m',1e-11)
            checks_add(raw_checks,name+'.pre_velocity_previous_interval',arrays['actual_encoder_velocity_si'][1:],(after[:-1]-before[:-1])/(ends-starts)[:-1,None],'rad,rad,m per s',1e-11)
            checks_add(raw_checks,name+'.pre_velocity_state_history',(arrays['state'][2:]-arrays['state'][1:-1])/.05,velocities,'deg,deg,mm per s',1e-9)
            checks_add(raw_checks,name+'.motor_pre_velocity',np.asarray(tr['velocities']),velocities,'deg,deg,mm per s',1e-12)
            checks_add(raw_checks,name+'.previous_accepted',arrays['previous_action'][3:],labels[:-1],'deg,deg,mm')
            checks_add(raw_checks,name+'.same_tick_OCT',np.asarray(tr['oct_navigation_mm']),octs[2:,:3],'mm')
            checks_add(raw_checks,name+'.pre_time_motor',arrays['timestamp_s'][2:],starts,'s')
            checks_add(raw_checks,name+'.execution_duration',ends-starts,np.full(40,.05),'s',1e-12)
            checks_add(raw_checks,name+'.post_score_next_pre',post[:-1],error[1:],'um')
            final_norm=float(np.linalg.norm(np.asarray(ep['final_tip_world_mm'])-ep['target_world_mm'])*1000)
            checks_add(raw_checks,name+'.final_true_distance',np.asarray([post[-1]]),np.asarray([final_norm]),'um')
            episode_rows=[]
            for i in range(40):
                j=i+2
                incoming_truth=(error[i]-error[i-1])/.05 if i else None
                incoming_oct=float(current_oct_delta[j-1])
                incoming='stationary' if abs(velocities[i,2])<=1e-6 else 'receding' if incoming_oct>0 else 'approaching' if incoming_oct<0 else 'ambiguous'
                crossed=bool(octs[j,2]*octs[j+1,2]<0) if j+1<len(octs) else False
                group='hold' if abs(labels[i,2])<=1e-9 and abs(velocities[i,2])<=1e-6 else 'crossing' if crossed else 'approaching' if post[i]<error[i] else 'receding'
                result='approaching' if post[i]<error[i] else 'receding' if post[i]>error[i] else 'stationary'
                actual_delta=(after[i]-before[i])/unit
                row=dict(episode=name,target_id=f'E{target}',segment_name=kind,step=i,global_motor_step=ep['motor_records'][i]['step_index'],
                    pre_time_s=float(starts[i]),post_time_s=float(ends[i]),pre_true_distance_um=float(error[i]),post_true_distance_um=float(post[i]),
                    true_distance_change_um=float(post[i]-error[i]),incoming_truth_radial_um_s=incoming_truth,
                    incoming_OCT_norm_radial_um_s=incoming_oct,incoming_physical_phase=incoming,
                    original_group=group,outgoing_distance_phase=result,crossing_next_OCT_observed=j+1<len(octs),
                    signed_OCT_xyz_mm=octs[j,:3].tolist(),next_signed_OCT_xyz_mm=octs[j+1,:3].tolist() if j+1<len(octs) else None,
                    signed_OCT_change_xyz_um=((octs[j+1,:3].astype(float)-octs[j,:3])*1000).tolist() if j+1<len(octs) else None,
                    observed_OCT_norm_um=float(octnorm[j]),OCT_norm_minus_true_distance_um=float(octnorm[j]-error[i]),
                    ABC_state_deg_deg_mm=states[i].tolist(),actual_encoder_velocity_deg_deg_mm_s=velocities[i].tolist(),
                    previous_accepted_deg_deg_mm=arrays['previous_action'][j].tolist(),raw_teacher_deg_deg_mm=np.asarray(tr['raw_actions'][i]).tolist(),
                    label_accepted_deg_deg_mm=labels[i].tolist(),actual_execution_delta_deg_deg_mm=actual_delta.tolist(),
                    raw_accepted_max_abs_diff=float(np.max(np.abs(labels[i]-tr['raw_actions'][i]))),
                    actual_encoder_stationary_next_interval=bool(np.all(actual_delta==0)),encoder_settled_pre=bool(tr['encoder_settled'][i]),
                    zero_accepted_C=bool(abs(labels[i,2])<=1e-9),original_selected_first=i<=24,
                    future_label_occurrences=sum(s<=i<s+16 for s in range(25)),drop_reason='' if i<=24 else f'insufficient_future_actions:{40-i}<16; no_padding',
                    history_frames=[j-1,j],history_ready=True,RGB_structurally_valid=True,OCT_valid=bool(octs[j,3]>.5))
                raw_rows.append(row)
                episode_rows.append(row)
            for j in (0,1):
                prefix_rows.append(dict(episode=name,observation_index=j,global_motor_step=24+j,
                    time_s=float(arrays['timestamp_s'][j]),ABC_state_deg_deg_mm=arrays['state'][j].tolist(),signed_OCT_xyz_mm=octs[j,:3].tolist(),
                    previous_accepted=arrays['previous_action'][j].tolist(),accepted_setup_command=arrays['previous_action'][j+1].tolist(),
                    teacher_training_candidate=False,reason='Setup prefix only; not a recorded teacher action label',
                    role='history-only; index1 is used by first teacher window' if j else 'extra prefix history; first window uses index1'))
            group_counts=dict(Counter(r['original_group'] for r in episode_rows))
            physical_counts=dict(Counter(r['incoming_physical_phase'] for r in episode_rows))
            selected_counts=dict(Counter(r['incoming_physical_phase'] for r in episode_rows[:25]))
            false_recede=sum(r['original_group']=='receding' and r['true_distance_change_um']==0 for r in episode_rows)
            data=dict(name=name,target=target,kind=kind,folder=folder,arrays=arrays,ep=ep,rows=episode_rows)
            episodes[name]=data
            episode_summaries.append(dict(episode=name,original_classes=group_counts,original_class_intervals=intervals([r['original_group'] for r in episode_rows]),
                incoming_physical_classes=physical_counts,incoming_class_intervals=intervals([r['incoming_physical_phase'] for r in episode_rows]),
                outgoing_distance_classes=dict(Counter(r['outgoing_distance_phase'] for r in episode_rows)),
                selected_incoming_classes=selected_counts,selected_original_classes=dict(Counter(r['original_group'] for r in episode_rows[:25])),
                original_receding_but_distance_equal=false_recede,strict_zero_command_hold_count=sum(r['original_group']=='hold' for r in episode_rows),
                physically_stationary_post_intervals=sum(r['actual_encoder_stationary_next_interval'] for r in episode_rows),
                tail_first_actions_excluded=15,tail_actions_missing_from_any_supervision=0,
                first_teacher_history_ready=True,first_teacher_incoming_phase=episode_rows[0]['incoming_physical_phase'],
                last_OCT_post_missing=True))
            # Descriptive nearest signed-OCT pair: no new gate and no claim of exact state equality.
            outward=[r for r in episode_rows[:25] if r['incoming_physical_phase']=='receding']
            inward=[r for r in episode_rows[:25] if r['incoming_physical_phase']=='approaching']
            if outward and inward:
                x,y=min(((x,y) for x in outward for y in inward),key=lambda pair:np.linalg.norm(np.asarray(pair[0]['signed_OCT_xyz_mm'])-pair[1]['signed_OCT_xyz_mm']))
                candidates.append(dict(episode=name,steps=[x['step'],y['step']],both_history_valid=True,both_selected=True,
                    incoming_phases=[x['incoming_physical_phase'],y['incoming_physical_phase']],
                    true_distance_difference_um=abs(x['pre_true_distance_um']-y['pre_true_distance_um']),
                    signed_OCT_vector_distance_um=float(np.linalg.norm(np.asarray(x['signed_OCT_xyz_mm'])-y['signed_OCT_xyz_mm'])*1000),
                    actual_C_velocity_um_s=[x['actual_encoder_velocity_deg_deg_mm_s'][2]*1000,y['actual_encoder_velocity_deg_deg_mm_s'][2]*1000],
                    accepted_C_labels_um=[x['label_accepted_deg_deg_mm'][2]*1000,y['label_accepted_deg_deg_mm'][2]*1000],
                    exact_same_signed_error=False,interpretation='Descriptive near-neighbor only, not independently matched velocity-controlled branches'))
    assert all(c['passed'] for c in raw_checks), 'Raw alignment check failed; retain measured mismatches'
    save('raw_state_evidence.json',raw_rows)
    csv_save('raw_state_evidence.csv',raw_rows)
    save('prefix_history_evidence.json',prefix_rows)
    save('episode_classifications.json',episode_summaries)
    save('near_error_opposite_velocity_examples.json',candidates)
    save('raw_alignment_checks.json',raw_checks)
    complete('check2_full_raw_episodes',dict(status='passed',episodes=4,states=160,checks=len(raw_checks),
        original_classes=dict(Counter(r['original_group'] for r in raw_rows)),incoming_phases=dict(Counter(r['incoming_physical_phase'] for r in raw_rows)),
        original_receding_but_equal_distance=sum(s['original_receding_but_distance_equal'] for s in episode_summaries)))
    status('window_reconstruction_and_CPU_input_verification')
    windows=read(BASE/'window_manifest.json')
    batches=torch.load(BASE/'real_input_windows.pt',map_location='cpu',weights_only=False)
    before=torch.load(BASE/'before.pth',map_location='cpu',weights_only=False)
    after=torch.load(BASE/'after.pth',map_location='cpu',weights_only=False)
    config=before['config']
    window_checks=[]
    window_rows=[]
    input_rows=[]
    direct=read(BASE/'asset_audit.json')['direct_224_preprocessing']
    stats={k:(config[k],after['config'][k]) for k in config if k.endswith('_mean') or k.endswith('_std')}
    assert all(a==b for a,b in stats.values())
    for split,target in (('train',40),('heldout',41)):
        batch=batches[split]
        manifest=windows[split]
        assert len(manifest)==50
        for kind in ('below_recede','above_recede'):
            name=f'E{target}_{kind}'
            e=episodes[name]
            a=e['arrays']
            selected=[(k,r) for k,r in enumerate(manifest) if r['kind']==kind]
            assert [r['start'] for _,r in selected]==list(range(40-16+1))
            image_map={}
            with np.load(e['folder']/'observations.npz',allow_pickle=False) as archive:
                rgb=archive['rgb']
                assert rgb.dtype==np.uint8 and rgb.shape==(42,1440,1920,3)
                for j in range(1,27):
                    image=Image.fromarray(rgb[j])
                    if not direct:
                        image=image.resize((640,480),Image.Resampling.BILINEAR)
                    image=image.resize((224,224),Image.Resampling.BILINEAR)
                    tensor=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255.
                    tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
                    image_map[j]=tensor
                del rgb
            for k,old in selected:
                s=old['start'];j=s+2
                assert old['history_frames']==[j-1,j] and old['action_frames_inclusive']==[s,s+15]
                assert old['group']==e['rows'][s]['original_group']
                assert old['moving_encoder']==(abs(e['rows'][s]['actual_encoder_velocity_deg_deg_mm_s'][2])>1e-6)
                expected_state=torch.tensor(a['state'][j-1:j+1],dtype=torch.float32)
                expected_oct=torch.tensor(a['oct'][j-1:j+1],dtype=torch.float32)
                expected_action=torch.tensor(a['accepted_teacher_action'][s:s+16],dtype=torch.float32)
                expected_velocity=(expected_state[-1]-expected_state[-2])/.05
                expected_previous=torch.tensor(a['previous_action'][j],dtype=torch.float32)
                expected_image=torch.stack([image_map[j-1],image_map[j]])
                for key,expected,units in (('state',expected_state,'deg,deg,mm'),('oct',expected_oct,'mm+valid'),
                    ('velocity',expected_velocity,'deg,deg,mm per s'),('previous_action',expected_previous,'deg,deg,mm'),('image',expected_image,'normalized_RGB')):
                    measured=batch['observation.'+key][k]
                    checks_add(window_checks,f'{name}/s{s}/{key}',measured.numpy(),expected.numpy(),units)
                checks_add(window_checks,f'{name}/s{s}/label_horizon16',batch['action'][k].numpy(),expected_action.numpy(),'deg,deg,mm')
                model_velocity=batch['observation.velocity'][k].numpy()
                actual_velocity=a['actual_encoder_velocity_si'][s]/unit
                input_rows.append(dict(episode=name,window_index=k,step=s,
                    actual_encoder_velocity_deg_deg_mm_s=actual_velocity.tolist(),production_float32_velocity=model_velocity.tolist(),
                    float32_minus_float64_velocity_deg_deg_mm_s=(model_velocity-actual_velocity).tolist(),
                    physical_velocity_evidence='Actual finite difference, not desired command; model calculation follows original float32 production formula'))
                window_rows.append(dict(split=split,target=f'E{target}',episode=name,window_index=k,raw_step=s,
                    actual_original_group=old['group'],incoming_physical_phase=e['rows'][s]['incoming_physical_phase'],
                    input_time_s=float(a['timestamp_s'][j]),history_times_s=a['timestamp_s'][j-1:j+1].tolist(),
                    history_stop_time_s=float(a['timestamp_s'][j]),observation_indices=[j-1,j],
                    previous_accepted_source_step=s-1 if s else 'prefix25',label_first_source_step=s,label_last_source_step=s+15,
                    executed_first_label_source_step=s,padding=0,history_complete=True,RGB_structural_and_reprocessed_tensor_valid=True,
                    OCT_valid=bool(np.all(a['oct'][j-1:j+1,3]>.5)),supervised_execution_interval_s=[float(a['timestamp_s'][j]),float(a['timestamp_s'][j]+.05)],
                    future_observation_input_leak=False))
            del image_map
    assert all(c['passed'] for c in window_checks), 'Saved input mismatch'
    states_drop=[dict(episode=r['episode'],raw_step=r['step'],selected_as_first=r['original_selected_first'],
        original_group=r['original_group'],incoming_phase=r['incoming_physical_phase'],history_available=r['history_ready'],
        future_actions_available=40-r['step'],horizon_required=16,padding=0,reason=r['drop_reason'],
        appears_as_future_label=r['future_label_occurrences']) for r in raw_rows]
    save('window_evidence.json',window_rows)
    csv_save('window_evidence.csv',window_rows)
    csv_save('all_candidate_states_and_exclusions.csv',states_drop)
    save('saved_input_checks.json',window_checks)
    csv_save('velocity_float32_evidence.csv',input_rows)
    maxv=np.max(np.abs([r['float32_minus_float64_velocity_deg_deg_mm_s'] for r in input_rows]),axis=0).tolist()
    complete('check3_windows_and_inputs',dict(status='passed',raw_teacher_candidates=160,selected=100,excluded_first_states=60,
        missing_history_first_states=0,padding=0,CPU_tensor_checks=len(window_checks),tensor_mismatch_elements=sum(c['mismatch_elements'] for c in window_checks),
        selected_incoming_phases=dict(Counter(r['incoming_physical_phase'] for r in window_rows)),
        hypothesis_initial_recede_lost_to_history=False,max_float32_vs_actual_velocity_difference_deg_deg_mm_s=maxv))
    status('training_code_and_metric_consistency')
    trace=[json.loads(line) for line in (BASE/'training.jsonl').read_text().splitlines()]
    generator=torch.Generator().manual_seed(card['training']['seed'])
    count=Counter();future_count=Counter();successful=0;index_mismatches=0
    for row in trace:
        expected=torch.randint(50,(4,),generator=generator).tolist()
        index_mismatches+=int(expected!=row['sample_indices'])
        if row['optimizer_updated']:
            successful+=1
            for k in row['sample_indices']:
                meta=windows['train'][k]
                count[(meta['kind'],meta['start'])]+=1
                for j in range(meta['start'],meta['start']+16):
                    future_count[(meta['kind'],j)]+=1
    assert not index_mismatches and successful==128 and after['progress']['optimizer_updates']==128
    sampling_rows=[]
    for k,meta in enumerate(windows['train']):
        sampling_rows.append(dict(train_window_index=k,episode=f"E40_{meta['kind']}",raw_step=meta['start'],
            incoming_phase=episodes[f"E40_{meta['kind']}"]['rows'][meta['start']]['incoming_physical_phase'],
            times_sampled_in_successful_updates=count[(meta['kind'],meta['start'])]))
    csv_save('training_sampling_evidence.csv',sampling_rows)
    csv_save('training_future_label_support.csv',[dict(episode='E40_'+kind,raw_action_step=j,
        occurrences_in_successful_training_batches=future_count[(kind,j)]) for kind in ('below_recede','above_recede') for j in range(40)])
    checks=[]
    metrics={}
    for split,target in (('train',40),('heldout',41)):
        for label in ('before','after'):
            path=BASE/'offline'/label/f'{"train_E40" if split=="train" else "heldout_E41"}_predictions.npz'
            with np.load(path,allow_pickle=False) as archive:
                a={k:archive[k].copy() for k in archive.files}
            teacher=batches[split]['action'][:,0].numpy()
            checks_add(checks,f'{split}/{label}/physical_teacher_labels',a['teacher'],teacher,'deg,deg,mm')
            clipped=np.clip(a['raw'],-np.asarray(card['model_caps_deg_deg_mm']),np.asarray(card['model_caps_deg_deg_mm']))
            checks_add(checks,f'{split}/{label}/production_clipping',clipped,a['production_clipped'],'deg,deg,mm')
            groups={}
            meta=windows[split]
            masks=dict(all=np.ones(50,bool),moving_encoder=np.asarray([m['moving_encoder'] for m in meta]),nonzero_C=np.asarray([m['nonzero_C'] for m in meta]))
            masks.update({g:np.asarray([m['group']==g for m in meta]) for g in ('approaching','receding','crossing','hold')})
            masks.update({kind:np.asarray([m['kind']==kind for m in meta]) for kind in ('below_recede','above_recede')})
            masks['teacher_C_positive']=teacher[:,2]>1e-9
            masks['teacher_C_negative']=teacher[:,2]<-1e-9
            for name,mask in masks.items():
                signs=mask&masks['nonzero_C']
                groups[name]=dict(windows=int(mask.sum()),physical_first_action_MAE_deg_deg_um=(np.abs(clipped[mask]-teacher[mask,None]).mean((0,1))*physical_scale).tolist() if mask.any() else None,
                    C_direction_accuracy=float((np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).mean()) if signs.any() else None)
            saved=read(path.with_name(path.name.replace('_predictions.npz','_metrics.json')))
            for name,value in groups.items():
                old=saved['groups'][name]
                assert value['windows']==old['windows']
                assert value['physical_first_action_MAE_deg_deg_um']==old['production_clipped_first_action_mae_deg_deg_um']
                assert value['C_direction_accuracy']==old['C_direction_accuracy']
            flow=float(a['flow_loss'].mean())
            assert flow==saved['flow_loss_4_fixed_probes']
            cfg=before['config'] if label=='before' else after['config']
            mean=np.asarray(cfg['action_mean']);std=np.asarray(cfg['action_std'])
            assert np.all(std>0)
            checks_add(checks,f'{split}/{label}/label_normalization_roundtrip',((teacher-mean)/std)*std+mean,teacher,'deg,deg,mm',1e-15)
            checks_add(checks,f'{split}/{label}/prediction_normalization_roundtrip',((a['raw']-mean)/std)*std+mean,a['raw'],'deg,deg,mm',1e-15)
            metrics[split+'_'+label]=dict(groups=groups,flow_loss=flow,
                normalized_first_C_label_range=[float(((teacher[:,2]-mean[2])/std[2]).min()),float(((teacher[:,2]-mean[2])/std[2]).max())])
    assert all(c['passed'] for c in checks)
    save('metric_and_unit_checks.json',checks)
    save('recomputed_original_metrics.json',metrics)
    gate_checks=[]
    b=metrics['heldout_before']['groups'];a=metrics['heldout_after']['groups']
    for name,group,index,op in [('heldout_moving_C_MAE_um','moving_encoder',2,'<'),
        ('heldout_moving_nonzero_C_direction_accuracy','moving_encoder',None,'>')]+[
            (f'heldout_{group}_{axis}_MAE_deg',group,i,'<=') for group in ('all','moving_encoder') for i,axis in enumerate(('A','B'))]:
        x=b[group]['physical_first_action_MAE_deg_deg_um'][index] if index is not None else b[group]['C_direction_accuracy']
        y=a[group]['physical_first_action_MAE_deg_deg_um'][index] if index is not None else a[group]['C_direction_accuracy']
        gate_checks.append(dict(name=name,before=x,after=y,passed=bool(y<x if op=='<' else y>x if op=='>' else y<=x)))
    old_gate=read(BASE/'offline_gate.json')
    assert all(all(c[k]==old[k] for k in c) for c,old in zip(gate_checks,old_gate['checks']))
    recomputed_gate=dict(status='passed' if all(c['passed'] for c in gate_checks) else 'failed',checks=gate_checks,
        original_gate_preserved=True,original_gate_exact_match=True,no_criterion_changed=True)
    assert recomputed_gate['status']==old_gate['status']
    save('recomputed_gate.json',recomputed_gate)
    block_specs=[('pilot.py','build_windows',None),('pilot.py','arguments',None),('pilot.py','configure_training',None),('pilot.py','train_once',None),
        ('pilot.py','evaluator',None),('pilot.py','offline',None),('pilot.py','neural_rollout',None),
        ('lerobot_training/dit_rectified_flow.py','rectified_flow_loss','DiTRectifiedFlowPolicy'),
        ('lerobot_training/dit_rectified_flow.py','sample','DiTRectifiedFlowPolicy'),
        ('lerobot_training/dit_rectified_flow.py','_normalize_observation','DiTRectifiedFlowPolicy'),
        ('lerobot_training/dit_rectified_flow.py','_condition','DiTRectifiedFlowPolicy'),
        ('lerobot_training/dit_state_attention.py','_condition','StateConditionedDiTRectifiedFlowPolicy'),
        ('lerobot_training/eval_policy.py','get_action','PolicyEvaluator'),('lerobot_training/eval_policy.py','__init__','PolicyEvaluator'),
        ('lerobot_training/eval_policy.py','evaluate_episode','PolicyEvaluator'),('rcm/motor_execution.py','preview_robot','MotorExecution')]
    blocks=[source_block(BASE/path if path=='pilot.py' else ROOT/path,name,parent) for path,name,parent in block_specs]
    save('readonly_code_evidence.json',blocks)
    loss_block=next(b['text'] for b in blocks if b['path'].endswith('dit_rectified_flow.py') and 'def rectified_flow_loss' in b['text'])
    assert 'F.mse_loss(predicted, target_velocity)' in loss_block and 'actions - self.action_mean' in loss_block
    get_action=next(b['text'] for b in blocks if b['path'].endswith('eval_policy.py') and 'def get_action' in b['text'])
    assert 'return action_chunk[0]' in get_action and 'steps=self.flow_integration_steps' in get_action
    assert 'velocity.expand' in get_action and 'previous_action.expand' in get_action
    assert all(not m.requires_grad for b in batches.values() for m in b.values())
    bit_exact=read(BASE/'short_gate/first_input_parity.json')
    assert bit_exact['status']=='passed' and bit_exact['first_action_max_abs_difference']==0
    checkpoint_evidence=read(REC/'checkpoint_training_audit.json')
    assert checkpoint_evidence['vision_all_weights_buffers_bit_exact'] and checkpoint_evidence['optimizer_updates']==128
    vision=[k for k in before['model_state_dict'] if k.startswith('vision.')]
    assert all(torch.equal(before['model_state_dict'][k],after['model_state_dict'][k]) for k in vision)
    code_result=dict(status='passed',train_loss='Unmodified all16x3 normalized velocity MSE, reduction mean; first action nominal1/16 contribution, not explicit first-action objective',
        inference='Original Euler midpoint4-flow sampler; one independent noise draw K1; exec1; no teacher in model action',
        condition='Same _condition: normalized state/OCT/encoder finite-difference velocity/previous accepted command plus actual RGB2; velocity and previous action each broadcast across both history frames',
        train_mode='model.train with vision.eval; no BatchNorm in frozen vision (GroupNorm20); no dropout; step0 eval/train parity previously bit-exact',
        precision='Training original FP16 autocast with dynamic scaler; offline/production FP32; recipe choice, not evidence of time/label shift',
        loading='Both original and after through untouched PolicyEvaluator state-attention architecture; after loader proof reused from recovery; no new model instance',
        sampling='Uniform replacement batch4;129 recorded attempts128 updates1 AMP skip; all recorded indices exactly reconstructed from original CPU seed',
        sampling_index_mismatches=index_mismatches,successful_updates=successful,
        outward_incoming_first_windows_in_training_batches=sum(r['times_sampled_in_successful_updates'] for r in sampling_rows if r['incoming_phase']=='receding'),
        first_action_slots_in_successful_batches=512,vision_tensors_unchanged=len(vision),normalization_keys_unchanged=list(stats),
        reused_evidence=[str(BASE/'step0_parity.json'),str(BASE/'short_gate/first_input_parity.json'),str(REC/'checkpoint_training_audit.json')],
        no_model_inference_executed=True)
    save('training_code_audit.json',code_result)
    complete('check4_training_and_saved_metrics',dict(status='passed',sampling_index_mismatches=0,optimizer_updates_confirmed=128,
        original_metrics_exactly_reproduced=True,normalization_roundtrip_checks=len(checks),
        outward_training_first_action_slots=code_result['outward_incoming_first_windows_in_training_batches'],all_training_slots=512,
        existing_loader_and_bit_exact_proofs_reused=True,new_inference_calls=0))
    status('report_and_final_integrity')
    final_hashes={str(p):sha(p) for p in files}
    changed=[p for p,row in hashes.items() if final_hashes[p]!=row['sha256']]
    assert not changed
    assert not (BASE/'closed_loop').exists()
    save('input_hash_verification.json',dict(status='passed',time=now(),files=len(files),changed_paths=changed,all_original_files_unchanged=True,hashes=final_hashes))
    selected=[r for r in raw_rows if r['original_selected_first']]
    defects=[dict(type='classification_semantics',confirmed=True,
        finding='Original class uses post-minus-pre distance for action outcome, not incoming motion; equal distance also falls into receding unless commandC==0 and pre_velocityC==0',
        impact='4 history-valid outward incoming first windows are reported approaching; stationary nonzero-command tail is incorrectly called receding, not physical outward recovery'),
        dict(type='coverage_and_horizon',confirmed=True,
        finding='Only4/100 selected anchors have outgoing-from-goal incoming velocity; final15 anchors per episode lack full16 future labels and are omitted without padding',
        impact='No first-action stationary/hold validation windows; tail is still present as future supervision; no evidence history warmup removed initial away states'),
        dict(type='time_or_label_shift',confirmed=False,finding='No observed mismatch in stored t/t-1/action horizon and motor interval checks; physical requested/accepted action differs from actual encoder motion by design')]
    limitations=[
        'Intermediate independent MuJoCo needle-site XYZ were not saved. Audit recomputes all stored pre/post distance differences and final XYZ norm, but cannot independently reconstruct each intermediate3D norm without additional kinematics/simulation; none was run.',
        'No terminal post-action RGB/OCT after step39; final crossing cannot be directly assessed from measured OCT. Original fallback copies last OCT; explicitly flagged.',
        'Simulation acquisition timestamps are recorded; true camera latency/hardware clock synchronization and biological/physical robot dynamics are not identified.',
        'RGB shape/dtype and all selected history tensors were independently reprocessed on CPU; this does not independently establish needle visibility or recalibrate OCT3D geometry.',
        'Saved predictions are physical units; latent normalized sampled trajectories are absent. Normalization inverse checked algebraically and static code checked; sampler not rerun.',
        'Original raw strict hold group is zero because accepted C remains nonzero even when encoders settle. Physical settled low-error tail exists, but cannot be relabeled as a demonstrated zero-command hold under the original definition.',
        'Near-error opposite-velocity neighbors exist but are few and not independent matched same-observation branches; no exact equal-signed-error velocity intervention was tested.',
        'Flow-loss improvement versus first-action degradation is a measured association. Horizon16 objective/exec1 evaluation, coverage imbalance and FP16/FP32 difference are candidate explanations, not causal proof.'
    ]
    next_spec=dict(action='只建议：先补齐并验收连续RGB2的同误差、不同实际速度配对终端教师数据；本次不采集、不训练',
        registry='继续使用已核实的homogeneous训练登记与原目标ID；E40拟合、E41整目标留出，禁止冒用/改号',
        history='从运动前缀开始保存全部真实RGB/同拍signedOCT/ABC/上一受理指令，目标锚点前至少2帧真实历史，不清零速度或命令',
        moving='保持MotorExecution实际编码器有限差分速度定义；在history有效以后，以已执行前缀保证每侧至少3个真实远离输入锚点，再接同观测可执行教师制动/恢复',
        matching='同目标、同signedOCT侧，显式保存误差向量接近且实际速度相反的两条分支；验收同误差范围/ABC/OCT距离和实际速度，不以片段名或指令符号代替',
        horizon='每个期望训练/验证锚点之后保存至少16个连续受理教师动作及下一状态；可保留40步主体，但必须追加真实连续尾部使保持锚点也有16步未来；禁止拼段/伪padding',
        hold='实际编码器停稳且<=20um的连续保持段；若要求原strict hold，受理C必须真为0并保留至少16步未来。非零指令但量化不动须单列，不擅自改label',
        admission='采集后先用本审计区分incoming motion和post-action outcome，逐侧核对远离/制动/恢复/保持均有可训练首窗与留出窗，达标前不训练')
    summary=dict(status='completed',time=now(),source=str(BASE),output=str(OUT),raw_episodes=4,raw_teacher_states=160,selected_windows=100,
        train_windows=50,heldout_windows=50,episode_summaries=episode_summaries,
        raw_original_group_counts=dict(Counter(r['original_group'] for r in raw_rows)),
        raw_incoming_phases=dict(Counter(r['incoming_physical_phase'] for r in raw_rows)),
        selected_incoming_phases=dict(Counter(r['incoming_physical_phase'] for r in selected)),
        selected_original_groups=dict(Counter(r['original_group'] for r in selected)),
        excluded_tail_anchors=60,excluded_for_history=0,hypothesis_initial_recede_lost_to_history=False,
        confirmed_findings=defects,no_confirmed_timing_or_label_shift=all(c['passed'] for c in raw_checks+window_checks),
        alignment_checks=len(raw_checks),saved_input_checks=len(window_checks),input_tensor_mismatches=sum(c['mismatch_elements'] for c in window_checks),
        max_float32_vs_float64_actual_velocity_deg_deg_mm_s=maxv,
        original_metrics=metrics,recomputed_original_gate=recomputed_gate,training_code=code_result,next_step_only_recommended=next_spec,limitations=limitations,
        model_inference_calls=0,policy_samples=0,gradient_updates=0,new_simulation_cycles=0,
        input_hashes_verified=len(files),all_input_hashes_unchanged=True,compute_seconds=time.monotonic()-START)
    assert summary['compute_seconds']<60, 'CPU audit60s budget exceeded; report actual duration'
    save('recommended_collection_spec_not_executed.json',next_spec)
    write_report(summary,raw_checks,window_checks,candidates)
    save('summary.json',summary)
    complete('check5_final_integrity_and_report',dict(status='passed',input_files_hash_verified=len(files),changed_original_files=0,
        compute_seconds=summary['compute_seconds'],report=str(OUT/'report.md'),summary=str(OUT/'summary.json'),
        model_inference_calls=0,gradient_updates=0,new_simulation_cycles=0))
    # Read back both logs and explicitly verify each completed check was persisted.
    logs={}
    for path in (OUT/'TEST_LOG.md',ROOT/'TEST_LOG.md'):
        content=path.read_bytes()
        logs[str(path)]=dict(bytes=path.stat().st_size,all_stage_records_present=all(
            json.dumps(r,ensure_ascii=False,allow_nan=False).encode() in content for r in STAGES),
            verified_stages=[r['stage'] for r in STAGES])
        assert logs[str(path)]['all_stage_records_present']
    save('test_log_write_verification.json',dict(status='passed',time=now(),logs=logs))
    status('finished','completed',summary_path=str(OUT/'summary.json'),report_path=str(OUT/'report.md'),TEST_LOG_append_verified=True)
    print('AUDIT_COMPLETED',summary['compute_seconds'],str(OUT/'summary.json'),flush=True)


def write_report(s, raw_checks,window_checks,pairs):
    def maximum(token):
        rows=[x for x in raw_checks if token in x['name']]
        return max(x['max_abs_difference'] for x in rows)
    phases=s['selected_incoming_phases']
    lines=['# 原神经末端最小实验：现有数据与训练链路离线核查','',
        '**为什么没有形成“同误差、不同速度”的独立检验：** 本轮未构造成对、按输入运动方向分层验收的同误差分支，且把“本拍执行后距离减小”当作 approaching，混淆了输入时速度与动作结果。',
        f"实际100个首动作窗含 {phases.get('receding',0)} 个输入时远离、{phases.get('approaching',0)} 个输入时靠近；不是完全没有远离输入。初始远离第0步的真实历史已齐全、也被选中，故“history尚未有效，远离已经结束”的假设被否定。",'',
        '**时序/标签是否存在已证实缺陷：** 未发现保存的 t、t−1、label t、horizon16 起点或0.05s受理执行间隔错位。已证实的是分类语义缺陷和覆盖不足：等距离也被原 else 分支叫作 receding；停稳而受理C非零不满足原 strict hold。',
        '每段末15个状态因未来不足16步不能作为首动作窗，但它们仍在较早窗口的未来监督中。不能把“未作为首窗”写成“全部未训练”。','',
        '**当前唯一最值得做的下一步（仅建议）：** 先补齐并验收连续视觉教师数据中的同误差异速配对及可作首窗的保持段，再考虑任何训练。必须从真实前缀开始记录history，使history有效后仍保留实际远离/制动；每个目标锚点后保留至少16步真实连续监督。保持原速度定义和homogeneous目标登记，不用改定义/改ID解决。完整最小规格见 `recommended_collection_spec_not_executed.json`；本次未执行。','',
        '## 范围与证据','',
        f"CPU确定性文件复算；{s['raw_episodes']} 条原始teacher episode、{s['raw_teacher_states']} 个teacher状态、{s['selected_windows']} 个原保存窗口。模型推理0次、策略重采样0次、梯度更新0次、新仿真0周期。", 
        '逐状态证据：`raw_state_evidence.csv/json`；前缀：`prefix_history_evidence.json`；逐窗：`window_evidence.csv/json`；所有状态丢弃原因：`all_candidate_states_and_exclusions.csv`。','',
        '## 全40步分类、连续区间与真实方向','',
        '原类定义保持不动：先判 abs(label_C)<=1e−9mm 且 abs(pre_actual_vC)<=1e−6mm/s 为hold；再判前/后signed OCT Z反号为crossing；再判 post_error<pre_error 为approaching；其余均receding。',
        '因此它描述的是执行区间结果，不是输入时刻的径向运动；最后一个分支还包含 post==pre。物理说明另外列 incoming_phase 与 outgoing_distance_phase，未改原标签、gate或速度。',
        'incoming速度仍取记录的真实编码器上一执行区间有限差分。方向用相邻同源signed OCT向量模长变化说明；teacher step>=1另有独立真值距离差核对。step0缺前缀独立site XYZ真值，故其incoming truth径向值留空，使用已有signedOCT/编码器方向为证。','',
        '|episode|原分类（完整40步）|输入运动（完整40步）|原分类连续区间（含端点）|',
        '|---|---|---|---|']
    for ep in s['episode_summaries']:
        ranges='；'.join(f"{r['class']} {r['start_step']}–{r['end_step']}" for r in ep['original_class_intervals'])
        lines.append(f"|{ep['episode']}|{json.dumps(ep['original_classes'])}|{json.dumps(ep['incoming_physical_classes'])}|{ranges}|")
    lines+=['',
        '原始 strict hold 的确没有采到：tail虽误差稳定/电机不动，但受理C仍非零；不能说 strict hold 只是被过滤。原始真远离的执行后区间也不能由片段名证明，须查看距离变化。',
        '片段名 below_recede/above_recede 指固定前缀制造的入场状态，不保证随后教师40步都在远离。实际teacher第一拍已经使距离减小，但输入仍包含前一拍朝外速度。','',
        '|episode|已保留的异速近邻原始步|真值距离差µm|signed OCT向量差µm|实际C速度µm/s|教师受理Cµm|',
        '|---|---|---:|---:|---|---|']
    for p in pairs:
        lines.append(f"|{p['episode']}|{p['steps']}|{p['true_distance_difference_um']:.9f}|{p['signed_OCT_vector_distance_um']:.9f}|{p['actual_C_velocity_um_s']}|{p['accepted_C_labels_um']}|")
    lines+=['','这些是少量描述性近邻，并非相同RGB/OCT/ABC的独立速度干预配对；不据此声称已完成同误差异速检验。','',
        '## 40步到25窗的完整过程','',
        '- observations 共42帧：索引0/1是实际前缀motor24/25，索引2对应teacher step0（motor26）。两个prefix是history，不是teacher标签。',
        '- teacher候选step s=0..39；原代码恰好 `range(25)`，与40−16+1相符。窗口历史帧为[s+1,s+2]，输入截止为step s；label为teacher[s:s+16]。',
        '- step0有前缀index1可用，实际全100窗均无history缺失。不存在额外warmup丢step0。',
        '- step0..24有16步真实未来；step25..39分别只剩15..1步，未padding。每段15个、共60个只作为首窗被排除；无有效首窗遗漏的证据。',
        '- 已选窗的两帧RGB均以CPU按原PIL/ToTensor/Normalize路径独立重处理，与全部保存图像张量逐元素核对；原OCT有效位、索引、动作16步连续性同步核对。',
        '- 没有按类别筛选：原代码先收所有25窗，分类只写元数据且不进入loss或采样权重。尾部与输入运动分层覆盖不足是实际数据/固定horizon边界问题，不能称为class筛选bug。','',
        '## 时序、标签、受理与实际位移','',
        f"共 {s['alignment_checks']} 项原轨迹核对、{s['saved_input_checks']} 项逐窗张量核对，容差外错位/张量错配均0。逐项单位、容差、最大差与逐元素差异见 `raw_alignment_checks.json`、`saved_input_checks.json`。", 
        f"原NPZ/traj受理标签最大差 {maximum('.label_accepted'):.9g} deg/deg/mm；历史上一指令最大差 {maximum('.previous_accepted'):.9g}；motor请求回推指令最大差 {maximum('.recovered_accepted_command'):.9g} deg/deg/mm。",
        f"pre时间相对motor.start最大差 {maximum('.pre_time_motor'):.9g}s；执行间隔相对0.05s最大差 {maximum('.execution_duration'):.9g}s；post评分对下一pre评分最大差 {maximum('.post_score_next_pre'):.9g}µm。", 
        'label t 是当拍受理的增量请求；不是“执行后的编码器位移”。Motor G10量化/执行动力学可使两者不同，尤其尾部受理非零但真实编码器零运动；这是原控制接口语义，不可偷偷把label换成实际位移。',
        'ABC命令单位为度/度/mm；送Motor时乘[π/180,π/180,0.001]。motor输出actual velocity是(actual_after−actual_before)/duration；输入t使用的是上一执行区间实际速度，上一受理指令来自t−1。',
        f"原网络实际速度输入来自float32 ABC相减/0.05，与原生产代码一致；相对float64真实速度最大差A/B/C={s['max_float32_vs_float64_actual_velocity_deg_deg_mm_s']}（deg/deg/mm每秒）。这是已有float32量化差，不冒称为零差或改变实际速度定义。", 
        '归一化统计before/after相同；label和物理预测按(mean,std)正/反变换复算，数值误差按明确1e−15容差记录。没有保存采样latent，故不宣称重新验证采样内部的每个反归一化数值。',
        '模型观测只含真实RGB2/ABC/signedOCT/有限差分速度/上一受理指令；真值距离仅进评分、离线分组元数据，不是模型输入。训练不会读取未来图像或未来状态。','',
        '## 训练链路：已证实、设计选择与未知原因','',
        '- 已证实分类语义错误：输入远离与本拍靠近被合并，等距离停稳尾部被称receding。报告保留原组再另列物理说明，没有重写旧指标。',
        '- 已证实链路一致：同一个原架构条件构建与归一化；原入口step0/短门禁bit-exact和after checkpoint加载复用已有证据；本次没有模型实例化、前向或策略采样。',
        '- 原配方设计选择：loss为全部16步×3轴的normalized flow velocity MSE，首步名义权重1/16；部署是同一模型4步flow/K1只执行首步。训练连续随机tau/noise与4步数值采样不是相同计算，但符合原rectified-flow配方。',
        '- 条件中的速度/上一受理指令是当前向量，模型对两帧广播；训练与生产一致。图像编码器冻结/eval、GroupNorm20无BN漂移；其62个张量before/after再次逐位核对。',
        f"- CPU种子重放训练抽样，与129次记录全部一致；128次实际更新的512个首步样本槽中，输入远离仅{s['training_code']['outward_incoming_first_windows_in_training_batches']}槽。见 `training_sampling_evidence.csv`。", 
        '- 训练FP16/autocast、评估FP32是原配方；覆盖不均、horizon目标与首动作验收差异均是待验证原因，不足以从loss下降与MAE变差的相关性推断因果。','',
        '|既有指标复算|C首动作MAE µm before→after|flow loss before→after|','|---|---:|---:|']
    for split,label in (('train','E40训练'),('heldout','E41留出')):
        b=s['original_metrics'][split+'_before'];a=s['original_metrics'][split+'_after']
        lines.append(f"|{label}|{b['groups']['all']['physical_first_action_MAE_deg_deg_um'][2]:.7f} → {a['groups']['all']['physical_first_action_MAE_deg_deg_um'][2]:.7f}|{b['flow_loss']:.9f} → {a['flow_loss']:.9f}|")
    lines+=['','原gate未修改，128更新负结果不外推为架构不可能。','',
        '## 证据局限','']+['- '+x for x in s['limitations']]+['',
        '## 文件与执行验收','',
        f"输入文件 {s['input_hashes_verified']} 个，恢复前/后SHA-256全部一致；含原脚本、原始数据、before/after checkpoint、旧metrics及恢复报告。唯一新产物目录为本目录；部署根TEST_LOG按授权追加。", 
        '每项检查完成后立即写本目录及部署根TEST_LOG并fsync；最终 `test_log_write_verification.json` 会读回核对每项记录确实存在。',
        f"确定性CPU核查耗时 {s['compute_seconds']:.3f}s（小于60s，不含阅读编写）；命令见 `command.json`，脚本见 `audit.py`。", 
        '本次只建议下一步，未采集、未训练、未推理、未仿真、未修改生产代码或原产物、未上传。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        problem=dict(time=now(),status='failed',error=repr(exc),traceback=traceback.format_exc(),
            model_inference_calls=0,gradient_updates=0,new_simulation_cycles=0,elapsed_seconds=time.monotonic()-START)
        path=OUT/'diagnostic_failure.json'
        if not path.exists():
            save(path.name,problem)
        complete('diagnostic_failed',problem)
        status('failed','failed',failure_reason=repr(exc))
        raise
