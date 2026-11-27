"""Authorized first DiT terminal-precision batch; all additions are task-local."""
from __future__ import annotations
import argparse
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FROZEN_DAY = ROOT/'runs/20261126'
PRIOR = FROZEN_DAY/'neural_paired_terminal_minifit_v1'
BASE = FROZEN_DAY/'neural_exec124_50um_v1'
ORIGINAL = FROZEN_DAY/'original_neural_terminal_pilot_v1'
SOURCE = ROOT/'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
REF = Path(json.loads((PRIOR/'experiment_card.json').read_text())['init_checkpoint'])
REF_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
KEYS = ('observation.image', 'observation.state', 'observation.oct', 'observation.velocity', 'observation.previous_action')
KINDS = ('below_approaching', 'below_receding', 'above_approaching', 'above_receding')
SEED = 20261126
DESKTOP = dict(DISPLAY=':1', XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
RUNTIME = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
               RCM_DRIVER_VARIANT='g10', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', TORCH_DISABLE_NATIVE_JIT='1', MPLBACKEND='Agg')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
sys.modules['campaign'] = sys.modules[__name__]
START = None
STOP_REQUESTED = False
TRAIN_RNG = None
TRAIN_GRADIENTS = {}

def request_training_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def now(): return datetime.now().astimezone().isoformat()
def default(v):
    if hasattr(v, 'tolist'): return v.tolist()
    if isinstance(v, Path): return str(v)
    raise TypeError(type(v).__name__)
def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p = Path(p)
    if not p.is_absolute(): p = OUT/p
    assert p.is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=default)+'\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
def append(p, value):
    with Path(p).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=default)+'\n'); f.flush()
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def imp(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
def record(stage, result):
    row = dict(time=now(), experiment=OUT.name, stage=stage, result=result, card=str(OUT/'experiment_card.json'),
               command=sys.argv, far_checkpoint_sha256=REF_SHA,
               init_sha256=read(OUT/'experiment_card.json').get('init_sha256') if (OUT/'experiment_card.json').exists() else None,
               output=str(OUT))
    line = ('\n- '+json.dumps(row, ensure_ascii=False, allow_nan=False, default=default)+'\n').encode()
    receipts = []
    for p in (OUT/'TEST_LOG.md', ROOT/'TEST_LOG.md'):
        with p.open('a+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.seek(0, 2); offset = f.tell()
            f.write(line); f.flush(); os.fsync(f.fileno()); f.seek(offset)
            assert f.read(len(line)) == line
        receipts.append(dict(path=str(p), offset=offset, bytes=len(line), exact_readback=True))
    append(OUT/'test_log_receipts.jsonl', dict(time=now(), stage=stage, receipts=receipts))
    print(json.dumps(row, ensure_ascii=False, default=default), flush=True)
def status(phase, **kw):
    value = dict(time=now(), phase=phase, pid=os.getpid(), budget_elapsed_s=None if START is None else time.monotonic()-START, **kw)
    save('task_status.json', value); append(OUT/'status_history.jsonl', value)
def torch_setup():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    return torch
def evaluator():
    from lerobot_training.eval_policy import PolicyEvaluator
    e = PolicyEvaluator(str(REF), device='cuda', num_action_samples=1, n_action_execute_steps=1,
                        flow_integration_steps=4, measure_inference=True, controller_variant='feedback')
    assert e.fine_evaluator is None and len(e.model.blocks)==6 and e.model.dit_dim==384
    assert e.is_dit_rectified_flow and e.n_obs_steps==2 and e.n_action_steps==16
    return e
def checkpoint_load(e, path):
    import torch
    from model import install
    cp = torch.load(path, map_location='cuda', weights_only=False)
    e.model.load_state_dict(cp['model'], strict=True)
    install(e.model, cp['action_mean'], cp['action_std']); e.model.eval()
    return cp


def train(candidate):
    phase_started=time.monotonic()
    torch=torch_setup()
    from model import integrate,install,NOISE_SEED
    assert candidate=='candidate1'
    global TRAIN_RNG, TRAIN_GRADIENTS
    signal.signal(signal.SIGTERM, request_training_stop)
    signal.signal(signal.SIGINT, request_training_stop)
    signal.signal(signal.SIGHUP, request_training_stop)
    folder=OUT/candidate;folder.mkdir(exist_ok=True)
    e=evaluator();m=e.model;install(m)
    data=torch.load(OUT/'relative_windows.pt',map_location='cpu',weights_only=False)
    manifest=read(OUT/'relative_window_manifest.json')
    assert set(v['target'] for v in manifest['train'])=={'E40','E42'}
    assert all(v['target']=='E41' for v in manifest['development_E41'])
    sampler=imp('relative_balanced',OUT/'data_pipeline.py')
    if not (folder/'input_ablation_before.json').exists():
        sampler.verify_inputs(m,data,folder/'input_ablation_before.json')
    # Only the current training subset determines physical output mean/std.
    y=data['train']['action'][:,0]
    mean=y.mean(0);raw_std=y.std(0,unbiased=False);std=raw_std.clamp_min(torch.tensor([1e-4,1e-4,1e-5]))
    global_scales={k:v.detach().cpu().clone() for k,v in m.named_buffers() if any(k.startswith(n) for n in ('state_','oct_','velocity_','previous_action_'))}
    install(m,mean,std)
    save(folder/'scales.json',dict(action_mean=mean,action_std=std,raw_std=raw_std,floors=[1e-4,1e-4,1e-5],
          action_statistics_source='current train only; development excluded',global_input_buffers=global_scales,precision='FP32',
          OCT_normalization='relative50um; valid retained; RGB/absoluteABC fixed; velocity0.1deg_deg_mm_s; previous0.005deg_deg_mm',noise_seed=NOISE_SEED))
    for n,p in m.named_parameters():p.requires_grad_(not n.startswith('vision.'))
    m.train();m.vision.eval()
    initial_path=folder/'initial_weights.pth'
    if initial_path.exists():
        before=torch.load(initial_path,map_location='cpu',weights_only=False)['model']
    else:
        before={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
    parameter_rows=[dict(name=n,count=p.numel(),trainable=p.requires_grad) for n,p in m.named_parameters()]
    save(folder/'parameters.json',dict(class_name=type(m).__name__,blocks=len(m.blocks),dim=m.dit_dim,horizon=m.n_action_steps,parameters=parameter_rows))
    active=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(active,lr=3e-5,weight_decay=1e-4)
    pools=sampler.pools_for(manifest['train']);rng=torch.Generator().manual_seed(SEED)
    # Data tensors fit on GPU after RGB has been replaced by frozen full tokens.
    train_data={k:v.cuda() for k,v in data['train'].items()}
    updates=0;gradient_evidence={}
    TRAIN_RNG=rng
    if (folder/'checkpoint.pth').exists():
        cp=torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=False)
        assert cp['init_sha256']==REF_SHA and cp['training_targets']==['E40','E42']
        m.load_state_dict(cp['model'],strict=True); opt.load_state_dict(cp['optimizer'])
        updates=int(cp['actual_updates']); assert 0<=updates<=3000
        rng.set_state(cp['sampler_rng'])
        torch.set_rng_state(cp['torch_rng'])
        torch.cuda.set_rng_state_all(cp['cuda_rng'])
        gradient_evidence=cp['gradient_evidence']
        record('training_restored',dict(actual_updates=updates,checkpoint=str(folder/'checkpoint.pth'),optimizer_restored=True,rng_restored=True))
    else:
        torch.save(dict(model={n:v.detach().cpu() for n,v in m.state_dict().items()},action_mean=mean,action_std=std),initial_path)
    TRAIN_GRADIENTS=gradient_evidence
    checkpoint_save(folder,m,opt,updates,mean,std)
    started=time.monotonic()
    time_limit=float(os.environ.get('DIT_STEP_SECONDS','850'))-25-(started-phase_started)
    budget_reason=None
    try:
        for step in range(updates+1,3001):
            if STOP_REQUESTED or time.monotonic()-started>=time_limit:
                budget_reason='stage_time_reserve';break
            idx=sampler.draw_batch(pools,rng,size=8)
            b={k:v[idx] for k,v in train_data.items()};args=tuple(b[k] for k in KEYS)
            opt.zero_grad(set_to_none=True)
            condition=m._condition(*args)
            prediction=integrate(m,args,condition=condition)
            first=(((prediction[:,0]-b['action'][:,0])/m.action_std)**2).mean()
            normalized=(b['action']-m.action_mean[None,None])/m.action_std[None,None]
            noise=torch.randn_like(normalized);tau=torch.rand(len(idx),device='cuda');t=tau[:,None,None]
            velocity=m._predict_velocity_from_condition((1-t)*noise+t*normalized,tau,*condition)
            rf=((velocity-(normalized-noise))**2).mean()
            loss=first+.005*rf
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if step in (1,1000,2000,3000):
                gradient_evidence[str(step)]={n:float(p.grad.norm()) for n,p in m.named_parameters()
                    if p.grad is not None and ('self_attention.q.weight' in n or n=='output.weight' or 'temporal_bias.templates' in n)}
            gradient=torch.nn.utils.clip_grad_norm_(active,1.)
            if not bool(torch.isfinite(gradient)):raise FloatingPointError('Nonfinite gradients')
            opt.step();updates+=1
            row=dict(time=now(),update=updates,loss=float(loss.detach()),first_action_mse=float(first.detach()),rf_mse=float(rf.detach()),
                     grad_norm=float(gradient),elapsed_s=time.monotonic()-started)
            append(folder/'training_curve.jsonl',row)
            if step==1 or step%25==0:print(json.dumps(row),flush=True)
            if step%500==0:
                save(folder/'progress.json',dict(actual_updates=updates,planned_updates=3000,elapsed_s=time.monotonic()-started))
                # Save weights before any statistics/reporting, even on later failure.
                checkpoint_save(folder,m,opt,updates,mean,std)
    finally:
        checkpoint_save(folder,m,opt,updates,mean,std)
    del opt,train_data
    save(folder/'resume_progress.json',dict(actual_updates=updates,planned_updates=3000,interrupted=updates<3000,optimizer_rng_saved=True))
    if updates<3000:
        record('training_budget_partial',dict(actual_updates=updates,planned_updates=3000,checkpoint=str(folder/'checkpoint.pth')))
        raise TimeoutError('Training paused below 3000 updates; saved weights optimizer and RNG; evaluations blocked')
    delta={n:float((p.detach().cpu()-before[n]).norm()) for n,p in m.named_parameters()}
    assert all(delta[n]==0 for n in delta if n.startswith('vision.'))
    for k,v in global_scales.items():assert torch.equal(dict(m.named_buffers())[k].cpu(),v)
    assert all(any(delta[n]>0 for n in delta if n.startswith(f'blocks.{i}.self_attention.')) for i in range(6))
    assert all(any(v>0 for probe in gradient_evidence.values() for n,v in probe.items()
                   if n.startswith(f'blocks.{i}.self_attention.')) for i in range(6))
    evidence=dict(actual_updates=updates,planned_updates=3000,stage_budget_reason=budget_reason,minimum_prescribed_updates=3000,
                  all_six_attention_blocks_changed=True,vision_bit_exact=True,global_input_buffers_bit_exact=True,
                  gradient_probes=gradient_evidence,parameter_delta_l2=delta,checkpoint_sha256=sha(folder/'checkpoint.pth'))
    save(folder/'parameter_update_evidence.json',evidence);record('DiT_training_checkpoint_saved',dict(candidate=candidate,actual_updates=updates,checkpoint=str(folder/'checkpoint.pth'),sha256=evidence['checkpoint_sha256']))
    m.eval();diagnostics={}
    with torch.no_grad():
        for split,b in data.items():
            pred=[]
            for start in range(0,len(b['action']),8):
                args=tuple(b[k][start:start+8].cuda() for k in KEYS)
                pred.append(integrate(m,args)[:,0].cpu())
            pred=torch.cat(pred);err=(pred-b['action'][:,0]).abs()
            diagnostics[split]=dict(windows=len(pred),MAE_deg_deg_mm=err.mean(0),C_MAE_um=float(err[:,2].mean()*1000),
                                     used_as_closed_loop_gate=False)
    save(folder/'offline_diagnostic.json',diagnostics)
    record('offline_action_diagnostic_no_rejection_gate',dict(candidate=candidate,metrics=diagnostics))
    save(folder/'summary.json',dict(status='trained',actual_updates=updates,minimum_updates_reached=updates==3000,diagnostics=diagnostics))
    sampler.verify_inputs(m,data,folder/'input_ablation_after.json')
    if updates!=3000:raise TimeoutError('Actual updates below3000; budget partial, no closed-loop authorization')


def checkpoint_save(folder,m,opt,updates,mean,std):
    import torch
    cp=dict(model={n:v.detach().cpu() for n,v in m.state_dict().items()},action_mean=mean.cpu(),action_std=std.cpu(),
            actual_updates=updates,init_sha256=REF_SHA,policy='target_relative_DiT_RF',noise_seed=434341,
            optimizer=opt.state_dict(),sampler_rng=TRAIN_RNG.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),gradient_evidence=TRAIN_GRADIENTS,time=now(),relative_input_scales=__import__('model').INPUT_SCALES,training_targets=['E40','E42'],development_only=['E41'])
    tmp=folder/'checkpoint.tmp';torch.save(cp,tmp);tmp.replace(folder/'checkpoint.pth')
    if updates%500==0:
        path=folder/f'step_{updates:06d}.pth'
        if not path.exists():torch.save(cp,path)


def score(errors,expected):
    import numpy as np
    a=np.asarray(errors);run=longest=0
    for x in a<=20:
        run=run+1 if x else 0;longest=max(longest,run)
    return dict(completed_cycles=len(a),endpoint_um=float(a[-1]) if len(a) else None,
          first_enter20_post_cycle=int(np.flatnonzero(a<=20)[0]+1) if np.any(a<=20) else None,
          last200_max_post_error_um=float(a[-200:].max()) if len(a)>=200 else None,
          longest_le20_cycles=longest,terminal_le20_cycles=run,last200_includes_final_post=True,
          strict20um_pass=bool(len(a)==expected and len(a)>=200 and max(a[-200:])<=20))


def evaluate(candidate,kind=None,index=None):
    torch=torch_setup();import numpy as np
    from lerobot_training.eval_policy import StrictRGBDSensor,visual_servo_settings,TASK_TO_DEG_DEG_MM,DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    from scripts.research.homog_replay_common import QUALITY
    from rcm.scene_v9 import RCMSceneV9
    original=imp('original_history',ORIGINAL/'pilot.py')
    near=kind is not None;name=kind if near else f'E{index:02d}'
    folder=OUT/candidate/('near' if near else 'standard')/name;folder.mkdir(parents=True,exist_ok=False)
    training_metadata=read(OUT/candidate/'summary.json')
    e=evaluator();far=e.model
    if not near:e.model=evaluator().model
    checkpoint_load(e,OUT/candidate/'checkpoint.pth');local=e.model
    checkpoint_before=sha(OUT/candidate/'checkpoint.pth')
    save(folder/'weights_before.json',dict(time=now(),checkpoint_sha256=checkpoint_before,far_v5_sha256=sha(REF),
        input_scales=__import__('model').INPUT_SCALES,training_updates=training_metadata['actual_updates'],eval_updates=0,truth_setup_scoring_only=True))
    scene=sensor=None;rows=[];previous=np.zeros(3);pending={};history=[];cache={};active=near;sample_calls=0
    expected=240 if near else 1600;seed=434341 if near else 42+index
    route_counts=dict(original_v5_DiT_RF=0,fixed_noise_near_DiT_RF=0,original_RGB2_startup_zero=0,teacher=0,geometry=0)
    try:
        np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        scene=RCMSceneV9()
        if near:
            task=read(OUT/'experiment_card.json')['near_task'];world=np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
            scene.set_target_world(world)
            specs=imp('paired_specs',PRIOR/'pilot.py').plans()
            spec=next(s for s in specs if s['target']=='E41' and s['level_um']==50 and s['side']+'_'+s['requested_incoming']==kind)
            q=scene.target_task(validate=True).copy();q[2]+=spec['initial_C_offset_m'];scene.set_task(q)
        else:
            task=read(ROOT/'configs/v12_reachable_development_targets_0p35um.json')['targets'][index]
            world=np.asarray(scene.robot_to_world(np.asarray(task['reachable_target_robot_mm'])));scene.set_target_world(world)
        if scene.motor_execution is None:scene.enable_motor_execution();scene.reset_motor_time(0.)
        settings=visual_servo_settings(scene);settings.update(perception_backend='strict_rgbd',rgbd_render_quality=QUALITY.copy())
        sensor=StrictRGBDSensor(scene,settings);real_observe=sensor.observe
        def capture():
            t=float(scene.motor_execution.time_s)
            if cache.get('t')==t:return cache['obs'],cache['row']
            sensor.observation_time_s=t;obs=real_observe();oct4=np.zeros(4,np.float32)
            nav=obs.get('oct_navigation_depth');delta=None if nav is None else nav.get('delta_xyz_m')
            if delta is not None and len(delta)==3 and np.isfinite(delta).all():oct4[:3]=np.asarray(delta)*1000;oct4[3]=1
            tip=scene.site_mm('needle_tip').copy()
            row=dict(time_s=t,state_deg_deg_mm=scene.current_task()*TASK_TO_DEG_DEG_MM,signed_OCT_mm=oct4.copy(),
                 actual_velocity_deg_deg_mm_s=scene.motor_execution.actual_velocity*TASK_TO_DEG_DEG_MM,
                 previous_accepted_deg_deg_mm=previous.copy(),encoder_ABCE=list(scene.motor_execution.backend.robot),
                 needle_site_world_mm=tip,target_world_mm=world,
                 true_error_xyz_um=(tip-world)*1000,true_error_um=float(np.linalg.norm(tip-world)*1000))
            history.append((obs,row));cache.update(t=t,obs=obs,row=row)
            if len(history)>3:history.pop(0)
            return obs,row
        sensor.observe=lambda:capture()[0]
        capture()
        if near:
            unit=np.asarray([np.pi/180,np.pi/180,.001])
            for raw in spec['prefix_actions_deg_deg_mm']:
                _,pre=capture();action=np.asarray(raw)
                scene.apply_prepared_motor(scene.motor_execution.preview(scene.current_task()+action*unit,.05));previous=action.copy()
                _,post=capture();append(folder/'setup_prefix.jsonl',dict(pre=pre,post=post,accepted_action=action))
            current=history[-1][1];past=history[-2][1]
            distance=current['true_error_um'];radial=(distance-past['true_error_um'])/.05
            radial_oct=(np.linalg.norm(current['signed_OCT_mm'][:3])-np.linalg.norm(past['signed_OCT_mm'][:3]))*1000/.05
            actual_side='below' if scene.current_task()[2]<scene.target_task(validate=True)[2] else 'above'
            gate=bool(49<=distance<=51 and abs(current['actual_velocity_deg_deg_mm_s'][2])>1e-6 and actual_side==spec['side']
               and current['signed_OCT_mm'][3]==past['signed_OCT_mm'][3]==1
               and (radial<0 and radial_oct<0 if spec['requested_incoming']=='approaching' else radial>0 and radial_oct>0))
            save(folder/'start_gate.json',dict(passed=gate,initial=current,radial_XYZ_um_s=radial,radial_OCT_um_s=float(radial_oct),spec=spec))
            if not gate:raise RuntimeError('Original 50um real-motion start gate mismatch')
        prefix_previous=previous.copy();initial=capture()[1];save(folder/'initial_state.json',initial)
        original_reset=e.reset_buffer;original_action=e.get_action;original_accepted=e.record_applied_action
        if near:
            past_obs,past=history[-2]
            def reset(seed=None):
                original_reset(seed=seed)
                original.seed_history(e,[dict(rgb=past_obs['rgb'],state=past['state_deg_deg_mm'],oct=past['signed_OCT_mm'])],prefix_previous)
            e.reset_buffer=reset
        def action(rgb,state,oct4):
            nonlocal active,sample_calls
            _,pre=capture();norm=float(np.linalg.norm(oct4[:3])) if oct4[3]>.5 else None
            if not active and norm is not None and norm<=.100:
                active=True
                # Only route changes; velocity, accepted action and RGB2 history are continuous.
                assert not e.action_queue
                save(folder/'switch_receipt.json',dict(cycle=len(rows)+1,observable_OCT_norm_mm=norm,state=state,
                     history_length=len(e.obs_buffer),previous_accepted=e.previous_action,actual_velocity=pre['actual_velocity_deg_deg_mm_s'],
                     no_history_or_motor_reset=True))
            source='fixed_noise_near_DiT_RF' if active else 'original_v5_DiT_RF'
            e.model=local if active else far
            if active and not near and not read(OUT/'coverage.json')['switch100um_supported']:
                raise RuntimeError('Near route outside validated training coverage')
            before=e.previous_action.copy();raw=original_action(rgb,state,oct4)
            if len(e.obs_buffer)==2:sample_calls+=1
            else:source='original_RGB2_startup_zero'
            route_counts[source]+=1
            pending.update(pre=pre,source=source,oct_norm_mm=norm,previous_input=before,
                 production_velocity=(e.obs_buffer[-1]['state']-e.obs_buffer[-2]['state']).numpy()/.05 if len(e.obs_buffer)==2 else None,
                 history_state=[v['state'].tolist() for v in e.obs_buffer])
            return raw
        def accepted(value):
            nonlocal previous
            original_accepted(value);previous=np.asarray(value).copy()
        def observer(event):
            _,post=capture()
            assert event['motor_record']['driver']['variant']=='g10'
            assert np.isclose(post['time_s']-pending['pre']['time_s'],.05,atol=1e-12,rtol=0)
            row=dict(cycle=len(rows)+1,target_id='E41' if near else f'E{index:02d}',seed=seed,
                     model_source=pending['source'],pre=pending['pre'],post=post,event=event,
                     production_velocity_deg_deg_mm_s=pending['production_velocity'],previous_action_input=pending['previous_input'],
                     input_history_state=pending['history_state'])
            rows.append(row);append(folder/'cycles.jsonl',row)
        e.get_action=action;e.record_applied_action=accepted;e.step_observer=observer
        ep=e.evaluate_episode(scene,world,max_steps=expected,error_threshold_mm=.020,control_period_s=.05,
             action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,max_consecutive_safety_limited=None,
             early_stop_error_mm=30.,seed=seed,sensor=sensor,stop_on_success=False,progress_every=80)
        metrics=score([r['post']['true_error_um'] for r in rows],expected)
        assert len(rows)==expected and ep['early_stop_reason'] is None,'Incomplete fixed-cycle rollout'
        assert sample_calls==(240 if near else 1599),'Unexpected K1/exec1 sampling count'
        result=dict(status='completed',candidate=candidate,target=name,metrics=metrics,route_counts=route_counts,
              seed=seed,expected_cycles=expected,model_checkpoint_sha256=sha(OUT/candidate/'checkpoint.pth'),
              all_policy_ABC_produced_by_DiT=True,input_ablation=str(OUT/candidate/'input_ablation_after.json'),
              original_startup_zero_cycles=route_counts['original_RGB2_startup_zero'],
              truth_setup_scoring_only=True,simulation_only=True)
        if near:
            baseline=read(BASE/'arms'/kind/'exec1'/'summary.json')['metrics']
            result['baseline_comparison']=dict(path=str(BASE/'arms'/kind/'exec1/summary.json'),before=baseline,
                delta_last200_max_um=metrics['last200_max_post_error_um']-baseline['last200_max_post_error_um'])
        save(folder/'summary.json',result);record('near_test_completed' if near else 'standard_test_completed',result)
    except BaseException as err:
        failure=dict(status='technical_or_protocol_failure',candidate=candidate,target=name,error=repr(err),traceback=traceback.format_exc(),
             saved_cycles=len(rows),partial_metrics=score([r['post']['true_error_um'] for r in rows],expected),scientific_negative=False)
        save(folder/'failure.json',failure);record('closed_loop_failed',failure);raise
    finally:
        checkpoint_after=sha(OUT/candidate/'checkpoint.pth')
        save(folder/'weights_after.json',dict(time=now(),checkpoint_sha256=checkpoint_after,
            checkpoint_bit_exact=checkpoint_after==checkpoint_before,far_v5_sha256=sha(REF),eval_updates=0,saved_cycles=len(rows)))
        if sensor is not None:sensor.close()
        if scene is not None:scene.close()


def expansion_plans():
    return imp('relative_data',OUT/'data_pipeline.py').plans()


def collect(branch):
    helper=imp('paired_collection',PRIOR/'pilot.py');helper.OUT=OUT
    local=imp('relative_collection',OUT/'collection_helper.py')
    # Reuse the original teacher and recorder. Writes are task-local and atomic.
    local.__dict__.update({k:v for k,v in helper.__dict__.items() if not k.startswith('__')})
    local.OUT=OUT;local.record=record;local.save=lambda p,v,update=False:save(p,v)
    spec=next(s for s in expansion_plans() if s['id']==branch)
    assert spec['target'] in ('E40','E42')
    local.campaign_collect(spec)


def build_expanded():
    imp('relative_data',OUT/'data_pipeline.py').build()


def diagnostic():
    imp('relative_data',OUT/'data_pipeline.py').diagnostic()


def group(stage, commands, limit, concurrency):
    start=time.monotonic();deadline=min(START+2696,start+min(limit,895));queue=list(commands);active=[];results=[]
    status(stage,remaining_batch_s=max(0,START+2700-start))
    try:
        while queue or active:
            if time.monotonic()>=deadline:raise TimeoutError(f'{stage}: shared batch or stage time exhausted')
            while queue and len(active)<concurrency:
                name,args=queue.pop(0);log=OUT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
                stream=log.open('x');cmd=[str(ROOT/'.venv/bin/python'),'-E','-B','-u',str(OUT/'campaign.py'),*args]
                remaining=max(.01,deadline-time.monotonic())
                env={**os.environ,**RUNTIME,'DIT_STEP_SECONDS':str(remaining)}
                show_training=args[0]=='train'
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE if show_training else stream,
                    stderr=subprocess.STDOUT,start_new_session=True,text=show_training,bufsize=1 if show_training else -1)
                if show_training:
                    def tee(process=p,output=stream):
                        for line in process.stdout:
                            output.write(line);output.flush();sys.stdout.write(line);sys.stdout.flush()
                    p._log_thread=threading.Thread(target=tee,daemon=True);p._log_thread.start()
                append(OUT/'commands.jsonl',dict(time=now(),stage=stage,command=cmd,pid=p.pid,timeout_s=remaining,log=str(log)))
                active.append((name,p,stream,time.monotonic()))
            for item in active[:]:
                name,p,stream,t=item;rc=p.poll()
                if rc is not None:
                    if hasattr(p,'_log_thread'):p._log_thread.join(timeout=3)
                    stream.close();active.remove(item)
                    entry=dict(name=name,exit_code=rc,elapsed_s=time.monotonic()-t,log=str(OUT/'logs'/f'{name}.log'))
                    results.append(entry)
                    if rc:record('external_step_failed',entry);raise RuntimeError(f'{name} exit {rc}; retained failure is engineering evidence, not science')
            if active:time.sleep(.2)
        return results
    finally:
        for name,p,f,t in active:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,p,f,t in active:
            try:p.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            if hasattr(p,'_log_thread'):p._log_thread.join(timeout=1)
            f.close()
        append(OUT/'budget_ledger.jsonl',dict(time=now(),stage=stage,elapsed_s=time.monotonic()-start,
                batch_used_s=time.monotonic()-START,batch_remaining_s=max(0,START+2700-time.monotonic()),finished=results))


def near_results(candidate):
    results=[read(OUT/candidate/'near'/kind/'summary.json') for kind in KINDS]
    ratios=[r['metrics']['last200_max_post_error_um']/r['baseline_comparison']['before']['last200_max_post_error_um'] for r in results]
    passed=sum(r['metrics']['strict20um_pass'] for r in results)
    improved=passed>0 or (sum(v<=.8 for v in ratios)>=2 and math.prod(ratios)**.25<=.8)
    value=dict(candidate=candidate,completed=4,strict20um_pass=passed,denominator=4,
          ratios_last200_max_vs_saved_original_exec1=ratios,substantial_improvement=improved,
          all_passed=passed==4,scope='E41 development only; excluded from train and all scale estimation')
    save(OUT/candidate/'near_summary.json',value);record('near_four_state_summary',value)
    return value


def run():
    global START
    with (OUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(),'Do not start duplicate campaign'
        # Static validation is charged too; four seconds remain for cleanup/reporting.
        preparation=read(OUT/'preparation_budget.json')['validation_s']
        START=time.monotonic()-preparation
        actual=dict(time=now(),pid=os.getpid(),ppid=os.getppid(),actual_start=True,stage='campaign_orchestrator',
             gpu=0,venv=str(ROOT/'.venv'),budget_s=2700,actual_training_updates=0)
        save('actual_start_receipt.json',actual);status('actual_started',actual_start=actual)
        try:
            fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK);os.write(fd,(json.dumps(actual)+'\n').encode());os.close(fd)
        except OSError:pass
        record('actual_start',actual)
        summary=dict(workflow_status='running',goal_achieved=False,first_research_batch_only=True,actual_start=actual,
            all_control_ABC_DiT=True,teacher_used_for_training_collection_only=True,truth_setup_offline_scoring_only=True)
        try:
            assert sha(REF)==REF_SHA
            changed=[p for p,digest in read(OUT/'input_fingerprints.json').items() if sha(p)!=digest]
            assert not changed, f'Inputs changed after preflight: {changed}'
            record('runtime_input_fingerprints_verified',dict(status='passed',changed=[]))
            try:
                group('old_same_window_diagnostic',[('old_diagnostic',['diagnostic'])],150,1)
            except Exception as diag_error:
                record('old_diagnostic_nonblocking_failure',dict(error=repr(diag_error),used_as_stop_gate=False))
            group('E40_E42_real_teacher_collection',[(f'collect_{s["id"]}',['collect','--branch',s['id']]) for s in expansion_plans()],750,12)
            group('build_relative_windows',[('relative_windows',['build-expanded'])],200,1)
            group('relative_DiT_3000_updates',[('candidate1_train',['train','--candidate','candidate1'])],780,1)
            group('E41_four_50um',[(f'candidate1_{k}',['evaluate','--candidate','candidate1','--kind',k]) for k in KINDS],400,4)
            first=near_results('candidate1');summary['near_E41']=first
            summary['C_bias']=imp('relative_analysis',OUT/'data_pipeline.py').bias_report()
            if first['all_passed']:
                group('standard_five',[(f'standard_E{i:02d}',['evaluate','--candidate','candidate1','--index',str(i)]) for i in (0,2,6,8,10)],890,5)
                standards=[read(OUT/'candidate1'/'standard'/f'E{i:02d}'/'summary.json') for i in (0,2,6,8,10)]
                summary['standard_five']=dict(completed=5,cycles_each=1600,
                    strict20um_pass=sum(r['metrics']['strict20um_pass'] for r in standards),denominator=5,results=standards)
                summary['goal_achieved']=summary['standard_five']['strict20um_pass']==5
                summary['workflow_status']='complete_strict_pass' if summary['goal_achieved'] else 'complete_standard_negative'
            else:
                summary.update(workflow_status='complete_near_negative',standard_five='not_run_E41_gate_failed',
                    failure_trajectories=[str(OUT/'candidate1'/'near'/k/'cycles.jsonl') for k in KINDS],
                    interpretation='Joint data/input feasibility experiment, not single-factor attribution; original0of4 retained and not retrained')
        except BaseException as err:
            summary.update(workflow_status='budget_partial' if isinstance(err,TimeoutError) else 'technical_failure',
                 error=repr(err),traceback=traceback.format_exc(),scientific_negative=False,
                 goal_achieved=False,next_step='Retain saved checkpoints and partial evidence; repair diagnosed engineering issue within remaining shared budget or continue next authorized batch')
            record('batch_exception',dict(status=summary['workflow_status'],error=repr(err),traceback=summary['traceback']))
        finally:
            summary.update(time=now(),runtime_s=time.monotonic()-START,budget_s=2700,simulation_only=True)
            save('summary.json',summary)
            report=f"# DiT relative terminal v2 — joint data/input feasibility\n\nWorkflow: {summary['workflow_status']}\n\nGoal achieved: {summary['goal_achieved']}\n\nBudget used: {summary['runtime_s']:.2f} /2700 seconds.\n\n"
            for c in ('candidate1',):
                p=OUT/c/'near_summary.json'
                if p.exists():report+=f"{c}: `{json.dumps(read(p),ensure_ascii=False)}`\n\n"
            if 'C_bias' in summary:
                report+=f"C bias disappearance (independent postXYZ criterion): {summary['C_bias']['disappearance_of_74um_C_bias']}. Details: C_bias_report.json.\n\n"
            report+="Old0/4 retained read-only; no old-model retraining. Training and scales: E40/E42 only; E41 development only. Joint input/data change, no single-factor attribution.\n\n"
            report+=f"Next: {summary.get('next_step','Review five-target per-cycle evidence.')}\n\nFull details: summary.json; curves/checkpoints and per-cycle XYZ/accepted/velocity/model source remain in candidate directories. Offline action errors are diagnostic. MuJoCo simulation only. Batch completion does not imply completion of the research direction.\n"
            (OUT/'report.md').write_text(report)
            status(summary['workflow_status'],goal_achieved=summary['goal_achieved'],summary_path=str(OUT/'summary.json'))
            record('batch_finished',dict(status=summary['workflow_status'],goal_achieved=summary['goal_achieved'],runtime_s=summary['runtime_s'],summary=str(OUT/'summary.json')))


def launch():
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(OUT/'preflight.json')['status']=='passed'
        assert not (OUT/'gnome_launch_request.json').exists() and not (OUT/'actual_start_receipt.json').exists()
        os.mkfifo(OUT/'startup_ack.fifo',0o600);fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=DiT relative terminal v2 2700s','--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        save('gnome_launch_request.json',dict(time=now(),command=cmd,desktop=DESKTOP,single_launch=True))
        try:
            done=subprocess.run(cmd,env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=15)
            save('gnome_launch_receipt.json',dict(time=now(),returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,command=cmd))
            assert done.returncode==0,done.stderr
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ);ready=selector.select(20)
            message=os.read(fd,65536).decode() if ready else '';selector.close()
            ack=dict(time=now(),actual_start=json.loads(message.splitlines()[0]) if message else None,
                  one_startup_ack_only=True,no_postlaunch_polling=True,timed_out=not bool(message))
            save('launch_startup_acknowledgement.json',ack);record('GNOME_actual_start_acknowledged',ack)
            print('ACTUAL_START_RECEIPT '+json.dumps(ack),flush=True)
            return 0 if message else 1
        except BaseException as err:
            save('launch_failure.json',dict(time=now(),error=repr(err),no_background_fallback=True));raise
        finally:os.close(fd)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','launch','train','evaluate','collect','build-expanded','diagnostic','probe'])
    p.add_argument('--candidate',choices=['candidate1']);p.add_argument('--kind',choices=KINDS)
    p.add_argument('--index',type=int,choices=[0,2,6,8,10]);p.add_argument('--branch');a=p.parse_args()
    if a.mode=='probe':
        torch=torch_setup();print(json.dumps(dict(output_dir=str(OUT),cuda_device=torch.cuda.get_device_name(0),model=__import__('model').__file__,data_output=str(__import__('data_pipeline').OUT))))
    elif a.mode=='run':run()
    elif a.mode=='launch':sys.exit(launch())
    elif a.mode=='train':train(a.candidate)
    elif a.mode=='evaluate':evaluate(a.candidate,a.kind,a.index)
    elif a.mode=='collect':collect(a.branch)
    elif a.mode=='build-expanded':build_expanded()
    elif a.mode=='diagnostic':diagnostic()
if __name__=='__main__':
    import model, data_pipeline
    proof=dict(time=now(),pid=os.getpid(),ppid=os.getppid(),mode=sys.argv[1:],output_dir=str(OUT),
        campaign_file=__file__,model_file=model.__file__,data_pipeline_file=data_pipeline.__file__,
        data_output=str(data_pipeline.OUT),prior=str(PRIOR),base=str(BASE),original=str(ORIGINAL))
    assert Path(model.__file__).resolve().parent==OUT and Path(data_pipeline.__file__).resolve().parent==OUT
    assert data_pipeline.OUT==OUT and OUT.name=='dit_relative_v2_resume'
    save(OUT/'process_receipts'/f'{os.getpid()}.json',proof)
    main()
