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
PRIOR = OUT.parent/'neural_paired_terminal_minifit_v1'
BASE = OUT.parent/'neural_exec124_50um_v1'
ORIGINAL = OUT.parent/'original_neural_terminal_pilot_v1'
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
START = None


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
               command=sys.argv, init_sha256=REF_SHA, output=str(OUT))
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
    torch = torch_setup()
    from model import integrate, install, allow_visual_cache, NOISE_SEED
    folder = OUT/candidate; folder.mkdir(exist_ok=False)
    e = evaluator(); m = e.model
    sampler = imp('paired_sampler', PRIOR/'pilot.py')
    if candidate=='candidate1':
        data = torch.load(PRIOR/'real_input_windows.pt', map_location='cpu', weights_only=False)
        manifest = read(PRIOR/'window_manifest.json')
        assert len(data['train']['action'])==len(data['development_E41']['action'])==657
        # Actual original implementation vs differentiable implementation on both sides/classes.
        checks=[]
        with torch.no_grad():
            for split in ('train','development_E41'):
                for side in ('below','above'):
                    for cls in ('approaching','receding','settled'):
                        i=next(i for i,v in enumerate(manifest[split]) if v['side']==side and v['input_class']==cls)
                        args=tuple(data[split][k][i:i+1].cuda() for k in KEYS)
                        velocity_impl=m._predict_velocity_from_condition
                        traces=[]
                        def traced(*values):
                            output=velocity_impl(*values)
                            traces.append((values[0].detach().clone(),values[1].detach().clone(),output.detach().clone()))
                            return output
                        m._predict_velocity_from_condition=traced
                        try:
                            original=m.sample(*args, steps=4, generator=torch.Generator(device='cuda').manual_seed(NOISE_SEED))
                            different=integrate(m,args)
                        finally:m._predict_velocity_from_condition=velocity_impl
                        assert len(traces)==8
                        per_step=[]
                        for a,b in zip(traces[:4],traces[4:]):
                            for av,bv in zip(a,b):torch.testing.assert_close(av,bv,atol=1e-7,rtol=1e-6)
                            per_step.append(float((a[0]-b[0]).abs().max()))
                        torch.testing.assert_close(original,different,atol=1e-7,rtol=1e-6)
                        checks.append(dict(split=split, window=i, side=side, cls=cls, max_abs=float((original-different).abs().max()),
                                           each_four_intermediate_state_max_abs=per_step))
        save(folder/'original_sampler_parity.json',dict(status='passed',four_steps=checks,FP32=True))
        record('original_four_step_sampler_parity',dict(candidate=candidate,status='passed',checks=len(checks)))
        # Cache the unmodified full visual tokens, not pooled/spatial-softmax features.
        cache={}
        with torch.no_grad():
            for split in ('train','development_E41'):
                blocks=[]
                for start in range(0,len(data[split]['action']),8):
                    blocks.append(m.vision(data[split][KEYS[0]][start:start+8].cuda()).cpu())
                cache[split]=torch.cat(blocks)
        allow_visual_cache(m)
        with torch.no_grad():
            cache_checks=[]
            for check in checks:
                split=check['split'];i=check['window'];b=data[split]
                args=tuple(b[k][i:i+1].cuda() for k in KEYS)
                cached=(cache[split][i:i+1].cuda(),*args[1:])
                direct=integrate(m,args);derived=integrate(m,cached)
                torch.testing.assert_close(direct,derived,atol=1e-6,rtol=1e-5)
                cache_checks.append(dict(split=split,window=i,max_abs=float((direct-derived).abs().max())))
        save(folder/'cache_parity.json',dict(status='passed',checks=cache_checks,full_tokens=[2,49,384],FP32=True))
        record('FP32_full_visual_cache_parity',dict(candidate=candidate,status='passed',checks=len(cache_checks),full_tokens=[2,49,384]))
        for split in cache:
            data[split][KEYS[0]]=cache[split]
        torch.save(data,OUT/'original_visual_cache.pt')
        del cache
    else:
        cp=checkpoint_load(e,OUT/'candidate1/checkpoint.pth')
        data=torch.load(OUT/'expanded_visual_cache.pt',map_location='cpu',weights_only=False)
        manifest=read(OUT/'expanded_window_manifest.json')
        allow_visual_cache(m)
    # Only the current training subset determines physical output mean/std.
    y=data['train']['action'][:,0]
    mean=y.mean(0);raw_std=y.std(0,unbiased=False);std=raw_std.clamp_min(torch.tensor([1e-4,1e-4,1e-5]))
    global_scales={k:v.detach().cpu().clone() for k,v in m.named_buffers() if any(k.startswith(n) for n in ('state_','oct_','velocity_','previous_action_'))}
    install(m,mean,std)
    save(folder/'scales.json',dict(action_mean=mean,action_std=std,raw_std=raw_std,floors=[1e-4,1e-4,1e-5],
          action_statistics_source='current train only; development excluded',global_input_buffers=global_scales,precision='FP32',
          OCT_normalization='original including original local/symlog branches',noise_seed=NOISE_SEED))
    for n,p in m.named_parameters():p.requires_grad_(not n.startswith('vision.'))
    m.train();m.vision.eval()
    before={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
    parameter_rows=[dict(name=n,count=p.numel(),trainable=p.requires_grad) for n,p in m.named_parameters()]
    save(folder/'parameters.json',dict(class_name=type(m).__name__,blocks=len(m.blocks),dim=m.dit_dim,horizon=m.n_action_steps,parameters=parameter_rows))
    active=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(active,lr=3e-5,weight_decay=1e-4)
    pools=sampler.pools_for(manifest['train']);rng=torch.Generator().manual_seed(SEED)
    # Data tensors fit on GPU after RGB has been replaced by frozen full tokens.
    train_data={k:v.cuda() for k,v in data['train'].items()}
    updates=0;gradient_evidence={};started=time.monotonic()
    time_limit=float(os.environ.get('DIT_STEP_SECONDS','850'))-35-(started-phase_started)
    budget_reason=None
    try:
        for step in range(1,3001):
            if time.monotonic()-started>=time_limit:
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
            if step==1 or step%25==0:
                row=dict(time=now(),update=updates,loss=float(loss.detach()),first_action_mse=float(first.detach()),rf_mse=float(rf.detach()),
                         grad_norm=float(gradient),elapsed_s=time.monotonic()-started)
                append(folder/'training_curve.jsonl',row);print(json.dumps(row),flush=True)
            if step%500==0:
                save(folder/'progress.json',dict(actual_updates=updates,planned_updates=3000,elapsed_s=time.monotonic()-started))
                # Save weights before any statistics/reporting, even on later failure.
                checkpoint_save(folder,m,opt,updates,mean,std)
    finally:
        checkpoint_save(folder,m,opt,updates,mean,std)
    del opt,train_data
    delta={n:float((p.detach().cpu()-before[n]).norm()) for n,p in m.named_parameters()}
    assert all(delta[n]==0 for n in delta if n.startswith('vision.'))
    for k,v in global_scales.items():assert torch.equal(dict(m.named_buffers())[k].cpu(),v)
    assert all(any(delta[n]>0 for n in delta if n.startswith(f'blocks.{i}.self_attention.')) for i in range(6))
    assert all(any(v>0 for probe in gradient_evidence.values() for n,v in probe.items()
                   if n.startswith(f'blocks.{i}.self_attention.')) for i in range(6))
    evidence=dict(actual_updates=updates,planned_updates=3000,stage_budget_reason=budget_reason,minimum_prescribed_updates=2000,
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
    save(folder/'summary.json',dict(status='trained',actual_updates=updates,minimum_updates_reached=updates>=2000,diagnostics=diagnostics))
    if updates<2000:raise TimeoutError('Actual updates below2000; budget partial, no unlearnability conclusion')


def checkpoint_save(folder,m,opt,updates,mean,std):
    import torch
    cp=dict(model={n:v.detach().cpu() for n,v in m.state_dict().items()},action_mean=mean.cpu(),action_std=std.cpu(),
            actual_updates=updates,init_sha256=REF_SHA,policy='fixed_noise_DiT_RF',noise_seed=434341,
            optimizer=opt.state_dict(),time=now())
    tmp=folder/'checkpoint.tmp';torch.save(cp,tmp);tmp.replace(folder/'checkpoint.pth')


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
    e=evaluator();far=e.model
    if not near:e.model=evaluator().model
    checkpoint_load(e,OUT/candidate/'checkpoint.pth');local=e.model
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
                 previous_accepted_deg_deg_mm=previous.copy(),needle_site_world_mm=tip,target_world_mm=world,
                 true_error_um=float(np.linalg.norm(tip-world)*1000))
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
              all_policy_ABC_produced_by_DiT=True,original_startup_zero_cycles=route_counts['original_RGB2_startup_zero'],
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
        if sensor is not None:sensor.close()
        if scene is not None:scene.close()


def expansion_plans():
    specs=[]
    for target in (40,41,42):
        for side,sign in (('below',-1),('above',1)):
            for j,level in enumerate((20,70,140)):
                for motion in ('approaching','receding'):
                    direction=-sign if motion=='approaching' else sign
                    lateral=(1 if motion=='approaching' else -1)*.01*math.pi/180
                    specs.append(dict(id=f'E{target}_{side}_{level}_{motion}_AB',target=f'E{target}',role='training',side=side,
                        level_um=level,requested_incoming=motion,initial_C_offset_m=sign*(level+(4.4 if motion=='approaching' else -4.4))*1e-6,
                        initial_AB_offset_rad=[lateral,(-1 if j%2 else 1)*lateral],
                        prefix_actions_deg_deg_mm=[[0.,0.,direction*.003]]*2,teacher_steps=80,designated_anchor_teacher_step=0))
    return specs


def collect(branch):
    # Reuse the registered homogeneous true-RGB2 teacher recorder, with only setup AB offsets added.
    helper=imp('paired_collection',PRIOR/'pilot.py');helper.OUT=OUT
    helper.record=record
    # The locally saved helper is a transparent surgical copy of collect(), not a new teacher.
    local=imp('campaign_collection',OUT/'collection_helper.py')
    local.__dict__.update({k:v for k,v in helper.__dict__.items() if not k.startswith('__')})
    local.collect=local.campaign_collect
    local.collect(next(s for s in expansion_plans() if s['id']==branch))


def build_expanded():
    torch=torch_setup();import numpy as np
    from PIL import Image
    from model import allow_visual_cache
    e=evaluator();checkpoint_load(e,OUT/'candidate1/checkpoint.pth');m=e.model;m.eval()
    sampler=imp('paired_input_class',PRIOR/'pilot.py')
    train={k:[] for k in (*KEYS,'action')};metas=[];parity=[];coverage=[]
    for spec in expansion_plans():
        folder=OUT/'data'/spec['id'];obs=[json.loads(v) for v in (folder/'observations.jsonl').read_text().splitlines()]
        actions=[json.loads(v) for v in (folder/'actions.jsonl').read_text().splitlines()]
        selected=list(range(2,len(actions)-15));indices=sorted({j for i in selected for j in (i-1,i)})
        features={};image_tensors={}
        for start in range(0,len(indices),8):
            js=indices[start:start+8];images=[]
            for j in js:
                with Image.open(OUT/obs[j]['rgb_path']) as im:images.append(e.preprocess_image(np.asarray(im.convert('RGB')).copy()))
            # Encode individual frames as one-step vision sequences; GroupNorm has no batch stats.
            imgs=torch.stack(images)[:,None].cuda()
            with torch.no_grad():f=m.vision(imgs).cpu()[:,0]
            for k,j in enumerate(js):features[j]=f[k];image_tensors[j]=images[k]
        for j in selected:
            state=torch.tensor([obs[j-1]['state_deg_deg_mm'],obs[j]['state_deg_deg_mm']],dtype=torch.float32)
            oct4=torch.tensor([obs[j-1]['signed_OCT_mm'],obs[j]['signed_OCT_mm']],dtype=torch.float32)
            velocity=(state[-1]-state[-2])/.05;previous=torch.tensor(obs[j]['previous_accepted_deg_deg_mm'],dtype=torch.float32)
            assert actions[j]['pre_observation']==j and actions[j]['teacher_step']==j-2
            y=torch.tensor([r['accepted_deg_deg_mm'] for r in actions[j:j+16]],dtype=torch.float32)
            cached=torch.stack([features[j-1],features[j]])
            values=[cached,state,oct4,velocity,previous,y]
            for k,v in zip((*KEYS,'action'),values):train[k].append(v)
            cls=sampler.input_class(obs[j],obs[j-1])['input_class']
            if cls=='ambiguous':cls='approaching' if np.linalg.norm(oct4[-1,:3])<np.linalg.norm(oct4[-2,:3]) else 'receding'
            metas.append(dict(window_index=len(metas),episode=spec['id'],target=spec['target'],role='training',
                split='train',input_class=cls,side=spec['side'],teacher_step=j-2,pre_observation=j,
                label='same-tick accepted ABC',image_paths=[obs[j-1]['rgb_path'],obs[j]['rgb_path']]))
            coverage.append(dict(target=spec['target'],role='training',OCT_valid=bool(oct4[-1,3]>.5),
                observable_distance_um=float(oct4[-1,:3].norm()*1000),true_error_um=obs[j]['true_error_um'],
                actual_velocity_si=obs[j]['actual_velocity_si']))
        # Per-episode FP32 RGB2 online vs individually cached frames through original complete RF.
        j=selected[0]
        args=(torch.stack([image_tensors[j-1],image_tensors[j]])[None].cuda(),
              *[train[k][-len(selected)][None].cuda() for k in KEYS[1:]])
        with torch.no_grad():
            direct=m.sample(*args,steps=4)
            original_forward=allow_visual_cache(m)
            cached_args=(train[KEYS[0]][-len(selected)][None].cuda(),*args[1:])
            derived=m.sample(*cached_args,steps=4)
            torch.testing.assert_close(direct,derived,atol=1e-6,rtol=1e-5)
            m.vision.forward=original_forward
        parity.append(dict(episode=spec['id'],max_abs=float((direct-derived).abs().max())))
    train={k:torch.stack(v) for k,v in train.items()}
    # E41 is explicitly promoted from selected development to training by this campaign authorization.
    # It cannot remain a development holdout after collection; no0..19 development targets are collected.
    old=torch.load(OUT/'original_visual_cache.pt',map_location='cpu',weights_only=False)
    om=read(PRIOR/'window_manifest.json')
    for split in ('train','development_E41'):
        offset=len(metas)
        for v in om[split]:metas.append(dict(v,role='training',split='train',window_index=len(metas),prior_role=split))
        for k in train:train[k]=torch.cat([train[k],old[split][k]])
    torch.save(dict(train=train),OUT/'expanded_visual_cache.pt')
    save('expanded_window_manifest.json',dict(train=metas,E41_prior_development_now_training=True,no_independent_development_split=True))
    valid=[r['observable_distance_um'] for r in coverage if r['OCT_valid']]
    support={f'E{i}':sum(r['target']==f'E{i}' and r['OCT_valid'] and 90<=r['observable_distance_um']<=150 for r in coverage) for i in (40,41,42)}
    result=dict(status='passed',windows=len(metas),new_windows=len(coverage),targets=['E40','E41','E42'],role='training',
        E41_reclassified_from_previously_selected_development=True,no_standard_or_0_to19_development_target_collection=True,
        distance_observed_um=[min(valid),max(valid)],per_target_90to150um_windows=support,
        switch100um_supported=all(n>=4 for n in support.values()) and min(valid)<20 and max(valid)>=100,
        online_cache_parity=parity,real_RGB2=True,same_tick_accepted_labels=True)
    save('coverage.json',result);record('expanded_training_data_and_cache_completed',result)
    assert result['switch100um_supported'],'100um route needs observed training coverage'


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
          all_passed=passed==4,scope='E41 selected development' if candidate=='candidate1' else 'E41 now training: seen-target closed-loop probe')
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
            group('candidate1_train_and_parity',[('candidate1_train',['train','--candidate','candidate1'])],860,1)
            group('candidate1_near',[(f'candidate1_{k}',['evaluate','--candidate','candidate1','--kind',k]) for k in KINDS],480,4)
            first=near_results('candidate1');summary['candidate1']=first;selected='candidate1'
            if first['substantial_improvement']:
                summary['expansion_branch']='selected'
                group('homogeneous_RGB2_collection',[(f'collect_{s["id"]}',['collect','--branch',s['id']]) for s in expansion_plans()],600,12)
                group('expanded_cache',[('expanded_cache',['build-expanded'])],300,1)
                group('candidate2_train',[('candidate2_train',['train','--candidate','candidate2'])],800,1)
                group('candidate2_near',[(f'candidate2_{k}',['evaluate','--candidate','candidate2','--kind',k]) for k in KINDS],480,4)
                second=near_results('candidate2');summary['candidate2']=second;selected='candidate2'
                if second['all_passed']:
                    group('standard_five',[(f'standard_E{i:02d}',['evaluate','--candidate',selected,'--index',str(i)]) for i in (0,2,6,8,10)],900,5)
                    standards=[read(OUT/selected/'standard'/f'E{i:02d}'/'summary.json') for i in (0,2,6,8,10)]
                    summary['standard_five']=dict(completed=5,cycles_each=1600,strict20um_pass=sum(r['metrics']['strict20um_pass'] for r in standards),
                         denominator=5,results=standards)
                    summary['goal_achieved']=summary['standard_five']['strict20um_pass']==5
                    summary['workflow_status']='batch_complete_strict_pass' if summary['goal_achieved'] else 'batch_complete_standard_negative'
                else:
                    summary.update(workflow_status='batch_complete_near_negative',
                        hypothesis_rejected='Expanded E40-E42 coverage plus the fixed-noise first-action objective did not meet all four near-state hold tests; this is not evidence that DiT cannot learn the task',
                        next_step='Analyze saved per-axis drift and class-conditioned action bias to select the minimum next DiT conditioning or data change; no MLP substitution')
            else:
                summary.update(workflow_status='batch_complete_near_negative',expansion_branch='not_selected',
                    hypothesis_rejected='3000-update first-action fixed-noise local-output-scale objective did not materially improve the four original near starts',
                    next_step='Use saved DiT gradients, action curves and postXYZ to distinguish insufficient observable conditioning from state-distribution coverage; next batch may test near OCT conditioning with original DiT backbone')
        except BaseException as err:
            summary.update(workflow_status='budget_partial' if isinstance(err,TimeoutError) else 'technical_failure',
                 error=repr(err),traceback=traceback.format_exc(),scientific_negative=False,
                 goal_achieved=False,next_step='Retain saved checkpoints and partial evidence; repair diagnosed engineering issue within remaining shared budget or continue next authorized batch')
            record('batch_exception',dict(status=summary['workflow_status'],error=repr(err),traceback=summary['traceback']))
        finally:
            summary.update(time=now(),runtime_s=time.monotonic()-START,budget_s=2700,simulation_only=True)
            save('summary.json',summary)
            report=f"# DiT terminal precision campaign — first batch\n\nWorkflow: {summary['workflow_status']}\n\nGoal achieved: {summary['goal_achieved']}\n\nBudget used: {summary['runtime_s']:.2f} /2700 seconds.\n\n"
            for c in ('candidate1','candidate2'):
                p=OUT/c/'near_summary.json'
                if p.exists():report+=f"{c}: `{json.dumps(read(p),ensure_ascii=False)}`\n\n"
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
        cmd=['gnome-terminal','--window','--title=DiT terminal precision campaign 2700s','--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
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
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','launch','train','evaluate','collect','build-expanded'])
    p.add_argument('--candidate',choices=['candidate1','candidate2']);p.add_argument('--kind',choices=KINDS)
    p.add_argument('--index',type=int,choices=[0,2,6,8,10]);p.add_argument('--branch');a=p.parse_args()
    if a.mode=='run':run()
    elif a.mode=='launch':sys.exit(launch())
    elif a.mode=='train':train(a.candidate)
    elif a.mode=='evaluate':evaluate(a.candidate,a.kind,a.index)
    elif a.mode=='collect':collect(a.branch)
    elif a.mode=='build-expanded':build_expanded()
if __name__=='__main__':main()
