"""Authorized original-v5 exec1/2/4 comparison; production modules read-only.

Wrappers observe, verify and log. Production PolicyEvaluator owns sampling,
queue popping, clipping, safety preview/commit and accepted-action feedback.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import selectors
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/'runs/20261126/neural_paired_terminal_minifit_v1'
ORIGINAL = ROOT/'runs/20261126/original_neural_terminal_pilot_v1'
SOURCE = ROOT/'data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2'
REF = Path(json.loads((PRIOR/'experiment_card.json').read_text())['init_checkpoint'])
REF_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
SEED = 434341
KINDS = ('below_approaching', 'below_receding', 'above_approaching', 'above_receding')
CAPS = [.2897999882698059, .14434358477592468, .04271206632256508]
DESKTOP = dict(DISPLAY=':1', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus', XDG_RUNTIME_DIR='/run/user/1000')
RUNTIME = dict(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID', MUJOCO_GL='egl',
    MUJOCO_EGL_DEVICE_ID='0', RCM_DRIVER_VARIANT='g10', OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
    TORCH_DISABLE_NATIVE_JIT='1', MPLBACKEND='Agg')
sys.path.insert(0, str(ROOT))


class ProtocolError(RuntimeError):
    pass


def now():
    return datetime.now().astimezone().isoformat()


def default(v):
    if hasattr(v, 'tolist'):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v).__name__)


def plain(v):
    return json.loads(json.dumps(v, default=default, allow_nan=False))


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v, update=False):
    p = Path(p)
    if not p.is_absolute():
        p = OUT/p
    assert p.is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False, default=default)+'\n'
    if update:
        tmp = p.with_suffix(p.suffix+'.tmp')
        with tmp.open('w') as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        tmp.replace(p)
    else:
        with p.open('x') as f:
            f.write(value); f.flush(); os.fsync(f.fileno())


def append(p, row):
    with Path(p).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=default)+'\n')
        f.flush()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, default=default, allow_nan=False).encode()).hexdigest()


def record(stage, result):
    row = dict(time=now(), experiment=OUT.name, stage=stage, result=result,
        command=sys.argv, card=str(OUT/'experiment_card.json'), checkpoint_sha256=REF_SHA,
        target='E41 development previously used for selection', seed=SEED, output=str(OUT))
    line = ('\n- '+json.dumps(row, ensure_ascii=False, default=default, allow_nan=False)+'\n').encode()
    receipts = []
    for p in (OUT/'TEST_LOG.md', ROOT/'TEST_LOG.md'):
        with p.open('a+b') as f:
            fcntl.flock(f, fcntl.LOCK_EX); f.seek(0, 2); offset = f.tell()
            f.write(line); f.flush(); os.fsync(f.fileno()); f.seek(offset)
            assert f.read(len(line)) == line
        receipts.append(dict(path=str(p), offset=offset, bytes=len(line), exact_readback=True))
    with (OUT/'test_log_receipts.jsonl').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(dict(time=row['time'], stage=stage, receipts=receipts))+'\n')
        f.flush(); os.fsync(f.fileno())
    print(json.dumps(row, ensure_ascii=False, default=default), flush=True)


def status(stage, state='running', **extra):
    row = dict(time=now(), status=state, stage=stage, pid=os.getpid(), training_updates=0,
        planned_arms=12, completion_means_execution_only=True, **extra)
    save('task_status.json', row, update=True)
    append(OUT/'status_history.jsonl', row)


def import_file(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def specs():
    prior = import_file('readonly_paired_prefix_plan', PRIOR/'pilot.py')
    selected = [p for p in prior.plans() if p['target']=='E41' and p['level_um']==50]
    assert len(selected)==4
    return {p['side']+'_'+p['requested_incoming']:p for p in selected}


def identity_files():
    paths = [REF, PRIOR/'experiment_card.json', PRIOR/'paired_anchor_evidence.json', PRIOR/'pilot.py',
        ORIGINAL/'pilot.py', ORIGINAL/'experiment_card.json', ROOT.parent/'AGENTS.md',
        ROOT/'docs/EXPERIMENT_SOP.md', SOURCE/'shards/contract.json',
        SOURCE/'shards/tasks/collection_E41/E41.json', OUT/'pilot.py', OUT/'launch.sh']
    for spec in specs().values():
        paths += [PRIOR/'data'/spec['id']/name for name in ('observations.jsonl','actions.jsonl')]
    for tree in ('rcm','lerobot_training','mujoco_model'):
        paths += list((ROOT/tree).rglob('*.py'))
    for pattern in ('*.json','*.xml','mujoco_model/**/*.xml','mujoco_model/**/*.json'):
        paths += list(ROOT.glob(pattern))
    return sorted(set(p for p in paths if p.is_file()), key=str)


def verify_identity():
    expected = read(OUT/'input_hashes.json')
    differences = [p for p, value in expected.items() if not Path(p).exists() or sha(p)!=value]
    return dict(status='passed' if not differences else 'failed', checked_files=len(expected),
        changed_or_missing=differences, original_checkpoint_sha256=sha(REF))


def metrics(rows):
    import numpy as np
    errors = np.array([r['post']['true_error_um'] for r in rows],float)
    count=longest=0
    for x in errors<=20:
        count=count+1 if x else 0; longest=max(longest,count)
    raw=np.array([r['event']['raw_action_deg_deg_mm'] for r in rows],float).reshape(-1,3)
    accepted=np.array([r['event']['accepted_action_deg_deg_mm'] for r in rows],float).reshape(-1,3)
    vel=np.array([r['post']['actual_velocity_deg_deg_mm_s'] for r in rows],float).reshape(-1,3)
    csign=np.array([r['post']['axial_error_um_from_initial_axis'] for r in rows],float)
    before=rows[0]['pre']['axial_error_um_from_initial_axis'] if rows else 0.
    signs=np.sign(np.r_[before,csign]); nonzero=signs[signs!=0]
    incoming=[r['pre']['time_s'] for r in rows if r['sampled_this_cycle']]
    def axes(a):
        return dict(max_abs=np.max(np.abs(a),axis=0).tolist() if len(a) else None,
            rms=np.sqrt(np.mean(a*a,axis=0)).tolist() if len(a) else None,
            units=['deg','deg','mm'])
    return dict(completed_cycles=len(rows), endpoint_um=float(errors[-1]) if len(errors) else None,
        last200_max_post_error_um=float(errors[-200:].max()) if len(errors)>=200 else None,
        last200_includes_final_post=True, strict20um_pass=bool(len(rows)==240 and np.all(errors[-200:]<=20)),
        first_enter20_post_cycle=int(np.flatnonzero(errors<=20)[0]+1) if np.any(errors<=20) else None,
        longest_le20_cycles=longest, terminal_le20_cycles=count,
        error_quantiles_um={str(q):float(np.quantile(errors,q)) for q in (0,.25,.5,.75,.9,.95,.99,1)} if len(errors) else {},
        axial_target_plane_crossings=int(np.sum(nonzero[1:]*nonzero[:-1]<0)),
        overshoot_definition='world target plane normal=initial needle C direction; crossing sign changes ignoring exact zeros',
        max_overshoot_um=float(max(0.,-csign.min())) if len(csign) else None,
        beyond100um_post_cycles=int(np.sum(errors>100)),
        beyond100um_is_diagnostic_not_stop=True,
        accepted_action_jump=axes(np.diff(accepted,axis=0)), raw_action_jump=axes(np.diff(raw,axis=0)),
        actual_velocity=dict(**axes(vel),units_per_second=['deg/s','deg/s','mm/s']),
        sample_calls=len(incoming), feedback_observation_hz=20.,
        policy_feedback_resample_hz=len(incoming)/(len(rows)*.05) if rows else None,
        resample_interval_s=np.diff(incoming).tolist(),
        safety_limited_cycles=sum(r['event']['safety_limited'] for r in rows))


def prepare():
    started=time.monotonic()
    assert not (OUT/'gnome_launch_request.json').exists() and not (OUT/'actual_start_receipt.json').exists()
    status('preflight_assets_syntax_queue', 'preparing', completed_arms=0)
    ast.parse(Path(__file__).read_text())
    syntax=subprocess.run(['bash','-n',str(OUT/'launch.sh')],capture_output=True,text=True,timeout=5)
    assert syntax.returncode==0, syntax.stderr
    save('syntax_check.json', dict(status='passed',python='ast.parse',shell='bash -n',model_calls=0,simulation_cycles=0))
    record('syntax_check_passed',read(OUT/'syntax_check.json'))
    assert sha(REF)==REF_SHA
    old=read(PRIOR/'experiment_card.json')
    task=next(x['task'] for x in old['targets'] if x['task']['id']=='E41')
    registry=read(SOURCE/'shards/contract.json')['groups']['collection']['tasks'][41]
    assert task==registry==read(SOURCE/'shards/tasks/collection_E41/E41.json')['task']
    s=specs()
    # Structural checks of the actual production queue, no model or scene run.
    source=(ROOT/'lerobot_training/eval_policy.py').read_text()
    tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PolicyEvaluator')
    methods={x.name:ast.get_source_segment(source,x) for x in cls.body if isinstance(x,ast.FunctionDef)}
    get=methods['get_action']
    assert get.index('self.obs_buffer.append') < get.index('if self.action_queue:')
    assert 'self.action_queue.pop(0)' in get and '1:self.n_action_execute_steps' in get
    assert 'steps=self.flow_integration_steps' in get and 'generator=self.action_generator' in get
    assert 'self.record_applied_action(applied_action)' in methods['evaluate_episode']
    assert 'n_action_execute_steps' in methods['__init__']
    queue={str(n):[i for _ in range(240//n) for i in range(n)] for n in (1,2,4)}
    assert all(len(v)==240 for v in queue.values())
    # Pure metrics check independently exercises final-post inclusion and dwell.
    fake=[]
    for i in range(240):
        post=dict(true_error_um=30. if i==239 else 10., actual_velocity_deg_deg_mm_s=[0.,0.,0.],
            axial_error_um_from_initial_axis=10.)
        fake.append(dict(post=post,pre=dict(time_s=i*.05,axial_error_um_from_initial_axis=10.),
            sampled_this_cycle=i%4==0,event=dict(raw_action_deg_deg_mm=[0.]*3,
            accepted_action_deg_deg_mm=[0.]*3,safety_limited=False)))
    m=metrics(fake)
    assert m['last200_max_post_error_um']==30. and not m['strict20um_pass']
    assert m['terminal_le20_cycles']==0 and m['longest_le20_cycles']==239 and m['sample_calls']==60
    save('static_queue_metric_checks.json',dict(status='passed',queue_indices=queue,
        production_queue_owns_execution=True,model_calls=0,simulation_cycles=0,
        final_post_and_dwell_fixture=m))
    record('queue_seed_metric_static_checks_passed',dict(queue='production append-before-pop and original parameter support verified',
        sampling_seed=SEED, expected_sample_calls={'exec1':240,'exec2':120,'exec4':60},synthetic_final_post_test='passed',
        simulation_smoke_cycles=0,model_calls=0))
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader'],
        capture_output=True,text=True,timeout=8)
    assert gpu.returncode==0 and '4090' in gpu.stdout
    display=subprocess.run(['xdpyinfo','-display',':1'],env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=8)
    assert display.returncode==0,display.stderr
    assert shutil.which('gnome-terminal') and Path('/run/user/1000/bus').exists()
    assert shutil.disk_usage(OUT).free>5*1024**3
    card=dict(name=OUT.name,time=now(),question='Original-v5 16-action prediction, execute1/2/4 from50um real moving starts',
        init_checkpoint=str(REF),init_sha256=REF_SHA,target=task,target_role='E41 DEVELOPMENT previously used for selection',
        prefix_source=str(PRIOR/'pilot.py'),prefix_evidence=str(PRIOR/'paired_anchor_evidence.json'),starts=s,
        original_entry='lerobot_training.eval_policy.PolicyEvaluator.evaluate_episode/get_action',
        variable='n_action_execute_steps only:1,2,4',unchanged=dict(K=1,flow_integration_steps=4,n_obs_steps=2,
            n_action_steps=16,preprocess='original224 production transform',caps_deg_deg_mm=CAPS,
            control_period_s=.05,driver='g10',controller_variant='feedback',seed=SEED),
        total_arms=12,cycles_per_arm=240,neural_duration_s=12.,max_concurrent_workers=4,
        scheduling='one worker per start; exec1/2/4 sequential; one unchanged model instance per worker; real prefix replay each arm',
        parity_gate='exact canonical scene/driver/history/RNG state and first complete16-action chunk; abort dependent arms on mismatch',
        start_gate='independent needle-site3D distance50±1um, RGB2 and actual incoming XYZ/OCT motion direction',
        stop_on_success=False,early_stop_error_mm=30.,max_consecutive_safety_limited=None,
        limits='production clipping and safety scales retained; invalidation changes expected schedule => protocol failure and dependent comparison stops',
        strict_gate='all final200 POST errors<=20um including final; not endpoint-only',
        out_of_range_definition='post3D error>100um diagnostic plus native safety rejection/early stop, no added stop',
        budget_s=900,budget_includes='preflight execution, model loading, prefix setup, failures and all scoring; writing/reading excluded',
        prior_preparation_runtime_charge_s=1.,training_updates=0,new_weights=False,
        no_teacher_after_prefix=True,no_smoke_simulation=True,no_dev5_dev20=True,no_upload=True,
        concurrency_reason='4 independent start workers,4 original models on physical0 RTX4090;~49GiB GPU current occupancy saved in preflight',
        prior_parity_evidence=[str(ORIGINAL/'short_gate/first_input_parity.json'),str(ORIGINAL/'step0_parity.json')],
        environment={**DESKTOP,**RUNTIME},known_limitations='one selected target and one seed; simulation only; no generalization or hardware claims')
    save('experiment_card.json',card)
    hashes={str(p):sha(p) for p in identity_files()}
    save('input_hashes.json',hashes)
    preflight=dict(status='passed',time=now(),gpu=gpu.stdout.strip(),display=':1 available',
        checkpoint_sha256=hashes[str(REF)],hashed_files=len(hashes),production_files_modified=0,
        duplicate_launch_detected=False,concurrency=4,actual_preflight_model_calls=0,actual_preflight_simulation_cycles=0,
        start_motion_reference=[x for x in read(PRIOR/'paired_anchor_evidence.json') if x['target']=='E41' and x['level_um']==50],
        runtime_s=time.monotonic()-started+1.,command=[str(ROOT/'.venv/bin/python'),'-E','-B',str(OUT/'pilot.py'),'run'])
    save('preflight.json',preflight)
    save('preparation_budget.json',dict(elapsed_s=preflight['runtime_s'],writing_reading_excluded=True))
    record('asset_target_desktop_preflight_passed',preflight)
    status('ready_for_single_gnome_launch','prepared',completed_arms=0,strict20um_result='not_evaluated')
    (OUT/'report.md').write_text('# 原v5 exec1/2/4：50µm真实运动初态对照\n\n准备与静态检查完成，尚未启动。12臂尚未执行，20µm尚未评定。\n')


def rng_snapshot(e):
    import numpy as np
    import torch
    return dict(policy_generator=e.action_generator.get_state().cpu().tolist(),
        torch_cpu=torch.get_rng_state().tolist(),torch_cuda=[x.cpu().tolist() for x in torch.cuda.get_rng_state_all()],
        numpy=plain(np.random.get_state()),python=plain(random.getstate()))


def arm(e, kind, execute, spec):
    import numpy as np
    import torch
    from PIL import Image
    from rcm.scene_v9 import RCMSceneV9
    from lerobot_training.eval_policy import StrictRGBDSensor,visual_servo_settings,TASK_TO_DEG_DEG_MM,DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    from scripts.research.homog_replay_common import QUALITY
    original=import_file('readonly_original_history',ORIGINAL/'pilot.py')
    folder=OUT/'arms'/kind/f'exec{execute}'
    folder.mkdir(parents=True,exist_ok=False); (folder/'rgb').mkdir()
    scene=sensor=None; observations=[]; rows=[]; chunks=[]; previous=np.zeros(3)
    pending={}; cache={}; base=OUT/'arms'/kind/'exec1'
    originals=(e.reset_buffer,e.get_action,e.model.sample,e.record_applied_action,e.invalidate_action_chunk)
    start_wall=time.monotonic()
    try:
        save(folder/'task_status.json',dict(status='running',stage='fresh_prefix_replay',time=now(),pid=os.getpid()))
        # Reset RNG before each fresh scene. Do not reset motor velocity/history in flight.
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        scene=RCMSceneV9(); task=read(OUT/'experiment_card.json')['target']
        world=np.asarray(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
        np.testing.assert_allclose(world,task['target_world_mm'],atol=1e-9,rtol=0)
        scene.set_target_world(world); q=scene.target_task(validate=True).copy(); q[2]+=spec['initial_C_offset_m']
        scene.set_task(q);scene.enable_motor_execution();scene.reset_motor_time(0.)
        settings=visual_servo_settings(scene);settings.update(perception_backend='strict_rgbd',rgbd_render_quality=QUALITY.copy())
        sensor=StrictRGBDSensor(scene,settings); actual_observe=sensor.observe
        unit=np.array([np.pi/180,np.pi/180,.001])
        # Fixed scoring axis only; pure C prefix leaves its orientation intact.
        initial_delta=scene.site_mm('needle_tip')-world
        initial_axis=initial_delta/np.linalg.norm(initial_delta)

        def capture():
            t=float(scene.motor_execution.time_s)
            if cache.get('time_s')==t:
                return cache['obs'],cache['row']
            sensor.observation_time_s=t; obs=actual_observe()
            rgb=np.asarray(obs['rgb'],np.uint8); oct4=np.zeros(4,np.float32)
            navigation=obs.get('oct_navigation_depth'); delta=None if navigation is None else navigation.get('delta_xyz_m')
            if delta is not None and len(delta)==3 and np.isfinite(delta).all():
                oct4[:3]=np.asarray(delta,np.float64)*1000;oct4[3]=1.
            tip=scene.site_mm('needle_tip').copy(); image=folder/'rgb'/f'{len(observations):04d}.png'
            Image.fromarray(rgb).save(image,compress_level=1)
            row=plain(dict(index=len(observations),time_s=t,rgb_path=str(image.relative_to(OUT)),rgb_file_sha256=sha(image),
                rgb_array_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),rgb_shape=list(rgb.shape),
                state_deg_deg_mm=np.asarray(scene.current_task())*TASK_TO_DEG_DEG_MM,signed_OCT_mm=oct4,
                actual_velocity_deg_deg_mm_s=np.asarray(scene.motor_execution.actual_velocity)*TASK_TO_DEG_DEG_MM,
                previous_accepted_deg_deg_mm=previous.copy(),encoder_ABCE=list(scene.motor_execution.backend.robot),
                needle_site_world_mm=tip,target_world_mm=world,true_error_um=float(np.linalg.norm(tip-world)*1000),
                axial_error_um_from_initial_axis=float(np.dot(tip-world,initial_axis)*1000) if initial_axis is not None else None,
                OCT_metadata=obs.get('oct_metadata'),RGB_depth_source=obs.get('oct_depth_source')))
            observations.append(row);append(folder/'observations.jsonl',row)
            cache.update(time_s=t,obs=obs,row=row)
            return obs,row

        sensor.observe=lambda:capture()[0]
        capture()
        for raw in spec['prefix_actions_deg_deg_mm']:
            _,pre=capture(); action=np.array(raw,float)
            scene.apply_prepared_motor(scene.motor_execution.preview(scene.current_task()+action*unit,.05))
            previous=action.copy();_,post=capture()
            append(folder/'prefix_cycles.jsonl',dict(pre=pre,post=post,raw_action_deg_deg_mm=raw,
                accepted_action_deg_deg_mm=action,motor_record=scene.motor_execution.last_record))
        # Axis used only for offline overshoot scoring. No truth reaches policy.
        prefix_previous=previous.copy(); past=observations[-2]; current=observations[-1]
        radial_xyz=(current['true_error_um']-past['true_error_um'])/.05
        radial_oct=(np.linalg.norm(current['signed_OCT_mm'][:3])-np.linalg.norm(past['signed_OCT_mm'][:3]))*1000/.05
        actual_side='below' if scene.current_task()[2]<scene.target_task(validate=True)[2] else 'above'
        gate=dict(status='passed',distance_um=current['true_error_um'],velocity_deg_deg_mm_s=current['actual_velocity_deg_deg_mm_s'],
            incoming_XYZ_radial_um_s=radial_xyz,incoming_OCT_radial_um_s=float(radial_oct),previous_accepted=prefix_previous.tolist(),
            history_times_s=[past['time_s'],current['time_s']],registered=spec,actual_side=actual_side,
            truth_used_only_for_setup_and_scoring=True)
        passed=(49<=current['true_error_um']<=51 and abs(current['actual_velocity_deg_deg_mm_s'][2])>1e-6
            and actual_side==spec['side']
            and current['signed_OCT_mm'][3]==past['signed_OCT_mm'][3]==1
            and (radial_xyz<0 and radial_oct<0 if spec['requested_incoming']=='approaching' else radial_xyz>0 and radial_oct>0))
        gate['status']='passed' if passed else 'failed';save(folder/'start_gate.json',gate)
        record('start_gate_checked',dict(arm=str(folder.relative_to(OUT)),**gate))
        if not passed:raise ProtocolError('50±1um / real incoming motion / RGB2 start gate failed')
        e.n_action_execute_steps=execute
        e.reset_buffer=originals[0]
        e.reset_buffer(seed=SEED)
        # Seed only true t-1; production appends t on the first get_action.
        past_rgb=np.asarray(Image.open(OUT/past['rgb_path']),np.uint8)
        history=[dict(rgb=past_rgb,state=np.array(past['state_deg_deg_mm']),oct=np.array(past['signed_OCT_mm']))]
        def reset(seed=None):
            originals[0](seed=seed);original.seed_history(e,history,prefix_previous)
        e.reset_buffer=reset;reset(seed=SEED)
        motor=scene.motor_execution
        def physical_history_row(row):
            value={k:v for k,v in row.items() if k not in ('rgb_path','rgb_file_sha256')}
            # Provider UUID and wall acquisition time are provenance, never model
            # inputs or simulated state; retain originals in observations.jsonl.
            metadata=value.get('OCT_metadata')
            if metadata is not None:
                value['OCT_metadata']={k:v for k,v in metadata.items()
                    if k not in ('frame_id','stream_id','acquired_monotonic_s')}
            return value
        scene_state=dict(xml_sha256=hashlib.sha256(scene.xml.encode()).hexdigest(),
            scene_qpos=scene.data.qpos.copy(),scene_qvel=scene.data.qvel.copy(),scene_time=float(scene.data.time),
            camera_positions=scene.model.cam_pos.copy(),camera_quaternions=scene.model.cam_quat.copy(),
            target_world_mm=world,scene_config=scene.config,
            motor_backend={k:v for k,v in motor.backend.__dict__.items() if k!='kin'},
            motor_driver=motor.driver_identity,motor_time_s=motor.time_s,motor_actual_velocity=motor.actual_velocity,
            motor_revision=motor.revision,motor_epoch=motor.epoch,motor_records=motor.records,
            sensor_sequence=sensor.sequence,sensor_settings=settings,
            history=[physical_history_row(r) for r in observations],
            previous_accepted=prefix_previous,obs_buffer=[{k:v.cpu().tolist() for k,v in x.items()} for x in e.obs_buffer],
            rng=rng_snapshot(e))
        scene_state=plain(scene_state);save(folder/'initial_state.json',scene_state)
        initial_identity=digest(scene_state)
        initial_parity=dict(status='reference' if execute==1 else 'passed',canonical_sha256=initial_identity,
            exact_all_fields=True,comparison_reference=str(base/'initial_state.json'),
            excluded_provenance_only=['artifact paths','OCT provider UUID/frame_id','wall acquisition monotonic time'],
            full_original_metadata_preserved_in='observations.jsonl')
        if execute!=1:
            expected=read(base/'initial_state.json')
            initial_parity['exact_all_fields']=scene_state==expected
            if scene_state!=expected:
                initial_parity['status']='failed'
                initial_parity['different_top_level_keys']=[k for k in expected if scene_state.get(k)!=expected[k]]
        save(folder/'init_parity.json',initial_parity);record('init_parity_checked',dict(arm=str(folder.relative_to(OUT)),**initial_parity))
        if not initial_parity['exact_all_fields']:raise ProtocolError('Exact initial scene/driver/history/RNG parity failed')
        pending.update(cycle=0,chunk=None,index=None,invalidations=[])

        def sample(*args,**kwargs):
            if kwargs.get('steps')!=4:raise ProtocolError('Flow integration steps changed')
            if args[0].shape!=(1,2,3,224,224):raise ProtocolError('Production RGB2/K1/224 shape mismatch')
            before=rng_snapshot(e)['policy_generator']
            try:
                output=originals[2](*args,**kwargs)
            except torch.cuda.OutOfMemoryError as err:
                # Prevent production's OOM fallback from silently consuming another draw.
                raise RuntimeError('OOM: no resampling/restart authorized') from err
            array=output.detach().cpu().numpy()
            if array.shape!=(1,16,3) or not np.isfinite(array).all():raise ProtocolError('Expected finite1x16x3 original chunk')
            chunk=plain(dict(chunk_id=len(chunks),sample_cycle=pending['cycle'],sample_time_s=float(motor.time_s),
                sampling_seed=SEED,K=1,flow_integration_steps=4,actions_deg_deg_mm=array[0],
                generator_before_sha256=digest(before),generator_after_sha256=digest(rng_snapshot(e)['policy_generator']),
                input_sha256=[hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for x in args],
                input_shapes=[list(x.shape) for x in args]))
            if not chunks:
                first=dict(actions_deg_deg_mm=chunk['actions_deg_deg_mm'],input_sha256=chunk['input_sha256'],
                    generator_before_sha256=chunk['generator_before_sha256'])
                save(folder/'first_sample.json',first)
                torch.save({str(i):x.detach().cpu() for i,x in enumerate(args)},folder/'first_real_production_input.pt')
                same=True if execute==1 else first==read(base/'first_sample.json')
                save(folder/'first_sample_parity.json',dict(status='reference' if execute==1 else 'passed' if same else 'failed',
                    full16_and_first_action_exact=same,no_extra_inference=True))
                record('first_sample_parity_checked',dict(arm=str(folder.relative_to(OUT)),full16_and_first_action_exact=same,extra_inference=0))
                if not same:raise ProtocolError('First16/first-action production input/RNG parity mismatch')
            chunks.append(chunk);append(folder/'action_chunks.jsonl',chunk)
            pending.update(chunk=chunk,index=0)
            return output

        def action(rgb,state,oct4):
            _,pre=capture(); before_count=len(chunks); queue_before=len(e.action_queue)
            prev=e.previous_action.copy()
            result=originals[1](rgb,state,oct4)
            sampled=len(chunks)>before_count
            if sampled: idx=0
            else:
                if pending['index'] is None:raise ProtocolError('Queue used without a recorded sample')
                idx=pending['index']+1
            pending['index']=idx
            if not 0<=idx<execute:raise ProtocolError('Chunk execution index outside registered exec horizon')
            np.testing.assert_array_equal(np.asarray(result),np.asarray(pending['chunk']['actions_deg_deg_mm'][idx]))
            assert len(e.obs_buffer)==2
            np.testing.assert_array_equal(e.obs_buffer[-1]['state'].cpu().numpy(),np.asarray(state,np.float32))
            np.testing.assert_array_equal(prev,np.asarray(pre['previous_accepted_deg_deg_mm'],np.float32))
            history_velocity=(e.obs_buffer[-1]['state']-e.obs_buffer[-2]['state']).cpu().numpy()/.05
            pending.update(pre=pre,sampled=sampled,queue_before=queue_before,queue_after=len(e.action_queue),
                production_float32_velocity_deg_deg_mm_s=history_velocity.tolist(),previous_action_input=prev.tolist(),
                input_history_state=[x['state'].cpu().tolist() for x in e.obs_buffer])
            return result

        def accepted(value):
            nonlocal previous
            originals[3](value);previous=np.asarray(value).copy()

        def invalidate():
            pending['invalidations'].append(dict(cycle=pending['cycle'],chunk_id=pending['chunk']['chunk_id'] if pending['chunk'] else None,
                remaining_queue=len(e.action_queue)))
            originals[4]()

        def observer(event):
            _,post=capture(); row=plain(dict(cycle=len(rows)+1,pre=pending['pre'],post=post,event=event,
                sampled_this_cycle=pending['sampled'],sample_calls_so_far=len(chunks),chunk_id=pending['chunk']['chunk_id'],
                chunk_sample_time_s=pending['chunk']['sample_time_s'],chunk_index=pending['index'],
                queue_before=pending['queue_before'],queue_after_get_action=pending['queue_after'],
                production_float32_velocity_deg_deg_mm_s=pending['production_float32_velocity_deg_deg_mm_s'],
                previous_action_input=pending['previous_action_input'],input_history_state=pending['input_history_state']))
            rows.append(row);append(folder/'cycles.jsonl',row)
            record_time=event['motor_record']
            if not np.isclose(post['time_s']-pending['pre']['time_s'],.05,rtol=0,atol=1e-12):raise ProtocolError('Not one50ms actual tick')
            if record_time['driver']['variant']!='g10' or record_time['duration_s']!=.05:raise ProtocolError('Driver/timing mismatch')
            if event['safety_limited']:
                raise ProtocolError('Production safety invalidated chunk; dependent exec comparison stopped; executed tick retained')
            if row['chunk_index']!=(len(rows)-1)%execute:raise ProtocolError('Actual queue index schedule mismatch')
            pending['cycle']=len(rows)
            if len(rows)==1:
                save(folder/'actual_first_action_receipt.json',dict(time=now(),status='first_neural_tick_executed',row=row))
                parity_keys=('raw_action_deg_deg_mm','clipped_action_deg_deg_mm','accepted_action_deg_deg_mm','motor_record')
                reference=row if execute==1 else read(base/'actual_first_action_receipt.json')['row']
                actual_same=all(row['event'][k]==reference['event'][k] for k in parity_keys)
                actual_same=actual_same and row['post']['needle_site_world_mm']==reference['post']['needle_site_world_mm']
                save(folder/'actual_first_action_parity.json',dict(status='reference' if execute==1 else 'passed' if actual_same else 'failed',
                    raw_clipped_accepted_driver_postXYZ_exact=actual_same))
                record('actual_first_action_parity_checked',dict(arm=str(folder.relative_to(OUT)),exact=actual_same))
                if not actual_same:raise ProtocolError('First actual accepted action/motor/postXYZ mismatch')
            if len(rows)%40==0:
                save(folder/'task_status.json',dict(time=now(),status='running',stage='neural_cycles',cycles=len(rows),sample_calls=len(chunks)),True)

        e.get_action=action;e.model.sample=sample;e.record_applied_action=accepted;e.invalidate_action_chunk=invalidate;e.step_observer=observer
        ep=e.evaluate_episode(scene,world,max_steps=240,error_threshold_mm=.020,control_period_s=.05,
            action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,max_consecutive_safety_limited=None,
            early_stop_error_mm=30.,seed=SEED,sensor=sensor,stop_on_success=False,progress_every=40)
        # Native results keep all true motor records including2 actual prefix ticks.
        save(folder/'production_evaluation.json',ep)
        validation=dict(status='passed',cycles=len(rows),actual_model_sample_calls=len(chunks),expected_model_sample_calls=240//execute,
            actual_queue_indices=[r['chunk_index'] for r in rows],expected_queue_indices=[i%execute for i in range(240)],
            first_sample_exact=True,initial_state_exact=True,all_neural_actions=True,teacher_action_calls_after_prefix=0,
            invalidations=pending['invalidations'],early_stop_reason=ep['early_stop_reason'])
        if len(rows)!=240 or len(chunks)!=240//execute or ep['early_stop_reason'] is not None:
            validation['status']='failed'
        save(folder/'protocol_validation.json',validation)
        if validation['status']!='passed':raise ProtocolError('Fixed240/sample-count/early-stop protocol failed')
        result=dict(status='completed',kind=kind,exec_steps=execute,initial=gate,metrics=metrics(rows),protocol=validation,
            elapsed_s=time.monotonic()-start_wall,model_sha256=REF_SHA,wall_time=now())
        save(folder/'summary.json',result);save(folder/'task_status.json',dict(status='completed',stage='arm_finished',time=now(),cycles=len(rows),sample_calls=len(chunks)),True)
        record('arm_completed',result)
        return result
    except BaseException as err:
        failure=dict(time=now(),status='protocol_failure' if isinstance(err,ProtocolError) else 'technical_failure',
            error=repr(err),traceback=traceback.format_exc(),kind=kind,exec_steps=execute,
            completed_cycles=len(rows),sample_calls=len(chunks),partial_metrics=metrics(rows) if rows else None,
            no_retry=True,dependent_exec_arms_stopped=True)
        save(folder/'failure.json',failure)
        save(folder/'task_status.json',failure,True)
        record('arm_failed',failure)
        raise
    finally:
        e.reset_buffer,e.get_action,e.model.sample,e.record_applied_action,e.invalidate_action_chunk=originals
        if hasattr(e,'step_observer'):del e.step_observer
        if sensor is not None:sensor.close()
        if scene is not None:scene.close()


def worker(kind):
    import numpy as np
    import torch
    from lerobot_training.eval_policy import PolicyEvaluator,DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    torch.set_num_threads(1)
    started=time.monotonic();done=[]
    try:
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
        np.testing.assert_array_equal(DEFAULT_ACTION_LIMIT_DEG_DEG_MM,np.array(CAPS))
        e=PolicyEvaluator(str(REF),device='cuda',num_action_samples=1,n_action_execute_steps=1,
            flow_integration_steps=4,controller_variant='feedback')
        assert e.is_dit_rectified_flow and e.n_obs_steps==2 and e.n_action_steps==16
        assert not e.model.training and not any(x.training for x in e.model.modules())
        assert e.fine_evaluator is None and not e.guided_oct and not e.is_regression
        spec=read(OUT/'experiment_card.json')['starts'][kind]
        record('worker_original_loader_and_frozen_settings_checked',dict(kind=kind,pid=os.getpid(),
            model_sha256=sha(REF),model_eval=True,all_modules_eval=True,K=e.num_action_samples,
            flow=e.flow_integration_steps,preprocessing='original transform',direct_224_preprocessing=e.direct_224_preprocessing,
            n_obs=e.n_obs_steps,n_action=e.n_action_steps,training_updates=0))
        for execute in (1,2,4):
            done.append(arm(e,kind,execute,spec))
        save(OUT/'groups'/f'{kind}.json',dict(status='completed',kind=kind,completed_arms=len(done),
            elapsed_s=time.monotonic()-started,arms=done))
        return 0
    except BaseException as err:
        failure=dict(status='protocol_failure' if isinstance(err,ProtocolError) else 'technical_failure',
            kind=kind,time=now(),error=repr(err),traceback=traceback.format_exc(),completed_arms=len(done),
            no_retry=True,not_completed_exec_arms=[x for x in (1,2,4) if x not in [r['exec_steps'] for r in done]])
        save(OUT/'groups'/f'{kind}.json',failure);record('start_group_failed',failure)
        return 1


def finish(workflow_status, **extra):
    arms=[];failures=[];groups=[]
    for kind in KINDS:
        p=OUT/'groups'/f'{kind}.json'
        if p.exists():groups.append(read(p))
        for n in (1,2,4):
            p=OUT/'arms'/kind/f'exec{n}'
            if (p/'summary.json').exists():arms.append(read(p/'summary.json'))
            if (p/'failure.json').exists():failures.append(read(p/'failure.json'))
    identity=verify_identity()
    save('final_input_hash_verification.json',identity)
    if identity['status']!='passed':workflow_status='protocol_failure'
    validation=dict(time=now(),status='passed' if len(arms)==12 and not failures and all(g['status']=='completed' for g in groups)
        and identity['status']=='passed' else 'failed_or_incomplete',groups=groups,completed_arms=len(arms),planned_arms=12,
        original_inputs_unchanged=identity['status']=='passed',source_hash_verification=str(OUT/'final_input_hash_verification.json'))
    save('protocol_validation.json',validation)
    result=dict(time=now(),workflow_status=workflow_status,execution_completed=len(arms)==12,
        completed_arms=len(arms),planned_arms=12,protocol_status=validation['status'],
        strict20um_pass_arms=sum(a['metrics']['strict20um_pass'] for a in arms),
        strict20um_denominator=12,training_updates=0,model='original v5 A4000',model_sha256=REF_SHA,
        target='E41 development previously used for selection',seed=SEED,arms=arms,failures=failures,groups=groups,
        limitations='one selected target and one seed, no generalization; MuJoCo only',**extra)
    save('summary.json',result)
    lines=['# 原v5预测16、执行1/2/4：50µm局部闭环对照','',
        f"流程状态：{workflow_status}；完整轨迹 {len(arms)}/12；协议状态 {validation['status']}。",
        f"最后200个POST误差全部≤20µm：{result['strict20um_pass_arms']}/12（未完成不计成功）。训练更新0次。",'',
        'E41为已经参与选择的开发目标；仅一个采样种子434341。该结果不代表泛化，也不代表硬件或临床精度。',
        '前缀用于设置初态，之后所有ABC动作均由原始神经网络给出。保留生产限幅、安全预览、量化、惯性和动作队列。',
        '每周期真实观测/速度/上一受理指令均更新；20µm按POST评分并包含最后一拍，不以终点代替持续判据。','',
        '| 起点 | exec | 初始µm | 初速ABC(deg/s,deg/s,mm/s) | 周期 | sample次数 | 末200maxµm | 终点µm | 首入≤20周期 | 最长/终末≤20 | 严格通过 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|']
    for a in arms:
        m=a['metrics'];g=a['initial']
        lines.append(f"| {a['kind']} | {a['exec_steps']} | {g['distance_um']:.9f} | {g['velocity_deg_deg_mm_s']} | {m['completed_cycles']} | {m['sample_calls']} | {m['last200_max_post_error_um']:.9f} | {m['endpoint_um']:.9f} | {m['first_enter20_post_cycle']} | {m['longest_le20_cycles']}/{m['terminal_le20_cycles']} | {m['strict20um_pass']} |")
    lines += ['', '技术/协议错误原文、未完成组、预算和产物身份见summary.json及各臂failure.json；缺失轨迹不填充。',
        '误差分位数、逐轴实际速度/动作跳变、轴向过冲和>100µm计数、实际反馈频率见各臂summary.json。',
        '完整逐拍XYZ/target、原始/受理动作、电机时间与chunk索引：arms/<起点>/execN/cycles.jsonl。',
        '完整16动作与真实采样输入指纹：action_chunks.jsonl；初态精确校验：initial_state.json/init_parity.json。',
        'RGB2及每拍真实图像在各臂rgb/；observations.jsonl含末拍真实post，不复制前帧。',
        '原始脚本/权重/旧数据hash见input_hashes.json及final_input_hash_verification.json。',
        '逐状态日志已在任务和部署根TEST_LOG追加，字节级读回凭据见test_log_receipts.jsonl。','',
        '未添加训练、教师接管、ridge、几何控制、距离切换或新模型。真实前缀每臂重放，未在运动中清零状态。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    record('workflow_finished',dict(status=workflow_status,completed_arms=len(arms),protocol=validation['status'],
        strict20um_pass_arms=result['strict20um_pass_arms'],summary=str(OUT/'summary.json'),input_hashes=identity))
    status('finished',workflow_status,completed_arms=len(arms),protocol_status=validation['status'],
        strict20um_pass_arms=result['strict20um_pass_arms'])


def run():
    children=[];logs=[];started=time.monotonic()
    with (OUT/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(),'Refuse repeated execution'
        receipt=dict(time=now(),status='actual_process_started',pid=os.getpid(),ppid=os.getppid(),argv=sys.argv,
            stage='runtime_identity_then_four_start_workers',desktop={k:os.environ.get(k) for k in DESKTOP},
            GNOME_TERMINAL_SCREEN=os.environ.get('GNOME_TERMINAL_SCREEN'),not_scientific_success=True)
        save('actual_start_receipt.json',receipt)
        try:
            fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK)
            os.write(fd,(json.dumps(receipt)+'\n').encode());os.close(fd)
        except OSError:pass
        try:
            assert receipt['GNOME_TERMINAL_SCREEN'],'Not a real GNOME terminal'
            for k,v in {**DESKTOP,**RUNTIME}.items():assert os.environ.get(k)==v,(k,os.environ.get(k))
            status('runtime_identity_verification',completed_arms=0)
            identity=verify_identity();save('runtime_input_hash_verification.json',identity)
            if identity['status']!='passed':raise ProtocolError('Asset drift before execution')
            record('actual_execution_started',receipt)
            prior=read(OUT/'preparation_budget.json')['elapsed_s']+read(OUT/'prelaunch_budget.json')['elapsed_s']
            remaining=900-prior-(time.monotonic()-started)
            assert remaining>0,'Budget exhausted before workers'
            deadline=time.monotonic()+remaining
            save('runtime_budget.json',dict(total_s=900,prior_compute_s=prior,remaining_at_workers_s=remaining,
                starts_monotonic=started,deadline_monotonic=deadline))
            def interrupted(signum,frame):raise InterruptedError(f'signal {signum}')
            signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
            status('closed_loop_4_start_groups',completed_arms=0,workers=4)
            for kind in KINDS:
                log=(OUT/f'worker_{kind}.log').open('x');logs.append(log)
                cmd=[str(ROOT/'.venv/bin/python'),'-E','-B','-u',str(OUT/'pilot.py'),'worker','--kind',kind]
                p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**RUNTIME},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append((kind,p))
                append(OUT/'worker_launches.jsonl',dict(time=now(),kind=kind,pid=p.pid,command=cmd))
            # This wait is internal experiment management, never agent-side polling.
            for kind,p in children:
                try:p.wait(timeout=max(.001,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:raise TimeoutError('900s total budget exhausted')
                completed=sum((OUT/'arms'/k/f'exec{n}'/'summary.json').exists() for k in KINDS for n in (1,2,4))
                status('closed_loop_4_start_groups',completed_arms=completed,group_returncode={k:q.returncode for k,q in children})
            failures=[dict(kind=k,returncode=p.returncode) for k,p in children if p.returncode]
            finish('completed' if not failures else 'partial_failure',worker_failures=failures,
                runtime_s=time.monotonic()-started,total_budget_used_s=prior+time.monotonic()-started)
            return 1 if failures else 0
        except BaseException as err:
            for _,p in children:
                if p.poll() is None:
                    try:os.killpg(p.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for _,p in children:
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),no_retry=True,child_processes_stopped=True)
            save('failure.json',failure);record('workflow_exception',failure)
            finish('budget_exceeded' if isinstance(err,TimeoutError) else 'protocol_failure' if isinstance(err,ProtocolError) else 'technical_failure',
                error=repr(err),runtime_s=time.monotonic()-started)
            return 1
        finally:
            for f in logs:f.close()


def launch():
    started=time.monotonic()
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'gnome_launch_request.json').exists() and not (OUT/'actual_start_receipt.json').exists(),'Refuse duplicate launch'
        assert read(OUT/'preflight.json')['status']=='passed'
        assert verify_identity()['status']=='passed'
        os.mkfifo(OUT/'startup_ack.fifo',0o600)
        fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=Original v5 E41 50um exec1 exec2 exec4 seed434341',
            '--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        save('gnome_launch_request.json',dict(time=now(),command=cmd,desktop=DESKTOP))
        save('prelaunch_budget.json',dict(elapsed_s=time.monotonic()-started+1.,
            launch_ipc_reserve_s=1.,startup_wait_overlaps_runtime=True))
        try:
            result=subprocess.run(cmd,env={**os.environ,**DESKTOP},capture_output=True,text=True,timeout=15)
            receipt=dict(time=now(),command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            save('gnome_launch_receipt.json',receipt)
            if result.returncode:
                raise RuntimeError(f'GNOME launch failed: {result.stderr}')
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ)
            ready=selector.select(20);message=os.read(fd,65536).decode() if ready else '';selector.close()
            ack=dict(time=now(),one_startup_ack_only=True,no_phase_polling=True,
                actual_start=json.loads(message.splitlines()[0]) if message else None,timed_out=not bool(message))
            save('launch_startup_acknowledgement.json',ack);record('gnome_actual_start_acknowledgement',ack)
            print('ACTUAL_START_RECEIPT',json.dumps(ack,ensure_ascii=False),flush=True)
            if not message:
                record('startup_ack_missing',dict(no_retry=True,no_background_fallback=True,actual_phase_unknown=True))
            return 0 if message else 1
        except BaseException as err:
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),no_retry=True,no_background_fallback=True)
            save('launch_failure.json',failure);record('gnome_launch_failed',failure)
            # A launched child might be running; never clobber its task status.
            if not (OUT/'actual_start_receipt.json').exists():
                status('launch_failed','technical_failure',completed_arms=0,error=repr(err))
                save('summary.json',dict(workflow_status='technical_failure',stage='gnome_launch',
                    completed_arms=0,planned_arms=12,strict20um_result='not_evaluated',**failure))
                (OUT/'report.md').write_text('# 启动失败\n\n'+json.dumps(failure,ensure_ascii=False,indent=2)+'\n无后台替代或重试。\n')
            return 1
        finally:os.close(fd)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','launch','run','worker'))
    parser.add_argument('--kind',choices=KINDS);a=parser.parse_args()
    if a.mode=='prepare':
        try:prepare();return 0
        except BaseException as err:
            failure=dict(time=now(),error=repr(err),traceback=traceback.format_exc(),stage='preparation',no_simulation_or_inference_started=True)
            save(OUT/f'preparation_failure_{time.time_ns()}.json',failure)
            status('preparation_failed','technical_failure',error=repr(err));record('preparation_failed',failure)
            raise
    if a.mode=='launch':return launch()
    if a.mode=='worker':return worker(a.kind)
    return run()


if __name__=='__main__':
    sys.exit(main())
