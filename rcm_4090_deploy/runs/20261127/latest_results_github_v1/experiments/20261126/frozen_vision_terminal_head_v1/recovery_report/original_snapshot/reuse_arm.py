"""Task-local adaptation of verified exec1 loop; production control is unchanged. Generated during implementation, never at runtime."""

from __future__ import annotations
import argparse,ast,fcntl,hashlib,importlib.util,json,os,random,sys,time,traceback
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/him1/mujoco_bc/rcm_4090_deploy")
OUT=Path(__file__).resolve().parent
BASE=ROOT/"runs/20261126/neural_exec124_50um_v1"
PRIOR=ROOT/"runs/20261126/neural_paired_terminal_minifit_v1"
ORIGINAL=ROOT/"runs/20261126/original_neural_terminal_pilot_v1"
REF=Path(json.loads((OUT/"experiment_card.json").read_text())["init_checkpoint"])
REF_SHA="6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11"
SEED=434341

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
        expected_state=read(BASE/'arms'/kind/'exec1'/'initial_state.json')
        differences=[key for key in expected_state if scene_state.get(key)!=expected_state[key]]
        # RNG is not a model input for this deterministic head; physical state/history/driver are exact.
        physical_differences=[key for key in differences if key!='rng']
        parity=dict(status='passed' if not physical_differences else 'failed',different_keys=physical_differences,
            excluded_nonphysical_keys=['rng'],baseline=str(BASE/'arms'/kind/'exec1'/'initial_state.json'))
        save(folder/'saved_exec1_initial_parity.json',parity);record('saved_exec1_initial_physical_history_parity',dict(kind=kind,**parity))
        if physical_differences:raise ProtocolError('Saved exec1 physical initialization differs')
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
            # Compatibility kwargs are ignored by deterministic TerminalHead.sample; no flow or RNG.
            if args[0].shape!=(1,2,3,224,224):raise ProtocolError('Production RGB2/K1/224 shape mismatch')
            before=rng_snapshot(e)['policy_generator']
            try:
                output=originals[2](*args,**kwargs)
            except torch.cuda.OutOfMemoryError as err:
                # Prevent production's OOM fallback from silently consuming another draw.
                raise RuntimeError('OOM: no resampling/restart authorized') from err
            array=output.detach().cpu().numpy()
            if array.shape!=(1,1,3) or not np.isfinite(array).all():raise ProtocolError('Expected finite1x1x3 deterministic first action')
            chunk=plain(dict(chunk_id=len(chunks),sample_cycle=pending['cycle'],sample_time_s=float(motor.time_s),
                sampling_seed=SEED,K=1,flow_integration_steps=0,deterministic=True,action_horizon=1,actions_deg_deg_mm=array[0],
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
                    first_deterministic_action_exact=same,no_extra_inference=True))
                record('first_sample_parity_checked',dict(arm=str(folder.relative_to(OUT)),first_deterministic_action_exact=same,extra_inference=0))
                if not same:raise ProtocolError('First-action input parity mismatch')
            chunks.append(chunk);append(folder/'deterministic_actions.jsonl',chunk)
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
            elapsed_s=time.monotonic()-start_wall,model_sha256=sha(OUT/'candidate.pth'),original_visual_sha256=REF_SHA,wall_time=now())
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
