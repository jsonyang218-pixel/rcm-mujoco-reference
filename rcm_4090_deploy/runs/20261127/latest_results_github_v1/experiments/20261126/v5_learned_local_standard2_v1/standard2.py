"""Two authorized standard-reset live rollouts; archives are read only after control."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
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
LOCAL = ROOT/'runs/20261125/learned_terminal_recovery_pilot_v1/recovery_v1/run'
ARCHIVE = ROOT/'runs/20261125/terminal_round_v1/v5_A4000'
CHECKPOINT = ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
MODEL = LOCAL/'model.json'
SWITCH_MM = 0.03813982154943184
MODEL_SHA = '8d0679fc706c8a1a384977c59af5ec131e22e281afa9d02bd2f6625baccfbfc1'
V5_SHA = '6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11'
DESKTOP = dict(DISPLAY=':1', XDG_RUNTIME_DIR='/run/user/1000', DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL))


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def default(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    raise TypeError(type(x).__name__)


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=default)+'\n')
    temp.replace(path)


def record(stage, value):
    entry = (f'\n- {now()}: v5_learned_local_standard2_v1 {stage}; '
             f'config={OUT}/experiment_card.json; result={json.dumps(value,ensure_ascii=False,default=default)}; '
             f'output={OUT}; MuJoCo simulation only.\n')
    for path in (ROOT/'TEST_LOG.md', ROOT.parent/'TEST_LOG.md'):
        with path.open('ab') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(entry.encode())
    print(entry.strip(), flush=True)


def acknowledge(value):
    # This pipe is used only for bounded startup acknowledgement, never run polling.
    try:
        fd = os.open(OUT/'startup_ack.fifo', os.O_WRONLY | os.O_NONBLOCK)
        os.write(fd, (json.dumps(value,default=default)+'\n').encode())
        os.close(fd)
    except OSError:
        pass


def radius_metrics(post_um, radius):
    import numpy as np
    inside = np.asarray(post_um) <= radius
    entries = np.flatnonzero(inside & ~np.r_[False,inside[:-1]])+1
    exits = np.flatnonzero(~inside & np.r_[False,inside[:-1]])+1
    run = longest = 0
    for value in inside:
        run = run+1 if value else 0
        longest = max(longest,run)
    complete = len(post_um)==1600
    return dict(radius_um=radius, first_enter_post_action_cycle=int(entries[0]) if len(entries) else None,
                entry_cycles=entries.tolist(), first_exit_post_action_cycle=int(exits[0]) if len(exits) else None,
                exit_cycles=exits.tolist(), longest_hold_cycles=longest, longest_hold_s=longest*.05,
                terminal_hold_cycles=run, terminal_hold_s=run*.05,
                last200_max_um=float(max(post_um[-200:])) if len(post_um)>=200 else None,
                last200_cycles_inclusive=[1401,1600] if complete else None,
                last200_duration_s=10.0 if complete else None,
                passed=bool(complete and max(post_um[-200:])<=radius))


def preflight():
    import numpy as np
    import torch
    from scripts.research.homog_replay_common import check_identity
    assert not list(OUT.glob('E*/actual_start_receipt.json')), 'Existing execution; refuse duplicate'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid(): continue
        try:
            argv=[v.decode(errors='replace') for v in (process/'cmdline').read_bytes().split(b'\0') if v]
            assert not (str(Path(__file__)) in argv and any(v in argv for v in ('worker','supervise','launch'))), f'Already active: {process.name}'
        except (OSError,ProcessLookupError):
            pass
    for name in ('workflow_status.json','supervisor_start_receipt.json','gnome_launch_receipt.json'):
        assert not (OUT/name).exists(), f'Existing launch state: {name}'
    assert sha(CHECKPOINT)==V5_SHA
    assert sha(MODEL)==MODEL_SHA
    assert sha(LOCAL/'student_policy.py')=='7909a2932b682d38c688c1846177ceb424a5c436d822d56e4e16e7e8461f71a5'
    gate = read(LOCAL/'local_metrics.json')
    assert len(gate)==6
    assert all(r['cycles']==240 and r['last200_max_true_error_um']<=20 and r['early_stop_reason'] is None for r in gate)
    assert sum(r['controller']=='student' for r in gate)==3
    with np.load(LOCAL/'training_arrays.npz') as a:
        assert float(np.linalg.norm(a['X'][:,:3],axis=1).max())==SWITCH_MM
    spec=read(ARCHIVE/'shards/contract.json')['groups']['eval']['spec']
    check_identity(spec['identity'])
    assert spec['checkpoint_sha256']==V5_SHA and spec['samples']==1 and spec['controller']=='feedback'
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    baselines={}
    for i in (6,8):
        row=read(ARCHIVE/f'E{i:02d}.json')
        assert row['seed']==42+i and row['actions']==1600 and row['checkpoint']==str(CHECKPOINT)
        assert sha(row['result_path'])==row['result_sha256']
        baselines[f'E{i:02d}']=row
    env=os.environ.copy()
    env.update(DESKTOP)
    probe=subprocess.run(['xdpyinfo','-display',':1'],env=env,capture_output=True,text=True,timeout=8)
    assert probe.returncode==0,probe.stderr
    files=[Path(__file__),CHECKPOINT,MODEL,LOCAL/'student_policy.py',LOCAL/'training_arrays.npz',LOCAL/'local_metrics.json',
           ROOT/'lerobot_training/eval_policy.py',ROOT/'rcm/scene_v9.py',ROOT/'rcm/motor_execution.py',
           ROOT/'configs/v12_reachable_development_targets_0p35um.json',ARCHIVE/'shards/contract.json']
    card=dict(name='v5_learned_local_standard2_v1',created=now(),targets=[dict(episode_index=6,seed=48),dict(episode_index=8,seed=50)],
              start='Fresh RCMSceneV9 standard reset exactly as PolicyEvaluator.evaluate; no start_error_mm or set_task',
              steps=1600,control_period_s=.05,K=1,execute_steps=1,flow_integration_steps=4,driver='g10',controller='feedback',
              action_caps_deg_deg_mm=[.2,.1,.03],stop_on_success=False,max_consecutive_safety_limited=None,early_stop_error_mm=30.,
              v5_checkpoint=str(CHECKPOINT),v5_sha256=V5_SHA,local_model=str(MODEL),local_sha256=MODEL_SHA,
              switch_first_valid_measured_oct_norm_le_mm=SWITCH_MM,switch_once=True,
              continuity='No scene, velocity, previous accepted command, clock or RNG reset at switch',
              primary='Every post-action live needle_tip world XYZ versus target world XYZ; cycles1401..1600 maximum<=200um',
              secondary='Identical post-action scoring at20um; original strict0.35um+3settled frames also reported',
              max_workers=2,additional_targets_authorized=False,training=False,publication=False,
              archive_access='Only parent supervisor reads archived actions/states after workers terminate; no worker archive reads',
              baseline=baselines,source_protocol=spec,desktop_environment=DESKTOP)
    save(OUT/'experiment_card.json',card)
    save(OUT/'preflight.json',dict(status='passed',time=now(),gpu=torch.cuda.get_device_name(0),
         local_gate_rows=gate,frozen_files={str(p):sha(p) for p in files},source_identity=spec['identity'],
         historical_baseline='Complete provenance matched originals, scoring and pre-switch comparison only'))
    record('preflight passed',dict(targets=['E06/48','E08/50'],primary_radius_um=200,switch_mm=SWITCH_MM))


def worker(index):
    folder=OUT/f'E{index:02d}'
    folder.mkdir(exist_ok=False)
    pid=os.getpid()
    save(folder/'process_receipt.json',dict(pid=pid,ppid=os.getppid(),time=now(),argv=sys.argv,phase='loading'))
    scene=sensor=trace=None
    rows=[]
    try:
        import numpy as np
        import torch
        from student_policy import LearnedPolicy, measured_features
        from lerobot_training.eval_policy import PolicyEvaluator, StrictRGBDSensor, visual_servo_settings, TASK_TO_DEG_DEG_MM
        from rcm.scene_v9 import RCMSceneV9
        from scripts.research.clean_oct_workers import QUALITY,egl_identity
        assert sha(MODEL)==MODEL_SHA and sha(CHECKPOINT)==V5_SHA
        assert torch.cuda.is_available()
        class Hybrid(PolicyEvaluator):
            def reset_buffer(self,seed=None):
                super().reset_buffer(seed=seed)
                self.local_active=False
                self.local_previous=np.zeros(3,dtype=np.float64)
                self.switch_info=None
                self.step_index=0
                self.route_counts=dict(v5=0,learned_local=0,teacher=0,geometry=0)

            def get_action(self,rgb,state,oct_feature):
                valid=bool(oct_feature[3]>.5)
                norm=float(np.linalg.norm(np.asarray(oct_feature[:3],dtype=np.float64))) if valid else None
                velocity=scene.motor_execution.actual_velocity*TASK_TO_DEG_DEG_MM
                x=measured_features(oct_feature[:3],state,velocity,self.local_previous)
                if not self.local_active and valid and norm<=SWITCH_MM:
                    self.local_active=True
                    self.invalidate_action_chunk()
                    self.switch_info=dict(action_cycle=self.step_index+1,pre_action_index=self.step_index,
                        measured_oct_norm_mm=norm,measured_input12=x.copy(),simulation_time_s=scene.motor_execution.time_s,
                        velocity_deg_deg_mm_s=velocity.copy(),previous_accepted_deg_deg_mm=self.local_previous.copy())
                    save(folder/'switch_receipt.json',self.switch_info)
                    print('ONE_TIME_MEASURED_SWITCH',json.dumps(self.switch_info,default=default),flush=True)
                if self.local_active:
                    if not valid:
                        raise ValueError('Invalid OCT in learned local mode; no fallback authorized')
                    action=self.learned.predict(x)
                    mode='learned_local'
                else:
                    action=super().get_action(rgb,state,oct_feature)
                    mode='v5'
                self.route_counts[mode]+=1
                self.latest=dict(mode=mode,measured_input12=x.copy(),oct_valid=valid,oct_norm_mm=norm,
                                 rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb)).hexdigest() if self.step_index==0 else None)
                self.step_index+=1
                return action

            def record_applied_action(self,action):
                super().record_applied_action(action)
                self.local_previous=np.asarray(action,dtype=np.float64).copy()

        evaluator=Hybrid(str(CHECKPOINT),device='cuda',num_action_samples=1,n_action_execute_steps=1,
                         flow_integration_steps=4,measure_inference=True,action_aggregation='median',
                         controller_variant='feedback',rgbd_render_quality=QUALITY.copy())
        evaluator.learned=LearnedPolicy(MODEL)
        assert evaluator.fine_evaluator is None
        targets=read(ROOT/'configs/v12_reachable_development_targets_0p35um.json')['targets']
        target=targets[index]
        scene=RCMSceneV9()
        world=np.asarray(scene.robot_to_world(np.asarray(target['reachable_target_robot_mm'])))
        scene.set_target_world(world)
        settings=visual_servo_settings(scene)
        settings.update(perception_backend='strict_rgbd',rgbd_render_quality=QUALITY.copy())
        sensor=StrictRGBDSensor(scene,settings)
        if scene.motor_execution is None:
            scene.enable_motor_execution()
            scene.reset_motor_time(0.)
        trace=(folder/'cycles.jsonl').open('x',buffering=1)
        initial_tip=scene.site_mm('needle_tip').copy()
        initial=dict(tip_world_mm=initial_tip,target_world_mm=world,initial_error_um=float(np.linalg.norm(initial_tip-world)*1000),
                     state_deg_deg_mm=scene.current_task()*TASK_TO_DEG_DEG_MM,start='untouched production standard constructor reset')
        save(folder/'initial_state.json',initial)
        def observe_step(event):
            tip=scene.site_mm('needle_tip').copy()
            post=float(np.linalg.norm(tip-world)*1000)
            previous=initial_tip if not rows else rows[-1]['post_tip_world_mm']
            row=dict(action_cycle=event['step']+1,**evaluator.latest,**event,
                     pre_tip_world_mm=previous,post_tip_world_mm=tip,target_world_mm=world,
                     post_true_error_um=post,post_state_deg_deg_mm=scene.current_task()*TASK_TO_DEG_DEG_MM,
                     post_velocity_deg_deg_mm_s=scene.motor_execution.actual_velocity*TASK_TO_DEG_DEG_MM,
                     post_simulation_time_s=scene.motor_execution.time_s,
                     encoder_settled=bool(scene.motor_execution.settled()))
            trace.write(json.dumps(row,default=default,allow_nan=False)+'\n')
            rows.append(row)
            if len(rows)==1:
                receipt=dict(status='actual_first_action_completed',time=now(),pid=pid,ppid=os.getppid(),target=f'E{index:02d}',
                    seed=42+index,action_cycle=1,simulation_time_s=scene.motor_execution.time_s,
                    initial_error_um=initial['initial_error_um'],post_true_error_um=post,
                    driver=event['motor_record']['driver']['variant'],gpu=torch.cuda.get_device_name(0),
                    receipt_path=str(folder/'actual_start_receipt.json'))
                save(folder/'actual_start_receipt.json',receipt)
                acknowledge(receipt)
                record('actual first action',receipt)
        evaluator.step_observer=observe_step
        started=time.monotonic()
        ep=evaluator.evaluate_episode(scene,world,sensor=sensor,seed=42+index,max_steps=1600,
            error_threshold_mm=.00035,control_period_s=.05,action_limit_deg_deg_mm=(.2,.1,.03),
            max_consecutive_safety_limited=None,progress_every=100,stop_on_success=False)
        ep.update(episode_index=index,target_id=f'E{index:02d}',seed=42+index,target_manifest=target,
                  initial_boundary=initial,route_counts=evaluator.route_counts,switch=evaluator.switch_info,
                  checkpoint_sha256=V5_SHA,local_model_sha256=MODEL_SHA,egl=egl_identity(),elapsed_s=time.monotonic()-started)
        save(folder/'evaluation_results.json',ep)
        assert len(rows)==len(ep['trajectory']['actions'])==len(ep['motor_records'])==1600
        assert ep['early_stop_reason'] is None
        post=np.asarray([r['post_true_error_um'] for r in rows])
        aligned=np.r_[ep['trajectory']['errors'][1:],ep['final_error_mm']]*1000
        np.testing.assert_allclose(post,aligned,rtol=0,atol=1e-8)
        np.testing.assert_allclose([r['post_simulation_time_s'] for r in rows],np.arange(1,1601)*.05,atol=1e-8,rtol=0)
        assert all(r['motor_record']['driver']['variant']=='g10' for r in rows)
        summary=dict(status='completed',target=f'E{index:02d}',seed=42+index,actions=1600,initial_true_error_um=initial['initial_error_um'],
            final_true_error_um=float(post[-1]),last200_max_true_error_um=float(post[-200:].max()),
            last200_includes_final_action=True,primary_200um=radius_metrics(post,200),long_term_20um=radius_metrics(post,20),
            original_strict_0p35um_success=ep['success'],original_terminal_success=ep['terminal_success'],
            switch=evaluator.switch_info,route_counts=evaluator.route_counts,teacher_calls=0,geometry_calls=0,
            model_sha256=dict(v5=V5_SHA,local=MODEL_SHA),elapsed_s=ep['elapsed_s'],
            independent_truth='Live scene.site_mm(needle_tip) after each apply_prepared_motor; worldXYZ 3D norm to target',
            control_inputs='Measured RGB/OCT + ABC; ridge only measured OCT/ABC/actual velocity/previous accepted command',
            source_protocol=str(OUT/'experiment_card.json'),trajectory=str(folder/'cycles.jsonl'),raw=str(folder/'evaluation_results.json'))
        if evaluator.switch_info:
            k=evaluator.switch_info['pre_action_index']
            summary['switch_scoring_only']=dict(pre_tip_world_mm=rows[k]['pre_tip_world_mm'],target_world_mm=world,
                pre_true_error_um=rows[k]['error_before_mm']*1000,post_true_error_um=rows[k]['post_true_error_um'],
                true_world_delta_mm=world-np.asarray(rows[k]['pre_tip_world_mm']),raw_action=rows[k]['raw_action_deg_deg_mm'],
                accepted_action=rows[k]['accepted_action_deg_deg_mm'])
        save(folder/'summary.json',summary)
        record('target completed',summary)
        return 0
    except BaseException as error:
        failure=dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',time=now(),pid=pid,
                     target=f'E{index:02d}',seed=42+index,completed_actions=len(rows),error=repr(error),traceback=traceback.format_exc())
        save(folder/'failure.json',failure)
        acknowledge(failure)
        record('target failed or interrupted',failure)
        traceback.print_exc()
        return 1
    finally:
        if trace is not None: trace.close()
        if sensor is not None: sensor.close()
        if scene is not None: scene.close()


def finish_comparison(index):
    # Entirely separate process from actors. No archived control/state is available to the actor.
    import numpy as np
    folder=OUT/f'E{index:02d}'
    summary=read(folder/'summary.json')
    ep=read(folder/'evaluation_results.json')
    row=read(ARCHIVE/f'E{index:02d}.json')
    assert sha(row['result_path'])==row['result_sha256']
    archive=read(row['result_path'])
    old=archive['episodes'][0]
    assert old['seed']==42+index
    np.testing.assert_allclose(old['target_world_mm'],ep['target_world_mm'],atol=1e-10,rtol=0)
    count=summary['route_counts']['v5']
    checks={}
    for key in ('raw_actions','actions','states','errors','velocities','oct_navigation_mm','simulation_time_s'):
        end=count if key in ('raw_actions','actions') else min(1600,count+1)
        a=np.asarray(ep['trajectory'][key][:end],dtype=float)
        b=np.asarray(old['trajectory'][key][:end],dtype=float)
        maximum=float(np.max(np.abs(a-b))) if a.size else 0.
        checks[key]=dict(max_abs_difference=maximum,exact_equal=bool(np.array_equal(a,b)),allclose_atol_1e_8=bool(np.allclose(a,b,rtol=0,atol=1e-8)))
    post=np.r_[old['trajectory']['errors'][1:],old['final_error_mm']]*1000
    summary['pre_switch_baseline_comparison']=dict(status='passed' if all(v['allclose_atol_1e_8'] for v in checks.values()) else 'mismatch',
        compared_v5_actions=count,source=row['result_path'],source_sha256=row['result_sha256'],checks=checks,
        archive_used_for_control=False,comparison_performed_after_all_workers_exit=True)
    summary['baseline']=dict(final_true_error_um=float(post[-1]),primary_200um=radius_metrics(post,200),long_term_20um=radius_metrics(post,20),
                             source_contract=archive.get('evaluation_contract'),source=row['result_path'])
    summary['paired_final_delta_um']=summary['final_true_error_um']-float(post[-1])
    save(folder/'summary.json',summary)
    return summary


def supervise():
    with (OUT/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'supervisor_start_receipt.json').exists(), 'Refuse duplicate execution'
        for path,value in read(OUT/'preflight.json')['frozen_files'].items():
            assert sha(path)==value, f'Frozen input changed: {path}'
        save(OUT/'supervisor_start_receipt.json',dict(time=now(),pid=os.getpid(),ppid=os.getppid(),desktop={k:os.environ.get(k) for k in DESKTOP}))
        save(OUT/'workflow_status.json',dict(status='running',time=now(),pid=os.getpid(),targets=[6,8],requested_actions_each=1600))
        children=[]
        threads=[]
        def tee(pipe,path):
            with path.open('x',buffering=1) as log:
                for line in pipe:
                    log.write(line)
                    print(line,end='',flush=True)
        def interrupt(signum,frame):
            for child in children:
                if child.poll() is None: child.send_signal(signal.SIGINT)
        signal.signal(signal.SIGTERM,interrupt)
        signal.signal(signal.SIGHUP,interrupt)
        signal.signal(signal.SIGINT,interrupt)
        try:
            for i in (6,8):
                p=subprocess.Popen([sys.executable,'-E','-B','-u',str(Path(__file__)),'worker','--index',str(i)],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                children.append(p)
                t=threading.Thread(target=tee,args=(p.stdout,OUT/f'E{i:02d}.terminal.log'))
                t.start()
                threads.append(t)
            save(OUT/'parallel_process_receipt.json',dict(time=now(),supervisor_pid=os.getpid(),workers=[dict(target=f'E{i:02d}',seed=42+i,pid=p.pid) for i,p in zip((6,8),children)]))
            codes=[p.wait() for p in children]
            for t in threads: t.join()
            results=[]
            for i in (6,8):
                if (OUT/f'E{i:02d}/summary.json').exists(): results.append(finish_comparison(i))
            complete=len(results)==2 and all(c==0 for c in codes)
            valid=complete and all(r['pre_switch_baseline_comparison']['status']=='passed' for r in results)
            result=dict(status='completed' if valid else 'failed_or_incomplete',time=now(),completed_targets=len(results),requested_targets=2,
                worker_exit_codes=codes,protocol_and_baseline_verified=valid,
                primary_200um_passes=sum(r['primary_200um']['passed'] for r in results),
                long_term_20um_passes=sum(r['long_term_20um']['passed'] for r in results),
                additional_targets_run=False,additional_targets_require_separate_planner_action=True,
                both20_gate_for_original_plan=bool(valid and all(r['long_term_20um']['passed'] for r in results)),targets=results)
            save(OUT/'summary.json',result)
            save(OUT/'workflow_status.json',result)
            record('final completeness and acceptance',result)
            (OUT/'report.md').write_text('# 两个标准起点验收\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print('FINAL_RESULT_PATH',OUT/'summary.json',flush=True)
            return 0 if valid else 1
        except BaseException as error:
            failure=dict(status='failed',time=now(),error=repr(error),traceback=traceback.format_exc())
            save(OUT/'supervisor_failure.json',failure)
            save(OUT/'workflow_status.json',failure)
            record('supervisor failure',failure)
            traceback.print_exc()
            return 1


def launch():
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight()
        os.mkfifo(OUT/'startup_ack.fifo',0o600)
        fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
        env=os.environ.copy()
        env.update(DESKTOP)
        cmd=['gnome-terminal','--window','--title=v5 learned local standard2 E06-48 E08-50','--working-directory='+str(ROOT),
             '--','bash',str(OUT/'launch.sh')]
        save(OUT/'gnome_launch_request.json',dict(time=now(),command=cmd,desktop_environment=DESKTOP))
        p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=20)
        save(OUT/'gnome_launch_receipt.json',dict(time=now(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,command=cmd))
        if p.returncode:
            os.close(fd)
            record('GNOME launch failed',dict(returncode=p.returncode,stderr=p.stderr,experiment_started=False))
            raise RuntimeError(p.stderr)
        sel=selectors.DefaultSelector()
        sel.register(fd,selectors.EVENT_READ)
        deadline=time.monotonic()+90
        buffer=''
        receipts=[]
        while len(receipts)<2 and time.monotonic()<deadline:
            events=sel.select(max(0,deadline-time.monotonic()))
            if not events: break
            buffer+=os.read(fd,16384).decode()
            while '\n' in buffer:
                line,buffer=buffer.split('\n',1)
                receipt=json.loads(line)
                receipts.append(receipt)
                print(json.dumps(receipt,ensure_ascii=False),flush=True)
        os.close(fd)
        sel.close()
        save(OUT/'startup_acknowledgements.json',dict(time=now(),receipts=receipts,actual_started=sum(r.get('status')=='actual_first_action_completed' for r in receipts)))
        print('STARTUP_ACK_PATH',OUT/'startup_acknowledgements.json',flush=True)
        return 0 if len(receipts)==2 and all(r.get('status')=='actual_first_action_completed' for r in receipts) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['launch','supervise','worker'])
    parser.add_argument('--index',type=int,choices=[6,8])
    args=parser.parse_args()
    if args.mode=='worker': sys.exit(worker(args.index))
    if args.mode=='supervise': sys.exit(supervise())
    sys.exit(launch())
