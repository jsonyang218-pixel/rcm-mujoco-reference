"""Bounded v3 quantization-recovery run, launched once in GNOME per SOP."""
from __future__ import annotations
import argparse
import ast
import fcntl
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
import campaign as c

OUT=c.OUT;ROOT=c.ROOT;OLD=ROOT/'runs/20261127/dit_relative_v2_resume'
PYTHON=str(ROOT/'.venv/bin/python');START=None;ACTIVE=[];STOP=False

def interrupted(signum,frame):
    global STOP
    STOP=True

def group(stage,commands,limit,concurrency):
    assert 1<=concurrency<=12
    started=time.monotonic();deadline=min(START+2670,started+limit)
    queue=list(commands);results=[]
    c.status(stage,remaining_batch_s=max(0,START+2700-started))
    try:
        while queue or ACTIVE:
            if STOP:raise InterruptedError('Terminal signal received')
            if time.monotonic()>=deadline:raise TimeoutError(f'{stage}: stage/shared2700s deadline')
            while queue and len(ACTIVE)<concurrency:
                name,args=queue.pop(0);path=OUT/'logs'/f'{name}.log';path.parent.mkdir(exist_ok=True)
                stream=path.open('x');cmd=[PYTHON,'-E','-B','-u',str(OUT/'v3.py'),*args]
                env={**os.environ,**c.RUNTIME,'DIT_STEP_SECONDS':str(deadline-time.monotonic()-10)}
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    start_new_session=True,text=True,bufsize=1)
                def tee(process=p,output=stream,label=name):
                    for line in process.stdout:
                        output.write(line);output.flush();print(f'[{label}] {line}',end='',flush=True)
                thread=threading.Thread(target=tee,daemon=True);thread.start()
                ACTIVE.append((name,p,stream,thread,time.monotonic()))
                c.append(OUT/'commands.jsonl',dict(time=c.now(),stage=stage,command=cmd,pid=p.pid,
                    timeout_s=deadline-time.monotonic(),log=str(path),maximum_concurrency=concurrency))
            for item in ACTIVE[:]:
                name,p,stream,thread,t=item;rc=p.poll()
                if rc is None:continue
                thread.join(timeout=5);stream.close();ACTIVE.remove(item)
                result=dict(time=c.now(),name=name,pid=p.pid,exit_code=rc,elapsed_s=time.monotonic()-t)
                c.append(OUT/'worker_exit_receipts.jsonl',result);results.append(result)
                if rc:raise RuntimeError(f'{name} exited{rc}; see logs/{name}.log; engineering failure')
            if ACTIVE:time.sleep(.15)
        return results
    finally:
        for name,p,stream,thread,t in ACTIVE:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,p,stream,thread,t in ACTIVE:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            thread.join(timeout=2);stream.close()
            c.append(OUT/'worker_exit_receipts.jsonl',dict(time=c.now(),name=name,pid=p.pid,exit_code=p.returncode,
                interrupted=True,elapsed_s=time.monotonic()-t))
        ACTIVE.clear()
        c.append(OUT/'budget_ledger.jsonl',dict(time=c.now(),stage=stage,elapsed_s=time.monotonic()-started,
            batch_elapsed_s=time.monotonic()-START,batch_limit_s=2700,completed=results))

def validate_branch(spec):
    import numpy as np
    import legacy_data
    path=OUT/'data'/spec['id'];summary=c.read(path/'summary.json')
    assert summary['status']=='completed' and summary['branch']==spec
    obs=legacy_data.read_rows(path/'observations.jsonl');acts=legacy_data.read_rows(path/'actions.jsonl')
    assert len(obs)==spec['teacher_steps']+3 and len(acts)==spec['teacher_steps']+2
    for i,row in enumerate(obs):
        assert row['target_id'] in ('E40','E42') and row['index']==i
        assert abs(row['time_s']-.05*i)<1e-9
        image=OUT/row['rgb_path'];assert c.sha(image)==row['rgb_sha256']
    for i,a in enumerate(acts):
        assert a['pre_observation']==i and a['post_observation']==i+1
        np.testing.assert_array_equal(obs[i+1]['previous_accepted_deg_deg_mm'],a['accepted_deg_deg_mm'])
        assert a['motor_record']['actual_after_robot_ABCE']==obs[i+1]['encoder_ABCE']
    return summary

def verify_predicted_executability(folder,predictions,ids,manifest):
    """Offline encoding audit only; never modifies or supplies policy actions."""
    import numpy as np
    import legacy_data as d
    from rcm.scene_v9 import RCMSceneV9
    from motor_backend import MotorBackend
    unit=np.array([np.pi/180,np.pi/180,.001]);scene=RCMSceneV9();cache={};evidence=[]
    try:
        for prediction,index in zip(predictions.numpy(),ids):
            meta=manifest['train'][index];episode=meta['episode'];path=Path(meta['source'])
            if episode not in cache:
                obs=d.read_rows(path/'observations.jsonl');acts=d.read_rows(path/'actions.jsonl')
                cache[episode]=(obs,acts,MotorBackend(initial_robot=tuple(acts[0]['motor_record']['actual_before_robot_ABCE'])))
            obs,acts,backend=cache[episode];j=meta['pre_observation'];pre=obs[j]
            q=np.array(pre['state_deg_deg_mm'])*unit;raw=np.clip(prediction,[-.2,-.1,-.03],[.2,.1,.03])
            requested=tuple(round(float(v)) for v in scene.encoders(q+raw*unit))+(0,)
            command=backend.command_for_robot_target(requested)
            handles=tuple(getattr(command,'handleEncoder'+axis) for axis in 'ABCE')
            quantized=list(backend.kin.validate_manual_jog(backend._anchor_robot,handles))
            before=pre['encoder_ABCE'];teacher=acts[j]
            evidence.append(dict(episode=episode,observation=j,OCT_mm=pre['signed_OCT_mm'],
                input_velocity=pre['actual_velocity_si'],input_previous_accepted=pre['previous_accepted_deg_deg_mm'],
                prediction_raw_deg_deg_mm=prediction.tolist(),production_clipped_deg_deg_mm=raw.tolist(),
                requested_ABCE=requested,quantized_target_ABCE=quantized,current_ABCE=before,
                executable=quantized!=before,teacher_accepted_deg_deg_mm=teacher['accepted_deg_deg_mm'],
                teacher_quantized_ABCE=teacher['motor_record']['quantized_target_robot_ABCE']))
    finally:scene.close()
    executable=sum(r['executable'] for r in evidence)
    result=dict(status='completed',stationary_recovery_inputs=len(evidence),predicted_executable=executable,
        executable_fraction=executable/max(1,len(evidence)),evidence=evidence,
        only_training_targets=['E40','E42'],new_motor_ticks=0,used_as_external_action_adjustment=False,
        no_rejection_gate=True,method='unchanged production clipping then exact pinned G10 encoding at actual saved state/session anchor')
    c.save(folder/'offline_stationary_quantization_check.json',result)
    c.record('v3_offline_actual_G10_predicted_executability',dict(candidate=folder.name,
        stationary_recovery_inputs=len(evidence),predicted_executable=executable,
        executable_fraction=result['executable_fraction'],evidence_path=str(folder/'offline_stationary_quantization_check.json')))

def train(candidate,repair=False):
    import torch
    import data_pipeline as dp
    from model import integrate
    torch=c.torch_setup();folder=OUT/candidate;folder.mkdir(exist_ok=False)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,c.request_training_stop)
    init=OUT/'candidate1/checkpoint.pth' if repair else OLD/'candidate1/checkpoint.pth'
    e=c.evaluator();cp=c.checkpoint_load(e,init);m=e.model
    data=torch.load(OUT/'relative_windows.pt',map_location='cpu',weights_only=False)
    manifest=c.read(OUT/'relative_window_manifest.json')
    assert {r['target'] for r in manifest['train']}=={'E40','E42'}
    assert manifest['development_used_for_training'] is False
    dp.verify_inputs(m,data,folder/'input_ablation_before.json')
    initsha=c.sha(init);initial={n:p.detach().cpu().clone() for n,p in m.named_parameters()}
    for name,p in m.named_parameters():p.requires_grad_(not name.startswith('vision.'))
    trainable=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=1e-5 if repair else 3e-5,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(20261127+(1 if repair else 0))
    td={k:v.cuda() for k,v in data['train'].items()};m.train();m.vision.eval()
    target_steps=1200 if repair else 2400;start=time.monotonic();gradient_evidence={};updates=0
    # Keep v2 denormalization bit-exact. It is inside the original DiT RF
    # sampler, not an external post-processing gain or alternate controller.
    mean=m.action_mean.detach().cpu().clone();std=m.action_std.detach().cpu().clone()
    c.save(folder/'scales.json',dict(action_mean=mean,action_std=std,source=str(init),
        source_sha256=initsha,unchanged_v2_trainonly_output_scales=True,heldout_scale_estimation=False))
    c.TRAIN_RNG=generator;c.TRAIN_GRADIENTS=gradient_evidence
    def save_checkpoint():
        cp=dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},action_mean=mean,action_std=std,
            actual_updates=updates,planned_updates=target_steps,init_checkpoint=str(init),init_sha256=initsha,
            optimizer=opt.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),gradient_evidence=gradient_evidence,
            training_targets=['E40','E42'],excluded_targets=['E41','E00','E02','E06','E08','E10'],
            architecture='original6block_DiT_four_midpoint_RF',policy_inference_postprocessor=None,
            recovery_teacher=str(OUT/'teacher.py'),time=c.now())
        tmp=folder/'checkpoint.tmp';torch.save(cp,tmp);tmp.replace(folder/'checkpoint.pth')
    interrupted=False
    try:
        for step in range(1,target_steps+1):
            if c.STOP_REQUESTED or time.monotonic()-start>float(os.environ.get('DIT_STEP_SECONDS','420'))-20:
                interrupted=True;break
            idx=dp.draw(data['pools'],generator,size=12,repair=repair)
            b={k:v[idx] for k,v in td.items()};args=tuple(b[k] for k in c.KEYS)
            opt.zero_grad(set_to_none=True);condition=m._condition(*args)
            pred=integrate(m,args,condition=condition)
            first=(((pred[:,0]-b['action'][:,0])/m.action_std)**2).mean()
            sequence=(((pred-b['action'])/m.action_std)**2).mean()
            norm=(b['action']-m.action_mean)/m.action_std;noise=torch.randn_like(norm)
            tau=torch.rand(len(idx),device='cuda');t=tau[:,None,None]
            velocity=m._predict_velocity_from_condition((1-t)*noise+t*norm,tau,*condition)
            rf=((velocity-(norm-noise))**2).mean()
            loss=first+.02*sequence+.002*rf
            assert bool(torch.isfinite(loss)),'Nonfinite training loss'
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(trainable,1.)
            assert bool(torch.isfinite(gn)),'Nonfinite gradient'
            if step in (1,600,1200,2400):
                gradient_evidence[str(step)]={n:float(p.grad.norm()) for n,p in m.named_parameters()
                    if p.grad is not None and ('self_attention.q.weight' in n or n=='output.weight')}
            opt.step();updates=step
            entry=dict(time=c.now(),update=step,planned=target_steps,loss=float(loss.detach()),
                first_action_mse=float(first.detach()),sequence_mse=float(sequence.detach()),rf_mse=float(rf.detach()),
                gradient_norm=float(gn),elapsed_s=time.monotonic()-start)
            c.append(folder/'training_curve.jsonl',entry)
            if step==1 or step%50==0:print(json.dumps(entry),flush=True)
            if step%600==0:
                save_checkpoint();torch.save(dict(model={k:v.detach().cpu() for k,v in m.state_dict().items()},
                    action_mean=mean,action_std=std,actual_updates=updates),folder/f'step_{updates:06d}.pth')
    finally:save_checkpoint()
    if interrupted or updates!=target_steps:
        c.record('v3_training_budget_partial',dict(candidate=candidate,actual_updates=updates,planned=target_steps))
        raise TimeoutError('Weights optimizer and RNG saved; incomplete training blocks evaluation')
    delta={n:float((p.detach().cpu()-initial[n]).norm()) for n,p in m.named_parameters()}
    assert all(delta[n]==0 for n in delta if n.startswith('vision.'))
    assert all(any(delta[n]>0 for n in delta if n.startswith(f'blocks.{i}.self_attention.')) for i in range(6))
    assert torch.equal(m.action_mean.cpu(),mean) and torch.equal(m.action_std.cpu(),std)
    c.save(folder/'parameter_update_evidence.json',dict(actual_updates=updates,gradient_probes=gradient_evidence,
        parameter_delta_l2=delta,all_six_blocks_updated=True,vision_unchanged=True,init_sha256=initsha,
        checkpoint_sha256=c.sha(folder/'checkpoint.pth')))
    m.eval();offline={};stalled_prediction=None;stalled_ids=None
    with torch.no_grad():
        for pool in ('stalled_recovery','recovery','new_hold','retention'):
            ids=data['pools'][pool];ids=ids[:min(len(ids),256)]
            predictions=[]
            for k in range(0,len(ids),12):
                predictions.append(integrate(m,tuple(td[key][ids[k:k+12]] for key in c.KEYS))[:,0].cpu())
            prediction=torch.cat(predictions);labels=data['train']['action'][ids,0]
            if pool=='stalled_recovery':stalled_prediction=prediction;stalled_ids=ids
            offline[pool]=dict(windows=len(ids),MAE_deg_deg_mm=(prediction-labels).abs().mean(0),
                predicted_action_abs_mean=prediction.abs().mean(0),teacher_action_abs_mean=labels.abs().mean(0))
    c.save(folder/'offline_action_diagnostic.json',offline)
    verify_predicted_executability(folder,stalled_prediction,stalled_ids,manifest)
    dp.verify_inputs(m,data,folder/'input_ablation_after.json')
    summary=dict(status='trained',actual_updates=updates,init_checkpoint=str(init),init_sha256=initsha,
        checkpoint_sha256=c.sha(folder/'checkpoint.pth'),repair=repair,
        train_only_targets=['E40','E42'],heldout_inputs_consumed_for_training=False,offline=offline)
    c.save(folder/'summary.json',summary);c.record('v3_DiT_trained',summary)

def diagnose_near(candidate):
    import numpy as np
    import legacy_data as d
    report={}
    for kind in c.KINDS:
        path=OUT/candidate/'near'/kind
        data=d.read_rows(path/'cycles.jsonl');terminal=data[-200:]
        baseline=c.read(OLD/'candidate1/near'/kind/'summary.json')['metrics']
        report[kind]=dict(metrics=c.read(path/'summary.json')['metrics'],v2_metrics=baseline,
            mean_error_xyz_um=np.mean([r['post']['true_error_xyz_um'] for r in terminal],axis=0),
            stationary_cycles=sum(r['event']['motor_record']['actual_before_robot_ABCE']==r['event']['motor_record']['actual_after_robot_ABCE'] for r in terminal),
            mean_actions=np.mean([r['event']['accepted_action_deg_deg_mm'] for r in terminal],axis=0),
            mean_OCT_um=np.mean([r['pre']['signed_OCT_mm'][:3] for r in terminal],axis=0)*1000)
    result=dict(candidate=candidate,arms=report,
        correction='One preauthorized E40/E42-only retention-weighted repair; no E41 labels/inputs used in optimization',
        repair_data='Same frozen contiguous E40/E42 training data; increased axial/hold retention only')
    c.save(f'{candidate}/near_regression_diagnosis.json',result);c.record('near_regression_diagnosed',result)

def execute():
    global START
    with (OUT/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'actual_start_receipt.json').exists(),'Never silently overwrite prior run'
        START=time.monotonic();c.START=START
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
        receipt=dict(time=c.now(),pid=os.getpid(),ppid=os.getppid(),actual_start=True,output_dir=str(OUT),
            source=str(OLD),budget_s=2700,concurrency_max=12,interpreter=sys.executable,
            launch_method='GNOME new window, FIFO one-shot acknowledgement',simulation_only=True)
        c.save('actual_start_receipt.json',receipt)
        fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK)
        os.write(fd,(json.dumps(receipt)+'\n').encode());os.close(fd)
        c.record('v3_actual_start',receipt)
        summary=dict(workflow_status='running',goal_achieved=False,actual_start=receipt,simulation_only=True)
        try:
            changed=[p for p,digest in c.read(OUT/'input_fingerprints.json').items() if c.sha(p)!=digest]
            assert not changed,f'Preflight input drift: {changed}'
            import data_pipeline as dp
            specs=dp.plans()
            group('G10_continuous_teacher_collection',[(s['id'],['collect','--branch',s['id']]) for s in specs],760,12)
            for spec in specs:validate_branch(spec)
            summary['collection']=dict(completed=len(specs),planned=len(specs),only_targets=['E40','E42'])
            group('build_contiguous_windows',[('build',['build'])],160,1)
            assert c.read(OUT/'coverage.json')['status']=='passed'
            group('minimal_v2_warmstart_fit',[('train_candidate1',['train','--candidate','candidate1'])],450,1)
            selected='candidate1'
            group('E41_four_50um_first',[('near1_'+k,['evaluate','--candidate',selected,'--kind',k]) for k in c.KINDS],360,4)
            near=c.near_results(selected);summary['initial_near_E41']=near
            if not near['all_passed']:
                diagnose_near(selected)
                if START+2670-time.monotonic()<500:raise TimeoutError('Near regression diagnosed; insufficient budget for full preauthorized repair and evaluation')
                group('E40_E42_only_near_regression_repair',[('train_candidate2',['train','--candidate','candidate2','--repair'])],240,1)
                selected='candidate2'
                group('E41_four_50um_repaired',[('near2_'+k,['evaluate','--candidate',selected,'--kind',k]) for k in c.KINDS],360,4)
                near=c.near_results(selected);summary['repaired_near_E41']=near
                if not near['all_passed']:diagnose_near(selected)
            summary['selected_candidate']=selected;summary['near_E41']=near
            if not near['all_passed']:
                summary.update(workflow_status='complete_near_negative_after_repair',standard_five='blocked_by_E41_gate')
            else:
                group('five_standard_1600cycles',[(f'standard_E{i:02d}',['evaluate','--candidate',selected,'--index',str(i)]) for i in (0,2,6,8,10)],970,5)
                results={f'E{i:02d}':c.read(OUT/selected/'standard'/f'E{i:02d}'/'summary.json') for i in (0,2,6,8,10)}
                assert all(r['metrics']['completed_cycles']==1600 for r in results.values())
                passed=sum(r['metrics']['strict20um_pass'] for r in results.values())
                comparison={t:dict(v2=c.read(OLD/'candidate1/standard'/t/'summary.json')['metrics'],v3=r['metrics']) for t,r in results.items()}
                summary['standard_five']=dict(completed=5,denominator=5,strict20um_pass=passed,cycles_each=1600,paired=comparison)
                summary.update(workflow_status='complete_strict_pass' if passed==5 else 'complete_standard_negative',goal_achieved=passed==5)
                c.save('paired_standard_results.json',comparison)
        except BaseException as err:
            summary.update(workflow_status='budget_partial' if isinstance(err,TimeoutError) else 'interrupted' if isinstance(err,InterruptedError) else 'technical_failure',
                error=repr(err),traceback=traceback.format_exc(),scientific_negative=False)
        finally:
            # The stage supervisor has already terminated/waited all workers.
            summary.update(time=c.now(),elapsed_s=time.monotonic()-START,batch_limit_s=2700)
            c.save('summary.json',summary);c.status(summary['workflow_status'],summary_path=str(OUT/'summary.json'))
            c.record('v3_batch_finished',summary)
            (OUT/'report.md').write_text('# DiT quantization recovery v3\n\n'+json.dumps(summary,indent=2,ensure_ascii=False,default=c.default)+'\n')
            print(f'RESULTS {OUT}/summary.json {OUT}/report.md',flush=True)
        if summary['workflow_status'] not in ('complete_strict_pass','complete_standard_negative','complete_near_negative_after_repair'):
            raise SystemExit(1)

def preflight():
    import numpy as np
    import torch
    import data_pipeline as dp
    assert not (OUT/'actual_start_receipt.json').exists()
    for name in ('v3.py','campaign.py','model.py','collection_helper.py','teacher.py','data_pipeline.py','legacy_data.py','audit.py'):
        ast.parse((OUT/name).read_text())
    torch=c.torch_setup();e=c.evaluator();cp=c.checkpoint_load(e,OLD/'candidate1/checkpoint.pth')
    assert cp['actual_updates']==3000 and cp['training_targets']==['E40','E42']
    assert c.sha(OUT/'model.py')==c.sha(OLD/'model.py')
    data=torch.load(OLD/'relative_windows.pt',map_location='cpu',weights_only=False)
    dp.verify_inputs(e.model,data,OUT/'preflight_input_ablation.json')
    assert c.score([0.]*240,240)['strict20um_pass']
    assert not c.score([0.]*239+[20.01],240)['strict20um_pass']
    assert not c.score([0.]*1599,1600)['strict20um_pass']
    specs=dp.plans();assert len(specs)==92 and {s['target'] for s in specs}=={'E40','E42'}
    assert all(not (OUT/'data'/s['id']).exists() for s in specs)
    # Interface preflight uses real strict sensor observations and previews,
    # without committing a tick or generating training records.
    from rcm.scene_v9 import RCMSceneV9
    from scripts.research.homog_replay_common import make_teacher
    from teacher import QuantizedTeacher
    trials=[]
    for target in ('E40','E42'):
        task=next(t['task'] for t in c.read(OUT/'experiment_card.json')['targets'] if t['task']['id']==target)
        scene=RCMSceneV9();teacher=None
        try:
            scene.set_target_world(scene.robot_to_world(np.asarray(task['target_robot_mm'])))
            q=scene.target_task(validate=True).copy();q[:2]+=np.array([.15,-.15])*np.pi/180
            scene.set_task(q);scene.enable_motor_execution();scene.reset_motor_time(0.)
            teacher=QuantizedTeacher(make_teacher(scene),scene)
            obs=teacher.sensor.observe();q=scene.current_task().copy()
            result=teacher.geometry_action_from_observation(q,obs,previous_velocity=np.zeros(3),command_q=q)
            assert result['valid'];action=np.array(result['action_deg_deg_mm'])
            preview=scene.motor_execution.preview(q+action*np.array([np.pi/180,np.pi/180,.001]),.05)
            assert preview.record['quantized_target_robot_ABCE']==list(teacher.last_diagnostic['quantized_target_ABCE'])
            assert preview.record['quantized_target_robot_ABCE']!=preview.record['actual_before_robot_ABCE']
            assert scene.motor_execution.time_s==0 and len(scene.motor_execution.records)==0
            trials.append(dict(target=target,action=action,teacher=teacher.last_diagnostic,preview=preview.record,committed_ticks=0))
        finally:
            if teacher is not None:teacher.sensor.close()
            scene.close()
    c.save('preflight_teacher_driver_preview.json',trials)
    files=[OUT/name for name in ('v3.py','campaign.py','model.py','collection_helper.py','teacher.py','data_pipeline.py','legacy_data.py','audit.py','launch.sh','experiment_card.json')]
    files += [OLD/'candidate1/checkpoint.pth',OLD/'relative_windows.pt',OLD/'relative_window_manifest.json',c.REF,
        ROOT/'rcm/motor_execution.py',ROOT/'mujoco_model/motor_backend.py',ROOT/'rcm/scene_v9.py',ROOT/'rcm/reference.py',
        ROOT/'lerobot_training/eval_policy.py',ROOT/'lerobot_training/dit_rectified_flow.py',
        ROOT/'mujoco_model/vendor/simulation_driver/ros_msg/src/robot_control/kinematics.py',
        ROOT/'scripts/research/homog_replay_common.py',c.PRIOR/'pilot.py',c.ORIGINAL/'pilot.py',
        ROOT/'configs/v12_reachable_development_targets_0p35um.json']
    c.save('input_fingerprints.json',{str(p):c.sha(p) for p in files})
    result=dict(status='passed',time=c.now(),cuda_device=torch.cuda.get_device_name(0),warmstart=str(OLD/'candidate1/checkpoint.pth'),
        warmstart_sha256=c.sha(OLD/'candidate1/checkpoint.pth'),architecture_match=True,all6blocks=True,
        original4step_sampler_parity=True,teacher_preview_real_G10_pass=True,committed_motor_ticks=0,
        training_updates=0,collection_branches=92,parallel_max=12,total_hard_seconds=2700,
        no_geometry_or_accumulator_or_amplifier_in_policy=True,near_regression_repair_pre_authorized=True)
    c.save('preflight.json',result);c.record('v3_preflight_passed',result);print(json.dumps(result),flush=True)

def launch():
    assert c.read(OUT/'preflight.json')['status']=='passed'
    assert not (OUT/'actual_start_receipt.json').exists()
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'gnome_launch_request.json').exists(),'One launch only; failed requests are retained'
        fifo=OUT/'startup_ack.fifo';os.mkfifo(fifo,0o600);fd=os.open(fifo,os.O_RDWR|os.O_NONBLOCK)
        command=['gnome-terminal','--window','--title=DiT G10 quantization recovery v3 2700s',
            '--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
        c.save('gnome_launch_request.json',dict(time=c.now(),command=command,single_launch=True))
        try:
            run=subprocess.run(command,env={**os.environ,**c.DESKTOP},capture_output=True,text=True,timeout=15)
            c.save('gnome_launch_receipt.json',dict(time=c.now(),command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
            assert run.returncode==0,run.stderr
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ)
            ready=selector.select(25);message=os.read(fd,65536).decode() if ready else '';selector.close()
            assert message,'GNOME accepted but no actual-process acknowledgement; no background fallback'
            receipt=json.loads(message.splitlines()[0]);assert receipt['actual_start'] and receipt['output_dir']==str(OUT)
            ack=dict(time=c.now(),actual_start=receipt,one_startup_ack_only=True,no_postlaunch_polling=True)
            c.save('launch_startup_acknowledgement.json',ack);c.record('GNOME_v3_actual_start_acknowledged',ack)
            assert c.read(OUT/'launch_startup_acknowledgement.json')==ack
            c.save('launch_verified.json',dict(time=c.now(),pid=receipt['pid'],startup_receipt_readback=True,
                root_and_local_TEST_LOG_recorded=True,no_postlaunch_polling=True))
            print('ACTUAL_START_RECEIPT '+json.dumps(ack),flush=True)
        except BaseException as err:
            c.save('launch_failure.json',dict(time=c.now(),error=repr(err),no_background_fallback=True));raise
        finally:os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','launch','run','collect','build','train','evaluate'))
    p.add_argument('--candidate',choices=('candidate1','candidate2'));p.add_argument('--repair',action='store_true')
    p.add_argument('--kind',choices=c.KINDS);p.add_argument('--index',type=int,choices=(0,2,6,8,10));p.add_argument('--branch')
    a=p.parse_args()
    assert OUT.name=='dit_quantization_recovery_v3'
    if a.mode=='preflight':preflight()
    elif a.mode=='launch':launch()
    elif a.mode=='run':execute()
    elif a.mode=='collect':c.collect(a.branch)
    elif a.mode=='build':__import__('data_pipeline').build()
    elif a.mode=='train':train(a.candidate,a.repair)
    elif a.mode=='evaluate':c.evaluate(a.candidate,a.kind,a.index)

if __name__=='__main__':main()
