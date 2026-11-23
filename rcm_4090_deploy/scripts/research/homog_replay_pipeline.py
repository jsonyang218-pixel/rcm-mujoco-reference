"""Execute 规划5.txt once in GNOME Terminal with immutable gates and receipts."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_replay_common import (
    ABSOLUTE_DRIFT_KEY, BASE, BRANCH, CAPS, DATA, FAST5, HOLD_START, LATE_DRIFT_START, OLD, PUB, QUALITY, REPORTS,
    ROOT, SOURCE_DATA, STEPS, TARGETS, TASK, TEACHER_TOLERANCE_UM, UNIT, VAL,
    WEIGHTS, WORKERS, check_identity, dev20_gate, direction_gate, env_command, gpu_check,
    identity, make_teacher, pool, read, record, signature, write_json,
)

MODULE='scripts.research.homog_replay_pipeline'


def card():
    return {'schema':1,'task_file':str(TASK),'task_sha256':file_sha256(TASK),
        'base':BASE,'branch':BRANCH,
        'hypothesis':'Same-observation RGB/OCT/zero-velocity ABC replay reduces direction bias and final error',
        'main_change':'60 fresh same-render600-step trajectories at8%;12GPU workers; bounded fractional teacher intent; frozen model architecture/statistics',
        'repair_authorization':'2026-11-23 user: 修改这个问题; repair the diagnosed integer-handle deadzone',
        'budget_authorization':'2026-11-23 user:600步骤,12路并行; applies to collection and matched closed-loop evaluation; training remains4000batches',
        'original_pool':'Original episodes0-838 of merged dataset, full valid trajectories; no Plan-2 windows or old overlays',
        'data_source':str(SOURCE_DATA),'validation':str(VAL),'initial_checkpoint':str(OLD),
        'training':{'epochs':4,'batches_per_epoch':1000,'batch_size':16,
            'dagger_repeat':1,'new_replay_ratio':.08,'amp':True,
            'backbone_lr':1e-6,'attention_lr':1e-5,'save_steps':[1000,2000,3000,4000]},
        'collection':{'T1':20,'T2':20,'T3':20,'seeds':list(range(1000,1060)),
            'steps_per_trajectory':STEPS,'teacher_tolerance_um':TEACHER_TOLERANCE_UM,
            'T3_hold_start':HOLD_START,'T3_hold_frames':200,
            'label':'Zero-velocity geometry pre-acceleration expectation, capped at[.2deg,.1deg,.03mm] per50ms',
            'expert_execution':'StrictRGBD geometry teacher with actual previous velocity; T2/T3 bounded fractional intent committed only after accepted commands',
            'hold_execution':'Discard fractional intent on each explicit hold tick; command actual feedback pose without bypassing motor dynamics',
            'reuse':'No1600-step trajectory reuse or truncation; recollect all60trajectories at600steps in a new destination',
            'T2_sign_balance':'10 antithetic ABC sign pairs over20 isolated targets'},
        'quality_gates':{'random_frames':200,'oct_truth_median_um_lt':.2,
            'all_axes_reverse_fraction_lt':.02,'RGB_OCT_render_provenance_mismatch':False,
            'valid_window_contract_pass_fraction':1.},
        'evaluation':{'tier':'user_authorized_fast5_then_conditional_dev20','fast5':list(FAST5),
            'seeds':[42+i for i in FAST5],'steps':STEPS,'control_period_s':.05,
            'late_drift_start':LATE_DRIFT_START,
            'late_drift_definition':'Final post600-action error minus pre-action trace[400]; absolute mean gate<15um',
            'driver':'g10','quality':QUALITY,'flow_steps':4,'stop_on_success':False,
            'strict_success':'Diagnostic only, <0.35um and3 encoder-settled frames',
            'baseline':'Fresh task2_step4000 fast5 with identical current code/scene/protocol',
            'snapshot':'9 frozen dad76c5 states,32 explicit parallel CUDA seeds each',
            'direction_gate':'A_reverse<=2/9 AND toward>70% AND absolute_late_drift<15um',
            'optional_arms':'Selected V0 finalmean<paired old baseline OR all3 direction thresholds pass',
            'dev20_gate':'Best trained combination finalmean<35um AND absolute_late_drift<15um',
            'teacher_context_um':1.70,'teacher_context':'Historical approximate reference, not a fresh matched baseline'},
        'budgets':dict.fromkeys(('collection_timeout_s','training_timeout_s','episode_timeout_s',
                                'stage_timeout_s','initializer_timeout_s'),None),
        'gpu_parallelism':{'workers':WORKERS,'max_workers':WORKERS,
            'admission_min_gpu_free_mib':10240,'admission_min_ram_available_mib':4096,
            'training':'One CUDA/AMP trainer, original batch16'},
        'paths':{'dataset':str(DATA),'checkpoints':str(WEIGHTS),
            'reports':[str(p) for p in REPORTS]},
        'resume':'Only complete matching per-episode receipts reused; incomplete shards archived; training state every100 batches',
        'failure':'No automatic retry, parameter changes or dependent stages after failed gates',
        'simulation_scope':'Software/MuJoCo only; no clinical accuracy or hardware readiness claim'}


def preflight(output,runtime=False):
    from scripts.research.homog_replay_collect import collection_tasks,expected_label
    from scripts.research.a_axis_flip_audit import static_scene,snapshot_tasks
    from scripts.research.a_axis_flip_replay import windows
    from scripts.research.clean_oct_workers import egl_identity
    from lerobot_training.dit_experience_replay import _load_policy
    import torch
    gpu=gpu_check()
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('GNOME desktop display unavailable; no background fallback')
    if not Path('/usr/bin/gnome-terminal').exists():
        raise FileNotFoundError('gnome-terminal is required')
    info=read(PUB/'branch.json')
    if info['base']!=BASE or info['branch']!=BRANCH:
        raise ValueError('New branch has wrong base/name')
    repository=info['repository']
    if subprocess.check_output(['git','-C',repository,'merge-base',BRANCH,BASE],text=True).strip()!=BASE:
        raise ValueError('Branch does not descend from71040dd')
    assets={str(p):file_sha256(p) for p in (TASK,OLD,TARGETS,
        SOURCE_DATA/'data/dataset.npz',SOURCE_DATA/'metadata.json',SOURCE_DATA/'data/image_paths.json',
        VAL/'data/dataset.npz',VAL/'metadata.json',VAL/'data/image_paths.json')}
    for directory in (SOURCE_DATA,VAL):
        paths=read(directory/'data/image_paths.json')
        missing=[p for p in paths if not (directory/p.replace('\\','/')).is_file()]
        if missing: raise FileNotFoundError(f'Missing images: {directory}: {missing[:2]}')
    with np.load(SOURCE_DATA/'data/dataset.npz',allow_pickle=False) as f:
        a={k:f[k].copy() for k in f.files}
    indices,diagnostics=windows(a)
    original=indices[a['episode_index'][indices]<839]
    if len(original)==0: raise ValueError('Empty original complete-trajectory pool')
    tasks=collection_tasks()
    scene=static_scene()
    teacher=None
    sensor_checks=[]
    try:
        for i in (0,20,40):
            task=tasks[i]
            scene.set_target_world(np.asarray(task['target_world_mm']))
            q=np.asarray(task['initial_state_si']).copy()
            if i==0:
                q=np.asarray(task['goal_state_si']).copy(); q[2]-=.00015
            scene.set_task(q)
            teacher=make_teacher(scene)
            teacher.sensor.observation_time_s=0.
            obs=teacher.sensor.observe()
            label=expected_label(teacher,scene,q,obs)
            if label is None: raise ValueError(f'Actual strictRGBD preflight invalid forT{i//20+1}')
            nav=obs.get('oct_navigation_depth')
            if nav is None or not np.isfinite(nav['delta_xyz_m']).all():
                raise ValueError('Preflight invalid OCT')
            sensor_checks.append({'regime':task['regime'],'label':label['action'].tolist(),
                'measured_error_um':label['measured_error_um'],'egl':egl_identity(),
                'quality':QUALITY})
            teacher.sensor.close(); teacher=None
    finally:
        if teacher is not None: teacher.sensor.close()
        scene.close()
    model,config,epoch=_load_policy(OLD,torch.device('cuda'))
    if config.get('n_obs_steps')!=2 or config.get('n_action_steps')!=16:
        raise ValueError('Checkpoint observation/action architecture mismatch')
    architecture={'architecture':getattr(model,'architecture',config.get('architecture')),
        'policy_type':model.policy_type,'epoch':epoch,'n_obs_steps':2,'n_action_steps':16}
    del model; gc.collect(); torch.cuda.empty_cache()
    snapshots=snapshot_tasks()
    manifest={'card':card(),'assets_sha256':assets,'identity':identity(),
        'collection_tasks':tasks,'snapshot_tasks':snapshots,
        'original_windows':len(original),'original_window_rows_sha256':signature(original.tolist()),
        'source_window_diagnostics':diagnostics}
    path=output/'execution_manifest.json'
    if path.exists():
        if read(path)!=manifest:
            raise ValueError('Frozen execution manifest differs; no mixed resume')
    else:
        for p in (DATA,WEIGHTS,*REPORTS):
            if p.exists(): raise FileExistsError(f'Unregistered occupied destination: {p}')
        write_json(path,manifest)
    result={'status':'passed','gpu':gpu,'architecture':architecture,
        'original_windows':len(original),'collection_trajectories':60,
        'real_render_sensor_checks':sensor_checks,'no_wall_clock_limits':True,
        'manifest_sha256':signature(manifest)}
    write_json(output/('runtime_preflight.json' if runtime else 'preflight.json'),result)
    record('preflight','completed',output,result)
    return manifest


def completed_stage(output,name,fn):
    path=output/f'{name}_receipt.json'
    if path.exists():
        receipt=read(path)
        if any(not Path(p).is_file() or file_sha256(p)!=sha for p,sha in receipt['files'].items()):
            raise ValueError(f'Completed{name} evidence changed')
        return receipt['result']
    result=fn()
    directory=REPORTS[int(name[-1])-1]
    files={str(p):file_sha256(p) for p in directory.rglob('*') if p.is_file()
        and p.suffix in ('.json','.npz','.png','.csv','.md')}
    if name=='stage1':
        files.update({str(p):file_sha256(p) for p in DATA.rglob('*')
            if p.is_file() and p.suffix in ('.json','.npz','.png','.jsonl','.log')})
    if name=='stage2':
        files.update({str(p):file_sha256(p) for p in WEIGHTS.rglob('*')
            if p.is_file() and p.suffix in ('.pth','.json','.jsonl')})
    write_json(path,{'result':result,'files':files})
    record(name,result['status'],directory,result)
    return result


def stage2(replay,manifest):
    from scripts.research.homog_replay_train import train
    destination=REPORTS[1]
    destination.mkdir(parents=True,exist_ok=True)
    trained=train(replay,WEIGHTS,manifest['identity'])
    result={'status':'completed','training':trained,'checkpoint_dir':str(WEIGHTS)}
    write_json(destination/'summary.json',result)
    return result


def plots(result,destination):
    from matplotlib import pyplot as plt
    rows=result['episodes']
    fig,ax=plt.subplots(figsize=(12,5))
    for r in rows:
        ep=read(r['result_path'])['episodes'][0]
        errors=np.r_[np.asarray(ep['trajectory']['errors'])*1000,r['final_error_um']]
        ax.plot(np.arange(len(errors)),errors,label=f"E{r['episode_index']:02d}")
    ax.set(xlabel='Control step (50 ms)',ylabel='Tip error (um)',yscale='log')
    ax.legend(ncol=5,fontsize=8); fig.tight_layout()
    fig.savefig(destination/'error_curves_all.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    x=np.arange(len(rows))
    ax.bar(x-.2,[r['min_error_um'] for r in rows],.4,label='Minimum')
    ax.bar(x+.2,[r['final_error_um'] for r in rows],.4,label='Final')
    ax.set(xticks=x,xticklabels=[f"E{r['episode_index']:02d}" for r in rows],ylabel='Tip error (um)')
    ax.legend(); fig.tight_layout(); fig.savefig(destination/'best_and_final_error.png',dpi=150); plt.close(fig)
    with (destination/'episodes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Actual +/-one-handle protocol displacement, at each measured final pose.
    from scripts.research.a_axis_flip_audit import static_scene
    scene=static_scene()
    floors=[]
    try:
        from rcm.visual_servo import VisualServoController
        from motor_backend import MotorBackend
        from driver_loader import driver_identity
        geometry=VisualServoController.__new__(VisualServoController);geometry.reference=scene
        for r in rows:
            ep=read(r['result_path'])['episodes'][0]
            last=ep['motor_records'][-1]
            q=np.asarray(last['actual_after_task_rad_rad_m'])
            scene.set_task(q)
            origin=scene.site_mm('needle_tip').copy()
            robot=tuple(int(v) for v in last['actual_after_robot_ABCE'])
            displacements={}
            for axis,name in enumerate('ABC'):
                displacements[name]={}
                for sign in (-1,1):
                    backend=MotorBackend(initial_robot=robot,needle_profile='legacy')
                    delta=[0,0,0,0];delta[axis]=sign
                    try:
                        command=backend.kin.make_control_message(tuple(delta),'hold')
                        receipt=backend.consume(command,0.)
                        target=scene.task_from_encoders(receipt['target'][:3],reference_task=q)
                        scene.set_task(target)
                        endpoint=float(np.linalg.norm(scene.site_mm('needle_tip')-origin)*1000)
                        for tick in range(25): backend.step(.002,(tick+1)*.002)
                        actual=scene.task_from_encoders(backend.robot[:3],reference_task=q)
                        scene.set_task(actual)
                        after50=float(np.linalg.norm(scene.site_mm('needle_tip')-origin)*1000)
                        displacements[name][str(sign)]={'endpoint_tip_displacement_um':endpoint,
                            'after_one_50ms_packet_tip_displacement_um':after50,
                            'target_robot_ABCE':list(receipt['target'])}
                    except ValueError as error:
                        displacements[name][str(sign)]={'unavailable_at_pose':str(error)}
                    finally:
                        scene.set_task(q)
            floors.append({'episode_index':r['episode_index'],'state_si':q.tolist(),
                'driver':driver_identity(),'one_handle_steps':displacements,
                'local_jacobian_mm_per_task_si':geometry._model_jacobian(q).tolist()})
            # New scenes avoid advancing motor state; no commands are committed here.
    finally:
        scene.close()
    write_json(destination/'axis_floor_summary.json',{'protocol':'G10 50ms physical backend',
        'scope':'Final-pose +/-one-handle endpoint and50ms displacements; local protocol floor, no global/hardware precision claim',
        'episodes':floors})


def evaluate(checkpoint,samples,controller,indices,destination,manifest):
    from scripts.research.homog_replay_evaluate import evaluate as fixed_evaluate
    result=fixed_evaluate(checkpoint,samples,controller,indices,destination,manifest)
    plots(result,destination)
    result['hit_rates']={f'le_{t}_um':result[f'final_hits_le_{t}_um']/len(indices) for t in (5,10,30)}
    write_json(destination/'summary.json',result)
    record(f'paired_fixed{STEPS}_evaluation','completed',destination,
        {'checkpoint':str(checkpoint),'target_split':str(TARGETS),'indices':list(indices),
         'seeds':[42+i for i in indices],'steps':STEPS,
         **{k:result[k] for k in ('final_mean_um','min_mean_um',ABSOLUTE_DRIFT_KEY,'strict_success_count')}})
    return result


def stage3(trained,manifest):
    from scripts.research.a_axis_flip_audit import snapshots
    destination=REPORTS[2];destination.mkdir(parents=True,exist_ok=True)
    baseline=evaluate(OLD,1,'feedback',FAST5,destination/'baseline_task2_step4000',manifest)
    comparisons={f'V0_step{n}':evaluate(WEIGHTS/f'step_{n:06d}.pth',1,'feedback',FAST5,
        destination/f'V0_step{n}',manifest) for n in (2000,4000)}
    selected=min(comparisons,key=lambda k:comparisons[k]['final_mean_um'])
    checkpoint=comparisons[selected]['protocol']['checkpoint']
    snap_dir=destination/'selected_snapshot_recheck';snap_dir.mkdir(exist_ok=True)
    snap=snapshots({'BEST_CKPT':checkpoint},snap_dir,manifest['identity'])['BEST_CKPT']
    direction=direction_gate(snap,comparisons[selected])
    improved=comparisons[selected]['final_mean_um']<baseline['final_mean_um']
    if improved or direction:
        comparisons['I_med5']=evaluate(checkpoint,5,'feedback',FAST5,destination/'I_med5',manifest)
        comparisons['V3']=evaluate(checkpoint,1,'continuous',FAST5,destination/'V3',manifest)
    best_name=min(comparisons,key=lambda k:comparisons[k]['final_mean_um'])
    best=comparisons[best_name]
    result={'status':'completed','baseline':baseline,'comparisons':comparisons,
        'selected_V0':selected,'best_checkpoint':checkpoint,'snapshot_recheck':snap,
        'direction_gate_passed':direction,'paired_final_error_improved':improved,
        'optional_arms_status':'completed' if (improved or direction) else 'skipped_trigger_failed',
        'best_combination':best_name,'best':best,'dev20_gate_passed':dev20_gate(best),
        'paired_best_minus_baseline_um':best['final_mean_um']-baseline['final_mean_um']}
    write_json(destination/'summary.json',result)
    for name in ('error_curves_all.png','best_and_final_error.png','axis_floor_summary.json'):
        import shutil
        shutil.copy2(destination/best_name/name,destination/name)
    return result


def stage4(selected,manifest):
    if not selected['dev20_gate_passed']:
        raise ValueError('Development20 gate failed')
    destination=REPORTS[3];destination.mkdir(parents=True,exist_ok=True)
    p=selected['best']['protocol']
    result=evaluate(p['checkpoint'],p['samples'],p['controller'],tuple(range(20)),destination,manifest)
    result['excess_over_historical_teacher_approx_um']=result['final_mean_um']-1.70
    result['teacher_reference']='Historical approximate G10 visual teacher1.70um; not a fresh paired comparison'
    write_json(destination/'summary.json',result)
    return result


def summary(output,stages,status):
    lines=['# 同源新渲染近目标回放 — 2026-11-23','',f'流程状态：{status}。','',
        f'基点 `{BASE}`；独立分支 `{BRANCH}`。','',
        '全部阶段、训练、初始化及单episode均无wall-clock时长限制；0.35µm只用于诊断，不提前停止。','',
        '92%池仅使用原始episode0–838完整有效轨迹窗口，排除全部Plan-2窗口；8%为本轮同源新包。','']
    for stage,result in stages.items():
        brief={k:v for k,v in result.items() if k in ('status','reason','direction_gate_passed',
            'dev20_gate_passed','best_combination','paired_final_error_improved')}
        lines += [f'## {stage}','','```json',json.dumps(brief,ensure_ascii=False,indent=2),'```','']
    if 'stage3' in stages and stages['stage3']['status']=='completed':
        s=stages['stage3'];b=s['best']
        lines += [f"配对原模型终值均值：{s['baseline']['final_mean_um']:.6f}µm；本轮最优：{b['final_mean_um']:.6f}µm。",'',
            f"最小误差均值：{b['min_mean_um']:.6f}µm；{LATE_DRIFT_START}→{STEPS}步绝对漂移均值：{b[ABSOLUTE_DRIFT_KEY]:.6f}µm。",'',
            f"严格成功：{b['strict_success_count']}/{b['completed_targets']}。流程完成、指标改善、严格成功分别报告。",'']
    lines += ['结构化报告目录：'+', '.join(f'`{p.relative_to(ROOT)}`' for p in REPORTS)+'。','',
        '权重、优化器断点及原始纠偏RGB保留本机；发布路径/大小/SHA256清单。代码、配置、日志及结构化证据上传GitHub。','',
        '仅验证软件与MuJoCo仿真，不代表临床精度或硬件就绪。']
    text='\n'.join(lines)+'\n'
    (output/'SUMMARY.md').write_text(text)
    (PUB/'SUMMARY.md').write_text(text)


def run(output):
    # Scope the authorized worker setting to this workflow without
    # rewriting the older audit/evaluation modules or their frozen outputs.
    from scripts.research import a_axis_flip_audit, a_axis_flip_pipeline
    a_axis_flip_audit.pool = a_axis_flip_pipeline.pool = pool
    with (output/'workflow.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Workflow already running') from error
        stages={}
        status='failed'
        try:
            manifest=preflight(output,runtime=True)
            def interrupted(sig,_): raise KeyboardInterrupt(signal.Signals(sig).name)
            for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)
            from scripts.research.homog_replay_collect import collect
            for name,fn in [('stage1',lambda:collect(manifest)),
                ('stage2',lambda:stage2(stages['stage1'],manifest)),
                ('stage3',lambda:stage3(stages['stage2'],manifest)),
                ('stage4',lambda:stage4(stages['stage3'],manifest))]:
                if name!='stage1' and not stages['stage1']['audit']['gate_passed']:
                    stages[name]={'status':'skipped','reason':'Stage1 data quality gate failed'}
                elif name=='stage4' and not stages['stage3']['dev20_gate_passed']:
                    stages[name]={'status':'skipped','reason':'Fast5 finalmean<35um and absolute_late_drift<15um gate failed'}
                else:
                    check_identity(manifest['identity'])
                    write_json(output/'workflow_status.json',{'status':'running','stage':name,
                        'completed_stages':list(stages)})
                    stages[name]=completed_stage(output,name,fn)
                    check_identity(manifest['identity'])
                write_json(output/'stage_results.json',stages)
                print(f'STAGE {name}: {stages[name]["status"]}; results: {REPORTS[int(name[-1])-1]}',flush=True)
            status=('completed_with_gate_failure' if stages['stage1']['status']=='gate_failed' else
                'completed_with_conditional_skips' if any(s['status']=='skipped' for s in stages.values()) else 'completed')
            write_json(output/'workflow_status.json',{'status':status,'stages':{k:v['status'] for k,v in stages.items()}})
            record('workflow',status,output,{k:v['status'] for k,v in stages.items()})
        except BaseException as error:
            status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            write_json(output/'failure.json',{'status':status,'reason':repr(error),'completed_stages':list(stages)})
            write_json(output/'workflow_status.json',{'status':status,'reason':repr(error)})
            record('workflow',status,output,f'{error!r}; partial evidence retained; no automatic retry')
            raise
        finally:
            summary(output,stages,status)
            from scripts.research.homog_replay_publish import publish
            try:
                publish(output,push=True)
            except BaseException as error:
                write_json(output/'publication_failure.json',{'status':'failed','reason':repr(error),'retry':False})
                record('publication','failed',output,repr(error))
                raise
            print(f'FINAL SUMMARY: {output/"SUMMARY.md"}',flush=True)


def main():
    os.environ['RCM_DRIVER_VARIANT']='g10'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','launch','run'))
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();output=args.output_dir.resolve()
    if not output.is_relative_to(ROOT/'runs/20261123'):
        raise ValueError('Use purpose-specific runs/20261123 destination')
    output.mkdir(parents=True,exist_ok=True)
    if args.action=='run': return run(output)
    try: preflight(output)
    except BaseException as error:
        write_json(output/'preflight_failure.json',{'reason':repr(error)})
        record('preflight','failed',output,repr(error));raise
    if args.action=='preflight':
        print(f'PREFLIGHT PASSED: {output/"preflight.json"}',flush=True);return
    receipt_path=output/'launch_receipt.json'
    if receipt_path.exists() and read(receipt_path).get('accepted'):
        raise FileExistsError('Already launched; duplicate terminal/run refused')
    command=env_command()+['-m',MODULE,'run','--output-dir',str(output)]
    log=output/'terminal.log'
    shell=('set -o pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(command)
        +' 2>&1 | tee -a '+shlex.quote(str(log))
        +'\ncode=${PIPESTATUS[0]}\nprintf "\\nExperiment exited: %s\\n" "$code"\n'
        +'read -r -p "Press Enter to close this terminal..."\nexit "$code"\n')
    result=subprocess.run(['gnome-terminal','--title=Homogeneous RGB-D replay 20261123',
        '--working-directory='+str(ROOT),'--','bash','-c',shell],capture_output=True,text=True)
    receipt={'accepted':result.returncode==0,'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr,'command':command,'log':str(log),
        'timestamp':datetime.now().astimezone().isoformat(),'no_wall_clock_limits':True}
    write_json(receipt_path,receipt);record('launch','accepted' if receipt['accepted'] else 'failed',output,receipt)
    if not receipt['accepted']: raise RuntimeError(f'GNOME Terminal failed: {result.stderr}; no background fallback')
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
