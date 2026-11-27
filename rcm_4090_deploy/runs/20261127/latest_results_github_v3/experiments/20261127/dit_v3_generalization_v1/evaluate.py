"""Frozen independent-target evaluation, original v5 + candidate2; no training entry."""
from __future__ import annotations
import argparse,os,sys,json,time,signal,traceback,hashlib,fcntl,subprocess,threading,selectors
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];OLD=ROOT/'runs/20261127/dit_quantization_recovery_v3'
sys.path[:0]=[str(OLD),str(ROOT)]
import campaign as c
c.OUT=OUT
NEAR=OLD/'candidate2/checkpoint.pth';STOP=False

def read(p):return json.loads(Path(p).read_text())
def record(stage,result):c.record(stage,result)
def verify():
 card=read(OUT/'experiment_card.json')
 for name,spec in card['fingerprints'].items():
  assert c.sha(spec['path'])==spec['sha256'],f'Frozen input drift: {name}'
 assert c.sha(OUT/'frozen_plan.json')==card['plan_sha256']
 return card

def worker(job_id,resume=False):
 card=verify();plan=read(OUT/'frozen_plan.json');job=next(j for j in plan['jobs'] if j['id']==job_id)
 folder=OUT/'tasks'/job_id;folder.mkdir(parents=True,exist_ok=resume)
 old_rows=[]
 if (folder/'summary.json').exists():
  s=read(folder/'summary.json')
  if s['status']=='completed' and s['metrics']['completed_cycles']==1600:return
 if (folder/'cycles.jsonl').exists():
  assert resume,'Existing trajectory requires explicit resume'
  old_rows=[json.loads(x) for x in (folder/'cycles.jsonl').read_text().splitlines()]
 torch=c.torch_setup();import numpy as np
 from lerobot_training.eval_policy import StrictRGBDSensor,visual_servo_settings,TASK_TO_DEG_DEG_MM,DEFAULT_ACTION_LIMIT_DEG_DEG_MM
 from scripts.research.homog_replay_common import QUALITY
 from rcm.scene_v9 import RCMSceneV9
 e=c.evaluator();far=e.model;e.model=c.evaluator().model;c.checkpoint_load(e,NEAR);local=e.model
 scene=sensor=None;rows=[];previous=np.zeros(3);cache={};pending={};active=False;sample_calls=0;expected=1600;seed=job['seed'];began=time.monotonic()
 routes=dict(original_v5_DiT_RF=0,fixed_noise_near_DiT_RF=0,original_RGB2_startup_zero=0,teacher=0,geometry=0)
 try:
  np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  scene=RCMSceneV9();world=np.asarray(scene.robot_to_world(np.asarray(job['target_robot_mm'])));scene.set_target_world(world)
  # Same as original standard evaluations: fresh RCMSceneV9, zero ABC, no target-relative positioning.
  assert np.array_equal(scene.current_task(),np.zeros(3))
  if scene.motor_execution is None:scene.enable_motor_execution();scene.reset_motor_time(0.)
  settings=visual_servo_settings(scene);settings.update(perception_backend='strict_rgbd',rgbd_render_quality=QUALITY.copy())
  sensor=StrictRGBDSensor(scene,settings);real_observe=sensor.observe
  def capture():
   t=float(scene.motor_execution.time_s)
   if cache.get('t')==t:return cache['obs'],cache['row']
   sensor.observation_time_s=t;obs=real_observe();oct4=np.zeros(4,np.float32);nav=obs.get('oct_navigation_depth');delta=None if nav is None else nav.get('delta_xyz_m')
   if delta is not None and len(delta)==3 and np.isfinite(delta).all():oct4[:3]=np.asarray(delta)*1000;oct4[3]=1
   tip=scene.site_mm('needle_tip').copy();rgb=np.asarray(obs['rgb'])
   # Exact sensor evidence for every pre/post observation, including RGB; no truth passed to policy.
   idx=int(round(t/.05));sensor_path=folder/'sensors'/f'{idx:04d}.npz';sensor_path.parent.mkdir(exist_ok=True)
   if not sensor_path.exists():np.savez_compressed(sensor_path,rgb=rgb,signed_OCT_mm=oct4)
   else:
    with np.load(sensor_path) as saved:
     assert np.array_equal(saved['rgb'],rgb) and np.array_equal(saved['signed_OCT_mm'],oct4),'Resume sensor mismatch'
   row=dict(time_s=t,state_deg_deg_mm=scene.current_task()*TASK_TO_DEG_DEG_MM,signed_OCT_mm=oct4.copy(),
    actual_velocity_deg_deg_mm_s=scene.motor_execution.actual_velocity*TASK_TO_DEG_DEG_MM,
    previous_accepted_deg_deg_mm=previous.copy(),encoder_ABCE=list(scene.motor_execution.backend.robot),
    needle_site_world_mm=tip,target_world_mm=world,true_error_xyz_um=(tip-world)*1000,true_error_um=float(np.linalg.norm(tip-world)*1000),
    sensor_path=str(sensor_path.relative_to(OUT)),sensor_sha256=c.sha(sensor_path),sensor_keys=sorted(obs.keys()))
   cache.update(t=t,obs=obs,row=row);return obs,row
  sensor.observe=lambda:capture()[0]
  initial=capture()[1]
  if not old_rows:c.save(folder/'initial_state.json',initial)
  else:record('resume_replay_started',dict(job=job_id,saved_cycles=len(old_rows),method='deterministic same-seed replay from standard zero start; exact sensor/action/encoder checks; append only new cycles'))
  original_action=e.get_action;original_accepted=e.record_applied_action
  def action(rgb,state,oct4):
   nonlocal active,sample_calls
   if STOP or time.monotonic()-began>card['worker_timeout_s']:raise TimeoutError('Interrupted/time budget; trajectory retained for explicit resume')
   _,pre=capture();norm=float(np.linalg.norm(oct4[:3])) if oct4[3]>.5 else None
   if not active and norm is not None and norm<=.100:
    active=True;assert not e.action_queue
    if not (folder/'switch_receipt.json').exists():c.save(folder/'switch_receipt.json',dict(cycle=len(rows)+1,observable_OCT_norm_mm=norm,no_history_or_motor_reset=True))
   source='fixed_noise_near_DiT_RF' if active else 'original_v5_DiT_RF';e.model=local if active else far
   before=e.previous_action.copy();raw=original_action(rgb,state,oct4)
   if len(e.obs_buffer)==2:sample_calls+=1
   else:source='original_RGB2_startup_zero'
   routes[source]+=1;pending.update(pre=pre,source=source,oct_norm_mm=norm,previous_input=before,
    production_velocity=(e.obs_buffer[-1]['state']-e.obs_buffer[-2]['state']).numpy()/.05 if len(e.obs_buffer)==2 else None,
    history_state=[v['state'].tolist() for v in e.obs_buffer]);return raw
  def accepted(value):
   nonlocal previous
   original_accepted(value);previous=np.asarray(value).copy()
  def observer(event):
   _,post=capture();assert event['motor_record']['driver']['variant']=='g10'
   assert np.isclose(post['time_s']-pending['pre']['time_s'],.05,atol=1e-12,rtol=0)
   row=dict(cycle=len(rows)+1,target_id=job['target_id'],seed=seed,model_source=pending['source'],pre=pending['pre'],post=post,event=event,
    production_velocity_deg_deg_mm_s=pending['production_velocity'],previous_action_input=pending['previous_input'],input_history_state=pending['history_state'])
   if len(rows)<len(old_rows):
    old=old_rows[len(rows)];assert old['model_source']==row['model_source']
    np.testing.assert_array_equal(old['event']['raw_action_deg_deg_mm'],event['raw_action_deg_deg_mm'])
    np.testing.assert_array_equal(old['post']['needle_site_world_mm'],post['needle_site_world_mm'])
    assert old['post']['encoder_ABCE']==post['encoder_ABCE']
   else:c.append(folder/'cycles.jsonl',row)
   rows.append(row)
   if len(rows)%80==0:c.save(folder/'progress.json',dict(job=job_id,cycles=len(rows),expected=expected,time=c.now(),pid=os.getpid()))
  e.get_action=action;e.record_applied_action=accepted;e.step_observer=observer
  ep=e.evaluate_episode(scene,world,max_steps=expected,error_threshold_mm=.020,control_period_s=.05,
   action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,max_consecutive_safety_limited=None,
   early_stop_error_mm=30.,seed=seed,sensor=sensor,stop_on_success=False,progress_every=80)
  assert len(rows)==expected and ep['early_stop_reason'] is None,ep.get('early_stop_reason')
  assert sample_calls==1599
  metrics=c.score([r['post']['true_error_um'] for r in rows],expected)
  errs=np.array([r['post']['true_error_um'] for r in rows]);metrics.update(endpoint_le20um=bool(errs[-1]<=20),last200_le20_fraction=float(np.mean(errs[-200:]<=20)),
   first_terminal_hold_start_post_cycle=expected-metrics['terminal_le20_cycles']+1 if metrics['terminal_le20_cycles'] else None,
   terminal_hold_s=metrics['terminal_le20_cycles']*.05,longest_hold_s=metrics['longest_le20_cycles']*.05)
  invalid=sum(r['post']['signed_OCT_mm'][3]<.5 for r in rows)
  result=dict(status='completed',job=job,metrics=metrics,route_counts=routes,post_invalid_sensor_cycles=int(invalid),
   no_valid_sensor=all(r['pre']['signed_OCT_mm'][3]<.5 for r in rows),expected_cycles=expected,model_checkpoint_sha256=c.sha(NEAR),far_checkpoint_sha256=c.sha(c.REF),
   truth_setup_scoring_only=True,eval_updates=0,simulation_only=True,elapsed_s=time.monotonic()-began)
  c.save(folder/'summary.json',result);record('independent_job_completed',result)
 except BaseException as err:
  failure=dict(status='incomplete_or_failure',job=job,error=repr(err),traceback=traceback.format_exc(),saved_cycles=max(len(rows),len(old_rows)),
   partial_metrics=c.score([r['post']['true_error_um'] for r in rows],expected),included_in_frozen_denominator=True,time=c.now())
  c.save(folder/f'failure_{time.time_ns()}.json',failure);record('independent_job_failed_or_interrupted',failure);raise
 finally:
  c.save(folder/'weights_after.json',dict(time=c.now(),near_sha256=c.sha(NEAR),far_sha256=c.sha(c.REF),eval_updates=0,bit_exact=c.sha(NEAR)==card['fingerprints']['near']['sha256'] and c.sha(c.REF)==card['fingerprints']['far']['sha256']))
  if sensor is not None:sensor.close()
  if scene is not None:scene.close()

def summarize():
 import numpy as np,csv
 card=verify();jobs=read(OUT/'frozen_plan.json')['jobs'];allrows=[]
 for j in jobs:
  p=OUT/'tasks'/j['id'];s=read(p/'summary.json') if (p/'summary.json').exists() else {};m=s.get('metrics',{})
  allrows.append(dict(job_id=j['id'],target_id=j['target_id'],seed=j['seed'],status=s.get('status','missing_or_incomplete'),**m,no_valid_sensor=s.get('no_valid_sensor'),post_invalid_sensor_cycles=s.get('post_invalid_sensor_cycles')))
 def aggregate(rows):
  complete=[r for r in rows if r['status']=='completed'];n=len(rows)
  return dict(planned=n,completed=len(complete),strict20um_pass=sum(bool(r.get('strict20um_pass')) for r in rows),strict20um_success_rate=sum(bool(r.get('strict20um_pass')) for r in rows)/n,
   endpoint_le20um_count=sum(bool(r.get('endpoint_le20um')) for r in rows),endpoint_le20um_rate=sum(bool(r.get('endpoint_le20um')) for r in rows)/n,
   endpoint_mean_um_completed=float(np.mean([r['endpoint_um'] for r in complete])) if complete else None,
   endpoint_max_um_completed=max((r['endpoint_um'] for r in complete),default=None),last200_max_um_completed=max((r['last200_max_post_error_um'] for r in complete),default=None),
   scope='All planned jobs stay in denominator; incomplete jobs are not success; means explicitly completed-only')
 pertarget={t:aggregate([r for r in allrows if r['target_id']==t]) for t in dict.fromkeys(j['target_id'] for j in jobs)}
 perseed={str(s):aggregate([r for r in allrows if r['seed']==s]) for s in dict.fromkeys(j['seed'] for j in jobs)}
 summary=dict(time=c.now(),workflow_status='complete' if all(r['status']=='completed' for r in allrows) else 'partial_or_failed',total=aggregate(allrows),per_target=pertarget,per_seed=perseed,per_target_seed=allrows,
  qualification='10 distinct spatial targets x 2 sampling seeds = 20 episodes, not 20 independent spatial targets; limited frozen simulation generalization evidence, not universal generalization; no same-target historical v3 baseline exists; not original 0.35um settled criterion',simulation_only=True)
 c.save(OUT/'summary.json',summary)
 keys=list(dict.fromkeys(k for r in allrows for k in r))
 with (OUT/'per_target_seed.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
 lines=['# DiT v3 独立目标泛化测试',json.dumps(summary['total'],ensure_ascii=False,indent=2),'','|目标|种子|状态|终点µm|末200最大µm|20µm保持通过|首次到达周期|末尾持续周期|','|---|---|---|---|---|---|---|---|']
 for r in allrows:lines.append('|'+ '|'.join(str(r.get(k,'')) for k in ['target_id','seed','status','endpoint_um','last200_max_post_error_um','strict20um_pass','first_enter20_post_cycle','terminal_le20_cycles'])+'|')
 lines+=['',summary['qualification'],'','详见 existing_target_usage.json、target_provenance.json、frozen_plan.json、experiment_card.json、tasks/*/cycles.jsonl 及 sensors/*.npz。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');record('generalization_summary',summary);return summary

def interrupted(signum,frame):
 global STOP
 STOP=True

def run(resume=False):
 verify();start=dict(time=c.now(),pid=os.getpid(),ppid=os.getppid(),output=str(OUT),resume=resume,concurrency_max=12,simulation_only=True)
 if not resume:
  with (OUT/'actual_start_receipt.json').open('x') as f:json.dump(start,f,indent=2)
  fd=os.open(OUT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK);os.write(fd,(json.dumps(start)+'\n').encode());os.close(fd)
 record('generalization_started',start)
 jobs=read(OUT/'frozen_plan.json')['jobs'];queue=[j for j in jobs if not ((OUT/'tasks'/j['id']/'summary.json').exists() and read(OUT/'tasks'/j['id']/'summary.json')['status']=='completed')];active=[];began=time.monotonic()
 try:
  while queue or active:
   if STOP or time.monotonic()-began>read(OUT/'experiment_card.json')['batch_timeout_s']:raise TimeoutError('Batch interrupted/time budget; explicit resume available')
   while queue and len(active)<12:
    j=queue.pop(0);attempt=time.time_ns();log=OUT/'logs'/f'{j["id"]}_{attempt}.log';log.parent.mkdir(exist_ok=True);f=log.open('x')
    cmd=[str(ROOT/'.venv/bin/python'),'-E','-B','-u',str(OUT/'evaluate.py'),'worker','--job',j['id']]+(['--resume'] if resume else [])
    p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**c.RUNTIME},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,bufsize=1)
    def tee(process=p,stream=f,label=j['id']):
     for line in process.stdout:stream.write(line);stream.flush();print(f'[{label}] {line}',end='',flush=True)
    t=threading.Thread(target=tee,daemon=True);t.start();active.append((j,p,f,t));c.append(OUT/'commands.jsonl',dict(time=c.now(),job=j['id'],pid=p.pid,command=cmd,log=str(log)))
   for item in active[:]:
    j,p,f,t=item
    if p.poll() is None:continue
    t.join(5);f.close();active.remove(item);record('worker_exit',dict(job=j['id'],pid=p.pid,returncode=p.returncode))
   time.sleep(.25)
 finally:
  for j,p,f,t in active:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  for j,p,f,t in active:
   try:p.wait(30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   t.join(3);f.close();record('worker_interrupted',dict(job=j['id'],pid=p.pid,returncode=p.returncode))
  summarize()

def launch():
 verify()
 with (OUT/'.launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (OUT/'gnome_launch_request.json').exists(),'GNOME launch already requested; no duplicate launch'
  os.mkfifo(OUT/'startup_ack.fifo',0o600);fd=os.open(OUT/'startup_ack.fifo',os.O_RDWR|os.O_NONBLOCK)
  cmd=['gnome-terminal','--window','--title=DiT v3 generalization 10x2 K1 exec1 1600','--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh')]
  c.save(OUT/'gnome_launch_request.json',dict(time=c.now(),command=cmd,single_launch=True))
  try:
   done=subprocess.run(cmd,env={**os.environ,**c.DESKTOP},capture_output=True,text=True,timeout=20)
   c.save(OUT/'gnome_launch_receipt.json',dict(time=c.now(),returncode=done.returncode,stdout=done.stdout,stderr=done.stderr,command=cmd));assert done.returncode==0,done.stderr
   sel=selectors.DefaultSelector();sel.register(fd,selectors.EVENT_READ);ready=sel.select(25);message=os.read(fd,65536).decode() if ready else '';sel.close()
   ack=dict(time=c.now(),actual_start=json.loads(message.splitlines()[0]) if message else None,one_startup_ack_only=True,no_postlaunch_polling=True)
   c.save(OUT/'launch_startup_acknowledgement.json',ack);record('GNOME_actual_start_acknowledged',ack)
   assert message,'No actual start acknowledgement; no background fallback'
   print('ACTUAL_START_RECEIPT '+json.dumps(ack),flush=True)
  finally:os.close(fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['launch','run','worker','summarize']);p.add_argument('--job');p.add_argument('--resume',action='store_true');a=p.parse_args()
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
 if a.mode=='launch':launch()
 elif a.mode=='run':run(a.resume)
 elif a.mode=='worker':worker(a.job,a.resume)
 else:summarize()
