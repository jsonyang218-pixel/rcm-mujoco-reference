"""Bounded matched evaluation, offline pairing, and diagnostic teacher replay."""
import argparse, json, os, sys, time, traceback, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy')
sys.path.insert(0,str(ROOT))
INDICES=(0,2,6,8,10)
MODELS={'baseline':ROOT/'outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth',
 'v5_A4000':ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'}

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8');temp.replace(path)

def record(stage,status,out,details):
 with (ROOT/'TEST_LOG.md').open('a',encoding='utf-8') as f:
  f.write(f'\n- {datetime.now().astimezone().isoformat()}: terminal_round_v1 {stage} {status}; config={out}/card.json; result={details}; output={out}. MuJoCo only.\n')

def read_episode(path):return json.loads(Path(path).read_text())['episodes'][0]

def hold(errors):
 best=run=0
 for e in errors:
  run=run+1 if e<=20 else 0;best=max(best,run)
 return best

def audit(ep):
 tr=ep['trajectory'];err=np.asarray(tr['errors'])*1000
 octs=np.asarray(tr['oct_navigation_mm']);n=len(err)
 radial=np.r_[0,np.diff(err)/.05]
 near=np.flatnonzero((err<=100)&np.asarray(tr['oct_navigation_valid']))
 rows=[];pairs=[]
 for i in near:
  direction='approaching' if radial[i]<-5 else 'receding' if radial[i]>5 else 'low_radial_speed'
  rows.append({'step':int(i),'error_um':float(err[i]),'radial_speed_um_s':float(radial[i]),'regime':direction,
   'velocity_abc':tr['velocities'][i],'oct_mm':tr['oct_navigation_mm'][i],
   'previous_accepted_action':tr['actions'][i-1] if i else [0,0,0],
   'policy_raw_action':tr['raw_actions'][i],'accepted_action':tr['actions'][i]})
 # Match vector as well as distance; scalar distance alone confounds direction.
 approaching=near[radial[near]<-5];receding=near[radial[near]>5]
 for i in approaching[::10]:
  if not len(receding):break
  candidates=receding[(np.abs(err[receding]-err[i])<=5)&(np.abs(receding-i)>=10)]
  candidates=[int(j) for j in candidates if np.dot(octs[i],octs[j])/(np.linalg.norm(octs[i])*np.linalg.norm(octs[j])+1e-15)>=.95]
  if candidates:
   j=min(candidates,key=lambda j:np.linalg.norm(octs[i]-octs[j]))
   pairs.append({'approaching_step':int(i),'receding_step':j,'distance_difference_um':float(abs(err[i]-err[j])),'oct_vector_difference_um':float(np.linalg.norm(octs[i]-octs[j])*1000)})
  if len(pairs)>=20:break
 return {'rows':rows,'matched_pairs':pairs,'final_um':float(ep['final_error_mm']*1000),
  'max_hold_cycles':hold(np.r_[err,ep['final_error_mm']*1000]),
  'classification_limit':'Radial categories are kinematic diagnostics; low radial speed may include lateral motion; no braking correctness or label fault is inferred from action sign.'}

def teacher_replay(ep,index,out,steps=400):
 from rcm.scene_v9 import RCMSceneV9
 from scripts.research.homog_replay_common import make_teacher
 unit=np.array([np.pi/180,np.pi/180,.001])
 scene=RCMSceneV9();teacher=None
 try:
  scene.set_target_world(np.asarray(ep['target_world_mm']))
  scene.set_task(np.asarray(ep['trajectory']['states'][0])*unit)
  scene.enable_motor_execution();scene.reset_motor_time(0.)
  motor=scene.motor_execution
  for expected in ep['motor_records'][:index]:
   prepared=motor.preview(np.asarray(expected['requested_task_rad_rad_m']),.05)
   for key in ['requested_robot_ABCE','quantized_target_robot_ABCE','actual_after_robot_ABCE']:
    if prepared.record[key]!=expected[key]:raise ValueError(f'prefix mismatch {key} at {expected["step_index"]}')
   scene.apply_prepared_motor(prepared)
  np.testing.assert_allclose(scene.current_task(),np.asarray(ep['trajectory']['states'][index])*unit,atol=1e-10,rtol=0)
  np.testing.assert_allclose(motor.actual_velocity,np.asarray(ep['trajectory']['velocities'][index])*unit,atol=1e-10,rtol=0)
  teacher=make_teacher(scene);rows=[];first_labels=None
  for step in range(steps):
   teacher.sensor.observation_time_s=motor.time_s;obs=teacher.sensor.observe();q=scene.current_task().copy()
   action=teacher.geometry_action_from_observation(q,obs,previous_velocity=motor.actual_velocity.copy(),command_q=q)
   if not action['valid']:raise ValueError('teacher observation/action invalid')
   if step==0:
    zero=teacher.geometry_action_from_observation(q,obs,previous_velocity=np.zeros(3),command_q=q)
    first_labels={'actual_velocity_teacher':action,'zero_velocity_teacher':zero,
     'source_policy_raw_action':ep['trajectory']['raw_actions'][index],
     'label_limit':'Two teacher conventions are probed; neither is asserted to be the historical training label.'}
   increment=np.clip(np.asarray(action['action_deg_deg_mm']),[-.2,-.1,-.03],[.2,.1,.03])
   prepared=motor.preview(q+increment*unit,.05)
   scene.apply_prepared_motor(prepared)
   rows.append({'step':step,'true_error_um':float(np.linalg.norm(scene.site_mm('needle_tip')-np.asarray(ep['target_world_mm']))*1000),
    'action':increment.tolist(),'actual_velocity_si':motor.actual_velocity.tolist(),'motor_record':motor.last_record})
  result={'status':'completed','diagnostic_teacher_only':True,'policy_success_evidence':False,'prefix_exact':True,
   'source_step':index,'first_labels':first_labels,'final_um':rows[-1]['true_error_um'],
   'max_hold_cycles':hold([r['true_error_um'] for r in rows]),'trajectory':rows}
  save(out,result);return {k:v for k,v in result.items() if k not in ('trajectory','first_labels')}
 finally:
  if teacher is not None:teacher.sensor.close()
  scene.close()

def preflight(out):
 from scripts.research.homog_replay_common import identity,gpu_check
 for path in MODELS.values():
  if not path.is_file():raise FileNotFoundError(path)
 return {'status':'passed','gpu':gpu_check(),'identity':identity(),
  'checkpoint_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in MODELS.items()},
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
 out=args.output_dir
 if args.preflight_only:
  print(json.dumps(preflight(out)));return
 if out.exists():raise FileExistsError('Refusing to overwrite run')
 out.mkdir(parents=True);status={'status':'running','stage':'preflight'}
 save(out/'card.json',{'scope':'matched baseline + terminal audit + diagnostic teacher replay; no training or data collection',
  'models':{k:str(p) for k,p in MODELS.items()},'targets':list(INDICES),'seed':'42+index','steps':1600,'period_s':.05,
  'samples':1,'exec_steps':1,'flow_steps':4,'geometric_takeover':False,'hold_threshold_um':20,'hold_cycles':200,
  'legacy_strict_threshold_um':.35,'workers':5,'timeout_minutes':90,'teacher_replays':'E06/E10 per model, one selected state each, 400 steps; diagnostic only',
  'selection':'first receding frame at <=100um, else minimum-error pre-action frame; missing eligible states skipped',
  'stopping':'No retries; execution failure stops dependent work; no training or dev20 in this batch'})
 try:
  receipt=preflight(out);save(out/'preflight.json',receipt);record('preflight','passed',out,receipt['checkpoint_sha256'])
  from scripts.research.homog_replay_evaluate import evaluate
  results={}
  for name,checkpoint in MODELS.items():
   status['stage']='baseline_'+name;save(out/'workflow_status.json',status)
   summary=evaluate(checkpoint,1,'feedback',INDICES,out/name,{'identity':receipt['identity']},workers=5,max_workers=5,action_aggregation='median')
   results[name]={'final_mean_um':summary['final_mean_um'],'targets':{}}
   record(name,'completed',out,results[name]['final_mean_um'])
  status['stage']='terminal_audit';save(out/'workflow_status.json',status)
  selected=[]
  for name in MODELS:
   for i in INDICES:
    path=out/name/'shards/tasks'/f'eval_E{i:02d}/raw/evaluation_results.json';ep=read_episode(path);diagnosis=audit(ep)
    save(out/'audit'/name/f'E{i:02d}.json',diagnosis)
    results[name]['targets'][f'E{i:02d}']={k:diagnosis[k] for k in ('final_um','max_hold_cycles')}
    if i in (6,10) and diagnosis['rows']:
     receding=[r for r in diagnosis['rows'] if r['regime']=='receding']
     chosen=receding[0] if receding else min(diagnosis['rows'],key=lambda r:r['error_um'])
     selected.append((name,i,path,chosen['step']))
  save(out/'audit_summary.json',results);record('terminal_audit','completed',out,{'selected_replays':len(selected)})
  status['stage']='teacher_replay';save(out/'workflow_status.json',status)
  replays=[]
  for name,i,path,step in selected:
   destination=out/'teacher_replay'/name/f'E{i:02d}.json'
   result=teacher_replay(read_episode(path),step,destination);replays.append({'model':name,'target':i,**result})
   record('teacher_replay','completed',out,replays[-1]);print(json.dumps(replays[-1]),flush=True)
  save(out/'teacher_summary.json',replays)
  status.update(status='completed',stage='report',models=results,teacher_replays=replays)
  record('workflow','completed',out,'See workflow_status.json; no training performed')
 except BaseException as error:
  status.update(status='failed',error=str(error),traceback=traceback.format_exc());record(status['stage'],'failed',out,str(error));raise
 finally:save(out/'workflow_status.json',status)

if __name__=='__main__':main()
