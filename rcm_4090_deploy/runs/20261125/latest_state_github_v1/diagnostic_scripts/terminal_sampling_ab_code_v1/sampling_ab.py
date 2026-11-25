"""One-factor terminal-sampling comparison; immutable labels and execution."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy');sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/20261125/terminal_sampling_ab_v1'
SOURCE=ROOT/'data/homog_label_fix_20261124_v5_A_t3_w025'
BALANCED=ROOT/'data/terminal_sampling_ab_v1_balanced'
REFERENCE=ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
INDICES=(0,2,6,8,10)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=str));tmp.replace(p)
def record(stage,status,details):
 with (ROOT/'TEST_LOG.md').open('a') as f:f.write(f'\n- {datetime.now().astimezone().isoformat()}: terminal_sampling_ab_v1 {stage} {status}; command={sys.argv}; card={OUT}/card.json; result={details}; output={OUT}. MuJoCo simulation only.\n')

def command(arm,smoke=False):
 c=read(OUT/'card.json');base=c['reference_config']
 args=[str(ROOT/'.venv/bin/python'),'-E','-B','-u','-m','lerobot_training.dit_experience_replay','train']
 values={'data-dir':base['data_dir'],'val-data-dir':base['val_data_dir'],'init-from':base['init_from'],
  'correction-data-dir':str(SOURCE if arm=='A_original' else BALANCED),'original-max-episode':838,
  'output-dir':str(OUT/('smoke' if smoke else 'checkpoints')/arm),'mode':'ratio','correction-ratio':.15,
  'epochs':1 if smoke else 4,'steps-per-epoch':2 if smoke else 1000,'batch-size':16,'val-batch-size':64,
  'num-workers':4,'backbone-lr':0,'attention-lr':.00003,'lr-schedule':'constant','weight-decay':.0001,
  'seed':20261118,'validation-seed':20261119,'checkpoint-every-steps':2 if smoke else 1000,
  'log-every-steps':1 if smoke else 100,'device':'cuda'}
 for key,value in values.items():args.extend(['--'+key,str(value)])
 args.append('--require-temporal-bias-only')
 return args

def prepare():
 if OUT.exists() or BALANCED.exists():raise FileExistsError('Refusing existing output/dataset')
 OUT.mkdir(parents=True)
 try:
  from scripts.research.homog_replay_common import identity,gpu_check
  config=torch.load(REFERENCE,map_location='cpu',weights_only=False)['config']
  assert config['correction_ratio']==.15 and config['backbone_lr']==0 and config['require_temporal_bias_only']
  assert config.get('label_overlay') is None and config.get('oct_symlog') is False
  with np.load(SOURCE/'data/dataset.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
  with np.load(SOURCE/'data/label_repair_overlay.npz',allow_pickle=False) as f:overlay={k:f[k].copy() for k in f.files}
  starts=overlay['correction_window_indices'];original_weights=overlay['correction_window_weight'].copy()
  dist=np.linalg.norm(a['truth_navigation_mm'],axis=1)*1000
  dt=np.diff(a['timestamp_s']);rate=np.zeros(len(dist));adj=(a['episode_index'][1:]==a['episode_index'][:-1])&(a['source_frame_index'][1:]==a['source_frame_index'][:-1]+1)
  rate[1:]=np.divide(np.diff(dist),dt,out=np.zeros(len(dt)),where=adj&(dt>0))
  meta=read(SOURCE/'metadata.json');teacher_ids=[e['episode_index'] for e in meta['episodes'] if e['regime'] in ('T2','T3')]
  moving=np.isin(a['episode_index'][starts],teacher_ids)&(dist[starts]>20)&(dist[starts]<=100)&(rate[starts]<-5)
  cs=np.r_[0,np.cumsum(overlay['t3_stationary'].astype(int))]
  stationary=(cs[starts+16]-cs[starts])==16
  assert np.count_nonzero(moving)>0 and np.count_nonzero(stationary)>0 and not np.any(moving&stationary)
  weights=np.zeros(len(starts),np.float64);bins={}
  for name,mask in [('approach20to100',moving),('stationary16',stationary)]:
   episodes=a['episode_index'][starts[mask]];ids,counts=np.unique(episodes,return_counts=True)
   for ep,count in zip(ids,counts):weights[mask&(a['episode_index'][starts]==ep)]=.5/(len(ids)*count)
   bins[name]={'windows':int(mask.sum()),'episodes':ids.tolist(),'sampling_probability':float(weights[mask].sum())}
  np.testing.assert_allclose(weights.sum(),1,atol=1e-12)
  np.testing.assert_allclose([weights[moving].sum(),weights[stationary].sum()],[.5,.5],atol=1e-12)
  # Teacher provenance: T2/T3 execution is explicitly selected in CollectionWorker.
  teacher_rows=np.isin(a['episode_index'],teacher_ids)&~overlay['t3_stationary']
  np.testing.assert_allclose(a['action'][teacher_rows],a['executed_action'][teacher_rows],atol=1e-7,rtol=1e-6)
  targets=read(ROOT/'configs/v12_reachable_development_targets_0p35um.json')['targets']
  dev=np.asarray([t.get('reachable_target_robot_mm',t['point_mm']) for t in targets])
  train=np.asarray([e['target_robot_mm'] for e in meta['episodes']]);separation=float(np.min(np.linalg.norm(train[:,None,:]-dev[None,:,:],axis=-1))*1000)
  if separation<1:raise ValueError('correction target overlaps dev within1um')
  BALANCED.mkdir();(BALANCED/'data').mkdir()
  for child in SOURCE.iterdir():
   if child.name not in ('data','metadata.json'):(BALANCED/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
  for child in (SOURCE/'data').iterdir():
   if child.name!='label_repair_overlay.npz':(BALANCED/'data'/child.name).symlink_to(child.resolve(),target_is_directory=child.is_dir())
  overlay['correction_window_weight']=weights
  with (BALANCED/'data/label_repair_overlay.npz').open('wb') as f:np.savez_compressed(f,**overlay)
  meta['experiment_sampling_override']={'source':str(SOURCE),'only_changed_array':'correction_window_weight','bins':bins,'label_bytes_unchanged':True,'recovery_frames_available':False}
  save(BALANCED/'metadata.json',meta)
  source_sha=sha(SOURCE/'data/dataset.npz');assert source_sha==sha(BALANCED/'data/dataset.npz')
  from lerobot_training.train_diffusion_policy import RCMDataset,image_transform_for_dataset
  from lerobot_training.dit_experience_replay import load_correction_window_weights
  ds=RCMDataset(BALANCED,transform=image_transform_for_dataset(Path(config['data_dir'])),n_obs_steps=2,n_action_steps=16,include_velocity=True,include_previous_action=True)
  loaded,_,receipt=load_correction_window_weights(BALANCED,np.asarray(ds.valid_indices));np.testing.assert_array_equal(loaded,weights)
  sample=ds[int(np.flatnonzero(moving)[0])];assert sample['action'].shape==(16,3)
  assert all('temporal_bias' in key for key in config['freeze_config']['trainable_parameters'])
  save(OUT/'card.json',{'status':'prepared','hypothesis':'Original correction sampling is dominated by stationary frames; increasing executed teacher approach windows improves terminal convergence.',
   'only_changed_factor':'Correction window sampling distribution','arms':{'A_original':str(SOURCE),'B_balanced':str(BALANCED)},
   'reference_checkpoint':str(REFERENCE),'reference_config':{k:config[k] for k in ['data_dir','val_data_dir','init_from']},
   'same_initialization_sha256':sha(config['init_from']),'source_label_sha256':source_sha,'bins_B':bins,
   'budget':{'arm_train_batches':4000,'epochs':4,'batch_size':16,'total_full_train_batches':8000,'smoke_batches_per_arm':2,'wall_limit_minutes':90},
   'frozen':{'backbone_lr':0,'attention_lr':.00003,'trainable':'temporal_bias only','seed':20261118,'correction_ratio':.15},
   'evaluation':{'indices':list(INDICES),'seed':'42+index','steps':1600,'period_s':.05,'samples':1,'exec_steps':1,'flow_steps':4,'controller':'feedback','geometric_takeover':False,'hold_um':20,'hold_cycles':200},
   'dev20_rule':'B final_mean < A final_mean AND B sustained200>=2/5 AND B max_final<=A max_final AND at least60min remaining in90min budget; otherwise skipped',
   'data_limits':['Only existing approach/hold teacher fragments; no receding or transverse recovery coverage.','Repeated 230 overlapping approach windows; increased exposure is not new independent data.'],
   'failure_policy':'Stop dependent stages on execution/finite/sampling failure; no retries; failed improvement gate skips dev20.'})
  save(OUT/'preflight.json',{'status':'passed','gpu':gpu_check(),'identity':identity(),'source_npz_sha256':source_sha,'source_overlay_sha256':sha(SOURCE/'data/label_repair_overlay.npz'),
   'minimum_train_dev_target_distance_um':separation,'sample_parity':'weights exact and real image/sample loaded','weights_receipt':receipt,'bins':bins,'script_sha256':sha(__file__),
   'balanced_overlay_sha256':sha(BALANCED/'data/label_repair_overlay.npz')})
  save(OUT/'workflow_status.json',{'status':'prepared','stage':'preflight'})
  record('preflight','passed',{'bins':bins,'source_unchanged':source_sha,'target_separation_um':separation})
  print(json.dumps({'status':'prepared','bins':bins,'output':str(OUT)}),flush=True)
 except BaseException as e:
  save(OUT/'workflow_status.json',{'status':'preflight_failed','error':str(e),'traceback':traceback.format_exc()});record('preflight','failed',str(e));raise

def run():
 started=time.monotonic()
 status=read(OUT/'workflow_status.json')
 if status['status']!='prepared':raise ValueError('Not a fresh prepared run')
 receipt=read(OUT/'preflight.json');card=read(OUT/'card.json')
 from scripts.research.homog_replay_common import identity
 from scripts.research.homog_replay_evaluate import evaluate
 from scripts.research.eval_20um_hold_metrics import trajectory_metrics
 def check():
  assert identity()==receipt['identity'];assert sha(SOURCE/'data/dataset.npz')==receipt['source_npz_sha256']
  assert sha(SOURCE/'data/label_repair_overlay.npz')==receipt['source_overlay_sha256']
  assert sha(card['reference_config']['init_from'])==card['same_initialization_sha256']
  assert sha(BALANCED/'data/label_repair_overlay.npz')==receipt['balanced_overlay_sha256'];assert sha(__file__)==receipt['script_sha256']
 def stage(name):status.update(status='running',stage=name);save(OUT/'workflow_status.json',status);print('STAGE',name,flush=True);check()
 def train(arm,smoke=False):
  stage(('smoke_' if smoke else 'train_')+arm);cmd=command(arm,smoke);save(OUT/(('smoke_' if smoke else 'train_')+arm+'_command.json'),cmd)
  subprocess.run(cmd,cwd=ROOT,check=True,timeout=600 if smoke else 1200)
  cp=OUT/('smoke' if smoke else 'checkpoints')/arm/('step_000002.pth' if smoke else 'step_004000.pth')
  d=torch.load(cp,map_location='cpu',weights_only=False);c=d['config']
  if not all('temporal_bias' in x for x in c['freeze_config']['trainable_parameters']):raise ValueError('trainable set changed')
  if not np.isfinite([d.get('train_loss',np.nan)]).all():raise ValueError('nonfinite loss')
  if not smoke:
   tr=read(cp.parent/'training_receipt.json');assert tr['total_batches']==4000
   save(OUT/(arm+'_training_receipt.json'),tr)
  record(status['stage'],'completed',{'checkpoint':str(cp),'train_loss':d['train_loss']});return cp
 def assessment(arm,cp,indices):
  stage('eval_'+arm+('_dev20' if len(indices)==20 else '_fast5'));dest=OUT/('dev20_B' if len(indices)==20 else 'eval_'+arm)
  summary=evaluate(cp,1,'feedback',indices,dest,{'identity':receipt['identity']},workers=5,max_workers=5,action_aggregation='median')
  holds={f'E{i:02d}':trajectory_metrics(dest/'shards/tasks'/f'eval_E{i:02d}/raw/evaluation_results.json') for i in indices}
  result={'final_mean_um':summary['final_mean_um'],'min_mean_um':summary['min_mean_um'],
   'max_final_um':max(x['final_error_um'] for x in summary['episodes']),
   'sustained200':sum(x['sustained_200_pass'] for x in holds.values()),'hold_details':holds,'checkpoint':str(cp)}
  save(dest/'hold_summary.json',result);record(status['stage'],'completed',{k:v for k,v in result.items() if k!='hold_details'});return result
 try:
  for arm in ['A_original','B_balanced']:train(arm,True)
  cps={}
  for arm in ['A_original','B_balanced']:cps[arm]=train(arm)
  results={arm:assessment(arm,cps[arm],INDICES) for arm in cps}
  a=results['A_original'];b=results['B_balanced'];gate=b['final_mean_um']<a['final_mean_um'] and b['sustained200']>=2 and b['max_final_um']<=a['max_final_um']
  status['fast5_results']=results;status['dev20_gate']=bool(gate);save(OUT/'comparison.json',{'A':a,'B':b,'gate_passed':bool(gate)})
  remaining=90-(time.monotonic()-started)/60
  if gate and remaining>=60:status['dev20']=assessment('B_balanced',cps['B_balanced'],tuple(range(20)))
  else:status['dev20']={'status':'skipped_failed_gate' if not gate else 'skipped_time','remaining_minutes':remaining}
  status.update(status='completed',stage='report');record('workflow','completed',{'dev20_gate':bool(gate),'comparison':str(OUT/'comparison.json')})
 except BaseException as e:
  status.update(status='failed',error=str(e),traceback=traceback.format_exc());record(status['stage'],'failed',str(e));raise
 finally:save(OUT/'workflow_status.json',status)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
 prepare() if a.mode=='prepare' else run()
