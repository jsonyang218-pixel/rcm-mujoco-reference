"""Matched eight-window fit: full-horizon vs first-action Flow supervision."""
import copy,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy');sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'runs/20261125/terminal_tinyfit_v1'
OUT=ROOT/'runs/20261125/terminal_first_step_fit_v1'
spec=importlib.util.spec_from_file_location('tinyfit_base',ROOT/'runs/20261125/terminal_tinyfit_code_v1/tinyfit.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def save(name,d):
 p=OUT/name;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,allow_nan=False));temp.replace(p)
def record(stage,status,d):
 with (ROOT/'TEST_LOG.md').open('a') as f:f.write(f'\n- {datetime.now().astimezone().isoformat()}: terminal_first_step_fit_v1 {stage} {status}; command={sys.argv}; card={OUT}/card.json; result={d}; output={OUT}. Training-window fit diagnostic only; no rollout.\n')

def objective(m,b,c,first):
 normalized=(b['action']-m.action_mean.reshape(1,1,-1))/m.action_std.reshape(1,1,-1)
 noise=torch.randn_like(normalized);tau=torch.rand(len(normalized),device='cuda',dtype=normalized.dtype)
 x=(1-tau[:,None,None])*noise+tau[:,None,None]*normalized
 predicted=m._predict_velocity_from_condition(x,tau,*c);target=normalized-noise
 return F.mse_loss(predicted[:,0],target[:,0]) if first else F.mse_loss(predicted,target)

@torch.no_grad()
def measure(m,b,c):
 full=[];first=[];rows=[];scale=torch.tensor([1,1,1000],device='cuda')
 for seed in [91001,91002,91003,91004]:
  torch.manual_seed(seed);full.append(float(objective(m,b,c,False)))
  torch.manual_seed(seed);first.append(float(objective(m,b,c,True)))
 for seed in [92001,92002,92003,92004]:
  pred=m.sample(b['observation.image'],b['observation.state'],b['observation.oct'],b['observation.velocity'],b['observation.previous_action'],steps=4,generator=torch.Generator(device='cuda').manual_seed(seed))
  error=(pred[:,0]-b['action'][:,0]).abs()
  rows.append({'seed':seed,'all_mae_ABC_deg_deg_um':(error.mean(0)*scale).tolist(),'moving_mae_ABC_deg_deg_um':(error[:6].mean(0)*scale).tolist()})
 return {'full_flow_loss':float(np.mean(full)),'first_flow_loss':float(np.mean(first)),
  'moving_mae_ABC_deg_deg_um':np.mean([x['moving_mae_ABC_deg_deg_um'] for x in rows],axis=0).tolist(),
  'all_mae_ABC_deg_deg_um':np.mean([x['all_mae_ABC_deg_deg_um'] for x in rows],axis=0).tolist(),'per_seed':rows}

def prepare():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 try:
  old=json.loads((SOURCE/'preflight.json').read_text());assert old['cached_loss_parity']=='passed'
  assert hashlib.sha256(base.REF.read_bytes()).hexdigest()==old['reference_sha256']
  cached=SOURCE/'cached_training_batch.pt';d=torch.load(cached,map_location='cuda',weights_only=False);b=d['batch'];c=tuple(d['condition']);m=base.model();base.cache_condition(m,c)
  # Verify new exact loss implementation against the real model loss before use.
  with torch.no_grad():
   torch.manual_seed(8001);real=base.loss(m,b)
   torch.manual_seed(8001);ours=objective(m,b,c,False)
   torch.testing.assert_close(real,ours,rtol=1e-6,atol=1e-7)
   before=measure(m,b,c)
  np.testing.assert_allclose(before['moving_mae_ABC_deg_deg_um'],old['before']['moving_first_action_mae_ABC_deg_deg_um'],rtol=1e-6,atol=1e-7)
  card={'status':'prepared','source':str(SOURCE),'reference':str(base.REF),'reference_sha256':old['reference_sha256'],
   'batch_sha256':hashlib.sha256(cached.read_bytes()).hexdigest(),'windows':8,'training_episode_ids':old['training_episode_ids'],
   'updates_per_arm':128,'lr':3e-5,'seed_schedule':'73000+step, identical both arms',
   'arms':['full16','first1'],'trainable':'temporal_bias + output.weight/output.bias in BOTH arms',
   'only_changed_factor':'Flow MSE over16 actions vs action index0 only',
   'probe_seeds':[91001,91002,91003,91004],'sampler_seeds':[92001,92002,92003,92004],'sampler_steps':4,
   'budget_s':180,'loss_parity':'passed','initial_action_metrics_reproduction':'passed','before':before,
   'gate':'moving C MAE below initial and below full16; A/B each <=1.05*full16; C improves vs full16 on at least3/4 sampler seeds',
   'scope':'No held-out states, no closed-loop, no automatic long training.'}
  save('card.json',card);save('workflow_status.json',{'status':'prepared','stage':'preflight'})
  record('preflight','passed',{'loss_parity':'passed','initial_metrics_reproduced':True,'before':before});print(json.dumps({'status':'prepared','before':before}),flush=True)
 except BaseException as e:
  record('preflight','failed',str(e));save('workflow_status.json',{'status':'failed','stage':'preflight','error':str(e)});raise

def run():
 start=time.monotonic();status={'status':'running','stage':'load'}
 try:
  card=json.loads((OUT/'card.json').read_text());cached=SOURCE/'cached_training_batch.pt'
  assert hashlib.sha256(cached.read_bytes()).hexdigest()==card['batch_sha256']
  assert hashlib.sha256(base.REF.read_bytes()).hexdigest()==card['reference_sha256']
  d=torch.load(cached,map_location='cuda',weights_only=False);b=d['batch'];c=tuple(d['condition']);m=base.model();initial=copy.deepcopy(m.state_dict());base.cache_condition(m,c)
  results={}
  for arm in ['full16','first1']:
   status['stage']=arm;save('workflow_status.json',status);m.load_state_dict(initial)
   for name,p in m.named_parameters():p.requires_grad_('temporal_bias' in name or name.startswith('output.'))
   before=measure(m,b,c);assert before==card['before']
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-5,weight_decay=0)
   for step in range(128):
    torch.manual_seed(73000+step);opt.zero_grad(set_to_none=True);loss=objective(m,b,c,arm=='first1')
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
    if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient')
    opt.step()
   after=measure(m,b,c);results[arm]={'before':before,'after':after,'updates':128,'trainable_count':sum(p.numel() for p in m.parameters() if p.requires_grad)}
   save(arm+'.json',results[arm]);record(arm,'completed',results[arm]);print(json.dumps({arm:after}),flush=True)
  full=results['full16']['after'];first=results['first1']['after'];x=np.array(full['moving_mae_ABC_deg_deg_um']);y=np.array(first['moving_mae_ABC_deg_deg_um']);z=np.array(card['before']['moving_mae_ABC_deg_deg_um'])
  wins=sum(p['moving_mae_ABC_deg_deg_um'][2]<q['moving_mae_ABC_deg_deg_um'][2] for p,q in zip(first['per_seed'],full['per_seed']))
  gate=bool(y[2]<z[2] and y[2]<x[2] and np.all(y[:2]<=1.05*x[:2]) and wins>=3)
  status.update(status='completed',stage='report',elapsed_s=time.monotonic()-start,results=results,C_seed_wins_vs_full16=wins,preliminary_gate_passed=gate)
  save('summary.json',status);record('workflow','completed',{'elapsed_s':status['elapsed_s'],'gate_passed':gate,'C_seed_wins':wins,'summary':str(OUT/'summary.json')})
 except BaseException as e:
  status.update(status='failed',error=str(e),traceback=traceback.format_exc());record(status['stage'],'failed',str(e));raise
 finally:save('workflow_status.json',status)

if __name__=='__main__':
 torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
 prepare() if sys.argv[1]=='prepare' else run()
