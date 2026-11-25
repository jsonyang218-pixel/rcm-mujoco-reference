"""Eight-window cached-condition fit diagnostic; never deployment validation."""
import argparse,copy,hashlib,json,sys,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy');sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/20261125/terminal_tinyfit_v1'
REF=ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
DATA=ROOT/'data/homog_label_fix_20261124_v5_A_t3_w025'

def save(name,d):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=str));t.replace(p)
def record(stage,status,d):
 with (ROOT/'TEST_LOG.md').open('a') as f:f.write(f'\n- {datetime.now().astimezone().isoformat()}: terminal_tinyfit_v1 {stage} {status}; command={sys.argv}; reference={REF}; data={DATA}; result={d}; output={OUT}. Eight-training-window diagnostic only; no closed-loop claim.\n')
def model():
 from lerobot_training.eval_policy import PolicyEvaluator
 e=PolicyEvaluator(str(REF),device='cuda',num_action_samples=1,n_action_execute_steps=1,flow_integration_steps=4,controller_variant='feedback')
 return e.model.eval()
def loss(m,b):
 return m.rectified_flow_loss(b['observation.image'],b['observation.state'],b['observation.oct'],b['observation.velocity'],b['observation.previous_action'],b['action'])
def cache_condition(m,c):
 m._condition=lambda *args,**kwargs:c
@torch.no_grad()
def metrics(m,b):
 losses=[];errors=[]
 for seed in [91001,91002,91003,91004]:
  torch.manual_seed(seed);losses.append(float(loss(m,b)))
 for seed in [92001,92002,92003,92004]:
  generator=torch.Generator(device='cuda').manual_seed(seed)
  pred=m.sample(b['observation.image'],b['observation.state'],b['observation.oct'],b['observation.velocity'],b['observation.previous_action'],steps=4,generator=generator)
  errors.append((pred[:,0]-b['action'][:,0]).abs())
 errors=torch.stack(errors)
 return {'flow_loss_4_fixed_probes':float(np.mean(losses)),
  'first_action_mae_ABC_deg_deg_um':(errors.mean((0,1))*torch.tensor([1,1,1000],device='cuda')).tolist(),
  'moving_first_action_mae_ABC_deg_deg_um':(errors[:,:6].mean((0,1))*torch.tensor([1,1,1000],device='cuda')).tolist()}

def prepare():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 try:
  from lerobot_training.train_diffusion_policy import RCMDataset,image_transform_for_dataset
  ds=RCMDataset(DATA,transform=image_transform_for_dataset(DATA),n_obs_steps=2,n_action_steps=16,include_velocity=True,include_previous_action=True)
  with np.load(DATA/'data/dataset.npz',allow_pickle=False) as a:
   dist=np.linalg.norm(a['truth_navigation_mm'],axis=1)*1000;eps=a['episode_index'].copy()
  starts=np.asarray(ds.valid_indices);chosen=[]
  for ep in range(20,26):
   candidates=np.flatnonzero((eps[starts]==ep)&(dist[starts]>20)&(dist[starts]<=100))
   if not len(candidates):raise ValueError(f'missing approach window E{ep}')
   chosen.append(int(candidates[len(candidates)//2]))
  for ep in [40,41]:
   candidates=np.flatnonzero((eps[starts]==ep)&(dist[starts]<5))
   if not len(candidates):raise ValueError(f'missing hold window E{ep}')
   chosen.append(int(candidates[len(candidates)//2]))
  samples=[ds[i] for i in chosen];b={key:torch.stack([s[key] for s in samples]).cuda() for key in samples[0]}
  m=model()
  for p in m.parameters():p.requires_grad_(False)
  with torch.no_grad():
   condition=m._condition(b['observation.image'],b['observation.state'],b['observation.oct'],b['observation.velocity'],b['observation.previous_action'])
   torch.manual_seed(8001);full=loss(m,b)
   cache_condition(m,condition);torch.manual_seed(8001);cached=loss(m,b)
   torch.testing.assert_close(full,cached,rtol=1e-6,atol=1e-7)
   before=metrics(m,b)
  batch={k:v.cpu() for k,v in b.items() if k!='observation.image'}
  batch['observation.image']=torch.zeros((8,2,3,1,1))
  torch.save({'batch':batch,'condition':[x.cpu() for x in condition]},OUT/'cached_training_batch.pt')
  receipt={'status':'prepared','windows':8,'moving_windows':6,'hold_windows':2,'training_dataset_sample_indices':chosen,
   'source_rows':starts[chosen].tolist(),'training_episode_ids':eps[starts[chosen]].tolist(),
   'reference':str(REF),'reference_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),
   'updates_per_arm':128,'lr_both':3e-5,'arms':['temporal_only','temporal_plus_output_linear'],
   'additional_trainable_B':['output.weight','output.bias'],'conditions_cached':True,'cached_loss_parity':'passed',
   'evaluation':'same eight windows, four fixed probe RNG seeds and four single-sample inference seeds; no held-out state generalization',
   'before':before,'time_cap_seconds':180}
  save('preflight.json',receipt);save('workflow_status.json',{'status':'prepared','stage':'preflight'})
  record('preflight','passed',{'cache_parity':'passed','batch_windows':8,'before':before});print(json.dumps(receipt),flush=True)
 except BaseException as e:
  save('workflow_status.json',{'status':'failed','stage':'preflight','error':str(e)});record('preflight','failed',str(e));raise

def run():
 start=time.monotonic();status={'status':'running','stage':'load'}
 try:
  receipt=json.loads((OUT/'preflight.json').read_text());assert hashlib.sha256(REF.read_bytes()).hexdigest()==receipt['reference_sha256']
  data=torch.load(OUT/'cached_training_batch.pt',map_location='cuda',weights_only=False);b=data['batch'];c=tuple(data['condition'])
  m=model();initial=copy.deepcopy(m.state_dict());results={}
  for arm in receipt['arms']:
   status['stage']=arm;save('workflow_status.json',status);m.load_state_dict(initial);cache_condition(m,c)
   selected=[]
   for name,p in m.named_parameters():
    p.requires_grad_('temporal_bias' in name or (arm=='temporal_plus_output_linear' and name.startswith('output.')))
    if p.requires_grad:selected.append(name)
   frozen={name:p.detach().clone() for name,p in m.named_parameters() if not p.requires_grad and name.startswith('output.')}
   before=metrics(m,b);optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-5,weight_decay=0)
   for step in range(128):
    torch.manual_seed(73000+step);optimizer.zero_grad(set_to_none=True);value=loss(m,b)
    if not torch.isfinite(value):raise FloatingPointError('nonfinite loss')
    value.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
    if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient')
    optimizer.step()
    if (step+1)%32==0:print(json.dumps({'arm':arm,'step':step+1,'loss':float(value.detach()),'elapsed_s':time.monotonic()-start}),flush=True)
   for name,p in m.named_parameters():
    if name in frozen:torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
   after=metrics(m,b);results[arm]={'before':before,'after':after,'flow_loss_ratio':after['flow_loss_4_fixed_probes']/before['flow_loss_4_fixed_probes'],
    'trainable_parameters':selected,'trainable_count':sum(p.numel() for p in m.parameters() if p.requires_grad),'updates':128}
   save(arm+'.json',results[arm]);record(arm,'completed',results[arm]);print(json.dumps({arm:results[arm]}),flush=True)
  status.update(status='completed',stage='report',elapsed_s=time.monotonic()-start,results=results)
  save('summary.json',status);record('workflow','completed',{'elapsed_s':status['elapsed_s'],'summary':str(OUT/'summary.json')})
 except BaseException as e:
  status.update(status='failed',error=str(e),traceback=traceback.format_exc());record(status['stage'],'failed',str(e));raise
 finally:save('workflow_status.json',status)

if __name__=='__main__':
 torch.set_num_threads(1);torch.set_float32_matmul_precision('high');p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
 prepare() if a.mode=='prepare' else run()
