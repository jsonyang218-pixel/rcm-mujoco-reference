"""Read-only teacher/policy comparison and actual v5 sampling/label audit."""
import csv,hashlib,json,sys,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT=Path('/home/him1/mujoco_bc/rcm_4090_deploy')
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/20261125/terminal_round_v1'
OUT=ROOT/'runs/20261125/action_label_audit_v1'

def save(path,d):
 Path(path).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')

def record(status,details):
 with (ROOT/'TEST_LOG.md').open('a',encoding='utf-8') as f:
  f.write(f'\n- {datetime.now().astimezone().isoformat()}: action_label_audit_v1 {status}; command={sys.argv}; input={RUN}; config={OUT}/config.json; result={details}; output={OUT}. Read-only offline analysis, no training or simulation rollout.\n')

def load(path):return json.loads(Path(path).read_text())

def exposure(starts,weights,n):
 weights=np.asarray(weights,float);weights=weights/weights.sum()
 delta=np.zeros(n+1);np.add.at(delta,starts,weights/16);np.add.at(delta,starts+16,-weights/16)
 result=np.cumsum(delta)[:-1];result[np.abs(result)<1e-14]=0
 np.testing.assert_allclose(result.sum(),1,atol=1e-8)
 return result

def pool_summary(name,ds,starts,weights):
 starts=np.asarray(starts,int);weights=np.asarray(weights,float);prob=weights/weights.sum()
 mass=exposure(starts,weights,len(ds.states));octs=np.asarray(ds.oct_navigation,float)
 dist=np.linalg.norm(octs,axis=1)*1000
 label=np.asarray(ds.actions,float);episode=np.asarray(ds.episode_indices)
 velocity=np.zeros_like(ds.states,dtype=float)
 previous=np.zeros_like(ds.states,dtype=float)
 velocity[starts]=(np.asarray(ds.states[starts],float)-ds.states[starts-1])/(np.asarray(ds.timestamps_s[starts])-ds.timestamps_s[starts-1])[:,None]
 previous[starts]=ds.executed_actions[starts-1]
 def fraction(mask):return float(np.sum(prob[mask[starts]]))
 weighted={'name':name,'valid_windows':len(starts),'positive_weight_windows':int(np.count_nonzero(weights)),
  'first_action_zero_label_fraction':fraction(np.all(np.abs(label)<1e-10,axis=1)),
  'first_observation_under20_fraction':fraction(dist<=20),
  'first_observation_20to100_fraction':fraction((dist>20)&(dist<=100)),
  'first_observation_100to200_fraction':fraction((dist>100)&(dist<=200)),
  'action_block_zero_label_fraction':float(mass@np.all(np.abs(label)<1e-10,axis=1)),
  'action_block_under20_fraction':float(mass@(dist<=20))}
 rawpath=ds.data_dir/'data/dataset.npz'
 with np.load(rawpath,allow_pickle=False) as raw:
  if 'truth_navigation_mm' in raw:
   truth=np.linalg.norm(raw['truth_navigation_mm'],axis=1)*1000
   weighted['truth_action_block_under20_fraction']=float(mass@(truth<=20))
 overlay=ds.data_dir/'data/label_repair_overlay.npz'
 if overlay.exists():
  with np.load(overlay,allow_pickle=False) as o:
   weighted['action_block_t3_stationary_fraction']=float(mass@o['t3_stationary'])
   weighted['first_action_t3_stationary_fraction']=fraction(o['t3_stationary'])
   weighted['sampled_episode_ids']=sorted(set(episode[starts[weights>0]].tolist()))
 return weighted,velocity,previous

def nearest(ds,starts,weights,query,velocity,previous):
 # Physical gates; nearest neighbours are not the same observation or label.
 q=np.asarray(query['state']);octq=np.asarray(query['oct']);vq=np.asarray(query['velocity']);pq=np.asarray(query['previous'])
 starts=np.asarray(starts,int);positive=np.asarray(weights)>0;rows=starts[positive];w=np.asarray(weights,float)[positive]
 dq=np.abs(np.asarray(ds.states[rows],float)-q);do=np.linalg.norm(ds.oct_navigation[rows]-octq,axis=1)
 dv=np.abs(velocity[rows]-vq);dp=np.abs(previous[rows]-pq)
 match=(do<=.01)&np.all(dq<=np.array([.5,.5,1.]),axis=1)&np.all(dv<=np.array([.1,.1,.02]),axis=1)&np.all(dp<=np.array([.01,.01,.005]),axis=1)
 score=np.sum((dq/np.array([.5,.5,1.]))**2,axis=1)+(do/.01)**2+np.sum((dv/np.array([.1,.1,.02]))**2,axis=1)+np.sum((dp/np.array([.01,.01,.005]))**2,axis=1)
 order=np.argsort(score)[:5]
 neighbours=[]
 for k in order:
  i=int(rows[k]);neighbours.append({'row':i,'episode':int(ds.episode_indices[i]),'context_score':float(score[k]),
   'within_all_gates':bool(match[k]),'oct_difference_um':float(do[k]*1000),'state_difference_deg_deg_mm':dq[k].tolist(),
   'velocity_difference_deg_deg_mm_s':dv[k].tolist(),'previous_action_difference_deg_deg_mm':dp[k].tolist(),
   'actual_training_label_deg_deg_mm':np.asarray(ds.actions[i]).tolist(),'label_window_weight':float(w[k])})
 return {'matched_windows':int(match.sum()),'matched_episode_count':len(set(np.asarray(ds.episode_indices)[rows[match]].tolist())),
  'matched_sampling_probability_within_pool':float(w[match].sum()/w.sum()),'nearest':neighbours,
  'limitation':'Matching uses state/OCT/velocity/previous action; RGB and full observation history are not matched. Zero matches establish only absence within declared gates.'}

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 try:
  checkpoint=ROOT/'outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth'
  config=torch.load(checkpoint,map_location='cpu',weights_only=False)['config']
  save(OUT/'config.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
   'data_dir':config['data_dir'],'correction_data_dir':config['correction_data_dir'],'correction_ratio':config['correction_ratio'],
   'original_max_episode':config['original_max_episode'],'label_overlay':config.get('label_overlay'),
   'freeze_config':config['freeze_config'],'physical_gates':{'oct_difference_um':10,'state_deg_deg_mm':[.5,.5,1.],
    'velocity_deg_deg_mm_s':[.1,.1,.02],'previous_action_deg_deg_mm':[.01,.01,.005]},
   'scope':'Offline only; development trajectories never become training samples'})
  from lerobot_training.train_diffusion_policy import RCMDataset,image_transform_for_dataset
  from lerobot_training.dit_experience_replay import load_correction_window_weights
  transform=image_transform_for_dataset(Path(config['data_dir']))
  original=RCMDataset(Path(config['data_dir']),transform=transform,n_obs_steps=2,n_action_steps=16,include_velocity=True,include_previous_action=True)
  correction=RCMDataset(Path(config['correction_data_dir']),transform=transform,n_obs_steps=2,n_action_steps=16,include_velocity=True,include_previous_action=True)
  starts=np.asarray(original.valid_indices,int);starts=starts[original.episode_indices[starts]<=config['original_max_episode']]
  cstarts=np.asarray(correction.valid_indices,int);cw,_,receipt=load_correction_window_weights(correction.data_dir,cstarts)
  osummary,ov,op=pool_summary('original_85pct',original,starts,np.ones(len(starts)))
  csummary,cv,cp=pool_summary('correction_15pct',correction,cstarts,cw)
  # Confirm velocity and labels against the real __getitem__, including actual images.
  for ds,valid,vel,prev in [(original,starts,ov,op),(correction,cstarts,cv,cp)]:
   chosen=int(valid[len(valid)//2]);idx=ds.valid_indices.index(chosen) if isinstance(ds.valid_indices,list) else int(np.flatnonzero(ds.valid_indices==chosen)[0])
   sample=ds[idx]
   np.testing.assert_allclose(sample['action'][0].numpy(),ds.actions[chosen],atol=1e-7,rtol=1e-6)
   np.testing.assert_allclose(sample['observation.velocity'].numpy(),vel[chosen],atol=2e-5,rtol=1e-4)
   np.testing.assert_allclose(sample['observation.previous_action'].numpy(),prev[chosen],atol=1e-7,rtol=1e-6)
  checks={'actual_dataset_getitem_parity':'passed for one real sample per pool','exposure_probability_sum':'passed for both pools'}
  save(OUT/'sampling_audit.json',{'original':osummary,'correction':csummary,'weights_receipt':receipt,'checks':checks})
  record('sampling_checks_passed',{'checks':checks,'correction':csummary})
  comparisons=[];neighbours=[];csvrows=[]
  for model in ['baseline','v5_A4000']:
   for target in [6,10]:
    ep=load(RUN/model/'shards/tasks'/f'eval_E{target:02d}/raw/evaluation_results.json')['episodes'][0]
    replay=load(RUN/'teacher_replay'/model/f'E{target:02d}.json');index=replay['source_step'];tr=ep['trajectory']
    policy=np.asarray(tr['raw_actions'][index]);teacher=np.asarray(replay['first_labels']['actual_velocity_teacher']['action_deg_deg_mm'])
    first={'model':model,'target':target,'source_step':index,'source_error_um':tr['errors'][index]*1000,
     'policy_action_deg_deg_mm':policy.tolist(),'teacher_action_deg_deg_mm':teacher.tolist(),
     'A_policy_to_teacher_ratio':float(policy[0]/teacher[0]) if abs(teacher[0])>1e-10 else None,
     'C_policy_um':float(policy[2]*1000),'C_teacher_um':float(teacher[2]*1000),'same_sign_ABC':(np.sign(policy)==np.sign(teacher)).tolist(),
     'initial_state_same':replay['prefix_exact'],'teacher_final_um':replay['final_um'],'teacher_hold_cycles':replay['max_hold_cycles']}
    comparisons.append(first)
    for horizon in [1,5,10,20]:
     csvrows.append({'model':model,'target':target,'source_step':index,'horizon_cycles':horizon,
      'policy_error_um':tr['errors'][index+horizon]*1000,'teacher_error_um':replay['trajectory'][horizon-1]['true_error_um'],
      'same_start_only':'Branches diverge after start; not same-observation action comparison'})
    if model=='v5_A4000':
     for offset in [0,5,10,20]:
      i=index+offset;query={'state':tr['states'][i],'oct':tr['oct_navigation_mm'][i],'velocity':tr['velocities'][i],'previous':tr['actions'][i-1]}
      neighbours.append({'target':target,'step':i,'query':query,
       'policy_action_deg_deg_mm':tr['raw_actions'][i],
       'original_pool':nearest(original,starts,np.ones(len(starts)),query,ov,op),
       'correction_pool':nearest(correction,cstarts,cw,query,cv,cp)})
  save(OUT/'same_start_comparison.json',comparisons);save(OUT/'training_label_neighbours.json',neighbours)
  with (OUT/'branch_error_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
   writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
  with (OUT/'same_start_actions.csv').open('w',newline='',encoding='utf-8-sig') as f:
   fields=['model','target','step','error_um','policy_A_deg','teacher_A_deg','policy_B_deg','teacher_B_deg','policy_C_um','teacher_C_um']
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for row in comparisons:
    p=row['policy_action_deg_deg_mm'];t=row['teacher_action_deg_deg_mm']
    writer.writerow(dict(zip(fields,[row['model'],row['target'],row['source_step'],row['source_error_um'],p[0],t[0],p[1],t[1],p[2]*1000,t[2]*1000])))
  summary={'status':'completed','comparisons':comparisons,'sampling':{'original':osummary,'correction':csummary},
   'matched_windows':[{'target':r['target'],'step':r['step'],'original':r['original_pool']['matched_windows'],'correction':r['correction_pool']['matched_windows']} for r in neighbours],
   'checks':checks,'interpretation_limits':['Historical training labels are read from exact configured datasets. Nearest labels are not same-state truth.',
    'Only the first action of teacher/policy branches shares a state. Later errors compare trajectories from a shared start.',
    'Coverage and freeze configuration are measured; their causal contribution is not isolated.',
    'No label declared wrong and no training launched on this evidence alone.']}
  save(OUT/'summary.json',summary)
  lines=['# 动作与训练标签核查','', '状态：完成。没有训练、修改数据或启动新闭环。','',
   '| 模型 | 目标 | 策略 A (°) | 教师 A (°) | 策略 C (μm) | 教师 C (μm) |','|---|---|---:|---:|---:|---:|']
  for row in comparisons:
   p=row['policy_action_deg_deg_mm'];t=row['teacher_action_deg_deg_mm'];lines.append(f"| {row['model']} | E{row['target']:02d} | {p[0]:.6f} | {t[0]:.6f} | {p[2]*1000:.3f} | {t[2]*1000:.3f} |")
  lines+=['','## 实际采样','',f"v5 纠正池实际权重下，动作块 T3 静止帧占比 {csummary.get('action_block_t3_stationary_fraction',0)*100:.2f}%。这只是占总训练采样 15% 的纠正池，不等于全部训练。",
   f"纠正池首帧在 20–100 μm 区间的采样概率 {csummary['first_observation_20to100_fraction']*100:.4f}%。",
   '', '## 证据边界','', '同状态比较仅限分支起点；后续策略与教师状态已经不同。相似训练标签不是同状态正确答案。不能仅按符号把制动判为错误。',
   'v5 本轮只训练时序注意力，不能把数据覆盖与冻结范围的作用混为一个已证实根因。',
   '', '## 下一轮建议','', '先在训练目标上生成具有不同速度、运动方向的终末监督片段，验证教师可执行性。以同初始化、同可训练参数、同训练步数的原数据与补充数据两组实验检验覆盖假设；保持分支和恢复分支分层采样。',
   '本次不修改静止标签，也不使用开发目标生成训练数据。']
  (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  record('completed',{'summary':str(OUT/'summary.json'),'same_start_states':4,'training_context_queries':8})
  print(json.dumps(summary,ensure_ascii=False))
 except BaseException as e:
  save(OUT/'failure.json',{'status':'failed','error':str(e),'traceback':traceback.format_exc()});record('failed',str(e));raise

if __name__=='__main__':main()
