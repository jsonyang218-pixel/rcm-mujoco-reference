"""Freeze provenance, geometry-only target construction, seeds and evaluation protocol."""
import os,sys,json,hashlib,time,ast
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path[:0]=[str(OUT),str(ROOT)]
import evaluate as ev
c=ev.c
assert not (OUT/'frozen_plan.json').exists()
inv=ev.read(OUT/'history_inventory.json');usage=ev.read(OUT/'existing_target_usage.json');manifest=ev.read(ROOT/'configs/target_split_manifest.json')
ev.record('history_audit_completed',dict(files=len(inv['files']),historical_point_records=inv['point_count'],unreadable_json=len(inv['errors']),
 declared_training_targets=len(manifest['training_targets']),declared_dagger_targets=len(manifest['dagger_targets']),development=len(usage['development_targets']),
 development_with_historical_evaluation=sum(bool(t['historical_outputs_or_data_matches']) for t in usage['development_targets']),
 final_test=len(usage['final_test_targets']),final_with_historical_evaluation=sum(bool(t['historical_outputs_or_data_matches']) for t in usage['final_test_targets']),
 conclusion='No unused targets in existing split: training/dagger excluded by declared roles, development and all60 final targets have historical evaluation; create independent geometry-only fresh targets. No model/sensor-success screening.',
 evidence=str(OUT/'existing_target_usage.json'),limits=inv['limits']))
import numpy as np
from scipy.spatial import cKDTree
from rcm.scene_v9 import RCMSceneV9
from rcm.fundus import make_random_fundus_targets
points=ev.read(OUT/'history_points.json');scene=RCMSceneV9()
historical=np.asarray([r['point'] if r['frame']=='robot' else scene.world_to_robot(r['point']) for r in points])
historical=np.vstack([historical,np.asarray([t['point_mm'] for k in ['training_targets','dagger_targets','development_targets','final_test_targets'] for t in manifest[k]])]);tree=cKDTree(historical)
seed_candidates=[1926112701,1926112702];gen_seed=1926112700
assert not set([*seed_candidates,gen_seed]) & set(inv['seeds'])
# Also audit literals across historical text, including source/config/log files, before using the seeds.
import subprocess
res=subprocess.run(['rg','-l','--hidden','-F','-e',str(gen_seed),'-e',str(seed_candidates[0]),'-e',str(seed_candidates[1]),'runs','outputs','data','configs','github_reports','scripts','rcm','restricted_perception','lerobot_training','-g','*.json','-g','*.jsonl','-g','*.py','-g','*.md','-g','*.log','-g','!dit_v3_generalization_v1/**'],cwd=ROOT,capture_output=True,text=True)
hits=[x for x in res.stdout.splitlines() if not x.startswith(str(OUT.relative_to(ROOT))+'/')]
assert res.returncode in (0,1) and not hits,hits
rng=np.random.default_rng(gen_seed);targets=[];rejections=[];zero=np.zeros(3)
for attempt in range(10000):
 raw=make_random_fundus_targets(scene._ref,rng,1,min_separation_mm=0.,minimum_normalized_posterior_z=.9)['targets'][0]['point_mm']
 p=np.asarray(raw)
 if tree.query(p)[0]<.100:rejections.append(dict(attempt=attempt,reason='within100um_known_history',point_mm=raw));continue
 if targets and min(np.linalg.norm(p-t['target_robot_mm']) for t in targets)<1.:rejections.append(dict(attempt=attempt,reason='within1mm_selected_target',point_mm=raw));continue
 try:
  scene.set_task(zero);scene.set_target_world(scene.robot_to_world(p));goal=scene.target_task(validate=True)
  counts=np.rint(scene.encoders(goal)).astype('int64');q=scene.task_from_encoders(counts,reference_task=zero);scene.set_task(q)
  reachable=scene.world_to_robot(scene.site_mm('needle_tip')).copy();shift=float(np.linalg.norm(reachable-p)*1000)
  scene.set_task(zero);scene.set_target_world(scene.robot_to_world(reachable));checked=scene.target_task(validate=True)
  if shift>.352:raise ValueError(f'nearest integer endpoint shift {shift}um > original0.352um')
  distance=float(tree.query(reachable)[0]);assert distance>.099
 except Exception as e:rejections.append(dict(attempt=attempt,reason='geometry_or_integer_endpoint_rejected',error=str(e),point_mm=raw));continue
 targets.append(dict(id=f'G{len(targets):02d}',target_robot_mm=reachable.tolist(),nominal_robot_mm=raw,goal_state_si=checked.tolist(),reachable_encoder_counts_ABC=counts.tolist(),target_shift_um=shift,
   minimum_known_history_distance_um=distance*1000,generation_attempt=attempt,source='rcm.fundus.make_random_fundus_targets: compiled posterior eye surface proxy; normalized z>=0.9',
   prior_use='Newly generated after both frozen checkpoints; not from old development/final pools; no training/model-selection/tuning or sensor/model screening',
   minimum_other_selected_distance_mm=min((float(np.linalg.norm(reachable-t['target_robot_mm'])) for t in targets),default=None)))
 if len(targets)==10:break
scene.set_task(zero);scene.close();assert len(targets)==10
plan=dict(created_at=c.now(),generator_seed=gen_seed,sampling_seeds=seed_candidates,seed_audit=dict(historical_seed_metadata=len(inv['seeds']),literal_hits=hits,scope='local JSON/JSONL/source/report/log plus metadata; no claim about unavailable external history'),
 targets=targets,jobs=[dict(id=t['id']+'_s'+str(s),target_id=t['id'],target_robot_mm=t['target_robot_mm'],seed=s) for t in targets for s in seed_candidates],
 fixed_before_policy_evaluation=True,forbidden_original_targets=['E00','E02','E06','E08','E10','E40','E41','E42'],no_replacement_after_freeze=True)
c.save(OUT/'target_provenance.json',dict(created=c.now(),history_audit=str(OUT/'history_inventory.json'),existing_usage=str(OUT/'existing_target_usage.json'),
 targets=targets,geometry_only_preselection_rejections=rejections,history_limits=inv['limits'],same_two_sampling_seeds_per_target=True,independence='10 distinct spatial targets at least1mm apart; seeds are repeated measurements, not independent spatial targets',
 near_sampler_note='candidate2 keeps original fixed near noise seed434341; fresh episode seeds affect original far DiT stochastic sampling only',
 truth_use='Geometry and reachability only in setup; XYZ logged for offline score; no truth-based policy routing'))
c.save(OUT/'frozen_plan.json',plan)
# Capture the actual runtime and scene inputs as immutable evaluation dependencies.
paths={ROOT/'config.json',ROOT/'v9_alignment.json',ROOT/'v9_tool_mount.json',ROOT/'v9_side_profile.json',ROOT/'configs/target_split_manifest.json',
 OLD if False else ev.NEAR,c.REF,ev.OLD/'campaign.py',ev.OLD/'model.py',OUT/'evaluate.py',OUT/'launch.sh',OUT/'prepare.py',OUT/'audit_history.py'}
for base in ['rcm','lerobot_training','mujoco_model/vendor/simulation_driver']:
 for p in (ROOT/base).rglob('*'):
  if p.is_file() and p.suffix in {'.py','.json','.xml','.stl','.obj','.yaml','.urdf'} and '__pycache__' not in p.parts:paths.add(p)
for n in ['motor_backend.py','motor_execution.py','model_sync.py']:
 for p in ROOT.rglob(n):
  if 'latest_results' not in str(p) and '/source_snapshot/' not in str(p) and 'output_head_sanity_code' not in str(p):paths.add(p)
fingerprints={('near' if p==ev.NEAR else 'far' if p==c.REF else str(p.relative_to(ROOT))):dict(path=str(p),sha256=c.sha(p)) for p in sorted(paths)}
assert fingerprints['near']['sha256']=='dae7252fe5ee1af6a35d39974b395b674f9b6607175b8e6d984d5331812cd843'
assert fingerprints['far']['sha256']==c.REF_SHA
card=dict(name=OUT.name,created_at=c.now(),hypothesis='Evaluate frozen v3 candidate2 + original far v5 on independently generated, geometrically separated targets with no learning or tuning',
 plan_sha256=c.sha(OUT/'frozen_plan.json'),fingerprints=fingerprints,checkpoints=dict(near=str(ev.NEAR),far=str(c.REF)),
 protocol=dict(cycles=1600,control_period_s=.05,K=1,exec_steps=1,flow_integration_steps=4,switch='latch near once valid measured OCT norm<=0.100mm; no history reset',
 start='Fresh RCMSceneV9 default exact zero ABC SI; fresh G10 motor session and zero time; same method for all20 jobs; no target-relative offsets',
 control='Existing feedback controller, production action clipping and safety preview scales, G10, original strictRGBD/OCT settings',
 success='All final200 post-action XYZ errors<=20um including action1600; endpoint and per-cycle20um rate, first entry and longest/terminal hold also reported',
 original_early_stop_error_mm=30.,stop_on_success=False,max_consecutive_safety_limited=None,
 invalid_or_unreachable='Never replace/drop frozen job; preserve negative or incomplete in denominator, record sensor validity and all exceptions',
 truth='Setup and offline scoring only; both routes current DiT neural policy, no geometry/teacher control'),
 eval_tier='Independent frozen10targets x2sampling seeds, custom user-authorized; no old five-target, screen10/dev20 or historical final-test reruns',
 train_updates=0,no_results_based_tuning=True,concurrency=12,worker_timeout_s=7200,batch_timeout_s=18000,
 resume='Explicit run --resume: skip complete jobs, deterministically replay saved prefix from identical standard start with exact RGB/OCT/raw-action/postXYZ/encoder checks, append only new cycles; never silently discard mismatch or replace target',
 qualification='Limited simulation sample:10 spatial targets x2seeds; does not prove universal generalization or original0.35um settled acceptance',
 output=str(OUT),baseline='Original far v5 is the preserved controller component; no matched independent-target historical baseline, so no paired improvement claim')
c.save(OUT/'experiment_card.json',card)
ev.record('target_seed_protocol_frozen',dict(targets=targets,jobs=20,sampling_seeds=seed_candidates,plan_sha256=card['plan_sha256'],train_updates=0,concurrency=12,output=str(OUT)))
# Pure load/compatibility preflight, no policy action or observation filtering.
os.environ.update(c.RUNTIME)
torch=c.torch_setup();e=c.evaluator();assert len(e.model.blocks)==6;cp=c.checkpoint_load(e,ev.NEAR);assert cp['actual_updates']==1200
for p in [OUT/'evaluate.py',OUT/'prepare.py',OUT/'audit_history.py']:ast.parse(p.read_text())
ev.verify();preflight=dict(status='passed',time=c.now(),cuda_device=torch.cuda.get_device_name(0),architecture_match=True,checkpoint_hashes_verified=True,
 frozen_jobs=20,distinct_targets=10,nominal_cycles=32000,preflight_policy_actions=0,train_updates=0,geometry_only_target_selection=True,
 original5_not_run=True,runtime_fingerprints=len(fingerprints),original_far_v5_sha256=c.REF_SHA,near_sha256=fingerprints['near']['sha256'])
c.save(OUT/'preflight.json',preflight);ev.record('generalization_preflight_passed',preflight)
print(json.dumps(preflight),flush=True)
