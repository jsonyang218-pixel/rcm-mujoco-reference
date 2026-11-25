"""Execute Plan 8 once, with provenance, stage gates, deadlines and publication."""
from __future__ import annotations
import argparse
import contextlib
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import torch
import torch.nn.functional as F
from .head import RegressionHead, geom_condition, make_sample, symlog, symexp, SCALE

CODE = Path(__file__).resolve().parents[3]
RUNTIME = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
OUTPUT_ROOT = RUNTIME / 'runs/20261125/output_head_sanity_v1'
OUTPUT = Path(os.environ.get('RCM_OUTPUT_HEAD_SANITY_DIR', str(OUTPUT_ROOT))).resolve()
if OUTPUT != OUTPUT_ROOT and not OUTPUT.is_relative_to(OUTPUT_ROOT):
    raise ValueError('Plan8 output must remain under its designated artifact root')
BASELINE = RUNTIME / 'outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth'
BASE_SHA = '185a6ef02caea4ea74ba752d67618d459c906450f5b315cd8010c3c5f1703f1b'
DATA = RUNTIME / 'data/plan2_plus_original_train_224_20261119'
OVERLAY = RUNTIME / 'data/overnight_c_fix_symlog_081234_overlay/label_override.npz'
PAIR_ARCHIVE = RUNTIME / 'runs/20261125/train_direction_pair_v1'
TARGETS = CODE / 'configs/v12_reachable_development_targets_0p35um.json'
MODULE = 'scripts.research.output_head_sanity.workflow'
BRANCH = 'experiment/output-head-sanity-20261125'
FAST5 = (0, 2, 6, 8, 10)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v)) + '\n')
    temporary.replace(path)


def record(stage, status, detail):
    text = (f'\n- {datetime.now().astimezone().isoformat()}: output_head_sanity_v1 {stage} **{status}**; '
            f'command `python -m {MODULE} run --output-dir {OUTPUT}`; '
            f'data={DATA}; checkpoint={BASELINE} sha256={BASE_SHA}; seed=20261118, paired=43001, '
            f'fast5=42+index; details={json.dumps(detail, ensure_ascii=False, default=str)}; output={OUTPUT}. '
            'MuJoCo software simulation only.\n')
    with (RUNTIME / 'TEST_LOG.md').open('a') as stream:
        stream.write(text)
    print(f'[{stage}] {status}: {detail}', flush=True)


def evaluator():
    from lerobot_training.eval_policy import PolicyEvaluator
    from scripts.research.clean_oct_workers import QUALITY
    return PolicyEvaluator(str(BASELINE), device='cuda', num_action_samples=1,
        n_action_execute_steps=1, flow_integration_steps=4, controller_variant='feedback',
        rgbd_render_quality=QUALITY.copy())


def dataset():
    from lerobot_training.train_diffusion_policy import RCMDataset, image_transform_for_dataset
    from scripts.research.overnight_cfix_data import apply_overlay
    ds = RCMDataset(DATA, transform=image_transform_for_dataset(DATA), n_obs_steps=2,
        n_action_steps=16, include_velocity=True, include_previous_action=True)
    ds.actions = apply_overlay(DATA, ds.actions, OVERLAY)
    return ds


def resolve_target_with_raw(directory, index, cache):
    """Supplement frozen metadata with its exact archived collection report."""
    from scripts.research.motor_perception_gates import resolve_training_target
    resolved = resolve_training_target(directory, index, cache)
    if resolved is not None:
        return resolved
    directory = Path(directory)
    metadata_path = directory/'metadata.json'
    metadata = cache.setdefault(metadata_path, read(metadata_path))
    episode = metadata['episodes'][index]
    for source in metadata.get('source_datasets', []):
        start = int(source['episode_offset'])
        if start <= index < start+int(source['episodes']):
            return resolve_target_with_raw(directory.parent/Path(source['directory']).name,index-start,cache)
    if episode.get('source_dataset') and 'source_episode_index' in episode:
        return resolve_target_with_raw(directory.parent/Path(episode['source_dataset']).name,
            int(episode['source_episode_index']),cache)
    prefix='online_dagger_fixed_round1_'
    if directory.name.startswith(prefix):
        shard=directory.name.removeprefix(prefix).removesuffix('_224')
        raw=RUNTIME/'runs/20261115/online_dagger_v12/fixed_target_round1_shards'/shard
        original=int(episode.get('source_episode_index',index))
        directories=sorted(raw.glob('episode_*'))
        episode_dir=directories[original]  # converter records sorted-directory ordinal
        report_path=episode_dir/'control_report.json'
        report=read(report_path)
        if int(report['frames']) != int(episode['num_frames']):
            raise ValueError('Raw report/frame metadata mismatch')
        # Prove the report is for the actual converted trajectory, not a name guess.
        raw_npz=raw/'dataset'/report['data']
        with np.load(raw_npz,allow_pickle=False) as a, np.load(directory/'data/dataset.npz',allow_pickle=False) as b:
            values=b['state'][b['episode_index']==index]
            np.testing.assert_array_equal(values,a['observation.state'])
        cache[report_path]=report
        return {'frame':'robot_mm','point':report['target_robot_mm'],'source':str(report_path),
            'source_episode':int(report['episode_index']),'raw_state_exactly_matched':True,'raw_npz_sha256':sha(raw_npz)}
    raise ValueError(f'No target evidence for {directory}/E{index}')


def target_errors(ds):
    """Independent FK truth from explicit target ancestry, never noisy OCT."""
    from rcm.scene_v9 import RCMSceneV9
    scene = RCMSceneV9()
    provenance, targets, cache = [], {}, {}
    try:
        for episode in np.unique(ds.episode_indices):
            resolved = resolve_target_with_raw(DATA, int(episode), cache)
            if resolved is None:
                raise ValueError(f'Unresolved true target E{episode}; do not relabel OCT as truth')
            point = (scene.robot_to_world(resolved['point']) if resolved['frame'] == 'robot_mm'
                     else np.asarray(resolved['point']))
            targets[int(episode)] = point
            provenance.append({'episode': int(episode), **resolved})
        ref = scene._ref
        state = ds.states.astype(np.float64)
        phi = np.radians(state[:, 0] + ref.phi_home)
        theta = np.radians(state[:, 1] + ref.alpha_home + ref.nominal.geometry.tilt_offset_deg)
        length = state[:, 2] + ref.tip_reference_mm
        robot = length[:, None] * np.stack([np.sin(theta)*np.sin(phi), np.cos(theta),
                                           np.sin(theta)*np.cos(phi)], axis=1)
        # Read the public transform rather than assuming translation convention.
        origin = np.asarray(scene.robot_to_world(np.zeros(3)))
        rotation = np.stack([scene.robot_to_world(v)-origin for v in np.eye(3)], axis=1)
        world = robot @ rotation.T + origin
        site_max_error_mm=0.
        for row in np.linspace(0, len(state)-1, 12, dtype=int):
            scene.set_task(state[row] * [np.pi/180, np.pi/180, .001])
            scalar=scene.robot_to_world(ref.forward_kinematics(state[row]*[np.pi/180,np.pi/180,.001]))
            np.testing.assert_allclose(world[row],scalar,atol=1e-10,rtol=0)
            site_max_error_mm=max(site_max_error_mm,float(np.max(np.abs(world[row]-scene.site_mm('needle_tip')))))
            np.testing.assert_allclose(world[row], scene.site_mm('needle_tip'), atol=1e-6, rtol=0)
        target = np.stack([targets[int(e)] for e in ds.episode_indices])
        distance = np.linalg.norm(world-target, axis=1)*1000
        return distance, {'method': 'float64 analytic FK from stored float32 encoder state and explicit source/audit targets; 12 rows checked against MuJoCo site',
            'max_site_component_discrepancy_um':site_max_error_mm*1000,
            'frames_within_0p002um_of60_threshold':int((np.abs(distance-60)<.002).sum()),
            'target_provenance': provenance, 'source_sha256': {str(p): sha(p) for p in cache},
            'baseline_episode40_target_world_mm': targets[40]}
    finally:
        scene.close()


def unit_checks(model=None, ds=None):
    x = torch.linspace(-.2, .2, 2001, device='cuda')[:, None].expand(-1, 3)
    error = float((symexp(symlog(x))-x).abs().max())
    assert error < 1e-7, error
    torch.manual_seed(20261118)
    head = RegressionHead().cuda().eval()
    a = torch.randn(3, 384, device='cuda')
    state = torch.cuda.get_rng_state().clone()
    one = head(a); two = head(a)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert torch.equal(state, torch.cuda.get_rng_state())
    receipt = {'symlog_max_abs_roundtrip': error, 'deterministic_head_repeat': True,
        'no_sampling_rng_consumed': True, 'parameters': sum(p.numel() for p in head.parameters())}
    if model is not None:
        sample = ds[len(ds)//2]
        args = [sample[k][None].cuda() for k in ('observation.image','observation.state',
            'observation.oct','observation.velocity','observation.previous_action')]
        with torch.no_grad():
            full = model._condition(*args)[1]
            cached = geom_condition(model, *args[1:])
            torch.testing.assert_close(full, cached, rtol=0, atol=0)
            for arm in ('R1','R2'):
                fn = make_sample(head, model, arm)
                rng = torch.cuda.get_rng_state().clone()
                torch.testing.assert_close(fn(*args), fn(*args), rtol=0, atol=0)
                assert torch.equal(rng, torch.cuda.get_rng_state())
        receipt['F_geom_full_condition_exact_parity'] = True
        receipt['training_inference_transform_shared'] = True
        receipt['feature_precision']='highest FP32; caller setting restored; head AMP unchanged'
    return receipt


def preflight():
    if (OUTPUT/'preflight.json').exists() or (OUTPUT.exists() and any(p.name != 'preflight_checks' for p in OUTPUT.iterdir())):
        raise FileExistsError(OUTPUT)
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('RTX4090 CUDA required')
    assert sha(BASELINE) == BASE_SHA
    assert subprocess.check_output(['git','branch','--show-current'], cwd=CODE,text=True).strip()==BRANCH
    assets = [BASELINE, OVERLAY, DATA/'metadata.json', DATA/'data/dataset.npz', TARGETS,
              PAIR_ARCHIVE/'setup.json', PAIR_ARCHIVE/'pair.json',PAIR_ARCHIVE/'approach_policy.json',
              PAIR_ARCHIVE/'recede_policy.json',RUNTIME/'data/homog_label_fix_20261124_v5_A_t3_w025/metadata.json',
              RUNTIME/'data/homog_label_fix_20261124_v5_A_t3_w025/data/dataset.npz']
    regression_receipt=None
    if OUTPUT != OUTPUT_ROOT:
        regression_path=OUTPUT/'preflight_checks/regression/verification.json'
        regression_receipt=read(regression_path)
        assert regression_receipt['status']=='passed'
        assert regression_receipt['baseline_sha256']==BASE_SHA
        assert regression_receipt['cache_singleton_checks']>=219
        assert regression_receipt['full_cache_shape']==[301634,384]
        assert regression_receipt['source_sha256']['scripts/research/output_head_sanity/head.py']==sha(Path(__file__).with_name('head.py'))
        assert regression_receipt['caller_precision_restored'] and regression_receipt['outer_amp_feature_parity']
        assert all(v['updates']==2 for v in regression_receipt['smokes'].values())
    evidence=read(OUTPUT/'preflight_checks/resolved_targets.json')
    assert evidence['status']=='passed' and evidence['episodes']==1039 and evidence['frames']==319697
    receipt = {'cache_repair_regression':{'path':str(OUTPUT/'preflight_checks/regression/verification.json'),
        'checks':regression_receipt['cache_singleton_checks'],'max_abs':regression_receipt['cache_singleton_max_abs']} if regression_receipt else None,
        'truth_reconstruction_preflight':{'episodes':evidence['episodes'],'frames':evidence['frames'],
        'receipt_sha256':sha(OUTPUT/'preflight_checks/resolved_targets.json')},'gpu': torch.cuda.get_device_name(0), 'python': sys.version,
        'python_executable': sys.executable, 'asset_sha256': {str(p): sha(p) for p in assets},
        'source_sha256': {str(p.relative_to(CODE)): sha(p) for directory in ('rcm','lerobot_training','mujoco_model','scripts/research/output_head_sanity')
                          for p in (CODE/directory).rglob('*.py')},
        'units_test': unit_checks(), 'baseline_config': torch.load(BASELINE,map_location='cpu',weights_only=False)['config']}
    receipt['source_sha256'].update({str(p.relative_to(CODE)):sha(p) for p in CODE.glob('*.json')})
    receipt['source_sha256'].update({str(p.relative_to(CODE)):sha(p) for p in (CODE/'mujoco_model').glob('*.json')})
    return receipt


def audit(ds, model, distance, provenance):
    action = ds.actions.astype(np.float64)
    mean, std = model.action_mean.cpu().numpy(), model.action_std.cpu().numpy()
    normalized = (action-mean)/std
    energy = np.square(normalized).sum(axis=1)
    def group(mask):
        if not np.any(mask):
            return {'frames': 0}
        return {'frames': int(mask.sum()), 'physical_abs_median_p90_max_ABC': np.quantile(np.abs(action[mask]),[.5,.9,1],axis=0),
            'normalized_abs_median_p90_max_ABC': np.quantile(np.abs(normalized[mask]),[.5,.9,1],axis=0),
            'fraction_total_frames': float(mask.mean()), 'fraction_total_normalized_energy': float(energy[mask].sum()/energy.sum())}
    starts = np.asarray(ds.valid_indices)
    result = {'units': ['A_deg','B_deg','C_mm'], 'action_mean':mean, 'action_std':std,
        'all': group(np.ones(len(action),bool)), 'true_lt60':group(distance<60),
        'baseline_training_E40_lt60':group((ds.episode_indices==40)&(distance<60)),
        'energy_definition':'squared norm of (effective overlaid action - baseline mean) / baseline std; per-frame energy, not stochastic Flow training loss',
        'observation':'Nonzero action mean, especially C, can make near-zero physical labels large normalized targets; hypothesis is tested, not assumed.',
        'provenance':provenance, 'valid_windows':len(starts)}
    # Also audit episode40 of the archived replay source, explicitly distinct.
    path = RUNTIME/'data/homog_label_fix_20261124_v5_A_t3_w025/data/dataset.npz'
    with np.load(path,allow_pickle=False) as arrays:
        mask=(arrays['episode_index']==40)&(np.linalg.norm(arrays['truth_navigation_mm'],axis=1)*1000<60)
        values=arrays['action'][mask].astype(float)
        result['paired_v5_training_E40_lt60']={'frames':len(values), 'dataset':str(path), 'sha256':sha(path),
            'physical_abs_median_p90_max_ABC':np.quantile(np.abs(values),[.5,.9,1],axis=0),
            'normalized_abs_median_p90_max_ABC':np.quantile(np.abs((values-mean)/std),[.5,.9,1],axis=0)}
    return result


def training_cache(ds, model, distance):
    from lerobot_training.dit_experience_replay import split_experience_indices, RatioReplacementBatchSampler
    original, correction = split_experience_indices(ds)
    order = np.r_[original,correction]
    starts = np.asarray(ds.valid_indices)[order]
    batches = []
    for epoch in range(4):
        batches.extend(RatioReplacementBatchSampler(len(original),len(correction),.08,1000,16,20261118+epoch))
    batches = torch.as_tensor(np.asarray(batches),device='cuda')
    # Features use only proprioception, RGB-D-derived OCT, executed-action history.
    chunks = []
    with torch.no_grad():
        for part in np.array_split(starts, max(1,len(starts)//4096)):
            hist = part[:,None]+[-1,0]
            state = torch.as_tensor(ds.states[hist],device='cuda')
            octs = np.concatenate([ds.oct_navigation[hist],ds.oct_navigation_valid[hist][...,None]],axis=-1).astype(np.float32)
            octs = torch.as_tensor(octs,device='cuda')
            velocity = ((ds.states[part]-ds.states[part-1])/(ds.timestamps_s[part]-ds.timestamps_s[part-1])[:,None]).astype(np.float32)
            previous = ds.executed_actions[part-1]
            chunks.append(geom_condition(model,state,octs,torch.as_tensor(velocity,device='cuda'),
                torch.as_tensor(previous,device='cuda')).detach())
    features = torch.cat(chunks)
    with torch.no_grad():
        for index in np.linspace(0,len(starts)-1,5,dtype=int):
            sample=ds[int(order[index])]
            args=[sample[k][None].cuda() for k in ('observation.state','observation.oct','observation.velocity','observation.previous_action')]
            expected=geom_condition(model,*args)
            torch.testing.assert_close(features[index:index+1],expected,rtol=1e-5,atol=1e-6)
    actions = torch.as_tensor(ds.actions,device='cuda')
    rows = torch.as_tensor(starts,device='cuda')[:,None]+torch.arange(16,device='cuda')
    targets = actions[rows]
    near = torch.as_tensor(distance<80,device='cuda')[rows]
    save(OUTPUT/'sampling.json', {'original_windows':len(original),'correction_windows':len(correction),
        'frozen_feature_precision':'highest FP32 in shared train/inference helper; head AMP retained',
        'cached_vs_dataset_feature_checks':5,'feature_parity_rtol':1e-5,'feature_parity_atol':1e-6,
        'batches':4000,'batch_size':16,'original_draws':58880,'correction_draws':5120,
        'ordered_rows_sha256':hashlib.sha256(starts.tobytes()).hexdigest(),
        'batch_indices_sha256':hashlib.sha256(batches.cpu().numpy().tobytes()).hexdigest(),
        'feature_cache_shape':list(features.shape), 'zero_image_loading_reason':'F_geom_cond contains no visual features; exact full-condition parity checked',
        'direction_mask':'each future training label frame true_error_um <80 and nonzero action; no truth at inference'})
    return features,targets,near,batches


def objective(head, model, features, target, near, arm):
    with torch.autocast('cuda',dtype=torch.float16):
        output = head(features)
    y = ((target-model.action_mean)/model.action_std) if arm=='R1' else symlog(target)
    regression = F.smooth_l1_loss(output.float(), y.float())
    direction = regression.new_zeros(())
    if arm == 'R2':
        mask = near & (torch.linalg.vector_norm(target,dim=-1)>1e-10)
        if mask.any():
            # Divide physical axes by fixed scales before cosine to avoid mixing deg/mm.
            direction = (1-F.cosine_similarity(symexp(output.float()[mask])/output.new_tensor(SCALE),
                target[mask]/target.new_tensor(SCALE),dim=-1,eps=1e-8)).mean()
    return regression+.1*direction, regression, direction


def train(arm, model, cache, smoke=False):
    features,targets,near,batches = cache
    torch.manual_seed(20261118)
    head = RegressionHead().cuda().train()
    init_sha=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in head.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda',enabled=True)
    start=time.monotonic(); history=[]
    for step,indices in enumerate(batches[:2] if smoke else batches):
        optimizer.zero_grad(set_to_none=True)
        loss,reg,direction=objective(head,model,features[indices],targets[indices],near[indices],arm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{arm} nonfinite loss {step}')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('nonfinite head gradient')
        before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale()<before:
            raise FloatingPointError('AMP skipped update; no retry')
        if (step+1)%500==0 or smoke:
            row={'step':step+1,'loss':float(loss.detach()),'huber':float(reg.detach()),
                'direction':float(direction.detach()),'elapsed_s':time.monotonic()-start}
            history.append(row);print(arm, 'smoke' if smoke else 'train',json.dumps(row),flush=True)
    assert not any(p.grad is not None for p in model.parameters())
    receipt={'arm':arm,'updates':step+1,'elapsed_s':time.monotonic()-start,'initial_head_sha256':init_sha,
        'history':history,'lr':1e-4,'batch_size':16,'AMP':True,'frozen_encoder_no_grad':True}
    if not smoke:
        path=OUTPUT/'weights'/f'{arm}_step004000.pth';path.parent.mkdir(exist_ok=True)
        torch.save({'head':head.state_dict(),'arm':arm,'baseline_sha256':BASE_SHA,'training':receipt},path)
        receipt.update(checkpoint=str(path),sha256=sha(path))
    return head.eval(),receipt


def paired(model_evaluator, heads):
    from . import paired_reference as pair
    source=read(PAIR_ARCHIVE/'setup.json')
    point,_=pair.training_target()
    results={}
    for direction,step in (('approach',22),('recede',26)):
        results[direction]={}
        # Rerun baseline and teacher using this branch's exact simulator implementation.
        for control in ('teacher','baseline_flow','R1','R2'):
            sample=make_sample(heads[control],model_evaluator.model,control) if control in heads else None
            ep=pair.run_episode(model_evaluator,point,'teacher' if control=='teacher' else 'policy',step,branch_sample=sample)
            row=pair.summarize(source,ep,step,'teacher' if control=='teacher' else 'policy')
            save(OUTPUT/'paired'/f'{direction}_{control}_raw.json',ep)
            row['min_error_um']=min([row['initial_error_um']]+[v['error_um'] for v in row['cycles']])
            save(OUTPUT/'paired'/f'{direction}_{control}.json',row)
            results[direction][control]={k:v for k,v in row.items() if k!='cycles'}
            record('paired_'+direction+'_'+control,'completed',{'final_um':row['error_1s_um'],
                'min_um':row['min_error_um'],'prefix_verified':True})
        archive=read(PAIR_ARCHIVE/f'{direction}_policy.json')
        archive['min_error_um']=min([archive['initial_error_um']]+[v['error_um'] for v in archive['cycles']])
        results[direction]['archived_v5_A4000_flow']={k:archive[k] for k in ('initial_error_um','min_error_um','error_1s_um')}
    return results


def flips(actions):
    values=np.asarray(actions,float)[1000:1600]
    result={}
    for axis,name in enumerate('ABC'):
        signs=np.sign(values[:,axis]);valid=(signs[1:]!=0)&(signs[:-1]!=0)
        result[name]={'flips':int(((signs[1:]!=signs[:-1])&valid).sum()),
            'eligible_adjacent_nonzero_pairs':int(valid.sum()),
            'rate':float(((signs[1:]!=signs[:-1])&valid).sum()/valid.sum()) if valid.any() else None}
    return result


class EvaluationWorker:
    def __init__(self,spec):
        self.spec=spec
        assert sha(spec['head_checkpoint'])==spec['head_sha256']
        self.evaluator=evaluator()
        payload=torch.load(spec['head_checkpoint'],map_location='cuda',weights_only=False)
        assert payload['baseline_sha256']==BASE_SHA
        self.head=RegressionHead().cuda().eval();self.head.load_state_dict(payload['head'])
        self.evaluator.model.sample=make_sample(self.head,self.evaluator.model,spec['arm'])
        self.targets=read(TARGETS)['targets']

    def run(self,task,output):
        from scripts.research.homog_replay_evaluate import episode_metrics
        from scripts.research.eval_20um_hold_metrics import trajectory_metrics
        i=task['episode_index']
        self.evaluator.evaluate(num_episodes=1, output_dir=output/'raw',target_episodes=self.targets,
            target_start=i,target_end=i+1,seed=42,max_steps=1600,control_period_s=.05,
            error_threshold_mm=.00035,progress_every=200,stop_on_success=False,max_consecutive_safety_limited=None)
        path=output/'raw/evaluation_results.json';doc=read(path);ep=doc['episodes'][0]
        assert ep['episode_index']==i and ep['seed']==42+i
        assert doc['evaluation_contract']['stop_on_success'] is False
        row={**episode_metrics(ep,i),**trajectory_metrics(path),
            'late1000_1600_raw_ABC_sign_flip':flips(ep['trajectory']['raw_actions']),
            'late1000_1600_accepted_ABC_sign_flip':flips(ep['trajectory']['actions']),
            'raw_sha256':sha(path),'arm':self.spec['arm']}
        save(output/'metrics.json',row);record(f'fast5_{self.spec["arm"]}_E{i:02d}','completed',row)
        return row

    def close(self):
        pass


def fast_evaluate(arms,train_receipts,timeout):
    from scripts.research.gpu_episode_pool import run_pool
    groups={arm:{'factory':MODULE+':EvaluationWorker','spec':{'arm':arm,
        'head_checkpoint':train_receipts[arm]['checkpoint'],'head_sha256':train_receipts[arm]['sha256']},
        'tasks':[{'id':f'E{i:02d}','episode_index':i} for i in FAST5]} for arm in arms}
    values=run_pool(groups,OUTPUT/'fast5',workers=len(arms)*5,max_workers=12,
        timeout_s=None,initializer_timeout_s=120,execution_timeout_s=timeout,
        stage_timeout_s=timeout,admission_min_ram_mib=4096)
    result={}
    for arm in arms:
        rows=sorted([v for v in values.values() if v['arm']==arm],key=lambda x:x['episode_index'])
        assert [v['episode_index'] for v in rows]==list(FAST5)
        result[arm]={'status':'completed','targets':rows,'final_mean_um':float(np.mean([v['final_error_um'] for v in rows])),
            'final_le20_count':sum(v['final_error_um']<20 for v in rows),
            'sustained200_count':sum(v['sustained_200_pass'] for v in rows),
            'strict_success_count':sum(v['strict_success'] for v in rows)}
    return result


def historical_fast5():
    from scripts.research.eval_20um_hold_metrics import trajectory_metrics
    root=RUNTIME/'github_reports/homog_label_fix_20261123_v3_s3_fast5_1600/baseline_task2_step4000/shards/tasks'
    result=[]
    for i in FAST5:
        path=root/f'eval_E{i:02d}/raw/evaluation_results.json'
        if not path.is_file():
            return {'status':'missing_historical_raw','path':str(path)}
        ep=read(path)['episodes'][0]
        result.append({'episode_index':i,**trajectory_metrics(path),
            'late1000_1600_raw_ABC_sign_flip':flips(ep['trajectory']['raw_actions']),
            'late1000_1600_accepted_ABC_sign_flip':flips(ep['trajectory']['actions']), 'raw_sha256':sha(path)})
    return {'status':'historical_reference_only','identity':'baseline_task2_step4000; scene/code equivalence not certified for reuse',
        'targets':result,'final_mean_um':float(np.mean([v['final_error_um'] for v in result]))}


def report(metrics):
    lines=['# Plan 8: deterministic output-head and action-scale sanity check','',
        f'Workflow status: **{metrics["status"]}**. Elapsed: {metrics.get("elapsed_s",0)/60:.2f} minutes.',
        '','Frozen baseline: baseline_task2_step4000, SHA256 '+BASE_SHA+'.',
        'Training: same effective baseline dataset, C-only overlay, 8% correction replacement sampling, batch16, 4000 updates per arm.',
        'Head: 384→384→384→48, LayerNorm/SiLU, AdamW lr1e-4 (baseline config general lr), decay1e-4, AMP fp16, seed20261118.',
        'Encoder eval/frozen; F_geom_cond cache parity checked with unchanged rtol1e-5/atol1e-6, no global vision fusion. Ground truth only audit/training mask/post-hoc metrics.',
        'Recovery: shared frozen feature extraction uses highest FP32 with autocast disabled; caller precision restored; output-head AMP unchanged.',
        'R2 cosine uses physical action divided by fixed per-axis symlog scales; masked per future training frame <80µm and nonzero.',
        '','## Action-scale audit','']
    if 'audit' in metrics:
        a=metrics['audit']
        for key in ('all','true_lt60','baseline_training_E40_lt60','paired_v5_training_E40_lt60'):
            lines.extend([f'### {key}','', '```json',json.dumps(a[key],ensure_ascii=False,default=lambda x:x.tolist()),'```',''])
        lines.extend(['Dataset energy counts one action per stored frame, after C-overlay; normalized action energy is not realized Flow MSE.',
            'Baseline training episode40 and v5 paired replay episode40 are different datasets/targets and are reported separately.',''])
    lines+=['## Paired replay','', '| Start | Controller | min µm | final µm |','|---|---|---:|---:|']
    for direction,values in metrics.get('paired',{}).items():
        for name,row in values.items():
            lines.append(f'| {direction} | {name} | {row["min_error_um"]:.6f} | {row["error_1s_um"]:.6f} |')
    lines+=['','Archived 63.975/59.622µm belongs to v5 A4000, not baseline_task2. Exact source prefix/RGB/history verified before each new branch.',
        '','## fast5 hold and late flips','','| Arm | E | min µm | final µm | entries ≤20 | max hold cycles | last200 max µm | late raw flip A/B/C |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    groups=dict(metrics.get('fast5',{}))
    historical=metrics.get('historical_fast5',{})
    if 'targets' in historical:groups['historical_baseline']=historical
    for arm,values in groups.items():
        for row in values.get('targets',[]):
            rates=[row['late1000_1600_raw_ABC_sign_flip'][a]['rate'] for a in 'ABC']
            lines.append(f'| {arm} | {row["episode_index"]:02d} | {row["min_error_um"]:.3f} | {row["final_error_um"]:.3f} | {row["entries_le_20_um"]} | {row["max_consecutive_le_20_um_cycles"]} | {row["last_200_max_error_um"]:.3f} | {rates} |')
    lines+=['','Repeated-identical-observation action flips are zero by deterministic construction; temporal closed-loop flips are separately measured above.',
        'Fast5 gate: both paired finals must beat both the rerun baseline_task2 and the archived v5 reference. No retries or parameter changes. Below15min prioritize best eligible paired final mean.',
        '', '## Conclusions','',json.dumps(metrics.get('conclusion',{}),ensure_ascii=False,indent=2),
        '',f'Error/skip reason: {metrics.get("error",metrics.get("fast5_skip_reason","none"))}',
        '', 'A touch below20µm, final convergence, sustained200 cycles and strict0.35µm/3-frame settled success are separate outcomes.',
        'No dev20/final-test inference. These are software/MuJoCo simulation results, not hardware/clinical accuracy.']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n')


def publish(metrics):
    repo=CODE.parent
    rel=OUTPUT.relative_to(RUNTIME)
    destination=CODE/rel;destination.mkdir(parents=True,exist_ok=True)
    # Exact allowlist: lightweight summaries only, no weights/raw trajectories/secrets.
    files=['card.json','preflight.json','units.json','sampling.json','training.json','audit.json',
           'paired.json','fast5_gate.json','fast5_summary.json','metrics.json','report.md','workflow_status.json',
           'launch_receipt.json','terminal_launch_failure.json']
    for check in (OUTPUT/'preflight_checks').rglob('*.json'):
        target=destination/'preflight_checks'/check.relative_to(OUTPUT/'preflight_checks')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(check,target)
    for name in files:
        source=OUTPUT/name
        if source.is_file():
            if source.stat().st_size>10_000_000:raise ValueError(f'Oversized report {name}')
            shutil.copy2(source,destination/name)
    shutil.copy2(RUNTIME/'TEST_LOG.md',CODE/'TEST_LOG.md')
    subprocess.run(['git','add','rcm_4090_deploy/scripts/research/output_head_sanity','rcm_4090_deploy/TEST_LOG.md'],cwd=repo,check=True,timeout=20)
    subprocess.run(['git','add','-f',str(Path('rcm_4090_deploy')/rel)],cwd=repo,check=True,timeout=20)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    for path in staged:
        if Path(path).suffix.lower() in ('.pth','.npz','.npy','.mp4') or Path(path).name.startswith('.env'):
            raise ValueError('Forbidden staged artifact '+path)
    subprocess.run(['git','-c','user.name=jsonyang218-pixel','-c','user.email=jsonyang218-pixel@users.noreply.github.com',
        'commit','-m','Record deterministic head scale sanity results'],cwd=repo,check=True,timeout=30)
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -i /home/him1/.ssh/rcm_github_deploy_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20'
    subprocess.run(['git','push','-u','origin',BRANCH],cwd=repo,env=env,check=True,timeout=70)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,env=env,text=True,timeout=30).split()[0]
    assert remote==commit
    save(OUTPUT/'publication_receipt.json',{'status':'pushed_verified','commit':commit,'remote':remote,'branch':BRANCH})
    print('Published',BRANCH,commit,flush=True)


def run():
    receipt=read(OUTPUT/'preflight.json')
    for path,expected in receipt['asset_sha256'].items():
        assert sha(path)==expected, path
    for path,expected in receipt['source_sha256'].items():
        assert sha(CODE/path)==expected, path
    start=time.monotonic();deadline=start+2100
    metrics={'status':'running','started_at':datetime.now().astimezone().isoformat(),
        'budget_s':2100,'fast5':{},'conclusion':{}}
    def status(stage):
        metrics['stage']=stage
        save(OUTPUT/'workflow_status.json',{'status':metrics['status'],'stage':stage,'elapsed_s':time.monotonic()-start})
        print('Stage:',stage,'output:',OUTPUT,flush=True)
    def alarm(signum,frame):
        raise TimeoutError('35 minute total budget reached; no retry')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(1960)  # reserve140s for reports/push
    try:
        status('audit_and_feature_cache')
        ev=evaluator();model=ev.model.eval()
        for p in model.parameters():p.requires_grad_(False)
        ds=dataset();units=unit_checks(model,ds);save(OUTPUT/'units.json',units)
        distance,provenance=target_errors(ds)
        metrics['audit']=audit(ds,model,distance,provenance);save(OUTPUT/'audit.json',metrics['audit'])
        record('audit','completed',{'near60_fraction':metrics['audit']['true_lt60']['fraction_total_frames'],
            'near60_energy_share':metrics['audit']['true_lt60']['fraction_total_normalized_energy'],'units':units})
        cache=training_cache(ds,model,distance)
        heads={};training={}
        status('two_batch_smoke')
        for arm in ('R1','R2'):
            _,smoke=train(arm,model,cache,smoke=True);record(arm+'_2batch_smoke','passed',smoke)
        for arm in ('R1','R2'):
            status('training_'+arm)
            heads[arm],training[arm]=train(arm,model,cache)
            save(OUTPUT/'training.json',training);record('train_'+arm,'completed',training[arm])
        assert training['R1']['initial_head_sha256']==training['R2']['initial_head_sha256']
        metrics['training']=training
        del cache,ds;torch.cuda.empty_cache()
        status('paired_episode40')
        metrics['paired']=paired(ev,heads);save(OUTPUT/'paired.json',metrics['paired'])
        eligible=[]
        for arm in ('R1','R2'):
            if all(v[arm]['error_1s_um'] < min(v['baseline_flow']['error_1s_um'],
                v['archived_v5_A4000_flow']['error_1s_um']) for v in metrics['paired'].values()):eligible.append(arm)
        eligible.sort(key=lambda arm:np.mean([v[arm]['error_1s_um'] for v in metrics['paired'].values()]))
        remain=deadline-time.monotonic()-140
        if eligible and remain>=900:
            selected=eligible
        elif eligible and remain>=300:
            selected=eligible[:1]
        else:
            selected=[]
        save(OUTPUT/'fast5_gate.json',{'eligible':eligible,'selected':selected,'remaining_before_reserve_s':remain})
        record('fast5_gate','passed' if selected else 'skipped',{'eligible':eligible,'selected':selected,'remaining_s':remain})
        if selected:
            status('fast5_1600')
            del ev,model,heads;torch.cuda.empty_cache()
            metrics['fast5']=fast_evaluate(selected,training,max(1,deadline-time.monotonic()-140))
            save(OUTPUT/'fast5_summary.json',metrics['fast5'])
        else:
            metrics['fast5_skip_reason']='paired gate failed' if not eligible else 'insufficient remaining bounded budget'
        metrics['historical_fast5']=historical_fast5()
        metrics['status']='completed'
        for arm in ('R1','R2'):
            values=[v[arm] for v in metrics['paired'].values()]
            fast=metrics['fast5'].get(arm)
            metrics['conclusion'][arm]={'episode40_final_lt20_both':all(v['error_1s_um']<20 for v in values),
                'episode40_min_lt20_both':all(v['min_error_um']<20 for v in values),
                'episode40_final_lt15_both':all(v['error_1s_um']<15 for v in values),
                'fast5_demonstrated_final_lt20_all':bool(fast and fast['final_le20_count']==5),
                'fast5_sustained200_all':bool(fast and fast['sustained200_count']==5),
                'fast5_executed':fast is not None,'strict_success_count':fast['strict_success_count'] if fast else None}
    except BaseException as exc:
        partial_path=OUTPUT/'fast5/partial_summary.json'
        if partial_path.is_file():
            metrics['partial_fast5']=read(partial_path)
        metrics.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,TimeoutError)) else 'failed',
            error=repr(exc),traceback=traceback.format_exc())
        traceback.print_exc();record(metrics.get('stage','unknown'),metrics['status'],{'error':repr(exc),'partial':True})
    finally:
        signal.alarm(0)
        metrics['elapsed_s']=time.monotonic()-start
        save(OUTPUT/'metrics.json',metrics);report(metrics)
        save(OUTPUT/'workflow_status.json',{'status':metrics['status'],'stage':'finished','elapsed_s':metrics['elapsed_s']})
        record('workflow',metrics['status'],{'elapsed_s':metrics['elapsed_s'],'report':str(OUTPUT/'report.md'),'conclusion':metrics['conclusion']})
        signal.alarm(max(1,int(deadline-time.monotonic())))
        try:
            publish(metrics)
        except BaseException as exc:
            save(OUTPUT/'publication_receipt.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
            record('publication','failed',{'error':repr(exc)})
            raise
        finally:
            signal.alarm(0)
    return 0 if metrics['status']=='completed' else 1


def main():
    global OUTPUT
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('preflight','run'))
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    OUTPUT=args.output_dir.resolve()
    if OUTPUT != OUTPUT_ROOT and not OUTPUT.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Output must be inside '+str(OUTPUT_ROOT))
    os.environ['RCM_OUTPUT_HEAD_SANITY_DIR']=str(OUTPUT)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    os.environ['RCM_DRIVER_VARIANT']='g10'
    os.environ['RCM_RUNTIME_ROOT']=str(RUNTIME)
    if args.stage=='preflight':
        receipt=preflight();OUTPUT.mkdir(parents=True,exist_ok=True)
        card={'name':'Plan8 deterministic output-head sanity','source_base':'9fef16daca9b496d64b06803c0bc1dc8c94f9824',
            'branch':BRANCH,'budget_minutes':35,'checkpoint':str(BASELINE),'dataset':str(DATA),'overlay':str(OVERLAY),
            'batch_size':16,'updates_per_arm':4000,'lr':1e-4,'seed':20261118,'correction_ratio':.08,
            'symlog_scales_deg_deg_mm':SCALE,'direction_weight':.1,'direction_true_error_um':80,
            'steps':1600,'fast5':FAST5,'seed_eval':'42+index','hold200_cycles':200,'strict_error_um':.35,
            'stop_on_success':False,'K':1,'execute':1,'period_s':.05,'driver':'G10','max_workers':12,
            'paired_starts':{'approach':22,'recede':26},'paired_cycles':20,'paired_seed':43001,
            'fast5_gate':'both finals beat rerun baseline_task2 AND archived v5 A4000; prioritize lower mean if <15min',
            'scope':'no teacher takeover, no privileged truth policy input; no hyperparameter search or retries',
            'baseline_note':'Archived replay flow was v5 A4000; baseline training episode40 is a different source target; report both identities',
            'paths':{'output':str(OUTPUT),'code':str(CODE)},'training_head':'three linear layers384/384/48; LayerNorm SiLU; no visual fusion',
            'unspecified_defaults':'AdamW general baseline config lr1e-4,weight_decay1e-4,clip1; frozen encoder eval; AMPfp16',
            'feature_numeric_policy':'shared geometry extractor highest FP32 with autocast disabled; restore caller precision; head AMP unchanged',
            'recovery_of':str(OUTPUT_ROOT/'metrics.json') if OUTPUT != OUTPUT_ROOT else None}
        save(OUTPUT/'card.json',card);save(OUTPUT/'preflight.json',receipt)
        shutil.copy2(RUNTIME.parent/'规划8.txt',OUTPUT/'input_plan.txt')
        record('preflight','passed',{'gpu':receipt['gpu'],'unit_checks':receipt['units_test'],'source':str(CODE)})
        print('PREFLIGHT PASS',OUTPUT,flush=True)
    else:
        from scripts.research.gpu_episode_pool import Tee
        with (OUTPUT/'workflow.log').open('x',buffering=1) as log:
            sys.stdout=Tee(sys.__stdout__,log);sys.stderr=Tee(sys.__stderr__,log)
            try:
                code=run()
            finally:
                sys.stdout=sys.__stdout__;sys.stderr=sys.__stderr__
        return code
    return 0


if __name__=='__main__':
    sys.exit(main())
