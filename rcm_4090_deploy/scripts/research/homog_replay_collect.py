"""Sixty isolated targets, same-call RGB/OCT labels and independent quality gates."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lerobot_training.dataset_contract import validate_dataset_contract
from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_audit import pre_acceleration, static_scene
from scripts.research.a_axis_flip_replay import windows
from scripts.research.clean_oct_workers import egl_identity
from scripts.research.homog_replay_target import TeacherIntent
from scripts.research.homog_replay_common import (
    CAPS, COLLECTION_STEPS, DATA, HOLD_START, OLD, QUALITY, REPORTS, STEPS, TARGETS, UNIT,
    atomic_npz, check_identity, gpu_check, make_teacher, pool, quality_gate,
    collection_steps, read, record, signature, write_json,
)


def collection_tasks():
    """Deterministic fundus targets; no development target supplies supervision."""
    from rcm.fundus import make_random_fundus_targets
    scene = static_scene()
    dev = read(TARGETS)['targets']
    forbidden = np.asarray([scene.robot_to_world(np.asarray(t.get(
        'reachable_target_robot_mm', t['point_mm']))) for t in dev])
    tasks, targets = [], []
    try:
        for i in range(60):
            seed = 1000+i
            rng = np.random.default_rng(seed)
            target = make_random_fundus_targets(scene, rng, 1)['targets'][0]
            robot = np.asarray(target['point_mm'])
            world = scene.robot_to_world(robot)
            if np.min(np.linalg.norm(forbidden-world, axis=1)) < .001:
                raise ValueError(f'Training seed {seed} overlaps development within1um')
            if targets and np.min(np.linalg.norm(np.asarray(targets)-world, axis=1)) < .001:
                raise ValueError('Duplicate training targets within1um')
            targets.append(world)
            scene.set_target_world(world)
            goal = scene.target_task(validate=True)
            regime = 'T1' if i < 20 else 'T2' if i < 40 else 'T3'
            if regime == 'T1':
                start = goal.copy()
                start[2] -= .020
            else:
                # Antithetic ABC signs for each adjacent pair; equal magnitude
                # per pair, normalized by the local kinematic Jacobian.
                j = i-20 if regime == 'T2' else i-40
                pair_rng = np.random.default_rng(2026112300+j//2)
                vector = pair_rng.uniform(.25, 1., 3) * pair_rng.choice([-1., 1.], 3)
                vector *= 1. if j % 2 == 0 else -1.
                teacher = None
                from rcm.visual_servo import VisualServoController
                geometry = VisualServoController.__new__(VisualServoController)
                geometry.reference = scene
                jacobian = geometry._model_jacobian(goal)
                dq = vector * UNIT
                radius_um = float(pair_rng.uniform(10., 195.) if regime == 'T2'
                                  else pair_rng.uniform(10., 45.))
                dq *= (radius_um/1000) / np.linalg.norm(jacobian @ dq)
                start = goal+dq
            scene.validate_task(start)
            scene.set_task(start)
            error_um = float(np.linalg.norm(scene.site_mm('needle_tip')-world)*1000)
            if regime == 'T2' and not 10 <= error_um <= 200:
                raise ValueError(f'T2 shell constraint failed: {error_um}')
            if regime == 'T3' and not error_um < 50:
                raise ValueError(f'T3 initial error too large: {error_um}')
            tasks.append({'id': f'E{i:02d}', 'episode_index': i, 'seed': seed,
                'regime': regime, 'target_robot_mm': robot.tolist(),
                'target_world_mm': world.tolist(), 'initial_state_si': start.tolist(),
                'goal_state_si': goal.tolist(), 'initial_error_um': error_um,
                'abc_offset_sign': np.sign(start-goal).astype(int).tolist(),
                'min_development_distance_um': float(np.min(np.linalg.norm(forbidden-world, axis=1))*1000)})
        return tasks
    finally:
        scene.close()


def expected_label(teacher, scene, q, observation):
    zero = teacher.geometry_action_from_observation(q, observation,
        previous_velocity=np.zeros(3), command_q=q)
    if not zero['valid']:
        return None
    pre, jacobian, error = pre_acceleration(teacher, scene, q, observation, zero)
    label = np.clip(pre, -CAPS, CAPS)
    scene.validate_task(q+label*UNIT)
    if not np.isfinite(label).all() or float((jacobian @ (label*UNIT)) @ error) < -1e-14:
        raise ValueError('Invalid/nonforward pre-acceleration label')
    return {'action': label, 'zero': np.asarray(zero['action_deg_deg_mm']),
            'pre': pre, 'measured_error_um': zero['measured_error_um']}


class RecordingSensor:
    """One real observe() per policy tick; all three modalities share it."""
    def __init__(self, teacher, scene, evaluator, task, output):
        self.teacher, self.scene, self.evaluator = teacher, scene, evaluator
        self.sensor, self.task, self.output = teacher.sensor, task, output
        self.rows, self.index, self.pending = [], 0, None
        self.teacher_command = None
        self.teacher_intent = TeacherIntent(scene, CAPS) if task['regime'] != 'T1' else None
        self.execution_diagnostics = None
        self.invalid_reasons = Counter()
        (output/'images').mkdir()
        self.stream = (output/'frames.jsonl').open('w', buffering=1)
        self.image_parity = None

    @property
    def observation_time_s(self):
        return self.sensor.observation_time_s

    @observation_time_s.setter
    def observation_time_s(self, value):
        self.sensor.observation_time_s = value

    def observe(self):
        from PIL import Image
        import torch
        obs = self.sensor.observe()
        self.pending, self.teacher_command = None, None
        q = self.scene.current_task().copy()
        nav = obs.get('oct_navigation_depth')
        delta = None if nav is None else nav.get('delta_xyz_m')
        if delta is not None and not np.isfinite(delta).all():
            raise FloatingPointError('Nonfinite collection OCT')
        near = delta is not None and np.linalg.norm(delta)*1000 < .2
        label = expected_label(self.teacher, self.scene, q, obs) if near else None
        regime = self.task['regime']
        if regime != 'T1':
            if label is None:
                raise ValueError('Expert collection lost valid near-target strict RGB-D')
            execution = self.teacher.geometry_action_from_observation(q, obs,
                previous_velocity=self.scene.motor_execution.actual_velocity, command_q=q)
            if not execution['valid']:
                raise ValueError(f'Invalid teacher execution: {execution}')
            hold = regime == 'T3' and self.index >= HOLD_START
            increment = np.asarray(execution['action_deg_deg_mm'])
            if hold:
                if label['measured_error_um'] >= 5:
                    raise ValueError('T3 did not reach measured<5um before the200 explicit hold frames')
                increment = np.zeros(3)
            self.teacher_command, self.execution_diagnostics = self.teacher_intent.preview(
                q, increment, hold=hold)
        if label is not None:
            image = Image.fromarray(obs['rgb'].astype(np.uint8)).resize(
                (224, 224), Image.Resampling.BILINEAR)
            name = f'images/frame_{self.index:06d}.png'
            image.save(self.output/name)
            if self.image_parity is None:
                direct = self.evaluator.preprocess_image(obs['rgb'])
                stored = self.evaluator.transform(image)
                torch.testing.assert_close(stored, direct, rtol=0, atol=0)
                self.image_parity = float((stored-direct).abs().max())
            camera = self.scene.data.cam_xmat[self.scene.camera_id('eye_internal')].reshape(3,3)
            truth = (camera.T @ (np.asarray(self.task['target_world_mm'])-
                     self.scene.site_mm('needle_tip'))) * [1, -1, -1]
            action = label['action'].copy()
            hold = regime == 'T3' and self.index >= HOLD_START
            if hold:
                action[:] = 0
            raw_rgb_sha = hashlib.sha256(obs['rgb'].tobytes()).hexdigest()
            row = {'source_frame_index': self.index, 'timestamp_s': float(self.observation_time_s),
                'state': (q/UNIT).tolist(), 'state_si': q.tolist(), 'action': action.tolist(),
                'zero': label['zero'].tolist(), 'pre': label['pre'].tolist(),
                'oct_navigation': (np.asarray(delta)*1000).tolist(), 'truth_navigation_mm': truth.tolist(),
                'image_path': name, 'image_sha256': file_sha256(self.output/name),
                'raw_rgb_sha256': raw_rgb_sha, 'quality': QUALITY,
                'hold': hold, 'measured_error_um': label['measured_error_um'],
                'regime': regime, 'observation_calls': 1}
            if self.teacher_intent is not None:
                row['teacher_execution'] = self.execution_diagnostics
            row['observation_token'] = signature(row)
            self.pending = row
        elif near:
            self.invalid_reasons['invalid_teacher'] += 1
        self.index += 1
        return obs

    def after_step(self, event):
        if self.teacher_intent is not None:
            self.teacher_intent.commit(event['accepted_action_deg_deg_mm'])
        if self.pending is not None:
            if self.pending['source_frame_index'] != event['step']:
                raise ValueError('Observation/accepted-action alignment mismatch')
            row = {**self.pending, 'executed_action': event['accepted_action_deg_deg_mm']}
            self.rows.append(row)
            self.stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.pending = None

    def close(self):
        self.stream.close()
        self.sensor.close()


class CollectionWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        gpu_check()
        check_identity(spec['identity'])
        self.spec = spec
        if file_sha256(OLD) != spec['checkpoint_sha256']:
            raise ValueError('Initial checkpoint changed')
        from lerobot_training.eval_policy import PolicyEvaluator
        self.evaluator = PolicyEvaluator(str(OLD), device='cuda', num_action_samples=1,
            n_action_execute_steps=1, flow_integration_steps=4,
            measure_inference=False, controller_variant='feedback', rgbd_render_quality=QUALITY.copy())
        self.policy_action = self.evaluator.get_action
        if not self.evaluator.direct_224_preprocessing:
            raise ValueError('Expected direct224 checkpoint preprocessing')

    def run(self, task, output):
        steps = collection_steps(task['regime'])
        scene = static_scene()
        recorder = None
        try:
            scene.set_target_world(np.asarray(task['target_world_mm']))
            scene.set_task(np.asarray(task['initial_state_si']))
            teacher = make_teacher(scene)
            recorder = RecordingSensor(teacher, scene, self.evaluator, task, output)
            egl = egl_identity()
            self.evaluator.get_action = (self.policy_action if task['regime']=='T1'
                else lambda *_: recorder.teacher_command.copy())
            self.evaluator.step_observer = recorder.after_step
            ep = self.evaluator.evaluate_episode(scene, np.asarray(task['target_world_mm']),
                max_steps=steps, control_period_s=.05, seed=task['seed'],
                sensor=recorder, progress_every=100, stop_on_success=False,
                max_consecutive_safety_limited=None)
            if len(ep['trajectory']['actions']) != steps or ep['early_stop_reason'] is not None:
                raise ValueError('Incomplete/illegal collection trajectory')
            rows = recorder.rows
            if not rows:
                raise ValueError('No near-target observations')
            a = {k: np.asarray([r[k] for r in rows], dtype=d) for k,d in (
                ('state',np.float64), ('state_si',np.float64), ('action',np.float32),
                ('executed_action',np.float32), ('oct_navigation',np.float32),
                ('truth_navigation_mm',np.float64), ('zero',np.float64), ('pre',np.float64),
                ('source_frame_index',np.int64), ('timestamp_s',np.float64), ('hold',bool))}
            a.update(episode_index=np.full(len(rows),task['episode_index'],np.int64),
                frame_index=np.arange(len(rows),dtype=np.int32),
                oct_navigation_valid=np.ones(len(rows),bool))
            validate_dataset_contract(a, require_history=True)
            index, diagnostics = windows(a)
            if not len(index):
                raise ValueError('No valid2-observation/16-action contiguous windows')
            atomic_npz(output/'dataset.npz', **a)
            write_json(output/'image_paths.json',[r['image_path'] for r in rows])
            write_json(output/'policy_trajectory.json',ep)
            result = {'episode_index':task['episode_index'], 'regime':task['regime'],
                'frames':len(rows),'valid_windows':len(index),'actions':steps,
                'dataset_path':str(output/'dataset.npz'), 'dataset_sha256':file_sha256(output/'dataset.npz'),
                'final_error_um':ep['final_error_mm']*1000,
                'hold_frames':int(a['hold'].sum()), 'window_diagnostics':diagnostics,
                'task':task,'egl':egl,'stored_vs_policy_image_tensor_max_abs':recorder.image_parity,
                'RGB_OCT_render_provenance_mismatch':False, 'quality':QUALITY,
                'label':'Same observe() zero-velocity geometry, pre-acceleration safe expectation; no truth enters labels',
                'actual_history':'Accepted motor command stored separately as executed_action'}
            result['expert_execution'] = ('old_student_feedback' if task['regime'] == 'T1'
                else 'bounded_teacher_intent_v1')
            write_json(output/f"E{task['episode_index']:02d}.json",result)
            record('collection_episode','completed',output,result)
            return result
        finally:
            self.evaluator.get_action = self.policy_action
            self.evaluator.step_observer = None
            if recorder is not None:
                recorder.close()
            scene.close()

    def close(self):
        pass


class AuditWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT']='g10'
        gpu_check()
        check_identity(spec['identity'])

    def run(self, task, output):
        scene=static_scene()
        teacher=None
        try:
            scene.set_target_world(np.asarray(task['target_world_mm']))
            teacher=make_teacher(scene)
            errors=[]
            for entry in task['frames']:
                q=np.asarray(entry['state_si'])
                scene.set_task(q)
                teacher.sensor.observation_time_s=entry['timestamp_s']
                obs=teacher.sensor.observe()
                nav=obs.get('oct_navigation_depth')
                delta=None if nav is None else nav.get('delta_xyz_m')
                if delta is None or not np.isfinite(delta).all():
                    raise ValueError('Independent sampled render has invalid OCT')
                camera=scene.data.cam_xmat[scene.camera_id('eye_internal')].reshape(3,3)
                truth=(camera.T@(np.asarray(task['target_world_mm'])-scene.site_mm('needle_tip')))*[1,-1,-1]
                label=expected_label(teacher,scene,q,obs)
                if label is None:
                    raise ValueError('Independent sample has invalid teacher label')
                np.testing.assert_allclose(np.asarray(delta)*1000,entry['oct_navigation'],atol=1e-6,rtol=0)
                np.testing.assert_allclose(label['pre'],entry['pre'],atol=1e-7,rtol=1e-5)
                errors.append(float(np.linalg.norm(np.asarray(delta)*1000-truth)*1000))
            result={'episode_index':task['episode_index'],'sampled_frames':len(errors),
                'oct_truth_error_um':errors,'quality':QUALITY,'egl':egl_identity()}
            write_json(output/f"E{task['episode_index']:02d}.json",result)
            return result
        finally:
            if teacher is not None: teacher.sensor.close()
            scene.close()

    def close(self):
        pass


def axis_audit(a):
    zero, pre, action = a['zero'],a['pre'],a['action']
    eligible=(np.abs(zero)>1e-10)&(np.abs(pre)>1e-10)
    reverse=eligible&(action*zero<0)&(action*pre<0)
    return {axis:{'eligible_nonzero_geometry_frames':int(eligible[:,j].sum()),
        'reverse_frames':int(reverse[:,j].sum()),
        'reverse_fraction':float(reverse[:,j].sum()/max(1,eligible[:,j].sum()))}
        for j,axis in enumerate('ABC')}


def collect(manifest):
    report=REPORTS[0]
    report.mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'data').mkdir(exist_ok=True)
    results=pool({'collection':{'factory':__name__+':CollectionWorker',
        'spec':{'identity':manifest['identity'],'checkpoint_sha256':file_sha256(OLD),
                'quality':QUALITY,'steps_by_regime':COLLECTION_STEPS},'tasks':manifest['collection_tasks']}},DATA/'shards')
    if len(results)!=60:
        raise ValueError('Requires60 completed trajectories')
    blocks,images,episodes,audit_entries=[],[],[],[]
    for result in sorted(results.values(),key=lambda r:r['episode_index']):
        path=Path(result['dataset_path'])
        if file_sha256(path)!=result['dataset_sha256']:
            raise ValueError('Collection shard changed')
        with np.load(path,allow_pickle=False) as f:
            blocks.append({k:f[k].copy() for k in f.files})
        entries=[json.loads(line) for line in (path.parent/'frames.jsonl').read_text().splitlines()]
        if len(entries)!=result['frames']:
            raise ValueError('Frame evidence count mismatch')
        for row in entries:
            token=row['observation_token']
            original={k:v for k,v in row.items() if k not in {'executed_action','observation_token'}}
            if (signature(original)!=token or row['quality']!=QUALITY or row['observation_calls']!=1
                    or file_sha256(path.parent/row['image_path'])!=row['image_sha256']):
                raise ValueError('Observation provenance/image evidence changed')
        images.extend([os.path.relpath(path.parent/r['image_path'],DATA) for r in entries])
        audit_entries.extend(entries)
        episodes.append({'episode_index':result['episode_index'],'num_frames':result['frames'],
            'regime':result['regime'],'target_robot_mm':result['task']['target_robot_mm'],
            'seed':result['task']['seed'],'action_source':'same_observation_zero_velocity_pre_acceleration_teacher'})
    a={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    validate_dataset_contract(a,require_history=True)
    valid,diagnostics=windows(a)
    if len(valid)==0:
        raise ValueError('Empty combined replay pool')
    # Exact continuity of every retained training window, not fabricated frame renumbering.
    continuous=all(np.all(np.diff(a['source_frame_index'][i-1:i+16])==1)
        and np.allclose(np.diff(a['timestamp_s'][i-1:i+16]),.05,rtol=1e-3,atol=1e-6)
        and len(set(a['episode_index'][i-1:i+16]))==1 for i in valid)
    atomic_npz(DATA/'data/dataset.npz',**a)
    atomic_npz(DATA/'data/valid_window_indices.npz',rows=valid)
    write_json(DATA/'data/image_paths.json',images)
    write_json(DATA/'metadata.json',{'num_episodes':60,'num_frames':len(images),'fps':20,
        'data_contract_version':2,'episodes':episodes,'action_source':'teacher',
        'history_action_source':'executed_action','render_quality':QUALITY,
        'observation_space':{'image':{'shape':[224,224,3],'dtype':'uint8'}},
        'RGB_OCT_render_provenance_mismatch':False})
    # Sample200 rows uniformly over the complete retained package before inspecting errors.
    selected=np.sort(np.random.default_rng(2026112305).choice(len(a['state']),200,replace=False))
    tasks=[]
    for ep in range(60):
        rows=selected[a['episode_index'][selected]==ep]
        if len(rows):
            tasks.append({'id':f'E{ep:02d}','episode_index':ep,
                'target_world_mm':manifest['collection_tasks'][ep]['target_world_mm'],
                'frames':[{k:audit_entries[int(i)][k] for k in
                    ('state_si','timestamp_s','oct_navigation','pre')} for i in rows]})
    sampled=pool({'audit':{'factory':__name__+':AuditWorker',
        'spec':{'identity':manifest['identity'],'quality':QUALITY},'tasks':tasks}},report/'audit_shards')
    errors=[e for r in sampled.values() for e in r['oct_truth_error_um']]
    axes=axis_audit(a)
    regimes=Counter(r['regime'] for r in results.values())
    hold=int(a['hold'].sum())
    audit={'status':'completed','sampled_frames':len(errors),
        'sample_seed':2026112305,'sampled_row_indices':selected.tolist(),
        'oct_truth_median_error_um':float(np.median(errors)),
        'oct_truth_max_error_um':float(np.max(errors)),
        'axis_reverse_fractions':{k:v['reverse_fraction'] for k,v in axes.items()},
        'axes':axes,'RGB_OCT_render_provenance_mismatch':False,
        'valid_window_contract_pass_fraction':float(continuous),'window_diagnostics':diagnostics,
        'regimes':dict(regimes),'all_regimes_complete':dict(regimes)=={'T1':20,'T2':20,'T3':20},
        'T3_hold_frames_below_5um':hold,'render_quality':QUALITY,
        'frames':len(images),'valid_windows':len(valid),
        'provenance':'All RGB/OCT/ABC labels from one verified observe() per frame;200 independent new renders checked',
        'dataset_sha256':file_sha256(DATA/'data/dataset.npz')}
    audit['gate_passed']=quality_gate(audit)
    write_json(report/'ab_label_contamination.json',axes)
    write_json(report/'dataset_audit.json',audit)
    write_json(DATA/'dataset_audit.json',audit)
    result={'status':'completed' if audit['gate_passed'] else 'gate_failed',
        'branch':'H','data_dir':str(DATA),'dataset_sha256':audit['dataset_sha256'],
        'audit':audit,'dagger_repeat':1}
    write_json(report/'summary.json',result)
    write_json(DATA/'dataset_receipt.json',result)
    record('collection_and_audit',result['status'],report,result)
    return result
