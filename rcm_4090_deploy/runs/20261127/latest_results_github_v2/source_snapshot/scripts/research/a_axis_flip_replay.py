"""Independent ABC overlays or same-observation correction trajectories."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from lerobot_training.dataset_contract import valid_window_indices, validate_dataset_contract
from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_common import (
    DATA, EPS, OLD, QUALITY, TARGETS, UNIT, apply_abc_overlay, arrays, atomic_npz,
    gpu_check, near_mask, pool, read, record, write_json,
)
from scripts.research.a_axis_flip_audit import make_teacher
from scripts.research.clean_oct_workers import egl_identity, resolve_target, target_world


def windows(a):
    return valid_window_indices(a['episode_index'], a['source_frame_index'], a['timestamp_s'],
        n_obs_steps=2, n_action_steps=16, control_period_s=.05,
        oct_navigation=a['oct_navigation'], oct_navigation_valid=a['oct_navigation_valid'],
        return_diagnostics=True)


def build_abc_overlay(audit, destination):
    """Replace reverse, historically inertia-compatible axes, preserving other labels."""
    destination.mkdir(parents=True, exist_ok=True)
    a = arrays()
    with np.load(audit['arrays_path'], allow_pickle=False) as f:
        v = {key: f[key].copy() for key in f.files}
    if str(v['dataset_sha256']) != file_sha256(DATA / 'data/dataset.npz'):
        raise ValueError('source changed after audit')
    changes = (v['inertia_compatible'] & v['valid'][:, None] & v['toward'][:, None]
               & (np.abs(v['zero']) <= np.array([.2, .1, .03]) + 1e-8))
    proposals = np.where(changes, v['zero'], a['action'][v['rows']])
    forward = np.sum(proposals * v['first_order_gradient'], axis=1) >= -1e-14
    select = changes.any(1) & forward
    rows = v['rows'][select]
    before = a['action'][rows].copy()
    after = np.where(changes[select], v['zero'][select], before).astype(np.float32)
    if not len(rows):
        raise ValueError('Branch A has no safe history-replay-compatible ABC repairs; no training launched')
    path = destination / 'abc_override.npz'
    atomic_npz(path, rows=rows, old_action=before, new_action=after,
               changed_axes=changes[select], dataset_sha256=file_sha256(DATA / 'data/dataset.npz'))
    checked = apply_abc_overlay(a['action'], path, file_sha256(DATA / 'data/dataset.npz'))
    np.testing.assert_array_equal(checked[rows], after)
    before_windows, diagnostics = windows(a)
    a['action'] = checked
    after_windows, _ = windows(a)
    np.testing.assert_array_equal(before_windows, after_windows)
    # Replay windows must contain at least one actually repaired label in their future block.
    changed_frame = np.zeros(len(a['state']), np.int64)
    changed_frame[rows] = 1
    cumulative = np.r_[0, np.cumsum(changed_frame)]
    correction = after_windows[(cumulative[after_windows+16] - cumulative[after_windows]) > 0]
    if not len(correction):
        raise ValueError('No valid 16-step correction windows after ABC repair')
    atomic_npz(destination / 'correction_windows.npz', rows=correction)
    report = {'status': 'completed', 'branch': 'A', 'overlay': str(path),
              'overlay_sha256': file_sha256(path), 'source_dataset': str(DATA),
              'source_dataset_sha256': file_sha256(DATA / 'data/dataset.npz'),
              'corrected_frames': len(rows), 'corrected_axes': changes[select].sum(0).tolist(),
              'rejected_nonforward_composed_frames': int(np.sum(changes.any(1) & ~forward)),
              'correction_windows': len(correction), 'window_diagnostics': diagnostics,
              'label_rule': 'Only zero-velocity/pre-acceleration-reversed axes whose finite-difference history replay reproduces reversal. Other axes and executed-action history unchanged.',
              'correction_window_path': str(destination / 'correction_windows.npz'),
              'original_pool': 'All unmodified valid windows of merged base dataset, 92%; independent repaired replay 8%.'}
    write_json(destination / 'dataset_receipt.json', report)
    record('ABC_overlay', 'completed', destination, report)
    return report


def collection_tasks():
    a = arrays()
    near = near_mask(a)
    tasks = []
    development = read(TARGETS)['targets']
    dev_robot = np.asarray([t.get('reachable_target_robot_mm', t['point_mm']) for t in development])
    for i in range(839, 1039):
        rows = np.flatnonzero(near & (a['episode_index'] == i))
        if not len(rows):
            continue
        target = resolve_target(DATA, i)
        # Explicit target provenance is frozen below; geometry overlap is checked at runtime too.
        if target['kind'] == 'robot_mm' and np.any(np.linalg.norm(dev_robot-target['value'], axis=1) < 1e-6):
            continue
        distance = np.linalg.norm(a['oct_navigation'][rows], axis=1)
        candidates = rows[distance >= .15]
        row = int(candidates[0] if len(candidates) else rows[0])
        tasks.append({'id': f'E{len(tasks):02d}', 'episode_index': len(tasks),
                      'training_source_episode': i, 'source_row': row,
                      'initial_state_deg_deg_mm': a['state'][row].tolist(),
                      'seed': 2026112300 + len(tasks), 'target': target})
        if len(tasks) == 50:
            return tasks
    raise ValueError('Need 50 eligible training targets distinct from development targets')


class RecordingSensor:
    """Wrap the very sensor observed by the old policy; never render a second label frame."""
    def __init__(self, teacher, scene, evaluator, output):
        self.teacher, self.scene, self.evaluator = teacher, scene, evaluator
        self.sensor = teacher.sensor
        self.output = output
        self.rows = []
        self.pending = None
        self.index = 0
        self.image_parity = None
        (output / 'images').mkdir()
        self.stream = (output / 'frames.jsonl').open('w')

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
        self.pending = None
        nav = obs.get('oct_navigation_depth')
        delta = None if nav is None else nav.get('delta_xyz_m')
        if delta is not None and not np.isfinite(delta).all():
            raise FloatingPointError('nonfinite collection OCT')
        if delta is not None and np.linalg.norm(delta)*1000 < .2:
            q = self.scene.current_task().copy()
            z = self.teacher.geometry_action_from_observation(q, obs,
                previous_velocity=np.zeros(3), command_q=q)
            if z['valid']:
                action = np.asarray(z['action_deg_deg_mm'])
                if not np.isfinite(action).all():
                    raise FloatingPointError('nonfinite correction label')
                if np.any(np.abs(action) > np.array([.2, .1, .03]) + 1e-8):
                    raise ValueError('teacher exceeds frozen correction caps')
                image = Image.fromarray(obs['rgb'].astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
                name = f'images/frame_{self.index:06d}.png'
                image.save(self.output / name)
                if self.image_parity is None:
                    stored = self.evaluator.transform(image)
                    direct = self.evaluator.preprocess_image(obs['rgb'])
                    self.image_parity = float((stored-direct).abs().max())
                    torch.testing.assert_close(stored, direct, rtol=0, atol=0)
                self.pending = {'source_frame_index': self.index,
                    'timestamp_s': float(self.observation_time_s), 'state': (q / UNIT).tolist(),
                    'action': action.tolist(), 'oct_navigation': (np.asarray(delta)*1000).tolist(),
                    'image_path': name}
        self.index += 1
        return obs

    def after_step(self, event):
        if self.pending is not None:
            if self.pending['source_frame_index'] != event['step']:
                raise ValueError('label observation/action step misalignment')
            row = {**self.pending, 'executed_action': event['accepted_action_deg_deg_mm']}
            self.rows.append(row)
            self.stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.stream.flush()
            self.pending = None

    def close(self):
        self.stream.close()
        self.sensor.close()


class CollectionWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        gpu_check()
        from lerobot_training.eval_policy import PolicyEvaluator
        self.spec = spec
        if file_sha256(OLD) != spec['checkpoint_sha256']:
            raise ValueError('old checkpoint changed')
        self.evaluator = PolicyEvaluator(str(OLD), device='cuda', num_action_samples=1,
            n_action_execute_steps=1, flow_integration_steps=4, measure_inference=True,
            controller_variant='feedback', rgbd_render_quality=QUALITY.copy())
        if not self.evaluator.direct_224_preprocessing:
            raise ValueError('collection requires original direct224 preprocessing')

    def run(self, task, output):
        from rcm.scene_v9 import RCMSceneV9
        scene = RCMSceneV9()
        recorder = None
        try:
            q = np.asarray(task['initial_state_deg_deg_mm']) * UNIT
            world = target_world(scene, task['target'], q)
            scene.set_target_world(world)
            scene.set_task(q)
            # Keep collection and development target sets disjoint, including scanned-world targets.
            dev = read(TARGETS)['targets']
            dev_world = np.asarray([scene.robot_to_world(np.asarray(t.get('reachable_target_robot_mm', t['point_mm']))) for t in dev])
            if np.min(np.linalg.norm(dev_world-world, axis=1)) < 1e-6:
                raise ValueError('training target overlaps development target')
            teacher = make_teacher(scene)
            recorder = RecordingSensor(teacher, scene, self.evaluator, output)
            egl = egl_identity()
            self.evaluator.step_observer = recorder.after_step
            ep = self.evaluator.evaluate_episode(scene, world, max_steps=1600,
                error_threshold_mm=.00035, control_period_s=.05, seed=task['seed'],
                sensor=recorder, progress_every=100, stop_on_success=False,
                max_consecutive_safety_limited=None)
            if len(ep['trajectory']['actions']) != 1600 or ep['early_stop_reason'] is not None:
                raise ValueError('collection ended in illegal/incomplete state')
            write_json(output / 'policy_trajectory.json', ep)
            rows = recorder.rows
            if not rows:
                raise ValueError('no valid near-target same-observation frames collected')
            a = {key: np.asarray([r[key] for r in rows], dtype=dtype) for key, dtype in (
                ('state', np.float32), ('action', np.float32), ('oct_navigation', np.float32),
                ('executed_action', np.float32), ('source_frame_index', np.int64), ('timestamp_s', np.float64))}
            a.update(episode_index=np.full(len(rows), task['episode_index'], np.int64),
                     frame_index=np.arange(len(rows), dtype=np.int32),
                     oct_navigation_valid=np.ones(len(rows), bool))
            validate_dataset_contract(a, require_history=True)
            indices, diagnostics = windows(a)
            if not len(indices):
                raise ValueError('collection shard has no valid 2-observation/16-action windows')
            atomic_npz(output / 'dataset.npz', **a)
            write_json(output / 'image_paths.json', [r['image_path'] for r in rows])
            result = {'episode_index': task['episode_index'], 'frames': len(rows), 'actions': 1600,
                      'valid_windows': len(indices), 'window_diagnostics': diagnostics,
                      'dataset_path': str(output / 'dataset.npz'),
                      'dataset_sha256': file_sha256(output / 'dataset.npz'), 'egl': egl,
                      'task': task, 'stored_vs_policy_image_tensor_max_abs': recorder.image_parity,
                      'observation_provenance': 'RGB/OCT/zero-velocity ABC labels all from one actual policy sensor.observe call per retained frame; executed_action is the accepted command, never teacher label.',
                      'source_frame_gaps_preserved': True}
            write_json(output / f"E{task['episode_index']:02d}.json", result)
            record('same_observation_collection', 'completed', output, result)
            return result
        finally:
            self.evaluator.step_observer = None
            if recorder is not None:
                recorder.close()
            scene.close()

    def close(self):
        pass


def collect_dataset(destination, report_dir, tasks, identity):
    groups = {'collection': {'factory': __name__ + ':CollectionWorker',
        'spec': {'checkpoint_sha256': file_sha256(OLD), 'identity': identity,
                 'steps': 1600, 'quality': QUALITY, 'dataset_sha256': file_sha256(DATA / 'data/dataset.npz')},
        'tasks': tasks}}
    results = pool(groups, report_dir / 'collection_shards')
    if len(results) != 50:
        raise ValueError('requires exactly 50 complete correction trajectories')
    blocks, images, episodes = [], [], []
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'data').mkdir(exist_ok=True)
    for value in sorted(results.values(), key=lambda x: x['episode_index']):
        path = Path(value['dataset_path'])
        if file_sha256(path) != value['dataset_sha256']:
            raise ValueError('collection dataset shard changed')
        with np.load(path, allow_pickle=False) as f:
            blocks.append({key: f[key].copy() for key in f.files})
        paths = read(path.parent / 'image_paths.json')
        images.extend([os.path.relpath(path.parent / p, destination) for p in paths])
        task = value['task']
        episodes.append({'episode_index': value['episode_index'], 'num_frames': value['frames'],
                         'action_source': 'teacher', 'target_provenance': task['target'],
                         'source_dataset': str(DATA), 'source_episode_index': task['training_source_episode']})
    a = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
    validate_dataset_contract(a, require_history=True)
    index, diagnostics = windows(a)
    if not len(index):
        raise ValueError('combined correction data has no valid windows')
    atomic_npz(destination / 'data/dataset.npz', **a)
    write_json(destination / 'data/image_paths.json', images)
    write_json(destination / 'metadata.json', {'num_episodes': 50, 'num_frames': len(images),
        'fps': 20, 'data_contract_version': 2, 'episodes': episodes,
        'observation_space': {'image': {'shape': [224, 224, 3], 'dtype': 'uint8'}},
        'observation_provenance': 'Same new-render RGB-D observation drives policy and zero-velocity label',
        'action_source': 'teacher', 'history_action_source': 'executed_action', 'render_quality': QUALITY})
    result = {'status': 'completed', 'branch': 'B', 'data_dir': str(destination),
              'dataset_sha256': file_sha256(destination / 'data/dataset.npz'),
              'trajectories': 50, 'frames': len(images), 'valid_windows': len(index),
              'window_diagnostics': diagnostics, 'dagger_repeat': 1}
    write_json(destination / 'dataset_receipt.json', result)
    record('same_observation_dataset', 'completed', destination, result)
    return result
