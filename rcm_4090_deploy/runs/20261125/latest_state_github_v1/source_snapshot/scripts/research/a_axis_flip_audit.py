"""Full-coverage label audit and paired, fixed-noise CUDA snapshot audit."""
from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

import numpy as np

from lerobot_training.navigation_overlay import apply_dataset_overlays, file_sha256
from scripts.research.a_axis_flip_common import (
    AUDIT, CLEAN, DATA, EPS, FROZEN, NEW, OLD, QUALITY, SEEDS, SNAP_EPISODES,
    SNAP_STEPS, SOURCE, UNIT, arrays, atomic_npz, gpu_check, near_mask, pool,
    read, record, signature, write_json,
)
from scripts.research.clean_oct_workers import egl_identity, resolve_target, target_world


def pre_acceleration(teacher, scene, q, observation, zero):
    """Reproduce the real teacher's active-axis solve and pre-acceleration cap."""
    nav = scene.config['navigation']
    vmax = np.asarray(nav.get('max_velocity_A_deg_B_deg_C_mm_per_s', [4., 2., .6])) * UNIT
    amax = np.asarray(nav.get('max_acceleration_A_deg_B_deg_C_mm_per_s2', [10., 5., 1.5])) * UNIT
    geometry = observation['geometry']
    tip = np.asarray(geometry['tip_world_mm'])
    error = teacher._visual_goal(geometry['target_world_mm'], tip, observation) - tip
    j = teacher._model_jacobian(q)
    active = teacher.control_axes
    u, s, vt = np.linalg.svd(j[:, active] * vmax[active], full_matrices=False)
    correction = np.zeros(3)
    correction[active] = vmax[active] * (vt.T @ (s / (s*s + (s[0]*1e-4)**2) * (u.T @ error)))
    cap = np.minimum(vmax, .6 * np.sqrt(2 * amax * np.abs(correction)))
    wanted = np.clip(np.zeros(3) if zero['in_tolerance']
                     else float(teacher.settings['geometry_gain_per_s']) * correction, -cap, cap)
    wanted[~active] = 0
    expected_zero = np.clip(wanted, -amax*.05, amax*.05)
    np.testing.assert_allclose(expected_zero, zero['command_velocity_task_si'], rtol=0, atol=1e-12)
    return wanted * .05 / UNIT, j, error


def make_teacher(scene):
    from rcm.visual_servo import VisualServoController, visual_servo_settings
    settings = visual_servo_settings(scene)
    settings.update(perception_backend='strict_rgbd', position_feedback='geometry',
                    sample_period_s=.05, rgbd_render_quality=QUALITY.copy(), save_diagnostics=False)
    return VisualServoController(scene, settings=settings)


def static_scene():
    from rcm.scene_v9 import RCMSceneV9
    from scripts.research.a_axis_flip_common import ROOT
    config = read(ROOT / 'config.json')
    config['motor_execution'] = {**config.get('motor_execution', {}), 'enabled': False}
    return RCMSceneV9(config=config)


class LabelAuditWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        gpu_check()
        self.spec = spec
        if file_sha256(DATA / 'data/dataset.npz') != spec['dataset_sha256']:
            raise ValueError('dataset changed')
        self.a = arrays()
        self.scene = static_scene()

    def run(self, task, output):
        a, scene = self.a, self.scene
        rows = np.asarray(task['rows'], np.int64)
        world = target_world(scene, task['target'], a['state'][rows[0]] * UNIT)
        scene.set_target_world(world)
        scene.set_task(a['state'][rows[0]] * UNIT)
        teacher = make_teacher(scene)
        n = len(rows)
        zero, pre, replay, oct_mm = (np.zeros((n, 3), np.float64) for _ in range(4))
        first_order_gradient = np.zeros((n, 3), np.float64)
        valid = np.zeros(n, bool)
        velocity_known = np.zeros(n, bool)
        toward = np.zeros(n, bool)
        reasons = Counter()
        try:
            egl = egl_identity()
            for k, row in enumerate(rows):
                q = a['state'][row].astype(float) * UNIT
                scene.set_task(q)
                teacher.sensor.observation_time_s = k * .05
                obs = teacher.sensor.observe()
                nav = obs.get('oct_navigation_depth')
                delta = None if nav is None else nav.get('delta_xyz_m')
                if delta is not None and not np.isfinite(delta).all():
                    raise FloatingPointError(f'nonfinite rendered OCT at source row {row}')
                z = teacher.geometry_action_from_observation(q, obs,
                    previous_velocity=np.zeros(3), command_q=q)
                if not z['valid'] or delta is None:
                    reasons[z.get('reason', 'OCT_INVALID')] += 1
                    continue
                zero[k] = z['action_deg_deg_mm']
                pre[k], j, error = pre_acceleration(teacher, scene, q, obs, z)
                first_order_gradient[k] = (j * UNIT).T @ error
                oct_mm[k] = np.asarray(delta) * 1000
                valid[k] = True
                toward[k] = float((j @ (zero[k] * UNIT)) @ error) >= -1e-14
                if (row > 0 and a['episode_index'][row-1] == a['episode_index'][row]
                        and a['source_frame_index'][row] == a['source_frame_index'][row-1] + 1
                        and np.isclose(a['timestamp_s'][row] - a['timestamp_s'][row-1], .05,
                                       atol=1e-6, rtol=1e-3)):
                    v = (a['state'][row].astype(float)-a['state'][row-1]) * UNIT / .05
                    r = teacher.geometry_action_from_observation(q, obs,
                        previous_velocity=v, command_q=q)
                    if r['valid']:
                        replay[k] = r['action_deg_deg_mm']
                        velocity_known[k] = True
                if not np.isfinite(np.r_[zero[k], pre[k], replay[k], oct_mm[k]]).all():
                    raise FloatingPointError(f'nonfinite audit calculation at {row}')
                if (k + 1) % 100 == 0:
                    print(f"label audit E{task['episode_index']:04d}: {k+1}/{n}", flush=True)
            path = output / 'labels.npz'
            atomic_npz(path, rows=rows, zero=zero, pre=pre, replay=replay, valid=valid,
                       velocity_known=velocity_known, toward=toward, new_oct_mm=oct_mm,
                       first_order_gradient=first_order_gradient,
                       dataset_sha256=self.spec['dataset_sha256'])
            result = {'episode_index': task['episode_index'], 'frames': n, 'teacher_valid': int(valid.sum()),
                      'path': str(path), 'sha256': file_sha256(path), 'invalid_reasons': dict(reasons), 'egl': egl,
                      'target_provenance': task['target']}
            write_json(output / f"E{task['episode_index']:04d}.json", result)
            return result
        finally:
            teacher.sensor.close()

    def close(self):
        self.scene.close()


def seeded_parallel_actions(model, tensors, seeds=SEEDS):
    """One explicit CUDA noise tensor per seed; encode the frozen observation once."""
    import torch
    image, state, oct_feature, velocity, previous = tensors
    with torch.no_grad():
        tokens, global_condition = model._condition(image, state, oct_feature, velocity, previous)
        n = len(seeds)
        tokens = tokens.expand(n, *tokens.shape[1:])
        global_condition = global_condition.expand(n, *global_condition.shape[1:])
        x = torch.cat([torch.randn(1, model.n_action_steps, 3, device=state.device, dtype=state.dtype,
                      generator=torch.Generator(device=state.device).manual_seed(seed)) for seed in seeds])
        for k in range(4):
            tau = torch.full((n,), (k+.5)/4, device=state.device, dtype=state.dtype)
            x = x + .25 * model._predict_velocity_from_condition(x, tau, tokens, global_condition)
        result = x * model.action_std.reshape(1, 1, -1) + model.action_mean.reshape(1, 1, -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite parallel samples')
        return result


class SnapshotWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        gpu_check()
        from lerobot_training.dit_experience_replay import _load_policy
        import torch
        self.torch, self.spec = torch, spec
        if file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('snapshot checkpoint changed')
        self.model, _, _ = _load_policy(spec['checkpoint'], torch.device('cuda'))
        self.model.eval()
        self.scene = static_scene()
        # No synthetic sensor observations are passed to this controller.
        # It supplies only the model Jacobian; teacher directions are frozen evidence.
        from rcm.visual_servo import VisualServoController
        self.geometry = VisualServoController.__new__(VisualServoController)
        self.geometry.reference = self.scene

    def run(self, task, output):
        torch, scene = self.torch, self.scene
        i = task['episode_index']
        source = Path(task['source_path'])
        if file_sha256(source) != task['source_sha256']:
            raise ValueError('original trace changed')
        ep = read(source)['episodes'][0]
        folder = FROZEN / f'shards/tasks/fixed_E{i:02d}'
        references = read(folder / 'fixed_input.json')['records']
        records = []
        parity_max = 0.
        for k, step in enumerate(SNAP_STEPS):
            path = folder / f'state_{step}_reconstruction.npz'
            if file_sha256(path) != task['snapshots_sha256'][str(step)]:
                raise ValueError('frozen input changed')
            with np.load(path, allow_pickle=False) as f:
                tensors = [torch.from_numpy(f[key].copy()).to('cuda')
                           for key in ('images', 'states', 'oct', 'velocity', 'previous_action')]
            parallel = seeded_parallel_actions(self.model, tensors)
            # Verify every seed on one snapshot for each checkpoint/worker task.
            if k == 0:
                with torch.no_grad():
                    single = torch.cat([self.model.sample(*tensors, steps=4,
                        generator=torch.Generator(device='cuda').manual_seed(seed)) for seed in SEEDS])
                parity_max = float((parallel-single).abs().max())
                torch.testing.assert_close(parallel, single, rtol=2e-4, atol=2e-5)
            samples = parallel[:, 0].cpu().numpy().astype(float)
            q = np.asarray(ep['trajectory']['states'][step]) * UNIT
            scene.set_task(q)
            j = self.geometry._model_jacobian(q)
            camera = scene.data.cam_xmat[scene.camera_id('eye_internal')].reshape(3, 3)
            measured = camera @ (tensors[2][0, -1, :3].cpu().numpy().astype(float) * [1, -1, -1])
            ref = next(r for r in references if r['step'] == step)
            wanted = np.asarray(ref['teacher_action_deg_deg_mm'])
            toward = (samples @ (j * UNIT).T) @ measured
            row = {'episode_index': i, 'step': step, 'mean_action_deg_deg_mm': samples.mean(0).tolist(),
                   'std_action_deg_deg_mm': samples.std(0).tolist(),
                   'teacher_action_deg_deg_mm': wanted.tolist(),
                   'mean_opposes_teacher': ((np.sign(samples.mean(0))*np.sign(wanted) < 0)
                                            & (np.abs(wanted) > EPS)).tolist(),
                   'same_sign_as_teacher_fraction': (samples * wanted > 0).mean(0).tolist(),
                   'first_order_toward_measured_target_fraction': float(np.mean(toward > 0)),
                   'sampling_seeds': list(SEEDS), 'source_snapshot_sha256': file_sha256(path)}
            atomic_npz(output / f'samples_{step}.npz', actions=samples, first_order_dot=toward)
            records.append(row)
            write_json(output / 'partial_snapshot.json', {'records': records})
        result = {'episode_index': i, 'records': records, 'parallel_sequential_max_abs': parity_max,
                  'teacher_reference': 'Frozen recorded-velocity teacher from original diagnostic, unchanged',
                  'image_limitation': 'Original RGB was not retained; frozen reconstructed RGB/OCT used identically in both arms.'}
        write_json(output / f'E{i:02d}.json', result)
        return result

    def close(self):
        self.scene.close()


def snapshot_tasks():
    rows = {r['episode_index']: r for r in read(SOURCE)['episodes']}
    return [{'id': f'E{i:02d}', 'episode_index': i,
             'source_path': rows[i]['result_path'], 'source_sha256': file_sha256(rows[i]['result_path']),
             'snapshots_sha256': {str(s): file_sha256(FROZEN / f'shards/tasks/fixed_E{i:02d}/state_{s}_reconstruction.npz')
                                  for s in SNAP_STEPS}} for i in SNAP_EPISODES]


def snapshots(checkpoints, output, identity):
    groups = {name: {'factory': __name__ + ':SnapshotWorker',
              'spec': {'checkpoint': str(path), 'checkpoint_sha256': file_sha256(path), 'identity': identity},
              'tasks': snapshot_tasks()} for name, path in checkpoints.items()}
    results = pool(groups, output / 'shards')
    summary = {}
    for name in checkpoints:
        rows = [row for key, value in results.items() if key.startswith(name + '_') for row in value['records']]
        if len(rows) != 9:
            raise ValueError('snapshot coverage differs from 9')
        summary[name] = {'snapshots': 9, 'samples': 288,
            'axis_reverse_snapshot_counts': {axis: sum(r['mean_opposes_teacher'][j] for r in rows)
                                            for j, axis in enumerate('ABC')},
            'first_order_toward_fraction': float(np.mean([r['first_order_toward_measured_target_fraction'] for r in rows])),
            'records': rows, 'checkpoint': str(checkpoints[name]),
            'checkpoint_sha256': file_sha256(checkpoints[name])}
    write_json(output / 'snapshot_task2_vs_cleanoct.json', summary)
    record('snapshot_audit', 'completed', output, summary)
    return summary


def label_audit(output, identity):
    a = arrays()
    mask = near_mask(a)
    tasks = []
    for i in np.unique(a['episode_index'][mask]):
        rows = np.flatnonzero(mask & (a['episode_index'] == i))
        tasks.append({'id': f'E{i:04d}', 'episode_index': int(i), 'rows': rows.tolist(),
                      'target': resolve_target(DATA, int(i))})
    groups = {'labels': {'factory': __name__ + ':LabelAuditWorker',
                        'spec': {'dataset_sha256': file_sha256(DATA / 'data/dataset.npz'), 'identity': identity},
                        'tasks': tasks}}
    results = pool(groups, output / 'label_shards')
    blocks = []
    for r in results.values():
        if file_sha256(r['path']) != r['sha256']:
            raise ValueError('label shard changed')
        with np.load(r['path'], allow_pickle=False) as f:
            blocks.append({key: f[key].copy() for key in f.files if key != 'dataset_sha256'})
    v = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
    order = np.argsort(v['rows'])
    v = {key: value[order] for key, value in v.items()}
    if not np.array_equal(v['rows'], np.flatnonzero(mask)):
        raise ValueError('not all near valid source rows audited')
    source_ep = a['episode_index'][v['rows']]
    old = a['action'][v['rows']].astype(float)
    eligible = v['valid'][:, None] & (np.abs(v['pre']) > EPS) & (np.abs(v['zero']) > EPS)
    reverse = eligible & (old * v['zero'] < 0) & (old * v['pre'] < 0) & (np.abs(old) > EPS)
    compatible = (reverse & v['velocity_known'][:, None]
                  & (v['replay'] * v['pre'] < 0) & (v['replay'] * old > 0)
                  & (np.abs(v['replay']) > EPS))
    buckets = {}
    for name, select in [('all', np.ones(len(source_ep), bool)),
                         ('original', source_ep < 839), ('plan2', source_ep >= 839)]:
        stats = {}
        for j, axis in enumerate('ABC'):
            denom = int(eligible[select, j].sum())
            count = int(reverse[select, j].sum())
            stats[axis] = {'eligible_nonzero_geometry_frames': denom,
                          'reverse_frames': count, 'reverse_fraction': count / denom if denom else 0.,
                          'reverse_fraction_all_near_frames': count / int(select.sum()) if select.any() else 0.,
                          'history_replay_inertia_compatible_frames': int(compatible[select, j].sum())}
        buckets[name] = {'all_valid_near_source_frames': int(select.sum()),
                         'rerender_teacher_valid_frames': int(v['valid'][select].sum()), 'axes': stats}
    path = output / 'all_near_teacher_labels.npz'
    atomic_npz(path, **v, reverse=reverse, inertia_compatible=compatible,
               dataset_sha256=file_sha256(DATA / 'data/dataset.npz'))
    report = {'status': 'completed', 'source_dataset': str(DATA),
              'source_dataset_sha256': file_sha256(DATA / 'data/dataset.npz'),
              'covered_frames': int(mask.sum()), 'source_split_episode': 839, 'buckets': buckets,
              'branch_A_hit': any(buckets['all']['axes'][axis]['reverse_fraction'] >= .2 for axis in 'AB'),
              'denominator': 'All near frames whose rerendered zero-velocity and pre-acceleration axis expectations are both nonzero; zero/invalid cases counted separately.',
              'interpretation': 'Direction conflicts against a new-render same-state teacher are associations. Historical finite-difference velocity replay compatibility is separately recorded, not asserted causal proof.',
              'teacher_inputs': 'Actual new-render RGB-D observations at every near source state; no fabricated visibility or fit quality.',
              'arrays_path': str(path), 'arrays_sha256': file_sha256(path)}
    write_json(output / 'ab_label_contamination.json', report)
    record('all_near_label_audit', 'completed', output, report)
    return report


def consistency_audit(output):
    from lerobot_training.train_diffusion_policy import RCMDataset, image_transform_for_dataset
    d = RCMDataset(DATA, transform=image_transform_for_dataset(DATA), n_obs_steps=2, n_action_steps=16,
                   include_velocity=True, include_previous_action=True)
    a = arrays()
    image_paths_before = signature(d.image_paths)
    receipt = apply_dataset_overlays(d, oct_overlay=CLEAN / 'oct_override.npz', c_overlay=CLEAN / 'c_label_override.npz')
    if signature(d.image_paths) != image_paths_before:
        raise ValueError('actual overlay loader unexpectedly changed RGB paths')
    near = near_mask(a)
    adjacent = ((a['episode_index'][1:] == a['episode_index'][:-1])
                & (np.diff(a['source_frame_index']) == 1)
                & np.isclose(np.diff(a['timestamp_s']), .05, atol=1e-6, rtol=1e-3)
                & a['oct_navigation_valid'][1:] & a['oct_navigation_valid'][:-1])
    edges = np.flatnonzero(adjacent & (near[1:] != near[:-1])) + 1
    edges = edges[d.oct_navigation_valid[edges] & d.oct_navigation_valid[edges-1]]
    old_jump = np.linalg.norm(a['oct_navigation'][edges] - a['oct_navigation'][edges-1], axis=1)*1000
    new_jump = np.linalg.norm(d.oct_navigation[edges] - d.oct_navigation[edges-1], axis=1)*1000
    samples = []
    if len(d):
        for sample_index in np.unique(np.linspace(0, len(d)-1, 8, dtype=int)):
            sample = d[int(sample_index)]  # Exercise the real image/observation/history loader.
            source_row = int(d.valid_indices[int(sample_index)])
            expected = np.column_stack((d.oct_navigation[source_row-1:source_row+1],
                                         d.oct_navigation_valid[source_row-1:source_row+1]))
            np.testing.assert_allclose(sample['observation.oct'].numpy(), expected, rtol=0, atol=1e-8)
            images = []
            for row in (source_row-1, source_row):
                path = DATA / d.image_paths[row].replace('\\', '/')
                images.append({'path': str(path), 'sha256': file_sha256(path)})
            samples.append({'source_row': source_row, 'images': images})
    training = read(NEW.parent / 'training_config.json')
    if Path(training['oct_overlay']).resolve() != (CLEAN / 'oct_override.npz').resolve():
        raise ValueError('actual clean training OCT overlay differs from audited path')
    median_old = float(np.median(old_jump)) if len(edges) else None
    median_new = float(np.median(new_jump)) if len(edges) else None
    report = {'status': 'completed', 'actual_training_config': str(NEW.parent / 'training_config.json'),
              'training_config_sha256': file_sha256(NEW.parent / 'training_config.json'),
              'actual_training_keeps_original_RGB': True, 'RGB_paths_unchanged': True,
              'RGB_OCT_render_provenance_mismatch': True,
              'provenance_scope': 'OCT overlay was freshly rendered with the new quality; training uses original stored RGB paths. Different render provenance is established; exact pixel disagreement for all frames is not claimed.',
              'replaced_rows_only_original_valid_under_200um': None,
              'boundary_adjacent_pairs': len(edges), 'old_boundary_jump_median_um': median_old,
              'composed_boundary_jump_median_um': median_new,
              'boundary_jump_increased': bool(len(edges) and median_new > median_old),
              'loader_receipt': receipt, 'real_loader_samples': samples}
    with np.load(CLEAN / 'oct_override.npz', allow_pickle=False) as f:
        report['replaced_rows_only_original_valid_under_200um'] = bool(np.array_equal(f['rows'], np.flatnonzero(near)))
    atomic_npz(output / 'boundary_pairs.npz', right_rows=edges, old_jump_um=old_jump, composed_jump_um=new_jump)
    write_json(output / 'rgb_oct_consistency.json', report)
    record('rgb_oct_audit', 'completed', output, report)
    return report


def select_branch(label, snapshots_report, consistency):
    hit_a = bool(label['branch_A_hit'])
    old = snapshots_report['A_old']['axis_reverse_snapshot_counts']['A']
    new = snapshots_report['A_new']['axis_reverse_snapshot_counts']['A']
    hit_b = bool(old <= 3 and new >= 7 and
                 (consistency['RGB_OCT_render_provenance_mismatch'] or consistency['boundary_jump_increased']))
    return {'branch_A_hit': hit_a, 'branch_B_hit': hit_b,
            'selected_branch': 'B' if hit_b else 'A' if hit_a else None,
            'old_A_reverse_snapshots': old, 'new_A_reverse_snapshots': new,
            'rule': 'A: either A/B >=20% all-bucket eligible direction conflicts. B: old A<=3/9,new A>=7/9 and provenance mismatch or increased boundary median. B takes priority.'}
