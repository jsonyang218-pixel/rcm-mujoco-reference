"""Persistent EGL overlay and CUDA policy workers for the clean-OCT plan."""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import time

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.gpu_episode_pool import write_json

UNIT = np.array([np.pi / 180, np.pi / 180, .001])
QUALITY = {"offsamples": 0, "numslices": 112, "numstacks": 64}


@lru_cache(maxsize=128)
def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_target(data_dir, episode_index):
    """Follow explicit source indices; never infer a target from a noisy OCT."""
    directory, index = Path(data_dir).resolve(), int(episode_index)
    seen, evidence = set(), []
    while (str(directory), index) not in seen:
        seen.add((str(directory), index))
        metadata_path = directory / "metadata.json"
        metadata = read_json(str(metadata_path))
        episode = metadata['episodes'][index]
        if int(episode['episode_index']) != index:
            raise ValueError('source episode ordering mismatch')
        evidence.append({"path": str(metadata_path), "sha256": file_sha256(metadata_path), "episode_index": index})
        if episode.get('target_robot_mm') is not None:
            return {"kind": "robot_mm", "value": episode['target_robot_mm'], "evidence": evidence}
        audit_path = directory / "data/terminal_audits.json"
        if audit_path.is_file() and 'terminal_audit_index' in episode:
            audits = read_json(str(audit_path))
            audit = audits[int(episode['terminal_audit_index'])]
            if audit.get('terminal_target_world_m') is not None:
                evidence.append({"path": str(audit_path), "sha256": file_sha256(audit_path),
                                 "audit_index": int(episode['terminal_audit_index'])})
                return {"kind": "scanned_world_m", "value": audit['terminal_target_world_m'],
                        "terminal_encoders": audit.get('terminal_target_robot_ABCE'), "evidence": evidence}
        if episode.get('source_dataset') is None or 'source_episode_index' not in episode:
            raise ValueError(f'missing frozen target/source mapping: {directory}/E{index}')
        directory = directory.parent / Path(episode['source_dataset']).name
        index = int(episode['source_episode_index'])
    raise ValueError('source metadata cycle')


def target_world(scene, target, hint):
    value = np.asarray(target['value'], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('invalid frozen target')
    if target['kind'] == 'robot_mm':
        return scene.robot_to_world(value)
    if target['kind'] != 'scanned_world_m' or target.get('terminal_encoders') is None:
        raise ValueError('unsupported target reference or missing terminal encoder audit')
    task = scene.task_from_encoders(target['terminal_encoders'][:3], reference_task=hint)
    scene.set_task(task)
    if np.linalg.norm(scene.site_mm('needle_tip') - value * 1000) * 1000 > .2:
        raise ValueError('scanned-world target/encoder calibration no longer matches frozen audit')
    return value * 1000


def egl_identity():
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    identity = {"vendor": vendor.decode() if vendor else None,
                "renderer": renderer.decode() if renderer else None}
    if not identity['vendor'] or 'NVIDIA' not in identity['vendor']:
        raise RuntimeError(f'NVIDIA EGL context required: {identity}')
    return identity


class OverlayWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        from rcm.scene_v9 import RCMSceneV9
        from rcm.visual_servo import VisualServoController, visual_servo_settings
        self.spec = spec
        if file_sha256(Path(spec['data_dir']) / 'data/dataset.npz') != spec['dataset_sha256']:
            raise ValueError('dataset changed before render worker started')
        if file_sha256(spec['base_c_overlay']) != spec['base_c_sha256']:
            raise ValueError('base C overlay changed before render worker started')
        with np.load(Path(spec['data_dir']) / 'data/dataset.npz', allow_pickle=False) as archive:
            self.arrays = {key: archive[key].copy() for key in (
                'state', 'action', 'executed_action', 'oct_navigation', 'oct_navigation_valid',
                'episode_index', 'source_frame_index', 'timestamp_s')}
        self.c = self.arrays['action'][:, 2].copy()
        with np.load(spec['base_c_overlay'], allow_pickle=False) as archive:
            if str(archive['dataset_sha256']) != spec['dataset_sha256']:
                raise ValueError('base C overlay dataset changed')
            self.c[archive['rows']] = archive['new_dC_mm']
            self.base_rows = set(archive['rows'].tolist())
        self.scene = RCMSceneV9()
        settings = visual_servo_settings(self.scene)
        settings.update(rgbd_render_quality=QUALITY.copy(), sample_period_s=.05,
                        position_feedback='geometry', perception_backend='strict_rgbd', save_diagnostics=False)
        self.teacher = VisualServoController(self.scene, settings=settings)
        self.counter = 0
        self.egl = egl_identity()
        print(f'NVIDIA EGL overlay worker: {self.egl}', flush=True)

    def run(self, task, output):
        from scripts.research.overnight_cfix_data import _counterfactual
        rows = np.asarray(task['rows'], dtype=np.int64)
        world = target_world(self.scene, task['target'], self.arrays['state'][rows[0]] * UNIT)
        self.scene.set_target_world(world)
        self.scene.set_task(self.arrays['state'][rows[0]] * UNIT)
        target_c_mm = self.scene.target_task(validate=False)[2] / UNIT[2]
        observations, valid, errors, audit, corrections = [], [], [], [], []
        started = time.monotonic()
        for row in rows:
            q = self.arrays['state'][row].astype(float) * UNIT
            self.scene.set_task(q)
            self.counter += 1
            self.teacher.sensor.observation_time_s = self.counter * .05
            observation = self.teacher.sensor.observe()
            navigation = observation.get('oct_navigation_depth')
            delta = None if navigation is None else navigation.get('delta_xyz_m')
            is_valid = bool(delta is not None and np.asarray(delta).shape == (3,) and np.isfinite(delta).all())
            value = np.asarray(delta) * 1000 if is_valid else np.zeros(3)
            observations.append(value)
            valid.append(is_valid)
            # Truth remains isolated to this offline audit, never the overlay input.
            camera = self.scene.data.cam_xmat[self.scene.camera_id('eye_internal')].reshape(3, 3)
            truth = (camera.T @ (world - self.scene.site_mm('needle_tip'))) * [1, -1, -1]
            errors.append(float(np.linalg.norm(value - truth) * 1000) if is_valid else None)
            under_mm = target_c_mm - self.arrays['state'][row, 2]
            if (int(row) in self.base_rows and is_valid and 0 < under_mm < .050
                    and self.c[row] > 0):
                replay = _counterfactual(self.teacher, self.scene, q, observation, np.zeros(3), q)
                if replay['valid']:
                    wanted = max(0., min(.030, replay['pre_accel_dC_mm']))
                    audit.append({'row': int(row), 'under_mm': float(under_mm),
                                  'old_overlay_dC_mm': float(self.c[row]), 'wanted_dC_mm': wanted})
                    if wanted > float(self.c[row]) + 1e-9:
                        corrections.append([int(row), wanted])
            if self.counter % 100 == 0:
                print(f"overlay E{task['episode_index']:04d}: frame {int(row)}, total worker frames {self.counter}", flush=True)
        path = output / 'oct.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, rows=rows, old_oct_mm=self.arrays['oct_navigation'][rows],
                                old_valid=self.arrays['oct_navigation_valid'][rows].astype(bool),
                                new_oct_mm=np.asarray(observations, dtype=np.float32),
                                new_valid=np.asarray(valid, dtype=bool), dataset_sha256=self.spec['dataset_sha256'])
        path.with_suffix('.tmp').replace(path)
        write_json(output / 'audit.json', {'rows': rows.tolist(), 'error_um': errors, 'c_audit': audit,
                                         'c_proposals': corrections, 'egl': self.egl,
                                         'target': task['target']})
        finite = [value for value in errors if value is not None]
        return {'episode_index': task['episode_index'], 'frames': len(rows), 'valid_frames': int(sum(valid)),
                'median_error_um': float(np.median(finite)) if finite else None,
                'render_wall_s': time.monotonic() - started,
                'oct_path': str(path), 'audit_path': str(output / 'audit.json')}

    def close(self):
        self.teacher.sensor.close()
        self.scene.close()


class EvaluationWorker:
    def __init__(self, spec):
        from scripts.research.control_g15_diagnostics import VARIANTS, gpu_check
        driver, controller = VARIANTS[spec['variant']]
        os.environ['RCM_DRIVER_VARIANT'] = driver
        gpu_check()
        import torch
        torch.set_num_threads(1)
        from lerobot_training.eval_policy import PolicyEvaluator
        self.spec = spec
        if file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('checkpoint changed before evaluation worker started')
        self.evaluator = PolicyEvaluator(spec['checkpoint'], device='cuda', num_action_samples=1,
                                         flow_integration_steps=4, measure_inference=True,
                                         controller_variant=controller, rgbd_render_quality=QUALITY.copy())
        self.targets = read_json(spec['target_manifest'])['targets']

    def run(self, task, output):
        from scripts.research.control_g15_diagnostics import policy_episode
        # policy_episode creates its own directory and validates I0/P2 contracts.
        folder = output / f"E{task['episode_index']:02d}"
        row = policy_episode(self.evaluator, self.targets, task['episode_index'], self.spec['max_steps'], folder)
        if file_sha256(self.spec['checkpoint']) != self.spec['checkpoint_sha256']:
            raise ValueError('checkpoint changed during evaluation')
        write_json(output / 'episode.json', row)
        return row

    def close(self):
        pass
