"""Controlled physical occluder for renderer-consistent simulation experiments.

This is privileged SCENE INTERVENTION code. It may read the target's scene
position solely to place an opaque sphere on the camera ray. The controller
must never use that position or the audit placement log as a missing-target
measurement. No RGB, depth or segmentation arrays are edited.

Configuration under ``reference_scene.occlusion_fixture`` is opt-in. Events
use per-waypoint observation-frame indices, shared across tracking/recovery
phases. An event with ``relative_to_phase: true`` measures its interval from
the first frame of each waypoint/phase pair, retaining that origin through
recovery. The caller owns the monotonic frame clock; it must not reset on
entering recovery. ``update`` returns activity metadata only, never positions.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


GEOM_NAME = 'experiment_occluder'
MATERIAL_NAME = 'experiment_occluder_material'
HIDDEN_POSITION_M = np.array([10., 10., 10.])


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if int(value) < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')
    return int(value)


def fixture_settings(config):
    raw = config.get('reference_scene', {}).get('occlusion_fixture', {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError('reference_scene.occlusion_fixture must be an object')
    enabled = raw.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('occlusion_fixture.enabled must be boolean')
    radius = float(raw.get('radius_mm', .3))
    fraction = float(raw.get('ray_fraction', .75))
    if not math.isfinite(radius) or radius <= 0.:
        raise ValueError('occlusion_fixture.radius_mm must be positive')
    if not math.isfinite(fraction) or not .6 <= fraction <= .8:
        raise ValueError('occlusion_fixture.ray_fraction must lie in [0.6, 0.8]')
    events = raw.get('events', [])
    if not isinstance(events, list):
        raise ValueError('occlusion_fixture.events must be an array')
    parsed = []
    event_ids = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError('Each occlusion fixture event must be an object')
        event_id = str(event.get('id', f'occlusion_{index:03d}'))
        if not event_id or event_id in event_ids:
            raise ValueError('Occlusion fixture event IDs must be unique and nonempty')
        event_ids.add(event_id)
        waypoint = str(event.get('waypoint_id', '*'))
        if not waypoint:
            raise ValueError('Occlusion event waypoint_id must be nonempty')
        start = _positive_integer(event.get('start_frame', 0), 'start_frame', allow_zero=True)
        duration = _positive_integer(event.get('duration_frames', 1), 'duration_frames')
        relative = event.get('relative_to_phase', False)
        if not isinstance(relative, bool):
            raise ValueError('Occlusion event relative_to_phase must be boolean')
        phases = event.get('phases')
        if phases is None and 'phase' in event:
            phases = [event['phase']]
        if phases is not None:
            if not isinstance(phases, list) or not phases or not all(isinstance(p, str) and p for p in phases):
                raise ValueError('Occlusion event phases must be a nonempty string array')
            phases = tuple(phases)
        parsed.append({'id': event_id, 'waypoint_id': waypoint, 'start_frame': start,
                       'duration_frames': duration, 'phases': phases,
                       'relative_to_phase': relative})
    return {'enabled': enabled, 'radius_mm': radius, 'ray_fraction': fraction,
            'camera_name': str(raw.get('camera_name', 'eye_internal')), 'events': parsed}


def add_occluder_to_xml(asset, world, config):
    """Optional build_xml hook; the default scene receives no new geometry."""
    settings = fixture_settings(config)
    if not settings['enabled']:
        return
    ET.SubElement(asset, 'material', name=MATERIAL_NAME, rgba='.035 .035 .035 1',
                  specular='0', shininess='0')
    ET.SubElement(world, 'geom', name=GEOM_NAME, type='sphere', pos='10 10 10',
                  size=str(settings['radius_mm']*.001), material=MATERIAL_NAME,
                  group='3', contype='0', conaffinity='0')


class PhysicalOcclusionFixture:
    """Schedule one actual opaque sphere; observational state has no coordinates.

    ``audit`` includes privileged intervention placement records and belongs
    only in experimental evaluation artifacts, never policy/controller input.
    """
    def __init__(self, reference):
        self.reference = reference
        self.settings = fixture_settings(reference.config)
        self.enabled = self.settings['enabled']
        self.geom_id = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME)
        if self.enabled and self.geom_id < 0:
            raise RuntimeError('Enabled occlusion fixture geometry is missing from the scene')
        self.camera_id = reference.camera_id(self.settings['camera_name']) if self.enabled else None
        self.target_id = int(reference.model.geom('lesion_marker').id) if self.enabled else None
        self._last_frame = {}
        self._phase_origin = {}
        self._active_id = None
        self._audit = {'kind': 'controlled_opaque_sphere', 'enabled': self.enabled,
                       'scope': 'Privileged physical scene intervention, not natural needle occlusion or perception fallback.',
                       'radius_mm': self.settings['radius_mm'], 'ray_fraction': self.settings['ray_fraction'],
                       'events': copy.deepcopy(self.settings['events']),
                       'update_count': 0, 'transitions': [], 'current': {'active': False, 'event_id': None}}
        if self.enabled:
            self._move(HIDDEN_POSITION_M)

    @property
    def audit(self):
        return copy.deepcopy(self._audit)

    def _move(self, position_m):
        if self.enabled:
            self.reference.model.geom_pos[self.geom_id] = position_m
            mujoco.mj_forward(self.reference.model, self.reference.data)

    def hide(self, reason='manual_hide'):
        prior = self._active_id
        self._move(HIDDEN_POSITION_M)
        self._active_id = None
        state = {'active': False, 'event_id': None, 'reason': str(reason)}
        self._audit['current'] = state
        if prior is not None:
            self._audit['transitions'].append({'transition': 'hide', 'event_id': prior, 'reason': str(reason)})
        return copy.deepcopy(state)

    def reset(self):
        self.hide('episode_reset')
        self._last_frame.clear()
        self._phase_origin.clear()

    def update(self, waypoint_id, phase, frame_index):
        frame = _positive_integer(frame_index, 'frame_index', allow_zero=True)
        waypoint, phase = str(waypoint_id), str(phase)
        prior_frame = self._last_frame.get(waypoint)
        if prior_frame is not None and frame < prior_frame:
            raise ValueError('Occlusion fixture frame_index must remain monotonic across recovery phases')
        self._last_frame[waypoint] = frame
        phase_origin = self._phase_origin.setdefault((waypoint, phase), frame)
        phase_frame = frame-phase_origin
        self._audit['update_count'] += 1
        active = []
        if self.enabled:
            for event in self.settings['events']:
                if event['waypoint_id'] not in ('*', waypoint):
                    continue
                if event['phases'] is not None and phase not in event['phases']:
                    continue
                event_frame = phase_frame if event['relative_to_phase'] else frame
                if event['start_frame'] <= event_frame < event['start_frame']+event['duration_frames']:
                    active.append(event)
        if len(active) > 1:
            self.hide('overlapping_events_rejected')
            raise ValueError('Overlapping events request one occluder simultaneously')
        event = active[0] if active else None
        event_id = event['id'] if event else None
        state = {'active': event is not None, 'event_id': event_id, 'waypoint_id': waypoint,
                 'phase': phase, 'frame_index': frame,
                 'phase_origin_frame': phase_origin, 'phase_frame_index': phase_frame}
        placement = None
        if event is not None:
            # These scene coordinates ONLY place the controlled intervention.
            # They are never returned to or consumed by the feedback law.
            camera = self.reference.data.cam_xpos[self.camera_id].copy()
            target = self.reference.data.geom_xpos[self.target_id].copy()
            position = camera+self.settings['ray_fraction']*(target-camera)
            self._move(position)
            placement = {'occluder_world_mm': (position*1000.).tolist(),
                         'target_world_mm_privileged': (target*1000.).tolist()}
        elif self._active_id is not None:
            self._move(HIDDEN_POSITION_M)
        if event_id != self._active_id:
            transition = {'transition': 'show' if event else 'hide', **state}
            if placement is not None:
                transition.update(placement)
            self._audit['transitions'].append(transition)
        self._active_id = event_id
        self._audit['current'] = state
        return copy.deepcopy(state)

    def before_observation(self, waypoint_id, phase, iteration):
        return self.update(waypoint_id, phase, iteration)
