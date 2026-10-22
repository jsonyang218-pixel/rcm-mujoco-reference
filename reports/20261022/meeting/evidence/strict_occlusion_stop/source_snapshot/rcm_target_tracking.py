"""Bounded memory of a registered static target, never of a moving needle.

Only current rendered measurements enter this module. A stored target may
bridge a short visibility/fit gap under the explicit fixed-camera, static-
target assumption. Hidden target motion cannot be detected until it becomes
visible again. Raw image features and geometry dictionaries are never edited.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math

import numpy as np


class TargetMemory:
    """Register stable observations and bound later target-memory use.

    ``resolve`` always requires a fresh, usable needle measurement. Its return
    value is control/audit metadata, not a replacement sensor observation.
    The first registered point remains the identity/movement anchor for the
    entire goal. ``reacquire`` retains that anchor while requiring new stable
    observations; ``reset`` starts a different registration/goal.

    Fit-quality values are empirical budgets, not calibrated confidence bounds.
    """

    max_measurement_quality_mm = .01
    memory_quality_growth_mm_per_s = .001

    def __init__(self, sample_period_s=.05, max_age_s=1., min_observations=3,
                 max_shift_mm=.04, stability_mm=.01, max_travel_mm=.5):
        for name, value in [('sample_period_s', sample_period_s), ('max_age_s', max_age_s),
                            ('max_shift_mm', max_shift_mm), ('stability_mm', stability_mm),
                            ('max_travel_mm', max_travel_mm)]:
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if isinstance(min_observations, bool) or int(min_observations) != min_observations or min_observations < 1:
            raise ValueError('min_observations must be a positive integer')
        self.sample_period_s = float(sample_period_s)
        self.max_age_s = float(max_age_s)
        self.min_observations = int(min_observations)
        self.max_shift_mm = float(max_shift_mm)
        self.stability_mm = float(stability_mm)
        self.max_travel_mm = float(max_travel_mm)
        self.reset()

    def reset(self, goal_id=None):
        """Forget a previous goal, including its position and identity anchor."""
        self.goal_id = goal_id
        self._samples = deque(maxlen=self.min_observations)
        self._registration = None
        self._cached_point = None
        self._cached_quality = None
        self._cached_disagreement_mm = 0.
        self._last_target_fit_quality_mm = None
        self._ready = False
        self._ever_observed = False
        self._last_seen = None
        self._last_time = None
        self._last_tip = None
        self._travel_mm = 0.
        self._invalidated_reason = None
        self.reacquisition_shift_mm = None
        self.last_result = {}

    def reacquire(self):
        """Require fresh stable samples without forgetting the registered goal."""
        self._samples.clear()
        self._ready = False
        self._last_seen = None
        self._travel_mm = 0.
        self.reacquisition_shift_mm = None
        # An observed movement invalidation still requires a new goal/replan
        # and reset; reacquisition must not silently forgive target motion.

    @property
    def cache_ready(self):
        return self._ready and self._invalidated_reason is None

    def _reject(self, code, metadata):
        self.last_result = deepcopy({**metadata, 'target_source': 'unavailable',
                                     'used_target_world_mm': None, 'target_world_mm': None,
                                     'rejection_reason': code,
                                     'reacquisition_shift_mm': self.reacquisition_shift_mm,
                                     'cache_ready': self.cache_ready})
        raise RuntimeError(code)

    @staticmethod
    def _point(value):
        try:
            result = np.asarray(value, dtype=float)
        except (ValueError, TypeError):
            return None
        return result.copy() if result.shape == (3,) and np.isfinite(result).all() else None

    @staticmethod
    def _quality(value):
        if not isinstance(value, dict):
            return None
        try:
            quality = float(value['fit_quality_mm'])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(quality) or quality < 0.:
            return None
        return deepcopy(value)

    def resolve(self, observation, time_s):
        """Return a current or bounded registered target and current needle.

        Canonical source values are ``observed`` and ``registered_memory``.
        On error ``last_result`` retains audit fields such as movement shift.
        A repeated timestamp cannot manufacture additional registration samples.
        ``time_s`` is the caller's monotonic sensor/simulation clock; wall-clock
        latency is a separate quantity and must be recorded by the caller.
        """
        metadata = {'goal_id': self.goal_id, 'target_observed': False,
                    'cached_target_age_s': None, 'target_memory_travel_mm': self._travel_mm,
                    'target_last_observed_time_s': self._last_seen,
                    'registration_observations': len(self._samples)}
        try:
            time_s = float(time_s)
        except (ValueError, TypeError):
            self._reject('TARGET_TRACK_TIME_INVALID', metadata)
        if not math.isfinite(time_s) or time_s < 0.:
            self._reject('TARGET_TRACK_TIME_INVALID', metadata)
        if self._last_time is not None and time_s < self._last_time-1e-12:
            self._reject('TARGET_TRACK_TIME_REVERSED', metadata)
        advanced = self._last_time is None or time_s > self._last_time+1e-12
        geometry = observation.get('geometry', {})
        tip = self._point(geometry.get('tip_world_mm'))
        tip_quality = self._quality(geometry.get('tip_quality'))
        if tip is None or tip_quality is None or tip_quality['fit_quality_mm'] > self.max_measurement_quality_mm:
            if not self._ready:
                self._samples.clear()
            metadata['tip_measurement_error'] = (
                geometry.get('tip_error', 'TIP_NOT_OBSERVABLE') if tip is None else 'LOW_QUALITY_TIP')
            self._reject('TIP_NOT_OBSERVABLE', metadata)
        metadata['tip_world_mm'] = tip.tolist()
        if not advanced and self._last_tip is not None and np.linalg.norm(tip-self._last_tip) > 1e-10:
            self._reject('TARGET_TRACK_DUPLICATE_TIME_WITH_MOTION', metadata)
        if advanced and self._last_tip is not None:
            self._travel_mm += float(np.linalg.norm(tip-self._last_tip))
        self._last_time, self._last_tip = time_s, tip.copy()
        metadata['target_memory_travel_mm'] = self._travel_mm
        if self._invalidated_reason is not None:
            self._reject(self._invalidated_reason, metadata)

        target = self._point(geometry.get('target_world_mm'))
        quality = self._quality(geometry.get('target_quality'))
        pixel = observation.get('target_pixel')
        try:
            visible = pixel is not None and np.asarray(pixel, dtype=float).shape == (2,) and np.isfinite(pixel).all()
        except (TypeError, ValueError):
            visible = False
        good_target = bool(visible and target is not None and quality is not None
                           and quality['fit_quality_mm'] <= self.max_measurement_quality_mm)
        metadata['target_visible'] = bool(visible)
        if good_target:
            self.reacquisition_shift_mm = (None if self._registration is None else
                                          float(np.linalg.norm(target-self._registration)))
            metadata['target_observed'] = True
            if (self.reacquisition_shift_mm is not None
                    and self.reacquisition_shift_mm > self.max_shift_mm+1e-12):
                self._invalidated_reason = 'TARGET_MOVED_REPLAN_REQUIRED'
                self._reject(self._invalidated_reason, metadata)
            if advanced:
                self._samples.append((target.copy(), deepcopy(quality)))
            self._ever_observed = True
            self._last_seen = time_s
            self._travel_mm = 0.
            if len(self._samples) == self.min_observations and not self._ready:
                points = np.asarray([sample[0] for sample in self._samples])
                diameter = float(np.max(np.linalg.norm(points[:, None]-points[None, :], axis=2)))
                if diameter <= self.stability_mm:
                    center = points.mean(axis=0)
                    if self._registration is None:
                        self._registration = center.copy()
                    self._cached_point = center.copy()
                    self._cached_quality = deepcopy(self._samples[-1][1])
                    spread = float(np.max(np.linalg.norm(points-center, axis=1)))
                    self._cached_quality['fit_quality_mm'] = (
                        max(float(sample[1]['fit_quality_mm']) for sample in self._samples)+spread)
                    self._cached_quality['registration_spread_mm'] = spread
                    self._cached_quality['registration_sample_count'] = len(self._samples)
                    self._ready = True
            if self._ready:
                # A later accepted measurement can disagree with the fixed
                # registered point without crossing the movement threshold.
                # That disagreement still consumes the memory error budget;
                # visibility alone must not refresh a misleading old point.
                self._cached_disagreement_mm = float(np.linalg.norm(target-self._cached_point))
                self._last_target_fit_quality_mm = float(quality['fit_quality_mm'])
            result = {**metadata, 'target_source': 'observed', 'target_observed': True,
                      'cached_target_age_s': 0., 'target_memory_travel_mm': 0.,
                      'target_last_observed_time_s': self._last_seen,
                      'used_target_world_mm': target.tolist(), 'target_quality': quality}
        else:
            # Before registration, a missing/poor observation breaks the stable
            # acquisition sequence. It cannot be filled from scene coordinates.
            if not self._ready:
                self._samples.clear()
                code = 'TARGET_MEMORY_NOT_READY' if self._ever_observed else 'TARGET_MEMORY_UNAVAILABLE'
                self._reject(code, metadata)
            age = time_s-self._last_seen
            metadata['cached_target_age_s'] = age
            metadata['target_last_observed_time_s'] = self._last_seen
            if age > self.max_age_s+1e-12:
                self._reject('TARGET_MEMORY_EXPIRED', metadata)
            if self._travel_mm > self.max_travel_mm+1e-12:
                self._reject('TARGET_MEMORY_TRAVEL_EXCEEDED', metadata)
            cached_quality = deepcopy(self._cached_quality)
            cached_quality.update(source='registered_static_target_memory',
                                  quality_kind='empirical_fit_budget_not_statistical_confidence',
                                  last_observation_disagreement_mm=self._cached_disagreement_mm)
            cached_quality['fit_quality_mm'] = max(
                cached_quality['fit_quality_mm'],
                self._last_target_fit_quality_mm+self._cached_disagreement_mm,
            )+self.memory_quality_growth_mm_per_s*age
            if cached_quality['fit_quality_mm'] > self.max_measurement_quality_mm:
                self._reject('TARGET_MEMORY_LOW_QUALITY', metadata)
            result = {**metadata, 'target_source': 'registered_memory',
                      'used_target_world_mm': self._cached_point.tolist(),
                      'target_quality': cached_quality,
                      'target_measurement_error': geometry.get('target_error',
                          'TARGET_NOT_VISIBLE' if not visible else 'LOW_QUALITY_TARGET')}
        result.update(cache_ready=self.cache_ready,
                      registration_observations=len(self._samples),
                      reacquisition_shift_mm=self.reacquisition_shift_mm,
                      target_world_mm=list(result['used_target_world_mm']),
                      estimated_position_error_um=float(
                          np.linalg.norm(np.asarray(result['used_target_world_mm'])-tip)*1000.))
        self.last_result = deepcopy(result)
        return deepcopy(result)
