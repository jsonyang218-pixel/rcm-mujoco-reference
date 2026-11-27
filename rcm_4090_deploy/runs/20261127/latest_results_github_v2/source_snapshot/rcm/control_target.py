"""Policy-side ABC intent accumulation; protocol targets remain absolute ints."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TASK_TO_DEG_DEG_MM = np.array([180. / np.pi, 180. / np.pi, 1000.])


@dataclass(frozen=True)
class PreparedTarget:
    owner: object
    revision: int
    intent_deg_deg_mm: np.ndarray
    accepted_delta_deg_deg_mm: np.ndarray
    motion: object
    diagnostics: dict


class ContinuousControlTarget:
    """One accumulator per episode/session, never installed at the receiver.

    preview() is read-only. Only commit() accepts fractional intent. Rejected
    scales cannot advance the motor clock, action history, or the accumulator.
    """

    def __init__(self, reference, max_handle_delta=(32, 32, 32)):
        self.reference = reference
        self.motor = reference.motor_execution
        if self.motor is None:
            raise ValueError('Enable motor execution before creating a target accumulator')
        self.max_handle_delta = np.asarray(max_handle_delta, dtype=float)
        if (self.max_handle_delta.shape != (3,)
                or not np.all(np.isfinite(self.max_handle_delta))
                or np.any(self.max_handle_delta < 1)
                or np.any(self.max_handle_delta != np.round(self.max_handle_delta))):
            raise ValueError('Handle delta limits must be three positive integers')
        self.revision = 0
        self.reset()

    def reset(self):
        self.intent_deg_deg_mm = self.reference.current_task() * TASK_TO_DEG_DEG_MM
        self.epoch = self.motor.epoch
        self.anchor = (self.motor.backend._anchor_robot, self.motor.backend._anchor_handles)
        self.revision += 1

    def synchronize(self):
        """Call before candidate attempts; reset after a stop or new session."""
        anchor = (self.motor.backend._anchor_robot, self.motor.backend._anchor_handles)
        if self.motor.epoch != self.epoch or anchor != self.anchor or self.motor.backend.fault:
            self.reset()
        if self.motor.backend.fault:
            raise RuntimeError('Motor fault requires an explicit session reset')

    def preview(self, action_deg_deg_mm, scale=1.):
        action = np.asarray(action_deg_deg_mm, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError('Action must contain three finite ABC increments')
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError('Action scale must be between zero and one')
        anchor = (self.motor.backend._anchor_robot, self.motor.backend._anchor_handles)
        if self.motor.epoch != self.epoch or anchor != self.anchor or self.motor.backend.fault:
            raise ValueError('Synchronize the accumulator after a stop, fault or session reset')
        actual = self.reference.current_task() * TASK_TO_DEG_DEG_MM
        lead = np.asarray(self.motor.backend.speed_limits) * self.motor.control_period_s
        raw = self.intent_deg_deg_mm + float(scale) * action
        intent = np.clip(raw, actual - lead, actual + lead)
        # Zero scale still follows accepted pending intent. It never adds action.
        motion = self.motor.preview(intent / TASK_TO_DEG_DEG_MM)
        handles = np.asarray(motion.record['handles_ABCE'])
        handle_delta = handles[:3] - np.asarray(self.motor.backend.handles[:3])
        if np.any(np.abs(handle_delta) > self.max_handle_delta):
            raise ValueError('Candidate exceeds the per-packet handle delta limit')
        commanded = np.asarray(motion.record['commanded_task_rad_rad_m']) * TASK_TO_DEG_DEG_MM
        accepted = intent - self.intent_deg_deg_mm
        return PreparedTarget(self, self.revision, intent.copy(), accepted.copy(), motion, {
            'controller_variant': 'continuous_intent_v1',
            'raw_action_deg_deg_mm': action.tolist(), 'scale': float(scale),
            'accepted_delta_deg_deg_mm': accepted.tolist(),
            'intent_deg_deg_mm': intent.tolist(),
            'quantization_residual_deg_deg_mm': (intent - commanded).tolist(),
            'lead_limit_deg_deg_mm': lead.tolist(),
            'lead_limited_axes': (intent != raw).tolist(),
            'handle_delta_ABC': handle_delta.tolist(),
            'max_handle_delta_ABC': self.max_handle_delta.tolist(),
        })

    def commit(self, candidate):
        if candidate.owner is not self or candidate.revision != self.revision:
            raise ValueError('Target candidate is stale or belongs to another accumulator')
        self.reference.apply_prepared_motor(candidate.motion)
        self.intent_deg_deg_mm = candidate.intent_deg_deg_mm.copy()
        self.revision += 1
        self.motor.last_record['control_target'] = candidate.diagnostics
        return candidate.accepted_delta_deg_deg_mm.copy()
