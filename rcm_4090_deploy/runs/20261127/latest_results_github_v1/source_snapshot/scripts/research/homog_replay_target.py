"""Bounded fractional teacher intent, committed only after an accepted control tick."""
from __future__ import annotations

import numpy as np

UNIT = np.array([np.pi / 180., np.pi / 180., .001])


class TeacherIntent:
    """Keep sub-handle corrections on the sender; keep protocol packets unchanged."""

    def __init__(self, reference, caps_deg_deg_mm):
        self.reference = reference
        self.caps = np.asarray(caps_deg_deg_mm, dtype=float)
        if (self.caps.shape != (3,) or not np.isfinite(self.caps).all()
                or np.any(self.caps <= 0)):
            raise ValueError('Teacher intent caps must be three finite positive increments')
        self.intent_task_si = None
        self.pending = None

    def preview(self, q, increment_deg_deg_mm, *, hold=False):
        q = np.asarray(q, dtype=float)
        increment = np.asarray(increment_deg_deg_mm, dtype=float)
        if (q.shape != (3,) or increment.shape != (3,)
                or not np.isfinite(q).all() or not np.isfinite(increment).all()):
            raise ValueError('Finite ABC state and teacher increment are required')
        motor = self.reference.motor_execution
        if motor is None or motor.backend.fault:
            raise ValueError('Teacher intent requires a healthy explicit motor session')
        base = q if self.intent_task_si is None else self.intent_task_si
        # The lead is bounded by both one physical control interval and the
        # original teacher's per-axis increment caps. Clip before committing
        # to prevent a blocked axis from accumulating unbounded intent.
        lead = np.minimum(self.caps,
            np.asarray(motor.backend.speed_limits) * motor.control_period_s) * UNIT
        raw = q.copy() if hold else base + increment * UNIT
        target = self.reference.validate_task(np.clip(raw, q - lead, q + lead)).copy()
        self.pending = {'q': q.copy(), 'target': target.copy(), 'hold': bool(hold)}
        return (target - q) / UNIT, {
            'controller': 'bounded_teacher_intent_v1',
            'teacher_increment_deg_deg_mm': increment.tolist(),
            'intent_task_si': target.tolist(),
            'lead_limited_axes': (raw != target).tolist(),
            'hold_residual_reset': bool(hold)}

    def commit(self, accepted_action_deg_deg_mm):
        if self.pending is None:
            raise ValueError('No pending teacher target to commit')
        accepted = np.asarray(accepted_action_deg_deg_mm, dtype=float)
        if accepted.shape != (3,) or not np.isfinite(accepted).all():
            raise ValueError('Accepted teacher action must be three finite increments')
        # Safety may accept a scaled action. Preserve only the accepted target,
        # not a rejected candidate or the raw teacher request.
        self.intent_task_si = self.reference.validate_task(
            self.pending['q'] + accepted * UNIT).copy()
        self.pending = None
