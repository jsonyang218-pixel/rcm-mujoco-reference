"""T3 settlement through ordinary protocol commands before explicit zero hold."""
from __future__ import annotations

import numpy as np

from scripts.research.homog_replay_target import TeacherIntent, UNIT


class T3HoldError(ValueError):
    def __init__(self, details):
        self.details = details
        super().__init__(f"{details['code']}: frame={details['frame']}, "
                         f"measured_error_um={details['measured_error_um']:.6f}, "
                         f"encoder_settled={details['motor']['encoder_settled']}")


class T3HoldIntent(TeacherIntent):
    """Latch a candidate below 5 um, settle, then verify zero-action stability.

    Measurement and motor feedback alone control the transition. No direct-pose
    reset, simulator hold(), changed gains or extra control intervals are used.
    The latched candidate is committed only after the evaluator accepts a tick.
    """

    def __init__(self, reference, caps_deg_deg_mm, hold_start):
        super().__init__(reference, caps_deg_deg_mm)
        if not isinstance(hold_start, int) or hold_start < 1:
            raise ValueError('A positive integer hold start is required')
        self.hold_start = hold_start
        self.candidate_task_si = None
        self.pending_phase = None

    def zero_retains_target(self, q):
        from lerobot_training.eval_policy import state_abc_mm_to_task, task_to_state_abc_mm
        motor = self.reference.motor_execution
        # Match the evaluator's degree/mm round trip and use its real protocol
        # preview. This is read-only and does not advance the live motor clock.
        preview = motor.preview(state_abc_mm_to_task(task_to_state_abc_mm(q)))
        return (tuple(preview.record['quantized_target_robot_ABCE']) == motor.backend.robot
                and np.array_equal(preview.actual, q))

    def fail(self, code, frame, measured_error_um, q):
        motor = self.reference.motor_execution
        raise T3HoldError({'code': code, 'frame': int(frame),
            'measured_error_um': float(measured_error_um), 'state_si': q.tolist(),
            'candidate_task_si': (None if self.candidate_task_si is None
                                  else self.candidate_task_si.tolist()),
            'motor': motor.summary(), 'last_motor_record': motor.last_record})

    def preview_observed(self, q, increment_deg_deg_mm, *, frame, measured_error_um):
        q = np.asarray(q, dtype=float)
        increment = np.asarray(increment_deg_deg_mm, dtype=float)
        if (q.shape != (3,) or increment.shape != (3,)
                or not np.isfinite(q).all() or not np.isfinite(increment).all()
                or not np.isfinite(measured_error_um)):
            raise ValueError('Finite T3 observation and teacher increment are required')
        motor = self.reference.motor_execution
        if motor is None or motor.backend.fault:
            raise ValueError('T3 settlement requires a healthy explicit motor session')
        below = measured_error_um < 5.
        settled = motor.settled()
        candidate = self.candidate_task_si
        ready = candidate is not None and settled and below and self.zero_retains_target(q)
        hold = frame >= self.hold_start
        if hold:
            if not below:
                self.fail('HOLD_ENTRY_ERROR_EXCEEDED' if frame == self.hold_start
                          else 'HOLD_ERROR_EXCEEDED', frame, measured_error_um, q)
            if not ready:
                self.fail('HOLD_ENTRY_NOT_SETTLED' if frame == self.hold_start
                          else 'HOLD_MOTION_DETECTED', frame, measured_error_um, q)
            phase, desired = 'hold', q
        elif ready:
            phase, desired = 'ready', q
        elif candidate is not None and not settled:
            phase, desired = 'settling', candidate
        elif below and candidate is None:
            phase, desired = 'settling', q
        else:
            # Reject a settled candidate outside the measured gate (or one
            # whose zero command would move it), and resume from real feedback.
            phase = 'tracking'
            base = q if candidate is not None or self.intent_task_si is None else self.intent_task_si
            desired = base + increment * UNIT
        base = q if self.intent_task_si is None else self.intent_task_si
        command, diagnostics = super().preview(q, (desired - base) / UNIT, hold=hold)
        self.pending_phase = phase
        diagnostics.update(controller='bounded_teacher_intent_t3_settlement_v1',
            teacher_increment_deg_deg_mm=increment.tolist(), phase=phase,
            measured_error_um=float(measured_error_um), encoder_settled=bool(settled),
            zero_action_stable=bool(ready), hold_start=self.hold_start,
            candidate_task_si=None if candidate is None else candidate.tolist())
        return command, diagnostics

    def commit(self, accepted_action_deg_deg_mm):
        if self.pending_phase is None:
            raise ValueError('No pending T3 observation to commit')
        phase = self.pending_phase
        super().commit(accepted_action_deg_deg_mm)
        self.candidate_task_si = None if phase == 'tracking' else self.intent_task_si.copy()
        self.pending_phase = None
