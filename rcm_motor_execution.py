"""Protocol motor execution shared by the legacy renderer and current v9.

Only explicit control ticks advance time. Kinematic planning/reset operations
remain separate; failed motor commands never fall back to direct target poses.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mujoco_model'))
from motor_backend import MotorBackend


class MotorExecution:
    def __init__(self, reference, settings=None):
        self.reference = reference
        self.settings = dict(settings or {})
        self.control_period_s = float(self.settings.get('control_period_s',
            reference.config.get('navigation', {}).get('sample_period_s', .05)))
        self.integration_step_s = float(self.settings.get('integration_step_s', .002))
        for value in (self.control_period_s, self.integration_step_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Motor periods must be finite and positive')
        initial = tuple(int(round(float(v))) for v in reference.encoders()) + (0,)
        self.backend = MotorBackend(initial_robot=initial, needle_profile='legacy',
            tau_s=self.settings.get('tau_s', .04),
            timeout_s=self.settings.get('timeout_s', .5),
            speed_limits=self.settings.get('speed_limits_A_deg_B_deg_C_mm_per_s', (10., 10., 2.)))
        self.time_s = 0.
        self.records = []
        self.last_record = None
        self.actual_velocity = np.zeros(3)

    def advance(self, desired_task, duration_s=None):
        desired = self.reference.validate_task(desired_task).copy()
        duration = self.control_period_s if duration_s is None else float(duration_s)
        if not math.isfinite(duration) or not 0 < duration <= self.backend.timeout_s:
            raise ValueError('Motor control duration must be positive and no longer than the watchdog timeout')
        desired_robot = tuple(int(round(float(v))) for v in self.reference.encoders(desired)) + (0,)
        command = self.backend.command_for_robot_target(desired_robot)
        before_robot = self.backend.robot
        before_task = self.reference.current_task().copy()
        before_time = self.time_s
        # Validate before accepting the command. The uploaded driver performs
        # the protocol quantization and B-linkage branch validation.
        receipt = self.backend.consume(command, before_time)
        count = max(1, math.ceil(duration / self.integration_step_s))
        dt = duration / count
        try:
            commanded = self.reference.task_from_encoders(receipt['target'][:3], reference_task=before_task)
            self.reference.validate_task(commanded)
            for i in range(count):
                self.backend.step(dt, before_time + (i + 1) * dt)
            actual = self.reference.task_from_encoders(self.backend.robot[:3], reference_task=before_task)
            actual = self.reference.validate_task(actual).copy()
        except Exception:
            self.backend.hold()
            raise
        self.time_s = before_time + duration
        self.actual_velocity = (actual - before_task) / duration
        record = {
            'step_index': len(self.records), 'start_time_s': before_time,
            'end_time_s': self.time_s, 'duration_s': duration,
            'integration_steps': count, 'tau_s': self.backend.tau_s,
            'requested_task_rad_rad_m': desired.tolist(),
            'commanded_task_rad_rad_m': commanded.tolist(),
            'requested_robot_ABCE': list(desired_robot),
            'handles_ABCE': [getattr(command, 'handleEncoder'+a) for a in 'ABCE'],
            'quantized_target_robot_ABCE': list(receipt['target']),
            'actual_before_robot_ABCE': list(before_robot),
            'actual_after_robot_ABCE': list(self.backend.robot),
            'actual_before_task_rad_rad_m': before_task.tolist(),
            'actual_after_task_rad_rad_m': actual.tolist(),
            'actual_velocity_rad_rad_m_per_s': self.actual_velocity.tolist(),
            'status': self.backend.status, 'fault': self.backend.fault,
        }
        self.last_record = record
        self.records.append(record)
        return actual

    def settled(self):
        # Encoder-stationary AND at the quantized command, not merely a zero
        # desired velocity while the motor is still catching up.
        return (self.backend.robot == self.backend.target
                and bool(np.all(np.abs(self.actual_velocity) < 1e-12)))

    def hold(self):
        self.backend.hold()

    def reset_clock(self, time_s=0.):
        self.backend.reset_clock(time_s)
        self.time_s = float(time_s)
        self.records = []
        self.last_record = None
        self.actual_velocity[:] = 0

    def summary(self):
        return {'backend': 'MotorBackend', 'tau_s': self.backend.tau_s,
                'control_period_s': self.control_period_s,
                'integration_step_s': self.integration_step_s,
                'speed_limits_A_deg_B_deg_C_mm_per_s': list(self.backend.speed_limits),
                'timeout_s': self.backend.timeout_s,
                'simulation_time_s': self.time_s, 'control_steps': len(self.records),
                'robot_ABCE': list(self.backend.robot),
                'target_robot_ABCE': list(self.backend.target),
                'encoder_settled': self.settled(),
                'status': self.backend.status, 'fault': self.backend.fault,
                'physical_dynamics_identified': False,
                'direct_pose_fallback_used': False}
