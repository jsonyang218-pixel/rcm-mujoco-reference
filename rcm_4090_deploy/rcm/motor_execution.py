"""Protocol motor execution shared by the legacy renderer and current v9.

Only explicit control ticks advance time. Kinematic planning/reset operations
remain separate; failed motor commands never fall back to direct target poses.
"""
from __future__ import annotations

import math
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mujoco_model'))
from motor_backend import MotorBackend
from driver_loader import driver_identity


@dataclass(frozen=True)
class PreparedMotion:
    owner: object
    revision: int
    before_task: np.ndarray
    before_backend: dict
    backend: object
    actual: np.ndarray
    record: dict


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
        self.driver_identity = driver_identity()
        self.time_s = 0.
        self.records = []
        self.last_record = None
        self.actual_velocity = np.zeros(3)
        self.revision = 0
        self.epoch = 0

    def advance(self, desired_task, duration_s=None):
        return self.commit(self.preview(desired_task, duration_s))

    def preview(self, desired_task, duration_s=None):
        """Validate and simulate a candidate without changing live state."""
        desired = self.reference.validate_task(desired_task).copy()
        desired_robot = tuple(int(round(float(v))) for v in self.reference.encoders(desired)) + (0,)
        return self.preview_robot(desired_robot, duration_s, requested_task=desired)

    def advance_robot(self, desired_robot, duration_s=None, requested_task=None):
        """Execute one interval at an exact integer A/B/C/E target.

        A decoded terminal task pose does not necessarily re-encode to the same
        integer counts near a B-linkage branch boundary. Zero-error terminal
        locking therefore supplies the reproduced encoder vector directly while
        still issuing a normal MotorBackend protocol command.
        """
        return self.commit(self.preview_robot(desired_robot, duration_s, requested_task))

    def preview_robot(self, desired_robot, duration_s=None, requested_task=None):
        duration = self.control_period_s if duration_s is None else float(duration_s)
        if not math.isfinite(duration) or not 0 < duration <= self.backend.timeout_s:
            raise ValueError('Motor control duration must be positive and no longer than the watchdog timeout')
        values = np.asarray(desired_robot)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("desired robot target must be four finite integers")
        integer_robot = tuple(int(round(float(value))) for value in values)
        if not np.array_equal(values, np.asarray(integer_robot)):
            raise ValueError("desired robot target must contain integers")
        if integer_robot[3] != 0:
            raise ValueError("robot target extension channel E must be zero")
        requested_task = (
            self.reference.current_task().copy()
            if requested_task is None
            else self.reference.validate_task(requested_task).copy()
        )
        command = self.backend.command_for_robot_target(integer_robot)
        before_robot = self.backend.robot
        before_task = self.reference.current_task().copy()
        before_time = self.time_s
        # Exclude the append-only history from copying: preview must stay O(1)
        # in the number of prior control ticks. Driver modules are shared.
        trial = copy.copy(self.backend)
        trial.history = []
        receipt = trial.consume(command, before_time)
        count = max(1, math.ceil(duration / self.integration_step_s))
        dt = duration / count
        commanded = self.reference.task_from_encoders(receipt['target'][:3], reference_task=before_task)
        self.reference.validate_task(commanded)
        for i in range(count):
            trial.step(dt, before_time + (i + 1) * dt)
            actual = self.reference.task_from_encoders(trial.robot[:3], reference_task=before_task)
            actual = self.reference.validate_task(actual).copy()
        if trial.fault:
            raise ValueError('Motor preview fault: ' + trial.fault)
        velocity = (actual - before_task) / duration
        record = {
            'driver': dict(self.driver_identity),
            'step_index': len(self.records), 'start_time_s': before_time,
            'end_time_s': before_time + duration, 'duration_s': duration,
            'integration_steps': count, 'tau_s': self.backend.tau_s,
            'requested_task_rad_rad_m': requested_task.tolist(),
            'commanded_task_rad_rad_m': commanded.tolist(),
            'requested_robot_ABCE': list(integer_robot),
            'handles_ABCE': [getattr(command, 'handleEncoder'+a) for a in 'ABCE'],
            'quantized_target_robot_ABCE': list(receipt['target']),
            'actual_before_robot_ABCE': list(before_robot),
            'actual_after_robot_ABCE': list(trial.robot),
            'actual_before_task_rad_rad_m': before_task.tolist(),
            'actual_after_task_rad_rad_m': actual.tolist(),
            'actual_velocity_rad_rad_m_per_s': velocity.tolist(),
            'tip_rcm_radius_before_mm': float(np.linalg.norm(self.backend.tip_mm())),
            'tip_rcm_radius_after_mm': float(np.linalg.norm(trial.tip_mm())),
            'tip_displacement_um': float(np.linalg.norm(
                np.asarray(trial.tip_mm()) - np.asarray(self.backend.tip_mm())) * 1000.),
            'status': trial.status, 'fault': trial.fault,
        }
        state = {key: value for key, value in self.backend.__dict__.items()
                 if key not in ('history', 'kin')}
        return PreparedMotion(self, self.revision, before_task, state, trial, actual, record)

    def commit(self, prepared):
        """Publish one fully validated interval; reject stale previews."""
        state = {key: value for key, value in self.backend.__dict__.items()
                 if key not in ('history', 'kin')}
        if (prepared.owner is not self or prepared.revision != self.revision
                or state != prepared.before_backend
                or not np.array_equal(self.reference.current_task(), prepared.before_task)):
            raise ValueError('Motor preview is stale or belongs to another session')
        history = self.backend.history
        history.extend(prepared.backend.history)
        self.backend.__dict__.update(prepared.backend.__dict__)
        self.backend.history = history
        record = prepared.record
        self.time_s = record['end_time_s']
        self.actual_velocity = np.asarray(record['actual_velocity_rad_rad_m_per_s'])
        self.last_record = record
        self.records.append(record)
        self.revision += 1
        return prepared.actual.copy()

    def settled(self):
        # Encoder-stationary AND at the quantized command, not merely a zero
        # desired velocity while the motor is still catching up.
        return (self.backend.robot == self.backend.target
                and bool(np.all(np.abs(self.actual_velocity) < 1e-12)))

    def hold(self):
        self.backend.hold()
        self.epoch += 1
        self.revision += 1

    def reset_clock(self, time_s=0.):
        self.backend.reset_clock(time_s)
        self.time_s = float(time_s)
        self.records = []
        self.last_record = None
        self.actual_velocity[:] = 0
        self.epoch += 1
        self.revision += 1

    def summary(self):
        return {'backend': 'MotorBackend', 'driver': dict(self.driver_identity),
                'tau_s': self.backend.tau_s,
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
