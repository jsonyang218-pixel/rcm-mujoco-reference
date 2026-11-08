"""Continuous coarse/fine fusion with a bounded safe fallback state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import RestrictedPerceptionConfig
from .schema import FallbackState, RestrictedObservation
from .simulator import SensorFrame


@dataclass(frozen=True)
class FusionSafetyCommand:
    action_mask_xyz: np.ndarray
    retraction_command_robot_mm: np.ndarray
    retraction_active: bool


class CoarseFineFusion:
    def __init__(
        self,
        config: RestrictedPerceptionConfig,
        *,
        hard_switch: bool = False,
        precision_threshold_um: float | None = None,
    ):
        self.config = config
        self.hard_switch = bool(hard_switch)
        self.precision_threshold_um = float(
            precision_threshold_um
            if precision_threshold_um is not None
            else config.fusion.precision_levels_um[-1]
        )
        if not math.isfinite(self.precision_threshold_um) or self.precision_threshold_um <= 0.0:
            raise ValueError("precision_threshold_um must be finite and positive")
        self.filtered_weight = 0.0
        self.last_fine_vector_robot: np.ndarray | None = None
        self.last_fine_target_robot: np.ndarray | None = None
        self.last_fine_timestamp_s: float | None = None
        self.last_timestamp_s: float | None = None
        self.missing_fine_frames = 0
        self.fallback_started_s: float | None = None
        self.retraction_total_mm = 0.0
        self.state = FallbackState.PRECISION_HOLD

    @staticmethod
    def _as_state(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError("state is required to derive the safe hold/retraction command")
        return result.copy()

    def _is_precise(self, sigma_mm: float | None) -> bool:
        if sigma_mm is None or not math.isfinite(float(sigma_mm)):
            return False
        return (
            self.config.fusion.precision_sigma_multiplier * float(sigma_mm)
            <= self.precision_threshold_um / 1000.0
        )

    def _fine_weight(self, coarse_target_robot_mm: np.ndarray, tip_robot_mm: np.ndarray) -> float:
        distance = float(np.linalg.norm(np.asarray(coarse_target_robot_mm) - tip_robot_mm))
        switch = self.config.fusion.switch_distance_mm
        half_width = self.config.fusion.transition_half_width_mm
        lower = switch - half_width
        upper = switch + half_width
        if distance <= lower:
            return 1.0
        if distance >= upper:
            return 0.0
        return float((upper - distance) / (upper - lower))

    def _lowpass_weight(self, target_weight: float, dt_s: float) -> float:
        tau = self.config.fusion.weight_lowpass_tau_s
        alpha = min(1.0, max(0.0, dt_s / (tau + dt_s)))
        self.filtered_weight += alpha * (target_weight - self.filtered_weight)
        return self.filtered_weight

    def _fallback_elapsed(self, timestamp_s: float) -> float:
        if self.fallback_started_s is None:
            return 0.0
        return max(0.0, float(timestamp_s) - self.fallback_started_s)

    def step(
        self,
        sample: SensorFrame,
        *,
        state: np.ndarray,
        needle_axis_robot: np.ndarray | None = None,
    ) -> RestrictedObservation:
        q = self._as_state(state)
        if needle_axis_robot is None:
            axis = self.config.extrinsic.camera_to_robot_rotation @ np.array(
                [0.0, 0.0, 1.0]
            )
        else:
            axis = np.asarray(needle_axis_robot, dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError("needle_axis_robot must be a finite 3-vector")
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-12:
                raise ValueError("needle_axis_robot must be nonzero")
            axis = axis / axis_norm
        timestamp_s = float(sample.timestamp_s)
        dt_s = (
            self.config.control_period_s
            if self.last_timestamp_s is None
            else max(0.0, timestamp_s - self.last_timestamp_s)
        )
        tip = np.asarray(sample.kinematic_tip_robot_mm, dtype=float).copy()
        rotation = self.config.extrinsic.camera_to_robot_rotation
        fine_vector_robot = None
        if sample.fine_goal_vector_camera_mm is not None:
            measured_vector_robot = rotation @ np.asarray(
                sample.fine_goal_vector_camera_mm, dtype=float
            )
            if sample.fine_observation_tip_robot_mm is None:
                raise ValueError(
                    "fine measurement requires fine_observation_tip_robot_mm for delay alignment"
                )
            tip_motion = tip - np.asarray(
                sample.fine_observation_tip_robot_mm, dtype=float
            )
            fine_vector_robot = measured_vector_robot - tip_motion
        fine_target = None if fine_vector_robot is None else tip + fine_vector_robot
        if sample.fine_valid and fine_target is not None:
            self.last_fine_vector_robot = fine_vector_robot.copy()
            self.last_fine_target_robot = fine_target.copy()
            self.last_fine_timestamp_s = timestamp_s
            self.missing_fine_frames = 0
        else:
            self.missing_fine_frames += 1

        coarse_goal = (
            None
            if not sample.coarse_valid or sample.coarse_target_robot_mm is None
            else np.asarray(sample.coarse_target_robot_mm, dtype=float) - tip
        )

        if sample.fine_valid and sample.coarse_valid and sample.coarse_target_robot_mm is not None:
            requested_weight = self._fine_weight(sample.coarse_target_robot_mm, tip)
        elif sample.fine_valid:
            requested_weight = 1.0
        else:
            requested_weight = 0.0
        weight = self._lowpass_weight(requested_weight, dt_s)
        fine_recoverable = (
            self.last_fine_target_robot is not None
            and self.missing_fine_frames <= self.config.fusion.coast_max_frames
            and self.last_fine_timestamp_s is not None
            and timestamp_s - self.last_fine_timestamp_s
            <= self.config.fusion.coast_max_time_s + 1e-12
        )

        selected_goal: np.ndarray | None
        retraction = np.zeros(3)
        retraction_active = False
        if sample.fine_valid:
            self.state = FallbackState.FINE_TRACKING
            self.fallback_started_s = None
            self.retraction_total_mm = 0.0
            self.filtered_weight = weight
            if self.hard_switch:
                weight = 1.0 if requested_weight >= 0.5 or coarse_goal is None else 0.0
            else:
                weight = self.filtered_weight
            if weight >= 1.0 or coarse_goal is None:
                selected_goal = fine_vector_robot
            elif weight <= 0.0:
                selected_goal = coarse_goal
            else:
                selected_goal = weight * fine_vector_robot + (1.0 - weight) * coarse_goal
        elif fine_recoverable:
            self.state = FallbackState.FINE_COAST
            self.fallback_started_s = None
            selected_goal = self.last_fine_vector_robot
            weight = 1.0
        else:
            if self.fallback_started_s is None:
                self.fallback_started_s = timestamp_s
            self.filtered_weight = 0.0
            weight = 0.0
            coarse_precise = self._is_precise(sample.coarse_sigma_mm)
            if coarse_goal is not None and coarse_precise:
                self.state = FallbackState.COARSE_FALLBACK
                selected_goal = coarse_goal
            else:
                self.state = FallbackState.PRECISION_HOLD
                selected_goal = None
            elapsed = self._fallback_elapsed(timestamp_s)
            can_retract = (
                elapsed >= self.config.fusion.retract_after_loss_s - 1e-12
                and self.retraction_total_mm < self.config.fusion.retract_max_mm
            )
            if can_retract:
                step_mm = min(
                    self.config.fusion.retract_speed_mm_s * dt_s,
                    self.config.fusion.retract_max_mm - self.retraction_total_mm,
                )
                retraction = -step_mm * axis
                self.retraction_total_mm += step_mm
                retraction_active = step_mm > 0.0

        if self.state in (FallbackState.FINE_TRACKING, FallbackState.FINE_COAST):
            precision_sufficient = self._is_precise(sample.fine_sigma_mm)
        elif self.state == FallbackState.COARSE_FALLBACK:
            precision_sufficient = self._is_precise(sample.coarse_sigma_mm)
        else:
            precision_sufficient = False
        action_mask = (
            np.ones(3, dtype=float)
            if selected_goal is not None
            else np.zeros(3, dtype=float)
        )
        observation = RestrictedObservation(
            timestamp_s=timestamp_s,
            state=q,
            kinematic_tip_robot_mm=tip,
            coarse_target_robot_mm=(
                None
                if sample.coarse_target_robot_mm is None
                else np.asarray(sample.coarse_target_robot_mm, dtype=float).copy()
            ),
            coarse_goal_vector_robot_mm=coarse_goal,
            fine_goal_vector_camera_mm=(
                None
                if sample.fine_goal_vector_camera_mm is None
                else np.asarray(sample.fine_goal_vector_camera_mm, dtype=float).copy()
            ),
            fine_goal_vector_robot_mm=(
                None
                if sample.fine_goal_vector_camera_mm is None
                else fine_vector_robot
            ),
            coarse_valid=bool(sample.coarse_valid),
            fine_valid=bool(sample.fine_valid),
            coarse_age_s=sample.coarse_age_s,
            fine_age_s=sample.fine_age_s,
            coarse_sigma_mm=sample.coarse_sigma_mm,
            fine_sigma_mm=sample.fine_sigma_mm,
            coarse_bias_mm=np.asarray(sample.coarse_bias_mm, dtype=float).copy(),
            fine_bias_mm=np.asarray(sample.fine_bias_mm, dtype=float).copy(),
            switch_weight=float(weight),
            fallback_state=self.state,
            precision_sufficient=bool(precision_sufficient),
            selected_goal_vector_robot_mm=(
                None if selected_goal is None else np.asarray(selected_goal, dtype=float).copy()
            ),
            action_mask_xyz=action_mask,
            retraction_command_robot_mm=retraction,
            retraction_active=retraction_active,
            precision_threshold_um=self.precision_threshold_um,
        )
        observation.validate()
        self.last_timestamp_s = timestamp_s
        return observation
