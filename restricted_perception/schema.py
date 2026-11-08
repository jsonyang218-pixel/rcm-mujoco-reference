"""Strict schema and policy whitelist for restricted perception."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np


class FallbackState(str, Enum):
    FINE_TRACKING = "FINE_TRACKING"
    FINE_COAST = "FINE_COAST"
    COARSE_FALLBACK = "COARSE_FALLBACK"
    PRECISION_HOLD = "PRECISION_HOLD"


POLICY_FEATURE_KEYS = (
    "state",
    "kinematic_tip_robot_mm",
    "coarse_goal_vector_robot_mm",
    "fine_goal_vector_robot_mm",
    "coarse_valid",
    "fine_valid",
    "coarse_age_s",
    "fine_age_s",
    "coarse_sigma_mm",
    "fine_sigma_mm",
    "switch_weight",
    "fallback_state",
    "precision_sufficient",
)
POLICY_VECTOR_KEYS = (
    "coarse_goal_vector_robot_mm",
    "fine_goal_vector_robot_mm",
    "coarse_valid",
    "fine_valid",
    "coarse_age_s",
    "fine_age_s",
    "coarse_sigma_mm",
    "fine_sigma_mm",
    "switch_weight",
    "fallback_state",
    "precision_sufficient",
)


def _optional_vector(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be None or a finite 3-vector")
    return array.copy()


def _optional_nonnegative(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be None or finite and nonnegative")
    return result


@dataclass(frozen=True)
class RestrictedObservation:
    timestamp_s: float
    state: np.ndarray
    kinematic_tip_robot_mm: np.ndarray
    coarse_target_robot_mm: np.ndarray | None
    coarse_goal_vector_robot_mm: np.ndarray | None
    fine_goal_vector_camera_mm: np.ndarray | None
    fine_goal_vector_robot_mm: np.ndarray | None
    coarse_valid: bool
    fine_valid: bool
    coarse_age_s: float | None
    fine_age_s: float | None
    coarse_sigma_mm: float | None
    fine_sigma_mm: float | None
    coarse_bias_mm: np.ndarray | None
    fine_bias_mm: np.ndarray | None
    switch_weight: float
    fallback_state: FallbackState
    precision_sufficient: bool
    selected_goal_vector_robot_mm: np.ndarray | None
    action_mask_xyz: np.ndarray
    retraction_command_robot_mm: np.ndarray
    retraction_active: bool
    precision_threshold_um: float

    def validate(self) -> None:
        if not math.isfinite(float(self.timestamp_s)) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        for name in ("state", "kinematic_tip_robot_mm"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3-vector")
        for name in (
            "coarse_target_robot_mm",
            "coarse_goal_vector_robot_mm",
            "fine_goal_vector_camera_mm",
            "fine_goal_vector_robot_mm",
            "coarse_bias_mm",
            "fine_bias_mm",
            "selected_goal_vector_robot_mm",
        ):
            _optional_vector(getattr(self, name), name)
        for name in ("coarse_age_s", "fine_age_s", "coarse_sigma_mm", "fine_sigma_mm"):
            _optional_nonnegative(getattr(self, name), name)
        if not isinstance(self.fallback_state, FallbackState):
            raise ValueError("fallback_state must be a FallbackState")
        weight = float(self.switch_weight)
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("switch_weight must be in [0, 1]")
        mask = np.asarray(self.action_mask_xyz, dtype=float)
        if mask.shape != (3,) or not np.isfinite(mask).all() or not np.isin(mask, [0.0, 1.0]).all():
            raise ValueError("action_mask_xyz must contain three binary values")
        retraction = np.asarray(self.retraction_command_robot_mm, dtype=float)
        if retraction.shape != (3,) or not np.isfinite(retraction).all():
            raise ValueError("retraction_command_robot_mm must be a finite 3-vector")
        if not math.isfinite(float(self.precision_threshold_um)) or self.precision_threshold_um <= 0.0:
            raise ValueError("precision_threshold_um must be positive")

    def policy_features(self) -> dict[str, Any]:
        """Return the complete policy allowlist.

        Audit-only episode biases, simulator truth, and teacher state are not
        part of this mapping.
        """
        self.validate()
        return {
            "state": np.asarray(self.state, dtype=float).copy(),
            "kinematic_tip_robot_mm": np.asarray(self.kinematic_tip_robot_mm, dtype=float).copy(),
            "coarse_goal_vector_robot_mm": (
                np.zeros(3) if self.coarse_goal_vector_robot_mm is None
                else np.asarray(self.coarse_goal_vector_robot_mm, dtype=float).copy()
            ),
            "fine_goal_vector_robot_mm": (
                np.zeros(3) if self.fine_goal_vector_robot_mm is None
                else np.asarray(self.fine_goal_vector_robot_mm, dtype=float).copy()
            ),
            "coarse_valid": bool(self.coarse_valid),
            "fine_valid": bool(self.fine_valid),
            "coarse_age_s": float(self.coarse_age_s) if self.coarse_age_s is not None else -1.0,
            "fine_age_s": float(self.fine_age_s) if self.fine_age_s is not None else -1.0,
            "coarse_sigma_mm": (
                float(self.coarse_sigma_mm) if self.coarse_sigma_mm is not None else -1.0
            ),
            "fine_sigma_mm": float(self.fine_sigma_mm) if self.fine_sigma_mm is not None else -1.0,
            "switch_weight": float(self.switch_weight),
            "fallback_state": self.fallback_state.value,
            "precision_sufficient": bool(self.precision_sufficient),
        }

    def policy_vector(self) -> np.ndarray:
        features = self.policy_features()
        fallback = np.eye(len(FallbackState), dtype=np.float32)[
            list(FallbackState).index(self.fallback_state)
        ]
        values = [
            *features["coarse_goal_vector_robot_mm"].tolist(),
            *features["fine_goal_vector_robot_mm"].tolist(),
            float(features["coarse_valid"]),
            float(features["fine_valid"]),
            float(features["coarse_age_s"]),
            float(features["fine_age_s"]),
            float(features["coarse_sigma_mm"]),
            float(features["fine_sigma_mm"]),
            float(features["switch_weight"]),
            *fallback.tolist(),
            float(features["precision_sufficient"]),
        ]
        result = np.asarray(values, dtype=np.float32)
        from .sidecar import RESTRICTED_FEATURE_NAMES

        if result.shape != (len(RESTRICTED_FEATURE_NAMES),):
            raise ValueError("restricted policy vector has an inconsistent shape")
        return result

    def audit_record(self) -> dict[str, Any]:
        payload = self.policy_features()
        payload.update(
            {
                "timestamp_s": float(self.timestamp_s),
                "coarse_target_robot_mm": (
                    None if self.coarse_target_robot_mm is None
                    else np.asarray(self.coarse_target_robot_mm, dtype=float).tolist()
                ),
                "fine_goal_vector_camera_mm": (
                    None if self.fine_goal_vector_camera_mm is None
                    else np.asarray(self.fine_goal_vector_camera_mm, dtype=float).tolist()
                ),
                "coarse_bias_mm": (
                    None if self.coarse_bias_mm is None
                    else np.asarray(self.coarse_bias_mm, dtype=float).tolist()
                ),
                "fine_bias_mm": (
                    None if self.fine_bias_mm is None
                    else np.asarray(self.fine_bias_mm, dtype=float).tolist()
                ),
                "selected_goal_vector_robot_mm": (
                    None if self.selected_goal_vector_robot_mm is None
                    else np.asarray(self.selected_goal_vector_robot_mm, dtype=float).tolist()
                ),
                "action_mask_xyz": np.asarray(self.action_mask_xyz, dtype=float).tolist(),
                "retraction_command_robot_mm": np.asarray(
                    self.retraction_command_robot_mm, dtype=float
                ).tolist(),
                "retraction_active": bool(self.retraction_active),
                "precision_threshold_um": float(self.precision_threshold_um),
            }
        )
        return payload
