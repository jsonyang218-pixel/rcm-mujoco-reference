"""Validated configuration for restricted coarse/fine perception."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return tuple(float(item) for item in array)


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ExtrinsicConfig:
    camera_to_robot_rotation: np.ndarray
    translation_robot_mm: np.ndarray
    calibration_status: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExtrinsicConfig":
        rotation = np.asarray(payload["camera_to_robot_rotation"], dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("extrinsic.camera_to_robot_rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("extrinsic.camera_to_robot_rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise ValueError("extrinsic.camera_to_robot_rotation must be a proper rotation")
        return cls(
            camera_to_robot_rotation=rotation.copy(),
            translation_robot_mm=np.asarray(
                _finite_vector(payload["translation_robot_mm"], 3, "extrinsic.translation_robot_mm")
            ),
            calibration_status=str(payload.get("calibration_status", "unspecified")),
        )


@dataclass(frozen=True)
class CoarseSensorConfig:
    bias_limit_mm: np.ndarray
    drift_ar1_coefficient: float
    drift_innovation_sigma_mm: np.ndarray
    noise_floor_mm: float
    noise_slope_mm_per_mm: float
    delay_frames: int
    max_distance_mm: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CoarseSensorConfig":
        coefficient = float(payload["drift_ar1_coefficient"])
        if not 0.0 <= coefficient < 1.0:
            raise ValueError("coarse.drift_ar1_coefficient must be in [0, 1)")
        return cls(
            bias_limit_mm=np.asarray(_finite_vector(payload["bias_limit_mm"], 3, "coarse.bias_limit_mm")),
            drift_ar1_coefficient=coefficient,
            drift_innovation_sigma_mm=np.asarray(
                _finite_vector(
                    payload["drift_innovation_sigma_mm"],
                    3,
                    "coarse.drift_innovation_sigma_mm",
                )
            ),
            noise_floor_mm=_positive(payload["noise_floor_mm"], "coarse.noise_floor_mm"),
            noise_slope_mm_per_mm=_positive(
                payload["noise_slope_mm_per_mm"], "coarse.noise_slope_mm_per_mm"
            ),
            delay_frames=int(payload["delay_frames"]),
            max_distance_mm=_positive(payload["max_distance_mm"], "coarse.max_distance_mm"),
        )


@dataclass(frozen=True)
class FineSensorConfig:
    bias_limit_mm: float
    noise_floor_mm: float
    noise_slope_mm_per_mm: float
    delay_frames: int
    dropout_probability: float
    fov_half_angle_deg: float
    min_range_mm: float
    max_range_mm: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FineSensorConfig":
        dropout = float(payload["dropout_probability"])
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("fine.dropout_probability must be in [0, 1]")
        fov = _positive(payload["fov_half_angle_deg"], "fine.fov_half_angle_deg")
        if fov >= 90.0:
            raise ValueError("fine.fov_half_angle_deg must be below 90 degrees")
        min_range = _positive(payload["min_range_mm"], "fine.min_range_mm")
        max_range = _positive(payload["max_range_mm"], "fine.max_range_mm")
        if max_range <= min_range:
            raise ValueError("fine.max_range_mm must exceed fine.min_range_mm")
        return cls(
            bias_limit_mm=float(payload["bias_limit_mm"]),
            noise_floor_mm=_positive(payload["noise_floor_mm"], "fine.noise_floor_mm"),
            noise_slope_mm_per_mm=_positive(
                payload["noise_slope_mm_per_mm"], "fine.noise_slope_mm_per_mm"
            ),
            delay_frames=int(payload["delay_frames"]),
            dropout_probability=dropout,
            fov_half_angle_deg=fov,
            min_range_mm=min_range,
            max_range_mm=max_range,
        )


@dataclass(frozen=True)
class FusionConfig:
    switch_distance_mm: float
    transition_half_width_mm: float
    weight_lowpass_tau_s: float
    coast_max_frames: int
    coast_max_time_s: float
    retract_after_loss_s: float
    retract_speed_mm_s: float
    retract_max_mm: float
    precision_levels_um: tuple[float, ...]
    precision_sigma_multiplier: float
    required_closed_loop_success: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FusionConfig":
        levels = tuple(sorted(float(value) for value in payload["precision_levels_um"]))
        if not levels or any(not math.isfinite(value) or value <= 0.0 for value in levels):
            raise ValueError("fusion.precision_levels_um must contain positive finite values")
        success = float(payload["required_closed_loop_success"])
        if not 0.0 <= success <= 1.0:
            raise ValueError("fusion.required_closed_loop_success must be in [0, 1]")
        return cls(
            switch_distance_mm=_positive(
                payload["switch_distance_mm"], "fusion.switch_distance_mm"
            ),
            transition_half_width_mm=_positive(
                payload["transition_half_width_mm"], "fusion.transition_half_width_mm"
            ),
            weight_lowpass_tau_s=_positive(
                payload["weight_lowpass_tau_s"], "fusion.weight_lowpass_tau_s"
            ),
            coast_max_frames=int(payload["coast_max_frames"]),
            coast_max_time_s=_positive(payload["coast_max_time_s"], "fusion.coast_max_time_s"),
            retract_after_loss_s=_positive(
                payload["retract_after_loss_s"], "fusion.retract_after_loss_s"
            ),
            retract_speed_mm_s=_positive(payload["retract_speed_mm_s"], "fusion.retract_speed_mm_s"),
            retract_max_mm=_positive(payload["retract_max_mm"], "fusion.retract_max_mm"),
            precision_levels_um=levels,
            precision_sigma_multiplier=_positive(
                payload["precision_sigma_multiplier"], "fusion.precision_sigma_multiplier"
            ),
            required_closed_loop_success=success,
        )


@dataclass(frozen=True)
class RestrictedPerceptionConfig:
    schema_version: int
    name: str
    control_period_s: float
    extrinsic: ExtrinsicConfig
    coarse: CoarseSensorConfig
    fine: FineSensorConfig
    fusion: FusionConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RestrictedPerceptionConfig":
        if int(payload.get("schema_version")) != 1:
            raise ValueError("Unsupported restricted perception schema_version")
        return cls(
            schema_version=1,
            name=str(payload["name"]),
            control_period_s=_positive(payload["control_period_s"], "control_period_s"),
            extrinsic=ExtrinsicConfig.from_dict(payload["extrinsic"]),
            coarse=CoarseSensorConfig.from_dict(payload["coarse"]),
            fine=FineSensorConfig.from_dict(payload["fine"]),
            fusion=FusionConfig.from_dict(payload["fusion"]),
        )


def load_config(path: str | Path) -> RestrictedPerceptionConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RestrictedPerceptionConfig.from_dict(payload)
