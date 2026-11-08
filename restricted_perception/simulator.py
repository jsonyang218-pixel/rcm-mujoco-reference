"""Independent coarse RGB-D and fine OCT measurement simulators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .config import RestrictedPerceptionConfig


@dataclass(frozen=True)
class SensorFrame:
    timestamp_s: float
    kinematic_tip_robot_mm: np.ndarray
    fine_observation_tip_robot_mm: np.ndarray | None
    coarse_target_robot_mm: np.ndarray | None
    fine_goal_vector_camera_mm: np.ndarray | None
    coarse_valid: bool
    fine_valid: bool
    coarse_age_s: float | None
    fine_age_s: float | None
    coarse_sigma_mm: float | None
    fine_sigma_mm: float | None
    coarse_bias_mm: np.ndarray
    fine_bias_mm: np.ndarray


@dataclass
class _DelayedSample:
    timestamp_s: float
    tip_robot_mm: np.ndarray
    target_robot_mm: np.ndarray | None
    vector_camera_mm: np.ndarray | None
    valid: bool
    sigma_mm: float | None


class CoarseFinePerceptionSimulator:
    """Generate measurements while never exposing simulator target truth.

    The public ``update`` method accepts target truth and the kinematic tip as
    simulator inputs, but returns only sensor estimates and audit biases. The
    fusion layer has no truth parameter.
    """

    def __init__(self, config: RestrictedPerceptionConfig, seed: int, episode_id: str = ""):
        self.config = config
        material = f"{seed}:{episode_id}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        episode_seed = int.from_bytes(digest[:8], "little", signed=False)
        self.rng = np.random.default_rng(episode_seed)
        self.coarse_bias = self.rng.uniform(
            -config.coarse.bias_limit_mm, config.coarse.bias_limit_mm
        )
        self.fine_bias_mm = self.rng.uniform(
            -config.fine.bias_limit_mm, config.fine.bias_limit_mm, size=3
        )
        self.coarse_drift = np.zeros(3)
        self._coarse_queue: list[_DelayedSample] = []
        self._fine_queue: list[_DelayedSample] = []
        self._last_timestamp = -math.inf

    def _validate_inputs(
        self, target_robot_mm: np.ndarray, tip_robot_mm: np.ndarray, timestamp_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target_robot_mm, dtype=float)
        tip = np.asarray(tip_robot_mm, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_robot_mm must be a finite simulator-only 3-vector")
        if tip.shape != (3,) or not np.isfinite(tip).all():
            raise ValueError("tip_robot_mm must be a finite 3-vector")
        if not math.isfinite(float(timestamp_s)) or timestamp_s < self._last_timestamp:
            raise ValueError("timestamp_s must be finite and nondecreasing")
        return target.copy(), tip.copy()

    def _coarse_sigma(self, distance_mm: float) -> float:
        return (
            self.config.coarse.noise_floor_mm
            + self.config.coarse.noise_slope_mm_per_mm * max(0.0, distance_mm)
        )

    def _fine_sigma(self, distance_mm: float) -> float:
        return (
            self.config.fine.noise_floor_mm
            + self.config.fine.noise_slope_mm_per_mm * max(0.0, distance_mm)
        )

    def _coarse_sample(
        self, target_robot_mm: np.ndarray, tip_robot_mm: np.ndarray, timestamp_s: float
    ) -> _DelayedSample:
        truth_goal = target_robot_mm - tip_robot_mm
        distance = float(np.linalg.norm(truth_goal))
        sigma = self._coarse_sigma(distance)
        self.coarse_drift = (
            self.config.coarse.drift_ar1_coefficient * self.coarse_drift
            + self.rng.normal(0.0, self.config.coarse.drift_innovation_sigma_mm)
        )
        noise = self.rng.normal(0.0, sigma, size=3)
        target = target_robot_mm + self.coarse_bias + self.coarse_drift + noise
        valid = distance <= self.config.coarse.max_distance_mm
        return _DelayedSample(
            timestamp_s=timestamp_s,
            tip_robot_mm=tip_robot_mm.copy(),
            target_robot_mm=target if valid else None,
            vector_camera_mm=None,
            valid=valid,
            sigma_mm=sigma,
        )

    def _fine_sample(
        self,
        target_robot_mm: np.ndarray,
        tip_robot_mm: np.ndarray,
        timestamp_s: float,
    ) -> _DelayedSample:
        truth_goal = target_robot_mm - tip_robot_mm
        distance = float(np.linalg.norm(truth_goal))
        sigma = self._fine_sigma(distance)
        rotation = self.config.extrinsic.camera_to_robot_rotation
        vector_camera = rotation.T @ truth_goal
        in_range = self.config.fine.min_range_mm <= distance <= self.config.fine.max_range_mm
        if distance > 0.0:
            cosine = abs(float(vector_camera[2])) / distance
            angle_deg = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
        else:
            angle_deg = 0.0
        in_fov = angle_deg <= self.config.fine.fov_half_angle_deg
        acquired = (
            in_range
            and in_fov
            and self.rng.random() >= self.config.fine.dropout_probability
        )
        if acquired:
            bias_noise_robot = self.fine_bias_mm + self.rng.normal(0.0, sigma, size=3)
            measured = vector_camera + rotation.T @ bias_noise_robot
        else:
            measured = None
        return _DelayedSample(
            timestamp_s=timestamp_s,
            tip_robot_mm=tip_robot_mm.copy(),
            target_robot_mm=None,
            vector_camera_mm=measured,
            valid=acquired,
            sigma_mm=sigma,
        )

    def _delayed(
        self, queue: list[_DelayedSample], sample: _DelayedSample, delay_frames: int
    ) -> _DelayedSample:
        queue.append(sample)
        if delay_frames <= 0:
            return sample
        if len(queue) <= delay_frames:
            return _DelayedSample(
                timestamp_s=sample.timestamp_s,
                tip_robot_mm=sample.tip_robot_mm.copy(),
                target_robot_mm=None,
                vector_camera_mm=None,
                valid=False,
                sigma_mm=sample.sigma_mm,
            )
        return queue.pop(0)

    def update(
        self,
        target_robot_mm: np.ndarray,
        tip_robot_mm: np.ndarray,
        timestamp_s: float,
        *,
        coarse_enabled: bool = True,
        fine_enabled: bool = True,
    ) -> SensorFrame:
        target, tip = self._validate_inputs(target_robot_mm, tip_robot_mm, timestamp_s)
        self._last_timestamp = float(timestamp_s)
        coarse_now = self._coarse_sample(target, tip, timestamp_s)
        fine_now = self._fine_sample(target, tip, timestamp_s)
        coarse = self._delayed(self._coarse_queue, coarse_now, self.config.coarse.delay_frames)
        fine = self._delayed(self._fine_queue, fine_now, self.config.fine.delay_frames)
        if not coarse_enabled:
            coarse = _DelayedSample(
                timestamp_s, tip.copy(), None, None, False, coarse.sigma_mm
            )
        if not fine_enabled:
            fine = _DelayedSample(
                timestamp_s, tip.copy(), None, None, False, fine.sigma_mm
            )
        return SensorFrame(
            timestamp_s=float(timestamp_s),
            kinematic_tip_robot_mm=tip,
            fine_observation_tip_robot_mm=(
                fine.tip_robot_mm.copy() if fine.valid else None
            ),
            coarse_target_robot_mm=(
                None if coarse.target_robot_mm is None else coarse.target_robot_mm.copy()
            ),
            fine_goal_vector_camera_mm=(
                None if fine.vector_camera_mm is None else fine.vector_camera_mm.copy()
            ),
            coarse_valid=bool(coarse.valid),
            fine_valid=bool(fine.valid),
            coarse_age_s=float(timestamp_s - coarse.timestamp_s) if coarse.valid else None,
            fine_age_s=float(timestamp_s - fine.timestamp_s) if fine.valid else None,
            coarse_sigma_mm=float(coarse.sigma_mm) if coarse.sigma_mm is not None else None,
            fine_sigma_mm=float(fine.sigma_mm) if fine.sigma_mm is not None else None,
            coarse_bias_mm=self.coarse_bias.copy(),
            fine_bias_mm=self.fine_bias_mm.copy(),
        )
