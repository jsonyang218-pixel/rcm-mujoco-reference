"""Online adaptation from independent coarse/fine measurements to policy input."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import RestrictedPerceptionConfig
from .fusion import CoarseFineFusion
from .schema import RestrictedObservation
from .simulator import CoarseFinePerceptionSimulator, SensorFrame


class RestrictedPerceptionAdapter:
    def __init__(
        self,
        config: RestrictedPerceptionConfig,
        *,
        target_robot_mm: np.ndarray,
        seed: int,
        episode_id: str,
    ):
        self.config = config
        self.target_robot_mm = np.asarray(target_robot_mm, dtype=float)
        if self.target_robot_mm.shape != (3,) or not np.isfinite(
            self.target_robot_mm
        ).all():
            raise ValueError("target_robot_mm must be a finite simulator-only vector")
        self.simulator = CoarseFinePerceptionSimulator(
            config,
            seed=seed,
            episode_id=episode_id,
        )
        self.fusion = CoarseFineFusion(config)
        self._fine_queue: deque[tuple[float, np.ndarray, np.ndarray | None]] = deque()

    def observe(
        self,
        *,
        state: np.ndarray,
        kinematic_tip_robot_mm: np.ndarray,
        fine_goal_vector_camera_mm: np.ndarray | None,
        fine_valid: bool,
        timestamp_s: float,
    ) -> RestrictedObservation:
        tip = np.asarray(kinematic_tip_robot_mm, dtype=float)
        generated = self.simulator.update(
            self.target_robot_mm,
            tip,
            timestamp_s,
            fine_enabled=False,
        )
        measured = (
            np.asarray(fine_goal_vector_camera_mm, dtype=float).copy()
            if fine_valid and fine_goal_vector_camera_mm is not None
            else None
        )
        self._fine_queue.append((float(timestamp_s), tip.copy(), measured))
        delay = max(0, int(self.config.fine.delay_frames))
        if len(self._fine_queue) <= delay:
            delayed_timestamp, delayed_tip, delayed_vector = (
                float(timestamp_s),
                tip.copy(),
                None,
            )
        else:
            delayed_timestamp, delayed_tip, delayed_vector = self._fine_queue.popleft()
        sample = SensorFrame(
            timestamp_s=float(timestamp_s),
            kinematic_tip_robot_mm=tip.copy(),
            fine_observation_tip_robot_mm=(
                delayed_tip.copy() if delayed_vector is not None else None
            ),
            coarse_target_robot_mm=generated.coarse_target_robot_mm,
            fine_goal_vector_camera_mm=delayed_vector,
            coarse_valid=generated.coarse_valid,
            fine_valid=delayed_vector is not None,
            coarse_age_s=generated.coarse_age_s,
            fine_age_s=(
                float(timestamp_s) - delayed_timestamp
                if delayed_vector is not None
                else None
            ),
            coarse_sigma_mm=generated.coarse_sigma_mm,
            fine_sigma_mm=generated.fine_sigma_mm,
            coarse_bias_mm=generated.coarse_bias_mm,
            fine_bias_mm=generated.fine_bias_mm,
        )
        return self.fusion.step(sample, state=state)
