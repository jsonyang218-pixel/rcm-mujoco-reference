"""Precision-floor and switching-step acceptance calculations."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .config import RestrictedPerceptionConfig


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("percentile requires a nonempty finite sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q must be in [0, 100]")
    return float(np.percentile(array, q))


def single_frame_step_p95(goal_vectors_mm: Sequence[np.ndarray]) -> float:
    vectors = np.asarray(goal_vectors_mm, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) < 2:
        raise ValueError("goal_vectors_mm must contain at least two 3-vectors")
    if not np.isfinite(vectors).all():
        raise ValueError("goal_vectors_mm must be finite")
    return percentile(np.linalg.norm(np.diff(vectors, axis=0), axis=1), 95.0)


def switch_step_reduction(hard_steps_mm: Sequence[float], blended_steps_mm: Sequence[float]) -> dict:
    hard = np.asarray(hard_steps_mm, dtype=float)
    blended = np.asarray(blended_steps_mm, dtype=float)
    if hard.shape != blended.shape or hard.ndim != 1 or hard.size == 0:
        raise ValueError("hard and blended step sequences must be nonempty and aligned")
    if not np.isfinite(hard).all() or not np.isfinite(blended).all():
        raise ValueError("step sequences must be finite")
    hard_p95 = percentile(hard, 95.0)
    blended_p95 = percentile(blended, 95.0)
    reduction = 1.0 - blended_p95 / hard_p95 if hard_p95 > 0.0 else 0.0
    return {
        "hard_p95_step_mm": hard_p95,
        "blended_p95_step_mm": blended_p95,
        "relative_reduction": float(reduction),
        "minimum_relative_reduction": 0.8,
        "pass": bool(reduction >= 0.8),
    }


def precision_floor(
    config: RestrictedPerceptionConfig,
    p95_perception_error_um: float,
    deterministic_closed_loop_success: float,
) -> dict:
    error = float(p95_perception_error_um)
    success = float(deterministic_closed_loop_success)
    if not math.isfinite(error) or error < 0.0:
        raise ValueError("p95_perception_error_um must be finite and nonnegative")
    if not math.isfinite(success) or not 0.0 <= success <= 1.0:
        raise ValueError("deterministic_closed_loop_success must be in [0, 1]")
    rows = []
    for threshold in sorted(config.fusion.precision_levels_um):
        accuracy_pass = config.fusion.precision_sigma_multiplier * error <= threshold
        closed_loop_pass = success >= config.fusion.required_closed_loop_success
        rows.append(
            {
                "threshold_um": float(threshold),
                "accuracy_pass": bool(accuracy_pass),
                "closed_loop_pass": bool(closed_loop_pass),
                "pass": bool(accuracy_pass and closed_loop_pass),
            }
        )
    passing = [row for row in rows if row["pass"]]
    selected = min(passing, key=lambda row: row["threshold_um"])["threshold_um"] if passing else None
    return {
        "p95_perception_error_um": error,
        "sigma_multiplier": config.fusion.precision_sigma_multiplier,
        "deterministic_closed_loop_success": success,
        "levels": rows,
        "selected_threshold_um": selected,
        "pass": selected is not None,
    }
