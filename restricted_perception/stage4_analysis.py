"""Distribution and failure-mode analysis for the frozen Stage 4 test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import RestrictedPerceptionConfig
from .simulator import CoarseFinePerceptionSimulator


SAFETY_RANGE_TOKEN = "不在已确认安全范围"
RESOURCE_ERROR_TOKENS = (
    "MemoryError",
    "FatalError",
    "Could not allocate",
)


def summarize_values(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be nonempty and finite")
    percentiles = np.percentile(array, [0.0, 5.0, 50.0, 95.0, 100.0])
    return {
        "count": int(array.size),
        "min": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "max": float(percentiles[4]),
        "mean": float(array.mean()),
    }


def target_geometry(
    points_robot_mm: np.ndarray,
    *,
    center_world_mm: np.ndarray,
    rotation: np.ndarray,
    radii_mm: np.ndarray,
    robot_to_world,
) -> dict:
    points = np.asarray(points_robot_mm, dtype=float)
    world = np.asarray([robot_to_world(point) for point in points])
    local = (world - center_world_mm) @ rotation
    normalized = local / radii_mm
    normalized_norm = np.linalg.norm(normalized, axis=1)
    polar_angle_deg = np.degrees(
        np.arccos(
            np.clip(
                normalized[:, 2] / normalized_norm,
                -1.0,
                1.0,
            )
        )
    )
    surface_error_um = (
        np.abs(normalized_norm - 1.0) * float(np.min(radii_mm)) * 1000.0
    )
    return {
        "robot_xyz_mm": {
            axis: summarize_values(points[:, index])
            for index, axis in enumerate(("x", "y", "z"))
        },
        "normalized_eye_local_z": summarize_values(normalized[:, 2]),
        "polar_angle_from_posterior_axis_deg": summarize_values(
            polar_angle_deg
        ),
        "distance_from_eye_center_mm": summarize_values(
            np.linalg.norm(local, axis=1)
        ),
        "distance_to_ellipsoid_wall_um": summarize_values(surface_error_um),
        "_vectors": {
            "normalized_z": normalized[:, 2],
            "polar_angle_deg": polar_angle_deg,
            "surface_error_um": surface_error_um,
            "x_mm": points[:, 0],
            "y_mm": points[:, 1],
            "z_mm": points[:, 2],
        },
    }


def bias_summary(
    config: RestrictedPerceptionConfig,
    *,
    episodes: Iterable[dict],
    episode_id_prefix: str,
) -> dict:
    rows = []
    for episode in episodes:
        target_index = int(episode["target_index"])
        seed = int(episode["seed"])
        simulator = CoarseFinePerceptionSimulator(
            config,
            seed=seed,
            episode_id=f"{episode_id_prefix}_{target_index}",
        )
        rows.append(
            {
                "target_index": target_index,
                "seed": seed,
                "target_id": episode.get("target_id"),
                "coarse_bias_norm_um": float(
                    np.linalg.norm(simulator.coarse_bias) * 1000.0
                ),
                "fine_bias_norm_um": float(
                    np.linalg.norm(simulator.fine_bias_mm) * 1000.0
                ),
            }
        )
    coarse = [row["coarse_bias_norm_um"] for row in rows]
    fine = [row["fine_bias_norm_um"] for row in rows]
    return {
        "coarse_bias_norm_um": summarize_values(coarse),
        "fine_bias_norm_um": summarize_values(fine),
        "fine_bias_above_20um_episodes": int(sum(value > 20.0 for value in fine)),
        "rows": rows,
    }


def classify_failure(episode: dict) -> str:
    reason = str(episode.get("early_stop_reason", ""))
    if any(token in reason for token in RESOURCE_ERROR_TOKENS):
        return "RESOURCE_EXCEPTION"
    if SAFETY_RANGE_TOKEN in reason:
        return "UNSAFE_INITIALIZATION"
    if bool(
        episode.get("success_by_threshold_um", {}).get("20", False)
    ):
        return "SUCCESS_20UM"
    minimum_um = float(episode["minimum_error_mm"]) * 1000.0
    if (
        minimum_um >= 99.0
        and float(episode.get("final_error_mm", 0.0)) * 1000.0 >= 99.0
    ):
        return "NO_APPROACH"
    if minimum_um <= 20.0:
        return "REACHED_20UM_BUT_DID_NOT_HOLD"
    if minimum_um <= 30.0:
        return "NEAR_20_TO_30UM"
    if minimum_um <= 50.0:
        return "NEAR_30_TO_50UM"
    if minimum_um <= 100.0:
        return "NEAR_50_TO_100UM"
    return "FAR_OVER_100UM"


def failure_mode_summary(episodes: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for episode in episodes:
        groups.setdefault(classify_failure(episode), []).append(episode)
    result = {}
    for name, group in sorted(groups.items()):
        steps = np.asarray(
            [max(1, int(episode["steps"])) for episode in group],
            dtype=float,
        )
        fallback = {}
        for state in (
            "FINE_TRACKING",
            "FINE_COAST",
            "COARSE_FALLBACK",
            "PRECISION_HOLD",
        ):
            fractions = np.asarray(
                [
                    episode.get("fallback_state_frames", {}).get(state, 0)
                    / step
                    for episode, step in zip(group, steps)
                ],
                dtype=float,
            )
            fallback[state] = {
                "median_fraction": float(np.median(fractions)),
                "p95_fraction": float(np.percentile(fractions, 95.0)),
            }
        result[name] = {
            "episodes": len(group),
            "minimum_error_um": summarize_values(
                [
                    float(episode["minimum_error_mm"]) * 1000.0
                    for episode in group
                ]
            ),
            "final_error_um": summarize_values(
                [
                    float(episode["final_error_mm"]) * 1000.0
                    for episode in group
                ]
            ),
            "steps": summarize_values(steps),
            "fallback_fraction": fallback,
            "target_ids": sorted(
                {str(episode["target_id"]) for episode in group}
            ),
        }
    return result


def ks_comparison(left: Iterable[float], right: Iterable[float]) -> dict:
    from scipy.stats import ks_2samp

    result = ks_2samp(
        np.asarray(list(left), dtype=float),
        np.asarray(list(right), dtype=float),
    )
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def load_stage4_episodes(paths: Iterable[Path]) -> list[dict]:
    episodes = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        episodes.extend(payload["episodes"])
    unique = {}
    for episode in episodes:
        unique[episode["episode_key"]] = episode
    return list(unique.values())
