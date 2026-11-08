"""Frozen helpers shared by the Stage 3 and Stage 4 experiments."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import RestrictedPerceptionConfig


TASK_TO_SI = np.asarray(
    [math.pi / 180.0, math.pi / 180.0, 0.001],
    dtype=float,
)
SI_TO_ACTION = np.asarray(
    [180.0 / math.pi, 180.0 / math.pi, 1000.0],
    dtype=float,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_stage34_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported Stage 3/4 protocol schema")
    if payload.get("frozen_before_stage3_results") is not True:
        raise ValueError("Stage 3/4 protocol was not frozen before results")
    if int(payload["stage3"]["development_targets"]) != 40:
        raise ValueError("Stage 3 requires exactly 40 development targets")
    if int(payload["stage4"]["test_targets"]) != 60:
        raise ValueError("Stage 4 requires exactly 60 final-test targets")
    if int(payload["stage4"]["seeds_per_target"]) != 3:
        raise ValueError("Stage 4 requires exactly three seeds per target")
    return payload


def stage4_run_specs(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = manifest["final_test_targets"]
    if len(targets) != int(protocol["stage4"]["test_targets"]):
        raise ValueError("final-test target count does not match the frozen protocol")
    base_seed = int(protocol["seed"]) + 10000
    specs = []
    for target_index, target in enumerate(targets):
        for replicate in range(int(protocol["stage4"]["seeds_per_target"])):
            seed = base_seed + target_index * 3 + replicate
            specs.append(
                {
                    "target_index": target_index,
                    "target_id": target["id"],
                    "target_robot_mm": list(target["point_mm"]),
                    "replicate": replicate,
                    "seed": seed,
                    "episode_key": f"{target['id']}:{seed}",
                }
            )
    return specs


def median_control_specs(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    subset_ids = {item["id"] for item in manifest["median_subset_20"]}
    specs = [
        spec
        for spec in stage4_run_specs(manifest, protocol)
        if spec["target_id"] in subset_ids
    ]
    expected = int(protocol["stage4"]["median_control_episodes"])
    if len(specs) != expected:
        raise ValueError(
            f"median control requires {expected} runs, got {len(specs)}"
        )
    return specs


def scenario_configuration(
    config: RestrictedPerceptionConfig,
    scenario: str,
    protocol: dict[str, Any],
) -> tuple[RestrictedPerceptionConfig, dict[str, float]]:
    if scenario == "default":
        return config, {
            "gaussian_blur_radius": 1.5,
            "center_occlusion_fraction": 0.25,
        }
    scenarios = protocol["stage3"]["robustness"]["perturbation_scenarios"]
    if scenario not in scenarios:
        raise ValueError(f"unknown robustness scenario: {scenario}")
    settings = scenarios[scenario]
    coarse = config.coarse
    fine = config.fine
    image = {
        "gaussian_blur_radius": 1.5,
        "center_occlusion_fraction": 0.25,
    }
    if scenario == "noise":
        coarse = replace(
            coarse,
            noise_floor_mm=(
                coarse.noise_floor_mm
                * float(settings["coarse_noise_multiplier"])
            ),
            noise_slope_mm_per_mm=(
                coarse.noise_slope_mm_per_mm
                * float(settings["coarse_noise_multiplier"])
            ),
            drift_innovation_sigma_mm=(
                coarse.drift_innovation_sigma_mm
                * float(settings["coarse_noise_multiplier"])
            ),
        )
        fine = replace(
            fine,
            noise_floor_mm=fine.noise_floor_mm
            * float(settings["fine_noise_multiplier"]),
            noise_slope_mm_per_mm=(
                fine.noise_slope_mm_per_mm
                * float(settings["fine_noise_multiplier"])
            ),
        )
    elif scenario == "occlusion":
        image["center_occlusion_fraction"] = float(
            settings["center_occlusion_fraction"]
        )
    elif scenario == "oct_loss":
        fine = replace(
            fine,
            dropout_probability=float(settings["fine_dropout_probability"]),
        )
    elif scenario == "delay":
        coarse = replace(
            coarse,
            delay_frames=max(
                int(coarse.delay_frames),
                int(settings["coarse_delay_frames"]),
            ),
        )
        fine = replace(
            fine,
            delay_frames=max(
                int(fine.delay_frames),
                int(settings["fine_delay_frames"]),
            ),
        )
    return replace(config, coarse=coarse, fine=fine), image


def robot_task_jacobian(
    reference: Any,
    state_abc_mm: np.ndarray,
) -> np.ndarray:
    state = np.asarray(state_abc_mm, dtype=float)
    if state.shape != (3,) or not np.isfinite(state).all():
        raise ValueError("state_abc_mm must be a finite 3-vector")
    q_si = state * TASK_TO_SI
    steps = np.asarray([1e-4, 1e-4, 1e-5], dtype=float)
    jacobian = np.empty((3, 3), dtype=float)
    for axis, step in enumerate(steps):
        plus = q_si.copy()
        minus = q_si.copy()
        plus[axis] += step
        minus[axis] -= step
        jacobian[:, axis] = (
            np.asarray(reference.forward_kinematics(plus), dtype=float)
            - np.asarray(reference.forward_kinematics(minus), dtype=float)
        ) / (2.0 * step)
    if not np.isfinite(jacobian).all():
        raise ValueError("non-finite robot task Jacobian")
    return jacobian


def damped_pseudoinverse(
    jacobian: np.ndarray,
    error: np.ndarray,
    relative_damping: float = 1e-3,
) -> np.ndarray:
    matrix = np.asarray(jacobian, dtype=float)
    vector = np.asarray(error, dtype=float)
    if matrix.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("jacobian must be 3x3 and error must have length three")
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] <= 1e-12 or not np.isfinite(singular).all():
        raise ValueError("singular robot task Jacobian")
    damping = max(float(singular[0]) * float(relative_damping), 1e-12)
    return vt.T @ (
        singular / (singular**2 + damping**2) * (u.T @ vector)
    )


def geometric_teacher_action(
    reference: Any,
    state_abc_mm: np.ndarray,
    goal_vector_robot_mm: np.ndarray,
    *,
    gain: float,
    saturation: np.ndarray | None = None,
) -> np.ndarray:
    vector = np.asarray(goal_vector_robot_mm, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("goal_vector_robot_mm must be finite")
    if not math.isfinite(float(gain)) or gain < 0.0:
        raise ValueError("gain must be finite and nonnegative")
    limits = (
        np.asarray([0.2897999882698059, 0.14434358477592468, 0.04271206632256508])
        if saturation is None
        else np.asarray(saturation, dtype=float)
    )
    if np.linalg.norm(vector) <= 1e-12:
        return np.zeros(3, dtype=float)
    delta_si = damped_pseudoinverse(
        robot_task_jacobian(reference, state_abc_mm),
        vector,
    )
    action = float(gain) * (delta_si * SI_TO_ACTION)
    return np.clip(action, -limits, limits)


def fallback_state_of(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=float)
    if values.shape != (18,):
        raise ValueError("restricted feature vector must have length 18")
    names = (
        "FINE_TRACKING",
        "FINE_COAST",
        "COARSE_FALLBACK",
        "PRECISION_HOLD",
    )
    return names[int(np.argmax(values[13:17]))]
