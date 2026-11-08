"""Build restricted coarse/fine observations for existing recovery datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import RestrictedPerceptionConfig
from .fusion import CoarseFineFusion
from .schema import FallbackState
from .simulator import CoarseFinePerceptionSimulator, SensorFrame


STATE_TO_CODE = {state.value: index for index, state in enumerate(FallbackState)}
CODE_TO_STATE = {value: key for key, value in STATE_TO_CODE.items()}
RESTRICTED_FEATURE_NAMES = (
    "coarse_goal_robot_x_mm",
    "coarse_goal_robot_y_mm",
    "coarse_goal_robot_z_mm",
    "fine_goal_robot_x_mm",
    "fine_goal_robot_y_mm",
    "fine_goal_robot_z_mm",
    "coarse_valid",
    "fine_valid",
    "coarse_age_s",
    "fine_age_s",
    "coarse_sigma_mm",
    "fine_sigma_mm",
    "switch_weight",
    "fallback_fine_tracking",
    "fallback_fine_coast",
    "fallback_coarse_fallback",
    "fallback_precision_hold",
    "precision_sufficient",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def restricted_feature_matrix(data: Any) -> np.ndarray:
    """Convert a sidecar mapping to the fixed policy whitelist tensor."""
    count = len(data["episode_index"])
    coarse = np.asarray(data["coarse_goal_vector_robot_mm"], dtype=np.float32).copy()
    fine = np.asarray(data["fine_goal_vector_robot_mm"], dtype=np.float32).copy()
    coarse_valid = np.asarray(data["coarse_valid"], dtype=bool)
    fine_valid = np.asarray(data["fine_valid"], dtype=bool)
    coarse[~np.isfinite(coarse)] = 0.0
    fine[~np.isfinite(fine)] = 0.0
    coarse[~coarse_valid] = 0.0
    fine[~fine_valid] = 0.0
    ages = np.column_stack(
        [
            np.nan_to_num(np.asarray(data["coarse_age_s"], dtype=np.float32), nan=-1.0),
            np.nan_to_num(np.asarray(data["fine_age_s"], dtype=np.float32), nan=-1.0),
        ]
    )
    sigmas = np.column_stack(
        [
            np.nan_to_num(np.asarray(data["coarse_sigma_mm"], dtype=np.float32), nan=-1.0),
            np.nan_to_num(np.asarray(data["fine_sigma_mm"], dtype=np.float32), nan=-1.0),
        ]
    )
    weight = np.asarray(data["switch_weight"], dtype=np.float32)[:, None]
    state_codes = np.asarray(data["fallback_state"], dtype=np.int8)
    if ((state_codes < 0) | (state_codes >= len(STATE_TO_CODE))).any():
        raise ValueError("fallback_state contains an unknown code")
    state_one_hot = np.eye(len(STATE_TO_CODE), dtype=np.float32)[state_codes]
    precision = np.asarray(data["precision_sufficient"], dtype=np.float32)[:, None]
    features = np.column_stack(
        [
            coarse,
            fine,
            coarse_valid.astype(np.float32),
            fine_valid.astype(np.float32),
            ages,
            sigmas,
            weight,
            state_one_hot,
            precision,
        ]
    ).astype(np.float32)
    if features.shape != (count, len(RESTRICTED_FEATURE_NAMES)):
        raise ValueError(
            f"restricted feature shape mismatch: {features.shape}, "
            f"expected {(count, len(RESTRICTED_FEATURE_NAMES))}"
        )
    if not np.isfinite(features).all():
        raise ValueError("restricted feature tensor contains non-finite values")
    return features


def _kinematic_tips(states: np.ndarray, reference: Any) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3 or not np.isfinite(states).all():
        raise ValueError("states must be a finite Nx3 array")
    phi_deg = reference.phi_home + states[:, 0]
    alpha_deg = reference.alpha_home + states[:, 1]
    length_mm = reference.tip_reference_mm + states[:, 2]
    phi = np.radians(phi_deg)
    theta = np.radians(alpha_deg + reference.nominal.geometry.tilt_offset_deg)
    sin_theta = np.sin(theta)
    return np.column_stack(
        [
            length_mm * sin_theta * np.sin(phi),
            length_mm * np.cos(theta),
            length_mm * sin_theta * np.cos(phi),
        ]
    )


def build_restricted_sidecar(
    dataset_npz: Path,
    config: RestrictedPerceptionConfig,
    output_dir: Path,
    *,
    seed: int = 20261201,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Write a sidecar without persisting reconstructed target truth."""
    from rcm_reference import Reference

    dataset_npz = Path(dataset_npz).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(dataset_npz, allow_pickle=True)
    required = ("episode_index", "frame_index", "state", "oct_navigation", "oct_navigation_valid")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"dataset is missing required arrays: {missing}")
    reference = Reference()
    rotation = config.extrinsic.camera_to_robot_rotation
    collected: dict[str, list[Any]] = {
        "episode_index": [],
        "frame_index": [],
        "state": [],
        "kinematic_tip_robot_mm": [],
        "coarse_target_robot_mm": [],
        "coarse_goal_vector_robot_mm": [],
        "fine_goal_vector_camera_mm": [],
        "fine_goal_vector_robot_mm": [],
        "coarse_valid": [],
        "fine_valid": [],
        "coarse_age_s": [],
        "fine_age_s": [],
        "coarse_sigma_mm": [],
        "fine_sigma_mm": [],
        "coarse_bias_mm": [],
        "fine_bias_mm": [],
        "switch_weight": [],
        "fallback_state": [],
        "precision_sufficient": [],
        "fused_goal_vector_robot_mm": [],
        "action_mask_xyz": [],
        "retraction_command_robot_mm": [],
        "retraction_active": [],
        "precision_threshold_um": [],
    }
    excluded_episodes = []
    included_episodes = []
    episode_values = np.asarray(data["episode_index"], dtype=np.int64)
    unique_episodes = np.unique(episode_values)
    if max_episodes is not None:
        unique_episodes = unique_episodes[: int(max_episodes)]
    for episode_index in unique_episodes:
        rows = np.flatnonzero(episode_values == episode_index)
        rows = rows[np.argsort(np.asarray(data["frame_index"])[rows], kind="stable")]
        states = np.asarray(data["state"], dtype=float)[rows]
        fine_camera_all = np.asarray(data["oct_navigation"], dtype=float)[rows]
        fine_valid_all = np.asarray(data["oct_navigation_valid"], dtype=bool)[rows]
        if len(rows) == 0 or not np.any(fine_valid_all):
            excluded_episodes.append(
                {"episode_index": int(episode_index), "reason": "no_valid_fine_measurement"}
            )
            continue
        tips = _kinematic_tips(states, reference)
        first_valid = int(np.flatnonzero(fine_valid_all)[0])
        target_robot_mm = tips[first_valid] + rotation @ fine_camera_all[first_valid]
        simulator = CoarseFinePerceptionSimulator(
            config,
            seed=seed,
            episode_id=f"dataset_episode_{int(episode_index)}",
        )
        fusion = CoarseFineFusion(config)
        rng = np.random.default_rng(seed + int(episode_index))
        from scipy.signal import lfilter

        frame_count = len(rows)
        distance = np.linalg.norm(target_robot_mm[None, :] - tips, axis=1)
        coarse_sigma = (
            config.coarse.noise_floor_mm
            + config.coarse.noise_slope_mm_per_mm * distance
        )
        drift_innovation = simulator.rng.normal(
            0.0,
            config.coarse.drift_innovation_sigma_mm,
            size=(frame_count, 3),
        )
        coarse_drift = lfilter(
            [1.0],
            [1.0, -config.coarse.drift_ar1_coefficient],
            drift_innovation,
            axis=0,
        )
        coarse_noise = simulator.rng.normal(
            0.0,
            coarse_sigma[:, None],
            size=(frame_count, 3),
        )
        coarse_measurements = (
            target_robot_mm[None, :]
            + simulator.coarse_bias[None, :]
            + coarse_drift
            + coarse_noise
        )
        coarse_available = distance <= config.coarse.max_distance_mm
        coarse_delay = max(0, int(config.coarse.delay_frames))
        delayed_coarse = np.full_like(coarse_measurements, np.nan)
        delayed_coarse_valid = np.zeros(frame_count, dtype=bool)
        delayed_coarse_sigma = np.full(frame_count, np.nan)
        if coarse_delay == 0:
            delayed_coarse = coarse_measurements
            delayed_coarse_valid = coarse_available
            delayed_coarse_sigma = coarse_sigma
        else:
            delayed_coarse[coarse_delay:] = coarse_measurements[:-coarse_delay]
            delayed_coarse_valid[coarse_delay:] = coarse_available[:-coarse_delay]
            delayed_coarse_sigma[coarse_delay:] = coarse_sigma[:-coarse_delay]
        delayed_coarse[~delayed_coarse_valid] = np.nan
        fine_sigma = (
            config.fine.noise_floor_mm
            + config.fine.noise_slope_mm_per_mm * distance
        )
        camera_goal = rotation.T @ (
            target_robot_mm[None, :] - tips
        ).T
        cosine = np.abs(camera_goal[2]) / np.maximum(distance, 1e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        fine_available = (
            fine_valid_all
            & (distance >= config.fine.min_range_mm)
            & (distance <= config.fine.max_range_mm)
            & (angle_deg <= config.fine.fov_half_angle_deg)
            & (rng.random(frame_count) >= config.fine.dropout_probability)
        )
        fine_measurements = np.full((frame_count, 3), np.nan)
        fine_measurements[fine_available] = fine_camera_all[fine_available]
        fine_observation_tips = np.full((frame_count, 3), np.nan)
        fine_observation_tips[fine_available] = tips[fine_available]
        fine_delay = max(0, int(config.fine.delay_frames))
        delayed_fine = np.full_like(fine_measurements, np.nan)
        delayed_fine_tips = np.full_like(fine_observation_tips, np.nan)
        delayed_fine_valid = np.zeros(frame_count, dtype=bool)
        delayed_fine_sigma = np.full(frame_count, np.nan)
        if fine_delay == 0:
            delayed_fine = fine_measurements
            delayed_fine_tips = fine_observation_tips
            delayed_fine_valid = fine_available
            delayed_fine_sigma = fine_sigma
        else:
            delayed_fine[fine_delay:] = fine_measurements[:-fine_delay]
            delayed_fine_tips[fine_delay:] = fine_observation_tips[:-fine_delay]
            delayed_fine_valid[fine_delay:] = fine_available[:-fine_delay]
            delayed_fine_sigma[fine_delay:] = fine_sigma[:-fine_delay]
        included_frames = 0
        for local_index, row in enumerate(rows):
            timestamp_s = local_index * config.control_period_s
            tip = tips[local_index]
            coarse_valid = bool(delayed_coarse_valid[local_index])
            fine_measurement_valid = bool(delayed_fine_valid[local_index])
            sample = SensorFrame(
                timestamp_s=timestamp_s,
                kinematic_tip_robot_mm=tip.copy(),
                fine_observation_tip_robot_mm=(
                    delayed_fine_tips[local_index].copy()
                    if fine_measurement_valid
                    else None
                ),
                coarse_target_robot_mm=(
                    delayed_coarse[local_index].copy() if coarse_valid else None
                ),
                fine_goal_vector_camera_mm=(
                    delayed_fine[local_index].copy()
                    if fine_measurement_valid
                    else None
                ),
                coarse_valid=coarse_valid,
                fine_valid=fine_measurement_valid,
                coarse_age_s=(
                    coarse_delay * config.control_period_s if coarse_valid else None
                ),
                fine_age_s=(
                    fine_delay * config.control_period_s
                    if fine_measurement_valid
                    else None
                ),
                coarse_sigma_mm=(
                    float(delayed_coarse_sigma[local_index])
                    if coarse_valid
                    else None
                ),
                fine_sigma_mm=(
                    float(delayed_fine_sigma[local_index])
                    if fine_measurement_valid
                    else None
                ),
                coarse_bias_mm=simulator.coarse_bias,
                fine_bias_mm=simulator.fine_bias_mm,
            )
            observation = fusion.step(sample, state=states[local_index])
            collected["episode_index"].append(int(episode_index))
            collected["frame_index"].append(int(data["frame_index"][row]))
            collected["state"].append(states[local_index])
            collected["kinematic_tip_robot_mm"].append(observation.kinematic_tip_robot_mm)
            collected["coarse_target_robot_mm"].append(observation.coarse_target_robot_mm)
            collected["coarse_goal_vector_robot_mm"].append(
                observation.coarse_goal_vector_robot_mm
            )
            collected["fine_goal_vector_camera_mm"].append(
                observation.fine_goal_vector_camera_mm
            )
            collected["fine_goal_vector_robot_mm"].append(
                observation.fine_goal_vector_robot_mm
            )
            collected["coarse_valid"].append(observation.coarse_valid)
            collected["fine_valid"].append(observation.fine_valid)
            collected["coarse_age_s"].append(observation.coarse_age_s)
            collected["fine_age_s"].append(observation.fine_age_s)
            collected["coarse_sigma_mm"].append(observation.coarse_sigma_mm)
            collected["fine_sigma_mm"].append(observation.fine_sigma_mm)
            collected["coarse_bias_mm"].append(observation.coarse_bias_mm)
            collected["fine_bias_mm"].append(observation.fine_bias_mm)
            collected["switch_weight"].append(observation.switch_weight)
            collected["fallback_state"].append(STATE_TO_CODE[observation.fallback_state.value])
            collected["precision_sufficient"].append(observation.precision_sufficient)
            collected["fused_goal_vector_robot_mm"].append(
                observation.selected_goal_vector_robot_mm
            )
            collected["action_mask_xyz"].append(observation.action_mask_xyz)
            collected["retraction_command_robot_mm"].append(
                observation.retraction_command_robot_mm
            )
            collected["retraction_active"].append(observation.retraction_active)
            collected["precision_threshold_um"].append(observation.precision_threshold_um)
            included_frames += 1
        included_episodes.append(
            {"episode_index": int(episode_index), "frames": included_frames}
        )

    arrays = {}
    for key, values in collected.items():
        if key in ("episode_index", "frame_index"):
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key in ("state", "kinematic_tip_robot_mm", "coarse_bias_mm", "fine_bias_mm"):
            arrays[key] = np.asarray(values, dtype=np.float32)
        elif key in (
            "coarse_target_robot_mm",
            "coarse_goal_vector_robot_mm",
            "fine_goal_vector_camera_mm",
            "fine_goal_vector_robot_mm",
            "fused_goal_vector_robot_mm",
            "action_mask_xyz",
            "retraction_command_robot_mm",
        ):
            arrays[key] = np.asarray(
                [
                    np.full(3, np.nan, dtype=np.float32) if value is None else value
                    for value in values
                ],
                dtype=np.float32,
            )
        elif key in (
            "coarse_age_s",
            "fine_age_s",
            "coarse_sigma_mm",
            "fine_sigma_mm",
            "switch_weight",
            "precision_threshold_um",
        ):
            arrays[key] = np.asarray(
                [np.nan if value is None else value for value in values],
                dtype=np.float32,
            )
        elif key in (
            "coarse_valid",
            "fine_valid",
            "precision_sufficient",
            "retraction_active",
        ):
            arrays[key] = np.asarray(values, dtype=bool)
        else:
            arrays[key] = np.asarray(values, dtype=np.int8)
    np.savez_compressed(output_dir / "restricted_sidecar.npz", **arrays)
    metadata = {
        "schema_version": 1,
        "source_dataset": str(dataset_npz),
        "source_sha256": _sha256(dataset_npz),
        "config_sha256": _sha256(Path(__file__).resolve().parents[1] / "configs" / "restricted_perception.json"),
        "samples": int(len(arrays["state"])),
        "source_samples": int(len(episode_values)),
        "included_episodes": len(included_episodes),
        "excluded_episodes": excluded_episodes,
        "frame_fields": sorted(arrays),
        "policy_feature_fields": [
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
        ],
        "audit_only_fields": ["coarse_target_robot_mm", "coarse_bias_mm", "fine_bias_mm"],
        "truth_used_for_policy": False,
        "fallback_state_codes": STATE_TO_CODE,
        "scope": "Recovery data conditional measurements; simulator truth is not serialized.",
    }
    (output_dir / "restricted_sidecar_metadata.json").write_text(
        _json_text(metadata),
        encoding="utf-8",
    )
    return metadata
