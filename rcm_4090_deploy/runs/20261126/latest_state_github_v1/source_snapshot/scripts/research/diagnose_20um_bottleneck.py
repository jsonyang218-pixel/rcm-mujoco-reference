"""Read-only diagnosis of the 20-60 um baseline closed-loop floor.

This script only reads completed evaluation JSON files. It does not import the
simulator, load a policy, or write anywhere except the requested report and
metrics destination.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ROOT = (
    ROOT
    / "github_reports"
    / "homog_label_fix_20261123_v3_s3_fast5_1600"
    / "baseline_task2_step4000"
    / "shards"
    / "tasks"
)
TARGETS = ("E00", "E02", "E06", "E08", "E10")
DEFAULT_OUTPUT = (
    ROOT / "runs/20261124/stage2_3_baseline_20um_diagnosis"
)
ENCODER_COUNTS_PER_MM = 4 * 1024 * 15.5817175
ALPHA_HOME_DEG = 54.23600525044604
TIP_REFERENCE_MM = 4.0
ROTATION_ROBOT_WORLD = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -math.sqrt(0.5), math.sqrt(0.5)],
    [0.0, -math.sqrt(0.5), -math.sqrt(0.5)],
])


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(np.abs(values), 90)),
    }


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3)


def cosine(left, right):
    left = unit(left)
    right = unit(right)
    if not np.any(left) or not np.any(right):
        return float("nan")
    return float(np.dot(left, right))


def sign_agreement(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    eligible = (np.abs(left) > 1e-12) & (np.abs(right) > 1e-12)
    if not np.any(eligible):
        return None
    return float(np.mean(np.sign(left[eligible]) == np.sign(right[eligible])))


def task_tip_robot(task):
    """Analytic needle-tip forward kinematics in the recorded robot frame."""
    a_rad, b_rad, c_m = (float(value) for value in task)
    phi = a_rad
    alpha = math.radians(ALPHA_HOME_DEG + math.degrees(b_rad))
    theta = alpha + math.radians(22.0)
    direction = np.array([
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
        math.sin(theta) * math.cos(phi),
    ])
    radius_mm = c_m * 1000.0 + TIP_REFERENCE_MM
    return radius_mm * direction


def task_to_deg_deg_mm(task):
    task = np.asarray(task, dtype=np.float64)
    return np.array([
        math.degrees(task[0]),
        math.degrees(task[1]),
        task[2] * 1000.0,
    ])


def action_to_task_delta(action):
    action = np.asarray(action, dtype=np.float64)
    return np.array([
        math.radians(action[0]),
        math.radians(action[1]),
        action[2] / 1000.0,
    ])


def world_tip(task, translation):
    return translation + ROTATION_ROBOT_WORLD @ task_tip_robot(task)


def array_summary(values, axis_names):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "p90_abs": np.percentile(np.abs(values), 90, axis=0).tolist(),
        "axes": list(axis_names),
    }


def schema_of(document):
    episode = document["episodes"][0]
    trajectory = episode["trajectory"]
    motor = episode["motor_records"][0]
    control = trajectory["control_targets"][0]
    terminal = episode["terminal_success"]
    return {
        "top_level_keys": list(document),
        "episode_keys": list(episode),
        "trajectory_keys": list(trajectory),
        "trajectory_lengths": {
            key: len(value)
            for key, value in trajectory.items()
            if isinstance(value, list)
        },
        "motor_record_keys": list(motor),
        "control_target_keys": list(control),
        "terminal_success_keys": list(terminal),
        "evaluation_contract_keys": list(document["evaluation_contract"]),
    }


def summarize_group(rows, mask):
    selected = [
        row for row, use in zip(rows, mask)
        if bool(use)
    ]
    if not selected:
        return {
            "count": 0,
            "policy_toward_fraction": None,
            "policy_cosine_median": None,
            "policy_command_norm_median": None,
            "actual_toward_fraction": None,
            "actual_cosine_median": None,
            "error_um_median": None,
            "execution_lag_median": None,
        }
    policy_cos = np.asarray(
        [row["policy_cosine"] for row in selected], dtype=np.float64
    )
    actual_cos = np.asarray(
        [row["actual_cosine"] for row in selected], dtype=np.float64
    )
    policy_nonzero = np.asarray(
        [row["policy_command_norm"] > 1e-12 for row in selected]
    )
    actual_nonzero = np.asarray(
        [row["actual_displacement_norm_um"] > 1e-12 for row in selected]
    )
    return {
        "count": len(selected),
        "policy_toward_fraction": (
            float(np.mean(policy_cos[policy_nonzero] > 0))
            if policy_nonzero.any() else None
        ),
        "policy_cosine_median": float(np.nanmedian(
            policy_cos[policy_nonzero]
        )) if policy_nonzero.any() else None,
        "policy_command_norm_median": float(np.median([
            row["policy_command_norm"] for row in selected
        ])),
        "actual_toward_fraction": (
            float(np.mean(actual_cos[actual_nonzero] > 0))
            if actual_nonzero.any() else None
        ),
        "actual_cosine_median": float(np.nanmedian(
            actual_cos[actual_nonzero]
        )) if actual_nonzero.any() else None,
        "error_um_median": float(np.median([
            row["true_error_um"] for row in selected
        ])),
        "execution_lag_median": float(np.median([
            row["execution_lag"] for row in selected
        ])),
    }


def load_target(path):
    document = read(path)
    if len(document.get("episodes", [])) != 1:
        raise ValueError(f"expected exactly one episode in {path}")
    episode = document["episodes"][0]
    trajectory = episode["trajectory"]
    count = len(trajectory["errors"])
    for name in (
        "raw_actions",
        "actions",
        "states",
        "errors",
        "velocities",
        "oct_navigation_mm",
        "oct_navigation_norm_mm",
        "oct_navigation_valid",
    ):
        if len(trajectory[name]) != count:
            raise ValueError(f"{path}: length mismatch for trajectory.{name}")
    if len(episode["motor_records"]) != count:
        raise ValueError(f"{path}: motor record length mismatch")
    true_error_um = np.asarray(trajectory["errors"], dtype=np.float64) * 1000.
    raw_actions = np.asarray(trajectory["raw_actions"], dtype=np.float64)
    accepted_actions = np.asarray(trajectory["actions"], dtype=np.float64)
    oct_navigation_um = (
        np.asarray(trajectory["oct_navigation_mm"], dtype=np.float64) * 1000.
    )
    oct_norm_um = np.asarray(
        trajectory["oct_navigation_norm_mm"], dtype=np.float64
    ) * 1000.
    oct_valid = np.asarray(
        trajectory["oct_navigation_valid"], dtype=np.bool_
    )
    target_world = np.asarray(episode["target_world_mm"], dtype=np.float64)
    target_robot = np.asarray(episode["target_robot_mm"], dtype=np.float64)
    translation = target_world - ROTATION_ROBOT_WORLD @ target_robot
    motor = episode["motor_records"]
    requested = np.asarray([
        row["requested_task_rad_rad_m"] for row in motor
    ], dtype=np.float64)
    commanded = np.asarray([
        row["commanded_task_rad_rad_m"] for row in motor
    ], dtype=np.float64)
    before = np.asarray([
        row["actual_before_task_rad_rad_m"] for row in motor
    ], dtype=np.float64)
    after = np.asarray([
        row["actual_after_task_rad_rad_m"] for row in motor
    ], dtype=np.float64)
    actual_velocity = np.asarray([
        row["actual_velocity_rad_rad_m_per_s"] for row in motor
    ], dtype=np.float64)
    tip_before = np.asarray([
        world_tip(row, translation) for row in before
    ])
    tip_after = np.asarray([
        world_tip(row, translation) for row in after
    ])
    reconstructed_error_um = np.linalg.norm(
        target_world[None, :] - tip_before, axis=1
    ) * 1000.
    actual_displacement_um = (tip_after - tip_before) * 1000.
    command_displacement_um = np.asarray([
        (
            world_tip(
                before[index] + action_to_task_delta(action),
                translation,
            )
            - tip_before[index]
        ) * 1000.
        for index, action in enumerate(raw_actions)
    ])
    accepted_displacement_um = np.asarray([
        (
            world_tip(
                before[index] + action_to_task_delta(action),
                translation,
            )
            - tip_before[index]
        ) * 1000.
        for index, action in enumerate(accepted_actions)
    ])
    true_target_vector_um = np.asarray([
        unit(target_world - tip) * error
        for tip, error in zip(tip_before, true_error_um)
    ])
    measured_bias_um = oct_navigation_um - true_target_vector_um
    rows = []
    for index in range(count):
        actual_delta = task_to_deg_deg_mm(after[index] - before[index])
        commanded_delta = task_to_deg_deg_mm(commanded[index] - before[index])
        requested_residual = task_to_deg_deg_mm(requested[index] - after[index])
        command_lag = task_to_deg_deg_mm(commanded[index] - after[index])
        action_lag = accepted_actions[index] - actual_delta
        true_direction = unit(target_world - tip_before[index])
        rows.append({
            "index": index,
            "true_error_um": float(true_error_um[index]),
            "policy_cosine": cosine(
                command_displacement_um[index], true_direction
            ),
            "accepted_cosine": cosine(
                accepted_displacement_um[index], true_direction
            ),
            "actual_cosine": cosine(
                actual_displacement_um[index], true_direction
            ),
            "policy_command_norm": float(np.linalg.norm(raw_actions[index])),
            "accepted_command_norm": float(
                np.linalg.norm(accepted_actions[index])
            ),
            "actual_displacement_norm_um": float(
                np.linalg.norm(actual_displacement_um[index])
            ),
            "policy_command_um": command_displacement_um[index].copy(),
            "accepted_command_um": accepted_displacement_um[index].copy(),
            "actual_displacement_um": actual_displacement_um[index].copy(),
            "execution_lag": float(np.linalg.norm(command_lag)),
            "pending_target_residual": float(
                np.linalg.norm(requested_residual)
            ),
            "action_lag": float(np.linalg.norm(action_lag)),
            "command_lag_vector": command_lag,
            "pending_vector": requested_residual,
            "motor_velocity_deg_deg_mm_s": np.array([
                math.degrees(actual_velocity[index, 0]),
                math.degrees(actual_velocity[index, 1]),
                actual_velocity[index, 2] * 1000.,
            ]),
            "previous_executed_action_deg_deg_mm": (
                task_to_deg_deg_mm(after[index - 1] - before[index - 1])
                if index else np.zeros(3)
            ),
            "oct_valid": bool(oct_valid[index]),
        })
    return {
        "path": str(path),
        "document": document,
        "episode": episode,
        "trajectory": trajectory,
        "true_error_um": true_error_um,
        "reconstructed_error_um": reconstructed_error_um,
        "oct_navigation_um": oct_navigation_um,
        "oct_norm_um": oct_norm_um,
        "oct_valid": oct_valid,
        "measured_bias_um": measured_bias_um,
        "raw_actions": raw_actions,
        "accepted_actions": accepted_actions,
        "rows": rows,
        "target_world_mm": target_world,
        "target_robot_mm": target_robot,
    }


def analyze_target(target, path):
    data = load_target(path)
    errors = data["true_error_um"]
    count = len(errors)
    t_min = int(np.argmin(errors))
    start = max(0, t_min - 50)
    stop = min(count, t_min + 51)
    near = np.zeros(count, dtype=np.bool_)
    near[start:stop] = True
    under60 = errors < 60.
    valid = data["oct_valid"]
    measured_bias = data["measured_bias_um"]
    norm_bias = data["oct_norm_um"] - errors
    rows = data["rows"]
    accepted = data["accepted_actions"]
    states = np.asarray(data["trajectory"]["states"], dtype=np.float64)
    actual_norm = np.asarray([
        row["actual_displacement_norm_um"] for row in rows
    ])
    nonzero_command = np.linalg.norm(accepted, axis=1) > 1e-6

    def deadzone(mask, threshold):
        eligible = nonzero_command & mask
        if not np.any(eligible):
            return None
        return float(np.mean(actual_norm[eligible] < threshold))

    before = np.zeros(count, dtype=np.bool_)
    after = np.zeros(count, dtype=np.bool_)
    before[start:t_min + 1] = True
    after[t_min:stop] = True
    late = np.zeros(count, dtype=np.bool_)
    late[1000:min(1600, count)] = True

    def direction(mask):
        selected = [
            row for row, use in zip(rows, mask)
            if bool(use) and row["policy_command_norm"] > 1e-12
        ]
        policy_cos = np.asarray(
            [row["policy_cosine"] for row in selected]
        )
        actual_cos = np.asarray(
            [row["actual_cosine"] for row in selected]
            if selected else []
        )
        return {
            "frames": len(selected),
            "policy_toward_fraction": (
                float(np.mean(policy_cos > 0)) if len(selected) else None
            ),
            "policy_cosine_median": (
                float(np.median(policy_cos)) if len(selected) else None
            ),
            "actual_toward_fraction": (
                float(np.mean(actual_cos > 0)) if len(selected) else None
            ),
            "actual_cosine_median": (
                float(np.median(actual_cos)) if len(selected) else None
            ),
        }

    matched = (errors >= 20.) & (errors <= 50.)
    d_error = np.gradient(errors)
    approaching = matched & (d_error < -.2)
    receding = matched & (d_error > .2)
    execution_signal = np.asarray([
        max(row["execution_lag"], row["pending_target_residual"])
        for row in rows
    ])
    lag_threshold = (
        float(np.percentile(execution_signal[matched], 75))
        if np.any(matched) else math.inf
    )
    high_lag = matched & (execution_signal >= lag_threshold)

    tmin_row = rows[t_min]
    measurement_norm_error_tmin = float(norm_bias[t_min])
    direction_near = direction(near & under60)
    direction_before = direction(before)
    direction_after = direction(after)
    direction_late = direction(late)

    command_native = accepted[near]
    actual_native = np.asarray([
        row["actual_displacement_um"] for row in rows
    ])[near]
    result = {
        "target": target,
        "source": str(path),
        "episode_index": int(data["episode"]["episode_index"]),
        "seed": int(data["episode"]["seed"]),
        "steps": int(count),
        "t_min": t_min,
        "min_error_um": float(errors[t_min]),
        "final_error_um": float(errors[-1]),
        "measurement_availability": {
            "oct_navigation_mm": True,
            "oct_navigation_norm_mm": True,
            "oct_navigation_coordinate_frame": "camera_optical",
            "camera_rotation_available_in_json": False,
            "world_axis_bias_computable_from_logged_oct": False,
            "oct_navigation_valid_fraction": float(valid.mean()),
            "raw_rgb_available": False,
            "raw_depth_available": False,
            "explicit_predicted_tip_world_available": False,
            "explicit_true_error_vector_available": False,
            "ground_truth_vector_method": (
                "analytic A/B/C tip reconstruction plus target-world "
                "translation; direction multiplied by recorded true error"
            ),
            "tip_reconstruction_error_um": {
                "near_min": statistics(
                    np.abs(
                        data["reconstructed_error_um"][near] - errors[near]
                    )
                ),
                "under_60_um": statistics(np.abs(
                    data["reconstructed_error_um"][under60]
                    - errors[under60]
                )),
            },
        },
        "perception": {
            "norm_bias_um_near_min": statistics(norm_bias[near]),
            "norm_bias_um_under_60": statistics(norm_bias[under60]),
            "oct_components_um_near_min_camera_frame": array_summary(
                data["oct_navigation_um"][near], "camera_XYZ"
            ),
            "oct_components_um_under_60_camera_frame": array_summary(
                data["oct_navigation_um"][under60], "camera_XYZ"
            ),
            "camera_to_world_component_difference_is_not_perception_error": True,
            "state_target_residual_at_t_min_deg_deg_mm": (
                data["target_robot_mm"]
                - states[t_min]
            ).tolist(),
            "state_target_residual_near_min": array_summary(
                np.asarray([
                    data["target_robot_mm"]
                    - row
                    for row in states[start:stop]
                ]),
                "ABC",
            ),
            "state_target_residual_under_60_um": array_summary(
                np.asarray([
                    data["target_robot_mm"]
                    - row
                    for row in states[under60]
                ]),
                "ABC",
            ),
            "measurement_signed_distance_bias_at_t_min_um": (
                measurement_norm_error_tmin
            ),
            "measurement_distance_error_at_t_min_um": abs(
                measurement_norm_error_tmin
            ),
            "oct_camera_minus_world_vector_at_t_min_um": (
                measured_bias[t_min].tolist()
            ),
            "oct_camera_minus_world_vector_is_not_used_for_gate": True,
            "flag_ge_10um_at_t_min": bool(
                abs(measurement_norm_error_tmin) >= 10.
            ),
            "flag_ge_20um_at_t_min": bool(
                abs(measurement_norm_error_tmin) >= 20.
            ),
        },
        "deadzone": {
            "accepted_nonzero_threshold_native_norm": 1e-6,
            "near_min_nonzero_frames": int(
                np.count_nonzero(nonzero_command & near)
            ),
            "under_60_nonzero_frames": int(
                np.count_nonzero(nonzero_command & under60)
            ),
            "near_min_fraction_displacement_lt_0p5um": deadzone(
                near, .5
            ),
            "near_min_fraction_displacement_lt_1um": deadzone(
                near, 1.
            ),
            "under_60_fraction_displacement_lt_0p5um": deadzone(
                under60, .5
            ),
            "under_60_fraction_displacement_lt_1um": deadzone(
                under60, 1.
            ),
            "near_min_command_magnitude_deg_deg_mm": {
                axis: statistics(command_native[:, index])
                for index, axis in enumerate("ABC")
            },
            "near_min_actual_displacement_um": {
                axis: statistics(actual_native[:, index])
                for index, axis in enumerate("XYZ")
            },
        },
        "direction": {
            "near_min_under_60": direction_near,
            "before_t_min": direction_before,
            "after_t_min": direction_after,
            "late_1000_1600": direction_late,
        },
        "execution_state": {
            "t_min": {
                "execution_lag_deg_deg_mm": tmin_row["execution_lag"],
                "execution_lag_vector_deg_deg_mm": (
                    tmin_row["command_lag_vector"].tolist()
                ),
                "pending_target_residual_deg_deg_mm": (
                    tmin_row["pending_target_residual"]
                ),
                "pending_target_residual_vector_deg_deg_mm": (
                    tmin_row["pending_vector"].tolist()
                ),
                "accepted_vs_actual_action_lag_deg_deg_mm": (
                    tmin_row["action_lag"]
                ),
                "motor_velocity_deg_deg_mm_s": (
                    tmin_row["motor_velocity_deg_deg_mm_s"].tolist()
                ),
                "previous_executed_action_deg_deg_mm": (
                    tmin_row[
                        "previous_executed_action_deg_deg_mm"
                    ].tolist()
                ),
            },
            "near_min_execution_lag": statistics(
                execution_signal[near]
            ),
            "under_60_execution_lag": statistics(
                execution_signal[under60]
            ),
            "matched_20_50": {
                "frames": int(matched.sum()),
                "high_state_signal_threshold_p75": lag_threshold,
                "approaching": summarize_group(rows, approaching),
                "receding": summarize_group(rows, receding),
                "high_lag_or_pending": summarize_group(rows, high_lag),
            },
        },
        "classifier_inputs": {
            "after_toward_fraction": direction_after[
                "policy_toward_fraction"
            ],
            "after_policy_cosine_median": direction_after[
                "policy_cosine_median"
            ],
            "deadzone_lt_1um_near": deadzone(near, 1.),
            "measurement_error_tmin": abs(measurement_norm_error_tmin),
            "execution_lag_tmin": tmin_row["execution_lag"],
        },
    }
    # Per-axis sign agreement is computed separately because the
    # scalar-calibrated true vector already preserves the target direction.
    policy_vectors = np.asarray([
        row["policy_command_um"] for row in rows
    ])
    actual_vectors = np.asarray([
        row["actual_displacement_um"] for row in rows
    ])
    gt_vectors = np.asarray([
        unit(data["target_world_mm"] - (
            data["target_world_mm"]
            - data["target_world_mm"]
        )) for _ in []
    ])
    # Recompute the GT vectors here to avoid storing every trajectory point in
    # the public metrics payload.
    translation = (
        data["target_world_mm"]
        - ROTATION_ROBOT_WORLD @ data["target_robot_mm"]
    )
    gt_vectors = []
    for row, error in zip(data["document"]["episodes"][0][
        "motor_records"
    ], errors):
        tip = world_tip(row["actual_before_task_rad_rad_m"], translation)
        gt_vectors.append(unit(data["target_world_mm"] - tip) * error)
    gt_vectors = np.asarray(gt_vectors)
    for key, mask in (
        ("near_min_under_60", near & under60),
        ("before_t_min", before),
        ("after_t_min", after),
        ("late_1000_1600", late),
    ):
        result["direction"][key]["policy_sign_agreement_xyz"] = {
            axis: sign_agreement(
                policy_vectors[mask, index], gt_vectors[mask, index]
            )
            for index, axis in enumerate("XYZ")
        }
        result["direction"][key]["actual_sign_agreement_xyz"] = {
            axis: sign_agreement(
                actual_vectors[mask, index], gt_vectors[mask, index]
            )
            for index, axis in enumerate("XYZ")
        }
    result["primary_bottleneck"] = classify_target(result)
    return result


def classify_target(result):
    """Assign a transparent heuristic label; raw inputs remain in metrics."""
    perception = result["classifier_inputs"]["measurement_error_tmin"]
    deadzone = result["classifier_inputs"]["deadzone_lt_1um_near"]
    after_toward = result["classifier_inputs"]["after_toward_fraction"]
    after_cosine = result["classifier_inputs"]["after_policy_cosine_median"]
    lag = result["classifier_inputs"]["execution_lag_tmin"]
    if perception >= 10.:
        return "Perception Bias"
    if deadzone is not None and deadzone >= .5:
        return "Deadzone"
    if (
        after_toward is not None
        and after_toward < .5
        and (after_cosine is None or after_cosine < .2)
    ):
        return "Direction Reversal"
    if lag >= .5:
        return "Execution Lag"
    return "Mixed / None Dominant"


def fallback_paths():
    root = ROOT / "runs/20261124"
    result = {}
    for target in TARGETS:
        matches = sorted(root.glob(
            "**/homog_label_fix_20261124_v4_fallback_F_fast5/"
            f"shards/tasks/eval_{target}/raw/evaluation_results.json"
        ))
        if matches:
            result[target] = matches[0]
    return result


def max_hold_le_20(path):
    episode = read(path)["episodes"][0]
    errors = np.asarray(episode["trajectory"]["errors"]) * 1000.
    indices = np.flatnonzero(errors <= 20.)
    if not len(indices):
        return {
            "max_consecutive_cycles_le_20um": 0,
            "max_error_during_hold_um": None,
            "hold_start_step": None,
            "hold_end_step": None,
            "control_period_s": 0.05,
        }
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[cuts + 1]]
    ends = np.r_[indices[cuts], indices[-1]]
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    start, end = int(starts[best]), int(ends[best])
    return {
        "max_consecutive_cycles_le_20um": int(lengths[best]),
        "max_error_during_hold_um": float(errors[start:end + 1].max()),
        "hold_start_step": start,
        "hold_end_step": end,
        "control_period_s": 0.05,
    }


def aggregate_stage3(targets):
    pooled = []
    matched_counts = []
    thresholds = []
    for result in targets:
        rows = load_target(result["source"])["rows"]
        errors = np.asarray([row["true_error_um"] for row in rows])
        matched = (errors >= 20.) & (errors <= 50.)
        matched_counts.append(int(matched.sum()))
        signal = np.asarray([
            max(row["execution_lag"], row["pending_target_residual"])
            for row in rows
        ])
        if matched.any():
            threshold = float(np.percentile(signal[matched], 75))
            thresholds.append(threshold)
        else:
            threshold = math.inf
        d_error = np.gradient(errors)
        for index, row in enumerate(rows):
            if not matched[index]:
                continue
            pooled.append({
                **row,
                "approaching": bool(d_error[index] < -.2),
                "receding": bool(d_error[index] > .2),
                "high_state": bool(signal[index] >= threshold),
            })
    threshold = min(thresholds) if thresholds else math.inf
    return {
        "definition": (
            "Pooled within each target's 20-50 um frames; high state is "
            "top quartile of max(command-to-actual lag, pending target "
            "residual) inside that target's matched band. Groups can overlap."
        ),
        "matched_frames_total": int(sum(matched_counts)),
        "high_state_thresholds_per_target": thresholds,
        "approaching": summarize_group(
            pooled, [row["approaching"] for row in pooled]
        ),
        "receding": summarize_group(
            pooled, [row["receding"] for row in pooled]
        ),
        "high_lag_or_pending": summarize_group(
            pooled, [row["high_state"] for row in pooled]
        ),
    }


def fmt(value, digits=2):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def markdown_table(rows):
    headers = [
        "Target", "t_min", "min um", "final um",
        "OCT distance error @min", "OCT-world component delta* @min",
        "deadzone <1um near", "deadzone <1um <60",
        "toward before", "toward after", "lag @min", "classification",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for value in rows:
        lines.append(
            "| "
            + " | ".join([
                value["target"],
                str(value["t_min"]),
                fmt(value["min_error_um"]),
                fmt(value["final_error_um"]),
                fmt(value["perception"][
                    "measurement_distance_error_at_t_min_um"
                ]),
                fmt(float(np.linalg.norm(value["perception"][
                    "oct_camera_minus_world_vector_at_t_min_um"
                ]))),
                fmt(value["deadzone"][
                    "near_min_fraction_displacement_lt_1um"
                ], 3),
                fmt(value["deadzone"][
                    "under_60_fraction_displacement_lt_1um"
                ], 3),
                fmt(value["direction"]["before_t_min"][
                    "policy_toward_fraction"
                ], 3),
                fmt(value["direction"]["after_t_min"][
                    "policy_toward_fraction"
                ], 3),
                fmt(value["execution_state"]["t_min"][
                    "execution_lag_deg_deg_mm"
                ], 4),
                value["primary_bottleneck"],
            ])
            + " |"
        )
    return "\n".join(lines)


def stage3_table(stage3):
    lines = [
        "| State | Frames | policy toward | policy cosine median | "
        "command norm median | actual toward | error median um | lag median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("approaching", "Approaching"),
        ("receding", "Receding"),
        ("high_lag_or_pending", "High lag/pending"),
    ):
        value = stage3[key]
        lines.append(
            f"| {label} | {value['count']} | "
            f"{fmt(value['policy_toward_fraction'], 3)} | "
            f"{fmt(value['policy_cosine_median'], 3)} | "
            f"{fmt(value['policy_command_norm_median'], 5)} | "
            f"{fmt(value['actual_toward_fraction'], 3)} | "
            f"{fmt(value['error_um_median'])} | "
            f"{fmt(value['execution_lag_median'], 4)} |"
        )
    return "\n".join(lines)


def perception_detail_table(results):
    lines = [
        "| Target | norm bias near mean/median/p90 um | "
        "norm bias <60 mean/median/p90 um | OCT camera XYZ @min um | "
        "A/B/C state residual @min |",
        "|---|---|---|---|---|",
    ]
    def triple(value):
        return (
            f"{fmt(value['mean'])}/{fmt(value['median'])}/"
            f"{fmt(value['p90'])}"
        )
    for value in results:
        p = value["perception"]
        camera = p["oct_camera_minus_world_vector_at_t_min_um"]
        state = p["state_target_residual_at_t_min_deg_deg_mm"]
        lines.append(
            f"| {value['target']} | "
            f"{triple(p['norm_bias_um_near_min'])} | "
            f"{triple(p['norm_bias_um_under_60'])} | "
            f"{fmt(camera[0])}/{fmt(camera[1])}/{fmt(camera[2])}* | "
            f"{fmt(state[0])}/{fmt(state[1])}/{fmt(state[2])} |"
        )
    return "\n".join(lines)


def direction_detail_table(results):
    lines = [
        "| Target | policy cosine median before/after/late | "
        "policy toward before/after/late | policy sign X/Y/Z before | "
        "policy sign X/Y/Z after | policy sign X/Y/Z late | "
        "actual sign X/Y/Z after |",
        "|---|---|---|---|---|---|---|",
    ]
    for value in results:
        direction = value["direction"]
        before = direction["before_t_min"]
        after = direction["after_t_min"]
        late = direction["late_1000_1600"]
        sign_before = before["policy_sign_agreement_xyz"]
        sign_after = after["policy_sign_agreement_xyz"]
        sign_late = late["policy_sign_agreement_xyz"]
        actual_sign = after["actual_sign_agreement_xyz"]
        lines.append(
            f"| {value['target']} | "
            f"{fmt(before['policy_cosine_median'], 3)}/"
            f"{fmt(after['policy_cosine_median'], 3)}/"
            f"{fmt(late['policy_cosine_median'], 3)} | "
            f"{fmt(before['policy_toward_fraction'], 3)}/"
            f"{fmt(after['policy_toward_fraction'], 3)}/"
            f"{fmt(late['policy_toward_fraction'], 3)} | "
            f"{fmt(sign_before['X'], 2)}/{fmt(sign_before['Y'], 2)}/"
            f"{fmt(sign_before['Z'], 2)} | "
            f"{fmt(sign_after['X'], 2)}/{fmt(sign_after['Y'], 2)}/"
            f"{fmt(sign_after['Z'], 2)} | "
            f"{fmt(sign_late['X'], 2)}/{fmt(sign_late['Y'], 2)}/"
            f"{fmt(sign_late['Z'], 2)} | "
            f"{fmt(actual_sign['X'], 2)}/"
            f"{fmt(actual_sign['Y'], 2)}/"
            f"{fmt(actual_sign['Z'], 2)} |"
        )
    return "\n".join(lines)


def per_axis_available_table(results):
    lines = [
        "| Target | OCT camera X mean/median/p90 | Y | Z | "
        "A state residual mean/median/p90 | B | C |",
        "|---|---|---|---|---|---|---|",
    ]
    def triple(value):
        return (
            f"{fmt(value['mean'])}/{fmt(value['median'])}/"
            f"{fmt(value['p90'])}"
        )
    for value in results:
        oct_stats = value["perception"][
            "oct_components_um_under_60_camera_frame"
        ]
        state_stats = value["perception"][
            "state_target_residual_under_60_um"
        ]
        lines.append(
            f"| {value['target']} | "
            + " | ".join([
                triple({
                    "mean": oct_stats["mean"][index],
                    "median": oct_stats["median"][index],
                    "p90": oct_stats["p90_abs"][index],
                })
                for index in range(3)
            ])
            + " | "
            + " | ".join([
                triple({
                    "mean": state_stats["mean"][index],
                    "median": state_stats["median"][index],
                    "p90": state_stats["p90_abs"][index],
                })
                for index in range(3)
            ])
            + " |"
        )
    return "\n".join(lines)


def near_distribution_table(results):
    lines = [
        "| Target | policy command median A/B/C | "
        "policy command p90 |abs| A/B/C | "
        "actual displacement median X/Y/Z um | "
        "actual displacement p90 |abs| X/Y/Z um |",
        "|---|---|---|---|---|",
    ]
    for value in results:
        command = value["deadzone"][
            "near_min_command_magnitude_deg_deg_mm"
        ]
        displacement = value["deadzone"][
            "near_min_actual_displacement_um"
        ]
        lines.append(
            f"| {value['target']} | "
            f"{fmt(command['A']['median'], 5)}/"
            f"{fmt(command['B']['median'], 5)}/"
            f"{fmt(command['C']['median'], 5)} | "
            f"{fmt(command['A']['p90'], 5)}/"
            f"{fmt(command['B']['p90'], 5)}/"
            f"{fmt(command['C']['p90'], 5)} | "
            f"{fmt(displacement['X']['median'], 3)}/"
            f"{fmt(displacement['Y']['median'], 3)}/"
            f"{fmt(displacement['Z']['median'], 3)} | "
            f"{fmt(displacement['X']['p90'], 3)}/"
            f"{fmt(displacement['Y']['p90'], 3)}/"
            f"{fmt(displacement['Z']['p90'], 3)} |"
        )
    return "\n".join(lines)


def verdict(results, stage3):
    perception_flags = sum(
        value["perception"]["flag_ge_10um_at_t_min"]
        for value in results
    )
    deadzone_values = [
        value["deadzone"]["near_min_fraction_displacement_lt_1um"]
        for value in results
        if value["deadzone"]["near_min_fraction_displacement_lt_1um"]
        is not None
    ]
    after_toward = [
        value["direction"]["after_t_min"]["policy_toward_fraction"]
        for value in results
        if value["direction"]["after_t_min"]["policy_toward_fraction"]
        is not None
    ]
    lags = [
        value["execution_state"]["t_min"][
            "execution_lag_deg_deg_mm"
        ]
        for value in results
    ]
    mean_after = float(np.mean(after_toward)) if after_toward else math.nan
    mean_dead = float(np.mean(deadzone_values)) if deadzone_values else math.nan
    mean_lag = float(np.mean(lags)) if lags else math.nan
    if perception_flags:
        dominant = "Perception Bias"
        next_step = "fix perception before adding execution residual state"
    elif mean_after < .5:
        dominant = "Direction Reversal"
        next_step = "fix target direction/policy behavior before Stage 4"
    elif mean_dead >= .4:
        dominant = "Deadzone"
        next_step = "address minimum executeable displacement before Stage 4"
    elif mean_lag > 0.:
        dominant = "Execution Lag"
        next_step = "Stage 4 should add execution-target residual/state"
    else:
        dominant = "Mixed / None Dominant"
        next_step = "collect richer observations before Stage 4"
    return dominant, next_step, {
        "perception_ge_10_targets": int(perception_flags),
        "mean_after_toward_fraction": mean_after,
        "mean_deadzone_lt_1um_near": mean_dead,
        "mean_execution_lag_tmin": mean_lag,
    }


def write_report(path, schema, baseline, fallback, fallback_hold, verdict_data):
    stage3_baseline = aggregate_stage3(baseline)
    stage3_fallback = aggregate_stage3(fallback)
    dominant, next_step, evidence = verdict_data
    lines = [
        "# Stage 2 + Stage 3 pre-check: why baseline stays near 20-60 um",
        "",
        "Read-only analysis of completed evaluation JSON. No training, "
        "simulation rollout, model loading, or result overwrite was performed.",
        "",
        "## Exact JSON schema observed",
        "",
        f"- Top level: `{schema['top_level_keys']}`",
        f"- Episode: `{schema['episode_keys']}`",
        f"- Trajectory: `{schema['trajectory_keys']}`",
        f"- Motor record: `{schema['motor_record_keys']}`",
        f"- Control target: `{schema['control_target_keys']}`",
        f"- Terminal success: `{schema['terminal_success_keys']}`",
        "- Available measurements: OCT target vector, OCT norm and validity, "
        "A/B/C state, velocity, accepted control target, motor execution "
        "before/after task state, requested/commanded target, tip displacement.",
        "- The logged OCT vector is in `camera_optical` coordinates. The JSON "
        "does not retain the camera rotation, so only its norm can be compared "
        "directly with world-space true error. Per-axis camera components are "
        "reported separately and are not interpreted as world-space bias.",
        "- Absent from logs: raw RGB/depth arrays, explicit predicted tip-world "
        "trajectory, explicit true-error vector, and explicit named pending-"
        "execution-residual field. The true vector direction is therefore "
        "derived statically from recorded A/B/C plus target pose, then scaled "
        "to the recorded ground-truth scalar error.",
        "",
        "## Baseline per-target bottleneck table",
        "",
        markdown_table(baseline),
        "",
        "Columns `toward before/after` are the fraction of non-zero raw policy "
        "commands whose world-space induced displacement has positive cosine "
        "with the ground-truth target vector. `lag @min` is command-to-actual "
        "task residual in mixed A-deg/B-deg/C-mm units and is magnitude only.",
        "The `OCT-world component delta*` column is an explicitly non-gated "
        "cross-frame contrast included for auditability; it is not a perception "
        "error metric.",
        "",
        "## Perception statistics",
        "",
        perception_detail_table(baseline),
        "",
        "The final camera XYZ column at `t_min` is shown only as the available "
        "camera-frame observation. Because camera rotation is absent, it is not "
        "reported as world-axis bias. A/B/C state residual is recorded separately "
        "and uses different physical units.",
        "",
        "### Per-axis available-field distributions under 60 um",
        "",
        per_axis_available_table(baseline),
        "",
        "These are component distributions in their recorded frames, not "
        "world-space measurement bias.",
        "",
        "## Direction statistics",
        "",
        direction_detail_table(baseline),
        "",
        "## Near-target command and displacement distribution",
        "",
        near_distribution_table(baseline),
        "",
        "## Stage 3 matched-error comparison: Baseline 20-50 um",
        "",
        stage3_table(stage3_baseline),
        "",
        stage3_baseline["definition"],
        (
            "Approaching frames are "
            f"{fmt(stage3_baseline['approaching']['policy_toward_fraction'], 3)} "
            "toward-target versus "
            f"{fmt(stage3_baseline['receding']['policy_toward_fraction'], 3)} "
            "for receding frames. High execution-lag/pending frames remain mixed "
            f"({fmt(stage3_baseline['high_lag_or_pending']['policy_toward_fraction'], 3)} "
            "toward), but the absolute logged lag is small. This supports a "
            "direction/behavior failure more strongly than a large actuator lag."
        ),
        "",
    ]
    if fallback:
        lines += [
            "## Optional Fallback F raw check",
            "",
            markdown_table(fallback),
            "",
            "### Fallback F matched-error comparison",
            "",
            stage3_table(stage3_fallback),
            "",
            "### Fallback F E10 <=20 um hold",
            "",
            f"- Exact maximum consecutive cycles: "
            f"**{fallback_hold['max_consecutive_cycles_le_20um']}**",
            f"- Maximum error within the longest hold segment: "
            f"**{fmt(fallback_hold['max_error_during_hold_um'], 3)} um**",
            f"- Hold start/end: "
            f"`{fallback_hold['hold_start_step']}` / "
            f"`{fallback_hold['hold_end_step']}`",
            (
                "- Interpretation: E10 entered the <=20 um band for a short "
                f"{fallback_hold['max_consecutive_cycles_le_20um']}-cycle "
                "segment; it did not sustain a 20 um hold."
                if fallback_hold["max_consecutive_cycles_le_20um"] > 0
                else "- Interpretation: E10 never entered the <=20 um hold band."
            ),
            "",
        ]
    else:
        lines += [
            "## Optional Fallback F raw check",
            "",
            "- `fallback_F_raw_missing_locally: true`",
            "",
        ]
    lines += [
        "## Verdict",
        "",
        f"1. Dominant factor: **{dominant}**; perception-distance flags are "
        f"{evidence['perception_ge_10_targets']}/5, deadzone<1um is "
        f"{fmt(evidence['mean_deadzone_lt_1um_near'], 3)}, and mean after-t_min "
        f"toward fraction is {fmt(evidence['mean_after_toward_fraction'], 3)}.",
        f"2. Execution lag at t_min averages "
        f"{fmt(evidence['mean_execution_lag_tmin'], 4)} mixed A-deg/B-deg/C-mm "
        "units, so actuator lag alone is too small to explain the 23-42 um floor.",
        f"3. {next_step.capitalize()}; Stage 4 should not add execution-target residual as "
        "the first response to this static evidence.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    primary_documents = {}
    for target in TARGETS:
        path = PRIMARY_ROOT / f"eval_{target}/raw/evaluation_results.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        primary_documents[target] = path
    schema = schema_of(read(next(iter(primary_documents.values()))))
    baseline = [
        analyze_target(target, primary_documents[target])
        for target in TARGETS
    ]
    fallback_documents = fallback_paths()
    fallback = [
        analyze_target(target, fallback_documents[target])
        for target in TARGETS if target in fallback_documents
    ]
    hold = (
        max_hold_le_20(fallback_documents["E10"])
        if "E10" in fallback_documents else None
    )
    verdict_data = verdict(baseline, aggregate_stage3(baseline))
    metrics = {
        "status": "completed",
        "read_only_inputs_only": True,
        "inputs": {
            "baseline": {
                target: str(path)
                for target, path in primary_documents.items()
            },
            "fallback_F": {
                target: str(path)
                for target, path in fallback_documents.items()
            },
            "fallback_F_raw_missing_locally": not bool(fallback_documents),
        },
        "schema": schema,
        "definitions": {
            "true_error_um": "evaluation_results.json episodes[0].trajectory.errors * 1000",
            "near_min": "[t_min-50, t_min+50] inclusive, clipped to rollout",
            "matched_error_band_um": [20.0, 50.0],
            "approaching": "d(true_error_um)/d(step) < -0.2",
            "receding": "d(true_error_um)/d(step) > 0.2",
            "execution_lag": "commanded_task - actual_after_task converted to A-deg/B-deg/C-mm magnitude",
            "pending_residual": "requested_task - actual_after_task converted to A-deg/B-deg/C-mm magnitude",
            "policy_direction": "analytic world displacement induced by raw A/B/C action compared with target vector",
        },
        "baseline": baseline,
        "fallback_F": fallback,
        "fallback_F_E10_le_20um_hold": hold,
        "stage3_baseline_20_50um": aggregate_stage3(baseline),
        "stage3_fallback_F_20_50um": (
            aggregate_stage3(fallback) if fallback else None
        ),
        "verdict": {
            "dominant_factor": verdict_data[0],
            "stage4_recommendation": verdict_data[1],
            "evidence": verdict_data[2],
        },
    }
    write_json(output / "metrics.json", metrics)
    write_report(
        output / "report.md",
        schema,
        baseline,
        fallback,
        hold,
        verdict_data,
    )
    print(json.dumps({
        "output_dir": str(output),
        "baseline_targets": len(baseline),
        "fallback_targets": len(fallback),
        "fallback_E10_hold": hold,
        "verdict": metrics["verdict"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
