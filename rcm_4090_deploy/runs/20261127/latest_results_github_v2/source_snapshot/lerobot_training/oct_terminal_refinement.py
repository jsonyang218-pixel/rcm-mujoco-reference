"""Estimate a target from OCT and select a quantized reachable tip endpoint."""
from __future__ import annotations

from itertools import product

import numpy as np

from train_oct_guided_regression import (
    CAMERA_SIGN,
    camera_error_to_world,
)


TASK_TO_ABC = np.array(
    [180.0 / np.pi, 180.0 / np.pi, 1000.0],
    dtype=np.float64,
)


def _candidate_solution(
    reference,
    target_world_mm,
    current_task,
    encoder_counts,
):
    candidate_task = reference.task_from_encoders(
        encoder_counts,
        reference_task=current_task,
    )
    candidate_tip_robot_mm = reference.forward_kinematics(
        candidate_task
    )
    candidate_tip_world_mm = reference.robot_to_world(
        candidate_tip_robot_mm
    )
    predicted_error_um = float(
        np.linalg.norm(
            candidate_tip_world_mm - target_world_mm
        )
        * 1000.0
    )
    return {
        "encoder_counts": tuple(int(value) for value in encoder_counts),
        "candidate_task_rad_rad_m": candidate_task.tolist(),
        "candidate_state_deg_deg_mm": (
            candidate_task * TASK_TO_ABC
        ).tolist(),
        "candidate_tip_world_mm": candidate_tip_world_mm.tolist(),
        "predicted_error_um": predicted_error_um,
    }


def solve_oct_terminal_action(
    reference,
    state_deg_deg_mm,
    oct_error_camera_mm,
    camera_rotation,
    *,
    neighbor_counts=1,
    target_world_override_mm=None,
):
    """Return the best integer A/B/C target from current calibrated state+OCT."""
    state_abc = np.asarray(state_deg_deg_mm, dtype=np.float64)
    camera_error = np.asarray(oct_error_camera_mm, dtype=np.float64)
    if state_abc.shape != (3,) or camera_error.shape != (3,):
        raise ValueError("state and OCT error must each contain 3 values")
    if not np.all(np.isfinite(state_abc)) or not np.all(
        np.isfinite(camera_error)
    ):
        raise ValueError("state and OCT error must be finite")
    if int(neighbor_counts) < 0:
        raise ValueError("neighbor_counts must be nonnegative")

    current_task = reference.validate_task(
        state_abc / TASK_TO_ABC
    )
    current_tip_world_mm = reference.robot_to_world(
        reference.forward_kinematics(current_task)
    )
    if target_world_override_mm is None:
        world_error_mm = camera_error_to_world(
            camera_error,
            camera_rotation,
        )
        target_world_mm = current_tip_world_mm + world_error_mm
    else:
        target_world_mm = np.asarray(
            target_world_override_mm,
            dtype=np.float64,
        )
        if target_world_mm.shape != (3,) or not np.all(
            np.isfinite(target_world_mm)
        ):
            raise ValueError(
                "target_world_override_mm must contain three finite values"
            )
        world_error_mm = target_world_mm - current_tip_world_mm
    target_robot_mm = reference.world_to_robot(target_world_mm)
    target_joint_state = reference.nominal.inverse(
        target_robot_mm,
        enforce_limits=True,
        reference_state=reference.state_from_task(current_task),
    )
    target_task_continuous = reference.task_from_state(
        target_joint_state
    )
    requested_counts = np.rint(
        np.asarray(
            reference.encoders(target_task_continuous),
            dtype=np.float64,
        )
    ).astype(np.int64)
    if requested_counts.shape != (3,):
        raise ValueError("terminal inverse did not return three counts")

    candidates = []
    offsets = range(-int(neighbor_counts), int(neighbor_counts) + 1)
    for delta in product(offsets, repeat=3):
        counts = requested_counts + np.asarray(delta, dtype=np.int64)
        try:
            candidates.append(
                _candidate_solution(
                    reference,
                    target_world_mm,
                    current_task,
                    counts,
                )
            )
        except ValueError:
            continue
    if not candidates:
        raise ValueError("no valid neighboring integer endpoint")
    best = min(
        candidates,
        key=lambda value: value["predicted_error_um"],
    )
    best_abc = np.asarray(
        best["candidate_state_deg_deg_mm"],
        dtype=np.float64,
    )
    return {
        **best,
        "estimated_target_world_mm": target_world_mm.tolist(),
        "current_tip_world_mm": current_tip_world_mm.tolist(),
        "action_deg_deg_mm": (best_abc - state_abc).tolist(),
        "requested_integer_encoder_counts": requested_counts.tolist(),
        "searched_candidates": len(candidates),
        "neighbor_counts": int(neighbor_counts),
        "source": "calibrated forward kinematics + OCT target estimate",
        "used_target_ground_truth_for_control": False,
    }
