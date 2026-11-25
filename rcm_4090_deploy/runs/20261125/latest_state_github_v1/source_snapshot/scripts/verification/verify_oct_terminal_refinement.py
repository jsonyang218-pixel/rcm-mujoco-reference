"""Verify OCT terminal target estimation and integer endpoint search."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lerobot_training"))

from rcm.reference import Reference
from oct_terminal_refinement import solve_oct_terminal_action
from train_oct_guided_regression import CAMERA_SIGN, TASK_TO_SI


def main():
    reference = Reference()
    _, camera_rotation = reference.camera_pose()
    current_task = np.array([0.02, -0.01, 0.019], dtype=np.float64)
    current_counts = np.rint(
        np.asarray(reference.encoders(current_task), dtype=np.float64)
    ).astype(np.int64)
    target_task = reference.task_from_encoders(
        current_counts + np.array([2, -1, 3], dtype=np.int64),
        reference_task=current_task,
    )
    target_world_mm = reference.robot_to_world(
        reference.forward_kinematics(target_task)
    )
    current_world_mm = reference.robot_to_world(
        reference.forward_kinematics(current_task)
    )
    world_error = target_world_mm - current_world_mm
    camera_error = (
        CAMERA_SIGN * (camera_rotation.T @ world_error)
    )
    solution = solve_oct_terminal_action(
        reference,
        current_task / TASK_TO_SI,
        camera_error,
        camera_rotation,
        neighbor_counts=2,
    )
    assert solution["predicted_error_um"] <= 0.1
    assert np.asarray(solution["action_deg_deg_mm"]).shape == (3,)
    assert np.all(np.isfinite(solution["action_deg_deg_mm"]))
    print(json.dumps({
        "result": "PASS",
        "predicted_error_um": solution["predicted_error_um"],
        "searched_candidates": solution["searched_candidates"],
        "action_deg_deg_mm": solution["action_deg_deg_mm"],
        "used_target_ground_truth_for_control": solution[
            "used_target_ground_truth_for_control"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
