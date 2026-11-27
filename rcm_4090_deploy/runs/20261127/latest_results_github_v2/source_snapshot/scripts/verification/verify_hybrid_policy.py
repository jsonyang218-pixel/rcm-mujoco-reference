"""Verify coarse Flow to fine OCT-residual policy switching without rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lerobot_training"))

from eval_policy import PolicyEvaluator
from train_oct_guided_regression import CAMERA_SIGN, SI_TO_ACTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-model", type=Path, required=True)
    parser.add_argument("--fine-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    evaluator = PolicyEvaluator(
        str(args.coarse_model),
        device=args.device,
        num_action_samples=1,
        n_action_execute_steps=1,
        fine_model_path=str(args.fine_model),
        fine_switch_error_mm=0.5,
    )
    evaluator.reset_buffer(seed=20261116)
    rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(3, dtype=np.float32)
    coarse_oct = np.array([0.8, 0.2, 0.5, 1.0], dtype=np.float32)
    coarse_action = evaluator.get_action(rgb, state, coarse_oct)
    assert evaluator.control_mode_counts["coarse"] == 1
    evaluator.record_applied_action(coarse_action)

    fine_oct = np.array([0.1, 0.0, 0.2, 1.0], dtype=np.float32)
    fine_action = evaluator.get_action(rgb, state, fine_oct)
    assert evaluator.fine_active
    assert evaluator.control_mode_counts["fine"] == 1
    evaluator.record_applied_action(fine_action)

    reference = evaluator.fine_evaluator.reference
    _, camera_rotation = reference.camera_pose()
    current_task = state / SI_TO_ACTION
    current_counts = np.rint(
        np.asarray(reference.encoders(current_task), dtype=np.float64)
    ).astype(np.int64)
    terminal_task = reference.task_from_encoders(
        current_counts + np.array([2, -1, 3], dtype=np.int64),
        reference_task=current_task,
    )
    current_world = reference.robot_to_world(
        reference.forward_kinematics(current_task)
    )
    terminal_world = reference.robot_to_world(
        reference.forward_kinematics(terminal_task)
    )
    terminal_camera_error = (
        CAMERA_SIGN * (camera_rotation.T @ (terminal_world - current_world))
    )
    terminal_oct = np.r_[terminal_camera_error, 1.0].astype(np.float32)
    terminal_actions = []
    for _ in range(evaluator.terminal_confirm_frames):
        terminal_action = evaluator.get_action(
            rgb,
            state,
            terminal_oct,
        )
        terminal_actions.append(terminal_action)
        evaluator.record_applied_action(terminal_action)
    assert evaluator.terminal_refinement_active
    assert evaluator.control_mode_counts["terminal"] == 1
    assert evaluator.control_mode_counts["stabilize"] == (
        evaluator.terminal_confirm_frames - 1
    )
    assert (
        evaluator.last_terminal_solution["predicted_error_um"] <= 0.1
    )
    for name, action in (
        ("coarse", coarse_action),
        ("fine", fine_action),
        ("terminal", terminal_action),
    ):
        assert np.asarray(action).shape == (3,)
        assert np.all(np.isfinite(action))
    print(json.dumps({
        "result": "PASS",
        "coarse_action": np.asarray(coarse_action).tolist(),
        "fine_action": np.asarray(fine_action).tolist(),
        "terminal_action": np.asarray(terminal_action).tolist(),
        "terminal_confirm_frames": evaluator.terminal_confirm_frames,
        "terminal_predicted_error_um": evaluator.last_terminal_solution[
            "predicted_error_um"
        ],
        "fine_switch_error_mm": evaluator.fine_switch_error_mm,
        "fine_policy_type": evaluator.fine_evaluator.policy_type,
        "control_mode_counts": evaluator.control_mode_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
