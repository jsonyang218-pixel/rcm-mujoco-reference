"""Verify that the geometric teacher can recover from failed-policy states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rcm_reference import HERE
from rcm_scene_v9 import RCMSceneV9
from rcm_fundus import make_random_fundus_targets
from rcm_visual_servo import VisualServoController, visual_servo_settings
from run_waypoint_comparison import guard_scene_goal_inverse


UNITS = np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0])


def divergence_indices(result):
    trajectory = result["episodes"][0]["trajectory"]
    errors = np.asarray(trajectory["errors"], dtype=np.float64)
    minimum_step = int(np.argmin(errors))
    minimum_error = float(errors[minimum_step])
    divergence_step = None
    for index in range(minimum_step + 1, len(errors)):
        end = min(len(errors), index + 10)
        if (
            errors[index] >= minimum_error + 0.5
            and np.all(errors[index:end] >= minimum_error + 0.5)
        ):
            divergence_step = index
            break
    if divergence_step is None:
        divergence_step = minimum_step
    later_step = min(len(errors) - 1, divergence_step + 100)
    return minimum_step, minimum_error, divergence_step, later_step


def run_teacher(config, target_robot_mm, state_abc_mm, max_iterations, output_dir):
    scene = RCMSceneV9(
        {
            **config,
            "motor_execution": {"enabled": False},
        }
    )
    try:
        scene.set_task(np.zeros(3))
        target_robot_mm = np.asarray(target_robot_mm, dtype=np.float64)
        scene.set_target_world(scene.robot_to_world(target_robot_mm))
        scene.set_task(np.asarray(state_abc_mm, dtype=np.float64) / UNITS)
        scene.enable_motor_execution(config.get("motor_execution"))
        scene.reset_motor_time(0.0)
        settings = visual_servo_settings(scene)
        settings.update(
            max_iterations=max_iterations,
            save_diagnostics=False,
            position_feedback="geometry",
            occlusion_handling="stop",
            perception_backend="strict_rgbd",
        )
        guard_scene_goal_inverse(scene)
        controller = VisualServoController(
            scene,
            settings=settings,
            output_dir=output_dir,
        )
        try:
            report = controller.run()
        finally:
            controller.sensor.close()
        return {
            "result": report.get("result"),
            "reason": report.get("reason"),
            "final_error_um": report.get("final_tip_target_error_um"),
            "terminal_stop_validated": report.get("terminal_stop_validated"),
            "controller_iterations": report.get("iterations"),
            "final_task_A_deg_B_deg_C_mm": (
                scene.current_task() * UNITS
            ).tolist(),
        }
    finally:
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-result", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--max-iterations", type=int, default=1600)
    args = parser.parse_args()
    result = json.loads(args.episode_result.read_text())
    episode = result["episodes"][0]
    trajectory = episode["trajectory"]
    states = np.asarray(trajectory["states"], dtype=np.float64)
    errors = np.asarray(trajectory["errors"], dtype=np.float64)
    minimum_step, minimum_error, divergence_step, later_step = (
        divergence_indices(result)
    )

    scene = RCMSceneV9()
    try:
        rng = np.random.default_rng(args.seed)
        target = make_random_fundus_targets(scene, rng, 1)["targets"][0][
            "point_mm"
        ]
    finally:
        scene.close()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    scenarios = []
    for name, step in (
        ("divergence_onset", divergence_step),
        ("later_divergence", later_step),
    ):
        scenario_dir = output_dir / name
        scenario_dir.mkdir()
        teacher = run_teacher(
            config,
            target,
            states[step],
            args.max_iterations,
            scenario_dir,
        )
        scenarios.append(
            {
                "name": name,
                "source_step": step,
                "start_error_mm": float(errors[step]),
                "start_state_A_deg_B_deg_C_mm": states[step].tolist(),
                "teacher": teacher,
                "recovered_under_100um": bool(
                    teacher["final_error_um"] is not None
                    and teacher["final_error_um"] < 100.0
                ),
            }
        )
    summary = {
        "result": "PASS"
        if all(row["recovered_under_100um"] for row in scenarios)
        else "FAIL",
        "source_episode": str(args.episode_result),
        "seed": args.seed,
        "target_robot_mm": np.asarray(target).tolist(),
        "minimum_error_mm": minimum_error,
        "minimum_error_step": minimum_step,
        "divergence_step": divergence_step,
        "scenarios": scenarios,
    }
    (output_dir / "teacher_takeover_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
