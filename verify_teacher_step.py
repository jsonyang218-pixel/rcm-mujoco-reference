"""Compare one-shot teacher labels with the first real teacher command."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rcm_scene_v9 import RCMSceneV9
from rcm_fundus import make_random_fundus_targets
from rcm_visual_servo import VisualServoController, visual_servo_settings
from run_waypoint_comparison import guard_scene_goal_inverse


UNITS = np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0])


def target_for_seed(config, seed):
    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    try:
        rng = np.random.default_rng(seed)
        return np.asarray(
            make_random_fundus_targets(scene, rng, 1)["targets"][0]["point_mm"]
        )
    finally:
        scene.close()


def main():
    result = json.loads(
        Path(
            "outputs/oct_trace_diag_20261105_g1e0/evaluation_results.json"
        ).read_text()
    )
    state = np.asarray(
        result["episodes"][0]["trajectory"]["states"][819],
        dtype=np.float64,
    )
    config = json.loads(Path("config.json").read_text())
    target = target_for_seed(config, 20271105)

    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    try:
        scene.set_task(state / UNITS)
        scene.set_target_world(scene.robot_to_world(target))
        settings = visual_servo_settings(scene)
        settings.update(
            perception_backend="strict_rgbd",
            position_feedback="geometry",
            occlusion_handling="stop",
            save_diagnostics=False,
            max_iterations=2,
        )
        guard_scene_goal_inverse(scene)
        controller = VisualServoController(scene, settings=settings)
        controller.sensor.observation_time_s = 0.0
        observation = controller.sensor.observe()
        label = controller.geometry_action_from_observation(
            scene.current_task(),
            observation,
            previous_velocity=np.zeros(3),
            command_q=scene.current_task(),
        )
        controller.sensor.close()
    finally:
        scene.close()

    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    try:
        scene.set_task(np.zeros(3))
        scene.set_target_world(scene.robot_to_world(target))
        scene.set_task(state / UNITS)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.0)
        settings = visual_servo_settings(scene)
        settings.update(
            perception_backend="strict_rgbd",
            position_feedback="geometry",
            occlusion_handling="stop",
            save_diagnostics=False,
            max_iterations=2,
        )
        guard_scene_goal_inverse(scene)
        controller = VisualServoController(scene, settings=settings)
        report = controller.run()
        row = controller.rows[0]
    finally:
        scene.close()

    if not label["valid"] or "command_velocity_A_deg_B_deg_C_mm_per_s" not in row:
        raise AssertionError(
            json.dumps(
                {
                    "label": label,
                    "report": report,
                    "row": row,
                },
                indent=2,
                default=str,
            )
        )
    delta = (
        np.asarray(row["command_velocity_A_deg_B_deg_C_mm_per_s"])
        * settings["sample_period_s"]
    )
    np.testing.assert_allclose(
        label["action_deg_deg_mm"],
        delta,
        atol=1e-10,
        rtol=0,
    )
    print(
        json.dumps(
            {
                "result": "PASS",
                "teacher_label_deg_deg_mm": label["action_deg_deg_mm"],
                "teacher_first_command_deg_deg_mm": delta.tolist(),
                "depth_valid": label["depth_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
