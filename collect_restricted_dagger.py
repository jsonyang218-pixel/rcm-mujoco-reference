"""Collect near-target DAgger data for the restricted student policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from evaluate_restricted_policy import (
    DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
    TASK_TO_DEG_DEG_MM,
    LiveRestrictedPolicy,
    state_abc_mm_to_task,
    task_to_state_abc_mm,
)
from restricted_perception.config import load_config
from restricted_perception.image_ops import degrade_image
from restricted_perception.online import RestrictedPerceptionAdapter
from restricted_perception.sidecar import _kinematic_tips
from rcm_reference import Reference
from rcm_scene_v9 import RCMSceneV9
from rcm_visual_servo import VisualServoController, visual_servo_settings


def collect_episode(
    *,
    policy: LiveRestrictedPolicy,
    reference: Reference,
    perception_config,
    target: dict,
    start: dict,
    output_dir: Path,
    max_steps: int,
    beta: float,
    takeover_steps: int,
    image_resolution: int,
    blur_radius: float,
    occlusion_fraction: float,
) -> dict:
    scene = RCMSceneV9()
    teacher = None
    try:
        target_robot = np.asarray(target["point_mm"], dtype=float)
        target_world = scene.robot_to_world(target_robot)
        scene.set_target_world(target_world)
        target_task = scene.target_task(validate=True) * TASK_TO_DEG_DEG_MM
        start_task = target_task.copy()
        start_task[2] -= float(start["initial_error_um"]) / 1000.0
        scene.set_task(state_abc_mm_to_task(start_task))
        scene.enable_motor_execution()
        scene.reset_motor_time(0.0)
        settings = visual_servo_settings(scene)
        settings.update(
            save_diagnostics=False,
            position_feedback="geometry",
            occlusion_handling="stop",
            perception_backend="strict_rgbd",
        )
        teacher = VisualServoController(scene, settings=settings)
        adapter = RestrictedPerceptionAdapter(
            perception_config,
            target_robot_mm=target_robot,
            seed=int(start["seed"]),
            episode_id=start["start_id"],
        )
        images = []
        states = []
        velocities = []
        restricted_features = []
        student_actions = []
        teacher_actions = []
        executed_actions = []
        teacher_valid_flags = []
        errors_um = []
        fallback_codes = []
        takeover_left = 0
        worsening_streak = 0
        best_error = float("inf")
        success_streak = 0
        success = False
        stop_reason = None
        previous_velocity = np.zeros(3, dtype=float)
        rng = np.random.default_rng(int(start["seed"]) + 17)
        for step in range(max_steps):
            teacher.sensor.observation_time_s = scene.motor_execution.time_s
            observation = teacher.sensor.observe()
            state_abc = task_to_state_abc_mm(scene.current_task())
            velocity_si = np.asarray(scene.motor_execution.actual_velocity, dtype=float)
            velocity = velocity_si * TASK_TO_DEG_DEG_MM
            tip_robot = _kinematic_tips(state_abc[None, :], reference)[0]
            navigation = observation.get("oct_navigation_depth")
            delta_xyz_m = None if navigation is None else navigation.get("delta_xyz_m")
            fine_valid = bool(
                delta_xyz_m is not None
                and len(delta_xyz_m) == 3
                and np.all(np.isfinite(delta_xyz_m))
            )
            fine_camera = (
                np.asarray(delta_xyz_m, dtype=float) * 1000.0
                if fine_valid
                else None
            )
            restricted = adapter.observe(
                state=state_abc,
                kinematic_tip_robot_mm=tip_robot,
                fine_goal_vector_camera_mm=fine_camera,
                fine_valid=fine_valid,
                timestamp_s=scene.motor_execution.time_s,
            )
            restricted_vector = restricted.policy_vector()
            student_action = policy.action(observation["rgb"], state_abc, restricted_vector)
            student_action = np.clip(
                np.asarray(student_action, dtype=float),
                -DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
                DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            )
            teacher_label = teacher.geometry_action_from_observation(
                scene.current_task(),
                observation,
                previous_velocity=previous_velocity,
                command_q=scene.current_task(),
            )
            teacher_valid = bool(teacher_label["valid"])
            teacher_action = (
                np.asarray(teacher_label["action_deg_deg_mm"], dtype=float)
                if teacher_valid
                else np.zeros(3, dtype=float)
            )
            teacher_action = np.clip(
                teacher_action,
                -DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
                DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            )
            error_um = float(scene.metrics()["tip_target_error_um"])
            settled = bool(
                np.all(np.abs(velocity) <= np.array([0.05, 0.05, 0.01]))
            )
            if error_um < best_error - 1e-6:
                best_error = error_um
                worsening_streak = 0
            elif error_um > best_error + 100.0:
                worsening_streak += 1
            else:
                worsening_streak = 0
            if takeover_left == 0 and (worsening_streak >= 2 or not teacher_valid):
                takeover_left = int(takeover_steps)
            if takeover_left > 0:
                takeover_left -= 1
                executed_action = teacher_action
                executor = "teacher" if teacher_valid else "hold"
            elif teacher_valid and rng.random() < beta:
                executed_action = teacher_action
                executor = "beta_teacher"
            else:
                executed_action = student_action
                executor = "policy"
            if error_um <= 20.0:
                success_streak += 1
            else:
                success_streak = 0
            if success_streak > 0:
                executed_action = np.zeros(3, dtype=float)
                executor = "precision_hold"
            image = degrade_image(
                Image.fromarray(observation["rgb"], mode="RGB"),
                resolution=image_resolution,
                gaussian_blur_radius=blur_radius,
                center_occlusion_fraction=occlusion_fraction,
            )
            images.append(np.asarray(image, dtype=np.uint8))
            states.append(state_abc)
            velocities.append(velocity)
            restricted_features.append(restricted_vector)
            student_actions.append(student_action)
            teacher_actions.append(teacher_action)
            executed_actions.append(executed_action)
            teacher_valid_flags.append(teacher_valid)
            errors_um.append(error_um)
            fallback_codes.append(restricted.fallback_state.value)
            if error_um > 1000.0:
                stop_reason = "ERROR_EXCEEDED_1MM"
                break
            if success_streak >= 3:
                success = True
                stop_reason = "POSITION_ERROR_AND_HOLD"
                break
            previous_velocity = velocity_si
            try:
                scene.set_task(
                    state_abc_mm_to_task(state_abc + executed_action),
                    motor_dt=0.05,
                )
            except ValueError as error:
                stop_reason = f"SAFETY_LIMIT:{error}"
                break
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_path = output_dir / f"{start['start_id']}.npz"
        np.savez_compressed(
            episode_path,
            image=np.stack(images),
            state=np.asarray(states, dtype=np.float32),
            velocity=np.asarray(velocities, dtype=np.float32),
            restricted=np.asarray(restricted_features, dtype=np.float32),
            action_student=np.asarray(student_actions, dtype=np.float32),
            action_teacher=np.asarray(teacher_actions, dtype=np.float32),
            action_executed=np.asarray(executed_actions, dtype=np.float32),
            teacher_valid=np.asarray(teacher_valid_flags, dtype=bool),
            error_um=np.asarray(errors_um, dtype=np.float32),
            fallback_state=np.asarray(fallback_codes),
        )
        return {
            "start_id": start["start_id"],
            "target_id": target["id"],
            "target_robot_mm": target_robot.tolist(),
            "seed": int(start["seed"]),
            "initial_error_um": float(start["initial_error_um"]),
            "success": bool(success),
            "steps": len(errors_um),
            "final_error_um": float(errors_um[-1]),
            "teacher_valid_frames": int(np.sum(teacher_valid_flags)),
            "stop_reason": stop_reason,
            "episode_file": episode_path.name,
            "collect_wall_s": time.monotonic(),
        }
    finally:
        if teacher is not None:
            teacher.sensor.close()
        scene.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--takeover-steps", type=int, default=20)
    parser.add_argument("--image-resolution", type=int, default=112)
    parser.add_argument("--gaussian-blur-radius", type=float, default=1.5)
    parser.add_argument("--center-occlusion-fraction", type=float, default=0.25)
    args = parser.parse_args()
    manifest = json.loads(args.targets.read_text(encoding="utf-8"))
    target_end = len(manifest["targets"]) if args.target_end is None else args.target_end
    selected = manifest["targets"][args.target_start:target_end]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = LiveRestrictedPolicy(
        args.checkpoint,
        args.encoder_checkpoint,
        args.image_resolution,
        args.gaussian_blur_radius,
        args.center_occlusion_fraction,
        device,
    )
    reference = Reference()
    perception_config = load_config(
        Path(__file__).resolve().parent / "configs/restricted_perception.json"
    )
    episodes = []
    for target in selected:
        for start in target["starts"]:
            episode_file = output / f"{start['start_id']}.npz"
            if episode_file.is_file():
                continue
            result = collect_episode(
                policy=policy,
                reference=reference,
                perception_config=perception_config,
                target=target,
                start=start,
                output_dir=output,
                max_steps=args.max_steps,
                beta=args.beta,
                takeover_steps=args.takeover_steps,
                image_resolution=args.image_resolution,
                blur_radius=args.gaussian_blur_radius,
                occlusion_fraction=args.center_occlusion_fraction,
            )
            episodes.append(result)
            partial = {
                "schema_version": 1,
                "target_manifest": str(args.targets.resolve()),
                "policy_checkpoint": str(args.checkpoint.resolve()),
                "target_start": args.target_start,
                "target_end": target_end,
                "episodes": episodes,
            }
            (output / "manifest.json").write_text(
                json.dumps(partial, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
