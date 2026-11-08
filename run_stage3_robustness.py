"""Evaluate restricted students and a geometry teacher under frozen perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_restricted_policy import (
    DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
    LiveRestrictedPolicy,
    TASK_TO_DEG_DEG_MM,
    state_abc_mm_to_task,
    task_to_state_abc_mm,
)
from restricted_perception.config import load_config
from restricted_perception.fusion import CoarseFineFusion
from restricted_perception.online import RestrictedPerceptionAdapter
from restricted_perception.sidecar import _kinematic_tips
from restricted_perception.simulator import CoarseFinePerceptionSimulator
from restricted_perception.stage34 import (
    fallback_state_of,
    geometric_teacher_action,
    load_stage34_protocol,
    scenario_configuration,
)
from rcm_reference import Reference
from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings


ROOT = Path(__file__).resolve().parent


def _run_episode(
    *,
    policy_kind: str,
    student_policy: LiveRestrictedPolicy | None,
    reference: Reference,
    perception_config,
    image_settings: dict[str, float],
    teacher_gain: float,
    target_robot_mm: np.ndarray,
    target_id: str,
    target_index: int,
    seed: int,
    max_steps: int,
    threshold_mm: float,
    start_error_mm: float,
    scenario: str,
    stagnation_steps: int,
) -> dict:
    scene = RCMSceneV9()
    sensor = None
    fallback_counts = {
        "FINE_TRACKING": 0,
        "FINE_COAST": 0,
        "COARSE_FALLBACK": 0,
        "PRECISION_HOLD": 0,
    }
    try:
        target_world = scene.robot_to_world(target_robot_mm)
        scene.set_target_world(target_world)
        target_task_units = scene.target_task(validate=True) * TASK_TO_DEG_DEG_MM
        start_task_units = target_task_units.copy()
        start_task_units[2] -= float(start_error_mm)
        scene.set_task(state_abc_mm_to_task(start_task_units))
        scene.enable_motor_execution()
        scene.reset_motor_time(0.0)
        motor = scene.motor_execution
        if policy_kind == "student":
            settings = visual_servo_settings(scene)
            settings["perception_backend"] = "strict_rgbd"
            sensor = StrictRGBDSensor(scene, settings)
            adapter = RestrictedPerceptionAdapter(
                perception_config,
                target_robot_mm=target_robot_mm,
                seed=seed,
                episode_id=f"stage3_{scenario}_{policy_kind}_{target_index}",
            )
            simulator = None
            fusion = None
        else:
            adapter = None
            simulator = CoarseFinePerceptionSimulator(
                perception_config,
                seed=seed,
                episode_id=f"stage3_{scenario}_{policy_kind}_{target_index}",
            )
            fusion = CoarseFineFusion(perception_config)
        errors = []
        actions = []
        states = []
        success_streak = 0
        success = False
        settled_error_steps = []
        safety_limited_steps = 0
        safety_limited_streak = 0
        early_stop_reason = None
        best_error_mm = float("inf")
        last_improvement_step = 0
        for step_index in range(max_steps):
            state_abc = task_to_state_abc_mm(scene.current_task())
            tip_robot = _kinematic_tips(state_abc[None, :], reference)[0]
            if policy_kind == "student":
                sensor.observation_time_s = motor.time_s
                observation = sensor.observe()
                navigation = observation.get("oct_navigation_depth")
                delta_xyz_m = (
                    None
                    if navigation is None
                    else navigation.get("delta_xyz_m")
                )
                fine_valid = (
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
                    timestamp_s=motor.time_s,
                )
            else:
                frame = simulator.update(
                    target_robot_mm,
                    tip_robot,
                    motor.time_s,
                )
                restricted = fusion.step(frame, state=state_abc)
            vector = restricted.policy_vector().copy()
            fallback_counts[fallback_state_of(vector)] += 1
            if policy_kind == "student":
                if student_policy is None:
                    raise ValueError("student evaluation requires a checkpoint")
                velocity_abc = (
                    np.asarray(motor.actual_velocity, dtype=float)
                    * TASK_TO_DEG_DEG_MM
                )
                action = student_policy.action(
                    observation["rgb"],
                    state_abc,
                    vector,
                    velocity_abc,
                )
            elif policy_kind == "geometric_teacher":
                goal = (
                    np.zeros(3, dtype=float)
                    if restricted.selected_goal_vector_robot_mm is None
                    else restricted.selected_goal_vector_robot_mm
                )
                action = geometric_teacher_action(
                    reference,
                    state_abc,
                    goal,
                    gain=teacher_gain,
                )
            else:
                raise ValueError(f"unknown policy kind: {policy_kind}")
            if restricted.fallback_state.value == "PRECISION_HOLD":
                action = np.zeros(3, dtype=float)
            action = np.clip(
                np.asarray(action, dtype=float),
                -DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
                DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            )
            current_tip = scene.site_mm("needle_tip")
            error_mm = float(np.linalg.norm(current_tip - target_world))
            errors.append(error_mm)
            states.append(state_abc.tolist())
            actions.append(action.tolist())
            if error_mm < best_error_mm - 0.0005:
                best_error_mm = error_mm
                last_improvement_step = step_index
            if (
                int(stagnation_steps) > 0
                and step_index - last_improvement_step
                >= int(stagnation_steps)
            ):
                early_stop_reason = (
                    f"STAGNATION_{int(stagnation_steps)}_STEPS"
                )
                break
            if error_mm > 30.0:
                early_stop_reason = "ERROR_EXCEEDED_30MM"
                break
            velocity = np.asarray(motor.actual_velocity) * TASK_TO_DEG_DEG_MM
            if (
                error_mm < threshold_mm
                and np.all(
                    np.abs(velocity) <= np.array([0.05, 0.05, 0.01])
                )
            ):
                success_streak += 1
            else:
                success_streak = 0
            settled_error_steps.append(
                (
                    error_mm,
                    bool(
                        np.all(
                            np.abs(velocity)
                            <= np.array([0.05, 0.05, 0.01])
                        )
                    ),
                )
            )
            if success_streak >= 3:
                success = True
                break
            try:
                scene.set_task(
                    state_abc_mm_to_task(state_abc + action),
                    motor_dt=0.05,
                )
                safety_limited_streak = 0
            except ValueError:
                safety_limited_steps += 1
                safety_limited_streak += 1
            if safety_limited_streak >= 20:
                early_stop_reason = "SAFETY_LIMITED_20_STEPS"
                break
        final_tip = scene.site_mm("needle_tip")
        success_by_threshold = {}
        for threshold_um in (20.0, 10.0, 5.0):
            streak = 0
            maximum_streak = 0
            for error_mm, settled in settled_error_steps:
                if settled and error_mm < threshold_um / 1000.0:
                    streak += 1
                    maximum_streak = max(maximum_streak, streak)
                else:
                    streak = 0
            success_by_threshold[f"{threshold_um:g}"] = maximum_streak >= 3
        return {
            "success": success,
            "steps": len(errors),
            "final_error_mm": float(np.linalg.norm(final_tip - target_world)),
            "minimum_error_mm": float(np.min(errors)),
            "maximum_error_mm": float(np.max(errors)),
            "safety_limited_steps": safety_limited_steps,
            "early_stop_reason": early_stop_reason,
            "fallback_state_frames": fallback_counts,
            "success_by_threshold_um": success_by_threshold,
            "seed": int(seed),
            "target_id": target_id,
            "target_index": int(target_index),
            "target_robot_mm": target_robot_mm.tolist(),
            "policy_kind": policy_kind,
            "scenario": scenario,
            "measurement_source": (
                "strict_rgbd_plus_restricted_simulator"
                if policy_kind == "student"
                else "restricted_measurement_simulator"
            ),
            "ground_truth_used_for_control": False,
            "ground_truth_used_for_success": True,
            "states": states,
            "actions": actions,
            "errors_mm": errors,
        }
    finally:
        if sensor is not None:
            sensor.close()
        scene.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "stage34_protocol.json",
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=ROOT / "configs" / "target_split_manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
        ),
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v6_guided_oct_finetune/best_model.pth",
    )
    parser.add_argument(
        "--policy-kind",
        choices=("student", "geometric_teacher"),
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=("default", "noise", "occlusion", "oct_loss", "delay"),
        required=True,
    )
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    stage3 = protocol["stage3"]
    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    targets = manifest["development_targets"]
    target_start = int(args.target_start)
    target_end = min(int(args.target_end), len(targets))
    base_config = load_config(ROOT / "configs" / "restricted_perception.json")
    perception_config, image_settings = scenario_configuration(
        base_config,
        args.scenario,
        protocol,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_policy = None
    if args.policy_kind == "student":
        student_policy = LiveRestrictedPolicy(
            args.checkpoint,
            args.encoder_checkpoint,
            image_resolution=112,
            gaussian_blur_radius=image_settings["gaussian_blur_radius"],
            center_occlusion_fraction=image_settings[
                "center_occlusion_fraction"
            ],
            device=device,
        )
    reference = Reference()
    episodes = []
    existing_by_key = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_by_key = {
            (episode["target_id"], int(episode["seed"])): episode
            for episode in existing.get("episodes", [])
        }
    for target_index in range(target_start, target_end):
        target = targets[target_index]
        target_id = target["id"]
        seed = int(protocol["seed"]) + target_index
        key = (target_id, seed)
        if key in existing_by_key:
            episodes.append(existing_by_key[key])
            continue
        try:
            episode = _run_episode(
                policy_kind=args.policy_kind,
                student_policy=student_policy,
                reference=reference,
                perception_config=perception_config,
                image_settings=image_settings,
                teacher_gain=float(
                    stage3["robustness"]["geometric_teacher_gain"]
                ),
                target_robot_mm=np.asarray(target["point_mm"], dtype=float),
                target_id=target_id,
                target_index=target_index,
                seed=seed,
                max_steps=int(stage3["max_steps"]),
                threshold_mm=float(stage3["threshold_um"]) / 1000.0,
                start_error_mm=float(stage3["start_error_um"]) / 1000.0,
                scenario=args.scenario,
                stagnation_steps=int(stage3["stagnation_steps"]),
            )
        except Exception as error:
            episode = {
                "success": False,
                "steps": 0,
                "final_error_mm": 30.0,
                "minimum_error_mm": 30.0,
                "maximum_error_mm": 30.0,
                "safety_limited_steps": 0,
                "early_stop_reason": f"EXCEPTION:{type(error).__name__}:{error}",
                "fallback_state_frames": {},
                "seed": seed,
                "target_id": target_id,
                "target_index": target_index,
                "target_robot_mm": target["point_mm"],
                "policy_kind": args.policy_kind,
                "scenario": args.scenario,
                "ground_truth_used_for_control": False,
                "ground_truth_used_for_success": True,
                "states": [],
                "actions": [],
                "errors_mm": [],
            }
        episodes.append(episode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "result": "IN_PROGRESS",
                    "policy_kind": args.policy_kind,
                    "scenario": args.scenario,
                    "episodes": episodes,
                    "completed": len(episodes),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    successes = int(sum(episode["success"] for episode in episodes))
    result = {
        "result": "COMPLETE",
        "policy_kind": args.policy_kind,
        "scenario": args.scenario,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "target_start": target_start,
        "target_end": target_end,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": result["result"],
                "policy_kind": args.policy_kind,
                "scenario": args.scenario,
                "success_rate": result["success_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
