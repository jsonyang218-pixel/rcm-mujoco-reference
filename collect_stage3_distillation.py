"""Collect on-policy training states and seven-sample diffusion teacher labels."""

from __future__ import annotations

import argparse
import json
import sys
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
from restricted_perception.online import RestrictedPerceptionAdapter
from restricted_perception.sidecar import _kinematic_tips
from restricted_perception.stage34 import (
    load_stage34_protocol,
    sha256_file,
)
from rcm_reference import Reference
from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings


ROOT = Path(__file__).resolve().parent
if str(ROOT / "lerobot_training") not in sys.path:
    sys.path.insert(0, str(ROOT / "lerobot_training"))

from eval_policy import PolicyEvaluator  # noqa: E402


def _collect_run(
    *,
    target: dict,
    start: dict,
    student_policy: LiveRestrictedPolicy,
    teacher: PolicyEvaluator,
    reference: Reference,
    perception_config,
    protocol: dict,
) -> dict:
    scene = RCMSceneV9()
    sensor = None
    feature_rows = []
    teacher_actions = []
    states = []
    errors = []
    try:
        target_robot_mm = np.asarray(target["point_mm"], dtype=float)
        target_world_mm = scene.robot_to_world(target_robot_mm)
        scene.set_target_world(target_world_mm)
        target_task_units = (
            scene.target_task(validate=True) * TASK_TO_DEG_DEG_MM
        )
        start_task_units = target_task_units.copy()
        start_task_units[2] -= float(start["initial_error_um"]) / 1000.0
        scene.set_task(state_abc_mm_to_task(start_task_units))
        settings = visual_servo_settings(scene)
        settings["perception_backend"] = "strict_rgbd"
        sensor = StrictRGBDSensor(scene, settings)
        scene.enable_motor_execution()
        scene.reset_motor_time(0.0)
        motor = scene.motor_execution
        seed = int(start["seed"])
        teacher.reset_buffer(seed=seed)
        adapter = RestrictedPerceptionAdapter(
            perception_config,
            target_robot_mm=target_robot_mm,
            seed=seed,
            episode_id=f"stage3_distill_{start['start_id']}",
        )
        success_streak = 0
        success = False
        safety_limited_steps = 0
        safety_limited_streak = 0
        sample_every = int(
            protocol["stage3"]["distillation"]["sample_every_frames"]
        )
        for frame_index in range(
            int(protocol["stage3"]["distillation"]["max_rollout_steps"])
        ):
            sensor.observation_time_s = motor.time_s
            observation = sensor.observe()
            navigation = observation.get("oct_navigation_depth")
            delta_xyz_m = (
                None if navigation is None else navigation.get("delta_xyz_m")
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
            state_abc = task_to_state_abc_mm(scene.current_task())
            tip_robot = _kinematic_tips(state_abc[None, :], reference)[0]
            restricted = adapter.observe(
                state=state_abc,
                kinematic_tip_robot_mm=tip_robot,
                fine_goal_vector_camera_mm=fine_camera,
                fine_valid=fine_valid,
                timestamp_s=motor.time_s,
            )
            vector = restricted.policy_vector().copy()
            velocity_abc = (
                np.asarray(motor.actual_velocity, dtype=float)
                * TASK_TO_DEG_DEG_MM
            )
            features = student_policy.encode_features(
                observation["rgb"],
                state_abc,
                vector,
                velocity_abc,
            )
            oct_feature = np.zeros(4, dtype=np.float32)
            if fine_valid:
                oct_feature[:3] = np.asarray(delta_xyz_m, dtype=float) * 1000.0
                oct_feature[3] = 1.0
            teacher_action = teacher.get_action(
                observation["rgb"],
                state_abc,
                oct_feature,
            )
            if frame_index % sample_every == 0:
                feature_rows.append(
                    features.detach().float().cpu().numpy()[0].copy()
                )
                teacher_actions.append(
                    np.asarray(teacher_action, dtype=np.float32).copy()
                )
                states.append(state_abc.copy())
            student_action = student_policy.action_from_features(features)
            if restricted.fallback_state.value == "PRECISION_HOLD":
                student_action = np.zeros(3, dtype=float)
            student_action = np.clip(
                np.asarray(student_action, dtype=float),
                -DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
                DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            )
            current_tip = scene.site_mm("needle_tip")
            error_mm = float(np.linalg.norm(current_tip - target_world_mm))
            errors.append(error_mm)
            if error_mm > 30.0:
                break
            velocity = np.asarray(motor.actual_velocity) * TASK_TO_DEG_DEG_MM
            if (
                error_mm < float(protocol["stage3"]["threshold_um"]) / 1000.0
                and np.all(
                    np.abs(velocity) <= np.array([0.05, 0.05, 0.01])
                )
            ):
                success_streak += 1
            else:
                success_streak = 0
            if success_streak >= 3:
                success = True
                break
            try:
                scene.set_task(
                    state_abc_mm_to_task(state_abc + student_action),
                    motor_dt=0.05,
                )
                safety_limited_streak = 0
            except ValueError:
                safety_limited_steps += 1
                safety_limited_streak += 1
                if safety_limited_streak >= 20:
                    break
        return {
            "target_id": target["id"],
            "start_id": start["start_id"],
            "seed": seed,
            "success": success,
            "frames": int(len(errors)),
            "samples": int(len(teacher_actions)),
            "final_error_um": float(errors[-1] * 1000.0),
            "minimum_error_um": float(min(errors) * 1000.0),
            "safety_limited_steps": safety_limited_steps,
            "features": np.asarray(feature_rows, dtype=np.float32),
            "teacher_actions": np.asarray(
                teacher_actions,
                dtype=np.float32,
            ),
            "states": np.asarray(states, dtype=np.float32),
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
        default=ROOT / "configs/stage34_protocol.json",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/restricted_stage2_variants_v5_20261106/dagger_only/best_model.pth"
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v7_guided_dagger1_cached/best_model.pth",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=ROOT / "outputs/diffusion_policy_v6_guided_oct_finetune/best_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/stage3_distillation_v2_20261106",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    args = parser.parse_args()
    protocol = load_stage34_protocol(args.protocol)
    distillation = protocol["stage3"]["distillation"]
    target_manifest_path = ROOT / distillation["training_target_manifest"]
    target_manifest = json.loads(
        target_manifest_path.read_text(encoding="utf-8")
    )
    runs = [
        (target, start)
        for target in target_manifest["targets"]
        for start in target["starts"]
    ]
    end_index = len(runs) if args.end_index is None else int(args.end_index)
    if not 0 <= args.start_index < end_index <= len(runs):
        raise ValueError("invalid distillation run range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_policy = LiveRestrictedPolicy(
        args.student_checkpoint,
        args.encoder_checkpoint,
        image_resolution=112,
        gaussian_blur_radius=1.5,
        center_occlusion_fraction=0.25,
        device=device,
    )
    teacher = PolicyEvaluator(
        str(args.teacher_checkpoint),
        device=str(device),
        num_action_samples=int(distillation["teacher_samples"]),
    )
    reference = Reference()
    perception_config = load_config(
        ROOT / "configs/restricted_perception.json"
    )
    summaries = []
    for index in range(args.start_index, end_index):
        target, start = runs[index]
        output = run_dir / f"{start['start_id']}.npz"
        summary_path = run_dir / f"{start['start_id']}.json"
        if output.is_file() and summary_path.is_file():
            summaries.append(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
            continue
        result = _collect_run(
            target=target,
            start=start,
            student_policy=student_policy,
            teacher=teacher,
            reference=reference,
            perception_config=perception_config,
            protocol=protocol,
        )
        arrays = {
            "features": result.pop("features"),
            "teacher_actions": result.pop("teacher_actions"),
            "states": result.pop("states"),
        }
        np.savez_compressed(output, **arrays)
        result["run_npz"] = str(output.resolve())
        summary_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(result)
        print(
            json.dumps(
                {
                    "completed": len(summaries) + args.start_index,
                    "total": len(runs),
                    "start_id": start["start_id"],
                    "samples": result["samples"],
                }
            ),
            flush=True,
        )
    all_summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("*.json"))
    ]
    metadata = {
        "schema_version": 1,
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(args.teacher_checkpoint),
        "teacher_samples": int(distillation["teacher_samples"]),
        "aggregation": "median",
        "training_target_manifest": str(target_manifest_path.resolve()),
        "training_target_manifest_sha256": sha256_file(target_manifest_path),
        "runs": all_summaries,
        "total_runs": len(all_summaries),
        "total_samples": int(
            sum(item["samples"] for item in all_summaries)
        ),
        "training_split_only": True,
        "development_or_final_targets_used": False,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "runs": metadata["total_runs"],
                "samples": metadata["total_samples"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
