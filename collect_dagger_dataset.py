"""Collect teacher-labelled on-policy correction episodes for DAgger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from rcm_reference import HERE
from rcm_scene_v9 import RCMSceneV9
from rcm_fundus import make_random_fundus_targets
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import VisualServoController, visual_servo_settings
from rcm_waypoint_dataset import EpisodeWriter, make_policy_task
from run_waypoint_comparison import guard_scene_goal_inverse
sys.path.insert(0, str(HERE / "lerobot_training"))
from eval_policy import PolicyEvaluator


UNITS = np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0])


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def sample_target(config, seed):
    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    try:
        rng = np.random.default_rng(seed)
        return np.asarray(
            make_random_fundus_targets(scene, rng, 1)["targets"][0]["point_mm"],
            dtype=np.float64,
        )
    finally:
        scene.close()


def sample_start_errors(rng, count):
    values = []
    while len(values) < count:
        candidate = float(rng.uniform(3.0, 15.0))
        if all(abs(candidate-old) >= 2.0 for old in values):
            values.append(candidate)
    return sorted(values, reverse=True)


def reference_limits(data_dir):
    with np.load(Path(data_dir) / "data" / "dataset.npz", allow_pickle=False) as data:
        states = data["state"].astype(np.float64)
        actions = data["action"].astype(np.float64)
    return {
        "state_min": states.min(axis=0).tolist(),
        "state_max": states.max(axis=0).tolist(),
        "action_abs_max": np.abs(actions).max(axis=0).tolist(),
    }


def pre_roll_teacher(scene, target_error_mm, max_iterations):
    settings = visual_servo_settings(scene)
    settings.update(
        max_iterations=max_iterations,
        save_diagnostics=False,
        position_feedback="geometry",
        occlusion_handling="stop",
        perception_backend="strict_rgbd",
    )
    guard_scene_goal_inverse(scene)
    controller = VisualServoController(scene, settings=settings)

    def stop_near_target(iteration, observation):
        del observation
        return scene.metrics()["tip_target_error_um"] > target_error_mm * 1000.0

    controller.iteration_callback = stop_near_target
    report = controller.run()
    return report


def collect_rollout(
    config,
    policy,
    output_dir,
    episode_index,
    seed,
    target_robot_mm,
    start_error_mm,
    beta,
    takeover_steps,
    takeover_max_steps,
    hold_frames,
    max_steps,
    pre_roll_max_iterations,
    max_error_mm,
    state_min,
    state_max,
    action_limit,
):
    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    episode_dir = output_dir / f"episode_{episode_index:06d}"
    episode_dir.mkdir()
    dataset_dir = output_dir / "dataset"
    policy.reset_buffer(seed=seed)
    started_wall = time.monotonic()
    try:
        scene.set_task(np.zeros(3))
        scene.set_target_world(scene.robot_to_world(target_robot_mm))
        scene.enable_motor_execution(config.get("motor_execution"))
        pre_roll = pre_roll_teacher(
            scene,
            start_error_mm,
            pre_roll_max_iterations,
        )
        scene.reset_motor_time(0.0)
        sensor_settings = visual_servo_settings(scene)
        sensor_settings.update(
            save_diagnostics=False,
            position_feedback="geometry",
            occlusion_handling="stop",
            perception_backend="strict_rgbd",
        )
        teacher = VisualServoController(scene, settings=sensor_settings)
        sensor = teacher.sensor
        writer = EpisodeWriter(
            dataset_dir,
            episode_index,
            make_policy_task(
                "visual_current_target",
                [],
            ),
            "dagger_policy_teacher",
            seed,
            ["P1"],
        )
        rng = np.random.default_rng(seed + 17)
        rows = []
        teacher_previous_velocity = np.zeros(3)
        teacher_command_q = scene.current_task().copy()
        takeover_left = 0
        takeover_executed = 0
        hold_seen = 0
        hold_remaining = 0
        best_error = float("inf")
        worsening_streak = 0
        invalid_teacher_frames = 0
        success = False
        stop_reason = None
        for step in range(max_steps):
            sensor.observation_time_s = scene.motor_execution.time_s
            observation = sensor.observe()
            state_si = scene.current_task().copy()
            state_abc = state_si * UNITS
            velocity_si = np.asarray(
                scene.motor_execution.actual_velocity,
                dtype=np.float64,
            )
            error_mm = float(
                scene.metrics()["tip_target_error_um"] / 1000.0
            )
            navigation = observation.get("oct_navigation_depth")
            delta_xyz_m = None if navigation is None else navigation.get("delta_xyz_m")
            oct_valid = bool(
                delta_xyz_m is not None
                and len(delta_xyz_m) == 3
                and np.all(np.isfinite(delta_xyz_m))
            )
            oct_feature = np.zeros(4, dtype=np.float32)
            if oct_valid:
                oct_feature[:3] = np.asarray(delta_xyz_m) * 1000.0
                oct_feature[3] = 1.0
            policy_action = np.asarray(
                policy.get_action(observation["rgb"], state_abc, oct_feature),
                dtype=np.float64,
            )
            policy_action_limited = bool(
                np.any(np.abs(policy_action) > np.asarray(action_limit) + 1e-12)
            )
            policy_action = np.clip(policy_action, -action_limit, action_limit)
            if takeover_left > 0:
                teacher_label = teacher.geometry_action_from_observation(
                    state_si,
                    observation,
                    previous_velocity=teacher_previous_velocity,
                    command_q=teacher_command_q,
                )
            else:
                teacher_label = teacher.geometry_action_from_observation(
                    state_si,
                    observation,
                    previous_velocity=velocity_si,
                    command_q=state_si,
                )
            teacher_valid = bool(teacher_label["valid"])
            if not teacher_valid:
                invalid_teacher_frames += 1
            if error_mm > float(max_error_mm):
                stop_reason = f"ERROR_EXCEEDED_{max_error_mm:g}MM"
                break
            within_range = bool(
                np.all(state_abc >= np.asarray(state_min) - 1e-9)
                and np.all(state_abc <= np.asarray(state_max) + 1e-9)
            )
            if not within_range:
                stop_reason = "STATE_OUTSIDE_TRAINING_RANGE"
                break
            if error_mm < best_error - 1e-6:
                best_error = error_mm
                worsening_streak = 0
            elif error_mm > best_error + 0.1:
                worsening_streak += 1
            else:
                worsening_streak = 0
            settled = bool(
                np.all(
                    np.abs(velocity_si * UNITS)
                    <= np.array([0.05, 0.05, 0.01])
                )
            )
            if error_mm <= 0.020 and settled:
                hold_seen += 1
            else:
                hold_seen = 0
            if hold_seen >= 3 and hold_remaining == 0:
                hold_remaining = int(hold_frames)
            if hold_remaining == 0 and takeover_left == 0:
                if worsening_streak >= 2:
                    takeover_left = int(takeover_steps)
                    takeover_executed = 0
                    teacher_previous_velocity = velocity_si.copy()
                    teacher_command_q = state_si.copy()
                elif policy_action_limited:
                    takeover_left = int(takeover_steps)
                    takeover_executed = 0
                    teacher_previous_velocity = velocity_si.copy()
                    teacher_command_q = state_si.copy()
            if hold_remaining > 0:
                executed_action = np.zeros(3)
                executor = "hold"
                teacher_action = np.zeros(3)
                hold_remaining -= 1
            elif takeover_left > 0:
                if not teacher_valid:
                    stop_reason = f"TEACHER_INVALID: {teacher_label['reason']}"
                    break
                executed_action = np.asarray(
                    teacher_label["action_deg_deg_mm"],
                    dtype=np.float64,
                )
                executed_action = np.clip(
                    executed_action,
                    -action_limit,
                    action_limit,
                )
                executor = "teacher"
                teacher_action = executed_action
                teacher_previous_velocity = np.asarray(
                    teacher_label["command_velocity_task_si"],
                    dtype=np.float64,
                )
                teacher_command_q = np.asarray(
                    teacher_label["candidate_task_si"],
                    dtype=np.float64,
                )
                takeover_left -= 1
                takeover_executed += 1
                if takeover_left == 0:
                    recovered = bool(
                        teacher_label.get("measured_error_um") is not None
                        and teacher_label["measured_error_um"] <= 100.0
                    )
                    if recovered:
                        takeover_left = 0
                    elif takeover_executed >= int(takeover_max_steps):
                        stop_reason = "TEACHER_RECOVERY_TIMEOUT"
                        break
            elif teacher_valid and rng.random() < beta and not policy_action_limited:
                executed_action = np.clip(
                    np.asarray(
                        teacher_label["action_deg_deg_mm"],
                        dtype=np.float64,
                    ),
                    -action_limit,
                    action_limit,
                )
                executor = "beta_teacher"
                teacher_action = executed_action
            else:
                executed_action = policy_action
                executor = "policy"
                teacher_action = (
                    np.asarray(teacher_label["action_deg_deg_mm"], dtype=np.float64)
                    if teacher_valid else None
                )
            action_abs = state_abc + executed_action
            timestamp = float(scene.motor_execution.time_s)
            visibility = sensor.visibility_metrics(observation["segmentation"])
            writer.add(
                timestamp,
                state_abc,
                action_abs,
                "P1",
                observation["rgb"],
                terminal=bool(hold_remaining == 0 and hold_seen >= 3),
                phase="hold" if hold_remaining > 0 or hold_seen >= 3 else "dagger",
                visibility=visibility,
                perception={
                    "control_frame_index": step,
                    "true_position_error_um": error_mm * 1000.0,
                    "target_observed": bool(
                        observation.get("target_pixel") is not None
                    ),
                },
            )
            record = writer.records[-1]
            rows.append(
                {
                    "frame_index": step,
                    "timestamp_s": timestamp,
                    "state_A_deg_B_deg_C_mm": state_abc.tolist(),
                    "policy_action_deg_deg_mm": policy_action.tolist(),
                    "teacher_action_deg_deg_mm": (
                        None if teacher_action is None else np.asarray(
                            teacher_action,
                            dtype=np.float64,
                        ).tolist()
                    ),
                    "executed_action_deg_deg_mm": executed_action.tolist(),
                    "executor": executor,
                    "teacher_valid": teacher_valid,
                    "teacher_reason": teacher_label["reason"],
                    "teacher_depth_valid": teacher_label["depth_valid"],
                    "teacher_measured_error_um": teacher_label.get(
                        "measured_error_um"
                    ),
                    "oct_navigation_mm": oct_feature[:3].tolist(),
                    "oct_navigation_valid": oct_valid,
                    "automatic_truth_error_mm": error_mm,
                    "policy_action_limited": policy_action_limited,
                    "state_within_training_range": within_range,
                    "image_path": record["observation.images.microscope_rgb"],
                }
            )
            next_task = state_si + executed_action / UNITS
            try:
                scene.set_task(next_task, motor_dt=float(sensor_settings["sample_period_s"]))
            except ValueError as error:
                stop_reason = f"TASK_CONSTRAINT: {error}"
                break
            policy.record_applied_action(executed_action)
            if hold_remaining == 0 and hold_seen >= 3:
                success = True
                stop_reason = "POSITION_ERROR_AND_SETTLED"
                break
        if not rows:
            failed_dir = output_dir / (
                f"failed_episode_{episode_index:06d}_no_frames"
            )
            if not failed_dir.exists():
                episode_dir.replace(failed_dir)
            raise RuntimeError("DAgger episode produced no frames")
        result = "PASS" if success else "FAIL"
        episode = writer.finish(
            result,
            stop_constraint_summary={
                "pass": success,
                "attempted_targets": 1,
                "targets_with_validated_stop": int(success),
            },
        )
        episode.update(
            initial_error_mm=start_error_mm,
            final_error_mm=float(scene.metrics()["tip_target_error_um"] / 1000.0),
            stop_reason=stop_reason,
            teacher_invalid_frames=invalid_teacher_frames,
            teacher_invalid_fraction=(
                invalid_teacher_frames / len(rows) if rows else 1.0
            ),
            collection_wall_s=time.monotonic() - started_wall,
            target_robot_mm=np.asarray(target_robot_mm).tolist(),
            teacher_pre_roll=pre_roll,
        )
        write_jsonl(episode_dir / "dagger_samples.jsonl", rows)
        write_json(episode_dir / "control_report.json", episode)
        writer.finish
        return episode
    finally:
        try:
            sensor.close()
        except (AttributeError, UnboundLocalError):
            pass
        scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--starts-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20280101)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--teacher-takeover-steps", type=int, default=40)
    parser.add_argument("--teacher-takeover-max-steps", type=int, default=400)
    parser.add_argument("--hold-frames", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--pre-roll-max-iterations", type=int, default=1000)
    parser.add_argument("--max-error-mm", type=float, default=15.0)
    parser.add_argument("--num-action-samples", type=int, default=7)
    parser.add_argument(
        "--resume-start-episode",
        type=int,
        default=0,
        help=(
            "Resume an existing output directory at this zero-based episode; "
            "earlier episodes must have control_report.json."
        ),
    )
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument("--target-end", type=int, default=None)
    parser.add_argument(
        "--skip-episodes-before",
        type=int,
        default=0,
        help="Skip global episode indexes below this value while still "
        "advancing deterministic target/start sampling.",
    )
    parser.add_argument(
        "--start-errors-mm",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit teacher-to-policy handover error thresholds",
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=HERE / "data" / "lerobot_rcm_train_recovery_full_224",
    )
    args = parser.parse_args()
    if min(
        args.episodes,
        args.starts_per_episode,
        args.teacher_takeover_steps,
        args.teacher_takeover_max_steps,
        args.hold_frames,
        args.max_steps,
        args.pre_roll_max_iterations,
        args.num_action_samples,
    ) < 1:
        parser.error("count and iteration arguments must be positive")
    if args.resume_start_episode < 0:
        parser.error("--resume-start-episode must be nonnegative")
    if (
        args.target_start < 0
        or args.target_start >= args.episodes
        or (
            args.target_end is not None
            and not args.target_start <= args.target_end < args.episodes
        )
        or args.skip_episodes_before < 0
    ):
        parser.error("invalid target range or skip-episodes-before")
    if not 0 <= args.beta <= 1:
        parser.error("--beta must be in [0, 1]")
    if args.max_error_mm <= 0:
        parser.error("--max-error-mm must be positive")
    if args.pre_roll_max_iterations < 1:
        parser.error("--pre-roll-max-iterations must be positive")
    if args.start_errors_mm is not None:
        if len(args.start_errors_mm) != args.starts_per_episode:
            parser.error(
                "--start-errors-mm must provide one value per start"
            )
        if any(not 3.0 <= value <= 15.0 for value in args.start_errors_mm):
            parser.error("--start-errors-mm values must be in [3, 15]")
    output_dir = args.output_dir.resolve()
    if args.resume_start_episode:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config["motor_execution"] = {
        **config.get("motor_execution", {}),
        "enabled": True,
    }
    policy = PolicyEvaluator(
        str(Path(args.model).resolve()),
        device="cuda",
        num_action_samples=args.num_action_samples,
    )
    limits = reference_limits(args.reference_data_dir)
    state_min = np.asarray(limits["state_min"], dtype=np.float64)
    state_max = np.asarray(limits["state_max"], dtype=np.float64)
    action_limit = np.asarray(limits["action_abs_max"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    episodes = []
    skipped_empty_episodes = []
    resume_missing_prefix = []
    if args.resume_start_episode:
        for report_path in sorted(
            output_dir.glob("episode_*/control_report.json")
        ):
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            if int(report["episode_index"]) < args.resume_start_episode:
                episodes.append(report)
        existing_indexes = {
            int(report["episode_index"]) for report in episodes
        }
        expected_indexes = set(range(args.resume_start_episode))
        resume_missing_prefix = sorted(expected_indexes-existing_indexes)
    target_end = (
        args.episodes - 1
        if args.target_end is None
        else args.target_end
    )
    skip_episodes_before = max(
        args.resume_start_episode,
        args.skip_episodes_before,
    )
    episode_index = 0
    for target_index in range(target_end + 1):
        target = sample_target(config, args.seed + target_index)
        start_errors = (
            list(args.start_errors_mm)
            if args.start_errors_mm is not None
            else sample_start_errors(rng, args.starts_per_episode)
        )
        for start_index, start_error in enumerate(start_errors):
            if (
                target_index < args.target_start
                or episode_index < skip_episodes_before
            ):
                episode_index += 1
                continue
            print(
                json.dumps(
                    {
                        "target": target_index,
                        "start": start_index,
                        "episode": episode_index,
                        "start_error_mm": start_error,
                    }
                ),
                flush=True,
            )
            try:
                episode = collect_rollout(
                    config,
                    policy,
                    output_dir,
                    episode_index,
                    args.seed + episode_index,
                    target,
                    start_error,
                    args.beta,
                    args.teacher_takeover_steps,
                    args.teacher_takeover_max_steps,
                    args.hold_frames,
                    args.max_steps,
                    args.pre_roll_max_iterations,
                    args.max_error_mm,
                    state_min,
                    state_max,
                    action_limit,
                )
            except RuntimeError as error:
                if (
                    "DAgger episode produced no frames" in str(error)
                    and not (
                        output_dir
                        / f"episode_{episode_index:06d}"
                    ).exists()
                ):
                    skipped_empty_episodes.append(
                        {
                            "episode_index": episode_index,
                            "target_index": target_index,
                            "start_index": start_index,
                            "reason": str(error),
                        }
                    )
                    print(
                        json.dumps(
                            skipped_empty_episodes[-1],
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    episode_index += 1
                    continue
                raise
            episodes.append(episode)
            episode_index += 1
    selected_first = args.target_start * args.starts_per_episode
    selected_last = (target_end + 1) * args.starts_per_episode
    requested_episodes = max(
        0,
        selected_last - max(selected_first, skip_episodes_before),
    )
    manifest = {
        "schema_version": 1,
        "format": "DAgger episodes with per-frame policy, teacher and executed actions",
        "model": str(Path(args.model).resolve()),
        "policy_type": policy.policy_type,
        "policy_checkpoint_epoch": int(policy.checkpoint["epoch"]),
        "policy_guided_oct": bool(
            policy.checkpoint.get("config", {}).get(
                "guided_oct_mode",
                False,
            )
        ),
        "seed": args.seed,
        "beta": args.beta,
        "teacher_takeover_steps": args.teacher_takeover_steps,
        "teacher_takeover_max_steps": args.teacher_takeover_max_steps,
        "hold_frames": args.hold_frames,
        "max_error_mm": args.max_error_mm,
        "state_range_A_deg_B_deg_C_mm": [
            state_min.tolist(),
            state_max.tolist(),
        ],
        "action_abs_limit_deg_deg_mm": action_limit.tolist(),
        "episodes": episodes,
        "requested_episodes": requested_episodes,
        "completed_episodes": len(episodes),
        "skipped_empty_episodes": skipped_empty_episodes,
        "resume_missing_prefix": resume_missing_prefix,
        "resume_start_episode": args.resume_start_episode,
        "target_start": args.target_start,
        "target_end": target_end,
        "skip_episodes_before": skip_episodes_before,
        "total_frames": sum(row["frames"] for row in episodes),
        "teacher_invalid_frames": sum(
            row["teacher_invalid_frames"] for row in episodes
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
