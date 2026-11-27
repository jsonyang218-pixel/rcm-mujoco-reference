"""Collect teacher-labelled on-policy correction episodes for DAgger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from rcm.reference import HERE
from rcm.scene_v9 import RCMSceneV9
from rcm.fundus import make_random_fundus_targets
from rcm.strict_rgbd import StrictRGBDSensor
from rcm.visual_servo import VisualServoController, visual_servo_settings
from rcm.waypoint_dataset import EpisodeWriter, make_policy_task
from scripts.applications.run_waypoint_comparison import guard_scene_goal_inverse
from scripts.datasets.collect_v9_dataset import (
    RECOVERY_START_PROFILES,
    recovery_start_task,
)
sys.path.insert(0, str(HERE / "lerobot_training"))
from eval_policy import PolicyEvaluator


UNITS = np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0])


def install_policy_architecture(model_path):
    """Install the checkpoint's DiT architecture before PolicyEvaluator loads."""
    import torch
    import dit_rectified_flow as flow

    checkpoint = torch.load(
        model_path, map_location="cpu", weights_only=False
    )
    architecture = checkpoint.get("architecture")
    if architecture == "state_conditioned_temporal_attention_v1":
        import dit_state_attention as state_attention

        flow.DiTRectifiedFlowPolicy = (
            state_attention.StateConditionedDiTRectifiedFlowPolicy
        )
    elif architecture not in (None, "dit_rectified_flow_v1"):
        raise ValueError(
            f"unsupported collection policy architecture: {architecture}"
        )


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


def sample_target(
    config,
    seed,
    minimum_normalized_posterior_z=0.9,
):
    scene = RCMSceneV9({**config, "motor_execution": {"enabled": False}})
    try:
        rng = np.random.default_rng(seed)
        return np.asarray(
            make_random_fundus_targets(
                scene,
                rng,
                1,
                minimum_normalized_posterior_z=minimum_normalized_posterior_z,
            )["targets"][0]["point_mm"],
            dtype=np.float64,
        )
    finally:
        scene.close()


def make_reachable_target(config, requested_target_robot_mm):
    """Map a nominal target to a fixed, protocol-reachable needle-tip point."""
    requested_target_robot_mm = np.asarray(
        requested_target_robot_mm,
        dtype=np.float64,
    )
    planning_config = {
        **config,
        "motor_execution": {
            **config.get("motor_execution", {}),
            "enabled": False,
        },
    }
    inverse_scene = RCMSceneV9(planning_config)
    try:
        inverse_scene.set_task(np.zeros(3))
        requested_target_world_mm = inverse_scene.robot_to_world(
            requested_target_robot_mm
        )
        inverse_scene.set_target_world(requested_target_world_mm)
        requested_target_task = inverse_scene.target_task(validate=True)
        requested_terminal_robot_ABCE = (
            tuple(
                int(round(float(value)))
                for value in inverse_scene.encoders(requested_target_task)
            )
            + (0,)
        )
    finally:
        inverse_scene.close()

    scene = RCMSceneV9(planning_config)
    try:
        scene.set_task(np.zeros(3))
        scene.enable_motor_execution({"enabled": True})
        scene.reset_motor_time(0.0)
        motor = scene.motor_execution
        for _ in range(512):
            actual = motor.advance_robot(
                requested_terminal_robot_ABCE,
                motor.control_period_s,
                requested_task=requested_target_task,
            )
            scene._apply_task_pose(actual)
            if motor.settled():
                break
        else:
            raise RuntimeError("reachable-target planning did not settle")
        endpoint_robot_ABCE = tuple(motor.backend.robot)
        if endpoint_robot_ABCE != tuple(motor.backend.target):
            raise RuntimeError("reachable-target planning missed endpoint")
        endpoint_task = scene.task_from_encoders(
            endpoint_robot_ABCE[:3],
            reference_task=scene.current_task(),
        )
        reachable_tip_world_m = (
            scene.data.site_xpos[scene.sites["needle_tip"]].copy()
        )
        scene.set_target_world_m(reachable_tip_world_m)
        fixed_target_world_mm = scene.site_mm("lesion").copy()
        reachable_target_robot_mm = scene.world_to_robot(
            fixed_target_world_mm
        )
        shift_mm = (
            fixed_target_world_mm
            - requested_target_world_mm
        )
        return reachable_target_robot_mm, {
            "requested_target_robot_mm": requested_target_robot_mm.tolist(),
            "requested_target_world_mm": (
                requested_target_world_mm.tolist()
            ),
            "reachable_target_robot_mm": (
                reachable_target_robot_mm.tolist()
            ),
            "reachable_target_world_mm": fixed_target_world_mm.tolist(),
            "target_shift_mm": shift_mm.tolist(),
            "target_shift_norm_mm": float(np.linalg.norm(shift_mm)),
            "requested_terminal_robot_ABCE": list(
                requested_terminal_robot_ABCE
            ),
            "endpoint_robot_ABCE": list(endpoint_robot_ABCE),
            "endpoint_task_rad_rad_m": endpoint_task.tolist(),
        }
    finally:
        scene.close()


def sample_start_errors(rng, count):
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive")
    minimum, maximum, separation = 3.0, 15.0, 2.0
    required_span = separation * (count - 1)
    available_span = maximum - minimum
    if required_span > available_span + 1e-12:
        raise ValueError(
            f"cannot place {count} starts with {separation:g} mm separation "
            f"in [{minimum:g}, {maximum:g}] mm"
        )
    # Place ordered points with fixed gaps first, then distribute the
    # remaining span. This samples the feasible space without rejection.
    slack = rng.uniform(0.0, available_span - required_span, size=count)
    values = minimum + np.sort(slack) + separation * np.arange(count)
    return values[::-1].tolist()


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
        failure_hold_on_motor=False,
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
    stagnation_takeover_steps,
    stagnation_max_error_mm,
    stagnation_improvement_mm,
    state_min,
    state_max,
    action_limit,
    terminal_start_profile_index=None,
    terminal_robot_ABCE=None,
    reachable_target_report=None,
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
        # Capture the privileged terminal inverse before strict visual control
        # installs the goal-inverse guard. It is used only for the final
        # integer MotorBackend lock, never by the visual teacher.
        initial_target_task = scene.target_task(validate=True).copy()
        terminal_start_profile = None
        if terminal_start_profile_index is None:
            scene.enable_motor_execution(config.get("motor_execution"))
            pre_roll = pre_roll_teacher(
                scene,
                start_error_mm,
                pre_roll_max_iterations,
            )
        else:
            target_task_abc = initial_target_task * UNITS
            terminal_start_task, terminal_start_profile = recovery_start_task(
                target_task_abc,
                terminal_start_profile_index,
            )
            scene.set_task(terminal_start_task / UNITS)
            scene.enable_motor_execution(config.get("motor_execution"))
            pre_roll = {
                "result": "SKIPPED",
                "reason": "DIRECT_TERMINAL_START",
                "start_profile": terminal_start_profile,
            }
            start_error_mm = float(
                scene.metrics()["tip_target_error_um"] / 1000.0
            )
        # Keep the source-start MotorBackend anchor so the preplanned fixed
        # target endpoint remains exactly reachable. Pre-roll time is retained
        # in the audit timestamps and does not start another motor session.
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
        last_meaningful_improvement_step = 0
        worsening_streak = 0
        invalid_teacher_frames = 0
        success = False
        stop_reason = None
        takeover_trigger_reasons = {}
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
            if error_mm < best_error - float(stagnation_improvement_mm):
                best_error = error_mm
                worsening_streak = 0
                last_meaningful_improvement_step = step
            elif error_mm < best_error - 1e-6:
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
                    takeover_trigger_reasons["worsening"] = (
                        takeover_trigger_reasons.get("worsening", 0) + 1
                    )
                elif policy_action_limited:
                    takeover_left = int(takeover_steps)
                    takeover_executed = 0
                    teacher_previous_velocity = velocity_si.copy()
                    teacher_command_q = state_si.copy()
                    takeover_trigger_reasons["policy_action_limited"] = (
                        takeover_trigger_reasons.get(
                            "policy_action_limited", 0
                        )
                        + 1
                    )
                elif (
                    int(stagnation_takeover_steps) > 0
                    and error_mm <= float(stagnation_max_error_mm)
                    and step - last_meaningful_improvement_step
                    >= int(stagnation_takeover_steps)
                ):
                    takeover_left = int(takeover_steps)
                    takeover_executed = 0
                    teacher_previous_velocity = velocity_si.copy()
                    teacher_command_q = state_si.copy()
                    last_meaningful_improvement_step = step
                    takeover_trigger_reasons["near_target_stagnation"] = (
                        takeover_trigger_reasons.get(
                            "near_target_stagnation", 0
                        )
                        + 1
                    )
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
                # Terminal is reserved for the exact endpoint lock below.
                terminal=False,
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
        visual_success = bool(success)
        terminal_lock = None
        exact_terminal_error_um = None
        target_shift_mm = None
        target_shift_vector_mm = None
        if visual_success:
            if terminal_robot_ABCE is None:
                raise ValueError(
                    "exact DAgger collection requires a planned terminal encoder"
                )
            original_target_world_mm = scene.site_mm("lesion").copy()
            target_task = initial_target_task
            requested_terminal_robot_ABCE = tuple(terminal_robot_ABCE)
            motor = scene.motor_execution
            exact_steps = 0
            exact_settled = False
            for _ in range(128):
                sensor.observation_time_s = motor.time_s
                observation = sensor.observe()
                state_si = scene.current_task().copy()
                state_abc = state_si * UNITS
                actual = motor.advance_robot(
                    requested_terminal_robot_ABCE,
                    motor.control_period_s,
                    requested_task=target_task,
                )
                scene._apply_task_pose(actual)
                step_record = motor.last_record
                action_abs = (
                    np.asarray(
                        step_record["commanded_task_rad_rad_m"],
                        dtype=np.float64,
                    )
                    * UNITS
                )
                executed_action = action_abs - state_abc
                timestamp = float(step_record["start_time_s"])
                visibility = sensor.visibility_metrics(
                    observation["segmentation"]
                )
                writer.add(
                    timestamp,
                    state_abc,
                    action_abs,
                    "P1",
                    observation["rgb"],
                    terminal=False,
                    phase="exact_terminal_lock",
                    visibility=visibility,
                    perception={
                        "control_frame_index": step_record["step_index"],
                        "true_position_error_um": float(
                            scene.metrics()["tip_target_error_um"]
                        ),
                        "target_observed": bool(
                            observation.get("target_pixel") is not None
                        ),
                    },
                )
                image_path = writer.records[-1][
                    "observation.images.microscope_rgb"
                ]
                rows.append({
                    "frame_index": len(rows),
                    "timestamp_s": timestamp,
                    "state_A_deg_B_deg_C_mm": state_abc.tolist(),
                    "policy_action_deg_deg_mm": [0.0, 0.0, 0.0],
                    "teacher_action_deg_deg_mm": executed_action.tolist(),
                    "executed_action_deg_deg_mm": executed_action.tolist(),
                    "executor": "exact_terminal",
                    "teacher_valid": True,
                    "teacher_reason": "EXACT_TERMINAL_LOCK",
                    "teacher_depth_valid": True,
                    "teacher_measured_error_um": float(
                        scene.metrics()["tip_target_error_um"]
                    ),
                    "oct_navigation_mm": np.zeros(3).tolist(),
                    "oct_navigation_valid": False,
                    "automatic_truth_error_mm": float(
                        scene.metrics()["tip_target_error_um"] / 1000.0
                    ),
                    "policy_action_limited": False,
                    "state_within_training_range": True,
                    "image_path": image_path,
                })
                policy.record_applied_action(executed_action)
                exact_steps += 1
                if motor.settled():
                    exact_settled = True
                    break
            if exact_settled:
                terminal_tip_world_m = (
                    scene.data.site_xpos[scene.sites["needle_tip"]].copy()
                )
                terminal_target_world_m = (
                    scene.data.site_xpos[scene.sites["lesion"]].copy()
                )
                exact_terminal_error_um = float(
                    np.linalg.norm(
                        (
                            terminal_tip_world_m
                            - terminal_target_world_m
                        )
                        * 1_000_000.0
                    )
                )
                if exact_terminal_error_um > 0.1:
                    success = False
                    stop_reason = (
                        "EXACT_TERMINAL_ERROR_ABOVE_0P1UM: "
                        f"{exact_terminal_error_um}"
                    )
                shift_mm = (
                    terminal_target_world_m * 1000.0
                    - original_target_world_mm
                )
                target_shift_vector_mm = shift_mm.tolist()
                target_shift_mm = float(np.linalg.norm(shift_mm))

                if success:
                    sensor.observation_time_s = motor.time_s
                    final_observation = sensor.observe()
                    final_state_abc = scene.current_task() * UNITS
                    writer.add(
                        float(motor.time_s),
                        final_state_abc,
                        final_state_abc,
                        "P1",
                        final_observation["rgb"],
                        terminal=True,
                        phase="hold",
                        visibility=sensor.visibility_metrics(
                            final_observation["segmentation"]
                        ),
                        perception={
                            "control_frame_index": motor.last_record["step_index"],
                            "true_position_error_um": exact_terminal_error_um,
                            "target_observed": bool(
                                final_observation.get("target_pixel") is not None
                            ),
                        },
                    )
                    final_image_path = writer.records[-1][
                        "observation.images.microscope_rgb"
                    ]
                    rows.append({
                        "frame_index": len(rows),
                        "timestamp_s": float(motor.time_s),
                        "state_A_deg_B_deg_C_mm": final_state_abc.tolist(),
                        "policy_action_deg_deg_mm": [0.0, 0.0, 0.0],
                        "teacher_action_deg_deg_mm": [0.0, 0.0, 0.0],
                        "executed_action_deg_deg_mm": [0.0, 0.0, 0.0],
                        "executor": "hold",
                        "teacher_valid": True,
                        "teacher_reason": "EXACT_TERMINAL_HOLD",
                        "teacher_depth_valid": True,
                        "teacher_measured_error_um": exact_terminal_error_um,
                        "oct_navigation_mm": np.zeros(3).tolist(),
                        "oct_navigation_valid": False,
                        "automatic_truth_error_mm": (
                            exact_terminal_error_um / 1000.0
                        ),
                        "policy_action_limited": False,
                        "state_within_training_range": True,
                        "image_path": final_image_path,
                    })
                terminal_lock = {
                    "requested_terminal_robot_ABCE": list(
                        requested_terminal_robot_ABCE
                    ),
                    "terminal_robot_ABCE": list(motor.backend.robot),
                    "terminal_target_robot_ABCE": list(motor.backend.target),
                    "exact_settle_steps": exact_steps,
                    "terminal_tip_world_m": terminal_tip_world_m.tolist(),
                    "terminal_target_world_m": (
                        terminal_target_world_m.tolist()
                    ),
                    "original_target_world_mm": (
                        original_target_world_mm.tolist()
                    ),
                    "target_shift_mm": target_shift_mm,
                    "target_shift_vector_mm": target_shift_vector_mm,
                    "final_error_um": exact_terminal_error_um,
                    "reachable_target_planning": (
                        reachable_target_report
                    ),
                }
                if success:
                    stop_reason = "EXACT_TERMINAL_LOCK_WITHIN_0P1UM"
            else:
                success = False
                stop_reason = "EXACT_TERMINAL_LOCK_DID_NOT_SETTLE"
        if not rows:
            failed_dir = output_dir / (
                f"failed_episode_{episode_index:06d}_no_frames"
            )
            if not failed_dir.exists():
                episode_dir.replace(failed_dir)
            raise RuntimeError("DAgger episode produced no frames")
        result = "PASS" if success else "FAIL"
        executor_counts = {}
        for row in rows:
            executor = row["executor"]
            executor_counts[executor] = (
                executor_counts.get(executor, 0) + 1
            )
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
            stop_reason=stop_reason,
            teacher_invalid_frames=invalid_teacher_frames,
            teacher_invalid_fraction=(
                invalid_teacher_frames / len(rows) if rows else 1.0
            ),
            collection_wall_s=time.monotonic() - started_wall,
            target_robot_mm=np.asarray(target_robot_mm).tolist(),
            teacher_pre_roll=pre_roll,
            terminal_start_profile=terminal_start_profile,
            exact_terminal_lock=terminal_lock,
            executor_counts=executor_counts,
            takeover_trigger_reasons=takeover_trigger_reasons,
            strict_zero_error=bool(
                success
                and exact_terminal_error_um is not None
                and exact_terminal_error_um <= 0.1
            ),
            final_error_um=(
                float(exact_terminal_error_um)
                if exact_terminal_error_um is not None
                else float(scene.metrics()["tip_target_error_um"])
            ),
            expert_terminal_tolerance_um=0.1,
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
    parser.add_argument(
        "--stagnation-takeover-steps",
        type=int,
        default=0,
        help=(
            "Trigger a teacher takeover after this many near-target steps "
            "without at least --stagnation-improvement-mm improvement; "
            "0 disables the trigger."
        ),
    )
    parser.add_argument(
        "--stagnation-max-error-mm",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--stagnation-improvement-mm",
        type=float,
        default=0.005,
    )
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
        "--target-seeds-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a unique 'seeds' list. When set, "
            "the file must provide one target seed per --episodes target."
        ),
    )
    parser.add_argument(
        "--target-min-posterior-z",
        type=float,
        default=0.9,
        help="Minimum normalized posterior z for sampled fundus targets.",
    )
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
    parser.add_argument(
        "--terminal-start-profile-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Use direct near-target recovery starts instead of teacher pre-roll. "
            "Profile indices are cycled globally across collected episodes."
        ),
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
    if args.stagnation_takeover_steps < 0:
        parser.error("--stagnation-takeover-steps must be nonnegative")
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
    if args.stagnation_max_error_mm <= 0:
        parser.error("--stagnation-max-error-mm must be positive")
    if args.stagnation_improvement_mm <= 0:
        parser.error("--stagnation-improvement-mm must be positive")
    if args.pre_roll_max_iterations < 1:
        parser.error("--pre-roll-max-iterations must be positive")
    if not 0.0 < args.target_min_posterior_z < 1.0:
        parser.error("--target-min-posterior-z must be between 0 and 1")
    if args.start_errors_mm is not None:
        if len(args.start_errors_mm) != args.starts_per_episode:
            parser.error(
                "--start-errors-mm must provide one value per start"
            )
        if any(not 0.02 <= value <= 15.0 for value in args.start_errors_mm):
            parser.error("--start-errors-mm values must be in [0.02, 15]")
    if args.terminal_start_profile_indices is not None:
        if not args.terminal_start_profile_indices:
            parser.error("--terminal-start-profile-indices must not be empty")
        if any(
            index < 0 or index >= len(RECOVERY_START_PROFILES)
            for index in args.terminal_start_profile_indices
        ):
            parser.error(
                "--terminal-start-profile-indices must be in "
                f"[0, {len(RECOVERY_START_PROFILES) - 1}]"
            )
    target_seeds = None
    target_points = None
    if args.target_seeds_file is not None:
        target_seed_payload = json.loads(
            args.target_seeds_file.read_text(encoding="utf-8")
        )
        target_seeds = target_seed_payload.get("seeds")
        target_points = target_seed_payload.get("points_mm")
        if (
            not isinstance(target_seeds, list)
            or len(target_seeds) != args.episodes
            or any(
                not isinstance(seed, int) or isinstance(seed, bool)
                for seed in target_seeds
            )
        ):
            parser.error(
                "--target-seeds-file must contain one integer target seed "
                "per --episodes target"
            )
        if len(set(target_seeds)) != len(target_seeds):
            parser.error("--target-seeds-file target seeds must be unique")
        if target_points is not None:
            target_points = np.asarray(target_points, dtype=np.float64)
            if (
                target_points.shape != (args.episodes, 3)
                or not np.all(np.isfinite(target_points))
            ):
                parser.error(
                    "--target-seeds-file points_mm must contain one finite "
                    "3D point per --episodes target"
                )
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
    install_policy_architecture(str(Path(args.model).resolve()))
    policy = PolicyEvaluator(
        str(Path(args.model).resolve()),
        device="cuda",
        num_action_samples=args.num_action_samples,
    )
    print("policy_ready", flush=True)
    limits = reference_limits(args.reference_data_dir)
    print("reference_limits_ready", flush=True)
    state_min = np.asarray(limits["state_min"], dtype=np.float64)
    state_max = np.asarray(limits["state_max"], dtype=np.float64)
    action_limit = np.asarray(limits["action_abs_max"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    reachable_target_reports = []
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
        print(
            json.dumps(
                {
                    "phase": "select_target",
                    "target_index": target_index,
                    "episode_index": episode_index,
                }
            ),
            flush=True,
        )
        target_seed = (
            args.seed + target_index
            if target_seeds is None
            else target_seeds[target_index]
        )
        requested_target_robot_mm = (
            sample_target(
                config,
                target_seed,
                args.target_min_posterior_z,
            )
            if target_points is None
            else target_points[target_index]
        )
        target, reachable_target_report = make_reachable_target(
            config,
            requested_target_robot_mm,
        )
        reachable_target_reports.append({
            "target_index": target_index,
            "target_seed": target_seed,
            **reachable_target_report,
        })
        print(
            json.dumps({
                "target_index": target_index,
                "requested_target_robot_mm": (
                    requested_target_robot_mm.tolist()
                ),
                "reachable_target_robot_mm": target.tolist(),
                "target_shift_norm_mm": (
                    reachable_target_report["target_shift_norm_mm"]
                ),
            }, ensure_ascii=False),
            flush=True,
        )
        if args.terminal_start_profile_indices is not None:
            start_errors = [0.0] * args.starts_per_episode
        elif args.start_errors_mm is not None:
            start_errors = list(args.start_errors_mm)
        else:
            start_errors = sample_start_errors(
                rng,
                args.starts_per_episode,
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
            terminal_start_profile_index = (
                None
                if args.terminal_start_profile_indices is None
                else args.terminal_start_profile_indices[
                    episode_index
                    % len(args.terminal_start_profile_indices)
                ]
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
                    args.stagnation_takeover_steps,
                    args.stagnation_max_error_mm,
                    args.stagnation_improvement_mm,
                    state_min,
                    state_max,
                    action_limit,
                    terminal_start_profile_index,
                    reachable_target_report[
                        "requested_terminal_robot_ABCE"
                    ],
                    reachable_target_report,
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
        "target_seeds_file": (
            None
            if args.target_seeds_file is None
            else str(args.target_seeds_file.resolve())
        ),
        "target_seeds": target_seeds,
        "target_min_posterior_z": args.target_min_posterior_z,
        "reachable_targets": reachable_target_reports,
        "expert_terminal_tolerance_um": 0.1,
        "terminal_start_profile_indices": (
            args.terminal_start_profile_indices
        ),
        "terminal_start_profiles": (
            None
            if args.terminal_start_profile_indices is None
            else [
                {
                    "index": index,
                    "name": RECOVERY_START_PROFILES[index][0],
                    "delta_A_deg_B_deg_C_mm": list(
                        RECOVERY_START_PROFILES[index][1]
                    ),
                }
                for index in args.terminal_start_profile_indices
            ]
        ),
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
        "strict_zero_error_episodes": sum(
            row.get("strict_zero_error") is True for row in episodes
        ),
        "maximum_final_error_um": max(
            (
                row.get("final_error_um", float("inf"))
                for row in episodes
            ),
            default=None,
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
