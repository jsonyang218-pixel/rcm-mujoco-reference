"""Select a shared visit order, compare visual servo/RRT/PRM, export RGB datasets."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rcm_navigation import (Segment, load_targets, navigation_settings, solve_target,
                            task_si_units, task_user_units, write_json)
from rcm_reference import HERE, Reference
from rcm_fundus import (clearance_lower_bound_mm, no_surface_slide_edge,
                        standoff_task, verify_fundus_targets)
from rcm_sampling_planners import (PRM, TaskSpace, TrapezoidMotion, move_duration,
                                  rrt, select_visit_order, shortcut_path)
from rcm_visual_servo import VisualServoController, VisualServoSensor
from rcm_waypoint_dataset import (TASK_REGIMES, EpisodeWriter, make_policy_task,
                                  summarize_target_success_by_visibility,
                                  summarize_visibility, write_manifest)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_frame(frames, timestamp, q, waypoint_id, phase="motion"):
    frame = {"time_s": float(timestamp), "q": np.asarray(q).copy(), "waypoint_id": waypoint_id, "phase": phase}
    if frames and abs(timestamp-frames[-1]["time_s"]) < 1e-10:
        if not np.allclose(q, frames[-1]["q"], atol=1e-10, rtol=0):
            raise ValueError("Discontinuous pose at waypoint boundary")
        frames[-1] = frame
    else:
        frames.append(frame)


def goal_result(reference, solution, duration, computation, status, reason,
                terminal_stop_validated=False):
    reference.set_target_world(solution["requested_world_mm"])
    error = float(reference.metrics()["tip_target_error_um"])
    passed = status == "PASS" and error <= navigation_settings(reference)["endpoint_tolerance_um"]
    return {"waypoint_id": solution["id"], "result": "PASS" if passed else "FAIL",
            "reason": reason, "tip_error_um": error, "motion_time_s": float(duration),
            "algorithm_wall_s": float(computation),
            "terminal_stop_validated": bool(terminal_stop_validated),
            "success_with_stop": bool(passed and terminal_stop_validated)}


def perception_record(reference, observation, phase):
    """Audit only: keep truth, memory and current observations distinguishable."""
    geometry = observation.get("geometry", {})
    tracking = observation.get("tracking", {})
    target = tracking.get("used_target_world_mm", geometry.get("target_world_mm"))
    tip = geometry.get("tip_world_mm")
    estimated = tracking.get("estimated_position_error_um")
    if estimated is None and target is not None and tip is not None:
        estimated = float(np.linalg.norm(np.asarray(target)-tip)*1000.)
    observed = (observation.get("target_pixel") is not None
                and geometry.get("target_world_mm") is not None)
    return {
        "target_source": tracking.get("target_source", "observed" if observed else "unavailable"),
        "tip_source": geometry.get("tip_quality", {}).get("source", "unavailable"),
        "controller_state": tracking.get("controller_state", "model_"+phase),
        "cached_target_age_s": tracking.get("cached_target_age_s"),
        "estimated_position_error_um": estimated,
        "true_position_error_um": float(reference.metrics()["tip_target_error_um"]),
        "target_observed": bool(tracking.get("target_observed", observed)),
        "control_frame_index": observation.get("experiment_frame_index"),
        "occlusion_event_id": observation.get("occlusion_intervention", {}).get("event_id") or "unknown",
    }


def guard_scene_goal_inverse(reference):
    """Fail a strict run if control asks for a privileged scene-goal inverse.

    Forward kinematics and measured-point actuation Jacobians remain available.
    Scene construction precedes this guard; truth metrics stay evaluation-only.
    """
    nominal = reference.nominal

    class ForwardOnlyNominal:
        def __getattr__(self, name):
            if name == "inverse":
                raise RuntimeError("SCENE_GOAL_INVERSE_FORBIDDEN_IN_STRICT_VISUAL_CONTROL")
            return getattr(nominal, name)

    def forbidden_target_task(*args, **kwargs):
        raise RuntimeError("SCENE_TARGET_TASK_FORBIDDEN_IN_STRICT_VISUAL_CONTROL")

    reference.nominal = ForwardOnlyNominal()
    reference.target_task = forbidden_target_task


def run_visual(reference, solutions, directory, writer, strict_visual=False):
    frames, results, details = [], [], []
    elapsed, computation = 0., 0.
    velocity = np.zeros(3)
    max_velocity, max_acceleration = np.zeros(3), np.zeros(3)
    pending = None
    dt = float(reference.config["navigation"]["visual_servo"]["sample_period_s"])
    settings = navigation_settings(reference)
    servo_settings = settings.get("visual_servo", {})
    geometry_feedback = servo_settings.get("position_feedback", "geometry") == "geometry"
    feature_calibration = bool(not geometry_feedback and
                               servo_settings.get("use_target_pose_calibration", True))
    # Strict visual collection does not construct an analytic retract/transfer
    # trajectory from the scene target pose. Current camera measurements drive
    # the error; fixed calibration and the actuation Jacobian remain available.
    policy = None if strict_visual else settings.get("fundus_transition")
    fixture = None
    if reference.config.get("reference_scene", {}).get("occlusion_fixture", {}).get("enabled", False):
        from rcm_occlusion_fixture import PhysicalOcclusionFixture
        fixture = PhysicalOcclusionFixture(reference)
    if strict_visual:
        from rcm_multitarget_visual import run_strict_visual
        strict_settings = reference.config["navigation"]["strict_visual_navigation"]
        guard_scene_goal_inverse(reference)
        return run_strict_visual(reference, solutions, directory, writer, fixture=fixture,
                                 retract_mm=strict_settings["retract_mm"],
                                 minimum_transfer_clearance_mm=strict_settings["minimum_transfer_clearance_mm"])
    transition_details = []
    for waypoint_index, solution in enumerate(solutions):
        reference.set_target_world(solution["requested_world_mm"])
        if fixture is not None:
            fixture.hide()
        print(json.dumps({"waypoint": solution["id"], "event": "begin",
                          "dataset_frames": len(writer.records)}, ensure_ascii=False), flush=True)
        retreat_duration = 0.
        if policy:
            if np.max(np.abs(velocity)) > 1e-12:
                raise ValueError("Cannot begin fundus retraction before stopping")
            start = np.asarray(task_user_units(reference.current_task()))
            target_standoff = standoff_task(solution["requested_task_A_deg_B_deg_C_mm"], policy["retract_mm"])
            stages = []
            if waypoint_index:
                retract_end = standoff_task(start, policy["retract_mm"])
                stages.append(("retract", start, retract_end))
                start = retract_end
            # Once retracted, rotate inside the eye to the next target's
            # standoff pose.  The final visual-servo phase below changes only C.
            if not np.allclose(start, target_standoff, atol=1e-12, rtol=0):
                stages.append(("transfer", start, target_standoff))
            sensor = VisualServoSensor(reference)
            try:
                for phase_name, stage_start, stage_end in stages:
                    if not no_surface_slide_edge(reference, stage_start, stage_end,
                                                 policy["minimum_transfer_clearance_mm"]):
                        raise ValueError("Invalid fundus transfer/retraction")
                    motion = TrapezoidMotion(stage_start, stage_end, settings)
                    stage_duration = motion.duration_s
                    sample_times = [0., *np.arange(dt, stage_duration, dt), stage_duration]
                    states = [motion.sample(t)[0] for t in sample_times]
                    for sample_index, (t, q) in enumerate(zip(sample_times, states)):
                        append_frame(frames, elapsed+t, q, solution["id"], phase_name)
                        reference.set_task(task_si_units(q))
                        if sample_index < len(states)-1:
                            observation = sensor.observe()
                            visibility = sensor.visibility_metrics(observation["segmentation"])
                            writer.add(elapsed+t, q, states[sample_index+1], solution["id"],
                                       observation["rgb"], phase=phase_name, visibility=visibility,
                                       perception=perception_record(reference, observation, phase_name))
                    scale_si = np.array([math.pi/180, math.pi/180, .001])
                    max_velocity = np.maximum(max_velocity, motion.speed*np.abs(motion.delta)*scale_si)
                    max_acceleration = np.maximum(max_acceleration, motion.acceleration*np.abs(motion.delta)*scale_si)
                    transition_details.append({"waypoint_id": solution["id"], "phase": phase_name,
                                               "start_task": stage_start.tolist(), "end_task": stage_end.tolist(),
                                               "duration_s": stage_duration})
                    elapsed += stage_duration
                    retreat_duration += stage_duration
            finally:
                sensor.close()
        if strict_visual:
            def validator(a, b):
                start, end = np.asarray(task_user_units(a)), np.asarray(task_user_units(b))
                count = max(1, int(math.ceil(np.max(np.abs(end-start)/[.1, .1, .02]))))
                return all(clearance_lower_bound_mm(reference, q) >= -1e-8
                           for q in np.linspace(start, end, count+1))
        else:
            validator = None if not policy else lambda a, b: no_surface_slide_edge(
                reference, task_user_units(a), task_user_units(b), policy["minimum_transfer_clearance_mm"])
        pending = None
        duration = 0.
        wall = 0.
        report = {"result": "FAIL", "reason": "VISUAL_SERVO_NOT_RUN"}
        # At a fundus standoff, first align the needle in the image while C is
        # held fixed, then approach along C using the depth channel.  This
        # prevents perspective-induced A/B corrections from becoming an
        # illegal near-wall slide during the final approach.
        controller_phases = ([
            ("visual_approach", (False, False, True), (False, False, True)),
        ] if policy else [("visual_servo", (True, True, True), (True, True, True))])
        for phase_label, control_axes, completion_axes in controller_phases:
            phase_elapsed = elapsed
            controller = VisualServoController(
                reference, output_dir=directory/solution["id"]/phase_label,
                initial_velocity=velocity, command_validator=validator,
                control_axes=control_axes, completion_axes=completion_axes)
            controller.sensor.observation_time_origin_s = phase_elapsed
            if fixture is not None:
                controller.observation_hook = lambda iteration, state: fixture.update(
                    solution["id"], phase_label, iteration)

            def capture(iteration, observation, phase_label=phase_label, phase_elapsed=phase_elapsed):
                nonlocal pending
                q = task_user_units(reference.current_task())
                visibility = controller.sensor.visibility_metrics(observation["segmentation"])
                if pending is not None:
                    writer.add(pending[0], pending[1], q, solution["id"], pending[2], phase=pending[3],
                               visibility=pending[4], perception=pending[5])
                raw_time = phase_elapsed+iteration*dt
                # The first observation of a new controller phase shares the
                # previous phase's terminal instant.  Shift only the pending
                # dataset timestamp by machine epsilon so each action remains
                # strictly ordered without changing the simulation time.
                record_time = max(raw_time, writer.records[-1]["timestamp"]+1e-9) if writer.records else raw_time
                pending = (record_time, q, observation["rgb"].copy(), phase_label, visibility,
                           perception_record(reference, observation, phase_label))
                if iteration % 100 == 0:
                    print(json.dumps({"waypoint": solution["id"], "iteration": iteration,
                                      "perception": pending[5]}, ensure_ascii=False), flush=True)

            controller.iteration_callback = capture
            started = time.perf_counter()
            try:
                report = controller.run()
            except (RuntimeError, ValueError) as error:
                controller.sensor.close()
                report = {"result": "FAIL", "reason": str(error)}
            phase_wall = time.perf_counter()-started
            wall += phase_wall
            computation += phase_wall
            controller.write(report, directory/solution["id"]/phase_label)
            details.append({**report, "phase": phase_label})
            max_velocity = np.maximum(max_velocity, controller.max_velocity)
            max_acceleration = np.maximum(max_acceleration, controller.max_acceleration)
            for row in controller.rows:
                append_frame(frames, phase_elapsed+row["time_s"],
                             [row["A_deg"], row["B_deg"], row["C_mm"]], solution["id"], phase_label)
            phase_duration = controller.rows[-1]["time_s"] if controller.rows else 0.
            final_q = np.asarray(task_user_units(reference.current_task()))
            # A MAX_ITERATIONS exit may follow an applied command after the last observation.
            if controller.rows and not np.allclose(final_q, frames[-1]["q"], rtol=0, atol=1e-10):
                if pending is not None:
                    writer.add(pending[0], pending[1], final_q, solution["id"], pending[2], phase=pending[3],
                               visibility=pending[4], perception=pending[5])
                    pending = None
                phase_duration += dt
                append_frame(frames, phase_elapsed+phase_duration, final_q, solution["id"], phase_label)
            elapsed += phase_duration
            duration += phase_duration
            velocity = controller.final_velocity.copy()
            if report["result"] != "PASS":
                break
        failure_hold_validated = False
        failure_hold_steps = 0
        if report["result"] != "PASS":
            # A visibility/safety failure ends this episode at its current pose.
            # If the last accepted visual command still carries velocity, ramp
            # that command to zero while keeping every intermediate pose inside
            # the eye. No target feedback or target pose is used during braking.
            q = reference.current_task().copy()
            if pending is not None:
                # Close the last feedback observation before emitting brake
                # observations; otherwise the first failure-brake timestamp
                # can collide with the pending visual sample.
                pending_time = max(float(pending[0]),
                                   writer.records[-1]["timestamp"] + 1e-9
                                   if writer.records else float(pending[0]))
                writer.add(pending_time, pending[1], task_user_units(q), solution["id"],
                           pending[2], phase=pending[3], visibility=pending[4], perception=pending[5])
                pending = None
            acceleration = (np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"], dtype=float)
                            * np.array([math.pi/180, math.pi/180, .001]))
            sensor = VisualServoSensor(reference)
            failure_hold_validated = True
            try:
                while np.max(np.abs(velocity)) > 1e-12:
                    next_velocity = np.sign(velocity)*np.maximum(np.abs(velocity)-acceleration*dt, 0.)
                    candidate = q+next_velocity*dt
                    try:
                        candidate = reference.validate_task(candidate).copy()
                    except ValueError:
                        failure_hold_validated = False
                        break
                    if validator is not None and not validator(q, candidate):
                        failure_hold_validated = False
                        break
                    reference.set_task(q)
                    observation = sensor.observe()
                    visibility = sensor.visibility_metrics(observation["segmentation"])
                    q_user = task_user_units(q)
                    next_user = task_user_units(candidate)
                    record_time = max(float(elapsed),
                                      writer.records[-1]["timestamp"] + 1e-9
                                      if writer.records else float(elapsed))
                    writer.add(record_time, q_user, next_user, solution["id"], observation["rgb"],
                               phase="failure_brake", visibility=visibility,
                               perception=perception_record(reference, observation, "failure_brake"))
                    elapsed += dt
                    append_frame(frames, elapsed, task_user_units(candidate), solution["id"], "failure_brake")
                    q, velocity = candidate, next_velocity
                    failure_hold_steps += 1
                reference.set_task(q)
                failure_hold_validated = bool(
                    failure_hold_validated and np.max(np.abs(velocity)) <= 1e-12)
                if failure_hold_validated:
                    duration += failure_hold_steps*dt
            finally:
                sensor.close()
        results.append(goal_result(reference, solution, duration+retreat_duration, wall,
                                   report["result"], report["reason"],
                                   report.get("terminal_stop_validated", False)))
        results[-1]["failure_hold_validated"] = failure_hold_validated
        results[-1]["failure_hold_steps"] = failure_hold_steps
        print(json.dumps({"waypoint": solution["id"], "event": "end", "result": results[-1]},
                         ensure_ascii=False), flush=True)
        if results[-1]["result"] != "PASS":
            break
    if pending is not None:
        # A phase can end immediately after its last observation.  Keep the
        # terminal hold strictly after the last emitted action even when the
        # controller stopped on that same timestamp.
        terminal_time = max(float(pending[0]),
                            (writer.records[-1]["timestamp"] + dt) if writer.records else float(pending[0]))
        writer.add(terminal_time, pending[1], pending[1], results[-1]["waypoint_id"], pending[2], terminal=True,
                   phase="hold", visibility=pending[4], perception=pending[5])
    else:
        sensor = VisualServoSensor(reference)
        try:
            q = task_user_units(reference.current_task())
            append_frame(frames, elapsed, q, solutions[len(results)-1]["id"], "hold")
            observation = sensor.observe()
            visibility = sensor.visibility_metrics(observation["segmentation"])
            writer.add(elapsed, q, q, solutions[len(results)-1]["id"], observation["rgb"], terminal=True,
                       phase="hold", visibility=visibility,
                       perception=perception_record(reference, observation, "hold"))
        finally:
            sensor.close()
    if fixture is not None:
        fixture.hide()
    scale = np.array([180/math.pi, 180/math.pi, 1000.])
    return frames, results, {"algorithm_wall_s": computation,
        "execution_mode": ("strict_current_frame_visual_servo" if strict_visual else
                           "hybrid_scene_transfer_and_visual_depth_approach" if policy else "calibrated_visual_servo"),
        "known_target_pose_used_for_feature_calibration": feature_calibration,
        "occlusion_handling": servo_settings.get("occlusion_handling", "stop"),
        "controlled_occlusion_fixture": None if fixture is None else fixture.audit,
        "ground_truth_used_for_fundus_transfer": bool(policy),
        "visual_feedback_scope": ("current-frame RGB-D feedback throughout; no target-pose path or analytic transfer"
                                  if strict_visual else
                                  "C-only final approach; retract/transfer are analytic trajectories with RGB recorded but not used as feedback"
                                  if policy else
                                  "calibrated visible-geometry 3D feedback" if geometry_feedback else "legacy image/depth feedback"),
        "eye_containment_guard_enabled": bool(strict_visual),
        "max_velocity": (max_velocity*scale).tolist(),
        "max_acceleration": (max_acceleration*scale).tolist(),
        "terminal_velocity": (velocity*scale).tolist(),
        "terminal_stop_validated": bool(
            details and all(d.get("terminal_stop_validated", False) for d in details)
            or results and results[-1].get("failure_hold_validated", False)),
        "terminal_stop_scope": "validated zero command at the goal on success; bounded deceleration to a stationary failure hold on episode failure",
        "failure_hold_validated": bool(results and results[-1].get("failure_hold_validated", False)),
        "failure_hold_steps": int(results[-1].get("failure_hold_steps", 0)) if results else 0,
        "control_details": details, "transition_details": transition_details,
        "fundus_command_guard_enabled": bool(policy)}


def run_planner(method, reference, solutions, directory, writer, seed, space):
    rng = np.random.default_rng(seed)
    settings = navigation_settings(reference)
    start = np.asarray(task_user_units(reference.current_task()))
    goals = [np.asarray(s["requested_task_A_deg_B_deg_C_mm"]) for s in solutions]
    policy = settings.get("fundus_transition")
    transfer_goals = [standoff_task(q, policy["retract_mm"]) for q in goals] if policy else goals
    if policy:
        space.minimum_clearance_mm = policy["minimum_transfer_clearance_mm"]
    frames, results, paths = [], [], []
    elapsed, computation = 0., 0.
    max_velocity, max_acceleration = np.zeros(3), np.zeros(3)
    roadmap, build_wall = None, 0.
    if method == "PRM":
        started = time.perf_counter()
        roadmap = PRM(space, [start, *transfer_goals], rng)
        build_wall = time.perf_counter()-started
        computation += build_wall
    for index, (solution, goal) in enumerate(zip(solutions, goals)):
        started = time.perf_counter()
        transfer_start = standoff_task(start, policy["retract_mm"]) if policy and index else start
        transfer_goal = transfer_goals[index]
        if method == "RRT":
            path, info = rrt(space, transfer_start, transfer_goal, rng)
        else:
            path, info = roadmap.query(index, index+1), roadmap.info
        raw_path = [np.asarray(q).copy() for q in path]
        if settings.get("waypoint_trajectory") == "trapezoid":
            path = shortcut_path(space, path)
        phases = []
        if policy and index:
            phases.append(("retract", [start, transfer_start]))
        phases.append(("transfer" if policy else "motion", path))
        if policy:
            phases.append(("approach", [transfer_goal, goal]))
            for phase, phase_path in phases:
                if not all(no_surface_slide_edge(reference, a, b, policy["minimum_transfer_clearance_mm"])
                           for a, b in zip(phase_path, phase_path[1:])):
                    raise ValueError("Fundus phase violates clearance: "+phase)
        query_wall = time.perf_counter()-started
        computation += query_wall
        leg_start = elapsed
        edges = [(phase, left, right) for phase, points in phases for left, right in zip(points, points[1:])]
        for phase, left, right in edges:
            append_frame(frames, elapsed, left, solution["id"], phase)
            trapezoid = TrapezoidMotion(left, right, settings) if settings.get("waypoint_trajectory") == "trapezoid" else None
            duration = trapezoid.duration_s if trapezoid is not None else move_duration(left, right, settings)
            if duration == 0:
                continue
            segment = Segment(elapsed, duration, "planned_motion", index,
                              task_si_units(left), task_si_units(right))
            delta = np.abs(np.asarray(right)-left)
            max_velocity = np.maximum(max_velocity, trapezoid.speed*delta if trapezoid is not None else 1.875*delta/duration)
            max_acceleration = np.maximum(max_acceleration, trapezoid.acceleration*delta if trapezoid is not None else (10*math.sqrt(3)/3)*delta/duration**2)
            # RGB/state/action recording uses the configured nominal control period.
            dt = float(settings["visual_servo"]["sample_period_s"])
            for local in [*np.arange(dt, duration, dt), duration]:
                if trapezoid is None:
                    q, _, _ = segment.sample(elapsed+float(local))
                    user_q = task_user_units(q)
                else:
                    user_q, _, _ = trapezoid.sample(float(local))
                append_frame(frames, elapsed+float(local), user_q, solution["id"], phase)
            elapsed += duration
        reference.set_task(task_si_units(goal))
        results.append(goal_result(reference, solution, elapsed-leg_start,
                                   query_wall+(build_wall if index == 0 else 0.), "PASS",
                                   "PLANNED_TARGET_REACHED", terminal_stop_validated=True))
        full_path = [edges[0][1], *[edge[2] for edge in edges]]
        paths.append({"waypoint_id": solution["id"], "path_A_deg_B_deg_C_mm": [np.asarray(q).tolist() for q in full_path],
                      "raw_path_A_deg_B_deg_C_mm": [q.tolist() for q in raw_path],
                      "raw_path_scope": "transfer only; mandatory retract/approach are separate" if policy else "whole leg",
                      "phases": [{"phase": phase, "path_A_deg_B_deg_C_mm": [np.asarray(q).tolist() for q in points]}
                                 for phase, points in phases],
                      "search": info})
        start = goal
    write_json(directory/"paths.json", paths)
    sensor = VisualServoSensor(reference)
    by_id = {s["id"]: s for s in solutions}
    try:
        for index, frame in enumerate(frames):
            reference.set_target_world(by_id[frame["waypoint_id"]]["requested_world_mm"])
            reference.set_task(task_si_units(frame["q"]))
            action = frames[min(index+1, len(frames)-1)]["q"]
            observation = sensor.observe()
            visibility = sensor.visibility_metrics(observation["segmentation"])
            writer.add(frame["time_s"], frame["q"], action, frame["waypoint_id"],
                       observation["rgb"], terminal=index == len(frames)-1,
                       phase=frame["phase"] if index < len(frames)-1 else "hold", visibility=visibility)
            reference.set_task(task_si_units(action))
    finally:
        sensor.close()
    return frames, results, {"algorithm_wall_s": computation, "roadmap_build_wall_s": build_wall,
        "max_velocity": max_velocity.tolist(), "max_acceleration": max_acceleration.tolist(),
        "terminal_velocity": [0., 0., 0.], "terminal_stop_validated": True,
        "validity_checks": space.validity_checks,
        "path_postprocessing": ("validated shortcutting; synchronized trapezoidal motion with zero endpoint velocity"
                                if settings.get("waypoint_trajectory") == "trapezoid" else "none; quintic rest-to-rest interpolation on each graph/tree edge")}


def evaluate(reference, frames, solutions, directory):
    """Measure actual MuJoCo geometry along the issued position-command path."""
    by_id = {s["id"]: s for s in solutions}
    rows, previous_tip, previous_q = [], None, None
    tip_path_mm, joint_path = 0., np.zeros(3)
    max_geometry = {key: 0. for key in (
        "rcm_drift_um", "needle_axis_rcm_distance_um", "fk_agreement_um",
        "upper_link_closure_um", "lower_link_closure_um")}
    outside, proximal = 0, 0.
    navigation = navigation_settings(reference)
    policy = navigation.get("fundus_transition") or navigation.get("strict_visual_navigation")
    transition_violations = 0
    minimum_transfer_clearance = math.inf

    def measure(timestamp, q, waypoint, phase):
        nonlocal previous_tip, previous_q, tip_path_mm, outside, proximal, transition_violations, minimum_transfer_clearance
        reference.set_target_world(by_id[waypoint]["requested_world_mm"])
        reference.set_task(task_si_units(q))
        metrics = reference.metrics()
        tip = np.asarray(metrics["tip_world_mm"])
        if previous_tip is not None:
            tip_path_mm += float(np.linalg.norm(tip-previous_tip))
            joint_path[:] += np.abs(np.asarray(q)-previous_q)
        previous_tip, previous_q = tip, np.asarray(q)
        for key in max_geometry:
            max_geometry[key] = max(max_geometry[key], metrics[key])
        geometry = metrics["needle_eye_geometry"]
        outside += int(not geometry["tip_inside_eye"])
        proximal = max(proximal, geometry["proximal_centerline_inside_mm"])
        clearance = clearance_lower_bound_mm(reference, q)
        if policy and phase in ("transfer", "visual_align"):
            minimum_transfer_clearance = min(minimum_transfer_clearance, clearance)
            transition_violations += int(clearance < policy["minimum_transfer_clearance_mm"]-1e-8)
        rows.append({"time_s": timestamp, "waypoint_id": waypoint, "phase": phase,
                     "A_deg": q[0], "B_deg": q[1], "C_mm": q[2],
                     "tip_world_x_mm": tip[0], "tip_world_y_mm": tip[1], "tip_world_z_mm": tip[2],
                     "tip_target_error_um": metrics["tip_target_error_um"],
                     "rcm_drift_um": metrics["rcm_drift_um"],
                     "needle_axis_rcm_distance_um": metrics["needle_axis_rcm_distance_um"],
                     "tip_inside_eye": geometry["tip_inside_eye"],
                     "wall_clearance_lower_bound_mm": clearance,
                     "proximal_centerline_inside_mm": geometry["proximal_centerline_inside_mm"]})

    if frames:
        measure(frames[0]["time_s"], frames[0]["q"], frames[0]["waypoint_id"], frames[0]["phase"])
    for left, right in zip(frames, frames[1:]):
        duration = right["time_s"]-left["time_s"]
        count = max(1, math.ceil(duration/.02-1e-9))
        if policy:
            transition_violations += int(not no_surface_slide_edge(
                reference, left["q"], right["q"], policy["minimum_transfer_clearance_mm"]))
            if left["phase"] in ("retract", "approach", "visual_retract", "visual_approach"):
                transition_violations += int(np.max(np.abs(right["q"][:2]-left["q"][:2])) > 1e-9)
        for i in range(1, count+1):
            fraction = i/count
            measure(left["time_s"]+fraction*duration,
                    left["q"]+fraction*(right["q"]-left["q"]), left["waypoint_id"], left["phase"])
    write_csv(directory/"trajectory.csv", rows)
    return {"tip_path_length_mm": tip_path_mm,
            "joint_path_length_A_deg_B_deg_C_mm": joint_path.tolist(),
            "max_geometry_errors_um": max_geometry, "tip_outside_eye_samples": outside,
            "max_proximal_centerline_inside_mm": proximal, "validation_samples": len(rows),
            "motion_time_s": frames[-1]["time_s"] if frames else 0.,
            "fundus_transition_constraints_pass": transition_violations == 0,
            "fundus_transition_violation_count": transition_violations,
            "minimum_transfer_clearance_lower_bound_mm": minimum_transfer_clearance if math.isfinite(minimum_transfer_clearance) else None}


def summarize_stop_constraints(waypoints, method_info, requested_target_count):
    attempted = [row for row in waypoints if row.get("result") != "NOT_RUN"]
    stopped = [row for row in attempted if row.get("terminal_stop_validated", False)]
    terminal_velocity = np.asarray(method_info.get("terminal_velocity", [math.inf]*3), dtype=float)
    terminal_velocity_zero = bool(terminal_velocity.shape == (3,) and np.isfinite(terminal_velocity).all()
                                  and np.max(np.abs(terminal_velocity)) <= 1e-9)
    all_targets_reached_and_stopped = bool(
        len(waypoints) == requested_target_count and len(attempted) == requested_target_count
        and all(row.get("result") == "PASS" and row.get("terminal_stop_validated", False)
                for row in waypoints))
    passed = bool(all_targets_reached_and_stopped
                  and method_info.get("terminal_stop_validated", False)
                  and terminal_velocity_zero)
    return {
        "required_at_every_target": True,
        "requested_targets": int(requested_target_count),
        "attempted_targets": len(attempted),
        "not_attempted_targets": len(waypoints)-len(attempted),
        "targets_with_validated_stop": len(stopped),
        "target_stop_failures": len(attempted)-len(stopped),
        "terminal_velocity_A_deg_B_deg_C_mm_per_s": terminal_velocity.tolist(),
        "terminal_velocity_zero": terminal_velocity_zero,
        "all_targets_reached_and_stopped": all_targets_reached_and_stopped,
        "pass": passed,
        "stop_validation_source": "per-target controller stop reports or rest-to-rest planner endpoints; final command velocity must be zero",
    }


def write_comparison(directory, reports, solutions, order_report):
    protocol = json.loads((directory/"experiment_protocol.json").read_text(encoding="utf-8"))
    precision = protocol["profile"] == "precision"
    geometry_feedback = protocol.get("visual_servo_position_feedback", "image_depth") == "geometry"
    strict_visual = protocol.get("guidance_mode") in (
        "strict_current_frame_visual_servo", "strict_visual_multitarget")
    aggregate = []
    for method in dict.fromkeys(report["method"] for report in reports):
        trials = [r for r in reports if r["method"] == method]
        passed = [r for r in trials if r["result"] == "PASS"]
        row = {"method": method, "successful_episodes": len(passed), "total_episodes": len(trials)}
        for field in ("motion_time_s", "algorithm_wall_s", "tip_path_length_mm", "max_tip_error_um"):
            values = [r[field] for r in passed]
            row[field+"_mean"] = float(np.mean(values)) if values else None
            row[field+"_std"] = float(np.std(values)) if values else None
        row["max_rcm_drift_um"] = max((r["max_geometry_errors_um"]["rcm_drift_um"] for r in passed), default=None)
        for condition in ("visible", "needle_occluded", "other_visibility_issue", "unknown"):
            attempted = sum(r.get("target_success_by_visibility", {}).get("condition_exposure", {})
                            .get(condition, {}).get("target_visits", 0) for r in trials)
            successes = sum(r.get("target_success_by_visibility", {}).get("condition_exposure", {})
                            .get(condition, {}).get("successful_target_visits", 0) for r in trials)
            row[f"{condition}_successful_target_visits"] = successes
            row[f"{condition}_attempted_target_visits"] = attempted
            row[f"{condition}_success_rate"] = successes/attempted if attempted else None
        row["stop_validated_target_visits"] = sum(
            r.get("stop_constraint_summary", {}).get("targets_with_validated_stop", 0) for r in trials)
        row["attempted_target_visits"] = sum(
            r.get("stop_constraint_summary", {}).get("attempted_targets", 0) for r in trials)
        aggregate.append(row)
    write_csv(directory/"comparison.csv", aggregate)
    waypoint_rows = []
    for report in reports:
        for row in report["waypoints"]:
            waypoint_rows.append({"method": report["method"], "seed": report["seed"],
                                  "episode_index": report["episode_index"], **row})
    write_csv(directory/"waypoint_results.csv", waypoint_rows)
    lines = [f"# {len(solutions)} 个眼底目标访问与导航对比", "", f"输入为 {len(solutions)} 个仿真目标；从记录起点 {protocol.get('start_task_A_deg_B_deg_C_mm', [0., 0., 0.])} 出发，访问全部目标后结束，不回起点。目标逐个呈现。", "",
              ("给定输入顺序：" if strict_visual else "脚本选择并由各方法共享的访问顺序：")+(" → ".join(order_report["selected_order"]) or "未生成：视觉建图不完整，机器人未运动"), "",
             ("严格视觉按给定顺序访问，不枚举目标逆解路径；回撤、对准、接近目标均由视觉测量构造。" if strict_visual else
              ("顺序代价：关节限速/限加速度下，同步梯形速度轨迹的预计运动时间之和。" if precision else "顺序代价：关节限速/限加速度下，五次插值的预计运动时间之和。")+
              (f"枚举全部 {math.factorial(len(solutions))} 种顺序，三种方法共享选择结果。它不是对视觉/RRT/PRM 实际路径时间的全局最优证明."
               if len(solutions) <= 8 else "目标数超过 8 时使用确定性的最近目标启发式排序，三种方法共享选择结果。")), "",
             "| 路径点 | 机器人坐标 (mm) |", "|---|---|"]
    lines += [f"| {s['id']} | {s['requested_robot_mm']} |" for s in solutions]
    lines += ["", "| 候选顺序 | 预计时间 (s) |", "|---|---:|"]
    lines += [f"| {' → '.join(c['order'])} | {c['estimated_motion_time_s']:.4f} |" for c in order_report["candidates"]]
    lines += ["", "汇总指标只对完整 PASS 的轨迹统计，± 为总体标准差；视觉伺服是确定性的一次运行。", "",
              "| 方法 | 成功轨迹 | 仿真运动时间 (s) | 算法计算墙钟时间 (s) | 针尖路径长度 (mm) | 每轨迹最大到达误差 (μm) |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        values = []
        for field in ("motion_time_s", "algorithm_wall_s", "tip_path_length_mm", "max_tip_error_um"):
            mean, std = row[field+"_mean"], row[field+"_std"]
            values.append("—" if mean is None else f"{mean:.5g} ± {std:.3g}")
        lines.append(f"| {row['method']} | {row['successful_episodes']}/{row['total_episodes']} | "+" | ".join(values)+" |")
    lines += ["", "| 方法 | seed | 路径点 | 结果 | 到达误差 (μm) | 分段运动时间 (s) |", "|---|---:|---|---|---:|---:|"]
    for row in waypoint_rows:
        error = "—" if row["tip_error_um"] is None else f"{row['tip_error_um']:.6g}"
        lines.append(f"| {row['method']} | {row['seed']} | {row['waypoint_id']} | {row['result']} | {error} | {row['motion_time_s']:.4f} |")
    lines += ["", "## 比较口径", "",
              f"- 所有方法使用相同初始姿态、{len(solutions)} 个目标、访问顺序和 A/B/C 速度/加速度参数。",
              ("- 几何视觉伺服根据当前RGB-D和固定相机/RCM/针具尺寸重建物理针尖，目标切换时传递上一段指令速度；不使用已知目标姿态进行特征标定。"
               if geometry_feedback else "- 历史image_depth视觉伺服使用图像/深度特征控制；是否进行目标姿态特征标定以该次协议的先验开关为准。"),
              "- RRT 与 PRM 是实际随机采样规划；固定随机种子。PRM 一张路线图复用于全部转移段，建图开销仅计一次。",
              "- RRT/PRM 使用连续关节坐标和精确模型目标，未施加编码器量化与感知噪声；其接近零的终点误差不代表实机精度优于视觉伺服。",
              "- 采样范围是围绕起点/目标加边距的算法搜索域，不是新设定的设备限位。检查连杆有效性及针尖位于参考眼球内；没有额外障碍物，也没有针杆体积/组织接触检查。",
              ("- RRT/PRM 先做逐边有效性检查的路径简化，再使用同步梯形速度轨迹。每个指定目标均停稳；无法简化的转角保留内部停靠。是否能直连由实际眼球约束决定，详见 paths.json。速度连续、加速度有界，未限制 jerk。"
               if precision else "- 原始 RRT/PRM 路径没有快捷化或平滑优化，每条图边使用端点静止的五次插值。"),
              "- 算法时间：视觉包含渲染、测量拟合、控制计算及采集图像写盘；历史image_depth模式还可能包含目标姿态特征标定。RRT/PRM 包含搜索/建图但不含之后的 RGB 采集，不能直接当作实时控制性能排名。",
              f"- 数据集目标采样频率为 {1./protocol['dataset_sampling_period_s']:g} Hz；规划边界和阶段切换可能产生更短的事件帧间隔，精确时间戳保存在 JSONL/NPZ。几何验证对命令路径按不超过 0.02 s 加密。",
              ("- 所有指定目标都要求停止。视觉误差连续三帧达标且指令速度为零，三维位置另以 20 μm 验收。速率指标为视觉离散指令和规划轨迹解析极值；MuJoCo 使用 mj_forward，验证的是指令制动，不是实机响应。"
               if precision else "- 速率指标：视觉为离散指令速度/加速度；规划为五次轨迹解析极值。视觉最终保持姿态未验证减速停止。"),
              "", "## 数据集", "",
              "`dataset/manifest.json` 给出字段、单位、各轨迹结果与索引；`dataset/data/*.npz` 保存 float32 状态/动作及其他列，JSONL 便于逐行查看，PNG 为原始 uint8 RGB。",
              "图像/状态在动作执行前观测；action 是下一次实际下发的绝对位置目标，不是重复填入最终路径点。视觉图像在闭环观测回调中采集，规划图像在规划轨迹执行时采集。",
              "每帧记录标靶可见像素、隐藏针杆后重渲染得到的无遮挡像素、被针遮挡比例和状态。反事实重渲染只生成离线标签，不传入导航控制；遮挡来自 MuJoCo 场景中的针体几何，不用贴片或人工遮罩伪造。",
              "", "完整数据见 `run_reports.json`、`waypoint_results.csv`、各方法目录的 `trajectory.csv` 与路径/控制报告。", ""]
    lines += ["", "## 按可见状态统计目标成功率", "",
              "成功定义为该目标到达验收通过且停针约束已验证。分母是目标访问次数，不是帧数。按一次访问期间出现过的条件分组；同一访问可同时计入可见组和针体遮挡组，例如接近过程中先可见、随后被针遮挡。目标在针体隐藏的诊断渲染中仍不可见时计入其他不可见组，无法分类时计入未知组。", "",
              "| 方法 | 条件 | 成功目标数 | 尝试目标数 | 成功率 |", "|---|---|---:|---:|---:|"]
    for row in aggregate:
        for condition, label in (("visible", "可见条件"), ("needle_occluded", "针体遮挡"),
                                 ("other_visibility_issue", "其他不可见"), ("unknown", "未知状态")):
            attempted = row[f"{condition}_attempted_target_visits"]
            successes = row[f"{condition}_successful_target_visits"]
            rate = row[f"{condition}_success_rate"]
            rate_text = "—" if rate is None else f"{rate:.1%}"
            lines.append(f"| {row['method']} | {label} | {successes} | {attempted} | {rate_text} |")
    lines += ["", "## 逐帧遮挡统计", "",
              "统计仅包含有分割观测的帧；`partial_needle` 和 `full_needle` 代表物理针体遮挡，`not_visible` 表示隐藏针体后仍未检测到标靶。", "",
              "| 方法 | seed | 已标注帧 | 未知帧 | 无遮挡 | 部分遮挡 | 完全遮挡 | 未见标靶 | 最大遮挡比例 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        summary = report.get("visibility_summary", {})
        states = summary.get("target_occlusion_state_frames", {})
        maximum = summary.get("maximum_target_occlusion_fraction_by_needle")
        maximum_text = "—" if maximum is None else f"{maximum:.1%}"
        lines.append(f"| {report['method']} | {report['seed']} | {summary.get('known_frames', 0)} | "
                     f"{summary.get('unknown_frames', 0)} | {states.get('clear', 0)} | "
                     f"{states.get('partial_needle', 0)} | {states.get('full_needle', 0)} | "
                     f"{states.get('not_visible', 0)} | {maximum_text} |")
    if strict_visual:
        start, end = lines.index("## 比较口径")+2, lines.index("## 数据集")-1
        lines[start:end] = [
            "- 严格RGB-D视觉按输入目标顺序执行；无静止目标建图、无目标逆解路径、无RRT/PRM路线。",
            "- scene目标坐标仅布置标记、绘图和独立验收；控制误差来自当前针尖和当前/有界历史目标测量。",
            "- 回撤目标由实际测得针尖沿观测针轴后退构造；固定C的A/B对准使用测量目标方向，固定A/B的C接近使用测量距离。",
            "- 固定相机/RCM/针具尺寸、执行雅可比、限速限加速度和眼壁安全检查仍可用；不是RGB-only实机系统。",
            f"- 遮挡模式：{protocol.get('occlusion_handling', 'stop')}；记忆模式仅假设静态已注册目标，当前针尖仍须观测有效。",
            "- 各阶段连续停稳；仅最终接近按独立真值<20μm验收到达，回撤/对准的中间测量收敛不当作目标到达。",
        ]
    elif protocol.get("planner_feedback") == "vision":
        start, end = lines.index("## 比较口径")+2, lines.index("## 数据集")-1
        lines[start:end] = [
            "- RRT/PRM 默认使用视觉导航：从 RGB、仿真分割和深度重建标靶中心；顺序与规划目标来自测量，不读取目标三维真值。",
            f"- 当前感知依赖固定相机内外参、标靶尺寸及仿真深度/分割，尚非真实 RGB-only 系统。逐个呈现 {len(solutions)} 个标靶完成静止建图，不代表同时识别全部病灶。",
            "- 任一目标不可见或深度拟合无效，整体建图失败，不生成访问顺序、不执行规划；不能用输入坐标补齐缺失测量。",
            "- 建图成功时使用真实 RRT/PRM 搜索；执行过程中每条运动指令依据视觉针尖位置修正。目标丢失立即终止并保持，目标明显移动要求重新建图。",
            "- 对视觉 RRT/PRM，运动学用于视觉测量到轴位置的转换及机械/眼壁约束；目标与针尖真值用于布置场景、绘图和独立验收。",
            ("- 默认几何视觉伺服使用固定相机、RCM与针具尺寸标定，不做目标姿态特征校准；眼底解析回撤/转移仍有目标先验，不宣称三种控制器拥有完全相同的信息。"
             if geometry_feedback else "- 历史image_depth视觉伺服保留可选的目标姿态特征标定；具体先验以该次协议开关为准，不宣称三种控制器拥有完全相同的信息。"),
            "- 保留回撤、离壁转向、固定方向接近约束。视觉跟踪速率按实际指令差分统计；遮挡后的立即保持不等于已验证实机制动。",
            "- 算法墙钟时间包含在线视觉处理/图像保存；与旧模型规划时间不能直接排名。未完成不计入到达精度或完整导航耗时。",
            "- 默认眼底任务的建图检查与各目标结果见 visual_target_map.json；成功建图后的逐步反馈见各方法目录 visual_feedback.json。", ""]
        if not order_report["selected_order"]:
            lines[4] = "所有方法在共同的视觉建图检查阶段停止；静止采集顺序不是机器人访问顺序。"
            lines[6] = "目标测量不完整，未枚举访问顺序。下表坐标仅为仿真场景与验收真值，不曾作为缺失测量的替代。"
    if (directory/"fundus_target_verification.json").exists():
        lines += ["## 眼底目标位置", "",
                  f"{len(solutions)} 个目标均位于编译后眼球椭球的后部曲面（局部归一化 z ≥ 0.9），并以 0.001 μm 径向偏差验证表面位置。",
                  "眼球后部曲面是当前仿真中的眼底代理，模型没有独立视网膜层。目标落在眼底不等于视觉导航已成功；FAIL 轨迹不得作为成功示范训练。",
                  "本次目标与旧版眼球内部目标不同，不生成把二者当作同任务的前后性能对比图。位置报告见 fundus_target_verification.json。", ""]
    if protocol.get("fundus_transition"):
        policy = protocol["fundus_transition"]
        lines += ["## 禁止眼底横移", "",
                  f"点间先保持 A/B 不变沿 C 轴回撤 {policy['retract_mm']:g} mm，再在眼内转向，最后保持 A/B 不变沿 C 轴接近目标。",
                  f"转向时针尖到眼球表面的保守距离下界必须至少 {policy['minimum_transfer_clearance_mm']:g} mm。该下界不是针杆间隙，参数也不是临床安全距离。",
                  "访问顺序的代价包括回撤、眼内转移和再次接近。RRT/PRM 只规划有余量的转移段，路径简化不能跨过必需的回撤/接近阶段。",
                  "视觉伺服在成功且停稳后先回撤，并在每条新指令下发前拒绝近壁转向或出眼动作；该约束不解决目标遮挡，失败时不进入下一目标。",
                  "dataset 的 phase 字段和 trajectory.csv 标出动作阶段。独立验证会检查回撤/接近的 A/B 固定及转向时的针尖离壁余量。", ""]
    regime = protocol.get("task_regime")
    lines += ["## 研究任务与可得信息", "",
              f"策略任务模式：`{regime or 'legacy_unspecified'}`。名义采样频率 {1./protocol['dataset_sampling_period_s']:g} Hz。",
              ("任务文本含全部目标的机器人坐标先验；这些数值来自仿真输入，并非实测或已验证配准的 OCT。"
               if regime == "registered_coordinate_prior" else
               "任务文本不含目标坐标、目标 ID、访问顺序或随机种子，只要求导航到当前呈现的标靶。"
               if regime == "visual_current_target" else
               "此历史结果未声明策略输入模式，不能自动归入无坐标视觉实验。"), "",
              ("专家和学习策略的信息必须分开：几何视觉专家使用渲染深度、分割、固定相机/RCM和针具尺寸；不进行已知目标姿态特征校准。"
               if geometry_feedback else "专家和学习策略的信息必须分开：image_depth专家使用渲染深度、分割和标定参数，且可按协议开关使用已知目标姿态特征校准。"),
              ("本次strict-visual使用当前图像/深度反馈，不生成已知目标轴路径或解析回撤/转移；仍依赖标定信息和仿真分割/深度，不是实机RGB-only导航。"
               if strict_visual else
               "本次眼底视觉伺服在转移阶段根据已知目标轴位置构造离壁姿态，回撤/转移按解析轨迹执行；该阶段RGB只记录，最后C轴接近才使用视觉反馈，是混合先验流程。"
               if protocol.get("fundus_transition") else
               "本次无解析眼底转移策略的视觉接近使用图像/深度反馈；具体测量模式和目标先验以协议记录为准。"),
              "逐个呈现标靶后由脚本选定顺序，所有方法共享；这不证明学习模型能同时识别多个病灶或自主排序。", "",
              "训练输入仅取 `dataset/policy_inputs/*.npz`（RGB 路径、轴状态、任务文本）；路径须解码成像素，不能将文件名当作特征。",
              "`dataset/policy_inputs_no_image/*.npz` 只保留状态和任务；`dataset/action_labels/*.npz` 是逐行对齐的动作监督。完整 `dataset/data/`、坐标文件、phase、visibility、时间/帧索引及访问顺序仅供审计，不得直接拼入策略输入。", "",
              "去图像消融目前只完成输入导出，仓库没有策略训练/推理流程，尚无消融性能结果。须按目标集合划分训练/验证/测试（同一集合各方法不得跨 split），比较完整输入与无图像模型，并做零图像/打乱图像推理对照；报告成功率、到达误差、约束违规和遮挡分层表现。",
              "去掉坐标文本不能消除轴状态、固定相机、标靶外观和专家轨迹带来的捷径；也不能据此证明单目深度估计或完全遮挡恢复能力。", ""]
    (directory/"comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE/"examples"/"three_waypoints.json")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE/"runs"/"历史记录"/"04_sessions"/"waypoint_comparison"/f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--methods", nargs="+", choices=["visual_servo", "RRT", "PRM"], default=["visual_servo", "RRT", "PRM"])
    parser.add_argument("--profile", choices=["precision", "legacy"], default="precision")
    parser.add_argument("--start-task", type=float, nargs=3, metavar=("A_DEG", "B_DEG", "C_MM"),
                        help="Optional episode reset pose in task units (degrees, degrees, millimeters)")
    parser.add_argument("--strict-visual", action="store_true",
                        help="Use measured RGB-D targets/tips for multi-target retract, align and approach; no scene-goal path or inverse-kinematic goal")
    parser.add_argument("--sampling-frequency-hz", type=float, default=20.0,
                        help="Nominal RGB/state/action dataset rate (default: 20 Hz)")
    parser.add_argument("--task-regime", choices=TASK_REGIMES,
                        default="visual_current_target",
                        help="Declare whether simulated registered coordinates are policy inputs or only the current visible target is given")
    parser.add_argument("--planner-feedback", choices=["vision", "model"], default="vision",
                        help="Default: RGB-D target mapping and visual feedback; model reproduces historical oracle baseline")
    parser.add_argument("--retract-mm", type=float, default=2., help="Simulation axial retraction before fundus transfer")
    parser.add_argument("--transfer-clearance-mm", type=float, default=1., help="Minimum conservative tip-wall clearance during turning")
    parser.add_argument("--baseline-dir", type=Path, default=HERE/"runs"/"历史记录"/"02_baselines"/"three_waypoint_dataset_20261016")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    if args.strict_visual and (args.methods != ["visual_servo"] or args.planner_feedback != "vision"):
        parser.error("--strict-visual requires --methods visual_servo and --planner-feedback vision")
    if not math.isfinite(args.sampling_frequency_hz) or args.sampling_frequency_hz <= 0.:
        parser.error("--sampling-frequency-hz must be finite and positive")
    targets = load_targets(args.targets)
    target_region = json.loads(args.targets.read_text(encoding="utf-8")).get("target_region")
    if target_region and args.profile != "precision":
        parser.error("Fundus surface navigation requires precision profile with waypoint stops")
    if args.strict_visual and not target_region:
        parser.error("--strict-visual requires targets on the fundus surface")
    if any(not math.isfinite(x) or x <= 0 for x in (args.retract_mm, args.transfer_clearance_mm)):
        parser.error("Retraction and transfer clearance must be finite and positive")
    if len(targets) < 1:
        parser.error("Provide at least one unordered target point")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get('motor_execution', {}).get('enabled', False):
        parser.error('This research planner uses kinematic pose probes; use collect_v9_dataset.py for current-v9 motor datasets')
    occlusion_handling = config.get("navigation", {}).get("visual_servo", {}).get("occlusion_handling", "stop")
    fixture_enabled = config.get("reference_scene", {}).get("occlusion_fixture", {}).get("enabled", False)
    if fixture_enabled and args.methods != ["visual_servo"]:
        parser.error("Controlled occlusion fixtures currently require --methods visual_servo")
    sample_period_s = 1.0/args.sampling_frequency_hz
    config["navigation"]["sample_period_s"] = sample_period_s
    config["navigation"]["visual_servo"]["sample_period_s"] = sample_period_s
    if target_region and not args.strict_visual:
        config["navigation"]["fundus_transition"] = {
            "retract_mm": args.retract_mm, "minimum_transfer_clearance_mm": args.transfer_clearance_mm,
            "policy": "axial retract, interior transfer, fixed-orientation axial approach",
            "scope": "simulation tip clearance; not validated tissue/shaft clearance or clinical withdrawal distance",
        }
    if args.strict_visual:
        config["navigation"].pop("fundus_transition", None)
        config["navigation"]["strict_visual_navigation"] = {
            "retract_mm": args.retract_mm,
            "minimum_transfer_clearance_mm": args.transfer_clearance_mm,
            "policy": "measured-tip axial retract, measured-target angular alignment, measured-target axial approach",
            "scope": "RGB-D feedback goals; kinematics are actuation/safety only; no scene-goal joint trajectory",
        }
        # Require sufficient actual RGB-D sampling without doubling an already
        # prepared precision camera beyond the model's offscreen framebuffer.
        camera = config["reference_scene"]["eye_camera"]
        width, height = camera["resolution_px"]
        scale = max(1., 1280/width, 960/height)
        camera["resolution_px"] = [int(round(width*scale)), int(round(height*scale))]
    config["navigation"]["visual_servo"].update(require_settled=True, settle_frames=3)
    config["navigation"]["visual_servo"]["use_target_pose_calibration"] = bool(
        config["navigation"]["visual_servo"].get("position_feedback", "geometry") != "geometry"
        and not args.strict_visual)
    if args.profile == "precision":
        config["navigation"].update(endpoint_tolerance_um=20., waypoint_trajectory="trapezoid")
        config["navigation"]["visual_servo"].update(
            require_settled=True, settle_frames=3, pixel_tolerance=.5,
            depth_tolerance_mm=.003, step_gain=.3, endpoint_tolerance_um=20.)
    reference = Reference(copy.deepcopy(config))
    if args.start_task is not None:
        reference.set_task(task_si_units(args.start_task))
    settings = navigation_settings(reference)
    initial = task_user_units(reference.current_task())
    if args.strict_visual:
        # The controller receives no target joint solution. The scene target
        # and offline geometric acceptance retain only its world coordinates.
        solutions = []
        for target in targets:
            point = np.asarray(target["point_mm"], dtype=float)
            robot = point if target["frame"] == "robot" else reference.world_to_robot(point)
            world = reference.robot_to_world(robot)
            solutions.append({"id": target["id"], "frame": target["frame"],
                              "requested_robot_mm": robot.tolist(),
                              "requested_world_mm": world.tolist()})
    else:
        solutions = [solve_target(reference, t["point_mm"], t["frame"], t["id"]) for t in targets]
    if not args.strict_visual and not all(s["reachable"] for s in solutions):
        parser.error("Target preflight failed: "+json.dumps(solutions, ensure_ascii=False))
    region_report = verify_fundus_targets(reference, solutions, target_region) if target_region else None
    if region_report is not None and region_report["result"] != "PASS":
        parser.error("Targets must lie on the posterior eye surface: "+json.dumps(region_report))
    if target_region:
        if (not args.strict_visual and any(
                clearance_lower_bound_mm(reference, standoff_task(
                    s["requested_task_A_deg_B_deg_C_mm"], args.retract_mm))
                < args.transfer_clearance_mm for s in solutions)):
            parser.error("Retraction does not provide the requested tip-wall clearance")
    measured_solutions, survey = [], []
    survey_failed = False
    if args.strict_visual:
        ordered = solutions
        order_report = {
            "selected_order": [solution["id"] for solution in solutions],
            "input_order": [solution["id"] for solution in solutions],
            "candidates": [], "result": "PASS", "reason": "GIVEN_TARGET_ORDER_NO_SCENE_ROUTE_PLANNING",
            "target_source": "sequentially rendered targets; scene task poses are excluded from visual control",
            "scope": "Input visit order; retract, align and approach goals derive from measured geometry, without scene target inverse kinematics",
        }
        measured_solutions, survey = [], []
    elif args.planner_feedback == "vision":
        from rcm_vision_navigation import map_visual_targets
        measured_solutions, survey = map_visual_targets(reference, solutions)
        survey_failed = len(measured_solutions) != len(solutions)
        if survey_failed:
            ordered = solutions  # Display/acquisition sequence ONLY; never passed to a planner.
            order_report = {"selected_order": [], "input_order": [s["id"] for s in solutions],
                            "candidates": [], "result": "NOT_RUN", "reason": "VISUAL_MAP_INCOMPLETE",
                            "scope": "No autonomous visit order until every target has a valid visual measurement"}
        else:
            measured_solutions, order_report = select_visit_order(initial, measured_solutions, settings)
            by_id = {s["id"]: s for s in solutions}
            ordered = [by_id[s["id"]] for s in measured_solutions]
            order_report["target_source"] = "RGB-D measurements; not scene coordinates"
    else:
        ordered, order_report = select_visit_order(initial, solutions, settings)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/"config_snapshot.json", config)
    write_json(args.output_dir/"targets.json", {"schema_version": 1, "targets": targets,
                                               **({"target_region": target_region} if target_region else {})})
    if region_report is not None:
        write_json(args.output_dir/"fundus_target_verification.json", region_report)
    write_json(args.output_dir/"target_solutions.json", solutions)
    write_json(args.output_dir/"order_selection.json", order_report)
    if args.planner_feedback == "vision":
        write_json(args.output_dir/"visual_target_map.json", {"result": "FAIL" if survey_failed else "PASS",
                   "observations": survey, "measured_solutions": measured_solutions,
                   "scope": "Sequential stationary RGB-D survey; unknown targets are never filled from scene truth"})
    servo_settings = config["navigation"]["visual_servo"]
    geometry_feedback = servo_settings.get("position_feedback", "geometry") == "geometry"
    feature_calibration = bool(not geometry_feedback and
                               servo_settings.get("use_target_pose_calibration", True))
    feedback_kind = "visible-geometry 3D" if geometry_feedback else "legacy image-plane/depth"
    settle_frames = int(servo_settings.get("settle_frames", 3))
    write_json(args.output_dir/"experiment_protocol.json", {
        "profile": args.profile, "endpoint_tolerance_um": settings["endpoint_tolerance_um"],
        "occlusion_handling": occlusion_handling,
        "static_target_memory_assumption": ("fixed camera and static registered target during bounded occlusion; current measured tip remains required"
                                            if occlusion_handling == "static_memory_reobserve" else None),
        "occlusion_fixture": config.get("reference_scene", {}).get("occlusion_fixture"),
        "visual_servo_position_feedback": servo_settings.get("position_feedback", "geometry"),
        "depth_interface": {
            "source": "mujoco_ideal_registered_oct_simulation", "is_simulation": True,
            "raw_oct_available": False, "realtime_validated": False,
            "depth_unit": "m", "depth_convention": "positive_camera_optical_axis_z_not_ray_range",
            "clock_domain": "simulation", "nominal_rate_is_wall_clock_guarantee": False,
            "raw_depth_maps_exported_in_policy_dataset": False,
        },
        "task_regime": args.task_regime,
        "task_regime_scope": ("Registered robot-frame target-coordinate prior is part of task text; values are simulated inputs, not measured OCT"
                              if args.task_regime == "registered_coordinate_prior"
                              else "No goal coordinates or goal IDs in policy task text; one current target is presented at a time"),
        "required_stop_at_every_waypoint": True,
        "guidance_mode": ("strict_visual_multitarget" if args.strict_visual else
                          "visual_feedback" if args.planner_feedback == "vision" else "model_guidance"),
        "equation_navigation_used_for_control": False if args.strict_visual else args.planner_feedback == "model",
        "scene_goal_inverse_runtime_guard": bool(args.strict_visual),
        "strict_visual_control_contract": ({
            "target_detection": "current-frame RGB segmentation and rendered feature depth",
            "command_generation": ("measured visible-geometry 3D error with a normalized coupled actuation Jacobian"
                                   if geometry_feedback else "legacy image-plane/depth visual servo with online finite-difference image Jacobian"),
            "disabled": ["target inverse-kinematic goal for controller", "target-pose feature calibration",
                         "scene-goal analytic retract/transfer path", "RRT/PRM route planning", "scene truth fallback"],
            "retained": ["joint limits", "velocity/acceleration limits", "eye-containment safety guard",
                         "fixed camera/RCM/needle calibration", "actuation Jacobian", "independent post-run geometric acceptance"],
            "temporal_target_memory": ("bounded registered measurements from earlier images; static target assumption; never a replacement for current tip measurement"
                                       if occlusion_handling == "static_memory_reobserve" else "disabled"),
        } if args.strict_visual else None),
        "target_pose_usage": ("target world coordinates place the virtual marker and support independent post-run acceptance only; target inverse kinematics are not supplied to the visual controller or used to generate motion commands"
                              if args.strict_visual else "see method-specific privileges below"),
        "episode_definition": ("one reset start pose, one target, one navigation attempt, then a validated stop or failure hold"
                               if len(targets) == 1 else
                               "one reset start pose, sequential navigation to the supplied targets, then stop or failure hold"),
        "start_task_A_deg_B_deg_C_mm": list(initial),
        "scene_reset_scope": "fresh Reference/MuJoCo scene for each method run; target marker and robot task are reset to this episode's scenario",
        "stop_constraint": {
            "visual_controller": (f"measured controllable 3D error within {float(servo_settings.get('position_tolerance_um', 8.)):g} um for {settle_frames} consecutive zero-command-velocity observations"
                                  if geometry_feedback else f"legacy pixel/depth error within configured tolerances for {settle_frames} consecutive zero-command-velocity observations"),
            "sampled_planners": "rest-to-rest trajectory with zero endpoint velocity at every target",
            "independent_acceptance": "all targets reached, per-target stop reports valid, final command velocity is zero",
            "scope": "kinematic command validation; no hardware braking or tissue-contact validation",
        },
        "velocity_limits_changed": False, "dataset_sampling_period_s": sample_period_s,
        "dataset_sampling_frequency_hz": args.sampling_frequency_hz,
        "fundus_transition": config["navigation"].get("fundus_transition"),
        "planner_feedback": args.planner_feedback,
        "target_mapping_complete": None if args.strict_visual else not survey_failed,
        "perception": "simulated RGB + depth + geometry segmentation; calibrated fixed camera, RCM, needle and marker dimensions",
        "expert_perception_privileges": {
            "rgbd_mapping": args.planner_feedback == "vision" and not args.strict_visual,
            "segmentation_and_rendered_depth": args.planner_feedback == "vision" or "visual_servo" in args.methods,
            "visual_servo_known_target_pose_feature_calibration": "visual_servo" in args.methods and feature_calibration,
            "visual_servo_fundus_transfer_uses_scene_goal_pose": bool(target_region and "visual_servo" in args.methods and not args.strict_visual),
            "visual_servo_fundus_feedback_scope": (f"current-frame {feedback_kind} feedback throughout; no equation-generated target path or analytic transfer"
                                                    if args.strict_visual else
                                                    f"analytic retract/transfer; RGB recorded but not used for feedback or visibility stop; final C-only {feedback_kind} approach"
                                                    if target_region else f"current-frame {feedback_kind} feedback throughout the target approach"),
            "model_planner_scene_target_coordinates": args.planner_feedback == "model" and any(m in args.methods for m in ("RRT", "PRM")),
            "policy_exclusion": "expert depth, segmentation, feature calibration and scene poses are not policy input fields",
            "multi_target_scope": "targets are presented sequentially; all methods share one script-selected order; this does not demonstrate a learned policy detecting or ordering multiple lesions",
        },
        "image_ablation": {
            "status": "paired no-image inputs are exported; policy training/evaluation is not present in this repository",
            "full_input_bundle": "dataset/policy_inputs/episode_*.npz",
            "no_image_bundle": "dataset/policy_inputs_no_image/episode_*.npz",
            "paired_action_labels": "dataset/action_labels/episode_*.npz",
            "evaluation_requirement": "compare the same policy/splits with and without RGB; report success, endpoint error, constraint failures, and occlusion-stratified performance",
        },
    })
    sources = ["run_waypoint_comparison.py", "random_fundus_dataset.py", "verify_waypoint_comparison.py",
               "rcm_sampling_planners.py", "rcm_waypoint_dataset.py",
               "rcm_visual_servo.py", "rcm_reference.py", "rcm_recovered_model.py", "rcm_navigation.py", "rcm_fundus.py",
               "rcm_vision_navigation.py", "rcm_visual_geometry.py", "rcm_target_tracking.py",
               "rcm_occlusion_fixture.py", "rcm_multitarget_visual.py", "rcm_oct.py"]
    write_json(args.output_dir/"source_sha256.json", {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources})
    task = make_policy_task(args.task_regime, solutions, bool(target_region))
    print(json.dumps(order_report, ensure_ascii=False, indent=2), flush=True)
    reports, episodes = [], []
    for method in args.methods:
        for seed in ([None] if method == "visual_servo" else args.seeds):
            index = len(episodes)
            directory = args.output_dir/(method if seed is None else f"{method}_seed_{seed}")
            directory.mkdir()
            reference = Reference(copy.deepcopy(config))
            reference.set_task(task_si_units(initial))
            domain_solutions = measured_solutions if args.planner_feedback == "vision" else ordered
            space = TaskSpace(reference, ([initial] if args.strict_visual else
                                          [initial, *[s["requested_task_A_deg_B_deg_C_mm"] for s in domain_solutions]]))
            writer = EpisodeWriter(args.output_dir/"dataset", index, task, method, seed, order_report["selected_order"])
            started = time.perf_counter()
            if survey_failed:
                from rcm_vision_navigation import record_failed_survey
                frames, reached, info = record_failed_survey(reference, ordered, writer, survey)
            elif method == "visual_servo":
                frames, reached, info = run_visual(reference, ordered, directory, writer,
                                                   strict_visual=args.strict_visual)
            elif args.planner_feedback == "vision":
                from rcm_vision_navigation import run_vision_planner
                frames, reached, info = run_vision_planner(method, reference, ordered, measured_solutions,
                                                          directory, writer, seed)
            else:
                frames, reached, info = run_planner(method, reference, ordered, directory, writer, seed, space)
            metrics = evaluate(reference, frames, ordered, directory)
            rates_pass = bool(np.all(np.asarray(info["max_velocity"]) <= np.asarray(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])+1e-8)
                              and np.all(np.asarray(info["max_acceleration"]) <= np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])+1e-8))
            for missing in ordered[len(reached):]:
                reached.append({"waypoint_id": missing["id"], "result": "NOT_RUN", "reason": "PREVIOUS_WAYPOINT_FAILED",
                                "tip_error_um": None, "motion_time_s": 0., "algorithm_wall_s": 0.,
                                "terminal_stop_validated": False, "success_with_stop": False})
            stop_summary = summarize_stop_constraints(reached, info, len(ordered))
            visibility_success = summarize_target_success_by_visibility(writer.records, reached)
            visibility_visits = {row["waypoint_id"]: row for row in visibility_success["target_visits"]}
            for waypoint in reached:
                visit = visibility_visits[waypoint["waypoint_id"]]
                waypoint["visibility_exposure"] = visit["visibility_exposure"]
                waypoint["visibility_frame_counts"] = visit["visibility_frame_counts"]
                waypoint["success_with_stop"] = visit["success_with_stop"]
            passed = (all(r["result"] == "PASS" for r in reached) and rates_pass
                      and metrics["fundus_transition_constraints_pass"]
                      and metrics["tip_outside_eye_samples"] == 0
                      and max(metrics["max_geometry_errors_um"].values()) <= settings["geometry_tolerance_um"])
            passed = passed and stop_summary["pass"]
            report = {"method": method, "seed": seed, "episode_index": index,
                      "result": "PASS" if passed else "FAIL", "visit_order": order_report["selected_order"],
                      "waypoints": reached, "max_tip_error_um": max((r["tip_error_um"] for r in reached if r["tip_error_um"] is not None), default=None),
                      "trajectory_rate_limits_pass": rates_pass, "total_wall_s": time.perf_counter()-started,
                      "planning_domain_low_A_deg_B_deg_C_mm": space.low.tolist(),
                      "planning_domain_high_A_deg_B_deg_C_mm": space.high.tolist(),
                      "visibility_summary": summarize_visibility(writer.records),
                      "target_success_by_visibility": visibility_success,
                      "stop_constraint_summary": stop_summary, **metrics, **info}
            write_json(directory/"report.json", report)
            reports.append(report)
            episodes.append(writer.finish(report["result"], visibility_success, stop_summary))
            width, height = reference.eye_camera["resolution_px"]
            write_manifest(args.output_dir/"dataset", episodes, [height, width, 3], sample_period_s,
                           args.task_regime)
            write_json(args.output_dir/"run_reports.json", reports)
            print(json.dumps({"method": method, "seed": seed, "result": report["result"],
                              "max_tip_error_um": report["max_tip_error_um"],
                              "motion_time_s": report["motion_time_s"], "frames": len(writer.records)}, ensure_ascii=False), flush=True)
    aggregate = write_comparison(args.output_dir, reports, solutions, order_report)
    from plot_waypoint_results import render_results
    render_results(args.output_dir, args.baseline_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "comparison": aggregate}, ensure_ascii=False, indent=2))
    return 0 if all(r["result"] == "PASS" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
