"""RGB-D target mapping and closed-loop execution of RRT/PRM paths.

Camera intrinsics/extrinsics and marker dimensions are calibration inputs.
Segmentation/depth are simulated sensors, not a real RGB-only estimator.
No target pose, tip site position, or target-pose feature calibration is read
by the estimator or feedback controller. Kinematics remain the actuator and
mechanical-safety model; they do not supply the measured target/tip position.
"""
from __future__ import annotations

import math
import time

import numpy as np

from rcm_fundus import no_surface_slide_edge, standoff_task
from rcm_navigation import navigation_settings, task_si_units, task_user_units, write_json
from rcm_sampling_planners import PRM, TaskSpace, rrt, shortcut_path
from rcm_visual_servo import VisualServoSensor
from rcm_visual_geometry import NeedleGeometry, fit_marker_center


class RGBDPositionSensor(VisualServoSensor):
    def __init__(self, reference):
        super().__init__(reference)
        cid = reference.camera_id(self.camera_name)
        # Fixed, calibrated camera pose. No target/tip geom_xpos is consulted.
        self.camera_rotation = reference.data.cam_xmat[cid].reshape(3, 3).copy()
        self.camera_position_mm = reference.data.cam_xpos[cid].copy()*1000.
        self.focal_px = .5*self.height/math.tan(math.radians(float(reference.model.cam_fovy[cid]))/2)
        self.target_radius_mm = float(reference.model.geom("lesion_marker").size[0])*1000.
        self.tip_radius_mm = float(reference.model.geom_size[self.tip_geom_id, 0])*1000.

    def center_from_image(self, observation, target):
        if target and observation["target_pixel"] is None:
            raise RuntimeError("TARGET_NOT_VISIBLE")
        if (not target and observation["tip_pixel"] is None
                and self.settings.get('position_feedback') == 'geometry'):
            raise RuntimeError(observation.get("geometry", {}).get("tip_error", "TIP_NOT_VISIBLE"))
        # Recompute from supplied image arrays: injected sensor changes must
        # affect measurements, even when an observation has cached diagnostics.
        if self.geometry is None:
            self.geometry = NeedleGeometry.from_reference(
                self.reference, self.width, self.height, self.camera_name, self.settings)
        center, quality = (self.geometry.target(observation) if target else self.geometry.tip(observation))
        return center, quality["residual_mm"]

    def positions(self, observation):
        target, target_residual = self.center_from_image(observation, True)
        tip, tip_residual = self.center_from_image(observation, False)
        return target, tip, max(target_residual, tip_residual)


def task_from_measurement(reference, world_mm):
    """Inverse kinematics of a SENSOR measurement; not of the scene target."""
    state = reference.nominal.inverse(reference.world_to_robot(world_mm), enforce_limits=False,
                                      reference_state=reference.state_from_task(reference.current_task()))
    return np.asarray(task_user_units(reference.task_from_state(state)))


def map_visual_targets(reference, scene_solutions):
    """Scene harness presents each target in turn at the stationary start pose.

    Input coordinates are used ONLY to place the simulated markers. Perception
    receives rendered arrays, and ordering/planning receive measured positions.
    This is a sequential marker survey, not simultaneous multi-lesion detection.
    """
    measured, audit = [], []
    sensor = RGBDPositionSensor(reference)
    try:
        for scene in scene_solutions:
            reference.set_target_world(scene["requested_world_mm"])
            observation = sensor.observe()
            visibility = sensor.visibility_metrics(observation["segmentation"])
            try:
                point, residual = sensor.center_from_image(observation, True)
            except RuntimeError as exc:
                audit.append({"id": scene["id"], "result": "FAIL", "reason": str(exc),
                              **visibility})
                continue
            q = task_from_measurement(reference, point)
            measured.append({"id": scene["id"], "requested_world_mm": point.tolist(),
                             "requested_robot_mm": reference.world_to_robot(point).tolist(),
                             "requested_task_A_deg_B_deg_C_mm": q.tolist()})
            audit.append({"id": scene["id"], "result": "PASS", "measured_world_mm": point.tolist(),
                          "fit_residual_mm": residual, **visibility})
    finally:
        sensor.close()
    return measured, audit


def record_failed_survey(reference, scenes, writer, audit):
    """Record stationary failed acquisition; do not invent a visit order/path."""
    from run_waypoint_comparison import append_frame
    frames, results = [], []
    dt = float(navigation_settings(reference)["visual_servo"]["sample_period_s"])
    q = np.asarray(task_user_units(reference.current_task()))
    sensor = RGBDPositionSensor(reference)
    try:
        audit_by_id = {row["id"]: row for row in audit}
        for i, scene in enumerate(scenes):
            reference.set_target_world(scene["requested_world_mm"])
            observation = sensor.observe()
            visibility = sensor.visibility_metrics(
                observation["segmentation"])
            append_frame(frames, i*dt, q, scene["id"], "hold")
            writer.add(i*dt, q, q, scene["id"], observation["rgb"], terminal=i == len(scenes)-1,
                       phase="hold", visibility=visibility)
            results.append({"waypoint_id": scene["id"], "result": "NOT_RUN", "reason": "VISUAL_MAP_INCOMPLETE",
                            "tip_error_um": None, "motion_time_s": 0., "algorithm_wall_s": 0.})
    finally:
        sensor.close()
    return frames, results, {"algorithm_wall_s": 0., "max_velocity": [0., 0., 0.], "max_acceleration": [0., 0., 0.],
        "terminal_velocity": [0., 0., 0.], "terminal_stop_validated": False,
        "execution_mode": "visual_survey_rejected", "target_source": "RGB-D survey",
        "ground_truth_used_for_control": False, "planner_executed": False, "stop_reason": "VISUAL_MAP_INCOMPLETE",
        "survey": audit, "visual_feedback_samples": 0, "robot_moved": False,
        "motion_time_s": 0., "survey_time_s": (len(scenes)-1)*dt,
        "planning_domain_low_A_deg_B_deg_C_mm": None, "planning_domain_high_A_deg_B_deg_C_mm": None}


def feedback_command(reference, actual_q, measured_tip_mm, desired_q, previous_velocity, dt, phase):
    """Measured tip error determines every issued command, with rate limits."""
    settings = navigation_settings(reference)
    error = np.asarray(desired_q)-task_from_measurement(reference, measured_tip_mm)
    if phase in ("retract", "approach"):
        error[:2] = 0.  # Rotation close to the fundus is forbidden.
    vmax = np.asarray(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
    amax = np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
    wanted = np.clip(1.5*error, -vmax, vmax)
    braking = .6*np.sqrt(2*amax*np.abs(error))
    wanted = np.clip(wanted, -braking, braking)
    velocity = np.clip(wanted, previous_velocity-amax*dt, previous_velocity+amax*dt)
    candidate = np.asarray(actual_q)+velocity*dt
    reference.validate_task(task_si_units(candidate))
    policy = settings.get("fundus_transition")
    if policy and not no_surface_slide_edge(reference, actual_q, candidate, policy["minimum_transfer_clearance_mm"]):
        raise RuntimeError("FUNDUS_MOTION_CONSTRAINT")
    return candidate, velocity, error


def run_vision_planner(method, reference, scene_solutions, measured_solutions, directory, writer, seed):
    """Actual sampled planning plus online visual tracking, with no truth fallback."""
    from run_waypoint_comparison import append_frame, goal_result

    settings = navigation_settings(reference)
    policy = settings.get("fundus_transition")
    dt = float(settings["visual_servo"]["sample_period_s"])
    rng = np.random.default_rng(seed)
    q = np.asarray(task_user_units(reference.current_task()))
    goals = [np.asarray(s["requested_task_A_deg_B_deg_C_mm"]) for s in measured_solutions]
    transfer_goals = [standoff_task(g, policy["retract_mm"]) for g in goals] if policy else goals
    space = TaskSpace(reference, [q, *goals])
    space.minimum_clearance_mm = policy["minimum_transfer_clearance_mm"] if policy else 0.
    sensor = RGBDPositionSensor(reference)
    frames, results, paths, log = [], [], [], []
    elapsed, computation = 0., 0.
    velocity, max_velocity, max_acceleration = np.zeros(3), np.zeros(3), np.zeros(3)
    roadmap = None
    reason, stopped = "MAX_ITERATIONS", False
    observation = None
    try:
        for index, (scene, goal) in enumerate(zip(scene_solutions, goals, strict=True)):
            # Scene placement and independent acceptance are outside perception/control.
            reference.set_target_world(scene["requested_world_mm"])
            leg_start, started = elapsed, time.perf_counter()
            waypoint = scene["id"]
            status = "FAIL"
            try:
                observation = sensor.observe()
                sensor.positions(observation)  # No planning/execution without a valid observation.
                depart = standoff_task(q, policy["retract_mm"]) if policy and index else q.copy()
                if method == "RRT":
                    path, search = rrt(space, depart, transfer_goals[index], rng)
                else:
                    if roadmap is None:
                        roadmap = PRM(space, [depart, *transfer_goals], rng)
                    # Use the actual stopped departure, not an assumed perfect arrival.
                    path = roadmap.query(0, index+1) if index == 0 else prm_from_departure(roadmap, space, depart, index+1)
                    search = roadmap.info
                path = shortcut_path(space, path)
                phases = []
                if policy and index:
                    phases.append(("retract", [depart]))
                phases.append(("transfer" if policy else "motion", path[1:]))
                if policy:
                    phases.append(("approach", [goal]))
                paths.append({"waypoint_id": waypoint, "search": search,
                              "path_A_deg_B_deg_C_mm": [p.tolist() for p in path],
                              "phases": [{"phase": phase, "path_A_deg_B_deg_C_mm": [p.tolist() for p in points]}
                                         for phase, points in phases]})
                for phase, points in phases:
                    for desired in points:
                        settled = 0
                        final_goal = phase == "approach" or (phase == "motion" and np.allclose(desired, goal, atol=1e-10, rtol=0))
                        for _ in range(1800):
                            observation = sensor.observe()
                            visibility = sensor.visibility_metrics(observation["segmentation"])
                            target_mm, tip_mm, residual = sensor.positions(observation)
                            # A changing target invalidates the mapped global plan. Never silently
                            # continue to old coordinates or substitute scene truth.
                            if np.linalg.norm(target_mm-measured_solutions[index]["requested_world_mm"]) > .15:
                                raise RuntimeError("VISUAL_TARGET_MOVED_REPLAN_REQUIRED")
                            desired = np.asarray(desired).copy()
                            if phase == "approach":
                                live_goal = task_from_measurement(reference, target_mm)
                                desired[2] = live_goal[2]
                            elif final_goal:
                                desired = task_from_measurement(reference, target_mm)
                            measured_q = task_from_measurement(reference, tip_mm)
                            error = desired-measured_q
                            tolerance = np.array([.025, .025, .008])
                            # During axial retract the commanded A/B orientation is
                            # intentionally held fixed.  The RGB-D inverse of the
                            # tiny tip marker has a small angular bias, so using its
                            # A/B residual to decide whether a retract finished can
                            # leave the phase spinning forever.  Keep A/B in the
                            # feedback command (feedback_command already zeros
                            # those components for retract), but judge completion
                            # only on the axis that is allowed to move.
                            completion_error = error.copy()
                            if phase == "retract":
                                completion_error[:2] = 0.
                            in_tolerance = bool(np.all(np.abs(completion_error) <= tolerance))
                            if final_goal:
                                in_tolerance = in_tolerance and np.linalg.norm(target_mm-tip_mm) <= .015
                            if in_tolerance:
                                amax = np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
                                new_velocity = np.sign(velocity)*np.maximum(0., np.abs(velocity)-amax*dt)
                                candidate = q+new_velocity*dt
                                if policy and not no_surface_slide_edge(reference, q, candidate, policy["minimum_transfer_clearance_mm"]):
                                    raise RuntimeError("FUNDUS_MOTION_CONSTRAINT")
                            else:
                                candidate, new_velocity, _ = feedback_command(reference, q, tip_mm, desired, velocity, dt, phase)
                            if phase in ("transfer", "motion") and not space.edge_valid(space.normalize(q), space.normalize(candidate)):
                                raise RuntimeError("VISUAL_TRACKING_PATH_CONSTRAINT")
                            settled = settled+1 if in_tolerance and np.max(np.abs(new_velocity)) < 1e-12 else 0
                            log.append({"time_s": elapsed, "waypoint_id": waypoint, "phase": phase,
                                        "measured_target_world_mm": target_mm.tolist(), "measured_tip_world_mm": tip_mm.tolist(),
                                        "fit_residual_mm": residual, **visibility,
                                        "visual_tip_target_error_mm": float(np.linalg.norm(target_mm-tip_mm)),
                                        "command_A_deg_B_deg_C_mm": candidate.tolist()})
                            append_frame(frames, elapsed, q, waypoint, phase)
                            writer.add(elapsed, q, candidate, waypoint, observation["rgb"], phase=phase,
                                       visibility=visibility)
                            max_velocity = np.maximum(max_velocity, np.abs(new_velocity))
                            max_acceleration = np.maximum(max_acceleration, np.abs(new_velocity-velocity)/dt)
                            velocity, q = new_velocity, candidate
                            reference.set_task(task_si_units(q))
                            elapsed += dt
                            if settled >= 3:
                                break
                        else:
                            raise RuntimeError("MAX_ITERATIONS")
                status, reason, stopped = "PASS", "VISUAL_POSITION_AND_SETTLED", True
            except (RuntimeError, ValueError) as exc:
                reason, stopped = str(exc), False
            wall = time.perf_counter()-started
            computation += wall
            results.append(goal_result(reference, scene, elapsed-leg_start, wall, status, reason,
                                       terminal_stop_validated=stopped))
            # Independent truth-based acceptance must not steer the controller.
            # Only the visual status decides whether another leg is attempted.
            if status != "PASS":
                break
        observation = sensor.observe()
        visibility = sensor.visibility_metrics(observation["segmentation"])
        # Immediate hold is a simulator abort, not a verified physical brake.
        # Include the discrete stop jump when reporting command rate bounds.
        if not stopped:
            max_acceleration = np.maximum(max_acceleration, np.abs(velocity)/dt)
        append_frame(frames, elapsed, q, waypoint, "hold")
        writer.add(elapsed, q, q, waypoint, observation["rgb"], terminal=True, phase="hold",
                   visibility=visibility)
    finally:
        sensor.close()
    write_json(directory/"paths.json", paths)
    write_json(directory/"visual_feedback.json", log)
    return frames, results, {"algorithm_wall_s": computation, "max_velocity": max_velocity.tolist(),
        "max_acceleration": max_acceleration.tolist(), "terminal_velocity": velocity.tolist(),
        "terminal_stop_validated": stopped, "feedback_controller_used": True,
        "target_source": "calibrated RGB-D marker reconstruction", "tip_source": "calibrated RGB-D marker reconstruction",
        "ground_truth_used_for_control": False, "known_target_pose_used_for_feature_calibration": False,
        "ground_truth_used_for_post_run_acceptance": True, "visibility_required_at_every_command": True,
        "occlusion_policy": "terminate and hold; physical emergency braking is not simulated",
        "execution_mode": "vision_guided_sampling_planner", "visual_feedback_samples": len(log),
        "validity_checks": space.validity_checks, "stop_reason": reason,
        "planning_domain_low_A_deg_B_deg_C_mm": space.low.tolist(),
        "planning_domain_high_A_deg_B_deg_C_mm": space.high.tolist()}


def prm_from_departure(roadmap, space, depart, goal_index):
    """Connect the observed departure to the reusable PRM without teleporting."""
    # A temporary query graph preserves the roadmap and uses its existing edges.
    import copy
    graph = copy.copy(roadmap)
    graph.graph = [list(edges) for edges in roadmap.graph]
    point = space.normalize(depart)
    neighbors = sorted(range(len(graph.nodes)), key=lambda i: np.linalg.norm(graph.nodes[i]-point))
    index = len(graph.nodes)
    graph.nodes = np.vstack([graph.nodes, point])
    graph.graph.append([])
    for neighbor in neighbors[:20]:
        if space.edge_valid(point, graph.nodes[neighbor]):
            distance = float(np.linalg.norm(point-graph.nodes[neighbor]))
            graph.graph[index].append((neighbor, distance))
            graph.graph[neighbor].append((index, distance))
    return graph.query(index, goal_index)
