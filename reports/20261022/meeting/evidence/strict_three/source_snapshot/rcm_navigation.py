"""Target solving, complete cycles and recorded MuJoCo kinematic execution.

All target coordinates are mm. Task coordinates use rad, rad, m internally.
This module uses mj_forward, not a motor plant or a feedback controller.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from rcm_reference import Reference, _vec


DEFAULT_NAVIGATION = {
    "require_inside_eye": True,
    "max_velocity_A_deg_B_deg_C_mm_per_s": [4.0, 2.0, 0.6],
    "max_acceleration_A_deg_B_deg_C_mm_per_s2": [10.0, 5.0, 1.5],
    "minimum_move_time_s": 0.3,
    "dwell_time_s": 0.4,
    "sample_period_s": 0.05,
    "endpoint_tolerance_um": 150.0,
    "geometry_tolerance_um": 0.000001,
}


def navigation_settings(reference):
    result = {**DEFAULT_NAVIGATION, **reference.config.get("navigation", {})}
    for key in ("max_velocity_A_deg_B_deg_C_mm_per_s",
                "max_acceleration_A_deg_B_deg_C_mm_per_s2"):
        if (_vec(result[key]) <= 0).any():
            raise ValueError(f"navigation.{key} must contain positive values")
    for key in ("minimum_move_time_s", "sample_period_s", "endpoint_tolerance_um", "geometry_tolerance_um"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"navigation.{key} must be finite and positive")
    if not math.isfinite(result["dwell_time_s"]) or result["dwell_time_s"] < 0:
        raise ValueError("navigation.dwell_time_s must be finite and nonnegative")
    if not isinstance(result["require_inside_eye"], bool):
        raise ValueError("navigation.require_inside_eye must be a boolean")
    return result


def task_user_units(q):
    return [math.degrees(q[0]), math.degrees(q[1]), float(q[2]*1000)]


def task_si_units(q):
    return np.asarray(q, dtype=float)*np.array([math.pi/180, math.pi/180, .001])


def inside_eye(reference, world_mm):
    """Test the actual compiled MuJoCo ellipsoid, including its orientation."""
    gid = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
    local_m = reference.data.geom_xmat[gid].reshape(3, 3).T @ (
        _vec(world_mm)*.001-reference.data.geom_xpos[gid])
    value = float(np.sum((local_m/reference.model.geom_size[gid])**2))
    return value <= 1.+1e-9, value


def _integer_encoders(reference, q):
    """Round to valid counts, retaining the requested B branch and L >= 0."""
    counts = [int(round(v)) for v in reference.encoders(q)]
    if reference.encoder_bounds is not None:
        for i, (lo, hi) in enumerate(reference.encoder_bounds):
            low, high = math.ceil(lo), math.floor(hi)
            if low > high:
                raise ValueError(f"Axis {'ABC'[i]} has no integer count in its configured range")
            counts[i] = min(high, max(low, counts[i]))
    # At the RCM, nearest rounding must not put the tip behind the entry.
    zero_length = np.asarray(q).copy()
    zero_length[2] = -reference.tip_reference_mm*.001
    c_at_rcm = reference.encoders(zero_length)[2]
    if reference.nominal.encoders.direction[2] > 0:
        counts[2] = max(counts[2], math.ceil(c_at_rcm))
    else:
        counts[2] = min(counts[2], math.floor(c_at_rcm))
    # A rounded B count just below the linkage minimum has no real inverse.
    # The nearest adjacent count is sufficient to stay on the real triangle.
    for b_count in (counts[1], counts[1]-1, counts[1]+1):
        candidate = [counts[0], b_count, counts[2]]
        try:
            reference.task_from_encoders(candidate, reference_task=q)
        except ValueError:
            continue
        return candidate
    raise ValueError("No nearby integer counts reproduce the requested B branch")


def solve_target(reference, point_mm, frame="world", target_id="target", require_inside_eye=None):
    """Non-mutating solve with explicit rejection reasons and integer counts.

The inverse is for the displayed NEEDLE TIP, including its 4 mm reference
offset. It is not the legacy controller's zero-length control point.
"""
    if frame not in ("world", "robot"):
        raise ValueError("frame must be 'world' or 'robot'")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("Each target needs a nonempty string id")
    result = {
        "id": target_id, "frame": frame, "input_mm": None,
        "requested_robot_mm": None, "requested_world_mm": None,
        "reachable": False, "issues": [], "requested_task_A_deg_B_deg_C_mm": None,
        "encoder_target_counts_ABC": None, "command_task_A_deg_B_deg_C_mm": None,
        "predicted_tip_world_mm": None, "quantization_error_um": None,
        "position_limits_enabled": reference.encoder_bounds is not None,
        "orientation_resolution": None, "b_inverse_candidates_deg": None,
    }
    try:
        point = _vec(point_mm)
    except (TypeError, ValueError):
        result["issues"].append({"code": "INVALID_POINT", "message": "目标须为三个有限数，单位 mm。"})
        return result
    robot = point if frame == "robot" else reference.world_to_robot(point)
    world = reference.robot_to_world(robot)
    result.update(input_mm=point.tolist(), requested_robot_mm=robot.tolist(), requested_world_mm=world.tolist())
    in_eye, surface_value = inside_eye(reference, world)
    result.update(inside_eye=in_eye, eye_ellipsoid_value=surface_value)
    require_inside_eye = navigation_settings(reference)["require_inside_eye"] if require_inside_eye is None else require_inside_eye
    if require_inside_eye and not in_eye:
        result["issues"].append({"code": "OUTSIDE_EYE", "message": "目标不在当前眼球椭球内部。"})
    try:
        state = reference.nominal.inverse(robot, enforce_limits=False,
                                          reference_state=reference.state_from_task(reference.current_task()))
        q = reference.task_from_state(state)
        result["requested_task_A_deg_B_deg_C_mm"] = task_user_units(q)
        result["orientation_resolution"] = ("retained_at_rcm" if np.linalg.norm(robot) < 1e-12
                                            else "azimuth_retained_on_Y_axis" if math.hypot(robot[0], robot[2]) < 1e-12
                                            else "position_inverse")
    except ValueError as error:
        result["issues"].append({"code": "UNDEFINED_ORIENTATION", "message": str(error)})
        return result
    if reference.joint_bounds is not None:
        wanted = np.asarray(task_user_units(q))
        lo = np.asarray(task_user_units(reference.joint_bounds[:, 0]))
        hi = np.asarray(task_user_units(reference.joint_bounds[:, 1]))
        for i, axis in enumerate("ABC"):
            if q[i] < reference.joint_bounds[i, 0]-1e-12 or q[i] > reference.joint_bounds[i, 1]+1e-12:
                result["issues"].append({
                    "code": f"AXIS_{axis}_LIMIT", "axis": axis,
                    "requested": float(wanted[i]), "allowed": [float(lo[i]), float(hi[i])],
                    "unit": "mm" if axis == "C" else "deg", "message": f"{axis} 轴超出配置的仿真范围。",
                })
    if result["issues"]:
        return result
    try:
        reference.validate_task(q)
        counts = _integer_encoders(reference, q)
        command = reference.task_from_encoders(counts, reference_task=q)
        displacement = reference.nominal.b_displacement(reference.state_from_task(command).alpha_deg)
        result["b_inverse_candidates_deg"] = list(reference.nominal.alpha_candidates_from_b(displacement))
        predicted = reference.robot_to_world(reference.nominal.forward(reference.state_from_task(command), enforce_limits=False))
    except ValueError as error:
        result["issues"].append({"code": "INVALID_BRANCH_OR_COUNTS", "message": str(error)})
        return result
    error_um = float(np.linalg.norm(predicted-world)*1000)
    if error_um > navigation_settings(reference)["endpoint_tolerance_um"]:
        result["issues"].append({"code": "ENCODER_RESOLUTION", "message": "整数编码器目标的量化误差超过配置容差。"})
        return result
    result.update(reachable=True, encoder_target_counts_ABC=counts,
                  command_task_A_deg_B_deg_C_mm=task_user_units(command),
                  predicted_tip_world_mm=predicted.tolist(), quantization_error_um=error_um)
    return result


def load_targets(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Target file must be a schema_version=1 object")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Target file must contain a nonempty targets list")
    default_frame = payload.get("frame", "world")
    parsed = []
    for i, target in enumerate(targets):
        if not isinstance(target, dict) or "point_mm" not in target:
            raise ValueError(f"Target {i} needs point_mm")
        parsed.append({"id": target.get("id", f"target_{i+1}"), "frame": target.get("frame", default_frame),
                       "point_mm": target["point_mm"]})
    ids = [t["id"] for t in parsed]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Target ids must be unique, nonempty strings")
    return parsed


class NavigationRejected(ValueError):
    def __init__(self, solutions):
        self.solutions = solutions
        ids = [s["id"] for s in solutions if not s["reachable"]]
        super().__init__("Navigation preflight rejected: "+", ".join(ids))


@dataclass(frozen=True)
class Segment:
    start_s: float
    duration_s: float
    phase: str
    target_index: int
    q_start: np.ndarray
    q_end: np.ndarray

    @property
    def end_s(self):
        return self.start_s+self.duration_s

    def sample(self, elapsed_s):
        u = min(1., max(0., (elapsed_s-self.start_s)/self.duration_s))
        # Quintic interpolation: zero velocity AND acceleration at both ends.
        s = u**3*(10.-15.*u+6.*u*u)
        ds = 30.*u*u*(1.-u)**2/self.duration_s
        dds = 60.*u*(1.-u)*(1.-2.*u)/self.duration_s**2
        delta = self.q_end-self.q_start
        return self.q_start+s*delta, ds*delta, dds*delta


@dataclass(frozen=True)
class NavigationPlan:
    solutions: list
    segments: list[Segment]
    settings: dict

    @property
    def duration_s(self):
        return self.segments[-1].end_s

    def sample(self, elapsed_s):
        if not math.isfinite(elapsed_s):
            raise ValueError("Sample time must be finite")
        t = min(self.duration_s, max(0., elapsed_s))
        for segment in self.segments:
            if t <= segment.end_s+1e-12:
                return (*segment.sample(t), segment)
        raise AssertionError("No segment for sample time")

    def sample_times(self, period_s=None):
        dt = self.settings["sample_period_s"] if period_s is None else period_s
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Sample period must be finite and positive")
        if self.duration_s/dt > 1_000_000:
            raise ValueError("More than one million samples requested; increase sample period")
        # Include every exact endpoint, even when it does not land on dt.
        values = sorted([*np.arange(0., self.duration_s, dt), *[s.end_s for s in self.segments]])
        unique = []
        for value in values:
            if not unique or value-unique[-1] > 1e-10:
                unique.append(float(value))
            elif any(abs(value-s.end_s) < 1e-12 for s in self.segments):
                unique[-1] = float(value)
        return unique

    def as_dict(self):
        return {
            "execution_mode": "kinematic_mj_forward", "duration_s": self.duration_s,
            "settings": self.settings, "target_solutions": self.solutions,
            "segments": [{"phase": s.phase, "target_index": s.target_index,
                          "start_s": s.start_s, "end_s": s.end_s,
                          "start_task_A_deg_B_deg_C_mm": task_user_units(s.q_start),
                          "end_task_A_deg_B_deg_C_mm": task_user_units(s.q_end)} for s in self.segments],
        }


def build_plan(reference, targets):
    """Preflight ALL targets before any model or marker mutation."""
    if not targets:
        raise ValueError("At least one target is required")
    ids = [t["id"] for t in targets]
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("Each target needs a nonempty string id")
    if len(set(ids)) != len(ids):
        raise ValueError("Target ids must be unique")
    settings = navigation_settings(reference)
    solutions = [solve_target(reference, t["point_mm"], t.get("frame", "world"), t["id"]) for t in targets]
    if not all(s["reachable"] for s in solutions):
        raise NavigationRejected(solutions)
    vmax = task_si_units(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
    amax = task_si_units(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
    previous = reference.validate_task(reference.current_task()).copy()
    segments = []
    elapsed = 0.

    def add(phase, target, index, hold=None):
        nonlocal previous, elapsed
        target = reference.validate_task(target).copy()
        delta = np.abs(target-previous)
        duration = float(hold) if hold is not None else max(
            settings["minimum_move_time_s"], float(np.max(1.875*delta/vmax)),
            float(np.max(np.sqrt((10.*math.sqrt(3.)/3.)*delta/amax))))
        if duration > 0:
            segments.append(Segment(elapsed, duration, phase, index, previous.copy(), target.copy()))
            elapsed += duration
        previous = target

    # Return the axial coordinate before changing orientation from an
    # arbitrary starting pose. C=0 retains the 4 mm reference tip offset.
    if not np.allclose(previous, reference.home, atol=1e-12, rtol=0):
        retracted = previous.copy()
        retracted[2] = reference.home[2]
        add("prepare_C_zero", retracted, -1)
        add("prepare_reset", reference.home, -1)
    for index, solution in enumerate(solutions):
        # Preserve the B branch resolved during preflight, including when the
        # current pose is on the other side of the displacement turning point.
        goal = task_si_units(solution["command_task_A_deg_B_deg_C_mm"])
        aimed = goal.copy()
        aimed[2] = reference.home[2]
        add("aim", aimed, index)
        add("insert", goal, index)
        add("dwell", goal, index, settings["dwell_time_s"])
        add("return_C_zero", aimed, index)
        add("reset", reference.home, index)
    return NavigationPlan(solutions, segments, settings)


class NavigationRecorder:
    """Record sampled geometry without calling it motor feedback."""
    def __init__(self, reference, plan):
        self.reference, self.plan = reference, plan
        self.rows, self.reached = [], {}
        self.active_target_index = None
        self.max_geometry = {k: 0. for k in (
            "rcm_drift_um", "needle_axis_rcm_distance_um", "fk_agreement_um",
            "upper_link_closure_um", "lower_link_closure_um", "distal_length_error_um")}
        self.max_velocity, self.max_acceleration = np.zeros(3), np.zeros(3)
        self.needle_geometry_summary = {"max_proximal_centerline_inside_mm": 0.,
                                       "max_entry_rcm_distance_mm": 0.,
                                       "tip_outside_eye_sample_count": 0,
                                       "scope": "centerline diagnostics; no port or tissue contact validation"}

    def step(self, elapsed_s):
        q, qd, qdd, segment = self.plan.sample(elapsed_s)
        solution = self.plan.solutions[segment.target_index] if segment.target_index >= 0 else None
        if solution is not None and segment.target_index != self.active_target_index:
            self.reference.set_target_world(solution["requested_world_mm"])
            self.active_target_index = segment.target_index
        self.reference.set_task(q)
        self.reference.data.time = float(elapsed_s)
        m = self.reference.metrics()
        needle = m["needle_eye_geometry"]
        summary = self.needle_geometry_summary
        summary["max_proximal_centerline_inside_mm"] = max(summary["max_proximal_centerline_inside_mm"], needle["proximal_centerline_inside_mm"])
        if needle["entry_rcm_distance_mm"] is not None:
            summary["max_entry_rcm_distance_mm"] = max(summary["max_entry_rcm_distance_mm"], needle["entry_rcm_distance_mm"])
        summary["tip_outside_eye_sample_count"] += int(not needle["tip_inside_eye"])
        for name in self.max_geometry:
            value = abs(m["distal_length_mm"]-self.reference.distal_mm)*1000 if name == "distal_length_error_um" else m[name]
            self.max_geometry[name] = max(self.max_geometry[name], value)
        velocity, acceleration = np.abs(task_user_units(qd)), np.abs(task_user_units(qdd))
        self.max_velocity = np.maximum(self.max_velocity, velocity)
        self.max_acceleration = np.maximum(self.max_acceleration, acceleration)
        row = {"time_s": float(elapsed_s), "phase": segment.phase,
               "target_id": "" if solution is None else solution["id"],
               "A_deg": math.degrees(q[0]), "B_deg": math.degrees(q[1]), "C_mm": float(q[2]*1000),
               "tip_target_error_um": m["tip_target_error_um"]}
        for name in ("junction_rcm_axial_offset_mm", "junction_inside_eye",
                     "entry_rcm_distance_mm", "entry_needle_part",
                     "distal_centerline_inside_mm", "proximal_centerline_inside_mm", "tip_inside_eye"):
            row[name] = needle[name]
        for prefix, values, axes in (
            ("model_encoder_float", m["model_robot_encoder_counts_float"], "ABC"),
            ("tip_world_mm", m["tip_world_mm"], "XYZ"),
            ("junction_world_mm", m["junction_world_mm"], "XYZ"),
            ("target_world_mm", m["target_world_mm"], "XYZ"),
            ("velocity_deg_deg_mm_per_s", task_user_units(qd), "ABC"),
            ("acceleration_deg_deg_mm_per_s2", task_user_units(qdd), "ABC"),
        ):
            row.update({f"{prefix}_{axis}": value for axis, value in zip(axes, values)})
        for name in self.max_geometry:
            row[name] = abs(m["distal_length_mm"]-self.reference.distal_mm)*1000 if name == "distal_length_error_um" else m[name]
        self.rows.append(row)
        if segment.phase == "insert" and abs(elapsed_s-segment.end_s) < 1e-9:
            self.reached[solution["id"]] = {
                "id": solution["id"], "reached_time_s": float(elapsed_s),
                "target_world_mm": solution["requested_world_mm"], "actual_model_tip_world_mm": m["tip_world_mm"],
                "encoder_target_counts_ABC": solution["encoder_target_counts_ABC"],
                "tip_error_um": m["tip_target_error_um"],
                "pass": m["tip_target_error_um"] <= self.plan.settings["endpoint_tolerance_um"],
                "needle_eye_geometry": needle,
            }

    def report(self):
        completed = bool(self.rows) and abs(self.rows[-1]["time_s"]-self.plan.duration_s) < 1e-9
        home_error = float(np.max(np.abs(self.reference.current_task()-self.reference.home)))
        vmax = np.asarray(self.plan.settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
        amax = np.asarray(self.plan.settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
        limits_pass = bool(np.all(self.max_velocity <= vmax+1e-9) and np.all(self.max_acceleration <= amax+1e-9))
        passed = (completed and len(self.reached) == len(self.plan.solutions) and
                  all(v["pass"] for v in self.reached.values()) and home_error < 1e-12 and limits_pass and
                  max(self.max_geometry.values()) <= self.plan.settings["geometry_tolerance_um"])
        errors = [v["tip_error_um"] for v in self.reached.values()]
        return {
            "result": "PASS" if passed else ("ABORTED" if not completed else "FAIL"),
            "scope": "MuJoCo kinematic navigation and encoder quantization, not measured real-robot accuracy",
            "execution_mode": "kinematic_mj_forward", "feedback_controller_used": False,
            "mujoco_version": mujoco.__version__, "completed": completed,
            "target_count": len(self.plan.solutions), "reached_count": len(self.reached),
            "sample_count": len(self.rows), "planned_duration_s": self.plan.duration_s,
            "endpoint_tolerance_um": self.plan.settings["endpoint_tolerance_um"],
            "mean_reached_tip_error_um": None if not errors else float(np.mean(errors)),
            "max_reached_tip_error_um": None if not errors else max(errors),
            "max_geometry_errors_um": self.max_geometry,
            "trajectory_rate_limits_pass": limits_pass,
            "max_velocity_A_deg_B_deg_C_mm_per_s": self.max_velocity.tolist(),
            "max_acceleration_A_deg_B_deg_C_mm_per_s2": self.max_acceleration.tolist(),
            "final_task_A_deg_B_deg_C_mm": task_user_units(self.reference.current_task()),
            "final_model_encoder_counts_ABC": list(self.reference.encoders()),
            "returned_to_zero": completed and home_error < 1e-12,
            "C_zero_tip_distance_mm": self.reference.tip_reference_mm,
            "position_limits_enabled": self.reference.encoder_bounds is not None,
            "needle_eye_geometry_summary": dict(self.needle_geometry_summary),
            "b_encoder_decoding": "target/current pose selects the inverse branch",
            "target_results": list(self.reached.values()), "target_solutions": self.plan.solutions,
        }

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory/"navigation_plan.json", self.plan.as_dict())
        write_json(directory/"target_solutions.json", self.plan.solutions)
        write_json(directory/"navigation_report.json", self.report())
        if self.rows:
            with (directory/"trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)


def execute_plan(reference, plan, sample_period_s=None):
    recorder = NavigationRecorder(reference, plan)
    for t in plan.sample_times(sample_period_s):
        recorder.step(t)
    return recorder


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
