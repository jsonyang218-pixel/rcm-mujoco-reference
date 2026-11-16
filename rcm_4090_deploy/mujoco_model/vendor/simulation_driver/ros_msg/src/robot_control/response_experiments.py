"""Elapsed-time response signals and guarded feedback controller (no publishing).

Distances are mm internally; reference amplitudes and reports use micrometres.
The experiment owns a frozen coordinate origin, needle geometry and controller.
"""

from dataclasses import asdict, dataclass
import math
import threading

from .kinematics import (
    AXES, C_HANDLE_COUNTS_PER_MM, CONTROL_GEAR, ENCODER_COUNTS_PER_MM, ROBOT_ENCODER_LIMITS,
    b_alpha_from_robot_encoder, b_distance_from_alpha, model_tip_from_robot_encoders,
    cartesian_handle_target,
    needle_direction_from_robot_encoders, needle_pose_from_robot_encoders,
    stepped_values, validate_cartesian_delta, validate_manual_jog,
)


from .response_signals import ResponseSignal


def response_state_key(state):
    return (tuple(state["robot"]), tuple(state["handle"]), tuple(state["position"]),
            bool(state["forward"]), bool(state["backward"]))


def stationary_protocol_origin(history, state, now, timeout, duration=.5):
    """Attach once to a quiet existing wire session, never reset it to zero.

    This is only for starting a new response experiment. Running experiments
    still accumulate on their last successful publish, independently of echoes.
    """
    key = response_state_key(state)
    boundary = now - duration
    before = [i for i, (received, _) in enumerate(history) if received <= boundary]
    if not before:
        raise ValueError("实验启动前需要至少 0.5 s 连续静止反馈")
    recent = history[before[-1]:]
    if (now - recent[-1][0] > min(timeout, .1) or len(recent) < 3 or
            any(b[0] - a[0] > min(timeout, .1) for a, b in zip(recent, recent[1:]))):
        raise ValueError("实验启动前反馈不连续，请等待静止反馈")
    if key[-2] or key[-1] or any(sample != key for _, sample in recent):
        raise ValueError("实验启动前编码器、坐标与协议回显需连续静止 0.5 s")
    return tuple(stepped_values(state["handle"], AXES, (0, 0, 0, 0)))


@dataclass(frozen=True)
class ResponseControl:
    interval_s: float = 0.05
    tolerance_um: float = 1.0
    settle_samples: int = 3
    settle_s: float = 0.25
    state_timeout_s: float = 1.0
    no_progress_s: float = 15.0
    movement_um: float = 0.2
    correction_limit_um: float = 50.0
    envelope_margin_um: float = 20.0
    transverse_limit_um: float = 20.0
    posture_limit_deg: float = 0.02
    max_records: int = 200000

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("控制参数必须为有限正数")
        if self.interval_s != 0.05 or self.tolerance_um != 1.0:
            raise ValueError("基线固定 50 ms 周期与 1 µm 容差")
        if self.settle_samples < 3 or self.settle_s < 0.25:
            raise ValueError("稳定判据不得弱于三帧、250 ms")


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(sum(v * v for v in a))


def junction_handle_target(robot, handles, delta):
    """Solve a junction displacement in the aligned encoder model frame.

    Preserve the nearest signed radial branch. The needle-tip inverse rejects
    zero offsets, so junction targets have their own explicit geometry here.
    """
    junction = model_tip_from_robot_encoders(robot, 0.0)
    target = tuple(junction[i] + delta[i] for i in range(3))
    radius = _norm(target)
    direction = needle_direction_from_robot_encoders(robot)
    if radius > 1e-9:
        unit = tuple(v / radius for v in target)
        branch = 1 if sum(a * b for a, b in zip(unit, direction)) >= 0 else -1
        direction = tuple(branch * v for v in unit)
        distance = branch * radius
    else:
        distance = 0.0
    alpha = math.degrees(math.acos(max(-1, min(1, direction[1])))) - 22
    predicted = (round(math.degrees(math.atan2(direction[0], direction[2])) * 200000 / 360),
                 round(ENCODER_COUNTS_PER_MM * b_distance_from_alpha(alpha)),
                 round(ENCODER_COUNTS_PER_MM * (14 + distance)), robot[3])
    a_per_handle = (2 * CONTROL_GEAR * 3 / 4000) * (200000 / 360)
    delta_handles = (round((predicted[0] - robot[0]) / a_per_handle),
                     round((b_alpha_from_robot_encoder(predicted[1]) - b_alpha_from_robot_encoder(robot[1])) /
                           (CONTROL_GEAR * 3 / 4000)),
                     round((predicted[2] - robot[2]) / (3 * CONTROL_GEAR * 10000 / 4000)), 0)
    return stepped_values(handles, AXES, delta_handles), predicted


class ResponseExperiment:
    """Thread-safe recorder for guarded feedback control."""

    def __init__(self, signal, state, handles, started_mono, started_unix,
                 mode="junction_axis", tip_offset_mm=4.0, offset_sign=1.0,
                 control=None, metadata=None, xyz_amplitudes_um=None,
                 drive_vector_override=None):
        self.lock = threading.RLock()
        self.signal, self.control = signal, control or ResponseControl()
        if mode not in ("c_axis", "junction_axis", "junction_xyz", "junction_abc", "tip_x", "tip_y", "tip_z", "junction_x", "junction_y", "junction_z"):
            raise ValueError("未知响应坐标模式")
        if not (math.isfinite(tip_offset_mm) and tip_offset_mm >= 0 and offset_sign in (-1, 1)):
            raise ValueError("针型几何无效")
        self.mode, self.tip_offset_mm, self.offset_sign = mode, tip_offset_mm, offset_sign
        self.started_mono, self.started_unix = started_mono, started_unix
        self.start_robot = tuple(state["robot"])
        self.start_junction = tuple(state["position"])
        if len(self.start_junction) != 3 or not all(map(math.isfinite, self.start_junction)):
            raise ValueError("起始 XYZ 无效")
        self.start_direction = needle_direction_from_robot_encoders(self.start_robot)
        self.start_tip = self.tip(self.start_robot, self.start_junction)
        self.axial_control = mode in ("c_axis", "junction_axis")
        self.drive_vector = (self.start_direction if self.axial_control or mode == "junction_abc" else (1.0, 1.0, 1.0)
                             if mode == "junction_xyz" else
                             tuple(float(i == "xyz".index(mode[-1])) for i in range(3)))
        if xyz_amplitudes_um is not None:
            amplitudes = tuple(float(v) for v in xyz_amplitudes_um)
            if (mode != "junction_xyz" or len(amplitudes) != 3 or
                    not all(math.isfinite(v) and abs(v) <= 20 for v in amplitudes) or
                    not any(amplitudes)):
                raise ValueError("基点 XYZ 独立幅值需三项有限数，绝对值≤20 µm，至少一项非零")
            self.drive_vector = tuple(v / signal.amplitude_um for v in amplitudes)
        if drive_vector_override is not None:
            override = tuple(float(v) for v in drive_vector_override)
            if (len(override) != 3 or
                    not all(math.isfinite(v) for v in override) or
                    not any(abs(v) > 1e-12 for v in override)):
                raise ValueError("响应驱动向量必须为三个有限数且不能全为零")
            self.drive_vector = override
        self.reference_scale = _norm(self.drive_vector)
        self.direction = tuple(v / self.reference_scale for v in self.drive_vector)
        self.start_position = self.start_junction if mode.startswith("junction") else self.start_tip
        self.handles = tuple(handles)
        if len(handles) != 4:
            raise ValueError("需要四轴手柄值")
        stepped_values(handles, AXES, (0, 0, 0, 0))
        self.metadata = dict(metadata or {})
        self.feedback, self.decisions, self.commands = [], [], []
        self.status, self.reason, self.fault = "running", "", None
        self.latest = None
        self.last_sequence = None
        self.last_tick = None
        self.last_progress = started_mono
        self.progress_position = self.start_position
        self.settle_start, self.settle_count = None, 0
        self.final_stable = False
        self.stopped_mono = None
        self.preflight()

    def tip(self, robot, junction):
        return tuple(needle_pose_from_robot_encoders(
            robot, junction, self.tip_offset_mm, self.offset_sign)["display_tip"])

    def preflight(self):
        """Validate the complete scalar range before any publication."""
        validate_manual_jog(self.start_robot, (0, 0, 0, 0))
        low, high = self.signal.bounds_um
        model_tip = model_tip_from_robot_encoders(self.start_robot, self.tip_offset_mm, self.offset_sign)
        model_junction = model_tip_from_robot_encoders(self.start_robot, 0.0)
        self.tip_envelope_um = self.junction_envelope_um = 0.0
        for index in range(129):
            scalar = (low + (high - low) * index / 128) / 1000
            delta = tuple(scalar * v for v in self.drive_vector)
            try:
                if self.axial_control:
                    handles = (0, 0, round(scalar * C_HANDLE_COUNTS_PER_MM), 0)
                    robot = validate_manual_jog(self.start_robot, handles)
                else:
                    if self.mode.startswith("junction"):
                        handles, robot = junction_handle_target(self.start_robot, (0, 0, 0, 0), delta)
                    else:
                        _, handles, robot = cartesian_handle_target(
                            self.start_robot, (0, 0, 0, 0), self.start_junction, delta,
                            tip_offset_mm=self.tip_offset_mm, tip_offset_sign=self.offset_sign)
                    for axis, current, predicted in zip(AXES[:3], self.start_robot[:3], robot[:3]):
                        low_limit, high_limit = ROBOT_ENCODER_LIMITS[axis]
                        if not low_limit <= predicted <= high_limit:
                            raise ValueError(
                                "%s 轴预测目标编码器 %d 超出安全范围 [%d, %d]；当前反馈编码器 %d 在范围内" %
                                (axis, predicted, low_limit, high_limit, current))
                    validate_cartesian_delta(self.start_robot, handles, self.tip_offset_mm,
                                             self.offset_sign, max_step_mm=0.2)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(
                    "实验目标预检失败：当前反馈限位检查已通过；参考 XYZ=(%+.4f, %+.4f, %+.4f) µm，"
                    "模型起点距 RCM %.6f mm。%s。请调整可达的实验方向/幅值或起点后重新预检" %
                    (*tuple(v * 1000 for v in delta), _norm(model_junction), error)) from error
            self.tip_envelope_um = max(self.tip_envelope_um, 1000 * _norm(_sub(
                model_tip_from_robot_encoders(robot, self.tip_offset_mm, self.offset_sign), model_tip)))
            self.junction_envelope_um = max(self.junction_envelope_um, 1000 * _norm(_sub(
                model_tip_from_robot_encoders(robot, 0.0), model_junction)))

    def reference_at(self, elapsed):
        scalar, phase = self.signal.at(elapsed)
        xyz = tuple(scalar * v for v in self.drive_vector)
        return dict(reference_um=scalar * self.reference_scale, phase=phase, reference_xyz_um=xyz,
                    target_mm=tuple(self.start_position[i] + xyz[i] / 1000 for i in range(3)))

    def control_error(self, row, error, reference):
        return _norm(error) * 1000

    def stable_at_origin(self, row, displacement):
        return _norm(displacement) * 1000 <= self.control.tolerance_um

    def command_delta(self, row, correction, reference):
        if self.axial_control:
            # C encoder sign follows the physical needle direction.  The
            # response drive vector may be negative for a signed workpoint
            # correction, so project onto the fixed start direction rather
            # than onto the (possibly reversed) reference direction.
            axial = sum(a * b for a, b in zip(correction, self.start_direction))
            delta = (0, 0, round(axial * C_HANDLE_COUNTS_PER_MM), 0)
            return stepped_values(self.handles, AXES, delta), delta
        if self.mode.startswith("junction"):
            handles, _ = junction_handle_target(row["robot"], self.handles, correction)
        else:
            _, handles, _ = cartesian_handle_target(
                row["robot"], self.handles, row["junction_mm"], correction,
                tip_offset_mm=self.tip_offset_mm, tip_offset_sign=self.offset_sign)
        return handles, tuple(a - b for a, b in zip(handles, self.handles))

    def validate_dispatch(self, decision, now):
        """Validate a pending command immediately before publishing it.

        The base-position controller overrides this with its additional native
        joint envelope checks.  The shared response engine also needs the same
        freshness, fault, cumulative-origin and packet-limit checks for the
        C-axis workpoint preparation path and the other generic response modes.
        """
        with self.lock:
            if self.status != "running":
                return False
            reason = self.fault
            if (
                self.latest is None
                or now - self.latest["mono_time"] > self.control.state_timeout_s
            ):
                reason = reason or "feedback_timeout"
            if tuple(a - b for a, b in zip(decision["handles"], self.handles)) != tuple(
                decision["delta"]
            ):
                reason = reason or "stale_command_origin"
            if reason:
                self.stop(reason, now)
                return False
            try:
                validate_cartesian_delta(
                    self.latest["robot"],
                    decision["delta"],
                    self.tip_offset_mm,
                    self.offset_sign,
                    max_step_mm=self.control.correction_limit_um / 1000,
                )
            except (ValueError, TypeError, OverflowError) as error:
                self.stop("invalid_command: " + str(error), now)
                return False
            return True

    def feedback_details(self, row, reference):
        return {}

    def observe(self, state, unix_time, mono_time, sequence):
        with self.lock:
            if self.status != "running" or mono_time < self.started_mono:
                return
            if self.latest and sequence <= self.latest["sequence"]:
                return
            if len(self.feedback) >= self.control.max_records:
                self.fault = "record_capacity_exceeded"
                return
            try:
                junction = tuple(state["position"])
                robot, handle = tuple(state["robot"]), tuple(state["handle"])
                if not all(map(math.isfinite, junction)):
                    raise ValueError("反馈坐标非有限值")
                validate_manual_jog(robot, (0, 0, 0, 0))
                tip = self.tip(robot, junction)
                position = junction if self.mode.startswith("junction") else tip
                displacement = _sub(position, self.start_position)
                actual = sum(a * b for a, b in zip(displacement, self.direction)) * 1000
                transverse = _norm(tuple(displacement[i] * 1000 - actual * self.direction[i]
                                         for i in range(3)))
                if self.latest and mono_time - self.latest["mono_time"] > self.control.state_timeout_s:
                    self.fault = "feedback_gap_exceeded"
                elapsed = mono_time - self.started_mono
                reference = self.reference_at(elapsed)
                ref, phase = reference["reference_um"], reference["phase"]
                reference_xyz = reference["reference_xyz_um"]
                actual_xyz = tuple(v * 1000 for v in displacement)
                error_xyz = _sub(reference_xyz, actual_xyz)
                angle = math.degrees(math.acos(max(-1, min(1, sum(
                    a * b for a, b in zip(self.start_direction, needle_direction_from_robot_encoders(robot)))))))
                row = dict(elapsed_s=elapsed, unix_time=unix_time, mono_time=mono_time,
                           sequence=sequence, reference_um=ref, phase=phase,
                           reference_xyz_um=reference_xyz, actual_xyz_um=actual_xyz, error_xyz_um=error_xyz,
                           target_mm=tuple(self.start_position[i] + reference_xyz[i] / 1000 for i in range(3)),
                           actual_um=actual, error_um=ref - actual,
                           error_norm_um=_norm(error_xyz),
                           transverse_um=transverse, posture_deg=angle,
                           junction_mm=junction, tip_mm=tip, robot=robot, handle=handle,
                           forward=bool(state["forward"]), backward=bool(state["backward"]))
                row.update(self.feedback_details(row, reference))
                self.feedback.append(row)
                self.latest = row
                low, high = (v * self.reference_scale for v in self.signal.bounds_um)
                margin = self.control.envelope_margin_um
                if actual < low - margin or actual > high + margin:
                    self.fault = "position_envelope"
                if transverse > self.control.transverse_limit_um:
                    self.fault = "transverse_envelope"
                if _norm(_sub(tip, self.start_tip)) * 1000 > self.tip_envelope_um + margin:
                    self.fault = "needle_tip_envelope"
                if _norm(_sub(junction, self.start_junction)) * 1000 > self.junction_envelope_um + margin:
                    self.fault = "junction_envelope"
                if self.axial_control and angle > self.control.posture_limit_deg:
                    self.fault = "fixed_ab_posture_drift"
                if robot[3] != self.start_robot[3]:
                    self.fault = "unexpected_e_motion"
                stable = phase == "recovery" and self.stable_at_origin(row, displacement)
                if stable:
                    if self.settle_start is None:
                        self.settle_start = mono_time
                    self.settle_count += 1
                    self.final_stable = (self.settle_count >= self.control.settle_samples and
                                         mono_time - self.settle_start >= self.control.settle_s)
                else:
                    self.settle_start, self.settle_count, self.final_stable = None, 0, False
            except (TypeError, ValueError, OverflowError) as error:
                self.fault = "invalid_feedback: " + str(error)

    def stop(self, reason, now, status="stopped"):
        with self.lock:
            if self.status == "running":
                self.status, self.reason, self.stopped_mono = status, reason, now

    def tick(self, now):
        """Return a decision once per fresh feedback, or None. Never publish."""
        with self.lock:
            if self.status != "running":
                return None
            if self.fault:
                self.stop(self.fault, now)
                return None
            if self.latest is None or now - self.latest["mono_time"] > self.control.state_timeout_s:
                self.stop("feedback_timeout", now)
                return None
            elapsed = now - self.started_mono
            if elapsed >= self.signal.duration_s:
                self.stop("noise_recorded" if self.signal.kind == "noise" else
                          "schedule_finished" if self.final_stable else "schedule_finished_unsettled",
                          now, "completed" if self.signal.kind == "noise" or self.final_stable else "unsettled")
                return None
            if self.last_tick is not None and now - self.last_tick < self.control.interval_s - 1e-9:
                return None
            self.last_tick = now
            row = self.latest
            if row["sequence"] == self.last_sequence:
                return None
            self.last_sequence = row["sequence"]
            reference = self.reference_at(elapsed)
            ref, phase = reference["reference_um"], reference["phase"]
            reference_xyz = reference["reference_xyz_um"]
            position = row["junction_mm"] if self.mode.startswith("junction") else row["tip_mm"]
            target = reference["target_mm"]
            error = _sub(target, position)
            error_um = self.control_error(row, error, reference)
            decision = dict(elapsed_s=elapsed, mono_time=now, reference_um=ref, phase=phase,
                            reference_xyz_um=reference_xyz,
                            feedback_sequence=row["sequence"], feedback_age_s=now - row["mono_time"],
                            target_mm=target, error_norm_um=_norm(error) * 1000, delta=(0, 0, 0, 0),
                            handles=self.handles, action="noise" if self.signal.kind == "noise" else "deadband")
            decision.update({k: v for k, v in reference.items() if k.startswith("joint_")})
            self.decisions.append(decision)
            if len(self.decisions) >= self.control.max_records:
                self.stop("record_capacity_exceeded", now)
                return None
            if self.signal.kind == "noise":
                return decision
            # A fixed-start displacement watchdog works across direction
            # reversals; a moving reference cannot masquerade as progress.
            if error_um <= self.control.tolerance_um or _norm(_sub(position, self.progress_position)) * 1000 >= self.control.movement_um:
                self.last_progress, self.progress_position = now, position
            elif now - self.last_progress >= self.control.no_progress_s:
                self.stop("no_physical_progress", now)
                return None
            if error_um <= self.control.tolerance_um:
                return decision
            scale = min(1, self.control.correction_limit_um / error_um)
            correction = tuple(v * scale for v in error)
            try:
                handles, delta = self.command_delta(row, correction, reference)
                validate_cartesian_delta(row["robot"], delta, self.tip_offset_mm,
                                         self.offset_sign, max_step_mm=self.control.correction_limit_um / 1000)
                decision.update(delta=delta, handles=handles,
                                action="send" if any(delta) else "quantized_zero")
            except (ValueError, TypeError, OverflowError) as error_value:
                self.stop("invalid_command: " + str(error_value), now)
                return None
            return decision

    def published(self, decision, mono_time, unix_time, success=True):
        """Record publish API invocation, not hardware receipt or topic echo."""
        with self.lock:
            if success:
                self.handles = tuple(decision["handles"])
            self.commands.append(dict(decision, publish_mono=mono_time,
                                      publish_elapsed_s=mono_time - self.started_mono,
                                      publish_unix=unix_time, publish_success=success))

    def snapshot(self):
        with self.lock:
            return dict(schema_version=1, signal=asdict(self.signal), control=asdict(self.control),
                        mode=self.mode, metadata=self.metadata, started_unix=self.started_unix,
                        started_mono=self.started_mono, stopped_mono=self.stopped_mono,
                        status=self.status, reason=self.reason,
                        tip_offset_mm=self.tip_offset_mm, offset_sign=self.offset_sign,
                        origin_junction_mm=self.start_junction, origin_tip_mm=self.start_tip,
                        direction=self.direction, drive_vector=self.drive_vector, reference_scale=self.reference_scale,
                        control_object="junction" if self.mode.startswith("junction") else "needle_tip",
                        tip_envelope_um=self.tip_envelope_um,
                        junction_envelope_um=self.junction_envelope_um, feedback=list(self.feedback),
                        decisions=list(self.decisions), commands=list(self.commands))
