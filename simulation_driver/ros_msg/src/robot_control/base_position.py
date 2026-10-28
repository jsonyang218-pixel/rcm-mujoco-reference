"""Base XYZ residual controller using the normal trajectory update mechanism."""

import math

from .joint_response import joint_coordinates
from .kinematics import (
    AXES,
    model_tip_from_robot_encoders,
    validate_cartesian_delta,
    validate_manual_jog,
)
from .response_experiments import (
    ResponseExperiment, junction_handle_target, response_state_key,
    stationary_protocol_origin,
)


WORKPOINT_TARGET_UM = 500.0
WORKPOINT_TOLERANCE_UM = 25.0
# The real controller applies a changed handle target over several feedback
# frames.  Keep each response packet within the largest increments observed in
# the completed base-position sine runs; this also keeps the native A/B jump
# watchdog meaningful for step and pulse inputs.
MAX_HANDLE_STEP = (8, 9, 1, 0)


def stationary_base_protocol_origin(history, state, now, timeout,
                                    tip_offset_mm=4.0, offset_sign=-1.0):
    """Freeze an unchanged wire session despite bounded encoder hold ripple.

    One second of continuous feedback is required. Both base and reconstructed
    tip must stay in a 0.25 um box diagonal, with half-window mean drift <=
    0.05 um. Native limits and unchanged E also protect degenerate geometry.
    Legacy experiments retain their exact-state startup check.
    """
    key = response_state_key(state)
    before = [i for i, (received, _) in enumerate(history) if received <= now - 1.0]
    if not before:
        raise ValueError("基点实验启动前需要至少 1 s 连续静止反馈")
    recent = history[before[-1]:]
    # Reuse continuity, freshness, exact handle echo and direction checks.
    protocol = [(t, (key[0], sample[1], key[2], sample[3], sample[4]))
                for t, sample in recent]
    origin = stationary_protocol_origin(protocol, state, now, timeout, duration=1.0)
    samples = [sample for _, sample in recent] + [key]
    joints, base, tip = [], [], []
    for sample in samples:
        robot, _, point, _, _ = sample
        validate_manual_jog(robot, (0, 0, 0, 0))
        if robot[3] != key[0][3] or not all(map(math.isfinite, point)):
            raise ValueError("静止反馈含 E 轴变化或非法坐标")
        joints.append(joint_coordinates(robot))
        base.append(point)
        radial = model_tip_from_robot_encoders(robot, 0)
        modeled_tip = model_tip_from_robot_encoders(robot, tip_offset_mm, offset_sign)
        tip.append(tuple(point[i] + modeled_tip[i] - radial[i] for i in range(3)))
    span = lambda points: math.sqrt(sum(
        (max(p[i] for p in points) - min(p[i] for p in points)) ** 2
        for i in range(3)))
    if any(max(q[i] for q in joints) - min(q[i] for q in joints) > limit
           for i, limit in enumerate((0.003, 0.003, 0.25))):
        raise ValueError("启动前原生轴波动超出静止范围")
    split = sum(t < now - 0.5 for t, _ in recent)
    for points in (base, tip):
        if span(points) * 1000 > 0.25 or not 0 < split < len(points) - 1:
            raise ValueError("启动前基点或针尖波动超过 0.25 µm")
        means = [tuple(sum(p[i] for p in part) / len(part) for i in range(3))
                 for part in (points[:split], points[split:-1])]
        if math.dist(*means) * 1000 > 0.05:
            raise ValueError("启动前基点或针尖持续漂移")
    return origin


class BasePositionExperiment(ResponseExperiment):
    """Independent XYZ amplitudes; the last successful send owns the wire base."""

    def __init__(
        self,
        signal,
        state,
        handles,
        started_mono,
        started_unix,
        xyz_amplitudes_um=(10.0, 6.0, 4.0),
        **kwargs
    ):
        kwargs.pop("mode", None)
        # Active base-position experiments must start away from the RCM
        # singularity.  Workpoint preparation uses the dedicated C-axis
        # response controller in the CLI and therefore never needs an
        # exception here.
        kwargs.pop("allow_near_rcm", None)
        self.origin_handles = tuple(handles)
        super().__init__(
            signal,
            state,
            handles,
            started_mono,
            started_unix,
            mode="junction_xyz",
            xyz_amplitudes_um=xyz_amplitudes_um,
            **kwargs
        )
        radius_um = math.dist(
            model_tip_from_robot_encoders(self.start_robot, 0), (0, 0, 0)
        ) * 1000.0
        self.q0 = joint_coordinates(self.start_robot)
        self.progress_q = list(self.q0)
        self.progress_time = [started_mono] * 3
        self.metadata.update(
            controller="base_position_residual",
            input_kind="elapsed-time Cartesian position reference",
            workpoint_target_um=WORKPOINT_TARGET_UM,
            workpoint_tolerance_um=WORKPOINT_TOLERANCE_UM,
            workpoint_radius_um=radius_um,
            xyz_amplitudes_um=tuple(x * signal.amplitude_um for x in self.drive_vector),
            max_handle_step=MAX_HANDLE_STEP,
        )

    def preflight(self):
        radius_um = math.dist(
            model_tip_from_robot_encoders(self.start_robot, 0), (0, 0, 0)
        ) * 1000.0
        if (
            self.signal.kind != "noise"
            and math.dist(model_tip_from_robot_encoders(self.start_robot, 0), (0, 0, 0))
            < 0.2
        ):
            raise ValueError("基点距模型 RCM 小于 200 µm；请先预检并准备非零轴向工作点")
        if (
            self.signal.kind != "noise"
            and abs(radius_um - WORKPOINT_TARGET_UM) > WORKPOINT_TOLERANCE_UM
        ):
            raise ValueError(
                "基点工作点为 %.1f µm，要求与正弦实验一致为 %.0f±%.0f µm；"
                "请先准备 500 µm 轴向工作点" %
                (radius_um, WORKPOINT_TARGET_UM, WORKPOINT_TOLERANCE_UM)
            )
        super().preflight()
        q0 = joint_coordinates(self.start_robot)
        self.native_bounds = [[v, v] for v in q0]
        low, high = self.signal.bounds_um
        for k in range(257):
            fraction = low + (high - low) * k / 256
            delta = tuple(fraction * v / 1000 for v in self.drive_vector)
            _, robot = junction_handle_target(self.start_robot, self.handles, delta)
            for i, value in enumerate(joint_coordinates(robot)):
                self.native_bounds[i][0] = min(self.native_bounds[i][0], value)
                self.native_bounds[i][1] = max(self.native_bounds[i][1], value)

    def observe(self, state, unix_time, mono_time, sequence):
        with self.lock:
            previous = self.latest
            super().observe(state, unix_time, mono_time, sequence)
            row = self.latest
            if row is None or row is previous or self.status != "running":
                return
            q = joint_coordinates(row["robot"])
            if row["forward"] or row["backward"]:
                self.fault = self.fault or "unexpected_direction_flag"
            for i, (value, margin) in enumerate(
                zip(q, (0.05, 0.05, self.control.envelope_margin_um))
            ):
                if (
                    not self.native_bounds[i][0] - margin
                    <= value
                    <= self.native_bounds[i][1] + margin
                ):
                    self.fault = self.fault or "joint_" + AXES[i] + "_envelope"
                if (
                    previous
                    and abs(value - joint_coordinates(previous["robot"])[i])
                    > (0.03, 0.03, 5.0)[i]
                ):
                    self.fault = self.fault or "joint_" + AXES[i] + "_jump"
                if (
                    abs(value - self.progress_q[i])
                    >= (0.0002, 0.0002, self.control.movement_um)[i]
                ):
                    self.progress_q[i], self.progress_time[i] = value, mono_time

    def tick(self, now):
        with self.lock:
            decision = super().tick(now)
            if decision is None or self.status != "running":
                return decision
            if decision["error_norm_um"] <= self.control.tolerance_um:
                self.progress_time = [now] * 3
                return decision
            try:
                radius_um = (
                    math.dist(
                        model_tip_from_robot_encoders(self.latest["robot"], 0),
                        (0, 0, 0),
                    )
                    * 1000
                )
                angle_tolerance = math.degrees(
                    self.control.tolerance_um / max(radius_um, 1.0)
                )
                native_tolerance = (
                    max(0.01, angle_tolerance),
                    max(0.005, angle_tolerance),
                    1.0,
                )
                correction = tuple(
                    decision["target_mm"][i] - self.latest["junction_mm"][i]
                    for i in range(3)
                )
                _, desired = junction_handle_target(
                    self.latest["robot"], self.handles, correction
                )
                error = tuple(
                    a - b
                    for a, b in zip(
                        joint_coordinates(desired),
                        joint_coordinates(self.latest["robot"]),
                    )
                )
                for i, value in enumerate(error):
                    # A reference moving within tolerance must not accumulate
                    # fictitious inactivity; motion on C cannot mask stalled A/B.
                    if abs(value) <= native_tolerance[i]:
                        self.progress_time[i] = now
                    elif now - self.progress_time[i] >= self.control.no_progress_s:
                        self.stop("no_physical_progress_" + AXES[i], now)
                        return None
            except (ValueError, TypeError, OverflowError) as error:
                self.stop("invalid_reference: " + str(error), now)
                return None
            return decision

    def command_delta(self, row, correction, reference):
        error = None
        for power in range(9):
            step = tuple(v / (2**power) for v in correction)
            handles, _ = junction_handle_target(row["robot"], self.handles, step)
            delta = tuple(a - b for a, b in zip(handles, self.handles))
            try:
                validate_cartesian_delta(
                    row["robot"],
                    delta,
                    self.tip_offset_mm,
                    self.offset_sign,
                    max_step_mm=self.control.correction_limit_um / 1000,
                )
                limited = tuple(
                    max(-limit, min(limit, value))
                    for value, limit in zip(delta, MAX_HANDLE_STEP)
                )
                if limited != delta:
                    handles = tuple(a + b for a, b in zip(self.handles, limited))
                    validate_cartesian_delta(
                        row["robot"],
                        limited,
                        self.tip_offset_mm,
                        self.offset_sign,
                        max_step_mm=self.control.correction_limit_um / 1000,
                    )
                return handles, limited
            except ValueError as value:
                error = value
        raise ValueError("无法在单包针尖限幅内换算基点修正：" + str(error))

    def validate_dispatch(self, decision, now):
        """Check under the same lock as feedback immediately before send."""
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

    def published(self, decision, mono_time, unix_time, success=True):
        super().published(decision, mono_time, unix_time, success)
        if not success:
            self.stop("publish_failed", mono_time)

    def snapshot(self):
        result = super().snapshot()
        result.update(
            origin_robot=self.start_robot,
            origin_handles=self.origin_handles,
            native_bounds=self.native_bounds,
            final_stable=self.final_stable,
            controller="base_position_residual",
        )
        return result
