"""A/B/C reference feedback control; XYZ is a forward-model observation.

A is phi in degrees, B is linkage alpha in degrees, C is axial micrometres.
No junction inverse is used, including at the RCM origin.
"""

import math

from .kinematics import (
    AXES, CONTROL_GEAR, C_HANDLE_COUNTS_PER_MM, ENCODER_COUNTS_PER_MM, ROBOT_ENCODER_LIMITS,
    b_alpha_from_robot_encoder, b_distance_from_alpha, model_tip_from_robot_encoders,
    stepped_values, validate_cartesian_delta, validate_manual_jog,
)
from .response_experiments import ResponseExperiment, _norm, _sub


def joint_coordinates(robot):
    return (robot[0] * 360.0 / 200000, b_alpha_from_robot_encoder(robot[1]),
            robot[2] / ENCODER_COUNTS_PER_MM * 1000)


def joint_reference(signal, start_robot, amplitudes, elapsed):
    value, phase = signal.at(elapsed)
    fraction = value / signal.amplitude_um
    offsets = tuple(fraction * a for a in amplitudes)
    origin = joint_coordinates(start_robot)
    target = tuple(a + b for a, b in zip(origin, offsets))
    robot = (round(target[0] * 200000 / 360),
             round(b_distance_from_alpha(target[1]) * ENCODER_COUNTS_PER_MM),
             round(target[2] / 1000 * ENCODER_COUNTS_PER_MM), start_robot[3])
    xyz = tuple(v * 1000 for v in _sub(model_tip_from_robot_encoders(robot, 0),
                                      model_tip_from_robot_encoders(start_robot, 0)))
    return offsets, robot, xyz, phase


class JointResponseExperiment(ResponseExperiment):
    tolerances = (.01, .005, 1.0)

    def __init__(self, *args, joint_amplitudes=(.05, .02, 10.0), **kwargs):
        self.joint_amplitudes = tuple(float(v) for v in joint_amplitudes)
        if (len(self.joint_amplitudes) != 3 or not all(map(math.isfinite, self.joint_amplitudes)) or
                any(abs(v) > .1 for v in self.joint_amplitudes[:2]) or
                self.joint_amplitudes[2] not in (10.0, 20.0)):
            raise ValueError("A/B 幅值必须在 ±0.1° 内，C 幅值必须为 10 或 20 µm")
        kwargs["mode"] = "junction_abc"
        super().__init__(*args, **kwargs)
        if self.joint_amplitudes[2] != self.signal.amplitude_um:
            raise ValueError("C 幅值必须与时间信号幅值一致")

    def reference_at(self, elapsed):
        offsets, robot, xyz, phase = joint_reference(
            self.signal, self.start_robot, self.joint_amplitudes, elapsed)
        return dict(reference_um=sum(a * b for a, b in zip(xyz, self.direction)), phase=phase,
                    reference_xyz_um=xyz, target_mm=tuple(self.start_junction[i] + xyz[i] / 1000 for i in range(3)),
                    joint_reference=offsets, joint_target_robot=robot)

    def feedback_details(self, row, reference):
        actual = _sub(joint_coordinates(row["robot"]), joint_coordinates(self.start_robot))
        error = _sub(reference["joint_reference"], actual)
        # The observed motor envelope is frozen at start, just like XYZ envelopes.
        low, high = self.signal.bounds_um
        for i, margin in enumerate((.05, .05, self.control.envelope_margin_um)):
            bounds = sorted((low / self.signal.amplitude_um * self.joint_amplitudes[i],
                             high / self.signal.amplitude_um * self.joint_amplitudes[i]))
            if not bounds[0] - margin <= actual[i] <= bounds[1] + margin:
                self.fault = "joint_%s_envelope" % AXES[i]
        return dict(joint_reference=reference["joint_reference"], joint_actual=actual, joint_error=error,
                    joint_target_robot=reference["joint_target_robot"])

    def control_error(self, row, error, reference):
        actual = _sub(joint_coordinates(row["robot"]), joint_coordinates(self.start_robot))
        return max(abs(a - b) / (tol + 1e-10) for a, b, tol in zip(reference["joint_reference"], actual, self.tolerances))

    def stable_at_origin(self, row, displacement):
        actual = _sub(joint_coordinates(row["robot"]), joint_coordinates(self.start_robot))
        return (super().stable_at_origin(row, displacement) and
                all(abs(v) <= tol + 1e-10 for v, tol in zip(actual, self.tolerances)))

    def command_delta(self, row, correction, reference):
        actual = _sub(joint_coordinates(row["robot"]), joint_coordinates(self.start_robot))
        error = _sub(reference["joint_reference"], actual)
        counts = (2 * CONTROL_GEAR * 3 / 4000, CONTROL_GEAR * 3 / 4000,
                  1000 / C_HANDLE_COUNTS_PER_MM)
        delta = tuple(0 if abs(e) <= tol + 1e-10 else round(e / resolution)
                      for e, tol, resolution in zip(error, self.tolerances, counts)) + (0,)
        return tuple(stepped_values(self.handles, AXES, delta)), delta

    def preflight(self):
        validate_manual_jog(self.start_robot, (0, 0, 0, 0))
        self.tip_envelope_um = self.junction_envelope_um = 0.0
        low, high = self.signal.bounds_um
        for index in range(129):
            value = low + (high - low) * index / 128
            # All four waveforms are parameterised by this same scalar fraction.
            fraction = value / self.signal.amplitude_um
            offsets = tuple(fraction * a for a in self.joint_amplitudes)
            origin = joint_coordinates(self.start_robot)
            nominal = (round((origin[0] + offsets[0]) * 200000 / 360),
                       round(b_distance_from_alpha(origin[1] + offsets[1]) * ENCODER_COUNTS_PER_MM),
                       round((origin[2] + offsets[2]) / 1000 * ENCODER_COUNTS_PER_MM), self.start_robot[3])
            for axis, current, target in zip(AXES[:3], self.start_robot[:3], nominal[:3]):
                lo, hi = ROBOT_ENCODER_LIMITS[axis]
                if not lo <= target <= hi:
                    raise ValueError("%s 轴参考目标编码器 %d 超出 [%d, %d]；当前反馈 %d，参考偏移 %s" %
                                     (axis, target, lo, hi, current, offsets))
            reference = dict(joint_reference=offsets)
            _, handles = self.command_delta(dict(robot=self.start_robot), (0, 0, 0), reference)
            quantized = validate_cartesian_delta(self.start_robot, handles, self.tip_offset_mm,
                                                self.offset_sign, max_step_mm=self.control.correction_limit_um / 1000)
            for robot in (nominal, quantized):
                for offset, attr in ((0, "junction_envelope_um"), (self.tip_offset_mm, "tip_envelope_um")):
                    distance = _norm(_sub(model_tip_from_robot_encoders(robot, offset, self.offset_sign),
                                          model_tip_from_robot_encoders(self.start_robot, offset, self.offset_sign))) * 1000
                    setattr(self, attr, max(getattr(self, attr), distance))

    def snapshot(self):
        result = super().snapshot()
        result.update(joint_amplitudes=self.joint_amplitudes, joint_units=("deg", "deg", "um"),
                      joint_tolerances=self.tolerances, origin_robot=self.start_robot,
                      reference_kind="forward kinematics of A/B/C time references; XYZ is observed")
        return result
