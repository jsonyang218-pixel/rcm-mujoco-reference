#!/usr/bin/env python3
"""Nominal RCM kinematics recovered from the two supplied dingdian programs.

Python 3.11+, standard library only. Lengths are mm; JointState uses degrees.
The encoder origin/sign defaults reproduce the source IK, NOT a calibrated robot.
This module is offline: no ROS import, device connection, or hardware publishing.
The handle formulas describe the algebra inferred from reverse_A/B/C; they do
not identify the motor plant or certify the downstream controller semantics.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

Vec3 = tuple[float, float, float]


def _triple(values: Sequence[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError("Expected exactly three values")
    result = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in result):
        raise ValueError("Values must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class Geometry:
    link1_mm: float = 60.0
    link2_mm: float = 52.0
    offset_mm: float = 16.0
    b_reference_mm: float = 130.0 - 31.52
    tilt_offset_deg: float = 22.0
    c_reference_mm: float = 160.45 - 146.45


@dataclass(frozen=True)
class EncoderMap:
    """E = origin + direction * nominal_encoder; origins/signs need calibration."""

    a_counts_per_degree: float = 2000.0 * 100.0 / 360.0
    bc_counts_per_mm: float = 4.0 * 1024.0 * 15.5817175
    origin: Vec3 = (0.0, 0.0, 0.0)
    direction: Vec3 = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        _triple(self.origin)
        _triple(self.direction)
        if any(s not in (-1.0, 1.0) for s in self.direction):
            raise ValueError("Encoder directions must each be +1 or -1")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.a_counts_per_degree, self.bc_counts_per_mm)):
            raise ValueError("Encoder scales must be finite and positive")


@dataclass(frozen=True)
class JointState:
    phi_deg: float
    alpha_deg: float
    length_mm: float

    def __post_init__(self) -> None:
        _triple((self.phi_deg, self.alpha_deg, self.length_mm))


class RCMModel:
    """Three task coordinates: phi, nonlinear B linkage angle alpha, and L.

    alpha=40..90 deg selects the positive-root linkage branch in the source.
    The limits below are source-declared limits, not verified hardware limits.
    No needle roll or full spatial linkage assembly can be identified here.
    """

    def __init__(self, geometry: Geometry | None = None,
                 encoders: EncoderMap | None = None) -> None:
        self.geometry = geometry or Geometry()
        self.encoders = encoders or EncoderMap()

    @staticmethod
    def within_source_limits(q: JointState, tolerance: float = 1e-9) -> bool:
        return (-45.0 - tolerance <= q.phi_deg <= 45.0 + tolerance
                and 40.0 - tolerance <= q.alpha_deg <= 90.0 + tolerance
                and -tolerance <= q.length_mm <= 44.0 + tolerance)

    def _check(self, q: JointState, enforce_limits: bool) -> None:
        if q.length_mm < 0:
            raise ValueError("Negative RCM-to-tip length is outside this model")
        if enforce_limits and not self.within_source_limits(q):
            raise ValueError("State is outside the source-declared workspace")

    def b_displacement(self, alpha_deg: float) -> float:
        g = self.geometry
        if not math.isfinite(alpha_deg):
            raise ValueError("Angle must be finite")
        a = math.radians(alpha_deg)
        vertical = g.link1_mm * math.sin(a) - g.offset_mm
        radicand = g.link2_mm ** 2 - vertical ** 2
        if radicand < -1e-9:
            raise ValueError("B linkage has no real-valued configuration")
        return (g.b_reference_mm - g.link1_mm * math.cos(a)
                - math.sqrt(max(0.0, radicand)))

    def alpha_candidates_from_b(self, displacement_mm: float) -> tuple[float, ...]:
        """Both triangle solutions that reproduce the positive-root B formula.

        B displacement is not one-to-one across its minimum. A previous or
        requested angle is needed to decode counts on the other branch.
        """
        g = self.geometry
        if not math.isfinite(displacement_mm):
            raise ValueError("Displacement must be finite")
        ad = g.b_reference_mm - displacement_mm
        ac = math.hypot(ad, g.offset_mm)
        if ac < 1e-12:
            raise ValueError("Degenerate B linkage triangle")
        c = (g.link1_mm ** 2 + ac ** 2 - g.link2_mm ** 2) / (2*g.link1_mm*ac)
        if not -1.0 - 1e-10 <= c <= 1.0 + 1e-10:
            raise ValueError("B displacement cannot form the linkage triangle")
        beta = math.acos(max(-1.0, min(1.0, c)))
        gamma = math.atan2(g.offset_mm, ad)
        candidates = []
        for angle in (gamma+beta, gamma-beta):
            a = math.degrees(angle)
            if abs(self.b_displacement(a)-displacement_mm) <= 1e-7 and not any(
                    abs(a-previous) < 1e-9 for previous in candidates):
                candidates.append(a)
        if not candidates:
            raise ValueError("Displacement belongs to a different assembly branch")
        return tuple(candidates)

    def alpha_from_b(self, displacement_mm: float,
                     enforce_limits: bool = False,
                     reference_alpha_deg: float | None = None) -> float:
        """Keep the source branch by default; use an angle hint when supplied."""
        candidates = self.alpha_candidates_from_b(displacement_mm)
        if reference_alpha_deg is None:
            a = candidates[0]
        else:
            if not math.isfinite(reference_alpha_deg):
                raise ValueError("Reference B angle must be finite")
            a = min(candidates, key=lambda angle: abs(angle-reference_alpha_deg))
        if enforce_limits and not 40.0 - 1e-9 <= a <= 90.0 + 1e-9:
            raise ValueError("B angle is outside the source-declared range")
        return a

    def needle_axis(self, q: JointState) -> Vec3:
        f = math.radians(q.phi_deg)
        t = math.radians(q.alpha_deg + self.geometry.tilt_offset_deg)
        return math.sin(t)*math.sin(f), math.cos(t), math.sin(t)*math.cos(f)

    def forward(self, q: JointState, enforce_limits: bool = True) -> Vec3:
        self._check(q, enforce_limits)
        return tuple(q.length_mm*v for v in self.needle_axis(q))  # type: ignore[return-value]

    def inverse(self, point_mm: Sequence[float],
                enforce_limits: bool = True,
                reference_state: JointState | None = None) -> JointState:
        x, y, z = _triple(point_mm)
        length = math.sqrt(x*x + y*y + z*z)
        if length < 1e-12:
            if reference_state is not None:
                q = JointState(reference_state.phi_deg, reference_state.alpha_deg, 0.)
                self._check(q, enforce_limits)
                return q
            raise ValueError("At the RCM point, position does not determine orientation")
        if math.hypot(x, z) < 1e-12:
            if reference_state is None:
                raise ValueError("On the Y axis, position does not determine phi")
            phi = reference_state.phi_deg
        else:
            phi = math.degrees(math.atan2(x, z))
        q = JointState(phi,
                       math.degrees(math.acos(max(-1.0, min(1.0, y/length))))
                       - self.geometry.tilt_offset_deg,
                       length)
        self._check(q, enforce_limits)
        return q

    def to_encoders(self, q: JointState, integer: bool = True,
                    enforce_limits: bool = True) -> tuple[int, int, int] | Vec3:
        self._check(q, enforce_limits)
        e, g = self.encoders, self.geometry
        nominal = (q.phi_deg*e.a_counts_per_degree,
                   self.b_displacement(q.alpha_deg)*e.bc_counts_per_mm,
                   (g.c_reference_mm+q.length_mm)*e.bc_counts_per_mm)
        result = tuple(o+s*v for o, s, v in zip(e.origin, e.direction, nominal))
        # int truncation reproduces the original zero-origin source IK.
        return tuple(int(v) for v in result) if integer else result  # type: ignore[return-value]

    def from_encoders(self, values: Sequence[float],
                      enforce_limits: bool = False,
                      reference_alpha_deg: float | None = None) -> JointState:
        e = self.encoders
        values = _triple(values)
        n = tuple((v-o)/s for v, o, s in zip(values, e.origin, e.direction))
        length = n[2]/e.bc_counts_per_mm-self.geometry.c_reference_mm
        # A source int-truncated L=0 can decode just below zero. Clamp only
        # this one-count quantization interval; larger negative L is rejected.
        if -1/e.bc_counts_per_mm < length < 0:
            length = 0.0
        q = JointState(n[0]/e.a_counts_per_degree,
                       self.alpha_from_b(n[1]/e.bc_counts_per_mm,
                                         reference_alpha_deg=reference_alpha_deg), length)
        # Integer truncation can place a boundary state slightly outside the
        # nominal range. Decode defaults to no workspace enforcement; callers
        # must evaluate limits with the calibration/quantization budget.
        self._check(q, enforce_limits)
        return q

    def forward_encoders(self, values: Sequence[float],
                         enforce_limits: bool = False) -> Vec3:
        return self.forward(self.from_encoders(values, enforce_limits), enforce_limits)

    def inverse_encoders(self, point_mm: Sequence[float], integer: bool = True,
                         enforce_limits: bool = True) -> tuple[int, int, int] | Vec3:
        return self.to_encoders(self.inverse(point_mm, enforce_limits), integer, enforce_limits)

    def jacobian_q(self, q: JointState) -> tuple[Vec3, Vec3, Vec3]:
        """Rows X,Y,Z; columns phi_rad, alpha_rad, L_mm (not degrees)."""
        f = math.radians(q.phi_deg)
        t = math.radians(q.alpha_deg+self.geometry.tilt_offset_deg)
        l, sf, cf, st, ct = q.length_mm, math.sin(f), math.cos(f), math.sin(t), math.cos(t)
        return ((l*st*cf, l*ct*sf, st*sf),
                (0.0, -l*st, ct),
                (-l*st*sf, l*ct*cf, st*cf))

    def jacobian_encoders(self, values: Sequence[float]) -> tuple[Vec3, Vec3, Vec3]:
        """Rows X,Y,Z in mm; columns E_A,E_B,E_C in counts."""
        q = self.from_encoders(values)
        g, e = self.geometry, self.encoders
        a = math.radians(q.alpha_deg)
        v = g.link1_mm*math.sin(a)-g.offset_mm
        root = math.sqrt(g.link2_mm**2-v**2)
        if root < 1e-12:
            raise ValueError("B linkage derivative is singular")
        db_da = g.link1_mm*math.sin(a)+v*g.link1_mm*math.cos(a)/root
        if abs(db_da) < 1e-12:
            raise ValueError("B displacement is locally insensitive to angle")
        scales = (e.direction[0]*math.pi/(180*e.a_counts_per_degree),
                  e.direction[1]/(e.bc_counts_per_mm*db_da),
                  e.direction[2]/e.bc_counts_per_mm)
        return tuple(tuple(v*s for v, s in zip(row, scales))
                     for row in self.jacobian_q(q))  # type: ignore[return-value]


def synchronized_encoder_step(current: Sequence[float], target: Sequence[float],
                              tolerances: Sequence[float] = (20, 150, 60),
                              max_steps: Sequence[float] = (200, 8000, 400)) -> Vec3:
    """Exact numeric step kernel from convergence_fixed; no hardware publishing."""
    current, target = _triple(current), _triple(target)
    tolerances, max_steps = _triple(tolerances), _triple(max_steps)
    if any(v < 0 for v in tolerances) or any(v <= 0 for v in max_steps):
        raise ValueError("Tolerances must be nonnegative and max steps positive")
    error = tuple(t-c if abs(t-c) > tol else 0.0
                  for c, t, tol in zip(current, target, tolerances))
    scale = max(1.0, *(abs(v)/s for v, s in zip(error, max_steps)))
    return tuple(v/scale for v in error)  # type: ignore[return-value]


def infer_handle_increment(model: RCMModel, current_encoders: Sequence[float],
                           requested_steps: Sequence[float], gear: float = 10.0,
                           b_gain: float = 0.8) -> Vec3:
    """Signed algebraic inverse of the INFERRED legacy downstream formula.

    Unlike the source's sign(step)*reverse_B(abs(step)), B is solved at the
    signed target. This is an offline candidate formula, not a tested driver.
    b_gain=0.8 may scale a velocity command in the missing downstream code;
    this routine does not interpret it as measured encoder movement per tick.
    Its mapping requires the original positive, zero-origin encoder convention.
    """
    current = _triple(current_encoders)
    steps = _triple(requested_steps)
    if model.encoders.origin != (0.0, 0.0, 0.0) or model.encoders.direction != (1.0, 1.0, 1.0):
        raise ValueError("Handle mapping is only recovered for the source convention")
    if not math.isfinite(gear) or gear <= 0 or not math.isfinite(b_gain) or b_gain <= 0:
        raise ValueError("Gear and inferred B gain must be finite and positive")
    a_scale = 2*gear*3/4000*model.encoders.a_counts_per_degree
    c_scale = 3*gear*10000/4000
    alpha_now = model.alpha_from_b(current[1]/model.encoders.bc_counts_per_mm)
    alpha_target = model.alpha_from_b(
        (current[1]+steps[1]/b_gain)/model.encoders.bc_counts_per_mm)
    b_handle = (alpha_target-alpha_now)*4000/(3*gear)
    return steps[0]/a_scale, b_handle, steps[2]/c_scale


def cartesian_command(actual_mm: Sequence[float], desired_mm: Sequence[float],
                      gain: float = 0.6, max_delta_mm: float = 0.08) -> Vec3:
    """Source Cartesian P feedback kernel; fixed version bypasses it near points."""
    actual, desired = _triple(actual_mm), _triple(desired_mm)
    if not math.isfinite(gain) or gain < 0 or not math.isfinite(max_delta_mm) or max_delta_mm <= 0:
        raise ValueError("Gain must be nonnegative; maximum delta must be positive")
    delta = tuple(gain*(d-a) for a, d in zip(actual, desired))
    length = math.sqrt(sum(v*v for v in delta))
    scale = min(1.0, max_delta_mm/length) if length > 0 else 1.0
    return tuple(a+scale*v for a, v in zip(actual, delta))  # type: ignore[return-value]


def rigid_transform(point_mm: Sequence[float], rotation: Sequence[Sequence[float]],
                    translation_mm: Sequence[float]) -> Vec3:
    """p_destination = R @ p_source + t; require a proper rigid rotation."""
    p, t = _triple(point_mm), _triple(translation_mm)
    if len(rotation) != 3:
        raise ValueError("Rotation must be 3 by 3")
    rows = tuple(_triple(row) for row in rotation)
    for i in range(3):
        for j in range(3):
            if not math.isclose(sum(rows[i][k]*rows[j][k] for k in range(3)),
                                1.0 if i == j else 0.0, abs_tol=1e-8):
                raise ValueError("Rotation must be orthonormal")
    det = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])
           - rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])
           + rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
    if not math.isclose(det, 1.0, abs_tol=1e-8):
        raise ValueError("Rotation must have determinant +1")
    return tuple(sum(row[k]*p[k] for k in range(3))+v
                 for row, v in zip(rows, t))  # type: ignore[return-value]


def demo() -> dict:
    model = RCMModel()
    q = JointState(20, 68, 18)
    tip = model.forward(q)
    encoders = model.to_encoders(q)
    back = model.forward_encoders(encoders)
    lesion_world = (9.5, 1.5, -16.0)
    direct = model.inverse(lesion_world, enforce_limits=False)
    return {
        "scope": "nominal offline model; not calibrated to physical robot",
        "joint_state": asdict(q), "tip_mm_robot_frame": tip,
        "source_convention_encoders": encoders,
        "reconstructed_tip_mm": back,
        "quantized_roundtrip_error_um": 1000*math.dist(tip, back),
        "jacobian_mm_per_encoder_count": model.jacobian_encoders(encoders),
        "current_lesion_if_misread_as_robot_frame": asdict(direct),
        "lesion_within_source_workspace_without_frame_transform": model.within_source_limits(direct),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
