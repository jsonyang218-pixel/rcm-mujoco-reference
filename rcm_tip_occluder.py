"""Rendered occluder coupled to the needle tip for occlusion experiments.

The occluder is placed with privileged simulator geometry. Its transform is
refreshed before every sensor render, but the perception/control path receives
only the normal RGB-D frame. No occlusion coordinates or coverage labels are
written into the observation.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


GEOM_NAME = "tip_coupled_occluder"
MATERIAL_NAME = "tip_coupled_occluder_material"
HIDDEN_POSITION_M = np.array([10., 10., 10.])


def _finite(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def fixture_settings(config):
    raw = config.get("reference_scene", {}).get("tip_occlusion_fixture", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("reference_scene.tip_occlusion_fixture must be an object")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("tip_occlusion_fixture.enabled must be boolean")
    radius = _finite(raw.get("radius_mm", .35), "tip_occlusion_fixture.radius_mm")
    trigger = _finite(raw.get("trigger_progress", .8),
                      "tip_occlusion_fixture.trigger_progress")
    coverage = _finite(raw.get("target_coverage", 1.),
                       "tip_occlusion_fixture.target_coverage")
    release = _finite(raw.get("release_coverage", .1),
                      "tip_occlusion_fixture.release_coverage")
    front = _finite(raw.get("front_offset_mm", .05),
                    "tip_occlusion_fixture.front_offset_mm")
    angle = _finite(raw.get("screen_direction_deg", 0.),
                    "tip_occlusion_fixture.screen_direction_deg")
    max_active = raw.get("max_active_s")
    if max_active is not None:
        max_active = _finite(max_active, "tip_occlusion_fixture.max_active_s")
    if radius <= 0.:
        raise ValueError("tip_occlusion_fixture.radius_mm must be positive")
    if not 0. <= trigger < 1.:
        raise ValueError("tip_occlusion_fixture.trigger_progress must be in [0, 1)")
    if not 0. < coverage <= 1.:
        raise ValueError("tip_occlusion_fixture.target_coverage must be in (0, 1]")
    if not 0. <= release < coverage:
        raise ValueError("tip_occlusion_fixture.release_coverage must be below target coverage")
    if front < 0.:
        raise ValueError("tip_occlusion_fixture.front_offset_mm must be nonnegative")
    if max_active is not None and max_active <= 0.:
        raise ValueError("tip_occlusion_fixture.max_active_s must be positive")
    return {
        "enabled": enabled,
        "radius_mm": radius,
        "trigger_progress": trigger,
        "target_coverage": coverage,
        "release_coverage": release,
        "front_offset_mm": front,
        "screen_direction_deg": angle,
        "max_active_s": max_active,
        "camera_name": str(raw.get("camera_name", "eye_internal")),
    }


def add_tip_occluder_to_xml(root, config):
    """Add the hidden renderable sphere before MJCF compilation."""
    settings = fixture_settings(config)
    if not settings["enabled"]:
        return
    world = root.find("worldbody")
    if world is None:
        raise ValueError("v9 model XML is missing worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find(f"material[@name='{MATERIAL_NAME}']") is None:
        ET.SubElement(asset, "material", name=MATERIAL_NAME,
                      rgba=".015 .015 .015 1", specular="0", shininess="0")
    if world.find(f"geom[@name='{GEOM_NAME}']") is None:
        ET.SubElement(world, "geom", name=GEOM_NAME, type="sphere",
                      pos=" ".join(str(value) for value in HIDDEN_POSITION_M),
                      size=f"{settings['radius_mm'] * .001:.15g}",
                      material=MATERIAL_NAME, group="3",
                      contype="0", conaffinity="0")


def circle_intersection_area(radius_a, radius_b, distance):
    """Area shared by two circles."""
    a, b, d = float(radius_a), float(radius_b), float(distance)
    if a <= 0. or b <= 0. or d < 0.:
        return 0.
    if d >= a + b:
        return 0.
    if d <= abs(a - b):
        return math.pi * min(a, b) ** 2
    first = a * a * math.acos((d * d + a * a - b * b) / (2. * d * a))
    second = b * b * math.acos((d * d + b * b - a * a) / (2. * d * b))
    triangle = .5 * math.sqrt(
        max(0., (-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)))
    return first + second - triangle


def center_distance_for_coverage(target_radius_px, occluder_radius_px,
                                 minimum_occluded_fraction):
    """Find image-plane center distance for a requested target occlusion."""
    fraction = float(minimum_occluded_fraction)
    if not 0. < fraction <= 1.:
        raise ValueError("minimum_occluded_fraction must be in (0, 1]")
    target_area = math.pi * target_radius_px ** 2
    maximum = math.pi * min(target_radius_px, occluder_radius_px) ** 2
    if maximum + 1e-12 < fraction * target_area:
        raise RuntimeError("OCCLUDER_TOO_SMALL_FOR_REQUESTED_COVERAGE")
    if fraction >= 1. - 1e-12:
        return max(0., occluder_radius_px - target_radius_px)
    low, high = 0., target_radius_px + occluder_radius_px
    for _ in range(60):
        middle = .5 * (low + high)
        value = circle_intersection_area(
            target_radius_px, occluder_radius_px, middle) / target_area
        if value > fraction:
            low = middle
        else:
            high = middle
    return .5 * (low + high)


def _unit(vector, code):
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-12:
        raise RuntimeError(code)
    return vector / length


class TipCoupledOcclusionFixture:
    """Place one opaque sphere in a tip-local occlusion configuration."""

    def __init__(self, reference):
        self.reference = reference
        settings_config = {
            "reference_scene": {
                "tip_occlusion_fixture": reference.config.get(
                    "reference_scene", {}).get("tip_occlusion_fixture", {})
            }
        }
        self.settings = fixture_settings(settings_config)
        self.enabled = bool(self.settings["enabled"])
        self.geom_id = mujoco.mj_name2id(
            reference.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME)
        self.target_id = mujoco.mj_name2id(
            reference.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_marker")
        if self.enabled and (self.geom_id < 0 or self.target_id < 0):
            raise RuntimeError("Enabled tip-coupled occluder geometry is missing")
        self.camera_id = (reference.camera_id(self.settings["camera_name"])
                          if self.enabled else None)
        self.radius_mm = float(self.settings["radius_mm"])
        self._initial_distance_mm = None
        self._trigger_time_s = None
        self._active_start_s = None
        self._active_frames = 0
        self._trigger_fired = False
        self._active = False
        self._local_offset_mm = None
        self._last_coverage = 0.
        self._max_coverage = 0.
        self._transitions = []
        self._last = {"active": False, "state": "waiting"}
        if self.enabled:
            self._move(HIDDEN_POSITION_M)

    @property
    def audit(self):
        return copy.deepcopy({
            "kind": "needle_tip_coupled_opaque_sphere",
            "enabled": self.enabled,
            "radius_mm": self.radius_mm,
            "trigger_progress": self.settings["trigger_progress"],
            "target_coverage": self.settings["target_coverage"],
            "release_coverage": self.settings["release_coverage"],
            "max_active_s": self.settings["max_active_s"],
            "triggered": bool(self._trigger_fired),
            "trigger_time_s": self._trigger_time_s,
            "active": bool(self._active),
            "active_frames": int(self._active_frames),
            "last_coverage_fraction": float(self._last_coverage),
            "max_coverage_fraction": float(self._max_coverage),
            "transitions": copy.deepcopy(self._transitions),
            "current": copy.deepcopy(self._last),
            "scope": ("Privileged rendered scene intervention; controller input "
                      "contains only normal RGB-D observations."),
        })

    def reset(self):
        if self.enabled:
            self._move(HIDDEN_POSITION_M)
        self._initial_distance_mm = None
        self._trigger_time_s = None
        self._active_start_s = None
        self._active_frames = 0
        self._trigger_fired = False
        self._active = False
        self._local_offset_mm = None
        self._last_coverage = 0.
        self._max_coverage = 0.
        self._transitions = []
        self._last = {"active": False, "state": "waiting"}

    def _move(self, position_m):
        if not self.enabled:
            return
        self.reference.model.geom_pos[self.geom_id] = np.asarray(position_m)
        mujoco.mj_forward(self.reference.model, self.reference.data)

    def _camera_geometry(self):
        camera_id = self.camera_id
        position = self.reference.data.cam_xpos[camera_id].copy() * 1000.
        rotation = self.reference.data.cam_xmat[camera_id].reshape(3, 3).copy()
        width, height = self.reference.eye_camera["resolution_px"]
        focal = .5 * height / math.tan(
            math.radians(float(self.reference.model.cam_fovy[camera_id])) / 2.)
        return position, rotation, float(focal), int(width), int(height)

    @staticmethod
    def _project(world_mm, camera_mm, rotation, focal, width, height):
        point = rotation.T @ (np.asarray(world_mm) - camera_mm)
        depth_mm = float(-point[2])
        if depth_mm <= 1e-9:
            raise RuntimeError("TIP_OCCLUDER_BEHIND_CAMERA")
        pixel = np.array([
            width / 2. - .5 + focal * point[0] / depth_mm,
            height / 2. - .5 - focal * point[1] / depth_mm,
        ])
        return pixel, depth_mm

    @staticmethod
    def _tool_basis(tip_mm, back_mm, camera_rotation):
        axis = _unit(np.asarray(back_mm) - np.asarray(tip_mm),
                     "TIP_OCCLUDER_NEEDLE_AXIS_UNAVAILABLE")
        side = camera_rotation[:, 0] - axis * float(axis @ camera_rotation[:, 0])
        if np.linalg.norm(side) < 1e-10:
            side = camera_rotation[:, 1] - axis * float(axis @ camera_rotation[:, 1])
        side = _unit(side, "TIP_OCCLUDER_SIDE_AXIS_UNAVAILABLE")
        up = _unit(np.cross(axis, side), "TIP_OCCLUDER_UP_AXIS_UNAVAILABLE")
        return np.column_stack((axis, side, up)), axis

    def _coverage(self, center_mm, target_mm, camera_mm, rotation,
                  focal, width, height):
        target_pixel, target_depth = self._project(
            target_mm, camera_mm, rotation, focal, width, height)
        occluder_pixel, occluder_depth = self._project(
            center_mm, camera_mm, rotation, focal, width, height)
        if occluder_depth >= target_depth:
            return 0.
        target_radius_px = focal * float(self.reference.model.geom_size[
            self.target_id, 0]) * 1000. / target_depth
        occluder_radius_px = focal * self.radius_mm / occluder_depth
        distance = float(np.linalg.norm(occluder_pixel - target_pixel))
        overlap = circle_intersection_area(
            target_radius_px, occluder_radius_px, distance)
        return min(1., overlap / (math.pi * target_radius_px ** 2))

    def _configure_trigger(self, tip_mm, back_mm, target_mm, camera_mm,
                           rotation, focal, width, height):
        target_pixel, target_depth = self._project(
            target_mm, camera_mm, rotation, focal, width, height)
        target_radius_px = focal * float(self.reference.model.geom_size[
            self.target_id, 0]) * 1000. / target_depth
        occluder_depth = target_depth - self.settings["front_offset_mm"]
        occluder_radius_px = focal * self.radius_mm / occluder_depth
        center_distance = center_distance_for_coverage(
            target_radius_px, occluder_radius_px,
            self.settings["target_coverage"])
        angle = math.radians(self.settings["screen_direction_deg"])
        pixel_offset = np.array([
            center_distance * math.cos(angle),
            center_distance * math.sin(angle),
        ])
        world_per_pixel = target_depth / focal
        lateral = (rotation[:, 0] * pixel_offset[0] * world_per_pixel
                   + rotation[:, 1] * -pixel_offset[1] * world_per_pixel)
        center = (np.asarray(target_mm) + rotation[:, 2] * self.settings["front_offset_mm"]
                  + lateral)
        basis, _ = self._tool_basis(tip_mm, back_mm, rotation)
        self._local_offset_mm = basis.T @ (center - np.asarray(tip_mm))
        return center, target_pixel, target_radius_px, occluder_radius_px

    def update(self, frame_index, timestamp_s):
        """Refresh the render geometry and return privileged audit metadata."""
        if not self.enabled:
            return copy.deepcopy(self._last)
        frame_index = int(frame_index)
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s) or timestamp_s < 0.:
            raise ValueError("tip occluder timestamp must be finite and nonnegative")
        mujoco.mj_forward(self.reference.model, self.reference.data)
        tip = np.asarray(self.reference.site_mm("needle_tip"), dtype=float)
        back = np.asarray(self.reference.site_mm("needle_base"), dtype=float)
        target = np.asarray(self.reference.site_mm("lesion"), dtype=float)
        camera, rotation, focal, width, height = self._camera_geometry()
        distance = float(np.linalg.norm(tip - target))
        if self._initial_distance_mm is None:
            self._initial_distance_mm = distance
        initial = max(self._initial_distance_mm, 1e-9)
        progress = float(np.clip(1. - distance / initial, 0., 1.))
        if not self._trigger_fired and progress >= self.settings["trigger_progress"]:
            self._trigger_fired = True
            self._trigger_time_s = timestamp_s
            try:
                center, _, target_radius_px, occluder_radius_px = (
                    self._configure_trigger(
                        tip, back, target, camera, rotation, focal, width, height))
                self._move(center * .001)
                self._last_coverage = self._coverage(
                    center, target, camera, rotation, focal, width, height)
                requested = self.settings["target_coverage"]
                if self._last_coverage + .02 < requested:
                    raise RuntimeError("TIP_OCCLUSION_TRIGGER_COVERAGE_NOT_REACHED")
                self._active = True
                self._active_start_s = timestamp_s
                self._active_frames = 0
                self._max_coverage = self._last_coverage
                transition = {
                    "transition": "show", "time_s": timestamp_s,
                    "frame_index": frame_index, "progress": progress,
                    "coverage_fraction": self._last_coverage,
                    "target_radius_px": target_radius_px,
                    "occluder_radius_px": occluder_radius_px,
                    "tip_target_distance_mm": distance,
                    "local_offset_mm": self._local_offset_mm.tolist(),
                }
                self._transitions.append(transition)
                self._last = {**transition, "active": True, "state": "active"}
            except RuntimeError as error:
                self._active = False
                self._last = {
                    "active": False, "state": "trigger_failed",
                    "time_s": timestamp_s, "frame_index": frame_index,
                    "progress": progress, "error": str(error),
                }
                self._transitions.append(copy.deepcopy(self._last))

        if self._active:
            basis, _ = self._tool_basis(tip, back, rotation)
            center = tip + basis @ self._local_offset_mm
            self._move(center * .001)
            coverage = self._coverage(
                center, target, camera, rotation, focal, width, height)
            self._last_coverage = coverage
            self._max_coverage = max(self._max_coverage, coverage)
            self._active_frames += 1
            elapsed = timestamp_s - self._active_start_s
            max_active = self.settings["max_active_s"]
            release_reason = None
            if coverage <= self.settings["release_coverage"] and self._active_frames > 1:
                release_reason = "LEFT_OCCLUSION_CONFIGURATION"
            elif max_active is not None and elapsed >= max_active:
                release_reason = "MAX_ACTIVE_DURATION_REACHED"
            self._last = {
                "active": release_reason is None, "state": (
                    "active" if release_reason is None else "released"),
                "time_s": timestamp_s, "frame_index": frame_index,
                "progress": progress, "coverage_fraction": coverage,
                "active_elapsed_s": elapsed,
                "release_reason": release_reason,
            }
            if release_reason is not None:
                self._active = False
                self._move(HIDDEN_POSITION_M)
                transition = copy.deepcopy(self._last)
                transition.update(transition="hide",
                                  duration_s=elapsed,
                                  active_frames=self._active_frames)
                self._transitions.append(transition)
        else:
            self._last = {
                "active": False,
                "state": ("released" if self._trigger_fired else "waiting"),
                "time_s": timestamp_s,
                "frame_index": frame_index,
                "progress": progress,
                "coverage_fraction": self._last_coverage,
            }
        return copy.deepcopy(self._last)
