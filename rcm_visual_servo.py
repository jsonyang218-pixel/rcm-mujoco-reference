"""RGB-D visual servoing for the fixed intraocular MuJoCo camera.

Current v9 defaults to RGB-derived feature masks and one normally visible
depth map. Historical simulator-ID perception is an explicit separate backend.
Calibrated geometry and an actuation Jacobian drive the A/B/C task interfaces;
MotorBackend execution is selected independently from perception. Historical
image/depth control remains available through position_feedback='image_depth'.
"""
from __future__ import annotations

import csv
import copy
import json
import math
import time
from itertools import count
from pathlib import Path

import numpy as np

from rcm_reference import Reference, _vec, save_png
from rcm_visual_geometry import NeedleGeometry
from rcm_oct import SyntheticOCTProvider, relative_depth
from rcm_occlusion_policy import (
    LEGACY_STATIC_MEMORY_REOBSERVE, S0A_STOP_INSTANT, S0B_STOP_BRAKE,
    S1_OPEN_LOOP, S2_TARGET_MEMORY, S3_RETRACT_REOBSERVE, STRATEGIES,
    normalize_occlusion_mode, strategy_settings,
)


DEFAULT_SERVO = {
    "camera_name": "eye_internal", "target_rgb_min": [180, 35, 0],
    "target_rgb_max": [255, 180, 100], "target_depth_weight": 0.35,
    "pixel_tolerance": 3.0, "depth_tolerance_mm": 0.05,
    "endpoint_tolerance_um": 20.0, "max_iterations": None,
    "step_gain": 0.65, "finite_difference_A_deg": 0.05,
    "finite_difference_B_deg": 0.05, "finite_difference_C_mm": 0.05,
    "damping": 0.02, "sample_period_s": 0.05, "save_diagnostics": True,
    "require_settled": True, "settle_frames": 3,
    "position_feedback": "geometry", "position_tolerance_um": 8.,
    "max_geometry_fit_quality_um": 10., "geometry_gain_per_s": 3.,
    "occlusion_handling": "stop",
    "perception_backend": "simulation_ids",
}


def visual_servo_settings(reference):
    settings = {**DEFAULT_SERVO,
                'perception_backend': getattr(reference, 'visual_perception_backend', 'simulation_ids'),
                **reference.config.get("navigation", {}).get("visual_servo", {})}
    settings['occlusion_handling'] = normalize_occlusion_mode(
        settings.get('occlusion_handling', 'stop'))
    if (settings.get('occlusion_handling') in STRATEGIES
            or settings.get('occlusion_handling') == LEGACY_STATIC_MEMORY_REOBSERVE):
        if settings['position_feedback'] != 'geometry':
            raise ValueError('Occlusion strategies require geometry feedback')
    if settings.get('occlusion_handling') in STRATEGIES:
        strategy_settings(settings)
    if settings['position_feedback'] not in ('geometry', 'image_depth'):
        raise ValueError('visual_servo.position_feedback must be geometry or image_depth')
    if settings["camera_name"] != "eye_internal":
        reference.camera_id(settings["camera_name"])
    if settings["max_iterations"] is not None and int(settings["max_iterations"]) <= 0:
        raise ValueError("visual_servo.max_iterations must be positive or null (unlimited)")
    if int(settings["settle_frames"]) < 1:
        raise ValueError("visual_servo.settle_frames must be positive")
    for key in ("pixel_tolerance", "depth_tolerance_mm", "endpoint_tolerance_um", "damping",
                "position_tolerance_um", "max_geometry_fit_quality_um", "geometry_gain_per_s",
                "sample_period_s"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"visual_servo.{key} must be positive")
    return settings


def _centroid(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    return np.array([float(np.mean(cols)), float(np.mean(rows))])


def _segmentation_geom_ids(segmentation):
    """Return MuJoCo geom IDs, rejecting background and site IDs."""
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 3 or segmentation.shape[2] < 2:
        raise ValueError("MuJoCo segmentation must contain object-ID and object-type channels")
    import mujoco
    object_ids = segmentation[:, :, 0].astype(np.int32)
    object_types = segmentation[:, :, 1]
    return np.where(object_types == int(mujoco.mjtObj.mjOBJ_GEOM), object_ids, -1)


def _draw_line(image, start, end, color, thickness=2):
    start = np.rint(start).astype(int)
    end = np.rint(end).astype(int)
    count = int(np.max(np.abs(end-start)))+1
    points = np.rint(np.linspace(start, end, max(count, 1))).astype(int)
    height, width = image.shape[:2]
    radius = max(0, int(thickness)//2)
    for x, y in points:
        image[max(0, y-radius):min(height, y+radius+1),
              max(0, x-radius):min(width, x+radius+1)] = color


def _draw_cross(image, center, color, radius=10, thickness=2):
    x, y = np.rint(center).astype(int)
    _draw_line(image, (x-radius, y), (x+radius, y), color, thickness)
    _draw_line(image, (x, y-radius), (x, y+radius), color, thickness)


def _dilate_mask(mask, radius=2):
    """Small display-only binary dilation without changing sensor data."""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask
    result = mask.copy()
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx*dx + dy*dy > radius*radius:
                continue
            source_y0, source_y1 = max(0, -dy), min(height, height-dy)
            source_x0, source_x1 = max(0, -dx), min(width, width-dx)
            target_y0, target_y1 = max(0, dy), min(height, height+dy)
            target_x0, target_x1 = max(0, dx), min(width, width+dx)
            result[target_y0:target_y1, target_x0:target_x1] |= \
                mask[source_y0:source_y1, source_x0:source_x1]
    return result


def guidance_overlay(observation):
    """Return an annotated RGB frame with a display-only enlarged needle aid."""
    image = observation["rgb"].copy()
    needle_mask = observation.get("needle_mask")
    if needle_mask is not None and np.asarray(needle_mask).any():
        # This yellow dilation is for human viewing only.  Raw camera frames,
        # segmentation, depth, and all controller decisions remain unchanged.
        image[_dilate_mask(needle_mask, radius=2)] = (255, 225, 35)
    target = observation["target_pixel"]
    tip = observation["tip_pixel"]
    if target is not None:
        _draw_cross(image, target, (0, 220, 255), radius=12, thickness=2)
    if tip is not None:
        _draw_cross(image, tip, (50, 255, 80), radius=9, thickness=2)
    if target is not None and tip is not None:
        _draw_line(image, tip, target, (255, 255, 255), thickness=2)
    return image


class VisualServoSensor:
    def __init__(self, reference, settings=None, depth_provider=None):
        self.reference = reference
        self.settings = settings or visual_servo_settings(reference)
        self.camera_name = self.settings["camera_name"]
        self.eye_group_visible = bool(getattr(reference, "sensor_eye_group_visible", False))
        self.width, self.height = reference.eye_camera["resolution_px"]
        self.oct_provider = depth_provider if depth_provider is not None else SyntheticOCTProvider(
            nominal_period_s=float(self.settings["sample_period_s"]),
            min_valid_depth_m=reference.eye_camera["znear_mm"]*.001,
            # Reverse-Z float rounding can return an empty pixel just below
            # zfar. Conservatively exclude the last 0.01% of the depth range.
            max_valid_depth_m=reference.eye_camera["zfar_mm"]*.001*(1.-1e-4))
        self.observation_time_s = None
        self.observation_time_origin_s = 0.
        self._oct_capture_count = 0
        self.last_oct_frame = None
        self.target_geom_ids = np.asarray([int(reference.model.geom("lesion_marker").id)])
        self.tip_geom_id = int(reference.model.geom("needle_tip_marker").id)
        self.tip_geom_ids = np.asarray([
            int(reference.model.geom(name).id) for name in ("needle_distal_tip", "needle_tip_marker")
        ])
        self.needle_geom_ids = np.asarray([
            int(reference.model.geom(name).id) for name in (
                "needle_proximal", "needle_distal", "needle_distal_tip", "needle_tip_marker")
        ])
        self.distal_geom_ids = np.asarray([
            int(reference.model.geom(name).id) for name in ("needle_distal", "needle_distal_tip")
        ])
        self.renderers = {}
        for mode in ("rgb", "depth", "segmentation"):
            renderer = __import__("mujoco").Renderer(reference.model, height=self.height, width=self.width)
            if mode == "depth":
                renderer.enable_depth_rendering()
            elif mode == "segmentation":
                renderer.enable_segmentation_rendering()
            self.renderers[mode] = renderer
        self.geometry = (NeedleGeometry.from_reference(
            reference, self.width, self.height, self.camera_name, self.settings)
            if self.settings.get('position_feedback') == 'geometry' else None)

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()

    def observe(self):
        render_started = time.monotonic()
        rgb = self._render("rgb")
        # Two ideal registered depth passes; these are not physical OCT scans.
        depth = self._render("depth", include_fundus=True)
        oct_depth = self._render("depth", include_fundus=False)
        # Segmentation identifies physical scene objects. The retina is an
        # RGB-only backdrop and must not occlude target/tip measurements.
        segmentation = self._render("segmentation", include_fundus=False)
        timestamp = (self._oct_capture_count*float(self.settings["sample_period_s"])
                     if self.observation_time_s is None else float(self.observation_time_s))
        repeated = self.last_oct_frame is not None and timestamp == self.last_oct_frame.timestamp_s
        frame = self.oct_provider.capture(
            depth, oct_depth, timestamp_s=timestamp, camera_name=self.camera_name,
            acquisition_wall_s=time.monotonic(), allow_repeated_timestamp=repeated,
            metadata={
                "timestamp_policy": ("synthetic_sensor_sequence_clock" if self.observation_time_s is None
                                     else "controller_simulation_clock"),
                "render_wall_duration_s": time.monotonic()-render_started,
                "feature_pass_excludes_fundus_backdrop": True,
                "visibility_gated_by_normal_scene_segmentation": True,
                "renderer_znear_m": self.reference.eye_camera["znear_mm"]*.001,
                "renderer_zfar_m": self.reference.eye_camera["zfar_mm"]*.001,
            })
        self._oct_capture_count += 1
        self.last_oct_frame = frame
        depth, oct_depth = frame.scene_surface_depth_m, frame.registered_feature_depth_m
        segmentation_ids = _segmentation_geom_ids(segmentation)
        target_mask = np.isin(segmentation_ids, self.target_geom_ids)
        target_pixel = _centroid(target_mask)
        target_depth_mask = target_mask & frame.feature_valid_mask
        tip_mask = np.isin(segmentation_ids, self.tip_geom_ids)
        needle_mask = np.isin(segmentation_ids, self.needle_geom_ids)
        marker_mask = segmentation_ids == self.tip_geom_id
        if marker_mask.any():
            tip_pixel = _centroid(marker_mask)
            tip_depth_mask = marker_mask
        else:
            # If the calibrated apex sphere is sub-pixel or occluded, use the
            # exposed endpoint of the real taper, away from its shaft.
            cone_mask = segmentation_ids == int(self.distal_geom_ids[1])
            shaft_center = _centroid(np.isin(segmentation_ids,
                                             self.needle_geom_ids))
            if cone_mask.any() and shaft_center is not None:
                rows, cols = np.where(cone_mask)
                endpoint = int(np.argmax((cols-shaft_center[0])**2+(rows-shaft_center[1])**2))
                tip_pixel = np.array([float(cols[endpoint]), float(rows[endpoint])])
                tip_depth_mask = np.zeros_like(cone_mask)
                tip_depth_mask[rows[endpoint], cols[endpoint]] = True
            else:
                tip_pixel = _centroid(tip_mask)
                tip_depth_mask = tip_mask
        tip_depth_mask = tip_depth_mask & frame.feature_valid_mask
        target_depth = None if not target_depth_mask.any() else float(np.median(oct_depth[target_depth_mask]))
        tip_depth = None if not tip_depth_mask.any() else float(np.median(oct_depth[tip_depth_mask]))
        if target_depth is not None and not np.isfinite(target_depth):
            target_depth = None
        if tip_depth is not None and not np.isfinite(tip_depth):
            tip_depth = None
        observation = {
            "target_pixel": None if target_pixel is None else target_pixel.tolist(),
            "tip_pixel": None if tip_pixel is None else tip_pixel.tolist(),
            "target_depth_m": target_depth, "tip_depth_m": tip_depth,
            "target_pixels": int(target_mask.sum()), "target_depth_pixels": int(target_depth_mask.sum()),
            "tip_pixels": int(tip_mask.sum()),
            "needle_mask": needle_mask,
            "target_occluded": target_pixel is None,
            "tip_occluded": tip_pixel is None,
            "rgb": rgb, "depth": depth, "oct_depth": oct_depth,
            "segmentation": segmentation, "depth_source": "scene_depth_with_fundus",
            "oct_depth_source": frame.source, "oct_depth_unit": frame.depth_unit,
            "oct_frame": frame, "oct_metadata": frame.metadata(),
            "oct_depth_valid_mask": frame.feature_valid_mask,
            "oct_navigation_depth": None,
        }
        if self.geometry is not None:
            # Raw image features remain available for diagnostics. Precision
            # feedback uses one calibrated physical endpoint, not a taper pixel.
            observation.update(raw_tip_pixel=observation['tip_pixel'],
                               raw_tip_depth_m=observation['tip_depth_m'])
            measured = {}
            for name in ('target', 'tip'):
                try:
                    point, quality = getattr(self.geometry, name)(observation)
                    measured[name+'_world_mm'] = point.tolist()
                    measured[name+'_quality'] = quality
                except RuntimeError as exc:
                    measured[name+'_world_mm'] = None
                    measured[name+'_error'] = str(exc)
            observation['geometry'] = measured
            if measured['tip_world_mm'] is not None:
                observation['tip_pixel'], observation['tip_depth_m'] = self.geometry.project(measured['tip_world_mm'])
            else:
                observation['tip_pixel'] = observation['tip_depth_m'] = None
            observation['tip_observable'] = measured['tip_world_mm'] is not None
            # Relative depth uses measured points only. No fresh hidden target
            # depth is invented when target visibility/geometry is unavailable.
            observation['oct_navigation_depth'] = None
            if measured['target_world_mm'] is not None and measured['tip_world_mm'] is not None:
                def optical(point):
                    local = self.geometry.camera_rotation.T@(
                        np.asarray(point)-self.geometry.camera_position_mm)*.001
                    return local*np.array([1., -1., -1.])
                axis = measured['tip_quality'].get('axis_world')
                optical_axis = (None if axis is None else
                                (self.geometry.camera_rotation.T@np.asarray(axis))*[1., -1., -1.])
                observation['oct_navigation_depth'] = relative_depth(
                    optical(measured['tip_world_mm']), optical(measured['target_world_mm']),
                    needle_axis_camera=optical_axis,
                    source_flags={"source": frame.source, "frame_id": frame.frame_id,
                                  "target_source": "current_visible_measurement",
                                  "is_simulation": frame.is_simulation})
        return observation

    def visibility_metrics(self, segmentation=None):
        """Measure target occlusion by the rendered physical needle geometry.

        Render a frame-specific counterfactual segmentation with only the
        physical needle geoms hidden.  The difference is therefore caused by geometric
        line-of-sight occlusion in MuJoCo, not a synthetic mask or noise label.
        This measurement is diagnostic only and never changes controller input.
        """
        if segmentation is None:
            segmentation = self._render("segmentation", include_fundus=False)
        ids = _segmentation_geom_ids(segmentation)
        target_pixels = int(np.isin(ids, self.target_geom_ids).sum())
        tip_pixels = int(np.isin(ids, self.tip_geom_ids).sum())
        needle_pixels = int(np.isin(ids, self.needle_geom_ids).sum())

        # Render the frame-specific counterfactual every time: apparent marker
        # size changes slightly with depth, so a fixed baseline can mislabel
        # ordinary perspective changes as occlusion (or miss real occlusion).
        groups = self.reference.model.geom_group[self.needle_geom_ids].copy()
        try:
            # Hide the needle in the group disabled by this backend's mask.
            hidden_group = 1 if self.eye_group_visible else 2
            self.reference.model.geom_group[self.needle_geom_ids] = hidden_group
            clear_segmentation = self._render("segmentation", include_fundus=False)
            clear_ids = _segmentation_geom_ids(clear_segmentation)
            unoccluded_pixels = int(np.isin(clear_ids,
                                             self.target_geom_ids).sum())
        finally:
            self.reference.model.geom_group[self.needle_geom_ids] = groups

        # The counterfactual should contain every target pixel visible in the
        # original render; guard against a stale reference or raster edge case.
        unoccluded_pixels = max(unoccluded_pixels, target_pixels)
        occluded_pixels = max(0, unoccluded_pixels-target_pixels)
        fraction = (float(occluded_pixels/unoccluded_pixels)
                    if unoccluded_pixels else None)
        if occluded_pixels:
            state = "full_needle" if target_pixels == 0 else "partial_needle"
        else:
            state = "clear" if target_pixels else "not_visible"
        return {
            # Compatibility aliases for feedback logs; canonical dataset names
            # below make the visible/unoccluded distinction explicit.
            "target_pixels": target_pixels,
            "tip_pixels": tip_pixels,
            "target_pixels_visible": target_pixels,
            "target_pixels_unoccluded": unoccluded_pixels,
            "target_pixels_occluded_by_needle": occluded_pixels,
            "target_occlusion_fraction_by_needle": fraction,
            "target_occlusion_state": state,
            "tip_pixels_visible": tip_pixels,
            "needle_pixels_visible": needle_pixels,
            "tip_fully_occluded": tip_pixels == 0,
            "counterfactual_no_needle_rendered": True,
        }

    def _render(self, mode, include_fundus=None):
        renderer = self.renderers[mode]
        if include_fundus is None:
            include_fundus = True
        old_znear, old_zfar = self.reference.model.vis.map.znear, self.reference.model.vis.map.zfar
        self.reference.model.vis.map.znear = self.reference.eye_camera["znear_mm"]*.001/self.reference.model.stat.extent
        self.reference.model.vis.map.zfar = self.reference.eye_camera["zfar_mm"]*.001/self.reference.model.stat.extent
        try:
            option = __import__("mujoco").MjvOption()
            option.geomgroup[0] = 0  # mechanical housing is outside the microscope FOV
            option.geomgroup[1] = 0  # external robot housing is outside the microscope FOV
            # The eye shell is a display-only element. It must be excluded from
            # every sensor pass because it is a depth/segmentation occluder.
            option.geomgroup[2] = 0
            option.geomgroup[3] = 1  # optical contents: needle and target markers
            option.geomgroup[4] = 1  # calibrated physical tip marker
            # The fundus backdrop is shown in RGB and in the complete scene
            # depth image, but excluded from the feature/OCT pass.
            option.geomgroup[5] = int(bool(include_fundus))
            option.sitegroup[:] = 0
            renderer.update_scene(self.reference.data, camera=self.camera_name, scene_option=option)
            return np.asarray(renderer.render()).copy()
        finally:
            self.reference.model.vis.map.znear, self.reference.model.vis.map.zfar = old_znear, old_zfar

    @staticmethod
    def error(observation, target_reference=None, allow_cached_target=False):
        # A cached target measurement may be used only for privileged feature
        # calibration at the known simulated goal pose. During feedback, the
        # current frame must contain a measurable target; otherwise navigation
        # fails closed instead of continuing through complete occlusion.
        target_pixel = observation["target_pixel"]
        target_depth = observation["target_depth_m"]
        if allow_cached_target and target_reference is not None:
            target_pixel = target_reference["target_pixel"]
            target_depth = target_reference["target_depth_m"]
        if target_pixel is None:
            raise RuntimeError("TARGET_NOT_VISIBLE")
        if observation["tip_pixel"] is None:
            raise RuntimeError("TIP_NOT_VISIBLE")
        if target_depth is None or observation["tip_depth_m"] is None:
            raise RuntimeError("DEPTH_NOT_VISIBLE")
        pixel = np.asarray(observation["tip_pixel"]) - np.asarray(target_pixel)
        depth_mm = (float(observation["tip_depth_m"])-float(target_depth))*1000.
        return np.r_[pixel, depth_mm]


class VisualServoController:
    def __init__(self, reference, settings=None, output_dir=None, initial_velocity=None,
                 command_validator=None, control_axes=None, completion_axes=None,
                 measurement_goal=None, target_transform=None,
                 independent_endpoint_acceptance=True):
        self.reference = reference
        self.settings = settings or visual_servo_settings(reference)
        # Strict visual collection disables the privileged target-pose feature
        # calibration.  In that mode every feedback error is computed from the
        # current RGB/segmentation/depth observation.
        self.use_target_pose_calibration = bool(
            self.settings.get('position_feedback') != 'geometry' and
            self.settings.get("use_target_pose_calibration", True))
        occlusion_mode = normalize_occlusion_mode(
            self.settings.get('occlusion_handling', 'stop'))
        self.settings['occlusion_handling'] = occlusion_mode
        if self.settings.get('perception_backend') == 'strict_rgbd':
            if self.settings.get('position_feedback') != 'geometry':
                raise ValueError('Strict RGB-D navigation requires current geometry feedback')
            if occlusion_mode == LEGACY_STATIC_MEMORY_REOBSERVE:
                raise ValueError(
                    'Strict RGB-D navigation requires a current target observation; '
                    'use S2_target_memory or S3_retract_reobserve for explicit strategies')
            if measurement_goal is not None or target_transform is not None:
                raise ValueError('Strict RGB-D navigation requires the currently observed target; external measurement_goal/target_transform injection is unsupported')
        if getattr(reference, 'motor_execution', None) is not None:
            if self.settings.get('position_feedback') != 'geometry':
                raise ValueError('Motor execution requires geometry feedback')
            if occlusion_mode == LEGACY_STATIC_MEMORY_REOBSERVE:
                raise ValueError(
                    'Motor execution does not support legacy static_memory_reobserve; '
                    'use S2_target_memory or S3_retract_reobserve')
        self.command_validator = command_validator
        # Intermediate strict-visual goals are constructed from measured needle
        # geometry, never from an inverse solution of the scene target.
        self.measurement_goal = (None if measurement_goal is None else np.asarray(measurement_goal, dtype=float).copy())
        if (self.measurement_goal is not None and
                (self.measurement_goal.shape != (3,) or not np.isfinite(self.measurement_goal).all())):
            raise ValueError('measurement_goal must be a finite measured 3D point in millimeters')
        if target_transform is not None and not callable(target_transform):
            raise ValueError('target_transform must be callable')
        self.target_transform = target_transform
        self.independent_endpoint_acceptance = bool(independent_endpoint_acceptance)
        self.final_observation = None
        self.control_axes = np.ones(3, dtype=bool) if control_axes is None else np.asarray(control_axes, dtype=bool)
        self.completion_axes = (self.control_axes.copy() if completion_axes is None
                                else np.asarray(completion_axes, dtype=bool))
        if (self.control_axes.shape != (3,) or self.completion_axes.shape != (3,)
                or not np.any(self.control_axes) or not np.any(self.completion_axes)):
            raise ValueError("control_axes and completion_axes must be non-empty length-3 masks")
        # Carry the previous interval's command across consecutive visual goals.
        # Units are rad/s, rad/s, m/s, matching the internal task coordinates.
        self.initial_velocity = np.zeros(3) if initial_velocity is None else _vec(initial_velocity).copy()
        if (self.settings.get('position_feedback') == 'geometry'
                and np.any(np.abs(self.initial_velocity[~self.control_axes]) > 1e-12)):
            raise ValueError('initial_velocity on disabled control axes must be zero')
        self.final_velocity = self.initial_velocity.copy()
        if self.settings.get('perception_backend') == 'strict_rgbd':
            from rcm_strict_rgbd import StrictRGBDSensor
            self.sensor = StrictRGBDSensor(reference, self.settings)
        elif self.settings.get('perception_backend', 'simulation_ids') == 'simulation_ids':
            self.sensor = VisualServoSensor(reference, self.settings)
        else:
            raise ValueError('Unknown visual perception backend')
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None and self.settings["save_diagnostics"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.oct_records = []
        self.max_velocity = np.zeros(3)
        self.max_acceleration = np.zeros(3)
        self.iteration_callback = None
        self.observation_hook = None

    def _record_oct(self, iteration, observation):
        if observation.get('oct_metadata') is not None:
            self.oct_records.append({
                'iteration': int(iteration),
                'controller_time_s': iteration*float(self.settings['sample_period_s']),
                'oct': observation['oct_metadata'],
                'navigation_depth': observation.get('oct_navigation_depth'),
            })

    def _prepare_observation(self, iteration, timestamp_s):
        """Refresh privileged scene interventions before rendering.

        The fixture returns audit data to the caller in tests/runners, but its
        coordinates and coverage are never inserted into the observation used
        by perception or control.
        """
        update = getattr(self.reference, 'update_tip_occlusion', None)
        return None if update is None else update(iteration, timestamp_s)

    def _visual_goal(self, target, tip, observation):
        goal = self.measurement_goal if self.measurement_goal is not None else target
        if self.target_transform is not None:
            goal = self.target_transform(np.asarray(goal).copy(), np.asarray(tip).copy(), observation)
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise RuntimeError('INVALID_VISUAL_PHASE_GOAL')
        return goal.copy()

    def _jacobian(self, q, base_error, feature_bias, target_reference):
        steps = np.asarray([
            math.radians(self.settings["finite_difference_A_deg"]),
            math.radians(self.settings["finite_difference_B_deg"]),
            self.settings["finite_difference_C_mm"]*.001,
        ])
        jacobian = np.zeros((3, 3))
        for axis, step in enumerate(steps):
            measured = {}
            for direction in (1., -1.):
                trial = q.copy()
                trial[axis] += direction*step
                try:
                    self.reference.set_task(trial)
                    error = VisualServoSensor.error(self.sensor.observe(), target_reference)
                    error -= feature_bias
                    measured[direction] = error
                except (RuntimeError, ValueError):
                    continue
                finally:
                    self.reference.set_task(q)
            if 1. in measured and -1. in measured:
                # Use both sides of the operating point.  A one-sided probe
                # can land on the same rendered pixel even when the opposite
                # direction is observable, producing an artificial zero
                # column in the image Jacobian.
                jacobian[:, axis] = (measured[1.] - measured[-1.])/(2.*step)
            elif measured:
                # Keep a usable one-sided estimate when one side is outside
                # the task domain or temporarily not visible.  The controller
                # still rejects a genuinely unobservable axis below.
                direction = next(iter(measured))
                jacobian[:, axis] = (measured[direction]-base_error)/(direction*step)
            else:
                jacobian[:, axis] = np.nan
        return jacobian

    def _model_jacobian(self, q):
        """Actuation Jacobian only; absolute model positions are not feedback."""
        steps = [1e-5, 1e-5, 1e-6]
        jacobian = np.zeros((3, 3))
        for i, step in enumerate(steps):
            points = []
            for sign in (1., -1.):
                trial = q.copy()
                trial[i] += sign*step
                state = self.reference.state_from_task(trial)
                points.append(self.reference.robot_to_world(
                    self.reference.nominal.forward(state, enforce_limits=False)))
            jacobian[:, i] = (points[0]-points[1])/(2*step)
        return jacobian

    def _run_geometry(self):
        """Measured 3D feedback with coupled actuation and a settled endpoint.

        No target-pose feature calibration is performed. Simulation truth is
        used in log/evaluation fields only, including the independent final
        accuracy gate after the measurement-driven controller has stopped.
        """
        q = self.reference.current_task().copy()
        initial = self.reference.metrics()
        dt = float(self.settings['sample_period_s'])
        navigation = self.reference.config.get('navigation', {})
        unit = np.array([math.pi/180, math.pi/180, .001])
        vmax = np.asarray(navigation.get('max_velocity_A_deg_B_deg_C_mm_per_s', [4., 2., .6]))*unit
        amax = np.asarray(navigation.get('max_acceleration_A_deg_B_deg_C_mm_per_s2', [10., 5., 1.5]))*unit
        previous_velocity = self.initial_velocity.copy()
        if np.any(np.abs(previous_velocity) > vmax+1e-12):
            self.sensor.close()
            raise ValueError('initial_velocity exceeds configured velocity limits')
        self.max_velocity = np.abs(previous_velocity)
        status, reason, settled = 'FAIL', 'MAX_ITERATIONS', 0
        final_observation = None
        measured_error = None
        max_iterations = self.settings['max_iterations']
        iterations = count() if max_iterations is None else range(int(max_iterations))
        motor = getattr(self.reference, 'motor_execution', None)
        command_q = q.copy()
        try:
            for iteration in iterations:
                if motor is None:
                    self.reference.set_task(q)
                elif iteration > 0:
                    self.reference.set_task(command_q, motor_dt=dt)
                if motor is not None:
                    q = self.reference.current_task().copy()
                self.sensor.observation_time_s = (motor.time_s if motor is not None else
                    getattr(self.sensor, 'observation_time_origin_s', 0.)+iteration*dt)
                if self.observation_hook is not None:
                    self.observation_hook(iteration, {
                        'controller_state': 'tracking', 'time_s': iteration*dt,
                        'task_A_rad_B_rad_C_m': q.tolist(), 'recovery_attempt': 0})
                self._prepare_observation(
                    iteration, motor.time_s if motor is not None else
                    getattr(self.sensor, 'observation_time_origin_s', 0.)+iteration*dt)
                observation = self.sensor.observe()
                self._record_oct(iteration, observation)
                final_observation = observation
                self.final_observation = observation
                geometry = observation.get('geometry', {})
                observation_error = None
                if (self.measurement_goal is None and
                        (observation['target_pixel'] is None or geometry.get('target_world_mm') is None)):
                    observation_error = geometry.get('target_error', 'TARGET_NOT_VISIBLE')
                elif geometry.get('tip_world_mm') is None:
                    observation_error = geometry.get('tip_error', 'TIP_NOT_OBSERVABLE')
                if observation_error is not None:
                    reason = observation_error
                    measured_error = None
                    observation['tracking'] = {
                        'target_source': 'unavailable', 'target_observed': False,
                        'cached_target_age_s': None, 'used_target_world_mm': None,
                        'estimated_position_error_um': None, 'controller_state': 'failed_observation',
                    }
                    self.rows.append({
                        'iteration': iteration, 'time_s': iteration*dt,
                        'u_error_px': None, 'v_error_px': None, 'depth_error_mm': None,
                        'pixel_error_px': None,
                        'tip_error_um': float(self.reference.metrics()['tip_target_error_um']),
                        'estimated_position_error_um': None, 'completion_error_um': None,
                        'geometry_fit_quality_um': None,
                        'tip_feature_source': geometry.get('tip_quality', {}).get('source'),
                        'tip_support_points': geometry.get('tip_quality', {}).get('support'),
                        'tip_fit_residual_mm': geometry.get('tip_quality', {}).get('residual_mm'),
                        'target_fit_residual_mm': geometry.get('target_quality', {}).get('residual_mm'),
                        'target_pixels': observation['target_pixels'], 'tip_pixels': observation['tip_pixels'],
                        'A_deg': math.degrees(q[0]), 'B_deg': math.degrees(q[1]), 'C_mm': q[2]*1000.,
                        'command_velocity_A_deg_B_deg_C_mm_per_s': None,
                        'command_acceleration_A_deg_B_deg_C_mm_per_s2': None,
                    })
                    if self.iteration_callback is not None and self.iteration_callback(iteration, observation) is False:
                        reason = 'USER_ABORTED'
                    break
                tip = np.asarray(geometry['tip_world_mm'])
                target = self._visual_goal(geometry.get('target_world_mm'), tip, observation)
                error_mm = target-tip
                measured_error = float(np.linalg.norm(error_mm)*1000)
                tip_quality = geometry['tip_quality']
                target_quality = ({'fit_quality_mm': .002, 'residual_mm': 0.}
                                  if self.measurement_goal is not None else geometry['target_quality'])
                observation['tracking'] = {
                    'target_source': 'visual_retraction_waypoint' if self.measurement_goal is not None else 'observed',
                    'target_observed': self.measurement_goal is None,
                    'cached_target_age_s': None if self.measurement_goal is not None else 0.,
                    'used_target_world_mm': target.tolist(),
                    'estimated_position_error_um': measured_error, 'controller_state': 'tracking',
                }
                quality_um = (tip_quality['fit_quality_mm']+target_quality['fit_quality_mm'])*1000
                if quality_um > self.settings['max_geometry_fit_quality_um']:
                    reason = 'LOW_QUALITY_VISUAL_GEOMETRY'
                    break
                jacobian = self._model_jacobian(q)
                active = self.control_axes
                # Normalize by per-axis velocity limits before SVD: radians and
                # meters must not receive arbitrary weights from their units.
                normalized = jacobian[:, active]*vmax[active]
                u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
                if not len(singular) or singular[-1] < max(1e-10, singular[0]*1e-6):
                    reason = 'SINGULAR_ACTUATION_JACOBIAN'
                    break
                damping = singular[0]*1e-4
                correction = np.zeros(3)
                correction[active] = vmax[active]*(vt.T@(
                    singular/(singular**2+damping**2)*(u.T@error_mm)))
                # Partial-axis phases (e.g. fixed-direction fundus insertion)
                # may complete only their controllable projection. Independent
                # final 3D acceptance still checks the complete endpoint.
                complete_j = jacobian[:, self.completion_axes]
                projected = complete_j@np.linalg.lstsq(complete_j, error_mm, rcond=None)[0]
                completion_error_um = float(np.linalg.norm(projected)*1000)
                in_tolerance = completion_error_um < float(self.settings['position_tolerance_um'])
                pixel_tip, depth_tip = self.sensor.geometry.project(tip)
                pixel_target, depth_target = self.sensor.geometry.project(target)
                feature_error = np.r_[np.asarray(pixel_tip)-pixel_target, (depth_tip-depth_target)*1000]
                # Evaluation only, not read by any control/stop expression.
                truth_error = float(self.reference.metrics()['tip_target_error_um'])
                self.rows.append({
                    'iteration': iteration, 'time_s': iteration*dt,
                    'u_error_px': float(feature_error[0]), 'v_error_px': float(feature_error[1]),
                    'depth_error_mm': float(feature_error[2]),
                    'pixel_error_px': float(np.linalg.norm(feature_error[:2])), 'tip_error_um': truth_error,
                    'estimated_position_error_um': measured_error,
                    'completion_error_um': completion_error_um, 'geometry_fit_quality_um': quality_um,
                    'tip_feature_source': tip_quality['source'], 'tip_support_points': tip_quality['support'],
                    'tip_fit_residual_mm': tip_quality['residual_mm'],
                    'target_fit_residual_mm': target_quality['residual_mm'],
                    'target_pixels': observation['target_pixels'], 'tip_pixels': observation['tip_pixels'],
                    'A_deg': math.degrees(q[0]), 'B_deg': math.degrees(q[1]), 'C_mm': q[2]*1000,
                })
                if self.iteration_callback is not None and self.iteration_callback(iteration, observation) is False:
                    reason = 'USER_ABORTED'
                    break
                motor_stopped = motor is None or motor.settled()
                settled = settled+1 if (in_tolerance and motor_stopped
                    and np.all(np.abs(previous_velocity) < 1e-12)) else 0
                if settled >= int(self.settings['settle_frames']):
                    status, reason = 'PASS', 'POSITION_ERROR_AND_SETTLED'
                    break
                if max_iterations is not None and iteration == int(max_iterations)-1:
                    break  # Never leave an applied but unobserved final command.
                wanted = np.zeros(3) if in_tolerance else float(self.settings['geometry_gain_per_s'])*correction
                braking = .6*np.sqrt(2*amax*np.abs(correction))
                wanted = np.clip(wanted, -np.minimum(vmax, braking), np.minimum(vmax, braking))
                wanted[~active] = 0.
                velocity = np.clip(wanted, previous_velocity-amax*dt, previous_velocity+amax*dt)
                acceleration = (velocity-previous_velocity)/dt
                # Retain sub-handle increments in the requested command. If
                # each increment is instead restarted at quantized feedback,
                # small valid corrections disappear in a protocol dead zone.
                candidate = (command_q if motor is not None else q)+velocity*dt
                try:
                    candidate = self.reference.validate_task(candidate).copy()
                except ValueError:
                    reason = 'TASK_CONSTRAINT'
                    break
                if self.command_validator is not None and not self.command_validator(q, candidate):
                    reason = 'FUNDUS_MOTION_CONSTRAINT'
                    break
                self.max_velocity = np.maximum(self.max_velocity, np.abs(velocity))
                self.max_acceleration = np.maximum(self.max_acceleration, np.abs(acceleration))
                self.rows[-1].update(
                    command_velocity_A_deg_B_deg_C_mm_per_s=(velocity/unit).tolist(),
                    command_acceleration_A_deg_B_deg_C_mm_per_s2=(acceleration/unit).tolist())
                q, previous_velocity = candidate, velocity
                command_q = candidate.copy()
                self.final_velocity = velocity.copy()
                if self.output_dir is not None and self.settings['save_diagnostics'] and iteration == 0:
                    save_png(self.output_dir/'camera_0000.png', observation['rgb'])
                    save_png(self.output_dir/'guidance_0000.png', guidance_overlay(observation))
            if motor is None:
                self.reference.set_task(q)
            elif status != 'PASS':
                motor.hold()  # Offline simulator hold, not validated physical braking.
            if final_observation is not None and self.output_dir is not None and self.settings['save_diagnostics']:
                save_png(self.output_dir/'camera_final.png', final_observation['rgb'])
                save_png(self.output_dir/'guidance_final.png', guidance_overlay(final_observation))
        finally:
            self.sensor.close()
        return self._finish_geometry(initial, final_observation, status, reason, measured_error,
                                     settled, dt, unit, vmax, amax)

    def geometry_action_from_observation(
        self,
        q,
        observation,
        previous_velocity=None,
        command_q=None,
    ):
        """Compute one teacher command from a current strict RGB-D observation.

        This method never moves the scene. It is used to create DAgger labels
        and to expose whether depth/geometry is valid for those labels.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite A/B/C task vector")
        dt = float(self.settings["sample_period_s"])
        navigation = self.reference.config.get("navigation", {})
        unit = np.array([math.pi / 180, math.pi / 180, 0.001])
        vmax = np.asarray(
            navigation.get("max_velocity_A_deg_B_deg_C_mm_per_s", [4.0, 2.0, 0.6])
        ) * unit
        amax = np.asarray(
            navigation.get(
                "max_acceleration_A_deg_B_deg_C_mm_per_s2",
                [10.0, 5.0, 1.5],
            )
        ) * unit
        previous_velocity = (
            np.zeros(3)
            if previous_velocity is None
            else np.asarray(previous_velocity, dtype=float)
        )
        command_q = q.copy() if command_q is None else np.asarray(command_q, dtype=float)
        geometry = observation.get("geometry", {})
        target_pixel = observation.get("target_pixel")
        target_depth = observation.get("target_depth_m")
        tip_depth = observation.get("tip_depth_m")
        target_world = geometry.get("target_world_mm")
        tip_world = geometry.get("tip_world_mm")
        depth_valid = bool(
            target_pixel is not None
            and target_depth is not None
            and tip_depth is not None
            and target_world is not None
            and tip_world is not None
        )
        if not depth_valid:
            reason = geometry.get(
                "target_error",
                "TARGET_NOT_VISIBLE"
                if target_pixel is None or target_world is None
                else "DEPTH_NOT_VISIBLE",
            )
            return {
                "valid": False,
                "reason": reason,
                "depth_valid": False,
                "action_deg_deg_mm": None,
            }
        tip = np.asarray(tip_world, dtype=float)
        target = self._visual_goal(target_world, tip, observation)
        error_mm = target - tip
        measured_error_um = float(np.linalg.norm(error_mm) * 1000.0)
        tip_quality = geometry.get("tip_quality", {})
        target_quality = (
            {"fit_quality_mm": 0.002, "residual_mm": 0.0}
            if self.measurement_goal is not None
            else geometry.get("target_quality", {})
        )
        quality_um = (
            float(tip_quality.get("fit_quality_mm", math.inf))
            + float(target_quality.get("fit_quality_mm", math.inf))
        ) * 1000.0
        if not np.isfinite(quality_um) or quality_um > float(
            self.settings["max_geometry_fit_quality_um"]
        ):
            return {
                "valid": False,
                "reason": "LOW_QUALITY_VISUAL_GEOMETRY",
                "depth_valid": True,
                "action_deg_deg_mm": None,
                "measured_error_um": measured_error_um,
                "geometry_fit_quality_um": quality_um,
            }
        jacobian = self._model_jacobian(q)
        active = self.control_axes
        normalized = jacobian[:, active] * vmax[active]
        u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
        if not len(singular) or singular[-1] < max(1e-10, singular[0] * 1e-6):
            return {
                "valid": False,
                "reason": "SINGULAR_ACTUATION_JACOBIAN",
                "depth_valid": True,
                "action_deg_deg_mm": None,
                "measured_error_um": measured_error_um,
                "geometry_fit_quality_um": quality_um,
            }
        damping = singular[0] * 1e-4
        correction = np.zeros(3)
        correction[active] = vmax[active] * (
            vt.T
            @ (
                singular
                / (singular**2 + damping**2)
                * (u.T @ error_mm)
            )
        )
        complete_j = jacobian[:, self.completion_axes]
        projected = complete_j @ np.linalg.lstsq(
            complete_j,
            error_mm,
            rcond=None,
        )[0]
        completion_error_um = float(np.linalg.norm(projected) * 1000.0)
        in_tolerance = completion_error_um < float(
            self.settings["position_tolerance_um"]
        )
        wanted = (
            np.zeros(3)
            if in_tolerance
            else float(self.settings["geometry_gain_per_s"]) * correction
        )
        braking = 0.6 * np.sqrt(2 * amax * np.abs(correction))
        wanted = np.clip(wanted, -np.minimum(vmax, braking), np.minimum(vmax, braking))
        wanted[~active] = 0.0
        velocity = np.clip(
            wanted,
            previous_velocity - amax * dt,
            previous_velocity + amax * dt,
        )
        candidate = command_q + velocity * dt
        try:
            candidate = self.reference.validate_task(candidate).copy()
        except ValueError as error:
            return {
                "valid": False,
                "reason": f"TASK_CONSTRAINT: {error}",
                "depth_valid": True,
                "action_deg_deg_mm": None,
                "measured_error_um": measured_error_um,
                "geometry_fit_quality_um": quality_um,
            }
        return {
            "valid": True,
            "reason": "POSITION_ERROR_AND_SETTLED_HINT"
            if in_tolerance
            else "TRACKING",
            "depth_valid": True,
            "action_deg_deg_mm": ((candidate - q) / unit).tolist(),
            "command_velocity_task_si": velocity.tolist(),
            "candidate_task_si": candidate.tolist(),
            "measured_error_um": measured_error_um,
            "completion_error_um": completion_error_um,
            "geometry_fit_quality_um": quality_um,
            "in_tolerance": bool(in_tolerance),
        }

    def _run_geometry_with_memory(self):
        """Opt-in static-target memory and bounded encoder-based reobservation.

        Memory supplies only the static target position. The tip is never
        predicted, and recovery cannot satisfy the arrival condition. Raw
        observations remain intact; their separate tracking record explains
        which information was actually used by the controller.
        """
        from rcm_target_tracking import TargetMemory

        tracker = TargetMemory(sample_period_s=float(self.settings['sample_period_s']))
        allow_memory = self.settings.get('occlusion_handling', 'stop') == 'static_memory_reobserve'
        q = self.reference.current_task().copy()
        initial = self.reference.metrics()
        dt = float(self.settings['sample_period_s'])
        navigation = self.reference.config.get('navigation', {})
        unit = np.array([math.pi/180, math.pi/180, .001])
        vmax = np.asarray(navigation.get('max_velocity_A_deg_B_deg_C_mm_per_s', [4., 2., .6]))*unit
        amax = np.asarray(navigation.get('max_acceleration_A_deg_B_deg_C_mm_per_s2', [10., 5., 1.5]))*unit
        velocity = self.initial_velocity.copy()
        if np.any(np.abs(velocity) > vmax+1e-12):
            self.sensor.close()
            raise ValueError('initial_velocity exceeds configured velocity limits')
        self.max_velocity = np.abs(velocity)
        state, status, reason = 'tracking', 'FAIL', 'MAX_ITERATIONS'
        recovery_attempts = recovery_successes = memory_frames = 0
        recovery_started = retract_origin = None
        observed_streak = settled = 0
        final_observation = None
        measured_error = None
        last_tracking = {}
        loss_reason = None
        max_iterations = self.settings['max_iterations']
        iterations = count() if max_iterations is None else range(int(max_iterations))
        try:
            for iteration in iterations:
                time_s = iteration*dt
                self.reference.set_task(q)
                self.sensor.observation_time_s = getattr(self.sensor, 'observation_time_origin_s', 0.)+time_s
                if self.observation_hook is not None:
                    self.observation_hook(iteration, {
                        'controller_state': state, 'time_s': time_s,
                        'task_A_rad_B_rad_C_m': q.tolist(), 'recovery_attempt': recovery_attempts})
                self._prepare_observation(iteration, time_s)
                observation = self.sensor.observe()
                self._record_oct(iteration, observation)
                final_observation = observation
                self.final_observation = observation
                geometry = observation.get('geometry', {})
                tracking = {
                    'target_source': 'unavailable', 'target_observed': False,
                    'cached_target_age_s': None, 'used_target_world_mm': None,
                    'estimated_position_error_um': None,
                }
                target = tip = None
                quality_um = None
                fatal = None
                try:
                    if self.measurement_goal is not None:
                        if geometry.get('tip_world_mm') is None:
                            raise RuntimeError(geometry.get('tip_error', 'TIP_NOT_OBSERVABLE'))
                        tip = np.asarray(geometry['tip_world_mm'], dtype=float)
                        target_quality = {'fit_quality_mm': .002, 'residual_mm': 0.}
                        tracking.update(target_source='visual_retraction_waypoint', target_observed=False,
                                        cached_target_age_s=None, tip_world_mm=tip.tolist())
                        target = self.measurement_goal.copy()
                    else:
                        tracking.update(tracker.resolve(observation, time_s))
                        if not allow_memory and tracking['target_source'] != 'observed':
                            raise RuntimeError(geometry.get('target_error', 'TARGET_NOT_VISIBLE'))
                        target = np.asarray(tracking['used_target_world_mm'], dtype=float)
                        tip = np.asarray(tracking['tip_world_mm'], dtype=float)
                        target_quality = tracking['target_quality']
                    target = self._visual_goal(target, tip, observation)
                    tracking['used_target_world_mm'] = target.tolist()
                    tip_quality = geometry['tip_quality']
                    quality_um = 1000.*(float(target_quality['fit_quality_mm'])+
                                        float(tip_quality['fit_quality_mm']))
                    if not math.isfinite(quality_um) or quality_um > self.settings['max_geometry_fit_quality_um']:
                        raise RuntimeError('LOW_QUALITY_VISUAL_GEOMETRY')
                    measured_error = float(np.linalg.norm(target-tip)*1000.)
                    tracking['estimated_position_error_um'] = measured_error
                    if tracking['target_source'] == 'registered_memory':
                        memory_frames += 1
                except RuntimeError as error:
                    loss_reason = str(error)
                    # A registered target may never be replaced by a newly
                    # observed position that violates the static assumption.
                    if loss_reason == 'TARGET_MOVED_REPLAN_REQUIRED' or not allow_memory:
                        fatal = loss_reason
                    target = tip = None
                    measured_error = None
                    tracking.update({
                        'target_source': 'unavailable', 'target_observed': False,
                        'used_target_world_mm': None, 'estimated_position_error_um': None})
                    if state == 'tracking' and fatal is None:
                        if recovery_attempts >= 3:
                            fatal = 'OCCLUSION_RECOVERY_ATTEMPTS_EXHAUSTED'
                        else:
                            recovery_attempts += 1
                            recovery_started = time_s
                            retract_origin = None
                            observed_streak = settled = 0
                            state = 'braking'
                            tracker.reacquire()

                valid = target is not None and tip is not None
                if state != 'tracking':
                    # A fixed visual retraction waypoint needs a fresh tip,
                    # not the marker. Physical goals still require a fresh
                    # target too: memory cannot count toward reacquisition.
                    reobserved = valid and (self.measurement_goal is not None
                                            or tracking.get('target_observed', False))
                    observed_streak = observed_streak+1 if reobserved else 0
                    if observed_streak >= 3:
                        state = 'reobserving'
                        if np.all(np.abs(velocity) < 1e-12):
                            state = 'tracking'
                            recovery_successes += 1
                            recovery_started = None
                            observed_streak = settled = 0
                    if state != 'tracking' and time_s-recovery_started >= 4.-1e-10:
                        fatal = 'OCCLUSION_RECOVERY_TIMEOUT'
                tracking['controller_state'] = state
                tracking['error_code'] = loss_reason if not valid else None
                observation['tracking'] = tracking
                last_tracking = tracking

                # The same columns are present on normal, occluded, recovery,
                # failed and terminal frames, including command-free frames.
                row = {
                    'iteration': iteration, 'time_s': time_s,
                    'u_error_px': None, 'v_error_px': None, 'depth_error_mm': None,
                    'pixel_error_px': None,
                    'tip_error_um': float(self.reference.metrics()['tip_target_error_um']),
                    'estimated_position_error_um': measured_error, 'completion_error_um': None,
                    'geometry_fit_quality_um': quality_um,
                    'tip_feature_source': geometry.get('tip_quality', {}).get('source'),
                    'tip_support_points': geometry.get('tip_quality', {}).get('support'),
                    'tip_fit_residual_mm': geometry.get('tip_quality', {}).get('residual_mm'),
                    'target_fit_residual_mm': geometry.get('target_quality', {}).get('residual_mm'),
                    'target_pixels': observation['target_pixels'], 'tip_pixels': observation['tip_pixels'],
                    'A_deg': math.degrees(q[0]), 'B_deg': math.degrees(q[1]), 'C_mm': q[2]*1000.,
                    'target_source': tracking['target_source'], 'target_observed': tracking['target_observed'],
                    'cached_target_age_s': tracking['cached_target_age_s'], 'controller_state': state,
                    'used_target_world_mm': tracking['used_target_world_mm'], 'error_code': tracking['error_code'],
                    'recovery_attempt': recovery_attempts,
                    'command_velocity_A_deg_B_deg_C_mm_per_s': None,
                    'command_acceleration_A_deg_B_deg_C_mm_per_s2': None,
                }
                correction = None
                in_tolerance = False
                if valid:
                    error_mm = target-tip
                    pixel_tip, depth_tip = self.sensor.geometry.project(tip)
                    pixel_target, depth_target = self.sensor.geometry.project(target)
                    feature_error = np.r_[np.asarray(pixel_tip)-pixel_target, (depth_tip-depth_target)*1000.]
                    row.update(u_error_px=float(feature_error[0]), v_error_px=float(feature_error[1]),
                               depth_error_mm=float(feature_error[2]),
                               pixel_error_px=float(np.linalg.norm(feature_error[:2])))
                    if state == 'tracking':
                        jacobian = self._model_jacobian(q)
                        active = self.control_axes
                        normalized = jacobian[:, active]*vmax[active]
                        u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
                        if not len(singular) or singular[-1] < max(1e-10, singular[0]*1e-6):
                            fatal = 'SINGULAR_ACTUATION_JACOBIAN'
                        else:
                            damping = singular[0]*1e-4
                            correction = np.zeros(3)
                            correction[active] = vmax[active]*(vt.T@(
                                singular/(singular**2+damping**2)*(u.T@error_mm)))
                            complete_j = jacobian[:, self.completion_axes]
                            projected = complete_j@np.linalg.lstsq(complete_j, error_mm, rcond=None)[0]
                            row['completion_error_um'] = float(np.linalg.norm(projected)*1000.)
                            in_tolerance = row['completion_error_um'] < float(self.settings['position_tolerance_um'])
                self.rows.append(row)
                if self.iteration_callback is not None and self.iteration_callback(iteration, observation) is False:
                    fatal = 'USER_ABORTED'
                if fatal is not None:
                    reason = fatal
                    break
                settled = (settled+1 if state == 'tracking' and valid and in_tolerance and
                           np.all(np.abs(velocity) < 1e-12) else 0)
                if settled >= int(self.settings['settle_frames']):
                    status, reason = 'PASS', 'POSITION_ERROR_AND_SETTLED'
                    break
                if max_iterations is not None and iteration == int(max_iterations)-1:
                    break

                if state == 'tracking':
                    wanted = np.zeros(3) if in_tolerance else float(self.settings['geometry_gain_per_s'])*correction
                    braking = .6*np.sqrt(2*amax*np.abs(correction))
                    wanted = np.clip(wanted, -np.minimum(vmax, braking), np.minimum(vmax, braking))
                    wanted[~self.control_axes] = 0.
                else:
                    wanted = np.zeros(3)
                    if state == 'braking' and np.all(np.abs(velocity) < 1e-12):
                        state = 'retracting'
                        retract_origin = float(q[2])
                    if state == 'retracting':
                        remaining = max(0., float(q[2])-(retract_origin-.0005))
                        wanted[2] = -min(.0002, vmax[2], .6*math.sqrt(2*amax[2]*remaining))
                        if remaining < 1e-10 and abs(velocity[2]) < 1e-12:
                            state = 'reobserving'
                    # Recovering A/B always decelerate to zero. C motion is
                    # encoder-based and bounded; no hidden tip/target is used.
                next_velocity = np.clip(wanted, velocity-amax*dt, velocity+amax*dt)
                candidate = q+next_velocity*dt
                if retract_origin is not None and state != 'tracking' and candidate[2] < retract_origin-.0005-1e-12:
                    reason = 'OCCLUSION_RETRACTION_LIMIT'
                    break
                try:
                    candidate = self.reference.validate_task(candidate).copy()
                except ValueError:
                    reason = 'TASK_CONSTRAINT'
                    break
                if self.command_validator is not None and not self.command_validator(q, candidate):
                    reason = 'FUNDUS_MOTION_CONSTRAINT'
                    break
                acceleration = (next_velocity-velocity)/dt
                self.max_velocity = np.maximum(self.max_velocity, np.abs(next_velocity))
                self.max_acceleration = np.maximum(self.max_acceleration, np.abs(acceleration))
                row.update(command_velocity_A_deg_B_deg_C_mm_per_s=(next_velocity/unit).tolist(),
                           command_acceleration_A_deg_B_deg_C_mm_per_s2=(acceleration/unit).tolist())
                q, velocity = candidate, next_velocity
                self.final_velocity = velocity.copy()
            self.reference.set_task(q)
            if final_observation is not None and self.output_dir is not None and self.settings['save_diagnostics']:
                save_png(self.output_dir/'camera_final.png', final_observation['rgb'])
                save_png(self.output_dir/'guidance_final.png', guidance_overlay(final_observation))
        finally:
            self.sensor.close()
        report = self._finish_geometry(initial, final_observation, status, reason, measured_error,
                                       settled, dt, unit, vmax, amax)
        report.update(
            occlusion_handling=self.settings.get('occlusion_handling', 'stop'),
            static_target_assumption=allow_memory, memory_frame_count=memory_frames,
            recovery_attempt_count=recovery_attempts, recovery_success_count=recovery_successes,
            final_target_source=last_tracking.get('target_source', 'unavailable'),
            final_target_observed=bool(last_tracking.get('target_observed', False)),
            final_tracking=last_tracking,
            target_detection=('current-frame calibrated geometry or explicitly labelled bounded static-target memory'
                              if allow_memory else 'current-frame calibrated geometry; no target-memory bridge'),
            recovery_limits={'max_attempts': 3, 'max_retract_mm_per_attempt': .5,
                             'max_duration_s_per_attempt': 4., 'reobserve_frames': 3,
                             'retract_speed_mm_per_s': .2},
            recovery_last_observation_error=loss_reason,
        )
        return report

    def _run_geometry_strategy(self, mode):
        """Run one explicit occlusion strategy with current RGB-D feedback.

        S0a is intentionally handled by the historical fail-closed path.
        S0b, S1, S2 and S3 share one observation/control loop so perception,
        actuation limits, endpoint acceptance and timing remain comparable.
        """
        from rcm_target_tracking import TargetMemory

        parameters = strategy_settings(self.settings)
        tracker = TargetMemory(
            sample_period_s=float(self.settings['sample_period_s']),
            max_age_s=parameters['memory_max_age_s'],
            min_observations=3,
            max_shift_mm=parameters['memory_max_shift_mm'],
            stability_mm=parameters['memory_stability_mm'],
            max_travel_mm=parameters['memory_max_travel_mm'])
        q = self.reference.current_task().copy()
        initial = self.reference.metrics()
        dt = float(self.settings['sample_period_s'])
        navigation = self.reference.config.get('navigation', {})
        unit = np.array([math.pi/180, math.pi/180, .001])
        vmax = np.asarray(navigation.get(
            'max_velocity_A_deg_B_deg_C_mm_per_s', [4., 2., .6]))*unit
        amax = np.asarray(navigation.get(
            'max_acceleration_A_deg_B_deg_C_mm_per_s2',
            [10., 5., 1.5]))*unit
        velocity = self.initial_velocity.copy()
        if np.any(np.abs(velocity) > vmax+1e-12):
            self.sensor.close()
            raise ValueError('initial_velocity exceeds configured velocity limits')
        command_q = q.copy()
        motor = getattr(self.reference, 'motor_execution', None)
        status, reason = 'FAIL', 'MAX_ITERATIONS'
        state = 'tracking'
        loss_started_s = None
        retract_origin_mm = None
        observed_streak = settled = 0
        memory_frames = memory_successes = 0
        recovery_attempts = recovery_successes = 0
        last_nonzero_velocity = velocity.copy()
        open_loop_velocity = None
        last_tracking = {}
        last_loss_reason = None
        measured_error = None
        final_observation = None
        orderly_stop = False
        self.max_velocity = np.abs(velocity)
        self.max_acceleration = np.zeros(3)
        max_iterations = self.settings['max_iterations']
        iterations = count() if max_iterations is None else range(int(max_iterations))
        try:
            for iteration in iterations:
                if motor is None:
                    self.reference.set_task(q)
                    time_s = (getattr(self.sensor, 'observation_time_origin_s', 0.)
                              + iteration*dt)
                else:
                    if iteration > 0:
                        self.reference.set_task(command_q, motor_dt=dt)
                    q = self.reference.current_task().copy()
                    time_s = float(motor.time_s)
                self.sensor.observation_time_s = time_s
                if self.observation_hook is not None:
                    self.observation_hook(iteration, {
                        'controller_state': state, 'time_s': time_s,
                        'task_A_rad_B_rad_C_m': q.tolist(),
                        'recovery_attempt': recovery_attempts})
                self._prepare_observation(iteration, time_s)
                observation = self.sensor.observe()
                self._record_oct(iteration, observation)
                final_observation = self.final_observation = observation
                geometry = observation.get('geometry', {})
                target_observed = bool(
                    observation.get('target_pixel') is not None
                    and geometry.get('target_world_mm') is not None)
                tip_observed = geometry.get('tip_world_mm') is not None
                target = tip = target_quality = None
                target_source = 'unavailable'
                current_observed = target_observed and tip_observed
                loss_reason = None

                if mode == S2_TARGET_MEMORY:
                    try:
                        tracking = tracker.resolve(observation, time_s)
                    except RuntimeError as error:
                        tracking = copy.deepcopy(tracker.last_result)
                        loss_reason = str(error)
                    else:
                        target = np.asarray(tracking['used_target_world_mm'], dtype=float)
                        tip = np.asarray(geometry['tip_world_mm'], dtype=float)
                        target_quality = tracking['target_quality']
                        target_source = tracking['target_source']
                        if target_source == 'registered_memory':
                            memory_frames += 1
                elif current_observed:
                    target = np.asarray(geometry['target_world_mm'], dtype=float)
                    tip = np.asarray(geometry['tip_world_mm'], dtype=float)
                    target_quality = geometry['target_quality']
                    target_source = 'observed'
                    tracking = {
                        'target_source': 'observed', 'target_observed': True,
                        'cached_target_age_s': 0.,
                        'used_target_world_mm': target.tolist(),
                    }
                else:
                    tracking = {
                        'target_source': 'unavailable', 'target_observed': False,
                        'cached_target_age_s': None, 'used_target_world_mm': None,
                    }
                    if not target_observed:
                        loss_reason = geometry.get('target_error', 'TARGET_NOT_VISIBLE')
                    else:
                        loss_reason = geometry.get('tip_error', 'TIP_NOT_OBSERVABLE')
                if loss_reason is not None:
                    last_loss_reason = loss_reason

                # Current-frame visibility drives reacquisition. A cached target
                # never counts as a fresh observation of the occluded marker.
                if current_observed:
                    observed_streak += 1
                else:
                    observed_streak = 0

                if state == 'tracking' and target is None:
                    loss_started_s = time_s
                    if recovery_attempts >= parameters['max_recovery_attempts']:
                        reason = 'OCCLUSION_RECOVERY_ATTEMPTS_EXHAUSTED'
                    else:
                        recovery_attempts += 1
                        if mode == S0B_STOP_BRAKE:
                            state = 'braking'
                        elif mode == S1_OPEN_LOOP:
                            state = 'open_loop'
                            open_loop_velocity = last_nonzero_velocity.copy()
                        elif mode == S3_RETRACT_REOBSERVE:
                            state = 'braking'
                        elif mode == S2_TARGET_MEMORY:
                            reason = loss_reason or 'TARGET_MEMORY_UNAVAILABLE'
                        else:
                            reason = loss_reason or 'OCCLUSION_OBSERVATION_UNAVAILABLE'
                elif state == 'memory':
                    if mode != S2_TARGET_MEMORY:
                        raise RuntimeError('TARGET_MEMORY_STATE_IN_INVALID_STRATEGY')
                    if target is None:
                        reason = loss_reason or 'TARGET_MEMORY_UNAVAILABLE'
                    elif target_observed and tip_observed:
                        state = 'reobserving'
                        observed_streak = 1
                        memory_successes += 1

                if mode == S2_TARGET_MEMORY and state == 'tracking' and target_source == 'registered_memory':
                    state = 'memory'

                if state == 'reobserving':
                    if observed_streak >= parameters['reacquire_frames'] and np.all(
                            np.abs(velocity) < 1e-12):
                        state = 'tracking'
                        recovery_successes += 1
                        loss_started_s = None
                        observed_streak = settled = 0
                    elif (loss_started_s is not None
                          and time_s-loss_started_s > parameters['recovery_timeout_s']):
                        reason = 'OCCLUSION_RECOVERY_TIMEOUT'

                if state == 'open_loop' and current_observed:
                    state = 'reobserving'
                    observed_streak = 1
                if state == 'open_loop' and (
                        loss_started_s is None
                        or time_s-loss_started_s > parameters['max_open_loop_s']):
                    reason = 'OPEN_LOOP_TIMEOUT'
                if (state in ('braking', 'retracting', 'reobserving')
                        and loss_started_s is not None
                        and time_s-loss_started_s > parameters['recovery_timeout_s']):
                    reason = 'OCCLUSION_RECOVERY_TIMEOUT'

                correction = np.zeros(3)
                completion_error_um = None
                in_tolerance = False
                quality_um = None
                if target is not None and tip is not None and state in ('tracking', 'memory'):
                    visual_goal = self._visual_goal(target, tip, observation)
                    target = np.asarray(visual_goal, dtype=float)
                    error_mm = target-tip
                    measured_error = float(np.linalg.norm(error_mm)*1000.)
                    tip_quality = geometry['tip_quality']
                    quality_um = 1000.*(
                        float(target_quality['fit_quality_mm'])
                        + float(tip_quality['fit_quality_mm']))
                    tracking['estimated_position_error_um'] = measured_error
                    if quality_um > float(self.settings['max_geometry_fit_quality_um']):
                        reason = 'LOW_QUALITY_VISUAL_GEOMETRY'
                    else:
                        jacobian = self._model_jacobian(q)
                        active = self.control_axes
                        normalized = jacobian[:, active]*vmax[active]
                        u, singular, vt = np.linalg.svd(normalized, full_matrices=False)
                        if (not len(singular)
                                or singular[-1] < max(1e-10, singular[0]*1e-6)):
                            reason = 'SINGULAR_ACTUATION_JACOBIAN'
                        else:
                            damping = singular[0]*1e-4
                            correction[active] = vmax[active]*(vt.T@(
                                singular/(singular**2+damping**2)*(u.T@error_mm)))
                            complete_j = jacobian[:, self.completion_axes]
                            projected = complete_j@np.linalg.lstsq(
                                complete_j, error_mm, rcond=None)[0]
                            completion_error_um = float(np.linalg.norm(projected)*1000.)
                            in_tolerance = completion_error_um < float(
                                self.settings['position_tolerance_um'])
                tracking['controller_state'] = state
                tracking['error_code'] = loss_reason if target is None else None
                observation['tracking'] = tracking
                last_tracking = copy.deepcopy(tracking)
                row = {
                    'iteration': iteration, 'time_s': time_s,
                    'u_error_px': None, 'v_error_px': None,
                    'depth_error_mm': None, 'pixel_error_px': None,
                    'tip_error_um': float(self.reference.metrics()['tip_target_error_um']),
                    'estimated_position_error_um': measured_error,
                    'completion_error_um': completion_error_um,
                    'geometry_fit_quality_um': quality_um,
                    'tip_feature_source': geometry.get('tip_quality', {}).get('source'),
                    'tip_support_points': geometry.get('tip_quality', {}).get('support'),
                    'tip_fit_residual_mm': geometry.get('tip_quality', {}).get('residual_mm'),
                    'target_fit_residual_mm': (
                        None if target_quality is None
                        else target_quality.get('residual_mm')),
                    'target_pixels': observation.get('target_pixels'),
                    'tip_pixels': observation.get('tip_pixels'),
                    'A_deg': math.degrees(q[0]), 'B_deg': math.degrees(q[1]),
                    'C_mm': q[2]*1000.,
                    'target_source': target_source,
                    'target_observed': target_observed,
                    'cached_target_age_s': tracking.get('cached_target_age_s'),
                    'controller_state': state,
                    'used_target_world_mm': (
                        None if target is None else target.tolist()),
                    'error_code': tracking.get('error_code'),
                    'recovery_attempt': recovery_attempts,
                    'command_velocity_A_deg_B_deg_C_mm_per_s': None,
                    'command_acceleration_A_deg_B_deg_C_mm_per_s2': None,
                }
                if target is not None and tip is not None:
                    pixel_tip, depth_tip = self.sensor.geometry.project(tip)
                    pixel_target, depth_target = self.sensor.geometry.project(target)
                    feature_error = np.r_[
                        np.asarray(pixel_tip)-pixel_target,
                        (depth_tip-depth_target)*1000.]
                    row.update(
                        u_error_px=float(feature_error[0]),
                        v_error_px=float(feature_error[1]),
                        depth_error_mm=float(feature_error[2]),
                        pixel_error_px=float(np.linalg.norm(feature_error[:2])))
                self.rows.append(row)
                if self.iteration_callback is not None and self.iteration_callback(
                        iteration, observation) is False:
                    reason = 'USER_ABORTED'
                if reason != 'MAX_ITERATIONS':
                    break

                settled = (settled+1 if (
                    state in ('tracking', 'memory') and in_tolerance
                    and np.all(np.abs(velocity) < 1e-12)) else 0)
                if settled >= int(self.settings['settle_frames']):
                    status, reason = 'PASS', 'POSITION_ERROR_AND_SETTLED'
                    break
                if max_iterations is not None and iteration == int(max_iterations)-1:
                    break

                wanted = np.zeros(3)
                if state in ('tracking', 'memory'):
                    wanted = (np.zeros(3) if in_tolerance
                              else float(self.settings['geometry_gain_per_s'])*correction)
                    braking = .6*np.sqrt(2*amax*np.abs(correction))
                    wanted = np.clip(
                        wanted, -np.minimum(vmax, braking), np.minimum(vmax, braking))
                    wanted[~self.control_axes] = 0.
                elif state == 'open_loop':
                    wanted = np.clip(open_loop_velocity, -vmax, vmax)
                elif state == 'braking':
                    target_velocity = np.zeros(3)
                    wanted = np.clip(target_velocity, velocity-amax*dt, velocity+amax*dt)
                    if np.all(np.abs(velocity) < 1e-12):
                        if mode == S0B_STOP_BRAKE:
                            reason = 'STOP_BRAKE_COMPLETE_WITHOUT_REACQUISITION'
                            orderly_stop = True
                            break
                        if mode == S3_RETRACT_REOBSERVE:
                            state = 'retracting'
                            retract_origin_mm = float(q[2]*1000.)
                elif state == 'retracting':
                    travelled = retract_origin_mm-float(q[2]*1000.)
                    remaining = parameters['retract_mm']-travelled
                    if remaining <= parameters['retract_tolerance_mm']:
                        state = 'reobserving'
                    else:
                        wanted[2] = -min(
                            parameters['retract_speed_mm_s']*.001,
                            vmax[2],
                            .6*math.sqrt(2*amax[2]*max(0., remaining*.001)))
                elif state == 'reobserving':
                    wanted = np.clip(
                        np.zeros(3), velocity-amax*dt, velocity+amax*dt)
                next_velocity = np.clip(
                    wanted, velocity-amax*dt, velocity+amax*dt)
                acceleration = (next_velocity-velocity)/dt
                candidate = (command_q if motor is not None else q)+next_velocity*dt
                if (state == 'retracting' and retract_origin_mm is not None
                        and float(candidate[2]*1000.) < retract_origin_mm-parameters['retract_mm']+1e-9):
                    candidate[2] = (retract_origin_mm-parameters['retract_mm'])*.001
                try:
                    candidate = self.reference.validate_task(candidate).copy()
                except ValueError:
                    reason = 'TASK_CONSTRAINT'
                    break
                if self.command_validator is not None and not self.command_validator(q, candidate):
                    reason = 'FUNDUS_MOTION_CONSTRAINT'
                    break
                self.max_velocity = np.maximum(self.max_velocity, np.abs(next_velocity))
                self.max_acceleration = np.maximum(
                    self.max_acceleration, np.abs(acceleration))
                row.update(
                    command_velocity_A_deg_B_deg_C_mm_per_s=(
                        next_velocity/unit).tolist(),
                    command_acceleration_A_deg_B_deg_C_mm_per_s2=(
                        acceleration/unit).tolist())
                if np.linalg.norm(next_velocity) > 1e-12:
                    last_nonzero_velocity = next_velocity.copy()
                q, velocity = candidate, next_velocity
                command_q = candidate.copy()
                self.final_velocity = velocity.copy()

            if motor is None:
                self.reference.set_task(q)
            elif status != 'PASS' and not orderly_stop:
                motor.hold()
            if final_observation is not None and self.output_dir is not None and self.settings['save_diagnostics']:
                save_png(self.output_dir/'camera_final.png', final_observation['rgb'])
                save_png(self.output_dir/'guidance_final.png', guidance_overlay(final_observation))
        finally:
            self.sensor.close()
        report = self._finish_geometry(
            initial, final_observation, status, reason, measured_error,
            settled, dt, unit, vmax, amax,
            simulation_abort_hold=not orderly_stop)
        report.update(
            occlusion_handling=mode,
            occlusion_strategy_label=mode,
            strategy_state=state,
            recovery_attempt_count=recovery_attempts,
            recovery_success_count=recovery_successes,
            memory_frame_count=memory_frames,
            memory_success_count=memory_successes,
            last_loss_reason=last_loss_reason,
            final_target_source=last_tracking.get('target_source', 'unavailable'),
            final_target_observed=bool(last_tracking.get('target_observed', False)),
            strategy_limits={
                'reacquire_frames': parameters['reacquire_frames'],
                'max_recovery_attempts': parameters['max_recovery_attempts'],
                'max_open_loop_s': parameters['max_open_loop_s'],
                'memory_max_age_s': parameters['memory_max_age_s'],
                'memory_max_travel_mm': parameters['memory_max_travel_mm'],
                'retract_mm': parameters['retract_mm'],
                'retract_speed_mm_s': parameters['retract_speed_mm_s'],
                'retract_tolerance_mm': parameters['retract_tolerance_mm'],
                'recovery_timeout_s': parameters['recovery_timeout_s'],
            },
            occlusion_handling_semantics={
                'S0a_stop_instant': 'immediate simulator hold; reports the resulting acceleration discontinuity',
                'S0b_stop_brake': 'maximum-acceleration command deceleration to zero, then fail closed',
                'S1_open_loop': 'continue the last admissible non-zero command until fresh observations return',
                'S2_target_memory': 'use a bounded registered static-target point while requiring a fresh tip measurement',
                'S3_retract_reobserve': 'brake, retract a bounded C-axis distance, then require fresh observations',
            }.get(mode),
        )
        return report

    def _finish_geometry(self, initial, final_observation, status, reason, measured_error,
                         settled, dt, unit, vmax, amax, simulation_abort_hold=True):
        abort_velocity = self.final_velocity.copy()
        if status != 'PASS' and simulation_abort_hold:
            # A lost observation/constraint violation is an immediate simulator
            # hold, not a physical brake. Account for that velocity discontinuity
            # instead of incorrectly reporting only the preceding bounded moves.
            self.max_acceleration = np.maximum(self.max_acceleration, np.abs(abort_velocity)/dt)
            self.final_velocity = np.zeros(3)
        metrics = self.reference.metrics()
        acceptance = metrics['tip_target_error_um'] < float(self.settings['endpoint_tolerance_um'])
        overall = status == 'PASS' and (acceptance or not self.independent_endpoint_acceptance)
        geometry = {} if final_observation is None else final_observation.get('geometry', {})
        return {
            'result': 'PASS' if overall else 'FAIL',
            'reason': ('INDEPENDENT_ACCURACY_REJECTED' if status == 'PASS' and self.independent_endpoint_acceptance and not acceptance else reason),
            'controller_result': status, 'controller_reason': reason,
            'execution_mode': ('visual_servo_motor' if getattr(self.reference, 'motor_execution', None)
                               is not None else 'visual_servo_kinematic'),
            'motor_execution': (self.reference.motor_execution.summary()
                                if getattr(self.reference, 'motor_execution', None) is not None else None),
            'feedback_controller_used': True,
            'position_feedback': 'calibrated_visible_geometry', 'camera_name': self.sensor.camera_name,
            'iterations': len(self.rows), 'initial_tip_world_mm': initial['tip_world_mm'],
            'final_tip_world_mm': metrics['tip_world_mm'], 'target_world_mm': metrics['target_world_mm'],
            'final_tip_target_error_um': metrics['tip_target_error_um'],
            'final_estimated_position_error_um': measured_error,
            'final_geometry_measurement': geometry,
            'final_oct_metadata': None if final_observation is None else final_observation.get('oct_metadata'),
            'final_oct_navigation_depth': None if final_observation is None else final_observation.get('oct_navigation_depth'),
            'final_visual_error': None if not self.rows else {
                'calibrated_u_v_depth_mm': [self.rows[-1][k] for k in ('u_error_px', 'v_error_px', 'depth_error_mm')]},
            'camera_resolution_px': [self.sensor.width, self.sensor.height],
            'max_velocity_A_deg_B_deg_C_mm_per_s': (self.max_velocity/unit).tolist(),
            'max_acceleration_A_deg_B_deg_C_mm_per_s2': (self.max_acceleration/unit).tolist(),
            'trajectory_rate_limits_pass': bool(np.all(self.max_velocity <= vmax+1e-12) and np.all(self.max_acceleration <= amax+1e-12)),
            'initial_command_velocity_A_deg_B_deg_C_mm_per_s': (self.initial_velocity/unit).tolist(),
            'final_command_velocity_A_deg_B_deg_C_mm_per_s': (self.final_velocity/unit).tolist(),
            'simulation_abort_hold': bool(status != 'PASS' and simulation_abort_hold),
            'velocity_before_abort_hold_A_deg_B_deg_C_mm_per_s': (
                (abort_velocity/unit).tolist()
                if status != 'PASS' and simulation_abort_hold else None),
            'physical_braking_validated': False,
            'settled_frames': settled,
            'terminal_stop_validated': bool(status == 'PASS' and np.all(np.abs(self.final_velocity) < 1e-12)),
            'independent_accuracy_acceptance_pass': bool(acceptance) if self.independent_endpoint_acceptance else None,
            'independent_endpoint_acceptance_required': self.independent_endpoint_acceptance,
            'measurement_goal_type': ('measured_tip_retraction_waypoint' if self.measurement_goal is not None else
                                      'transformed_measured_target' if self.target_transform is not None else 'physical_target'),
            'independent_accuracy_tolerance_um': float(self.settings['endpoint_tolerance_um']),
            'stop_condition': 'measured controllable 3D error and consecutive zero-command-velocity observations',
            'tip_detection': ('RGB-observed shaft/end plane or calibrated visible artificial green fiducial' if
                self.settings.get('perception_backend') == 'strict_rgbd' else
                'visible shaft cylinder plus identified end plane or actual apex sphere; no taper-pixel substitution'),
            'perception_backend': self.settings.get('perception_backend','simulation_ids'),
            'simulator_ids_used_for_control': self.settings.get('perception_backend') != 'strict_rgbd',
            'hidden_geometry_render_used_for_control': self.settings.get('perception_backend') != 'strict_rgbd',
            'target_detection': ('current RGB detector and single visible-scene depth' if
                self.settings.get('perception_backend') == 'strict_rgbd' else
                'current-frame RGB/segmentation and calibrated sphere depth fit'),
            'control_axes': self.control_axes.tolist(), 'completion_axes': self.completion_axes.tolist(),
            'known_target_pose_used_for_feature_calibration': False,
            'ground_truth_used_for_stop': False,
            'ground_truth_used_for_post_run_acceptance': self.independent_endpoint_acceptance,
            'calibration_inputs': 'fixed camera, RCM, needle lengths/radii and target marker radius',
            'scope': 'MuJoCo RGB-D geometry and kinematic control; geometric calibration and ideal depth are simulation assumptions, not hardware accuracy.',
        }

    def run(self):
        if self.settings.get('position_feedback') == 'geometry':
            mode = self.settings.get('occlusion_handling', 'stop')
            if mode == LEGACY_STATIC_MEMORY_REOBSERVE:
                return self._run_geometry_with_memory()
            if mode == S0A_STOP_INSTANT:
                report = self._run_geometry()
                report.update(
                    occlusion_handling=mode,
                    occlusion_strategy_label=mode,
                    occlusion_handling_semantics=(
                        'immediate simulator hold; failure report includes the '
                        'resulting command-velocity discontinuity'))
                return report
            if mode in STRATEGIES:
                return self._run_geometry_strategy(mode)
            return self._run_geometry()
        q = self.reference.current_task().copy()
        target_observation = self.sensor.observe()
        if target_observation["target_pixel"] is None or target_observation["target_depth_m"] is None:
            self.sensor.close()
            raise RuntimeError("TARGET_NOT_VISIBLE_AT_START")
        target_reference = None
        feature_bias = np.zeros(3)
        if self.use_target_pose_calibration:
            # Legacy/privileged mode. Strict visual collection leaves this
            # disabled so the controller never reads reference.target_task().
            target_q = self.reference.target_task()
            target_reference = {
                "target_pixel": target_observation["target_pixel"],
                "target_depth_m": target_observation["target_depth_m"],
            }
            try:
                self.reference.set_task(target_q)
                seed_observation = self.sensor.observe()
                feature_bias = VisualServoSensor.error(
                    seed_observation, target_reference,
                    allow_cached_target=True).copy()
            finally:
                self.reference.set_task(q)
        initial = self.reference.metrics()
        status, reason = "FAIL", "MAX_ITERATIONS"
        final_observation = None
        previous_velocity = self.initial_velocity.copy()
        dt = float(self.settings["sample_period_s"])
        navigation = self.reference.config.get("navigation", {})
        vmax = np.asarray(navigation.get("max_velocity_A_deg_B_deg_C_mm_per_s", [4., 2., .6])) * np.array([math.pi/180, math.pi/180, .001])
        amax = np.asarray(navigation.get("max_acceleration_A_deg_B_deg_C_mm_per_s2", [10., 5., 1.5])) * np.array([math.pi/180, math.pi/180, .001])
        if np.any(np.abs(previous_velocity) > vmax+1e-12):
            self.sensor.close()
            raise ValueError("initial_velocity exceeds configured velocity limits")
        self.max_velocity = np.maximum(self.max_velocity, np.abs(previous_velocity))
        settled_count = 0
        require_settled = bool(self.settings.get("require_settled", False))
        max_iterations = self.settings.get("max_iterations")
        iterations = count() if max_iterations is None else range(int(max_iterations))
        for iteration in iterations:
            self.reference.set_task(q)
            self._prepare_observation(iteration, (
                getattr(self.reference, 'motor_execution', None).time_s
                if getattr(self.reference, 'motor_execution', None) is not None
                else getattr(self.sensor, 'observation_time_origin_s', 0.)+iteration*dt))
            observation = self.sensor.observe()
            self._record_oct(iteration, observation)
            try:
                error = VisualServoSensor.error(observation, target_reference)
            except RuntimeError as exc:
                reason = str(exc)
                break
            control_error = error.copy()
            control_error -= feature_bias
            pixel_norm = float(np.linalg.norm(control_error[:2]))
            depth_abs = abs(float(control_error[2]))
            tip_error_um = float(self.reference.metrics()["tip_target_error_um"])
            self.rows.append({"iteration": iteration, "time_s": iteration*dt,
                              "u_error_px": float(control_error[0]),
                              "v_error_px": float(control_error[1]), "depth_error_mm": float(control_error[2]),
                              "pixel_error_px": pixel_norm, "tip_error_um": tip_error_um,
                              "target_pixels": observation["target_pixels"], "tip_pixels": observation["tip_pixels"],
                              "A_deg": math.degrees(q[0]), "B_deg": math.degrees(q[1]), "C_mm": q[2]*1000.})
            final_observation = observation
            if self.iteration_callback is not None:
                if self.iteration_callback(iteration, observation) is False:
                    reason = "USER_ABORTED"
                    break
            if (self.output_dir is not None and self.settings["save_diagnostics"] and
                    (iteration == 0 or (max_iterations is not None and iteration == int(max_iterations)-1))):
                save_png(self.output_dir/f"camera_{iteration:04d}.png", observation["rgb"])
                save_png(self.output_dir/f"guidance_{iteration:04d}.png", guidance_overlay(observation))
            tolerance = np.array([float(self.settings["pixel_tolerance"]),
                                  float(self.settings["pixel_tolerance"]),
                                  float(self.settings["depth_tolerance_mm"])])
            in_tolerance = bool(np.all(np.abs(control_error)[self.completion_axes]
                                       <= tolerance[self.completion_axes]))
            if in_tolerance and not require_settled:
                status, reason = "PASS", "VISUAL_ERROR_TOLERANCE"
                break
            settled_count = settled_count+1 if in_tolerance and np.all(np.abs(previous_velocity) < 1e-12) else 0
            if require_settled and settled_count >= int(self.settings["settle_frames"]):
                status, reason = "PASS", "VISUAL_ERROR_AND_SETTLED"
                break
            if in_tolerance and require_settled:
                wanted_velocity = np.zeros(3)
            else:
                jacobian = self._jacobian(q, control_error, feature_bias, target_reference)
                image_jacobian = jacobian[:2, :2]
                depth_c_gain = jacobian[2, 2]
                if (not np.isfinite(image_jacobian).all() or np.linalg.matrix_rank(image_jacobian) < 2 or
                        not np.isfinite(depth_c_gain) or abs(depth_c_gain) < 1e-9):
                    reason = "SINGULAR_VISUAL_JACOBIAN"
                    break
                damp = float(self.settings["damping"])
                gain = float(self.settings["step_gain"])
                update_ab = -gain * image_jacobian.T @ np.linalg.solve(
                    image_jacobian @ image_jacobian.T + damp*damp*np.eye(2), control_error[:2])
                update_c = -gain*control_error[2]/depth_c_gain
                update = np.r_[update_ab, update_c]
                update[~self.control_axes] = 0.
                wanted_velocity = np.clip(update/dt, -vmax, vmax)
                if require_settled:
                    # Local feature-based displacement estimate bounds approach speed.
                    braking_speed = .7*np.sqrt(2*amax*np.abs(update/gain))
                    wanted_velocity = np.clip(wanted_velocity, -braking_speed, braking_speed)
            velocity = np.clip(wanted_velocity, previous_velocity-amax*dt, previous_velocity+amax*dt)
            acceleration = (velocity-previous_velocity)/dt
            self.max_velocity = np.maximum(self.max_velocity, np.abs(velocity))
            self.max_acceleration = np.maximum(self.max_acceleration, np.abs(acceleration))
            self.rows[-1].update(command_velocity_A_deg_B_deg_C_mm_per_s=(velocity*np.array([180/math.pi, 180/math.pi, 1000.])).tolist(),
                                 command_acceleration_A_deg_B_deg_C_mm_per_s2=(acceleration*np.array([180/math.pi, 180/math.pi, 1000.])).tolist())
            update = velocity*dt
            candidate = q + update
            try:
                candidate = self.reference.validate_task(candidate).copy()
            except ValueError:
                reason = "TASK_CONSTRAINT"
                break
            if self.command_validator is not None and not self.command_validator(q, candidate):
                reason = "FUNDUS_MOTION_CONSTRAINT"
                break
            q = candidate
            previous_velocity = velocity
            self.final_velocity = velocity.copy()
        self.reference.set_task(q)
        metrics = self.reference.metrics()
        if self.output_dir is not None and self.settings["save_diagnostics"]:
            final_observation = self.sensor.observe()
            save_png(self.output_dir/"camera_final.png", final_observation["rgb"])
            save_png(self.output_dir/"guidance_final.png", guidance_overlay(final_observation))
        final_calibrated_error = None
        if final_observation is not None:
            try:
                final_calibrated_error = (VisualServoSensor.error(final_observation, target_reference)-feature_bias).tolist()
            except RuntimeError:
                pass  # Preserve the original failure report when the target is occluded.
        report = {
            "result": status, "reason": reason, "execution_mode": "visual_servo_kinematic",
            "feedback_controller_used": True, "camera_name": self.sensor.camera_name,
            "iterations": len(self.rows), "initial_tip_world_mm": initial["tip_world_mm"],
            "final_tip_world_mm": metrics["tip_world_mm"], "target_world_mm": metrics["target_world_mm"],
            "final_tip_target_error_um": metrics["tip_target_error_um"],
            "final_visual_error": None if final_observation is None else {
                "target_pixel": final_observation["target_pixel"], "tip_pixel": final_observation["tip_pixel"],
                "target_depth_m": final_observation["target_depth_m"], "tip_depth_m": final_observation["tip_depth_m"],
                "calibrated_u_v_depth_mm": final_calibrated_error,
            },
            "camera_resolution_px": [self.sensor.width, self.sensor.height],
            "feature_bias_at_target_calibration_u_v_depth_mm": feature_bias.tolist(),
            "depth_bias_at_target_calibration_mm": float(feature_bias[2]),
            "max_velocity_A_deg_B_deg_C_mm_per_s": (self.max_velocity*np.array([180/math.pi, 180/math.pi, 1000.])).tolist(),
            "max_acceleration_A_deg_B_deg_C_mm_per_s2": (self.max_acceleration*np.array([180/math.pi, 180/math.pi, 1000.])).tolist(),
            "trajectory_rate_limits_pass": bool(np.all(self.max_velocity <= vmax+1e-12) and np.all(self.max_acceleration <= amax+1e-12)),
            "initial_command_velocity_A_deg_B_deg_C_mm_per_s": (self.initial_velocity*np.array([180/math.pi, 180/math.pi, 1000.])).tolist(),
            "final_command_velocity_A_deg_B_deg_C_mm_per_s": (self.final_velocity*np.array([180/math.pi, 180/math.pi, 1000.])).tolist(),
            "settled_frames": settled_count,
            "terminal_stop_validated": bool(status == "PASS" and require_settled and np.all(np.abs(self.final_velocity) < 1e-12)),
            "target_detection": ("current-frame RGB centroid + segmentation depth; stop if target is fully occluded"
                                 if not self.use_target_pose_calibration else
                                 "current-frame RGB centroid + segmentation depth; stop if target is fully occluded; "
                                 "initial target measurement is used only for privileged feature calibration"),
            "tip_detection": "physical distal/taper geometry segmentation + depth; viewer-only aid excluded",
            "control_axes": "A/B from image-plane error; C from calibrated rendered-depth error",
            "stop_condition": ("calibrated pixel/depth tolerances and consecutive zero-command-velocity observations"
                               if require_settled else "calibrated pixel norm and rendered-depth error within configured tolerances"),
            "known_target_pose_used_for_feature_calibration": self.use_target_pose_calibration,
            "target_reference_pixel_cached_from_initial_observation": (
                None if target_reference is None else target_reference["target_pixel"]),
            "target_reference_depth_m_cached_from_initial_observation": (
                None if target_reference is None else target_reference["target_depth_m"]),
            "ground_truth_used_for_stop": False,
            "ground_truth_used_for_post_run_acceptance": True,
            "scope": "MuJoCo visual servo simulation; no hardware or tissue-contact validation",
        }
        self.sensor.close()
        return report

    def write(self, report, output_dir):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if self.oct_records:
            with (directory/'oct_observations.jsonl').open('w', encoding='utf-8') as handle:
                for record in self.oct_records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            report['oct_observation_audit'] = 'oct_observations.jsonl'
            report['oct_observation_count'] = len(self.oct_records)
        frame = getattr(self.sensor, 'last_oct_frame', None)
        if frame is not None:
            frame.save(directory/'final_oct_frame.npz')
            report['final_oct_frame_archive'] = 'final_oct_frame.npz'
            report['oct_archive_scope'] = 'One final registered simulation frame per phase; full raw depth sequence is not exported.'
        (directory/"visual_servo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if self.rows:
            with (directory/"visual_servo.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
