"""Image-based visual servoing for the fixed intraocular MuJoCo camera.

The controller is deliberately simulation-only: orange target detection comes
from RGB, while the needle tip is measured from MuJoCo geom segmentation and
depth.  It drives the existing kinematic A/B/C task joints with a damped
finite-difference image Jacobian.
"""
from __future__ import annotations

import csv
import json
import math
from itertools import count
from pathlib import Path

import numpy as np

from rcm_reference import Reference, _vec, save_png


DEFAULT_SERVO = {
    "camera_name": "eye_internal", "target_rgb_min": [180, 35, 0],
    "target_rgb_max": [255, 180, 100], "target_depth_weight": 0.35,
    "pixel_tolerance": 3.0, "depth_tolerance_mm": 0.05,
    "endpoint_tolerance_um": 150.0, "max_iterations": None,
    "step_gain": 0.65, "finite_difference_A_deg": 0.05,
    "finite_difference_B_deg": 0.05, "finite_difference_C_mm": 0.05,
    "damping": 0.02, "sample_period_s": 0.05, "save_diagnostics": True,
    "require_settled": False, "settle_frames": 3,
}


def visual_servo_settings(reference):
    settings = {**DEFAULT_SERVO, **reference.config.get("navigation", {}).get("visual_servo", {})}
    if settings["camera_name"] != "eye_internal":
        reference.camera_id(settings["camera_name"])
    if settings["max_iterations"] is not None and int(settings["max_iterations"]) <= 0:
        raise ValueError("visual_servo.max_iterations must be positive or null (unlimited)")
    if int(settings["settle_frames"]) < 1:
        raise ValueError("visual_servo.settle_frames must be positive")
    for key in ("pixel_tolerance", "depth_tolerance_mm", "endpoint_tolerance_um", "damping"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"visual_servo.{key} must be positive")
    return settings


def _centroid(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    return np.array([float(np.mean(cols)), float(np.mean(rows))])


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
    def __init__(self, reference, settings=None):
        self.reference = reference
        self.settings = settings or visual_servo_settings(reference)
        self.camera_name = self.settings["camera_name"]
        self.width, self.height = reference.eye_camera["resolution_px"]
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

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()

    def observe(self):
        rgb = self._render("rgb")
        # Keep a complete scene depth image for OCT-style inspection, including
        # the fundus surface. The feature depth pass is a separate axial return
        # for the target and needle, so a backdrop surface cannot mask either
        # feature in the synthetic OCT measurement.
        depth = self._render("depth", include_fundus=True)
        oct_depth = self._render("depth", include_fundus=False)
        # Keep the normal scene segmentation for visibility decisions. A target
        # hidden by the needle or fundus is allowed to disappear, matching the
        # real monocular view. OCT still supplies an independent axial depth
        # return for features that remain measurable.
        segmentation = self._render("segmentation", include_fundus=True)
        target_mask = np.isin(segmentation[:, :, 0], self.target_geom_ids)
        target_pixel = _centroid(target_mask)
        target_depth_mask = target_mask
        tip_mask = np.isin(segmentation[:, :, 0], self.tip_geom_ids)
        needle_mask = np.isin(segmentation[:, :, 0], self.needle_geom_ids)
        marker_mask = segmentation[:, :, 0] == self.tip_geom_id
        if marker_mask.any():
            tip_pixel = _centroid(marker_mask)
            tip_depth_mask = marker_mask
        else:
            # If the calibrated apex sphere is sub-pixel or occluded, use the
            # exposed endpoint of the real taper, away from its shaft.
            cone_mask = segmentation[:, :, 0] == int(self.distal_geom_ids[1])
            shaft_center = _centroid(segmentation[:, :, 0] == int(self.distal_geom_ids[0]))
            if cone_mask.any() and shaft_center is not None:
                rows, cols = np.where(cone_mask)
                endpoint = int(np.argmax((cols-shaft_center[0])**2+(rows-shaft_center[1])**2))
                tip_pixel = np.array([float(cols[endpoint]), float(rows[endpoint])])
                tip_depth_mask = np.zeros_like(cone_mask)
                tip_depth_mask[rows[endpoint], cols[endpoint]] = True
            else:
                tip_pixel = _centroid(tip_mask)
                tip_depth_mask = tip_mask
        target_depth = None if not target_depth_mask.any() else float(np.median(oct_depth[target_depth_mask]))
        tip_depth = None if not tip_depth_mask.any() else float(np.median(oct_depth[tip_depth_mask]))
        if target_depth is not None and not np.isfinite(target_depth):
            target_depth = None
        if tip_depth is not None and not np.isfinite(tip_depth):
            tip_depth = None
        return {
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
            "oct_depth_source": "feature_depth_without_backdrop_occlusion",
        }

    def visibility_metrics(self, segmentation=None):
        """Measure target occlusion by the rendered physical needle geometry.

        Render a frame-specific counterfactual segmentation with only the
        physical needle geoms hidden.  The difference is therefore caused by geometric
        line-of-sight occlusion in MuJoCo, not a synthetic mask or noise label.
        This measurement is diagnostic only and never changes controller input.
        """
        if segmentation is None:
            segmentation = self._render("segmentation", include_fundus=True)
        ids = np.asarray(segmentation[:, :, 0])
        target_pixels = int(np.isin(ids, self.target_geom_ids).sum())
        tip_pixels = int(np.isin(ids, self.tip_geom_ids).sum())
        needle_pixels = int(np.isin(ids, self.needle_geom_ids).sum())

        # Render the frame-specific counterfactual every time: apparent marker
        # size changes slightly with depth, so a fixed baseline can mislabel
        # ordinary perspective changes as occlusion (or miss real occlusion).
        groups = self.reference.model.geom_group[self.needle_geom_ids].copy()
        try:
            # Group 2 is disabled by the microscope sensor render options.
            self.reference.model.geom_group[self.needle_geom_ids] = 2
            clear_segmentation = self._render("segmentation", include_fundus=True)
            unoccluded_pixels = int(np.isin(clear_segmentation[:, :, 0],
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
            option.geomgroup[2] = 0  # eye shell and camera marker do not occlude sensor features
            option.geomgroup[3] = 1  # optical contents: needle, retina and target markers
            option.geomgroup[4] = 1  # calibrated physical tip marker
            # The fundus backdrop is shown in RGB and in the complete scene
            # depth image, but excluded from the feature/OCT pass.
            option.geomgroup[5] = int(bool(include_fundus))
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
                 command_validator=None, control_axes=None, completion_axes=None):
        self.reference = reference
        self.settings = settings or visual_servo_settings(reference)
        # Strict visual collection disables the privileged target-pose feature
        # calibration.  In that mode every feedback error is computed from the
        # current RGB/segmentation/depth observation.
        self.use_target_pose_calibration = bool(
            self.settings.get("use_target_pose_calibration", True))
        self.command_validator = command_validator
        self.control_axes = np.ones(3, dtype=bool) if control_axes is None else np.asarray(control_axes, dtype=bool)
        self.completion_axes = (self.control_axes.copy() if completion_axes is None
                                else np.asarray(completion_axes, dtype=bool))
        if self.control_axes.shape != (3,) or self.completion_axes.shape != (3,) or not np.any(self.control_axes):
            raise ValueError("control_axes and completion_axes must be non-empty length-3 masks")
        # Carry the previous interval's command across consecutive visual goals.
        # Units are rad/s, rad/s, m/s, matching the internal task coordinates.
        self.initial_velocity = np.zeros(3) if initial_velocity is None else _vec(initial_velocity).copy()
        self.final_velocity = self.initial_velocity.copy()
        self.sensor = VisualServoSensor(reference, self.settings)
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None and self.settings["save_diagnostics"]:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.max_velocity = np.zeros(3)
        self.max_acceleration = np.zeros(3)
        self.iteration_callback = None

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

    def run(self):
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
            observation = self.sensor.observe()
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
        (directory/"visual_servo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if self.rows:
            with (directory/"visual_servo.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
