#!/usr/bin/env python3
"""Deterministic checks for the intraocular camera and visual servo loop."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import mujoco

from rcm_reference import Reference
from rcm_visual_servo import DEFAULT_SERVO, VisualServoController, VisualServoSensor


def verify(output_dir=None):
    # No fixed iteration cap: exercise a run past the previous 400-step limit,
    # then stop it through the same callback used when the viewer closes.
    import rcm_visual_servo

    class StubReference:
        def __init__(self):
            self.q = np.array([2., 2., .04])
            self.config = {"navigation": {
                "max_velocity_A_deg_B_deg_C_mm_per_s": [4., 2., .6],
                "max_acceleration_A_deg_B_deg_C_mm_per_s2": [10., 5., 1.5],
            }}

        def current_task(self):
            return self.q.copy()

        def set_task(self, q):
            self.q = np.asarray(q, dtype=float).copy()

        def validate_task(self, q):
            return np.asarray(q, dtype=float)

        def metrics(self):
            tip = np.array([self.q[0], self.q[1], self.q[2]*1000.])
            return {"tip_world_mm": tip.tolist(), "target_world_mm": [0., 0., 0.],
                    "tip_target_error_um": float(np.linalg.norm(tip)*1000.)}

    class StubSensor:
        camera_name = "eye_internal"
        width = height = 640
        error = staticmethod(VisualServoSensor.error)

        def __init__(self, reference, settings):
            self.reference = reference

        def observe(self):
            q = self.reference.q
            return {"target_pixel": [0., 0.], "tip_pixel": q[:2].tolist(),
                    "target_depth_m": 0., "tip_depth_m": float(q[2]),
                    "target_pixels": 1, "tip_pixels": 1,
                    "rgb": np.zeros((1, 1, 3), dtype=np.uint8)}

        def close(self):
            pass

    unlimited_settings = {**DEFAULT_SERVO, "max_iterations": None,
                          "use_target_pose_calibration": False,
                          "pixel_tolerance": .001, "depth_tolerance_mm": .001,
                          "save_diagnostics": False}
    stub_reference = StubReference()
    original_sensor_class = rcm_visual_servo.VisualServoSensor
    try:
        rcm_visual_servo.VisualServoSensor = StubSensor
        unlimited_controller = VisualServoController(
            stub_reference, settings=unlimited_settings)
        unlimited_controller.iteration_callback = lambda iteration, observation: iteration < 405
        unlimited_report = unlimited_controller.run()
    finally:
        rcm_visual_servo.VisualServoSensor = original_sensor_class
    assert unlimited_report["iterations"] == 406, unlimited_report
    assert unlimited_report["reason"] == "USER_ABORTED", unlimited_report

    reference = Reference()
    camera_id = reference.camera_id("eye_internal")
    assert reference.model.ncam == 4
    assert np.isfinite(reference.data.cam_xpos[camera_id]).all()
    sensor = VisualServoSensor(reference)
    auxiliary_geom_id = int(reference.model.geom("needle_tip_visual_marker").id)
    physical_tip_geom_id = int(reference.model.geom("needle_tip_marker").id)
    physical_needle_geom_ids = {
        int(reference.model.geom(name).id)
        for name in ("needle_proximal", "needle_distal", "needle_distal_tip", "needle_tip_marker")
    }
    assert int(reference.model.geom_group[auxiliary_geom_id]) == 2
    assert sensor.tip_geom_id == physical_tip_geom_id
    assert auxiliary_geom_id not in sensor.needle_geom_ids
    assert set(sensor.needle_geom_ids.tolist()) == physical_needle_geom_ids
    # The enlarged 0.20 mm aid is viewer-only. Toggle its visual alpha and
    # prove that all three sensor passes are byte-for-byte unchanged.
    sensor_frames_with_aux = {
        mode: sensor._render(mode, include_fundus=True)
        for mode in ("rgb", "depth", "segmentation")
    }
    old_alpha = float(reference.model.geom_rgba[auxiliary_geom_id, 3])
    reference.model.geom_rgba[auxiliary_geom_id, 3] = 0.
    try:
        sensor_frames_without_aux = {
            mode: sensor._render(mode, include_fundus=True)
            for mode in ("rgb", "depth", "segmentation")
        }
    finally:
        reference.model.geom_rgba[auxiliary_geom_id, 3] = old_alpha
    assert all(np.array_equal(sensor_frames_with_aux[mode], sensor_frames_without_aux[mode])
               for mode in sensor_frames_with_aux)
    assert not np.any(sensor_frames_with_aux["segmentation"][:, :, 0] == auxiliary_geom_id)
    initial = sensor.observe()
    assert initial["target_pixel"] is not None and initial["target_pixels"] >= 1
    assert initial["tip_pixel"] is not None and initial["tip_pixels"] >= 1
    assert initial["target_depth_m"] > 0 and initial["tip_depth_m"] > 0
    retina_geom_id = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "retina_visual")
    assert retina_geom_id >= 0
    assert reference.model.geom_type[retina_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    retina_mesh_id = int(reference.model.geom_dataid[retina_geom_id])
    vertex_start = int(reference.model.mesh_vertadr[retina_mesh_id])
    vertex_count = int(reference.model.mesh_vertnum[retina_mesh_id])
    retina_vertices = reference.model.mesh_vert[vertex_start:vertex_start+vertex_count]
    assert np.ptp(retina_vertices[:, 2]) > .01
    absent_markers = [f"lesion_reticle_{index}" for index in range(4)] + [
        f"lesion_display_reticle_{index}" for index in range(4)]
    assert all(mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
               for name in absent_markers)
    reference.set_task(reference.target_task())
    reached = sensor.observe()
    occlusion = sensor.visibility_metrics(reached["segmentation"])
    assert reached["target_pixel"] is not None and reached["target_pixels"] > 0
    assert reached["tip_pixel"] is not None and reached["tip_pixels"] > 0
    assert occlusion["target_occlusion_state"] == "partial_needle", occlusion
    assert not occlusion["tip_fully_occluded"], occlusion
    assert occlusion["target_pixels_unoccluded"] > 0
    assert 0 < occlusion["target_pixels_occluded_by_needle"] < occlusion["target_pixels_unoccluded"]
    cached_target = {
        "target_pixel": initial["target_pixel"],
        "target_depth_m": initial["target_depth_m"],
    }
    uncached_error = VisualServoSensor.error(reached, cached_target)
    assert uncached_error.shape == (3,) and np.isfinite(uncached_error).all()
    calibrated_error = VisualServoSensor.error(reached, cached_target, allow_cached_target=True)
    assert calibrated_error.shape == (3,) and np.isfinite(calibrated_error).all()
    strict_config = copy.deepcopy(reference.config)
    strict_config["navigation"]["visual_servo"]["use_target_pose_calibration"] = False
    strict_reference = Reference(strict_config)
    strict_reference.set_task(strict_reference.target_task())
    strict_controller = VisualServoController(strict_reference)
    strict_report = strict_controller.run()
    assert strict_report["result"] == "FAIL"
    assert strict_report["reason"] in {"TIP_NOT_VISIBLE", "TARGET_NOT_VISIBLE"}
    missing_target = dict(reached, target_pixel=None, target_depth_m=None)
    try:
        VisualServoSensor.error(missing_target)
    except RuntimeError as exc:
        assert "TARGET_NOT_VISIBLE" in str(exc), str(exc)
    else:
        raise AssertionError("A missing target must fail closed without a visual reference")
    sensor.close()
    report = {
        "result": "PASS",
        "single_center_target_visible_at_initial_pose": "PASS",
        "curved_fundus_backdrop": "PASS",
        "peripheral_target_markers_absent": "PASS",
        "single_target_partially_occluded_at_goal": "PASS",
        "needle_tip_uses_physical_geometry": "PASS",
        "sensor_passes_exclude_viewer_only_auxiliary_marker": "PASS",
        "cached_visual_target_reference_is_limited_to_goal_calibration": "PASS",
        "missing_target_without_cached_visual_reference_rejected": "PASS",
        "strict_visual_controller_fails_closed_after_physical_tip_loss": "PASS",
        "unlimited_visual_servo_continues_past_400_until_explicit_abort": "PASS",
        "visual_servo_goal_execution": "see single-target motion report",
        "cached_target_pixel": cached_target["target_pixel"],
        "cached_target_depth_m": cached_target["target_depth_m"],
        "finite_calibration_error_after_target_occlusion": calibrated_error.tolist(),
        "target_pixels_initial": int(initial["target_pixels"]),
        "target_pixels_unoccluded_at_goal": int(occlusion["target_pixels_unoccluded"]),
        "target_pixels_occluded_at_goal": int(occlusion["target_pixels_occluded_by_needle"]),
        "strict_controller_failure_reason": strict_report["reason"],
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir)/"visual_servo_verification.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = verify(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
