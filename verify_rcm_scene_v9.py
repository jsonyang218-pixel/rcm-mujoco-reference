"""Construction and layer-1/layer-2 checks for the v9 Reference adapter.

Exercises the real adapter and the real v9 scanned model.  Full visual-servo
end-to-end (layer 3) is not claimed here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rcm_reference import Reference  # noqa: E402
from rcm_scene_v9 import RCMSceneV9  # noqa: E402
from rcm_visual_servo import VisualServoSensor  # noqa: E402


def verify():
    ref = Reference()
    scene = RCMSceneV9()
    checks = {}

    checks["home_task_norm"] = float(np.linalg.norm(scene.current_task()))
    checks["encoders_match_legacy_home"] = bool(
        np.allclose(scene.encoders(), ref.encoders(), atol=1e-6, rtol=0))

    mapped_legacy_tip = scene.robot_to_world(ref.forward_kinematics(ref.current_task()))
    checks["tip_world_mapping_diff_mm"] = float(
        np.linalg.norm(scene.site_mm("needle_tip") - mapped_legacy_tip))

    eye_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
    eye_center = scene.data.geom_xpos[eye_id] * 1000.0
    checks["eye_center_mapping_diff_mm"] = float(np.linalg.norm(
        eye_center - scene.robot_to_world(ref.config["reference_scene"]["eye_center_robot_mm"])))
    checks["target_mapping_diff_mm"] = float(np.linalg.norm(
        scene.site_mm("lesion") - scene.robot_to_world(ref.config["reference_scene"]["target_robot_mm"])))

    scene.set_task(np.zeros(3))
    mujoco.mj_step(scene.model, scene.data)
    expected_proximal_m = float(scene.config["needle"]["proximal_length_mm"]) * 0.001
    checks["needle_proximal_z_mm"] = float(
        scene.model.site_pos[scene.sites["needle_base"]][2] * 1000.0)
    checks["needle_proximal_matches_config"] = bool(
        abs(checks["needle_proximal_z_mm"] - expected_proximal_m * 1000.0) < 1e-9)
    checks["needle_tip_junction_distance_mm"] = float(np.linalg.norm(
        np.asarray(scene.site_mm("needle_tip"))
        - np.asarray(scene.site_mm("driver_junction"))))
    checks["needle_tip_junction_matches_config"] = bool(
        abs(checks["needle_tip_junction_distance_mm"]
            - float(scene.config["needle"]["distal_length_mm"])) < 1e-9)
    checks["needle_entry_rcm_distance_mm"] = float(
        scene.needle_eye_geometry()["entry_rcm_distance_mm"])
    checks["needle_entry_is_junction"] = bool(
        scene.needle_eye_geometry()["entry_needle_part"] == "junction")

    sensor_mapping = {
        "rcm_world_position": "rcm_site",
        "needle_tip_world_position": "needle_tip",
        "needle_junction_world_position": "driver_junction",
        "lesion_world_position": "lesion",
    }
    sensor_errors_um = {}
    for sensor_name, site_name in sensor_mapping.items():
        sensor_id = int(mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name))
        if sensor_id < 0:
            sensor_errors_um[sensor_name] = None
            continue
        address = int(scene.model.sensor_adr[sensor_id])
        dimension = int(scene.model.sensor_dim[sensor_id])
        reading_mm = scene.data.sensordata[address:address + dimension] * 1000.0
        sensor_errors_um[sensor_name] = float(
            np.linalg.norm(reading_mm - scene.site_mm(site_name)) * 1000.0)
    checks["position_sensor_errors_um"] = sensor_errors_um

    segmentation = np.asarray(scene.render_camera("eye_internal", "segmentation"))
    object_ids = segmentation[:, :, 0].astype(np.int32)
    geom_mask = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    scene_group_pixels = {}
    for group in range(6):
        ids = np.flatnonzero(scene.model.geom_group == group).astype(np.int32)
        scene_group_pixels[str(group)] = int(
            (geom_mask & np.isin(object_ids, ids)).sum())
    checks["scene_group_pixels"] = scene_group_pixels
    checks["retina_visual_exists"] = bool(
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "retina_visual") >= 0)
    checks["lens_visual_exists"] = bool(
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "lens_visual") >= 0)

    rgb = scene.render_camera("eye_internal", "rgb")
    checks["eye_rgb_shape"] = list(rgb.shape)
    camera_pos, _ = scene.camera_pose("eye_internal")
    checks["camera_pose_finite"] = bool(np.isfinite(camera_pos).all())

    m = scene.metrics()
    checks["metrics_keys"] = sorted(k for k in m if k in {
        "tip_target_error_um", "tip_world_mm", "target_world_mm",
        "needle_eye_geometry", "distal_length_mm", "rcm_drift_um", "fk_agreement_um"})

    try:
        sensor = VisualServoSensor(scene)
        checks["visual_servo_sensor_constructs"] = True
        observation = sensor.observe()
        checks["visual_servo_target_pixels"] = int(observation["target_pixels"])
        checks["visual_servo_tip_pixels"] = int(observation["tip_pixels"])
        checks["visual_servo_target_world_mm"] = observation.get("geometry", {}).get("target_world_mm")
        checks["visual_servo_tip_world_mm"] = observation.get("geometry", {}).get("tip_world_mm")
        scene.set_task(np.array([0.0, 0.0, 0.008]))
        insertion_observation = sensor.observe()
        insertion_segmentation = np.asarray(insertion_observation["segmentation"])
        insertion_geom_ids = np.where(
            insertion_segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
            insertion_segmentation[:, :, 0], -1)
        distal_pixels = int((insertion_geom_ids
                             == int(scene.model.geom("needle_distal").id)).sum())
        marker_pixels = int((insertion_geom_ids
                             == int(scene.model.geom("needle_tip_marker").id)).sum())
        checks["visual_servo_C8_distal_plus_marker_pixels"] = (
            distal_pixels + marker_pixels)
        checks["visual_servo_C8_tip_pixels"] = int(
            insertion_observation["tip_pixels"])
        checks["visual_servo_C8_tip_world_mm"] = insertion_observation.get(
            "geometry", {}).get("tip_world_mm")
        scene.set_task(np.zeros(3))
        sensor.close()
    except Exception as exc:
        checks["visual_servo_sensor_constructs"] = False
        checks["visual_servo_sensor_error"] = f"{type(exc).__name__}: {exc}"

    worst_roundtrip = 0.0
    for b_deg in (-5.0, 0.0, 5.0):
        for c_mm in (0.0, 4.0, 8.0):
            q = np.array([0.0, math.radians(b_deg), c_mm * 0.001])
            scene.set_task(q)
            worst_roundtrip = max(worst_roundtrip,
                                  float(np.linalg.norm(scene.current_task() - q)))
    checks["task_roundtrip_max_err"] = worst_roundtrip

    checks["result"] = "PASS" if (
        checks["encoders_match_legacy_home"]
        and checks["tip_world_mapping_diff_mm"] < 0.001
        and checks["eye_center_mapping_diff_mm"] < 0.001
        and checks["target_mapping_diff_mm"] < 0.001
        and checks["needle_proximal_matches_config"]
        and checks["needle_tip_junction_matches_config"]
        and checks["needle_entry_rcm_distance_mm"] < 1e-5
        and checks["needle_entry_is_junction"]
        and len(checks["position_sensor_errors_um"]) == 4
        and all(error is not None and error < 1e-6
                for error in checks["position_sensor_errors_um"].values())
        and checks["scene_group_pixels"]["0"] == 0
        and checks["scene_group_pixels"]["1"] == 0
        and checks["scene_group_pixels"]["5"] > 0
        and checks["retina_visual_exists"]
        and checks["lens_visual_exists"]
        and checks["eye_rgb_shape"] == [1440, 1920, 3]
        and checks["camera_pose_finite"]
        and len(checks["metrics_keys"]) == 7
        and checks["visual_servo_sensor_constructs"]
        and checks["visual_servo_target_pixels"] > 0
        and checks["visual_servo_tip_pixels"] > 0
        and checks["visual_servo_target_world_mm"] is not None
        and checks["visual_servo_tip_world_mm"] is not None
        and checks["visual_servo_C8_distal_plus_marker_pixels"] > 0
        and checks["visual_servo_C8_tip_pixels"] > 0
        and checks["visual_servo_C8_tip_world_mm"] is not None
        and checks["task_roundtrip_max_err"] < 1e-9
    ) else "FAIL"
    return checks


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["result"] == "PASS" else 1)
