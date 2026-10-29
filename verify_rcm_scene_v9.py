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
        and checks["eye_rgb_shape"] == [1440, 1920, 3]
        and checks["camera_pose_finite"]
        and len(checks["metrics_keys"]) == 7
        and checks["visual_servo_sensor_constructs"]
        and checks["task_roundtrip_max_err"] < 1e-9
    ) else "FAIL"
    return checks


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
