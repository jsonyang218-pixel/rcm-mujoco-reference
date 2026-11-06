#!/usr/bin/env python3
"""Verify the tip-coupled occluder in the real v9 RGB-D renderer."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings


OUTPUT = Path("runs/checks/tip_occluder")


def make_config(enabled):
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    config["reference_scene"]["eye_camera"]["resolution_px"] = [640, 480]
    config["reference_scene"]["tip_occlusion_fixture"] = {
        "enabled": bool(enabled),
        "radius_mm": .18,
        "trigger_progress": 0.,
        "target_coverage": 1.,
        "release_coverage": .1,
        "front_offset_mm": .05,
        "screen_direction_deg": 0.,
        "max_active_s": 3.,
    }
    config["navigation"]["visual_servo"].update({
        "perception_backend": "strict_rgbd",
        "position_feedback": "geometry",
        "occlusion_handling": "S0a_stop_instant",
        "save_diagnostics": False,
    })
    return config


def observe(scene):
    sensor = StrictRGBDSensor(scene, visual_servo_settings(scene))
    try:
        observation = sensor.observe()
        target = observation.get("geometry", {}).get("target_world_mm")
        return observation, None if target is None else np.asarray(target)
    finally:
        sensor.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    baseline = RCMSceneV9(make_config(False))
    try:
        before, before_target = observe(baseline)
    finally:
        baseline.close()

    scene = RCMSceneV9(make_config(True))
    try:
        if scene.tip_occlusion_fixture is None:
            raise RuntimeError("tip occlusion fixture was not constructed")
        scene.reset_tip_occlusion()
        during_event = scene.update_tip_occlusion(0, 0.)
        during, during_target = observe(scene)
        center = scene.model.geom_pos[scene.tip_occlusion_fixture.geom_id].copy()
        tip_before = scene.site_mm("needle_tip").copy()
        back_before = scene.site_mm("needle_base").copy()
        _, rotation_before = scene.camera_pose("eye_internal")

        scene.set_task(np.array([0., 0., -.0015]))
        after_event = scene.update_tip_occlusion(1, .05)
        after, after_target = observe(scene)
        audit = scene.tip_occlusion_audit()
    finally:
        scene.close()

    Image.fromarray(before["rgb"]).save(OUTPUT/"before.png")
    Image.fromarray(during["rgb"]).save(OUTPUT/"during.png")
    Image.fromarray(after["rgb"]).save(OUTPUT/"after.png")
    checks["baseline_target_visible"] = int(before["target_pixels"]) > 0
    checks["occluder_triggered"] = bool(during_event["active"])
    checks["target_hidden_during_occlusion"] = (
        int(during["target_pixels"]) < int(before["target_pixels"])
        and during_target is None)
    local_offset = np.asarray(
        audit["transitions"][0]["local_offset_mm"], dtype=float)
    basis, _ = scene.tip_occlusion_fixture._tool_basis(
        tip_before, back_before, rotation_before)
    expected_center_mm = tip_before+basis@local_offset
    checks["occluder_follows_tip_pose"] = bool(
        np.linalg.norm(center*1000.-expected_center_mm) < 1e-6)
    checks["coverage_reduced_after_retraction"] = bool(
        float(after_event.get("coverage_fraction", 1.))
        < float(during_event.get("coverage_fraction", 1.)) + 1e-12)
    checks["target_geometry_recovers_after_retraction"] = after_target is not None
    result = {
        "result": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "baseline_target_pixels": int(before["target_pixels"]),
        "during_target_pixels": int(during["target_pixels"]),
        "after_target_pixels": int(after["target_pixels"]),
        "during_event": during_event,
        "after_event": after_event,
        "fixture_audit": audit,
        "scope": ("Rendered v9 RGB-D occlusion check at 640x480; no hardware "
                  "or clinical validity claim."),
    }
    (OUTPUT/"tip_occluder_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
