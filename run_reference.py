#!/usr/bin/env python3
"""Launch the current eye-injected RCMSceneV9 MuJoCo simulation."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent

REQUIRED_GEOMS = (
    "scan_fixed_housing", "scan_parallel_cover", "scan_distal", "scan_C",
    "eye_reference", "retina_visual", "lens_visual", "lesion_marker",
    "needle_tip_marker",
)
REQUIRED_CAMERAS = ("overview", "eye_internal")


def _scene_summary(scene):
    metrics = scene.metrics()
    return {
        "status": "PASS",
        "model_backend": "RCMSceneV9",
        "scene": "v9 eye-injected scanned arm",
        "task_A_deg_B_deg_C_mm": (scene.current_task() * np.array(
            [180.0 / math.pi, 180.0 / math.pi, 1000.0])).tolist(),
        "geoms": {name: scene.model.geom(name).id for name in REQUIRED_GEOMS},
        "cameras": {name: scene.camera_id(name) for name in REQUIRED_CAMERAS},
        "rcm_world_mm": scene.site_mm("rcm_site").tolist(),
        "needle_tip_world_mm": scene.site_mm("needle_tip").tolist(),
        "lesion_world_mm": scene.site_mm("lesion").tolist(),
        "rcm_on_eye_surface": metrics["needle_eye_geometry"]["rcm_on_eye_surface"],
        "tip_inside_eye": metrics["needle_eye_geometry"]["tip_inside_eye"],
    }


def _set_pose(scene, pose):
    if pose is None:
        return
    a_deg, b_deg, c_mm = (float(value) for value in pose)
    if not all(math.isfinite(value) for value in (a_deg, b_deg, c_mm)):
        raise ValueError("--pose requires three finite values")
    scene.set_task(np.array([
        math.radians(a_deg), math.radians(b_deg), c_mm * 0.001,
    ]))


def _render(scene, destination, camera):
    from PIL import Image

    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    camera = "overview" if camera == "free" else camera
    with mujoco.Renderer(scene.model, width=1600, height=1080) as renderer:
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        option.geomgroup[:] = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        renderer.update_scene(scene.data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render().copy()).save(destination)
    return destination


def _viewer(scene, camera):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(
            scene.model, scene.data, show_left_ui=True, show_right_ui=True) as viewer:
        with viewer.lock():
            if camera == "free":
                mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = scene.camera_id(camera)
            viewer.opt.sitegroup[:] = 0
            # Hide only room/environment and illustrative linkage. Keep the
            # scanned robot, photograph-derived front tool, eye, lesion and needle.
            viewer.opt.geomgroup[:] = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true",
                        help="Evaluate the v9 scene without opening a GUI")
    parser.add_argument("--camera", choices=("free", "overview", "eye_internal"),
                        default="free")
    parser.add_argument("--pose", type=float, nargs=3,
                        metavar=("A_DEG", "B_RELATIVE_DEG", "C_MM"))
    parser.add_argument("--render", type=Path,
                        help="Render the v9 scene to a PNG and exit")
    parser.add_argument("--export", type=Path,
                        help="Export the current v9 runtime model as portable MJCF and exit")
    parser.add_argument("--verify", action="store_true",
                        help="Validate the current v9 eye scene and exit")
    parser.add_argument("--report", type=Path,
                        help="Write evaluation JSON (requires --no-viewer or --verify)")
    args = parser.parse_args(argv)
    if args.report is not None and not (args.no_viewer or args.verify):
        parser.error("--report requires --no-viewer or --verify")

    from rcm_scene_v9 import RCMSceneV9

    scene = RCMSceneV9()
    try:
        _set_pose(scene, args.pose)
        output = None
        if args.export is not None:
            from export_v9_eye_scene import export_scene
            output = export_scene(scene, args.export)
        elif args.render is not None:
            output = _render(scene, args.render, args.camera)
        elif args.verify or args.no_viewer:
            summary = _scene_summary(scene)
            if args.verify:
                for name in REQUIRED_GEOMS:
                    if scene.model.geom(name) is None:
                        raise ValueError(f"missing required geom: {name}")
                for name in REQUIRED_CAMERAS:
                    if scene.camera_id(name) < 0:
                        raise ValueError(f"missing required camera: {name}")
                if not summary["rcm_on_eye_surface"] or not summary["tip_inside_eye"]:
                    raise ValueError("v9 RCM/eye geometry check failed")
            content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            if args.report is not None:
                path = args.report.expanduser().resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            print(content, end="")
        else:
            _viewer(scene, args.camera)
        if output is not None:
            print(output)
        return 0
    finally:
        scene.close()


if __name__ == "__main__":
    raise SystemExit(main())
