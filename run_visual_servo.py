#!/usr/bin/env python3
"""Run the fixed intraocular-camera visual servo simulation headlessly."""
from __future__ import annotations

import argparse
import json
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco

from rcm_reference import HERE, Reference
from rcm_visual_servo import VisualServoController, guidance_overlay


def resize_nearest(image, width, height):
    """Resize an RGB image without adding a GUI/image dependency."""
    rows = np.linspace(0, image.shape[0]-1, height).astype(int)
    cols = np.linspace(0, image.shape[1]-1, width).astype(int)
    return np.ascontiguousarray(image[rows[:, None], cols])


def run_viewer(reference, controller, view_camera):
    """Run the visual servo loop while keeping a MuJoCo viewer responsive."""
    import mujoco.viewer

    events = queue.SimpleQueue()
    camera_names = ("overview", "side", "needle_close", "eye_internal")
    camera_index = 0 if view_camera == "free" else camera_names.index(view_camera)
    camera_free = view_camera == "free"
    with mujoco.viewer.launch_passive(reference.model, reference.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            if camera_free:
                mujoco.mjv_defaultFreeCamera(reference.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = reference.camera_id(view_camera)
            viewer.opt.sitegroup[:] = 0
            viewer.opt.geomgroup[2] = 1
        original_run = controller.run

        def handle_camera_events():
            nonlocal camera_free, camera_index
            with viewer.lock():
                while not events.empty():
                    key = events.get()
                    if key == 70:  # F
                        if camera_free:
                            camera_free = False
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = reference.camera_id(camera_names[camera_index])
                            print("Camera:", camera_names[camera_index])
                        else:
                            camera_free = True
                            mujoco.mjv_defaultFreeCamera(reference.model, viewer.cam)
                            print("Camera: free (mouse rotate/pan/zoom)")
                    elif key == 86:  # V
                        camera_index = (camera_index+1) % len(camera_names)
                        camera_free = False
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = reference.camera_id(camera_names[camera_index])
                        print("Camera:", camera_names[camera_index])
                viewer.opt.geomgroup[2] = 1

        def update_view(iteration, observation):
            if not viewer.is_running():
                return False
            handle_camera_events()
            viewport = viewer.viewport
            # Keep the eye-internal camera readable in the main MuJoCo window.
            # The physical 60 um distal segment is sub-pixel at the native
            # sensor scale, so a larger inset is a viewer aid only; it does
            # not change the geometry or the controller measurements.
            inset_width = min(480, max(1, int(viewport.width*.42)))
            inset_height = max(1, int(inset_width*controller.sensor.height/controller.sensor.width))
            inset = resize_nearest(guidance_overlay(observation), inset_width, inset_height)
            image_rect = mujoco.MjrRect(max(0, viewport.width-inset_width-10), 10,
                                       inset_width, inset_height)
            row = controller.rows[-1]
            viewer.set_images((image_rect, inset))
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Visual guidance\nPixel error\nDepth error\nA / B / C\nViewer\nController",
                (f"iteration {iteration}\n{row['pixel_error_px']:.3f} px\n"
                 f"{row['depth_error_mm']:.3f} mm\n"
                 f"{row['A_deg']:.2f} deg / {row['B_deg']:.2f} deg / {row['C_mm']:.2f} mm\n"
                 "F: free/fixed, V: fixed cameras\neye_internal\n"
                 "yellow: display-only needle aid"),
            ))
            viewer.sync()
            time.sleep(0.04)
            return viewer.is_running()

        controller.iteration_callback = update_view
        report = original_run()
        if viewer.is_running():
            with viewer.lock():
                viewer.opt.geomgroup[2] = 1
            viewer.sync()
            print("Visual servo finished. Close the viewer window to exit.")
            while viewer.is_running():
                handle_camera_events()
                viewer.sync()
                time.sleep(0.05)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--target-world", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"))
    parser.add_argument("--target-robot", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"))
    parser.add_argument("--output-dir", type=Path, default=HERE/"runs"/"历史记录"/"04_sessions"/"visual_servo"/f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--no-viewer", action="store_true", help="Run headlessly without opening a GUI")
    parser.add_argument("--camera", choices=("free", "eye_internal", "overview", "side", "needle_close"),
                        default="eye_internal", help="Viewer camera; control still uses eye_internal")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.target_robot is not None:
            config["reference_scene"]["target_robot_mm"] = list(args.target_robot)
        elif args.target_world is not None:
            reference = Reference(config)
            config["reference_scene"]["target_robot_mm"] = reference.world_to_robot(args.target_world).tolist()
        reference = Reference(config)
        controller = VisualServoController(reference, output_dir=args.output_dir)
        report = controller.run() if args.no_viewer else run_viewer(reference, controller, args.camera)
        controller.write(report, args.output_dir)
        print(json.dumps({**report, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0 if report["result"] == "PASS" else 1
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
