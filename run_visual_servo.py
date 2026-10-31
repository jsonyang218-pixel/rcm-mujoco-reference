#!/usr/bin/env python3
"""Run the fixed intraocular-camera visual servo simulation headlessly."""
from __future__ import annotations

import argparse
import copy
import json
import queue
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco

from rcm_reference import HERE, Reference
from rcm_scene_v9 import RCMSceneV9
from rcm_visual_servo import (
    VisualServoController,
    _centroid,
    _segmentation_geom_ids,
    guidance_overlay,
)


def resize_nearest(image, width, height):
    """Resize an RGB image without adding a GUI/image dependency."""
    rows = np.linspace(0, image.shape[0]-1, height).astype(int)
    cols = np.linspace(0, image.shape[1]-1, width).astype(int)
    return np.ascontiguousarray(image[rows[:, None], cols])


def _v9_viewer_observation(reference):
    """Build a display-only v9 eye view matching the visible sensor groups."""
    rgb = reference.render_camera("eye_internal", "rgb")
    segmentation = reference.render_camera("eye_internal", "segmentation")
    geom_ids = _segmentation_geom_ids(segmentation)
    target_id = int(reference.model.geom("lesion_marker").id)
    marker_id = int(reference.model.geom("needle_tip_marker").id)
    cone_id = int(reference.model.geom("needle_distal_tip").id)
    needle_ids = np.asarray([
        int(reference.model.geom(name).id)
        for name in ("needle_proximal", "needle_distal",
                     "needle_distal_tip", "needle_tip_marker")
    ])
    target_mask = geom_ids == target_id
    marker_mask = geom_ids == marker_id
    cone_mask = geom_ids == cone_id
    needle_mask = np.isin(geom_ids, needle_ids)
    tip_pixel = _centroid(marker_mask)
    if tip_pixel is None and cone_mask.any():
        shaft_center = _centroid(needle_mask)
        if shaft_center is not None:
            rows, cols = np.where(cone_mask)
            endpoint = int(np.argmax(
                (cols - shaft_center[0]) ** 2
                + (rows - shaft_center[1]) ** 2))
            tip_pixel = np.array([float(cols[endpoint]), float(rows[endpoint])])
    return {
        "rgb": rgb,
        "target_pixel": _centroid(target_mask),
        "tip_pixel": tip_pixel,
        "needle_mask": needle_mask,
    }


def run_viewer(view_reference, control_reference, controller, view_camera):
    """Run the visual servo loop while keeping a MuJoCo viewer responsive."""
    import mujoco.viewer

    events = queue.SimpleQueue()
    available_cameras = tuple(
        name for name in ("overview", "side", "needle_close", "eye_internal")
        if mujoco.mj_name2id(
            view_reference.model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0)
    if view_camera != "free" and view_camera not in available_cameras:
        view_camera = "overview"
    camera_names = available_cameras
    hidden_environment_group = 1 if isinstance(view_reference, RCMSceneV9) else 0
    viewer_groups = np.ones(6, dtype=np.uint8)
    viewer_groups[hidden_environment_group] = 0
    camera_index = 0 if view_camera == "free" else camera_names.index(view_camera)
    camera_free = view_camera == "free"
    with mujoco.viewer.launch_passive(view_reference.model, view_reference.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            if camera_free:
                mujoco.mjv_defaultFreeCamera(view_reference.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = view_reference.camera_id(view_camera)
            viewer.opt.sitegroup[:] = 0
            viewer.opt.geomgroup[:] = viewer_groups
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
                            viewer.cam.fixedcamid = view_reference.camera_id(camera_names[camera_index])
                            print("Camera:", camera_names[camera_index])
                        else:
                            camera_free = True
                            mujoco.mjv_defaultFreeCamera(view_reference.model, viewer.cam)
                            print("Camera: free (mouse rotate/pan/zoom)")
                    elif key == 86:  # V
                        camera_index = (camera_index+1) % len(camera_names)
                        camera_free = False
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = view_reference.camera_id(camera_names[camera_index])
                        print("Camera:", camera_names[camera_index])
                viewer.opt.geomgroup[:] = viewer_groups

        def update_view(iteration, observation):
            if not viewer.is_running():
                return False
            handle_camera_events()
            if view_reference is not control_reference:
                view_reference.set_task(control_reference.current_task())
            view_observation = (observation if view_reference is control_reference else
                                _v9_viewer_observation(view_reference))
            viewport = viewer.viewport
            # Keep the eye-internal camera readable in the main MuJoCo window.
            # The physical 60 um distal segment is sub-pixel at the native
            # sensor scale, so a larger inset is a viewer aid only; it does
            # not change the geometry or the controller measurements.
            inset_width = min(480, max(1, int(viewport.width*.42)))
            inset_height = max(1, int(inset_width*view_reference.eye_camera["resolution_px"][1]
                                       / view_reference.eye_camera["resolution_px"][0]))
            inset = resize_nearest(guidance_overlay(view_observation), inset_width, inset_height)
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
                viewer.opt.geomgroup[:] = viewer_groups
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
    parser.add_argument("--viewer-backend", choices=("v9", "legacy"), default="v9",
                        help="Model used by the viewer; a separate viewer never advances a motor")
    parser.add_argument("--control-backend", choices=("v9", "legacy"), default="v9")
    parser.add_argument("--motion-backend", choices=("motor", "kinematic"), default="motor")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--perception-backend", choices=("strict_rgbd", "simulation_ids"))
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config['motor_execution'] = {**config.get('motor_execution', {}), 'enabled': False}
        config['navigation']['visual_servo']['perception_backend'] = (args.perception_backend or
            ('strict_rgbd' if args.control_backend == 'v9' else 'simulation_ids'))
        factory = RCMSceneV9 if args.control_backend == 'v9' else Reference
        if args.target_robot is not None:
            config["reference_scene"]["target_robot_mm"] = list(args.target_robot)
        elif args.target_world is not None:
            converter = factory(copy.deepcopy(config))
            try:
                config["reference_scene"]["target_robot_mm"] = converter.world_to_robot(args.target_world).tolist()
            finally:
                if isinstance(converter, RCMSceneV9):
                    converter.close()
        if args.max_iterations is not None:
            if args.max_iterations < 1:
                parser.error('--max-iterations must be positive')
            config['navigation']['visual_servo']['max_iterations'] = args.max_iterations
        reference = factory(config)
        if args.motion_backend == 'motor':
            reference.enable_motor_execution(config.get('motor_execution'))
        controller = VisualServoController(reference, output_dir=args.output_dir)
        view_reference = None
        if not args.no_viewer:
            view_reference = (reference if args.viewer_backend == args.control_backend else
                (RCMSceneV9(copy.deepcopy(config)) if args.viewer_backend == 'v9' else Reference(copy.deepcopy(config))))
        try:
            report = (controller.run() if args.no_viewer else
                      run_viewer(view_reference, reference, controller, args.camera))
        finally:
            if isinstance(view_reference, RCMSceneV9) and view_reference is not reference:
                view_reference.close()
            if isinstance(reference, RCMSceneV9):
                reference.close()
        controller.write(report, args.output_dir)
        print(json.dumps({**report, "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))
        return 0 if report["result"] == "PASS" else 1
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
