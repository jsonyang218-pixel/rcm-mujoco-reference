#!/usr/bin/env python3
"""Launch the offline MuJoCo reference. This script never connects to a robot."""
from __future__ import annotations

import argparse
import json
import math
import queue
import time
from pathlib import Path

import mujoco
import numpy as np

from rcm_reference import HERE, Reference, demo_motion, navigation_motion, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--export", type=Path, help="Write a standalone MJCF scene using the current config")
    parser.add_argument("--no-viewer", action="store_true", help="Evaluate offline without opening a GUI")
    parser.add_argument("--verify", action="store_true", help="Run the bundled kinematic checks and exit")
    parser.add_argument("--report", type=Path, help="Save verification JSON, used with --verify")
    parser.add_argument("--render", type=Path, help="Save a 1280x960 PNG of the evaluated pose")
    parser.add_argument("--camera", choices=("free", "overview", "side", "needle_close", "eye_internal"), default="free")
    parser.add_argument("--sample-time", type=float, help="Offline time in seconds; default is the motion endpoint")
    parser.add_argument("--target-world", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"))
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--pose", type=float, nargs=3, metavar=("A_DEG", "B_DEG", "C_MM"))
    initial.add_argument("--encoders", type=float, nargs=3, metavar=("EA", "EB", "EC"), help="Robot feedback counts, NOT handle counts")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--demo", choices=("a", "b", "c", "abc"))
    action.add_argument("--auto-aim", action="store_true")
    action.add_argument("--auto-reach", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.target_world is not None:
            # Construct just the coordinate conversion before compiling the
            # scene, so its lesion geom and target site are updated together.
            from rcm_reference import installation_rotation
            rotation = installation_rotation(config["installation"]["euler_xyz_deg"]) @ np.diag([1., -1., -1.])
            config["reference_scene"]["target_robot_mm"] = (
                rotation.T @ (np.array(args.target_world)-np.array(config["installation"]["rcm_world_mm"]))).tolist()
        reference = Reference(config)
        if args.export is not None:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            args.export.write_text(reference.xml, encoding="utf-8")
        if args.verify:
            from verify_reference import verify
            result = verify(reference)
            content = json.dumps(result, ensure_ascii=False, indent=2)
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(content+"\n", encoding="utf-8")
            print(content)
            return
        if args.pose is not None:
            a, b, c = args.pose
            reference.set_task([math.radians(a), math.radians(b), c*.001])
        if args.encoders is not None:
            reference.set_task(reference.task_from_encoders(args.encoders))
        motion = None
        if args.demo:
            motion = demo_motion(reference, args.demo)
            reference.set_task(motion.start)
        elif args.auto_aim or args.auto_reach:
            motion = navigation_motion(reference, reach=args.auto_reach)
        if args.sample_time is not None and motion is None:
            raise ValueError("--sample-time requires --demo, --auto-aim or --auto-reach")
        if args.no_viewer or args.render is not None:
            if motion is not None:
                duration = motion.duration if args.sample_time is None else args.sample_time
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("--sample-time must be finite and nonnegative")
                reference.set_task(motion.sample(duration))
            if args.render is not None:
                render(reference, args.render, "overview" if args.camera == "free" else args.camera)
            print(json.dumps(reference.metrics(), ensure_ascii=False, indent=2))
            return
        if args.sample_time is not None:
            raise ValueError("--sample-time requires --no-viewer or --render")
        interactive(reference, motion, args.camera)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Configuration or command error: {error}\n")


def interactive(reference, initial_motion, camera):
    import mujoco.viewer

    events = queue.SimpleQueue()
    steps = reference.manual_steps
    print("MuJoCo kinematic reference (offline, no hardware I/O).")
    print("1: A demo | 2: B demo | 3: C demo | 4: A/B/C sequence")
    print(f"Left/Right: A +/-{steps['A_deg']:g} deg | Up/Down: B +/-{steps['B_deg']:g} deg")
    print(f"PageUp/PageDown: C +/-{steps['C_mm']:g} mm | Space: aim only | Enter: aim + insert")
    print("R: reset | P: pause/resume | M: cases on/off | V: fixed camera views | F: free/fixed camera")
    print("Free camera: left-drag rotate, right-drag pan, mouse wheel zoom.")
    print("Limits:", reference.config["measured_limits"].get("scope", "calibrated_workspace") if reference.encoder_bounds is not None else "disabled")
    if reference.reset_encoders is not None:
        print("Zero-pose robot encoders:", reference.reset_encoders.tolist(),
              "; user confirmed:", reference.reference_spec.get("zero_pose_confirmed_by_user", False))
        print("At reset: A/B/C task=0; B source angle=", reference.alpha_home, "deg")
        print("Needle reference:", reference.tip_reference_mm, "mm; physically confirmed:", reference.reference_spec.get("physical_needle_reference_confirmed", False))
    print("Installation:", reference.config["installation"]["euler_xyz_deg"], "deg (see calibration config)")
    pending, elapsed, paused, shell = initial_motion, 0., False, True
    camera_names = ("overview", "side", "needle_close", "eye_internal")
    camera_index = 0 if camera == "free" else camera_names.index(camera)
    last_stamp = time.perf_counter()
    last_valid = reference.current_task().copy()

    with mujoco.viewer.launch_passive(reference.model, reference.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            if camera == "free":
                mujoco.mjv_defaultFreeCamera(reference.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    reference.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            viewer.opt.sitegroup[:] = 0  # tip and junction have no visible spheres
            viewer.opt.jointgroup[3] = 0
        while viewer.is_running():
            stamp = time.perf_counter()
            delta, last_stamp = min(stamp-last_stamp, .05), stamp
            with viewer.lock():
                while not events.empty():
                    key = events.get()
                    try:
                        if key in (49, 50, 51, 52):
                            candidate = demo_motion(reference, ("a", "b", "c", "abc")[key-49])
                            reference.set_task(candidate.start)
                            pending, elapsed, paused = candidate, 0., False
                        elif key in (32, 257, 335):
                            pending = navigation_motion(reference, reach=key != 32)
                            elapsed, paused = 0., False
                        elif key == 82:  # R
                            reference.set_task(reference.home)
                            pending, elapsed, paused = None, 0., False
                            print("Reset:", json.dumps(reference.metrics()))
                        elif key == 80:  # P
                            paused = not paused
                        elif key == 77:  # M
                            shell = not shell
                        elif key == 86:  # V
                            if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                                camera_index = (camera_index+1) % len(camera_names)
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_names[camera_index])
                            print("Camera:", camera_names[camera_index])
                        elif key == 70:  # F
                            if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
                                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                                viewer.cam.fixedcamid = mujoco.mj_name2id(
                                    reference.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                    camera_names[camera_index])
                                print("Camera:", camera_names[camera_index])
                            else:
                                mujoco.mjv_defaultFreeCamera(reference.model, viewer.cam)
                                print("Camera: free (mouse enabled)")
                        elif key in (262, 263, 264, 265, 266, 267):
                            q = reference.current_task()
                            index, sign = {262: (0, 1), 263: (0, -1), 265: (1, 1), 264: (1, -1),
                                           266: (2, 1), 267: (2, -1)}[key]
                            step = sign * steps[("A_deg", "B_deg", "C_mm")[index]]
                            q[index] += math.radians(step) if index < 2 else step*.001
                            # Reject a limit crossing without changing the pose.
                            reference.set_task(q)
                            pending, elapsed, paused = None, 0., False
                    except ValueError as error:
                        print("Motion rejected:", error)
                if pending is not None and not paused:
                    elapsed = min(pending.duration, elapsed+delta)
                    reference.set_task(pending.sample(elapsed))
                    if elapsed >= pending.duration:
                        print("Motion complete:", json.dumps(reference.metrics()))
                        pending = None
                try:
                    # Also close the dependent visual chain after edits made
                    # through MuJoCo's native joint panel while idle/paused.
                    reference.set_task(reference.current_task())
                    last_valid = reference.current_task().copy()
                except ValueError as error:
                    reference.set_task(last_valid)
                    print("Native joint edit rejected:", error)
                # Numeral shortcuts in MuJoCo may also toggle geom groups;
                # restore this reference scene's visibility after callbacks.
                viewer.opt.geomgroup[0] = 1
                viewer.opt.geomgroup[1] = int(shell)
                viewer.opt.geomgroup[2] = 1
                reference.data.time += delta
            viewer.sync()
            time.sleep(.01)


if __name__ == "__main__":
    main()
