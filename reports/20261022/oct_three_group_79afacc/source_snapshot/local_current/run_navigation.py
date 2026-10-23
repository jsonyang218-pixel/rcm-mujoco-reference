#!/usr/bin/env python3
"""Run complete target navigation in MuJoCo, with preflight and CSV logs."""
from __future__ import annotations

import argparse
import json
import queue
import time
from datetime import datetime
from pathlib import Path

import mujoco

from rcm_reference import HERE, Reference, render
from rcm_navigation import (NavigationRecorder, NavigationRejected, build_plan,
                            execute_plan, load_targets, solve_target, task_si_units, write_json)


def interactive(reference, plan, period_s, camera):
    import mujoco.viewer

    recorder = NavigationRecorder(reference, plan)
    times = plan.sample_times(period_s)
    events = queue.SimpleQueue()
    index, elapsed, paused = 0, 0., False
    camera_names = ("overview", "side", "needle_close", "eye_internal")
    camera_index = 0 if camera == "free" else camera_names.index(camera)
    last = time.perf_counter()
    print("P: pause/resume | V: fixed camera views | F: free/fixed camera")
    print("Free camera: left-drag rotate, right-drag pan, mouse wheel zoom.")
    print("Close the window to save logs and exit.")
    print("Execution: kinematic navigation. C returns to its axial zero before orientation reset.")
    with mujoco.viewer.launch_passive(reference.model, reference.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            if camera == "free":
                mujoco.mjv_defaultFreeCamera(reference.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    reference.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            viewer.opt.sitegroup[:] = 0
            viewer.opt.jointgroup[3] = 0
        while viewer.is_running():
            now = time.perf_counter()
            delta, last = min(now-last, .05), now
            with viewer.lock():
                while not events.empty():
                    key = events.get()
                    if key == 80:  # P
                        paused = not paused
                    elif key == 86:  # V
                        if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                            camera_index = (camera_index+1) % len(camera_names)
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = mujoco.mj_name2id(
                            reference.model, mujoco.mjtObj.mjOBJ_CAMERA,
                            camera_names[camera_index])
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
                if not paused:
                    elapsed = min(plan.duration_s, elapsed+delta)
                while index < len(times) and times[index] <= elapsed+1e-12:
                    recorder.step(times[index])
                    index += 1
                viewer.opt.geomgroup[:3] = 1
            viewer.sync()
            time.sleep(.01)
    return recorder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--target-world", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"))
    source.add_argument("--target-robot", type=float, nargs=3, metavar=("X_MM", "Y_MM", "Z_MM"))
    source.add_argument("--targets", type=Path, help="JSON batch of named targets, coordinates in mm")
    parser.add_argument("--solve-only", action="store_true", help="Compute feasibility/counts without changing pose")
    parser.add_argument("--no-viewer", action="store_true", help="Run the complete cycle offline")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE/"runs"/"历史记录"/"04_sessions"/"navigation"/datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--sample-period", type=float, help="CSV sample period in seconds")
    parser.add_argument("--camera", choices=("free", "overview", "side", "needle_close", "eye_internal"), default="free")
    parser.add_argument("--render", type=Path, help="Save a MuJoCo PNG of the last reached target pose")
    args = parser.parse_args(argv)
    try:
        reference = Reference(json.loads(args.config.read_text(encoding="utf-8")))
        if args.targets is not None:
            targets = load_targets(args.targets)
        elif args.target_world is not None:
            targets = [{"id": "input_target", "frame": "world", "point_mm": args.target_world}]
        elif args.target_robot is not None:
            targets = [{"id": "input_target", "frame": "robot", "point_mm": args.target_robot}]
        else:
            targets = [{"id": "default_target", "frame": "robot",
                        "point_mm": reference.config["reference_scene"]["target_robot_mm"]}]
        if args.solve_only:
            solutions = [solve_target(reference, t["point_mm"], t.get("frame", "world"), t["id"]) for t in targets]
            write_json(args.output_dir/"target_solutions.json", solutions)
            print(json.dumps(solutions, ensure_ascii=False, indent=2, allow_nan=False))
            return 0 if all(s["reachable"] for s in solutions) else 2
        try:
            plan = build_plan(reference, targets)
        except NavigationRejected as rejected:
            write_json(args.output_dir/"target_solutions.json", rejected.solutions)
            write_json(args.output_dir/"navigation_report.json", {
                "result": "REJECTED", "execution_mode": "not_started",
                "pose_changed": False, "target_solutions": rejected.solutions,
            })
            print(json.dumps({"result": "REJECTED", "target_solutions": rejected.solutions},
                             ensure_ascii=False, indent=2, allow_nan=False))
            return 2
        # Validate sampling before opening a viewer or modifying the pose.
        plan.sample_times(args.sample_period)
        if args.no_viewer:
            recorder = execute_plan(reference, plan, args.sample_period)
        else:
            recorder = interactive(reference, plan, args.sample_period, args.camera)
        recorder.write(args.output_dir)
        report = recorder.report()
        if args.render is not None and report["completed"]:
            saved = reference.current_task().copy()
            last = plan.solutions[-1]
            reference.set_target_world(last["requested_world_mm"])
            reference.set_task(task_si_units(last["command_task_A_deg_B_deg_C_mm"]))
            try:
                render(reference, args.render, "overview" if args.camera == "free" else args.camera)
            finally:
                reference.set_task(saved)
        print(json.dumps({
            "result": report["result"], "target_count": report["target_count"],
            "reached_count": report["reached_count"], "sample_count": report["sample_count"],
            "max_reached_tip_error_um": report["max_reached_tip_error_um"],
            "returned_to_zero": report["returned_to_zero"], "output_dir": str(args.output_dir.resolve()),
            "execution_mode": report["execution_mode"],
        }, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["result"] == "PASS" else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
