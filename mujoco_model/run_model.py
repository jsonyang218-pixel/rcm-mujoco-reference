#!/usr/bin/env python3
"""View and evaluate the current scanned arm offline, with closed joint chains.

This entry never imports rospy, sources a network environment, or opens a motor
device. Historical navigation/configuration options belong to
run_legacy_reference.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DEMO_PERIOD_S = 8.0
DEMO_AMPLITUDES = (8.0, 5.0, 2.0)  # degrees, degrees, millimetres


class Parser(argparse.ArgumentParser):
    def error(self, message):
        super().error(message + "\nHistorical --config/--auto-reach and navigation options: use run_legacy_reference.py.")


def pose_from_scene(scene):
    """Return independent A/B-relative degrees and C axial millimetres."""
    return [scene.diagnostics["A_deg"], scene.diagnostics["B_relative_deg"],
            scene.diagnostics["C_axial_mm"]]


def encoders_from_pose(scene, pose):
    if len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose):
        raise ValueError("A/B/C pose must contain three finite numbers")
    a, b, c = map(float, pose)
    independent = (math.radians(a), math.radians(b), c * .001)
    for axis, value in zip("ABC", independent):
        low, high = scene.joint_bounds[axis]
        if not low - 1e-12 <= value <= high + 1e-12:
            raise ValueError(f"{axis} pose exceeds the source-derived software limits")
    kin = scene.kin
    return (round(a * 200000.0 / 360.0),
            round(kin.ENCODER_COUNTS_PER_MM * kin.b_distance_from_alpha(scene.alpha0 + b)),
            round(kin.ENCODER_COUNTS_PER_MM * (c + 14.0)), 0)


def synchronize(scene, robot):
    result = scene.sync(robot)
    # Display actuator targets consistently; this remains a kinematic viewer,
    # and intentionally never invokes mj_step or a hardware transport.
    import mujoco
    for axis in "ABC":
        actuator = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, axis + "_servo")
        joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, axis)
        scene.data.ctrl[actuator] = scene.data.qpos[scene.model.jnt_qposadr[joint]]
    return result


def demo_pose(base, name, seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Demo sample time must be finite and nonnegative")
    scalar = math.sin(2.0 * math.pi * seconds / DEMO_PERIOD_S)
    return [value + (amplitude * scalar if axis.lower() in name else 0.0)
            for value, amplitude, axis in zip(base, DEMO_AMPLITUDES, "ABC")]


def validate_demo(scene, base, name):
    # Check both sinusoid extrema before opening the viewer or changing state.
    for scalar in (-1, 1):
        values = [value + (amplitude * scalar if axis.lower() in name else 0.0)
                  for value, amplitude, axis in zip(base, DEMO_AMPLITUDES, "ABC")]
        encoders_from_pose(scene, values)


def export_scene(scene, destination):
    """Export the home/configuration model with destination-relative assets."""
    destination = Path(destination).expanduser().resolve()
    try:
        meshdir = os.path.relpath(HERE / "assets", destination.parent)
    except ValueError:
        # Windows cannot construct a relative path across drive letters.  An
        # absolute mesh directory keeps the export loadable and makes the
        # failure mode explicit; same-volume exports remain portable.
        meshdir = str((HERE / "assets").resolve())
    root = ET.fromstring(scene.xml)
    root.find("compiler").set("meshdir", Path(meshdir).as_posix())
    ET.indent(root, space="  ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="unicode")


def render_scene(scene, destination, camera):
    import mujoco
    from PIL import Image
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(scene.model, width=1100, height=850) as renderer:
        renderer.update_scene(scene.data, camera="overview" if camera == "free" else camera)
        Image.fromarray(renderer.render().copy()).save(path)


def verify_model(report_path):
    """Validate current assets and kinematics without historical model snapshots."""
    import subprocess
    import tempfile
    checks = []
    # Generated reports go to a temporary directory. Each check builds the
    # current model; historical model implementations are not loaded.
    with tempfile.TemporaryDirectory(prefix="scan-model-verify-") as temp:
        for script, arguments in (
                (HERE / "validate_frames.py", ["--output-dir", temp]),
                (HERE / "validate_geometry.py", ["--output-dir", temp]),
                (HERE.parent / "verify_enclosed_shells.py", []),
                (HERE.parent / "verify_v9_front_tool.py", [])):
            command = [sys.executable, str(script), *arguments]
            completed = subprocess.run(command, cwd=HERE.parent, text=True,
                                       capture_output=True, timeout=180)
            item = {"script": script.name, "returncode": completed.returncode}
            try:
                item["result"] = json.loads(completed.stdout)
            except json.JSONDecodeError:
                item["result"] = {}
            result = item["result"]
            if completed.returncode or result.get("status", result.get("result")) != "PASS":
                item["status"] = "FAIL"
                item["stderr"] = completed.stderr[-4000:]
                item["stdout"] = completed.stdout[-2000:]
            else:
                item["status"] = "PASS"
            checks.append(item)
    summary = {"status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
               "model_version": 9,
               "scope": "Current scanned-model frames, source ownership, closed skins and v9 tool; no historical snapshot comparison or hardware calibration",
               "checks": checks}
    content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if report_path is not None:
        path = Path(report_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(content, end="")
    return 0 if summary["status"] == "PASS" else 1


def interactive(scene, demo, base, camera):
    import queue
    import time
    import mujoco
    import mujoco.viewer
    events = queue.SimpleQueue()
    addresses = [int(scene.model.jnt_qposadr[mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_JOINT, axis)]) for axis in "ABC"]
    previous = tuple(scene.last_robot)
    elapsed = 0.0
    paused = False
    last = time.monotonic()
    print("Offline kinematic viewer: edit Joint sliders A/B/C; dependent edits are closed automatically.")
    print("H / Home / R: home and stop demo | P: pause/resume demo. No hardware I/O.")
    with mujoco.viewer.launch_passive(scene.model, scene.data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=True) as viewer:
        with viewer.lock():
            if camera == "free":
                mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            viewer.opt.sitegroup[:] = 0
        while viewer.is_running():
            now = time.monotonic()
            delta, last = min(now - last, .1), now
            with viewer.lock():
                while not events.empty():
                    key = events.get()
                    if key in (72, 268, 82):
                        from model_sync import HOME
                        previous = tuple(HOME)
                        synchronize(scene, previous)
                        demo, elapsed = None, 0.0
                    elif key == 80:
                        paused = not paused
                try:
                    if demo is not None:
                        if not paused:
                            elapsed += delta
                        robot = encoders_from_pose(scene, demo_pose(base, demo, elapsed))
                    else:
                        q = [float(scene.data.qpos[address]) for address in addresses]
                        robot = encoders_from_pose(scene, (math.degrees(q[0]), math.degrees(q[1]), q[2] * 1000.0))
                    synchronize(scene, robot)
                    previous = tuple(robot)
                except (ValueError, OverflowError) as error:
                    synchronize(scene, previous)
                    print("Joint edit rejected:", error)
                scene.data.time += delta
            viewer.sync()
            time.sleep(.01)


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true", help="Evaluate pose offline; no display required")
    parser.add_argument("--profile", choices=("new", "legacy"), default="new")
    parser.add_argument("--export", type=Path, help="Export the selected profile/model definition, at its home pose")
    parser.add_argument("--render", type=Path, help="Render the evaluated pose to a PNG; implies no viewer")
    parser.add_argument("--camera", choices=("free", "overview"), default="overview")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--pose", type=float, nargs=3, metavar=("A_DEG", "B_RELATIVE_DEG", "C_MM"))
    initial.add_argument("--encoders", type=int, nargs=3, metavar=("EA", "EB", "EC"), help="Integer robot feedback counts, not handles")
    parser.add_argument("--demo", choices=("a", "b", "c", "abc"),
                        help="Loop an eight-second in-phase sinusoid about the starting pose: A +/-8deg, B +/-5deg, C +/-2mm")
    parser.add_argument("--sample-time", type=float,
                        help="Sample a demo offline at this many seconds; default 2s (positive peak)")
    parser.add_argument("--verify", action="store_true", help="Verify current scanned-model frames and integral silver cover")
    parser.add_argument("--report", type=Path, help="Save verification JSON; requires --verify")
    args = parser.parse_args(argv)
    if args.report is not None and not args.verify:
        parser.error("--report requires --verify")
    if args.verify and any(value is not None for value in
                           (args.pose, args.encoders, args.demo, args.sample_time, args.export, args.render)):
        parser.error("--verify checks the full model; use it separately from pose/render/export options")
    if args.sample_time is not None and (args.demo is None or not (args.no_viewer or args.render)):
        parser.error("--sample-time requires --demo and --no-viewer or --render")
    if args.render is not None:
        os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "glfw")
        os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(HERE / ".cache"))
    try:
        if args.verify:
            return verify_model(args.report)
        from model_sync import DriverModel, HOME
        scene = DriverModel(needle_profile=args.profile)
        initial_robot = tuple(args.encoders) + (0,) if args.encoders is not None else HOME
        if args.pose is not None:
            initial_robot = encoders_from_pose(scene, args.pose)
        synchronize(scene, initial_robot)
        base = pose_from_scene(scene)
        if args.demo:
            validate_demo(scene, base, args.demo)
        if args.export is not None:
            export_scene(scene, args.export)
        if args.no_viewer or args.render is not None:
            if args.demo:
                seconds = 2.0 if args.sample_time is None else args.sample_time
                synchronize(scene, encoders_from_pose(scene, demo_pose(base, args.demo, seconds)))
                scene.data.time = seconds
            if args.render is not None:
                render_scene(scene, args.render, args.camera)
            print(json.dumps(scene.diagnostics, ensure_ascii=False, indent=2))
            return 0
        interactive(scene, args.demo, base, args.camera)
        return 0
    except (ValueError, KeyError, OSError, OverflowError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
