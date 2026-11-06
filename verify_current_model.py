"""Verify that the repository default entry is the current eye-injected v9 scene."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent

REQUIRED_GEOMS = (
    "scan_fixed_housing", "scan_parallel_cover", "scan_distal", "scan_C",
    "eye_reference", "retina_visual", "lens_visual", "lesion_marker",
    "needle_tip_marker",
)
REQUIRED_CAMERAS = ("overview", "eye_internal")


def run(cwd, *arguments):
    result = subprocess.run(
        [sys.executable, str(HERE / "run_reference.py"), *arguments],
        cwd=cwd, text=True, capture_output=True, timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"{arguments}: {result.stderr[-4000:] or result.stdout[-4000:]}"
        )
    return result.stdout


def check_scene(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    for name in ("A", "B", "C", "silver_B_linear"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name
    for name in REQUIRED_GEOMS:
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name
    for name in REQUIRED_CAMERAS:
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0, name
    return model


def main():
    assert not (HERE / "scene.xml").exists(), (
        "The obsolete standalone base-model export is still present"
    )
    check_scene(HERE / "v9_eye_scene.xml")

    with tempfile.TemporaryDirectory(prefix="current-v9-entry-", dir=HERE) as work:
        home = json.loads(run(work, "--no-viewer"))
        assert home["status"] == "PASS"
        assert home["model_backend"] == "RCMSceneV9"
        assert home["rcm_on_eye_surface"] and home["tip_inside_eye"]
        assert all(name in home["geoms"] for name in REQUIRED_GEOMS)
        assert all(name in home["cameras"] for name in REQUIRED_CAMERAS)

        verified = json.loads(run(work, "--verify"))
        assert verified["status"] == "PASS"
        assert verified["model_backend"] == "RCMSceneV9"

        posed = json.loads(run(work, "--no-viewer", "--pose", "4", "3", "1"))
        actual = posed["task_A_deg_B_deg_C_mm"]
        for value, expected in zip(actual, (4.0, 3.0, 1.0)):
            assert abs(value - expected) < 1e-9, (actual, expected)

        export = Path(work) / "nested" / "current_v9.xml"
        run(work, "--export", str(export))
        assert export.exists()
        check_scene(export)

        report = Path(work) / "nested" / "current_v9_report.json"
        run(work, "--verify", "--report", str(report))
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASS"

    assert not (HERE / "scene.xml").exists(), (
        "Evaluating v9 recreated the obsolete base-model export"
    )
    print(json.dumps({
        "status": "PASS",
        "default_model": "RCMSceneV9 eye-injected scanned arm",
        "default_entry": "run_reference.py",
        "portable_scene": "v9_eye_scene.xml",
        "obsolete_scene_xml_removed": True,
        "headless_entry_from_other_cwd": True,
        "exported_v9_scene_loads": True,
        "report_written": True,
        "live_desktop_GUI_tested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
