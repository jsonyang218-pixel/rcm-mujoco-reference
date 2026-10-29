"""Regression checks for the real-data calibration pipeline.

The measurements below are synthetic and exist only to test fitting, schema
validation, report generation, and the explicit configuration-application
guard.  They are never written into the repository's model parameters.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "mujoco_model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from calibrate_model import (  # noqa: E402
    CalibrationInputError,
    apply_affine_config,
    build_report,
    read_input,
)


def nominal_b(count: float) -> float:
    counts_per_mm = 4.0 * 1024.0 * 15.5817175
    distance = count / counts_per_mm
    ad = 98.48 - distance
    ac = math.hypot(ad, 16.0)
    gamma = math.degrees(math.atan2(16.0, ad))
    beta = math.degrees(math.acos((60.0**2 + ac**2 - 52.0**2) / (2.0 * 60.0 * ac)))
    return gamma + beta


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="calibration-pipeline-") as directory:
        work = Path(directory)
        csv_path = work / "samples.csv"
        rows = []
        for count in (-1000, 0, 1000, 2000):
            rows.append(["A", count, count / (2000.0 / 3.6), "deg", 1, "static", ""])
        for count in (893518, 925429, 957340, 989251):
            rows.append(["C", count, (count - 893518) / 63822.71488, "mm", 1, "static", ""])
        for count in (1466018, 1500000, 1540000, 1580000):
            correction = 0.02 + 1e-7 * (count - 1500000) ** 2 / 1e6
            rows.append(["B", count, nominal_b(count) + correction, "deg", 1, "forward", ""])
        for count in (1466018, 1500000, 1540000):
            rows.append(["SILVER", count, (count - 1466018) / 63822.71488 * 0.8, "mm", 1, "forward", ""])
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["axis", "encoder_count", "measured_value", "measured_unit", "repeat", "direction", "timestamp_s"])
            writer.writerows(rows)

        samples, pose_rows, metadata = read_input(csv_path)
        # A known rigid transform, expressed in millimetres.
        angle = math.radians(20.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                             [math.sin(angle), math.cos(angle), 0.0],
                             [0.0, 0.0, 1.0]])
        translation = np.array([12.0, -8.0, 4.0])
        pose_rows = []
        for point in ([0, 0, 0], [10, 0, 0], [0, 12, 0], [0, 0, 8]):
            robot = np.asarray(point, dtype=float)
            world = rotation @ robot + translation
            pose_rows.append({"robot_xyz_mm": robot.tolist(), "world_xyz_mm": world.tolist()})
        report = build_report(csv_path, samples, pose_rows, {"source": "synthetic regression only"})
        assert report["status"] == "PASS"
        assert set(report["axis_reports"]) == {"A", "B", "C", "SILVER"}
        assert abs(report["axis_reports"]["A"]["affine_parameters"]["counts_per_unit"] - 555.5555555556) < 1e-8
        assert abs(report["axis_reports"]["C"]["affine_parameters"]["counts_per_unit"] - 63822.71488) < 1e-5
        assert report["axis_reports"]["B"]["fit"]["degree"] == 2
        assert report["axis_reports"]["SILVER"]["fit"]["degree"] == 1
        assert report["pose_report"]["rmse_mm"] < 1e-10

        # Applying is explicit and must create a backup; the source config is untouched.
        source_config = HERE / "config.json"
        temporary_config = work / "config.json"
        source_before = source_config.read_bytes()
        temporary_config.write_bytes(source_before)
        before = temporary_config.read_bytes()
        applied = apply_affine_config(report, temporary_config)
        assert applied == temporary_config
        assert temporary_config.read_bytes() != before
        assert temporary_config.with_suffix(".json.before_calibration").is_file()
        updated = json.loads(temporary_config.read_text(encoding="utf-8"))
        assert abs(updated["encoders"]["a_counts_per_degree"] - 555.5555555556) < 1e-8
        assert abs(updated["encoders"]["bc_counts_per_mm"] - 63822.71488) < 1e-5
        assert updated["encoder_reference"]["robot_counts_ABC"] == [0, 1466018, 893518]
        assert updated["encoders"]["origin"] == [0.0, 0.0, 0.0]
        assert source_config.read_bytes() == source_before

        bad = work / "bad.csv"
        bad.write_text("axis,encoder_count,measured_value\nQ,1,2\n", encoding="utf-8")
        try:
            read_input(bad)
        except CalibrationInputError:
            pass
        else:
            raise AssertionError("invalid axis was accepted")
    print(json.dumps({
        "status": "PASS",
        "axes": ["A", "B", "C", "SILVER"],
        "rigid_pose_fit": "PASS",
        "explicit_apply_and_backup": "PASS",
        "invalid_schema_rejected": "PASS",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
