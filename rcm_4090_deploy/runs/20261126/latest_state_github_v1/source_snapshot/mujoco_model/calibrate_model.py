"""Fit physical calibration data for the v9 MuJoCo model.

The repository intentionally keeps the photo/video-derived model parameters
unchanged until real measurements are supplied.  This command turns encoder
and metrology data into a reviewable JSON report.  It can optionally apply the
two safe affine encoder fits (A and C) to a copy of ``config.json``; nonlinear
B-axis and installation fits always remain review-only.

Input formats
-------------
CSV files contain one encoder measurement per row::

    axis,encoder_count,measured_value,measured_unit,repeat,direction
    A,0,0,deg,1,static
    C,893518,0,mm,1,static
    B,1466018,54.236,deg,1,static
    SILVER,1466018,0,mm,1,static

JSON files may contain the same rows in ``axis_samples`` and optional rigid
pose pairs in ``pose_samples``::

    {"axis_samples": [...],
     "pose_samples": [{"robot_xyz_mm": [...], "world_xyz_mm": [...]}]}

No values in the example/template files are used as calibration data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ALLOWED_AXES = ("A", "B", "C", "SILVER")
EXPECTED_UNITS = {"A": "deg", "B": "deg", "C": "mm", "SILVER": "mm"}
REQUIRED_COLUMNS = ("axis", "encoder_count", "measured_value")


class CalibrationInputError(ValueError):
    """Raised when a calibration file is incomplete or physically invalid."""


@dataclass(frozen=True)
class AxisSample:
    axis: str
    encoder_count: float
    measured_value: float
    measured_unit: str
    repeat: int | None = None
    direction: str = "unknown"
    timestamp_s: float | None = None


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise CalibrationInputError(f"{name} must be finite: {value!r}")
    return result


def _optional_finite(value: Any, name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _finite(value, name)


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationInputError(f"{name} must be an integer: {value!r}") from exc
    return result


def _sample_from_mapping(row: Mapping[str, Any], index: int) -> AxisSample:
    missing = [key for key in REQUIRED_COLUMNS if key not in row]
    if missing:
        raise CalibrationInputError(
            f"axis sample {index} is missing columns: {', '.join(missing)}"
        )
    axis = str(row["axis"]).strip().upper()
    if axis == "SILVER_B" or axis == "SILVER_COVER":
        axis = "SILVER"
    if axis not in ALLOWED_AXES:
        raise CalibrationInputError(
            f"axis sample {index} has unsupported axis {axis!r}; "
            f"expected one of {', '.join(ALLOWED_AXES)}"
        )
    unit = str(row.get("measured_unit", EXPECTED_UNITS[axis])).strip().lower()
    if unit in ("degree", "degrees", "°"):
        unit = "deg"
    if unit not in ("deg", "mm"):
        raise CalibrationInputError(f"axis sample {index} has invalid unit {unit!r}")
    if unit != EXPECTED_UNITS[axis]:
        raise CalibrationInputError(
            f"axis sample {index} uses {unit!r} for {axis}; "
            f"expected {EXPECTED_UNITS[axis]!r}"
        )
    direction = str(row.get("direction", "unknown")).strip().lower() or "unknown"
    if direction not in {"unknown", "static", "forward", "backward", "positive", "negative"}:
        raise CalibrationInputError(
            f"axis sample {index} has invalid direction {direction!r}"
        )
    return AxisSample(
        axis=axis,
        encoder_count=_finite(row["encoder_count"], f"axis sample {index} encoder_count"),
        measured_value=_finite(row["measured_value"], f"axis sample {index} measured_value"),
        measured_unit=unit,
        repeat=_optional_int(row.get("repeat"), f"axis sample {index} repeat"),
        direction=direction,
        timestamp_s=_optional_finite(row.get("timestamp_s"), f"axis sample {index} timestamp_s"),
    )


def _read_csv(path: Path) -> tuple[list[AxisSample], list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CalibrationInputError("CSV input has no header")
        missing = [key for key in REQUIRED_COLUMNS if key not in reader.fieldnames]
        if missing:
            raise CalibrationInputError(
                f"CSV input is missing columns: {', '.join(missing)}"
            )
        samples = [_sample_from_mapping(row, i + 2) for i, row in enumerate(reader)]
    if not samples:
        raise CalibrationInputError("CSV input contains no axis samples")
    return samples, [], {}


def _read_json(path: Path) -> tuple[list[AxisSample], list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationInputError(f"invalid JSON input: {exc}") from exc
    if isinstance(payload, list):
        axis_rows = payload
        pose_rows: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        axis_rows = payload.get("axis_samples", payload.get("samples", []))
        pose_rows = payload.get("pose_samples", [])
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise CalibrationInputError("metadata must be an object")
    else:
        raise CalibrationInputError("JSON input must be an object or an array")
    if not isinstance(axis_rows, list) or not axis_rows:
        raise CalibrationInputError("JSON input contains no axis_samples")
    samples = []
    for i, row in enumerate(axis_rows):
        if not isinstance(row, Mapping):
            raise CalibrationInputError(f"axis sample {i + 1} must be an object")
        samples.append(_sample_from_mapping(row, i + 1))
    if not isinstance(pose_rows, list):
        raise CalibrationInputError("pose_samples must be an array")
    return samples, pose_rows, metadata


def read_input(path: str | os.PathLike[str]) -> tuple[list[AxisSample], list[dict[str, Any]], dict[str, Any]]:
    """Read and validate a calibration CSV or JSON input file."""
    input_path = Path(path)
    if not input_path.is_file():
        raise CalibrationInputError(f"calibration input does not exist: {input_path}")
    if input_path.suffix.lower() == ".csv":
        return _read_csv(input_path)
    if input_path.suffix.lower() == ".json":
        return _read_json(input_path)
    raise CalibrationInputError("calibration input must use .csv or .json")


def _normalized_fit(x: Sequence[float], y: Sequence[float], degree: int) -> dict[str, Any]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.size != y_array.size or x_array.size < degree + 1:
        raise CalibrationInputError(
            f"need at least {degree + 1} samples for degree-{degree} fit; "
            f"received {x_array.size}"
        )
    center = float(np.mean(x_array))
    scale = float(np.max(np.abs(x_array - center)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise CalibrationInputError("fit samples must contain at least two encoder values")
    z = (x_array - center) / scale
    design = np.column_stack([z**power for power in range(degree + 1)])
    coefficients, *_ = np.linalg.lstsq(design, y_array, rcond=None)
    prediction = design @ coefficients
    residual = prediction - y_array
    return {
        "degree": int(degree),
        "center_encoder_count": center,
        "scale_encoder_count": scale,
        "coefficients_ascending": [float(value) for value in coefficients],
        "sample_count": int(x_array.size),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "residuals": [float(value) for value in residual],
    }


def predict_fit(fit: Mapping[str, Any], encoder_count: float | Sequence[float]) -> np.ndarray | float:
    """Evaluate a normalized polynomial fit returned by :func:`_normalized_fit`."""
    values = np.asarray(encoder_count, dtype=float)
    center = float(fit["center_encoder_count"])
    scale = float(fit["scale_encoder_count"])
    coefficients = np.asarray(fit["coefficients_ascending"], dtype=float)
    z = (values - center) / scale
    result = np.zeros_like(z, dtype=float)
    for coefficient in reversed(coefficients):
        result = result * z + coefficient
    if np.ndim(encoder_count) == 0:
        return float(result)
    return result


def _affine_parameters(fit: Mapping[str, Any]) -> dict[str, float]:
    if int(fit["degree"]) != 1:
        raise ValueError("affine parameters require a degree-1 fit")
    coefficients = fit["coefficients_ascending"]
    scale = float(fit["scale_encoder_count"])
    center = float(fit["center_encoder_count"])
    slope = float(coefficients[1]) / scale
    intercept = float(coefficients[0]) - slope * center
    if abs(slope) <= 1e-15:
        raise CalibrationInputError("measured value does not change with encoder count")
    direction = 1.0 if slope > 0.0 else -1.0
    counts_per_unit = 1.0 / abs(slope)
    origin_count = -intercept / slope
    return {
        "slope_unit_per_count": slope,
        "intercept_unit": intercept,
        "counts_per_unit": counts_per_unit,
        "direction": direction,
        "zero_encoder_count": origin_count,
    }


def _nominal_b_angle(encoder_count: float) -> float:
    """Evaluate the unchanged c2a3ffc linkage inverse for comparison."""
    counts_per_mm = 4.0 * 1024.0 * 15.5817175
    distance = float(encoder_count) / counts_per_mm
    ad = 98.48 - distance
    ac = math.hypot(ad, 16.0)
    if ac <= 1e-12:
        raise CalibrationInputError("B encoder sample is at an invalid linkage singularity")
    gamma = math.degrees(math.atan2(16.0, ad))
    beta_cos = (60.0**2 + ac**2 - 52.0**2) / (2.0 * 60.0 * ac)
    if beta_cos < -1.000001 or beta_cos > 1.000001:
        raise CalibrationInputError("B encoder sample is outside linkage geometry")
    return gamma + math.degrees(math.acos(max(-1.0, min(1.0, beta_cos))))


def _axis_report(axis: str, samples: Sequence[AxisSample]) -> dict[str, Any]:
    if len(samples) < 2:
        raise CalibrationInputError(f"axis {axis} needs at least two samples")
    counts = [sample.encoder_count for sample in samples]
    measured = [sample.measured_value for sample in samples]
    degree = 2 if axis == "B" and len(samples) >= 3 else 1
    fit = _normalized_fit(counts, measured, degree)
    result: dict[str, Any] = {
        "axis": axis,
        "unit": EXPECTED_UNITS[axis],
        "sample_count": len(samples),
        "encoder_count_min": float(min(counts)),
        "encoder_count_max": float(max(counts)),
        "fit": fit,
        "direction_counts": {
            direction: sum(1 for sample in samples if sample.direction == direction)
            for direction in ("forward", "backward", "static", "unknown")
        },
    }
    if degree == 1:
        result["affine_parameters"] = _affine_parameters(fit)
    if axis == "B":
        nominal = np.asarray([_nominal_b_angle(value) for value in counts], dtype=float)
        measured_array = np.asarray(measured, dtype=float)
        correction = measured_array - nominal
        result["nominal_linkage"] = {
            "angle_deg": [float(value) for value in nominal],
            "correction_mean_deg": float(np.mean(correction)),
            "correction_rmse_deg": float(np.sqrt(np.mean(correction**2))),
            "correction_max_abs_deg": float(np.max(np.abs(correction))),
            "correction_values_deg": [float(value) for value in correction],
            "applyable_to_current_driver": False,
            "reason": "B uses a nonlinear linkage inverse; review the measured correction before changing kinematics.py",
        }
    if axis == "SILVER":
        result["applyable_to_current_driver"] = False
        result["reason"] = "silver travel is currently a provisional visual follower; confirm transmission and direction first"
    return result


def _pose_report(pose_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not pose_rows:
        return None
    robot_points: list[list[float]] = []
    world_points: list[list[float]] = []
    for index, row in enumerate(pose_rows, 1):
        if not isinstance(row, Mapping):
            raise CalibrationInputError(f"pose sample {index} must be an object")
        if "robot_xyz_mm" not in row or "world_xyz_mm" not in row:
            raise CalibrationInputError(
                f"pose sample {index} needs robot_xyz_mm and world_xyz_mm"
            )
        robot = [_finite(value, f"pose sample {index} robot_xyz_mm") for value in row["robot_xyz_mm"]]
        world = [_finite(value, f"pose sample {index} world_xyz_mm") for value in row["world_xyz_mm"]]
        if len(robot) != 3 or len(world) != 3:
            raise CalibrationInputError(f"pose sample {index} coordinates must have length 3")
        robot_points.append(robot)
        world_points.append(world)
    if len(robot_points) < 3:
        raise CalibrationInputError("pose_samples needs at least three corresponding points")
    source = np.asarray(robot_points, dtype=float)
    target = np.asarray(world_points, dtype=float)
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise CalibrationInputError("pose_samples must span at least two spatial dimensions")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _singular_values, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = (rotation @ source.T).T + translation
    residual = predicted - target
    norms = np.linalg.norm(residual, axis=1)
    return {
        "sample_count": len(pose_rows),
        "rotation_robot_to_world": rotation.tolist(),
        "translation_world_mm": translation.tolist(),
        "rmse_mm": float(np.sqrt(np.mean(norms**2))),
        "max_abs_error_mm": float(np.max(norms)),
        "residual_norms_mm": [float(value) for value in norms],
        "applyable_to_current_driver": False,
        "reason": "review RCM and installation convention before changing installation_config.json",
    }


def _repeatability(samples: Sequence[AxisSample]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for sample in samples:
        if sample.repeat is not None:
            grouped.setdefault((sample.axis, sample.repeat), []).append(sample.encoder_count)
    groups = []
    for (axis, repeat), counts in sorted(grouped.items()):
        if len(counts) >= 2:
            groups.append({
                "axis": axis,
                "repeat": repeat,
                "sample_count": len(counts),
                "mean_encoder_count": float(np.mean(counts)),
                "std_encoder_count": float(np.std(counts, ddof=1)),
                "span_encoder_count": float(max(counts) - min(counts)),
            })
    return {"groups": groups, "group_count": len(groups)}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    input_path: str | os.PathLike[str],
    samples: Sequence[AxisSample],
    pose_rows: Sequence[Mapping[str, Any]] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit all supplied measurements and return a serializable report."""
    grouped = {axis: [sample for sample in samples if sample.axis == axis] for axis in ALLOWED_AXES}
    axes = {}
    for axis, axis_samples in grouped.items():
        if axis_samples:
            axes[axis] = _axis_report(axis, axis_samples)
    if not axes:
        raise CalibrationInputError("input contains no supported axis samples")
    return {
        "schema_version": 1,
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(Path(input_path).resolve()),
        "input_sha256": sha256_file(input_path),
        "metadata": dict(metadata or {}),
        "sample_count": len(samples),
        "axis_reports": axes,
        "repeatability": _repeatability(samples),
        "pose_report": _pose_report(pose_rows),
        "application_policy": {
            "default": "review_only",
            "safe_affine_axes": ["A", "C"],
            "manual_review_axes": ["B", "SILVER"],
            "installation_and_RCM": "manual_review",
            "note": "A report does not change model or configuration files unless --apply-config is explicitly supplied.",
        },
    }


def apply_affine_config(report: Mapping[str, Any], config_path: str | os.PathLike[str]) -> Path:
    """Apply only reviewed A/C affine fits to a JSON config, with a backup."""
    path = Path(config_path)
    if not path.is_file():
        raise CalibrationInputError(f"config does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    encoder_config = config.setdefault("encoders", {})
    axis_reports = report.get("axis_reports", {})
    applied = []
    for axis, count_key, origin_index, direction_index in (
        ("A", "a_counts_per_degree", 0, 0),
        ("C", "bc_counts_per_mm", 2, 2),
    ):
        fit = axis_reports.get(axis, {}).get("affine_parameters")
        if not fit:
            continue
        encoder_config[count_key] = float(fit["counts_per_unit"])
        origin = list(encoder_config.get("origin", [0.0, 0.0, 0.0]))
        direction = list(encoder_config.get("direction", [1.0, 1.0, 1.0]))
        if len(origin) != 3 or len(direction) != 3:
            raise CalibrationInputError("config encoders.origin/direction must have length 3")
        direction[direction_index] = float(fit["direction"])
        encoder_config["direction"] = direction
        reference = config.get("encoder_reference", {})
        if reference.get("enabled") and isinstance(reference.get("robot_counts_ABC"), list):
            counts = list(reference["robot_counts_ABC"])
            if len(counts) != 3:
                raise CalibrationInputError("encoder_reference.robot_counts_ABC must have length 3")
            counts[origin_index] = int(round(fit["zero_encoder_count"]))
            reference["robot_counts_ABC"] = counts
            config["encoder_reference"] = reference
        else:
            # Legacy configurations without an explicit reference pose use the
            # encoder origin directly.
            origin[origin_index] = float(fit["zero_encoder_count"])
            encoder_config["origin"] = origin
        applied.append(axis)
    if not applied:
        raise CalibrationInputError("report contains no A/C affine fit to apply")
    backup = path.with_suffix(path.suffix + ".before_calibration")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_template(path: str | os.PathLike[str]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "axis,encoder_count,measured_value,measured_unit,repeat,direction,timestamp_s\n",
        encoding="utf-8",
    )
    return destination


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="real calibration CSV or JSON")
    parser.add_argument(
        "--output", type=Path, default=HERE / "outputs" / "calibration_report.json",
        help="JSON report destination (default: mujoco_model/outputs/calibration_report.json)",
    )
    parser.add_argument(
        "--write-template", type=Path,
        help="write a header-only CSV template and exit",
    )
    parser.add_argument(
        "--apply-config", action="store_true",
        help="apply reviewed A/C affine fits to --config and create a backup",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config.json",
        help="JSON config used with --apply-config (default: root config.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        if args.write_template:
            print(write_template(args.write_template))
            return 0
        if not args.input:
            raise CalibrationInputError("--input is required unless --write-template is used")
        samples, pose_rows, metadata = read_input(args.input)
        report = build_report(args.input, samples, pose_rows, metadata)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.apply_config:
            applied_path = apply_affine_config(report, args.config)
            report["applied_config"] = str(applied_path.resolve())
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "sample_count": report["sample_count"],
            "axes": sorted(report["axis_reports"]),
            "report": str(args.output.resolve()),
            "applied_config": report.get("applied_config"),
        }, ensure_ascii=False))
        return 0
    except (CalibrationInputError, OSError, json.JSONDecodeError) as exc:
        print(f"calibration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
