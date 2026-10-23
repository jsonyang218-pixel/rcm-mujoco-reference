"""Build the local OCT three-group experiment supplement.

This report builder is deliberately kept outside the navigation implementation. It
only reads recorded evidence and writes a self-contained, provenance-labelled
supplement.  The primary A/B/C evidence predates the OCT wrapper; local replays are
therefore reported separately and never replace the primary results.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MEETING = REPO / "reports" / "20261022" / "meeting"
EVIDENCE = MEETING / "evidence"
RUN_ROOT = REPO / "runs" / "oct_three_group_79afacc_20261022"
TARGET = REPO / "examples" / "three_waypoints.json"
CONFIG_DIR = EVIDENCE / "run_configuration"

PRIMARY = {
    "A_no_occlusion": EVIDENCE / "strict_three",
    "B_occlusion_stop": EVIDENCE / "strict_occlusion_stop",
    "C_occlusion_recovery": EVIDENCE / "strict_occlusion_recovery",
}
LOCAL = {
    "A_no_occlusion": RUN_ROOT / "A_no_occlusion",
    "B_occlusion_stop": RUN_ROOT / "B_occlusion_stop_local",
    "C_occlusion_recovery": RUN_ROOT / "C_occlusion_recovery_local",
}

CASE_LABELS = {
    "A_no_occlusion": "A: no occlusion",
    "B_occlusion_stop": "B: occlusion stop",
    "C_occlusion_recovery": "C: occlusion recovery",
}
PRIMARY_CASE_NAMES = {
    "A_no_occlusion": "strict_three",
    "B_occlusion_stop": "strict_occlusion_stop",
    "C_occlusion_recovery": "strict_occlusion_recovery",
}

FIG_DIR = OUT / "figures"
RAW_DIR = OUT / "raw_rgb"
EVIDENCE_OUT = OUT / "evidence"
INPUTS_OUT = OUT / "inputs"
SOURCE_OUT = OUT / "source_snapshot" / "local_current"

PHASE_COLORS = {
    "visual_align": "#0072B2",
    "visual_approach": "#D55E00",
    "visual_retract": "#009E73",
}
STATUS_COLORS = {"PASS": "#009E73", "FAIL": "#D55E00", "NOT_RUN": "#6C757D"}


def ensure_dirs() -> None:
    for path in (OUT, FIG_DIR, RAW_DIR, EVIDENCE_OUT, INPUTS_OUT, SOURCE_OUT):
        path.mkdir(parents=True, exist_ok=True)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def out_rel(path: Path) -> str:
    return path.resolve().relative_to(OUT.resolve()).as_posix()


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def run_command(args: list[str], cwd: Path = REPO) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.returncode, (result.stdout or result.stderr).strip()
    except OSError as exc:
        return 127, str(exc)


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def source_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "mujoco": package_version("mujoco"),
        "numpy": package_version("numpy"),
        "matplotlib": package_version("matplotlib"),
        "pillow": package_version("pillow"),
        "pip_freeze": run_command([sys.executable, "-m", "pip", "freeze"])[1].splitlines(),
    }


def recursive_diffs(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            key_path = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append(key_path)
            else:
                differences.extend(recursive_diffs(left[key], right[key], key_path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        return [] if left == right else [path]
    return [] if left == right else [path]


def config_hashes() -> dict[str, str]:
    return {
        "target_examples_three_waypoints.json": sha256_file(TARGET),
        "strict_config.json": sha256_file(CONFIG_DIR / "strict_config.json"),
        "stop_occlusion_config.json": sha256_file(CONFIG_DIR / "stop_occlusion_config.json"),
        "recovery_config.json": sha256_file(CONFIG_DIR / "recovery_config.json"),
    }


def load_endpoint_rows() -> list[dict[str, Any]]:
    rows = read_csv(MEETING / "endpoint_metrics.csv")
    selected: list[dict[str, Any]] = []
    for row in rows:
        case = row.get("case", "")
        mapped = next((key for key, value in PRIMARY_CASE_NAMES.items() if value == case), None)
        if mapped is None:
            continue
        selected.append(
            {
                "group": mapped,
                "primary_case": case,
                "episode_index": int(row["episode_index"]),
                "waypoint_id": row["waypoint_id"],
                "attempted": as_bool(row["attempted"]),
                "result": row["recorded_result"],
                "reason": row["reason"],
                "true_endpoint_error_um": as_float(row["true_endpoint_error_um"]),
                "terminal_stop_validated": as_bool(row["terminal_stop_validated"]),
                "settled_frames": int(row["settled_frames"]) if row["settled_frames"] else None,
                "final_command_velocity": row["final_command_velocity"],
                "independent_success": as_bool(row["independent_success_lt20um_and_3_zero_frames"]),
                "verification": row["verification"],
            }
        )
    return selected


def load_local_endpoint_rows() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group, path in LOCAL.items():
        for row in read_csv(path / "waypoint_results.csv"):
            result = row.get("result", "")
            attempted = as_bool(row.get("attempted"))
            # Local waypoint_results.csv predates the primary endpoint export and
            # has no attempted column. A NOT_RUN row is the explicit non-attempt.
            if attempted is None:
                attempted = result not in ("", "NOT_RUN")
            selected.append(
                {
                    "group": group,
                    "primary_case": "local_diagnostic_replay",
                    "episode_index": int(row.get("episode_index", 0)),
                    "waypoint_id": row["waypoint_id"],
                    "attempted": attempted,
                    "result": result,
                    "reason": row.get("reason"),
                    "true_endpoint_error_um": as_float(row.get("tip_error_um")),
                    "terminal_stop_validated": as_bool(row.get("terminal_stop_validated")),
                    "settled_frames": None,
                    "final_command_velocity": row.get("final_command_velocity", ""),
                    "independent_success": as_bool(row.get("success_with_stop")),
                    "verification": "LOCAL_REPLAY",
                }
            )
    return selected


def aggregate_group(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    selected = [row for row in rows if row["group"] == group]
    attempted = [row for row in selected if row["attempted"]]
    successful = [row for row in attempted if row["result"] == "PASS"]
    errors = [row["true_endpoint_error_um"] for row in successful if row["true_endpoint_error_um"] is not None]
    return {
        "group": group,
        "label": CASE_LABELS[group],
        "requested_targets": 3,
        "attempted_targets": len(attempted),
        "successful_targets": len(successful),
        "not_run_targets": 3 - len(attempted),
        "success_over_requested": len(successful) / 3.0,
        "endpoint_errors_um_successful_only": errors,
        "mean_endpoint_error_um_successful_only": float(np.mean(errors)) if len(errors) == 3 else None,
        "max_endpoint_error_um_successful_only": float(np.max(errors)) if len(errors) == 3 else None,
        "all_three_successful_for_average": len(errors) == 3,
        "results": selected,
    }


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                "time_s": float(row["time_s"]),
                "x": float(row["tip_world_x_mm"]),
                "y": float(row["tip_world_y_mm"]),
                "z": float(row["tip_world_z_mm"]),
                "error_um": float(row["tip_target_error_um"]),
            }
        )
    return parsed


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def target_world_points() -> dict[str, np.ndarray]:
    target_data = json_load(EVIDENCE / "strict_three" / "target_solutions.json")
    return {row["id"]: np.asarray(row["requested_world_mm"], dtype=float) for row in target_data}


def eye_center_world() -> np.ndarray:
    config = json_load(CONFIG_DIR / "strict_config.json")
    center = np.asarray(config["reference_scene"]["eye_center_robot_mm"], dtype=float)
    # The repository's installation transform maps robot (y,z) to world as below.
    return np.asarray([center[0], -np.cos(np.pi / 4) * center[1] + np.sin(np.pi / 4) * center[2],
                       -np.sin(np.pi / 4) * center[1] - np.cos(np.pi / 4) * center[2]])


def draw_eye_proxy(ax: Any) -> None:
    center = eye_center_world()
    config = json_load(CONFIG_DIR / "strict_config.json")
    radii = np.asarray(config["reference_scene"]["eye_radii_robot_mm"], dtype=float)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(-np.pi / 2, np.pi / 2, 24)
    x = center[0] + radii[0] * np.cos(v)[:, None] * np.cos(u)[None, :]
    y = center[1] + radii[1] * np.cos(v)[:, None] * np.sin(u)[None, :]
    z = center[2] + radii[2] * np.sin(v)[:, None]
    ax.plot_surface(x, y, z, color="#F4A3A8", alpha=0.14, linewidth=0, shade=False)


def make_figure_01() -> None:
    rows = load_trajectory(PRIMARY["A_no_occlusion"] / "visual_servo" / "trajectory.csv")
    points = target_world_points()
    start = np.asarray([rows[0]["x"], rows[0]["y"], rows[0]["z"]])
    rgb_path = MEETING / "rawframes" / "strict_three_ep000000_frame001288.png"
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    draw_eye_proxy(ax)
    xyz = np.asarray([[r["x"], r["y"], r["z"]] for r in rows])
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#344054", linewidth=1.3, alpha=0.78, label="A executed tip path")
    ax.scatter(*start, marker="*", s=95, color="#7F3C8D", label="start tip")
    ax.scatter(0, 0, 0, marker="x", s=70, color="#111827", label="RCM")
    for waypoint, point in points.items():
        ax.scatter(*point, s=58, color="#E69F00", edgecolor="#111827", linewidth=0.6)
        ax.text(*point, f" {waypoint}", fontsize=9)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title="A: eye / fundus proxy and executed path")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left", fontsize=8)
    ax2 = fig.add_subplot(gs[0, 1])
    if rgb_path.exists():
        ax2.imshow(plt.imread(rgb_path))
    ax2.set_title("MuJoCo needle view\nprimary A final RGB frame")
    ax2.axis("off")
    fig.suptitle("Figure 1. A overview; transparent ellipsoid is an illustrative eye proxy, not a measured anatomy model", fontsize=12)
    save_figure(fig, "figure_01_A_scene_overview")


def contiguous_segments(rows: list[dict[str, Any]], key: str) -> Iterable[list[dict[str, Any]]]:
    current: list[dict[str, Any]] = []
    previous: Any = object()
    for row in rows:
        value = row[key]
        if current and value != previous:
            yield current
            current = []
        current.append(row)
        previous = value
    if current:
        yield current


def make_figure_02() -> None:
    rows = load_trajectory(PRIMARY["A_no_occlusion"] / "visual_servo" / "trajectory.csv")
    points = target_world_points()
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    seen_phases: set[str] = set()
    for segment in contiguous_segments(rows, "phase"):
        xyz = np.asarray([[r["x"], r["y"], r["z"]] for r in segment])
        phase = segment[0]["phase"]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=PHASE_COLORS.get(phase, "#555555"), linewidth=2.0,
                label=phase.replace("visual_", "") if phase not in seen_phases else None)
        seen_phases.add(phase)
    for waypoint, point in points.items():
        ax.scatter(*point, s=65, color="#E69F00", edgecolor="#111827", linewidth=0.6)
        ax.text(*point, f" {waypoint}", fontsize=9)
    ax.scatter(0, 0, 0, marker="x", s=70, color="#111827", label="RCM")
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title="A executed 3-D tip trajectory")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Figure 2. Primary A trajectory; colors encode align / approach / retract phases", fontsize=12)
    save_figure(fig, "figure_02_A_executed_trajectory")


def make_figure_03() -> None:
    rows = load_trajectory(PRIMARY["A_no_occlusion"] / "visual_servo" / "trajectory.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained", sharey=True)
    for ax, waypoint in zip(axes, ("P1", "P2", "P3")):
        subset = [row for row in rows if row["waypoint_id"] == waypoint]
        if not subset:
            ax.text(0.5, 0.5, "NOT_RUN", ha="center", va="center", transform=ax.transAxes)
            continue
        t0 = subset[0]["time_s"]
        t = np.asarray([row["time_s"] - t0 for row in subset])
        error = np.maximum(np.asarray([row["error_um"] for row in subset]), 1e-3)
        ax.semilogy(t, error, color="#0072B2", linewidth=1.5)
        ax.axhline(20, color="#D55E00", linestyle="--", linewidth=1.2, label="20 um limit")
        ax.axvspan(t[-1] - min(0.2, t[-1]), t[-1], color="#009E73", alpha=0.12, label="terminal dwell")
        ax.set(title=waypoint, xlabel="phase time (s)", ylim=(1, 3e4))
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("true endpoint error (um, log scale)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 3. A target-wise error convergence; the final three samples are the reported zero-command stop interval", fontsize=12)
    save_figure(fig, "figure_03_A_error_convergence")


def make_figure_04(endpoint_rows: list[dict[str, Any]]) -> None:
    groups = ("A_no_occlusion", "B_occlusion_stop", "C_occlusion_recovery")
    waypoints = ("P1", "P2", "P3")
    by_key = {(row["group"], row["waypoint_id"]): row for row in endpoint_rows}
    fig = plt.figure(figsize=(13, 7.1), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])

    matrix_ax = fig.add_subplot(gs[0, 0])
    y_positions = {group: len(groups) - 1 - index for index, group in enumerate(groups)}
    for group in groups:
        for x_position, waypoint in enumerate(waypoints):
            row = by_key[(group, waypoint)]
            status = row["result"]
            value = row["true_endpoint_error_um"]
            matrix_ax.add_patch(
                plt.Rectangle(
                    (x_position - 0.48, y_positions[group] - 0.43),
                    0.96,
                    0.86,
                    facecolor=STATUS_COLORS.get(status, "#6C757D"),
                    alpha=0.22,
                    edgecolor="#344054",
                    linewidth=1.0,
                )
            )
            text = status
            if value is not None:
                text += f"\n{value:.2f} um"
            if status == "FAIL":
                text += "\nTARGET_NOT_VISIBLE"
            elif status == "NOT_RUN":
                text += "\nprevious phase failed"
            matrix_ax.text(x_position, y_positions[group], text, ha="center", va="center", fontsize=8)
    matrix_ax.set(
        xlim=(-0.5, 2.5),
        ylim=(-0.5, 2.5),
        xticks=range(3),
        xticklabels=waypoints,
        yticks=[y_positions[group] for group in groups],
        yticklabels=[CASE_LABELS[group] for group in groups],
        title="Recorded state by group and target",
    )
    matrix_ax.grid(False)
    matrix_ax.tick_params(length=0)
    matrix_ax.set_xlabel("requested target")

    error_ax = fig.add_subplot(gs[0, 1])
    group_centers = {group: index * 4 + 1 for index, group in enumerate(groups)}
    for group in groups:
        for waypoint_index, waypoint in enumerate(waypoints):
            row = by_key[(group, waypoint)]
            position = group_centers[group] - 1 + waypoint_index
            value = row["true_endpoint_error_um"]
            status = row["result"]
            if value is None:
                error_ax.scatter(position, 1.0, marker="s", s=65, color=STATUS_COLORS["NOT_RUN"], edgecolor="black", linewidth=0.5)
                error_ax.text(position, 1.25, "NOT_RUN", ha="center", va="bottom", fontsize=7, rotation=45)
            else:
                error_ax.scatter(position, value, marker="o" if status == "PASS" else "X", s=68,
                                 color=STATUS_COLORS.get(status, "#6C757D"), edgecolor="black", linewidth=0.5)
                error_ax.text(position, value * 1.14, waypoint, ha="center", va="bottom", fontsize=7)
    error_ax.axhline(20, color="#D55E00", linestyle="--", linewidth=1.2, label="20 um acceptance line")
    error_ax.set_xticks(list(group_centers.values()), ["A\nno occlusion", "B\nocclusion stop", "C\nocclusion recovery"])
    error_ax.set_xlim(-0.7, group_centers[groups[-1]] + 1.7)
    error_ax.set_yscale("symlog", linthresh=20)
    error_ax.set_ylim(0.5, 2500)
    error_ax.set_ylabel("true endpoint error (um; symlog)")
    error_ax.set_title("Measured errors; NOT_RUN retained")
    error_ax.grid(alpha=0.22, axis="y", which="both")
    error_ax.legend(handles=[
        Line2D([0], [0], color="#D55E00", linestyle="--", label="20 um acceptance line"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_COLORS["PASS"], markeredgecolor="black", label="PASS"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=STATUS_COLORS["FAIL"], markeredgecolor="black", label="FAIL"),
        Patch(facecolor=STATUS_COLORS["NOT_RUN"], alpha=0.5, label="NOT_RUN"),
    ], fontsize=8, loc="upper left")
    fig.suptitle("Figure 4. A/B/C endpoint states; B failure and P2/P3 NOT_RUN remain in the denominator", fontsize=12)
    save_figure(fig, "figure_04_endpoint_status")


def load_dataset_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_figure_05(raw_manifest: dict[str, Any]) -> dict[str, Any]:
    data_path = LOCAL["C_occlusion_recovery"] / "dataset" / "data" / "episode_000000.jsonl"
    rows = load_dataset_rows(data_path)
    p2 = [row for row in rows if row["waypoint_id"] == "P2" and row["phase"] == "visual_approach"]
    indices = np.asarray([row["frame_index"] for row in p2], dtype=int)
    times = np.asarray([row["timestamp"] for row in p2], dtype=float)
    state_order = {"braking": 0, "retracting": 1, "reobserving": 2, "tracking": 3}
    states = [row["perception"]["controller_state"] for row in p2]
    state_values = np.asarray([state_order.get(state, -1) for state in states])
    true_error = np.asarray([row["perception"].get("true_position_error_um") or np.nan for row in p2], dtype=float)
    c_state = np.asarray([row["observation.state"][2] for row in p2], dtype=float)
    event_frames = {"occlusion start": 328, "first braking": 344, "first clear": 363,
                    "first reobserving": 365, "tracking": 368, "P2 arrival": 426}
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.step(indices, state_values, where="post", color="#0072B2", linewidth=1.8)
    for frame, label in event_frames.items():
        ax.axvline(label, color="#D55E00" if "occlusion" in frame or "braking" in frame else "#009E73", alpha=0.65, linewidth=1)
        ax.text(label + 1, 3.25, frame, rotation=90, va="top", fontsize=8)
    ax.set_yticks(list(state_order.values()), list(state_order.keys()))
    ax.set_ylabel("controller state")
    ax.set_title("C local replay: P2 long occlusion and recovery")
    ax.grid(alpha=0.2, axis="x")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(indices, true_error, color="#D55E00", linewidth=1.6, label="true endpoint error (um)")
    ax2.set_ylabel("error (um)")
    ax2.set_xlabel("dataset frame index (20 Hz recording)")
    ax2b = ax2.twinx()
    ax2b.plot(indices, c_state, color="#009E73", linewidth=1.2, alpha=0.85, label="C state (mm)")
    ax2b.set_ylabel("C axis state (mm)")
    ax2.axvspan(328, 362, color="#D55E00", alpha=0.08, label="35-frame occlusion")
    ax2.grid(alpha=0.2)
    handles = [Line2D([0], [0], color="#D55E00", label="true endpoint error (um)"),
               Line2D([0], [0], color="#009E73", label="C state (mm)"),
               Line2D([0], [0], color="#D55E00", alpha=0.4, linewidth=8, label="occlusion interval")]
    ax2.legend(handles=handles, fontsize=8, loc="upper right")
    # Add four byte-preserved RGB snapshots below the axes as an inset row.
    inset_y = 0.01
    for i, item in enumerate(raw_manifest["items"][:4]):
        label = item["label"]
        image = plt.imread(OUT / item["copied_path"])
        inset = fig.add_axes([0.06 + i * 0.235, inset_y, 0.19, 0.12])
        inset.imshow(image)
        inset.set_title(label.replace("C_P2_", "").replace("_", " "), fontsize=7)
        inset.axis("off")
    fig.suptitle("Figure 5. C P2 recovery timeline; snapshots are copied without pixel edits", fontsize=12)
    save_figure(fig, "figure_05_C_P2_recovery_timeline")
    start_row = next(row for row in p2 if row["frame_index"] == 344)
    interval = [row for row in p2 if 344 <= row["frame_index"] <= 368]
    c_start = float(start_row["observation.state"][2])
    c_min = float(min(row["observation.state"][2] for row in interval))
    return {
        "p2_approach_dataset_frames": [int(indices[0]), int(indices[-1])],
        "occlusion_frames": [328, 362],
        "first_braking_frame": 344,
        "first_braking_time_s": float(next(row["timestamp"] for row in p2 if row["frame_index"] == 344)),
        "first_clear_frame": 363,
        "first_reobserving_frame": 365,
        "first_reobserving_time_s": float(next(row["timestamp"] for row in p2 if row["frame_index"] == 365)),
        "first_tracking_after_recovery_frame": 368,
        "first_tracking_after_recovery_time_s": float(next(row["timestamp"] for row in p2 if row["frame_index"] == 368)),
        "recovery_duration_s_from_braking_to_tracking": float(next(row["timestamp"] for row in p2 if row["frame_index"] == 368) - next(row["timestamp"] for row in p2 if row["frame_index"] == 344)),
        "retraction_distance_mm_start_to_min_c_axis": c_start - c_min,
        "arrival_frame": 426,
        "arrival_error_um": float(next(row for row in p2 if row["frame_index"] == 426)["perception"]["true_position_error_um"]),
    }


def make_figure_06() -> dict[str, Any]:
    phase_dir = LOCAL["C_occlusion_recovery"] / "visual_servo" / "P3" / "visual_approach"
    npz_path = phase_dir / "final_oct_frame.npz"
    metadata = json_load(phase_dir / "final_oct_frame.json")
    with np.load(npz_path, allow_pickle=False) as archive:
        depth = np.asarray(archive["registered_feature_depth_m"], dtype=np.float32)
        mask = np.asarray(archive["feature_valid_mask"], dtype=bool)
    shown = np.where(mask, depth, np.nan)[::4, ::4]
    rgb = plt.imread(phase_dir / "camera_final.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("C P3 final RGB")
    axes[0].axis("off")
    image = axes[1].imshow(shown, cmap="viridis", norm=Normalize(vmin=0.015, vmax=0.060), interpolation="nearest")
    axes[1].set_title("registered_feature_depth_m")
    axes[1].set_xlabel("pixel x (downsampled x4)")
    axes[1].set_ylabel("pixel y (downsampled x4)")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="depth (m)")
    axes[2].imshow(mask[::4, ::4], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("feature_valid_mask")
    axes[2].axis("off")
    fig.suptitle(
        "Figure 6. P3 final depth audit: ideal MuJoCo registered depth, metres, not physical OCT\n"
        f"frame_id={metadata['frame_id']} | source={metadata['source']} | valid pixels={int(mask.sum())}",
        fontsize=11,
    )
    save_figure(fig, "figure_06_P3_depth_audit")
    return {
        "source": metadata["source"],
        "is_simulation": metadata["is_simulation"],
        "raw_oct_available": metadata["raw_oct_available"],
        "frame_id": metadata["frame_id"],
        "depth_unit": metadata["depth_unit"],
        "shape_hw": metadata["shape_hw"],
        "feature_valid_pixels": int(mask.sum()),
        "npz": rel(npz_path),
        "warning": "ideal MuJoCo registered depth; not physical OCT",
    }


def copy_inputs_and_evidence() -> None:
    copy_file(TARGET, INPUTS_OUT / "three_waypoints.json")
    for name in ("strict_config.json", "stop_occlusion_config.json", "recovery_config.json"):
        copy_file(CONFIG_DIR / name, INPUTS_OUT / name)

    selected_files = [
        "comparison.csv",
        "comparison.md",
        "config_snapshot.json",
        "experiment_protocol.json",
        "run_reports.json",
        "source_sha256.json",
        "targets.json",
        "target_solutions.json",
        "waypoint_results.csv",
    ]
    for group, source in PRIMARY.items():
        destination = EVIDENCE_OUT / "primary" / group
        for name in selected_files:
            source_file = source / name
            if source_file.exists():
                copy_file(source_file, destination / name)
        trajectory = source / "visual_servo" / "trajectory.csv"
        if trajectory.exists():
            copy_file(trajectory, destination / "visual_servo_trajectory.csv")
    representative = {
        "A_no_occlusion": "strict_three_ep000000_frame001288.png",
        "B_occlusion_stop": "strict_occlusion_stop_ep000000_frame000028.png",
        "C_occlusion_recovery": "strict_occlusion_recovery_ep000000_frame000668.png",
    }
    for group, name in representative.items():
        copy_file(MEETING / "rawframes" / name, EVIDENCE_OUT / "primary" / group / "representative_rgb.png")

    for group, source in LOCAL.items():
        destination = EVIDENCE_OUT / "local_replay" / group
        for name in ("comparison.csv", "comparison.md", "config_snapshot.json", "experiment_protocol.json",
                     "run_reports.json", "waypoint_results.csv"):
            if (source / name).exists():
                copy_file(source / name, destination / name)
        (destination / "README.txt").write_text(
            "Diagnostic local replay metadata only; primary evidence remains under evidence/primary.\n"
            "The local run is not substituted for the original episode.\n",
            encoding="utf-8",
        )

    oct_interface = RUN_ROOT / "oct_interface_checks"
    oct_sensor = RUN_ROOT / "oct_sensor_checks"
    for source_dir, name in ((oct_interface, "oct_p1_interface"), (oct_sensor, "oct_sensor")):
        destination = EVIDENCE_OUT / name
        for source_file in source_dir.iterdir():
            if source_file.is_file():
                copy_file(source_file, destination / source_file.name)

    for source in sorted(REPO.glob("*.py")):
        copy_file(source, SOURCE_OUT / source.name)
    for source in (REPO / "config.json", REPO / "scene.xml", REPO / "requirements.txt", REPO / "AGENTS.md"):
        if source.exists():
            copy_file(source, SOURCE_OUT / source.name)
    manifest = {out_rel(path): sha256_file(path) for path in sorted(SOURCE_OUT.iterdir()) if path.is_file()}
    json_dump(SOURCE_OUT / "MANIFEST.json", manifest)


def copy_raw_rgb() -> dict[str, Any]:
    dataset = LOCAL["C_occlusion_recovery"] / "dataset" / "images" / "episode_000000"
    data_rows = load_dataset_rows(LOCAL["C_occlusion_recovery"] / "dataset" / "data" / "episode_000000.jsonl")
    selected = [
        ("C_P2_before_occlusion", 327, "P2 immediately before 35-frame occlusion"),
        ("C_P2_during_occlusion", 340, "P2 during long occlusion"),
        ("C_P2_first_reobserving", 365, "P2 first reobserving frame"),
        ("C_P2_arrival", 426, "P2 arrival / tracking"),
    ]
    items: list[dict[str, Any]] = []
    for label, index, description in selected:
        source = dataset / f"frame_{index:06d}.png"
        destination = RAW_DIR / f"{label}_frame_{index:06d}.png"
        copy_file(source, destination)
        row = data_rows[index]
        items.append(
            {
                "label": label,
                "frame_index": index,
                "description": description,
                "source": rel(source),
                "copied_path": out_rel(destination),
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(destination),
                "byte_identical": source.read_bytes() == destination.read_bytes(),
                "timestamp_s": row["timestamp"],
                "visibility": row["visibility"],
                "perception": row["perception"],
            }
        )
    p3_source = LOCAL["C_occlusion_recovery"] / "visual_servo" / "P3" / "visual_approach" / "camera_final.png"
    p3_destination = RAW_DIR / "C_P3_final_camera.png"
    copy_file(p3_source, p3_destination)
    items.append(
        {
            "label": "C_P3_final_camera",
            "description": "P3 visual-approach phase-end RGB",
            "source": rel(p3_source),
            "copied_path": out_rel(p3_destination),
            "source_sha256": sha256_file(p3_source),
            "copied_sha256": sha256_file(p3_destination),
            "byte_identical": p3_source.read_bytes() == p3_destination.read_bytes(),
        }
    )
    manifest = {"items": items, "raw_bytes_preserved": True}
    json_dump(OUT / "raw_rgb_manifest.json", manifest)
    return manifest


def audit_oct_file(jsonl_path: Path, npz_path: Path | None) -> dict[str, Any]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    oct_rows = [row["oct"] for row in rows]
    frame_ids = [row["frame_id"] for row in oct_rows]
    sequence_ids = [int(row["sequence_id"]) for row in oct_rows]
    timestamps = [float(row["timestamp_s"]) for row in oct_rows]
    sources = sorted(set(row["source"] for row in oct_rows))
    units = sorted(set(row["depth_unit"] for row in oct_rows))
    nav_units = sorted(
        {
            row["navigation_depth"]["unit"]
            for row in rows
            if isinstance(row.get("navigation_depth"), dict) and row["navigation_depth"].get("unit")
        }
    )
    navigation_depth_missing = sum(
        1 for row in rows if not isinstance(row.get("navigation_depth"), dict)
    )
    npz_check: dict[str, Any] | None = None
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            fields = list(archive.files)
            mask = np.asarray(archive["feature_valid_mask"], dtype=bool) if "feature_valid_mask" in archive else None
            npz_check = {
                "path": rel(npz_path),
                "allow_pickle_false_loaded": True,
                "fields": fields,
                "shape_hw": list(mask.shape) if mask is not None else None,
                "feature_valid_pixels": int(mask.sum()) if mask is not None else None,
            }
    return {
        "path": rel(jsonl_path),
        "rows": len(rows),
        "source": sources,
        "depth_units": units,
        "navigation_depth_units": nav_units,
        "navigation_depth_missing_rows": navigation_depth_missing,
        "unique_frame_ids": len(set(frame_ids)) == len(frame_ids),
        "sequence_strictly_increasing": all(b > a for a, b in zip(sequence_ids, sequence_ids[1:])),
        "timestamp_strictly_increasing": all(b > a for a, b in zip(timestamps, timestamps[1:])),
        "first_timestamp_s": timestamps[0] if timestamps else None,
        "last_timestamp_s": timestamps[-1] if timestamps else None,
        "first_sequence_id": sequence_ids[0] if sequence_ids else None,
        "last_sequence_id": sequence_ids[-1] if sequence_ids else None,
        "npz": npz_check,
    }


def build_oct_audit() -> dict[str, Any]:
    phase_audits: list[dict[str, Any]] = []
    for group, run_dir in LOCAL.items():
        for jsonl_path in sorted((run_dir / "visual_servo").glob("P*/*/oct_observations.jsonl")):
            npz_path = jsonl_path.with_name("final_oct_frame.npz")
            phase_audits.append({"group": group, "phase": str(jsonl_path.parent.relative_to(run_dir / "visual_servo")), **audit_oct_file(jsonl_path, npz_path)})
    interface = json_load(RUN_ROOT / "oct_interface_checks" / "oct_interface_verification.json")
    sensor = json_load(RUN_ROOT / "oct_sensor_checks" / "oct_sensor_verification.json")
    primary_status = {
        group: {
            "oct_log_available": False,
            "status": "PRE_OCT_WRAPPER_PRIMARY_EVIDENCE",
            "note": "Primary episode was recorded before the OCT wrapper was connected; do not relabel it as physical OCT.",
        }
        for group in PRIMARY
    }
    return {
        "audit_scope": "local replay logs plus the independent OCT interface/sensor checks",
        "primary_episode_status": primary_status,
        "local_phase_audits": phase_audits,
        "oct_interface_verification": interface,
        "oct_sensor_verification": sensor,
        "association_policy": "RGB and OCT are associated by phase and simulation timestamp; stage-boundary tolerance is 1 ns when required for strict ordering.",
        "interpretation": "All OCT-like data here are ideal MuJoCo registered depth in metres; raw physical OCT and hardware registration are absent.",
    }


def build_metrics(endpoint_rows: list[dict[str, Any]], local_rows: list[dict[str, Any]], recovery: dict[str, Any], depth: dict[str, Any]) -> dict[str, Any]:
    primary_summary = {group: aggregate_group(endpoint_rows, group) for group in PRIMARY}
    local_summary = {group: aggregate_group(local_rows, group) for group in LOCAL}
    # Report full row-level metrics as a flat CSV as well as JSON.
    csv_rows = endpoint_rows + local_rows
    fields = ["source", "group", "waypoint_id", "attempted", "result", "reason", "true_endpoint_error_um", "terminal_stop_validated", "settled_frames", "independent_success", "verification"]
    summary_rows: list[dict[str, Any]] = []
    for source_name, summary in (("primary", primary_summary), ("local_diagnostic", local_summary)):
        for group, data in summary.items():
            primary_case = "primary" if source_name == "primary" else "diagnostic"
            summary_rows.append({
                "source": primary_case,
                "group": group,
                "label": CASE_LABELS[group],
                "requested_targets": data["requested_targets"],
                "attempted_targets": data["attempted_targets"],
                "successful_targets": data["successful_targets"],
                "not_run_targets": data["not_run_targets"],
                "success_over_requested": data["success_over_requested"],
                "mean_endpoint_error_um_successful_only": data["mean_endpoint_error_um_successful_only"],
                "max_endpoint_error_um_successful_only": data["max_endpoint_error_um_successful_only"],
            })
    write_csv(OUT / "endpoint_metrics.csv", [{"source": "primary" if row in endpoint_rows else "local_diagnostic", **row} for row in csv_rows], fields)
    write_csv(OUT / "summary.csv", summary_rows, list(summary_rows[0].keys()))
    return {
        "schema_version": 1,
        "source_separation": {
            "primary": "reports/20261022/meeting/evidence; original A/B/C paired evidence",
            "local_diagnostic": "runs/oct_three_group_79afacc_20261022; replay after OCT wrapper connection",
        },
        "primary": primary_summary,
        "local_diagnostic": local_summary,
        "recovery_timing": recovery,
        "depth_audit_summary": depth,
        "denominator_rule": "All three requested targets remain in the denominator. FAIL and NOT_RUN are not silently removed.",
        "average_rule": "Three-point mean and maximum are reported only when all three targets are successful; otherwise null.",
        "claim_limits": [
            "Descriptive simulation pilot; no success-rate, generalization, or statistical-significance inference.",
            "No hardware execution, tissue contact, physical OCT, or real-time 20 Hz guarantee.",
            "B independent acceleration audit failed; the stop result is not described as safe braking.",
        ],
    }


def write_protocol() -> None:
    stop_config = json_load(CONFIG_DIR / "stop_occlusion_config.json")
    recovery_config = json_load(CONFIG_DIR / "recovery_config.json")
    diff = recursive_diffs(stop_config, recovery_config)
    protocol = {
        "title": "Strict visual and occlusion-recovery three-group pilot",
        "date": "2026-10-22",
        "output_root": "runs/oct_three_group_79afacc_20261022/",
        "groups": {
            "A_no_occlusion": {"config": "strict_config.json", "start_pose_ABC_mm": [0.0, 0.0, 0.0], "order": ["P1", "P2", "P3"]},
            "B_occlusion_stop": {"config": "stop_occlusion_config.json", "start_pose_ABC_mm": [6.481389210293787, 2.847750084902797, 17.8199835578327], "order": ["P1", "P2", "P3"]},
            "C_occlusion_recovery": {"config": "recovery_config.json", "start_pose_ABC_mm": [6.481389210293787, 2.847750084902797, 17.8199835578327], "order": ["P1", "P2", "P3"]},
        },
        "common_cli_flags": ["--strict-visual", "--methods", "visual_servo", "--planner-feedback", "vision", "--profile", "precision", "--task-regime", "visual_current_target", "--sampling-frequency-hz", "20", "--retract-mm", "2", "--transfer-clearance-mm", "1"],
        "occlusion_plan": {"P1": {"start_frame": 20, "duration_frames": 8}, "P2": {"start_frame": 20, "duration_frames": 35}, "relative_to_phase": True},
        "strict_config_structure_diff_stop_vs_recovery": diff,
        "expected_diff_only": ["/navigation/visual_servo/occlusion_handling"],
        "planner_policy": "No RRT or PRM was run in this pilot.",
        "evidence_policy": "Unexpected failures and occlusion failures are retained; reruns are diagnostic and never replace the original episode.",
        "expansion_status": "Pause after three groups pending confirmation before adding the other two layouts.",
    }
    json_dump(OUT / "protocol.json", protocol)


def write_provenance() -> None:
    git_code, git_head = run_command(["git", "rev-parse", "HEAD"])
    status_code, git_status = run_command(["git", "status", "--porcelain"])
    provenance = {
        "requested_commit_prefix": "79afacc",
        "verified_commit": False,
        "remote_fetch_status": "UNAVAILABLE_NETWORK",
        "local_fallback": True,
        "git_metadata": {"rev_parse_exit": git_code, "head_or_error": git_head, "status_exit": status_code, "status_or_error": git_status, "repository_present": (REPO / ".git").exists()},
        "source_note": "This checkout has no .git metadata and the remote could not be fetched; the report does not claim provenance at 79afacc.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO),
        "input_sha256": config_hashes(),
        "versions": source_versions(),
        "primary_evidence_paths": {group: rel(path) for group, path in PRIMARY.items()},
        "local_diagnostic_paths": {group: rel(path) for group, path in LOCAL.items()},
        "source_snapshot": "source_snapshot/local_current/",
    }
    json_dump(OUT / "provenance.json", provenance)


def write_readme(metrics: dict[str, Any]) -> None:
    primary = metrics["primary"]
    lines = [
        "# OCT Three-Group Supplement",
        "",
        "This directory is a local, descriptive simulation pilot supplement for the strict visual A/B/C episode set.",
        "",
        "> Provenance warning: the requested `79afacc` commit could not be fetched or verified. The checkout has no `.git` metadata, so this package uses a clearly labelled local fallback.",
        "",
        "## Evidence separation",
        "",
        "The primary A/B/C episodes are copied from `reports/20261022/meeting/evidence/`. They were recorded before the OCT wrapper was connected. The local replay directories under `runs/oct_three_group_79afacc_20261022/` are diagnostic only and are not substituted for primary results.",
        "",
        "| Group | Primary outcome | Targets | Error summary |",
        "| --- | --- | ---: | --- |",
    ]
    for group in ("A_no_occlusion", "B_occlusion_stop", "C_occlusion_recovery"):
        item = primary[group]
        errors = ", ".join(f"{value:.3f} um" for value in item["endpoint_errors_um_successful_only"])
        summary = errors if errors else "not computed (incomplete)"
        lines.append(f"| {CASE_LABELS[group]} | {item['successful_targets']}/3 successful | {item['attempted_targets']}/3 attempted | {summary} |")
    lines.extend([
        "",
        "A and C have three-point means/maxima only because all three targets succeeded. B retains P1 `FAIL: TARGET_NOT_VISIBLE` and P2/P3 `NOT_RUN`; its independent acceleration audit failed and is not described as safe braking.",
        "",
        "## OCT interpretation",
        "",
        "The OCT-like records are `mujoco_ideal_registered_oct_simulation` depth in metres. They are ideal rendered depth registered to a MuJoCo camera, not physical OCT, and do not validate hardware registration or real-time 20 Hz operation. Full depth is retained only as one final frame per phase; the JSONL audit preserves the sequence metadata.",
        "",
        "## Figures and raw data",
        "",
        "Figures are exported as both PNG and PDF. Figure 5 uses four byte-preserved C P2 RGB snapshots. Figure 6 shows C P3 final RGB, `registered_feature_depth_m`, and `feature_valid_mask` with the frame ID and simulation warning in the caption.",
        "",
        "The figure generation follows the Scientific Visualization and Matplotlib skills' integrity guidance (raw data retained, missing states explicit, units labelled). For the skill reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills*, arXiv:2609.00065, DOI 10.48550/arXiv.2609.00065.",
        "",
        "## Reproduction",
        "",
        "Run `python reports/20261022/oct_three_group_79afacc/build_supplement.py` from the repository root. The script reads existing evidence and writes this supplement; it does not run navigation or RRT/PRM.",
        "",
        "Do not infer hardware performance, real-time behavior, generalization, training performance, or statistical significance from this three-episode pilot.",
        "",
    ])
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_html(metrics: dict[str, Any], raw_manifest: dict[str, Any]) -> None:
    rows = []
    for group in ("A_no_occlusion", "B_occlusion_stop", "C_occlusion_recovery"):
        item = metrics["primary"][group]
        errors = ", ".join(f"{value:.3f}" for value in item["endpoint_errors_um_successful_only"]) or "-"
        rows.append(f"<tr><td>{html.escape(CASE_LABELS[group])}</td><td>{item['successful_targets']}/3</td><td>{item['attempted_targets']}/3</td><td>{html.escape(errors)}</td><td>{html.escape(str(item['max_endpoint_error_um_successful_only']))}</td></tr>")
    figure_titles = {
        "figure_01_A_scene_overview": "A scene overview",
        "figure_02_A_executed_trajectory": "A executed trajectory",
        "figure_03_A_error_convergence": "A error convergence",
        "figure_04_endpoint_status": "Endpoint status",
        "figure_05_C_P2_recovery_timeline": "C P2 recovery timeline",
        "figure_06_P3_depth_audit": "P3 depth audit",
    }
    figure_html = "\n".join(
        f'<figure><a href="figures/{name}.pdf"><img src="figures/{name}.png" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)} (PNG; click for PDF)</figcaption></figure>'
        for name, title in figure_titles.items()
    )
    raw_html = "\n".join(
        f'<figure><img src="{html.escape(item["copied_path"])}" alt="{html.escape(item["description"])}"><figcaption>{html.escape(item["label"])}; byte-identical={item["byte_identical"]}</figcaption></figure>'
        for item in raw_manifest["items"]
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OCT three-group supplement</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#17202a;line-height:1.45}}.warning{{border-left:5px solid #D55E00;background:#fff4e5;padding:1rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #cbd5e1;padding:.45rem;text-align:left}}figure{{display:inline-block;width:31%;vertical-align:top;margin:.7rem}}figure img{{width:100%;height:220px;object-fit:contain;background:#f8fafc;border:1px solid #cbd5e1}}figcaption{{font-size:.88rem}}.raw img{{height:145px}}code{{background:#eef2f7;padding:.1rem .25rem}}</style></head>
<body><h1>Strict visual and occlusion recovery: A/B/C</h1>
<div class="warning"><strong>Provenance and interpretation warning.</strong> Requested commit prefix <code>79afacc</code> was not verified because the remote was unavailable and this checkout has no Git metadata. This is a local fallback. Primary episodes predate the OCT wrapper. All OCT-like data shown here are ideal MuJoCo registered depth in metres, <strong>not physical OCT</strong>; no hardware execution or real-time 20 Hz guarantee is claimed.</div>
<h2>Primary descriptive outcomes</h2><table><thead><tr><th>Group</th><th>Successful targets</th><th>Attempted</th><th>Successful endpoint errors (um)</th><th>Three-point max (um)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>B retains its observed <code>FAIL: TARGET_NOT_VISIBLE</code> at P1 and <code>NOT_RUN</code> at P2/P3. Its independent acceleration audit failed, so it is not labelled safe braking. A and C means/maxima are computed only when all three targets succeeded.</p>
<h2>Figures</h2><section>{figure_html}</section>
<h2>Byte-preserved RGB checkpoints</h2><section class="raw">{raw_html}</section>
<h2>Scope</h2><p>This is a descriptive simulation pilot, not evidence of hardware accuracy, tissue contact safety, policy training, generalization, success-rate estimation, or statistical significance. The prior preview video, if used elsewhere, predates OCT interface connection and its 20 fps is playback speed only.</p>
<p>Machine-readable details: <a href="provenance.json">provenance</a>, <a href="protocol.json">protocol</a>, <a href="metrics.json">metrics</a>, <a href="oct_audit.json">OCT audit</a>, <a href="SHA256SUMS.json">SHA256 manifest</a>.</p>
</body></html>
"""
    (OUT / "slides_oct_update.html").write_text(page, encoding="utf-8")


def write_checksums() -> None:
    entries: dict[str, Any] = {}
    for path in sorted(OUT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "SHA256SUMS.json"
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        entries[out_rel(path)] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    json_dump(OUT / "SHA256SUMS.json", {"algorithm": "SHA256", "excludes": ["SHA256SUMS.json"], "files": entries})


def write_test_log_tail() -> None:
    source = REPO / "TEST_LOG.md"
    lines = source.read_text(encoding="utf-8").splitlines() if source.exists() else []
    (OUT / "TEST_LOG_tail.txt").write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    copy_inputs_and_evidence()
    raw_manifest = copy_raw_rgb()
    endpoint_rows = load_endpoint_rows()
    local_rows = load_local_endpoint_rows()
    recovery = make_figure_05(raw_manifest)
    make_figure_01()
    make_figure_02()
    make_figure_03()
    make_figure_04(endpoint_rows)
    depth = make_figure_06()
    oct_audit = build_oct_audit()
    json_dump(OUT / "oct_audit.json", oct_audit)
    write_protocol()
    write_provenance()
    metrics = build_metrics(endpoint_rows, local_rows, recovery, depth)
    json_dump(OUT / "metrics.json", metrics)
    write_readme(metrics)
    write_html(metrics, raw_manifest)
    write_test_log_tail()
    write_checksums()
    print(f"built {OUT}")
    print(f"primary endpoint rows: {len(endpoint_rows)}; local diagnostic rows: {len(local_rows)}")
    print(f"figures: {len(list(FIG_DIR.glob('*.png')))} PNG + {len(list(FIG_DIR.glob('*.pdf')))} PDF")


if __name__ == "__main__":
    main()
