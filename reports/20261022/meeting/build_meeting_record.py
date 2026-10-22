#!/usr/bin/env python3
"""Build the 2026-10-23 meeting record from recorded evidence, never run control.

Usage (run after collectors have exited):
  python runs/multitarget_occlusion_20261022/build_meeting_record.py
  python runs/multitarget_occlusion_20261022/build_meeting_record.py --allow-incomplete

Only meeting/ is written. Default mode requires final reports, manifests and
all declared JSONL episodes for every case. --allow-incomplete is for reviewing
an explicitly incomplete record, and never converts missing evidence to PASS.
The light archive contains selected raw PNGs, not the full RGB datasets.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcm-meeting-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rcm-meeting-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
CASE_NAMES = ("baseline_three", "strict_three", "strict_occlusion_stop", "strict_occlusion_recovery")
LABELS = {
    "baseline_three": "混合三点对照（场景解析转移 + 视觉接近）",
    "strict_three": "严格 RGB-D 三点（零位起始）",
    "strict_occlusion_stop": "严格视觉 + 实体遮挡 / stop",
    "strict_occlusion_recovery": "严格视觉 + 实体遮挡 / static_memory_reobserve",
}
SHORT = {"baseline_three": "Hybrid control", "strict_three": "Strict RGB-D", "strict_occlusion_stop": "Occlusion: stop", "strict_occlusion_recovery": "Occlusion: memory/reobserve"}
PERCEPTION_FIELDS = {"target_source", "tip_source", "controller_state", "cached_target_age_s", "estimated_position_error_um", "true_position_error_um", "target_observed", "control_frame_index", "occlusion_event_id"}
POLICY_FIELDS = {"observation.images.microscope_rgb", "observation.state", "task"}
VIS_STATES = ("clear", "partial_needle", "full_needle", "not_visible", "unknown")
VIS_COLORS = ("#49a585", "#e5ad35", "#b66f13", "#b84b63", "#999999")
COMMIT = "a20d76ef3979b804d7dbf599d577826957e0a8a9"
TEMPLATE_URL = f"https://github.com/HKUSTDial/Supervisor-Skills/blob/{COMMIT}/skills/benchmark-paper-template/SKILL.md"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def zero_velocity(value):
    return value is not None and len(value) == 3 and bool(np.allclose(value, 0, atol=1e-12, rtol=0))


def safe_dataset_path(root: Path, relative: str):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def phase_detail(report, waypoint_id, index):
    details = report.get("control_details", [])
    matched = [x for x in details if x.get("waypoint_id") == waypoint_id and x.get("phase") == "visual_approach"]
    if matched:
        return matched[-1]
    # Legacy hybrid details follow visit_order and do not contain waypoint_id.
    if all(x.get("waypoint_id") is None for x in details):
        approaches = [x for x in details if x.get("phase") == "visual_approach"]
        if index < len(approaches):
            return approaches[index]
    return None


def endpoint_metrics(report):
    rows = report.get("target_success_by_visibility", {}).get("target_visits") or report.get("waypoints", [])
    result = []
    for i, raw in enumerate(rows):
        wid = raw.get("waypoint_id", f"unknown_{i}")
        attempted = bool(raw.get("attempted", raw.get("result") not in ("NOT_RUN", None)))
        detail = phase_detail(report, wid, i)
        err = finite(raw.get("endpoint_error_um", raw.get("tip_error_um")))
        settled = detail.get("settled_frames") if detail else None
        velocity = detail.get("final_command_velocity_A_deg_B_deg_C_mm_per_s") if detail else None
        independent = bool(attempted and raw.get("result") == "PASS" and err is not None and err < 20
                           and raw.get("terminal_stop_validated") is True
                           and settled is not None and settled >= 3 and zero_velocity(velocity))
        result.append({"waypoint_id": wid, "attempted": attempted, "recorded_result": raw.get("result", "UNKNOWN"),
                       "reason": raw.get("reason"), "true_endpoint_error_um": err,
                       "terminal_stop_validated": raw.get("terminal_stop_validated"),
                       "settled_frames": settled, "final_command_velocity": velocity,
                       "independent_success_lt20um_and_3_zero_frames": independent,
                       "verification": "VERIFIED" if detail is not None else "PHASE_DETAIL_MISSING"})
    return result


def audit_episode(dataset: Path, episode: dict, rows: list, strict: bool):
    n = len(rows)
    states = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
    stamps = np.asarray([r["timestamp"] for r in rows], dtype=float)
    errors = np.abs(actions[:-1] - states[1:]) if n > 1 else np.empty((0, 3))
    aligned = bool(np.array_equal(actions[:-1], states[1:]))
    hold = bool(n and np.array_equal(actions[-1], states[-1]))
    dt = np.diff(stamps)
    issues = []
    if n != episode.get("frames"):
        issues.append(f"manifest.frames={episode.get('frames')} but JSONL rows={n}")
    if not aligned or not hold:
        issues.append("action/next-state alignment or terminal hold failed")
    if dt.size and not np.all(dt > 0):
        issues.append("timestamps are not strictly increasing")
    missing_png = [r["observation.images.microscope_rgb"] for r in rows if not safe_dataset_path(dataset, r["observation.images.microscope_rgb"]).is_file()]
    if missing_png:
        issues.append(f"missing raw PNGs: {len(missing_png)}")
    bundles = {}
    expectations = (("policy_inputs", POLICY_FIELDS), ("policy_inputs_no_image", {"observation.state", "task"}), ("action_labels", {"action"}))
    for kind, expected in expectations:
        rel = episode.get(kind)
        if rel is None:
            issues.append(f"missing manifest reference: {kind}")
            continue
        path = safe_dataset_path(dataset, rel)
        if not path.is_file():
            issues.append(f"missing bundle: {rel}")
            continue
        with np.load(path, allow_pickle=False) as bundle:
            fields = set(bundle.files)
            matching = fields == expected
            lengths = {k: int(len(bundle[k])) for k in bundle.files}
            same = all(value == n for value in lengths.values())
            for key in fields & expected:
                if key == "observation.state":
                    same = same and np.array_equal(bundle[key], states)
                elif key == "action":
                    same = same and np.array_equal(bundle[key], actions)
                elif key in ("task", "observation.images.microscope_rgb"):
                    same = same and np.array_equal(bundle[key], np.asarray([r[key] for r in rows]))
            bundles[kind] = {"path": rel, "fields": sorted(fields), "exact_expected_fields": matching,
                             "rows_and_values_match_jsonl": bool(same), "lengths": lengths}
            if not matching or not same:
                issues.append(f"policy/action bundle contract failed: {kind}")
    perception_missing = sum(not PERCEPTION_FIELDS.issubset(r.get("perception", {})) for r in rows)
    if strict and perception_missing:
        issues.append(f"schema-2 perception fields missing in {perception_missing} rows")
    return {"rows": n, "declared_frames": episode.get("frames"), "action_equals_next_state_exact_float32": aligned,
            "max_action_next_state_abs_difference_ABC": errors.max(axis=0).tolist() if errors.size else [0., 0., 0.],
            "terminal_action_holds_current_state": hold, "timestamps_strictly_increasing": bool(not dt.size or np.all(dt > 0)),
            "observed_dt_min_s": float(dt.min()) if dt.size else None,
            "observed_dt_max_s": float(dt.max()) if dt.size else None,
            "observed_dt_median_s": float(np.median(dt)) if dt.size else None,
            "shorter_than_nominal_intervals": int(np.sum(dt < .05 - 1e-8)),
            "perception_schema": "COMPLETE" if not perception_missing else ("LEGACY_UNAVAILABLE" if not strict else "INCOMPLETE"),
            "perception_rows_missing_fields": perception_missing, "missing_png_count": len(missing_png),
            "bundles": bundles, "issues": issues, "pass": not issues}


def load_case(name: str):
    root = BASE / name
    mandatory = [root / "experiment_protocol.json", root / "run_reports.json", root / "dataset/manifest.json"]
    missing = [str(p.relative_to(BASE)) for p in mandatory if not p.is_file()]
    case = {"name": name, "label": LABELS[name], "status": "INCOMPLETE", "missing": missing, "episodes": [],
            "protocol": read_json(mandatory[0]) if mandatory[0].is_file() else {},
            "dataset_path": str((root / "dataset").resolve()), "reports": [], "manifest": {}}
    verification = root / "dataset_verification.json"
    case["independent_dataset_verification"] = read_json(verification) if verification.is_file() else None
    records = []
    if missing:
        # Partial phase files are never an episode-level PASS.
        case["completed_phase_evidence"] = [
            {"file": str(p.relative_to(BASE)), "result": (r := read_json(p)).get("result"),
             "true_error_um": finite(r.get("final_tip_target_error_um"))}
            for p in sorted((root / "visual_servo").glob("*/*/visual_servo_report.json"))]
        return case, records
    reports = read_json(mandatory[1])
    if not isinstance(reports, list) or not reports:
        raise ValueError(f"{name}: run_reports must be a nonempty list")
    manifest = read_json(mandatory[2])
    case["reports"], case["manifest"] = reports, manifest
    if manifest.get("schema_version") != 2:
        case["missing"].append("dataset schema_version is not 2")
    for episode in manifest.get("episodes", []):
        jsonl = safe_dataset_path(root / "dataset", episode["jsonl"])
        if not jsonl.is_file():
            case["missing"].append(str(jsonl.relative_to(BASE)))
            continue
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        idx = episode["episode_index"]
        matching = [r for r in reports if r.get("episode_index") == idx and r.get("method") == episode.get("method")]
        if len(matching) != 1:
            case["missing"].append(f"episode {idx}: expected one matching report, found {len(matching)}")
            continue
        report = matching[0]
        endpoints = endpoint_metrics(report)
        audit = audit_episode(root / "dataset", episode, rows, name != "baseline_three")
        details = report.get("control_details", [])
        counters = {key: (sum(int(d.get(key, 0)) for d in details) if any(key in d for d in details) else None)
                    for key in ("memory_frame_count", "recovery_attempt_count", "recovery_success_count")}
        counts = Counter(r.get("visibility", {}).get("target_occlusion_state", "unknown") for r in rows)
        states = Counter(r.get("perception", {}).get("controller_state", "unknown") for r in rows)
        sources = Counter(r.get("perception", {}).get("target_source", "unknown") for r in rows)
        event_counts = Counter(r.get("perception", {}).get("occlusion_event_id", "unknown") for r in rows)
        observed_errors = [value for r in rows if (value := finite(r.get("perception", {}).get("true_position_error_um"))) is not None]
        em = {"episode_index": idx, "method": episode.get("method"), "recorded_result": report.get("result", "UNKNOWN"),
              "manifest_result": episode.get("result", "UNKNOWN"), "endpoints": endpoints,
              "requested_targets": len(endpoints), "attempted_targets": sum(x["attempted"] for x in endpoints),
              "successful_targets": sum(x["independent_success_lt20um_and_3_zero_frames"] for x in endpoints),
              "frames": len(rows), "audit": audit, "visibility_frame_counts": {k: counts[k] for k in VIS_STATES},
              "controller_state_frame_counts": dict(states), "target_source_frame_counts": dict(sources),
              "occlusion_event_frame_counts": dict(event_counts), "phase_report_counters": counters,
              "counter_scope": "controller phase counters and exported frame counts are distinct observations; neither substitutes for event success",
              "true_error_rows_available": len(observed_errors),
              "phase_results": [{k: d.get(k) for k in ("waypoint_id", "phase", "result", "reason", "memory_frame_count", "recovery_attempt_count", "recovery_success_count", "recovery_limits")} for d in details],
              "final_velocity_zero": zero_velocity(report.get("terminal_velocity", report.get("stop_constraint_summary", {}).get("terminal_velocity_A_deg_B_deg_C_mm_per_s"))),
              "jsonl": str(jsonl.relative_to(BASE))}
        if report.get("result") != episode.get("result"):
            audit["issues"].append("report/manifest outcome mismatch")
            audit["pass"] = False
        case["episodes"].append(em)
        records.append({"episode": em, "rows": rows, "report": report, "root": root})
    if not case["episodes"]:
        case["missing"].append("no final episodes loaded")
    if len(case["episodes"]) != len(reports):
        case["missing"].append("manifest/report episode count mismatch")
    if sum(e["frames"] for e in case["episodes"]) != manifest.get("total_frames"):
        case["missing"].append("manifest total_frames differs from loaded rows")
    files = [p for p in (root / "dataset").rglob("*") if p.is_file()]
    case["dataset_size_bytes"] = sum(p.stat().st_size for p in files)
    case["dataset_png_count"] = sum(p.suffix.lower() == ".png" for p in files)
    case["dataset_png_bytes"] = sum(p.stat().st_size for p in files if p.suffix.lower() == ".png")
    case["status"] = "COMPLETE" if not case["missing"] else "INCOMPLETE"
    return case, records


def paired_protocol(cases):
    left, right = (cases[n] for n in ("strict_occlusion_stop", "strict_occlusion_recovery"))
    fields = ("start_task_A_deg_B_deg_C_mm", "occlusion_fixture", "guidance_mode")
    matches = {key: left["protocol"].get(key) == right["protocol"].get(key) for key in fields}
    contracts = [dict(c["protocol"].get("strict_visual_control_contract") or {}) for c in (left, right)]
    memory_declarations = [contract.pop("temporal_target_memory", None) for contract in contracts]
    matches["strict_visual_control_contract_except_expected_memory_declaration"] = contracts[0] == contracts[1]
    target_paths = [BASE / n / "targets.json" for n in (left["name"], right["name"])]
    matches["targets"] = all(p.is_file() for p in target_paths) and read_json(target_paths[0]) == read_json(target_paths[1])
    config_paths = [BASE / n / "config_snapshot.json" for n in (left["name"], right["name"])]
    configs = [read_json(p) for p in config_paths] if all(p.is_file() for p in config_paths) else []
    differences = []
    def diff(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(a.keys() | b.keys()):
                diff(a.get(key), b.get(key), prefix + "." + key if prefix else key)
        elif a != b:
            differences.append({"field": prefix, "stop": a, "recovery": b})
    if configs:
        diff(*configs)
    expected = "navigation.visual_servo.occlusion_handling"
    matches["only_config_mode_differs"] = bool(configs and len(differences) == 1 and differences[0]["field"] == expected)
    return {"matching_fields": matches, "config_differences": differences, "expected_mode_dependent_memory_declarations": dict(zip(("stop", "recovery"), memory_declarations)), "pass": all(matches.values()),
            "scope": "matched scenario/reset/schedule; stop terminates early, so observed exposure durations need not match planned durations"}


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_endpoint_plot(cases, out):
    fig, ax = plt.subplots(figsize=(10, 4.7))
    pos, labels = [], []
    i = 0
    for name, case in cases.items():
        for ep in case["episodes"]:
            for item in ep["endpoints"]:
                i += 1
                err = item["true_endpoint_error_um"]
                labels.append(f"{SHORT[name]}\n{item['waypoint_id']}")
                pos.append(i)
                if err is not None:
                    good = item["independent_success_lt20um_and_3_zero_frames"]
                    ax.scatter(i, max(err, 1e-6), color="#247e73" if good else "#bc4c61", marker="o" if good else "x", s=65, zorder=3)
                    ax.annotate(f"{err:.2f}", (i, max(err, 1e-6)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
                else:
                    ax.text(i, 1.15, "NOT RUN", rotation=90, ha="center", fontsize=7)
    ax.axhline(20, color="#aa6720", linestyle="--", linewidth=1.2, label="Independent acceptance <20 um + 3 zero-velocity frames")
    ax.set_yscale("log")
    ax.set_ylabel("True endpoint error (um), log scale")
    ax.set_xticks(pos, labels, fontsize=7, rotation=25, ha="right")
    ax.set_title("Endpoint evidence by target (hybrid control is a separate comparator)")
    ax.grid(axis="y", alpha=.22)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, out / "endpoint_errors")


def make_executed_path(cases, output):
    """Plot logged execution and independent target markers, never a plan."""
    path = BASE / "strict_three/visual_servo/trajectory.csv"
    case = cases["strict_three"]
    if not path.is_file() or not case["reports"]:
        return {"status": "UNAVAILABLE", "source": str(path)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    points = np.asarray([[float(r[f"tip_world_{axis}_mm"]) for axis in "xyz"] for r in rows])
    targets = {}
    for report in case["reports"]:
        for i, wid in enumerate(report.get("visit_order", [])):
            detail = phase_detail(report, wid, i)
            if detail is not None and detail.get("target_world_mm") is not None:
                targets[wid] = detail["target_world_mm"]
    if not rows or not targets:
        return {"status": "UNAVAILABLE", "source": str(path), "reason": "no executed rows or independent target coordinates"}
    target_points = np.asarray(list(targets.values()))
    center = target_points.mean(axis=0)
    extent = max(float(np.ptp(target_points, axis=0).max()/2 + 2.5), 3.)
    phase_colors = {"visual_align": "#2d79b3", "visual_approach": "#368f79", "visual_retract": "#c47c36"}
    fig = plt.figure(figsize=(12, 5.4))
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    for plot_index, ax in enumerate(axes):
        seen = set()
        begin = 0
        for i in range(1, len(rows)+1):
            if i == len(rows) or rows[i]["phase"] != rows[begin]["phase"] or rows[i]["waypoint_id"] != rows[begin]["waypoint_id"]:
                phase = rows[begin]["phase"]
                segment = points[max(0, begin-1):i]
                if plot_index == 1:
                    inside_zoom = ((segment >= center-extent) & (segment <= center+extent)).all(axis=1)
                    segment = np.where(inside_zoom[:, None], segment, np.nan)
                ax.plot(*segment.T, color=phase_colors.get(phase, "#777777"), linewidth=1.4,
                        label=phase if phase not in seen else None)
                seen.add(phase)
                begin = i
        ax.scatter(*target_points.T, marker="*", s=100, color="#b6405d", label="Independent target markers", depthshade=False)
        for wid, point in targets.items():
            ax.text(*point, " " + wid, fontsize=9)
        ax.set_xlabel("World X (mm)")
        ax.set_ylabel("World Y (mm)")
        ax.set_zlabel("World Z (mm)", labelpad=3)
        ax.view_init(elev=22, azim=-60)
        ax.set_box_aspect((1, 1, 1))
    axes[0].scatter(*points[0], color="#263747", s=35, label="Start")
    axes[0].set_title("Full executed tip trajectory")
    axes[0].legend(fontsize=7, loc="upper left")
    for setter, c in zip((axes[1].set_xlim, axes[1].set_ylim, axes[1].set_zlim), center):
        setter(c-extent, c+extent)
    axes[1].set_title("Fundus region: retract / align / approach")
    fig.suptitle("Strict RGB-D: executed path from independent evaluation (not a planned path)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.subplots_adjust(right=.93, wspace=.16)
    save_figure(fig, output / "figures/executed_strict_path")
    return {"status": "AVAILABLE", "source": str(path), "source_sha256": digest(path), "samples": len(rows),
            "targets_world_mm": targets, "scope": "logged executed tip trajectory and independent target coordinates; not model-planned path"}


def make_timeline(name, record, out):
    rows = record["rows"]
    t = np.asarray([r["timestamp"] for r in rows], dtype=float)
    fig, axs = plt.subplots(4, 1, figsize=(11.8, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.4, .85, 1.4]})
    true = np.asarray([finite(r.get("perception", {}).get("true_position_error_um")) if finite(r.get("perception", {}).get("true_position_error_um")) is not None else np.nan for r in rows])
    estimated = np.asarray([finite(r.get("perception", {}).get("estimated_position_error_um")) if finite(r.get("perception", {}).get("estimated_position_error_um")) is not None else np.nan for r in rows])
    if np.isfinite(true).any():
        axs[0].plot(t, true, color="#285c91", linewidth=1.2, label="True error (evaluation only)")
        axs[0].plot(t, estimated, color="#d58c2b", linewidth=.9, alpha=.85, label="Controller error to phase goal")
        axs[0].set_yscale("symlog", linthresh=20)
    else:
        axs[0].text(.5, .5, "Legacy export: no per-frame perception/error record", ha="center", transform=axs[0].transAxes)
    axs[0].axhline(20, color="#9d4b4b", linestyle="--", linewidth=.8, label="20 um endpoint threshold")
    axs[0].set_ylabel("Position error (um)")
    axs[0].legend(fontsize=8, loc="upper right")
    px = [r.get("visibility", {}).get("target_pixels_visible", -1) for r in rows]
    axs[1].step(t, px, where="post", linewidth=1, color="#277668")
    axs[1].set_ylabel("Visible target\npixels")
    vis = np.asarray([VIS_STATES.index(r.get("visibility", {}).get("target_occlusion_state", "unknown")) for r in rows])
    # One colored segment per exported row; width is its actual outgoing interval.
    end = t[-1] + .05 if len(t) else .05
    edges = np.r_[t, end]
    axs[2].pcolormesh(edges, [0, 1], vis[None, :], cmap=ListedColormap(VIS_COLORS), vmin=-.5, vmax=4.5, shading="flat")
    axs[2].set_yticks([])
    axs[2].set_ylabel("Visibility")
    from matplotlib.patches import Patch
    axs[2].legend(handles=[Patch(color=c, label=s) for s, c in zip(VIS_STATES, VIS_COLORS)], ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.55))
    states = [r.get("perception", {}).get("controller_state", "unknown") for r in rows]
    unique = list(dict.fromkeys(states))
    axs[3].step(t, [unique.index(s) for s in states], where="post", color="#724c88", linewidth=1.1)
    axs[3].set_yticks(range(len(unique)), unique, fontsize=7)
    axs[3].set_ylabel("Controller state")
    axs[3].set_xlabel("Simulation time (s); phase endpoint samples may be closer than 0.05 s")
    for i in range(1, len(rows)):
        if rows[i].get("waypoint_id") != rows[i-1].get("waypoint_id"):
            for ax in axs:
                ax.axvline(t[i], color="#555555", linestyle=":", linewidth=.8)
            axs[0].annotate(rows[i].get("waypoint_id", "?"), (t[i], .96), xycoords=("data", "axes fraction"), fontsize=8)
    for ax in axs:
        ax.grid(axis="x", alpha=.15)
    fig.suptitle(f"{SHORT[name]} / episode {record['episode']['episode_index']} / {record['episode']['recorded_result']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    stem = f"timeline_{name}_ep{record['episode']['episode_index']:06d}"
    save_figure(fig, out / stem)
    return stem


def choose_raw_frames(records, output):
    candidates = []
    for name in CASE_NAMES:
        recs = records[name]
        if not recs:
            continue
        record = recs[0]
        rows = record["rows"]
        if not rows:
            continue
        selections = [("start", 0)]
        occluded = [i for i, r in enumerate(rows) if r.get("visibility", {}).get("target_occlusion_state") == "not_visible"]
        recovery = [i for i, r in enumerate(rows) if "recover" in r.get("perception", {}).get("controller_state", "").lower()]
        if occluded:
            selections.append(("first full target absence", occluded[0]))
        elif any(r.get("visibility", {}).get("target_occlusion_state") == "partial_needle" for r in rows):
            selections.append(("natural partial needle occlusion", next(i for i, r in enumerate(rows) if r.get("visibility", {}).get("target_occlusion_state") == "partial_needle")))
        if recovery:
            selections.append(("first recovery state", recovery[0]))
        selections.append(("final recorded frame", len(rows)-1))
        used = set()
        for role, i in selections:
            if i in used:
                continue
            used.add(i)
            row = rows[i]
            source = safe_dataset_path(record["root"] / "dataset", row["observation.images.microscope_rgb"])
            dest = output / "rawframes" / f"{name}_ep{record['episode']['episode_index']:06d}_frame{row['frame_index']:06d}.png"
            dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(source, dest)
            candidates.append({"case": name, "role": role, "frame_index": row["frame_index"], "timestamp_s": row["timestamp"],
                               "waypoint_id": row.get("waypoint_id"), "phase": row.get("phase"),
                               "visibility": row.get("visibility", {}).get("target_occlusion_state"),
                               "controller_state": row.get("perception", {}).get("controller_state", "unknown"),
                               "source": str(source), "copy": str(dest.relative_to(output)), "sha256": digest(dest),
                               "processing": "byte-identical raw PNG copy; montage uses display scaling only, no pixel overlay"})
    if candidates:
        cols = 3
        fig, axs = plt.subplots(math.ceil(len(candidates) / cols), cols, figsize=(12, 3.2 * math.ceil(len(candidates)/cols)), squeeze=False)
        for ax in axs.flat:
            ax.axis("off")
        for ax, item in zip(axs.flat, candidates):
            ax.imshow(plt.imread(output / item["copy"]))
            ax.set_title(f"{SHORT[item['case']]} / {item['waypoint_id']} / t={item['timestamp_s']:.2f}s\n{item['role']} / {item['visibility']}", fontsize=8)
        fig.suptitle("Raw controller RGB frames (uncropped; display scaling only; no overlays)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .97))
        save_figure(fig, output / "figures/raw_frame_montage")
    write_json(output / "rawframes.json", candidates)
    return candidates


def target_evidence():
    targets = read_json(BASE / "strict_three/targets.json")
    proof = read_json(BASE / "strict_three/fundus_target_verification.json")
    byid = {x["waypoint_id"]: x for x in proof["targets"]}
    return {"definition": proof["definition"], "recorded_result": proof["result"],
            "targets": [{**x, **byid[x["id"]]} for x in targets["targets"]]}


def summary_rows(cases):
    table = []
    for name, case in cases.items():
        eps = case["episodes"]
        requested = sum(e["requested_targets"] for e in eps)
        attempted = sum(e["attempted_targets"] for e in eps)
        success = sum(e["successful_targets"] for e in eps)
        finalerrors = [x["true_endpoint_error_um"] for e in eps for x in e["endpoints"] if x["attempted"] and x["true_endpoint_error_um"] is not None]
        table.append({"case": name, "label": LABELS[name], "evidence_status": case["status"],
                      "episode_results": [e["recorded_result"] for e in eps], "episodes": len(eps),
                      "requested_targets": requested, "attempted_targets": attempted, "successful_targets": success,
                      "success_over_requested": success/requested if requested else None,
                      "success_over_attempted": success/attempted if attempted else None,
                      "max_attempted_endpoint_error_um": max(finalerrors) if finalerrors else None,
                      "frames": sum(e["frames"] for e in eps), "dataset_size_bytes": case.get("dataset_size_bytes"),
                      "dataset_png_count": case.get("dataset_png_count"),
                      "visit_orders": [r.get("visit_order", []) for r in case["reports"]],
                      "motion_times_s": [r.get("motion_time_s") for r in case["reports"]],
                      "independent_rate_audit": (case.get("independent_dataset_verification") or {}).get("visual_boundary_rate_limits", "NOT_RUN"),
                      "audit_pass": all(e["audit"]["pass"] for e in eps) if eps else None})
    return table


def markdown_table(headers, rows):
    escape = lambda x: ("—" if x is None else str(x)).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join("| " + " | ".join(escape(x) for x in row) + " |" for row in rows)


def report_text(metrics):
    summary, cases = metrics["summary"], metrics["cases"]
    status = "完整最终证据" if metrics["complete"] else "未完成草稿：不能用作最终 PASS 结论"
    table = markdown_table(["实验", "落盘状态/episode结果", "成功/计划", "成功/已尝试", "最大已尝试终点误差 µm", "帧"],
        [[s["label"], f"{s['evidence_status']} / {','.join(s['episode_results']) or '未完成'}", f"{s['successful_targets']}/{s['requested_targets']}" if s["requested_targets"] else "未完成",
          f"{s['successful_targets']}/{s['attempted_targets']}" if s["attempted_targets"] else "未完成", number(s["max_attempted_endpoint_error_um"]), s["frames"]] for s in summary])
    targettable = markdown_table(["目标", "robot XYZ mm", "归一化 posterior Z", "椭球方程值", "径向表面差 µm", "验证"],
        [[t["id"], ", ".join(f"{v:.9f}" for v in t["point_mm"]), number(t["normalized_posterior_z"], 9), number(t["ellipsoid_value"], 15), f"{t['surface_radial_gap_um']:.3e}", t["result"]] for t in metrics["target_evidence"]["targets"]])
    sections = [f"# 2026-10-23 组会：多目标严格视觉控制与实体遮挡\n\n生成时间：{metrics['generated_at_utc']}。记录状态：**{status}**。所有结果数字读取本地最终 JSON/JSONL；本脚本不启动导航、不补写结果。\n\n## 真实结果与分母\n\n{table}\n\n成功定义：真实终点误差严格小于 20 µm，且该目标 phase report 记录至少 3 帧零命令速度停稳；失败 episode、未运行目标均保留。成功/计划将失败后未尝试目标计入分母；成功/已尝试只描述实际尝试，二者不能互换。结果 FAIL 表示实验结局，不表示材料构建失败。\n\n![True endpoint errors](figures/endpoint_errors.png)\n\n真实误差仅用于独立验收。回撤和方向对准的控制误差指各自视觉构造的阶段目标，不能把它解释为到物理标靶的真实误差。"]
    for name, case in cases.items():
        text = f"## {LABELS[name]}\n\n"
        verifier = case.get("independent_dataset_verification")
        if verifier:
            text += "独立复核（与导航结果分开）：" + "；".join(
                f"{label}={verifier.get(key, 'NOT_RUN')}" for key, label in
                (("result", "完整性总结果"), ("schema_dtypes_and_shapes", "schema"),
                 ("action_next_state_alignment", "动作对齐"), ("RGB_state_rerender_checks", "状态重渲次数"),
                 ("max_rerender_different_pixels", "重渲最大不同像素"), ("visual_boundary_rate_limits", "独立边界速率"))) + "。\n\n"
            if verifier.get("visual_boundary_rate_limits") == "FAIL":
                rate_audits = verifier.get("visual_rate_limit_audits", [])
                max_acc = [x.get("maximum_acceleration_A_deg_B_deg_C_mm_per_s2") for x in rate_audits]
                text += "**速率失败必须保留**：失去目标后仿真执行瞬停；独立审计最大 ABC 加速度为 `" + json.dumps(max_acc) + "` deg/s²、deg/s²、mm/s²。完整性 PASS 不代表运动速率 PASS，不能称为有界安全制动。\n\n"
        else:
            text += "独立 dataset_verification 尚未落盘：NOT_RUN；本脚本的文件/动作审计不能替代重渲或速率复核。\n\n"
        if case["status"] != "COMPLETE":
            text += "**未完成**：" + "; ".join(case["missing"]) + "。\n\n"
            if case.get("completed_phase_evidence"):
                text += "已落盘阶段仅作为局部证据，不等价于 episode 成功：\n\n" + markdown_table(["阶段文件", "阶段结果", "到物理标靶真实误差 µm"], [[x["file"], x["result"], number(x["true_error_um"])] for x in case["completed_phase_evidence"]]) + "\n\n"
        for ep in case["episodes"]:
            text += markdown_table(["目标", "尝试", "记录结果/原因", "真实误差 µm", "零速帧", "独立成功"],
                [[x["waypoint_id"], x["attempted"], f"{x['recorded_result']} / {x['reason']}", number(x["true_endpoint_error_um"]), x["settled_frames"], x["independent_success_lt20um_and_3_zero_frames"]] for x in ep["endpoints"]])
            text += "\n\n导出帧 visibility：`" + json.dumps(ep["visibility_frame_counts"], ensure_ascii=False) + "`；控制阶段计数：`" + json.dumps(ep["phase_report_counters"], ensure_ascii=False) + "`。这些计数不直接等价于成功率。\n\n"
            text += f"![Time series](figures/timeline_{name}_ep{ep['episode_index']:06d}.png)\n\n"
        sections.append(text)
    sections.append("## 已执行轨迹与协议可比性\n\n![Executed tip path from independent evaluation](figures/executed_strict_path.png)\n\n严格三点图取 `visual_servo/trajectory.csv` 的已执行 world XYZ，目标散点取验收报告；这是 independent evaluation 的执行轨迹，不是模型规划路径。展示跨点轴向回撤、转向、接近的实际日志。\n\n混合对照访问顺序为 P3→P2→P1，严格三点为 P1→P2→P3，且控制权限不同。不能把总运动时间差单独归因于严格视觉控制。\n\n[严格三点派生视频预览（若文件已生成）](strict_three_preview.mp4)：20 fps 展示预览；事件与端点的精确时间以原始 JSONL timestamp 为准。预览不替代 raw RGB、state/action 或真实误差验收。")
    fixture = cases["strict_occlusion_recovery"]["protocol"].get("occlusion_fixture") or {}
    events = fixture.get("events", [])
    period = cases["strict_occlusion_recovery"]["protocol"].get("dataset_sampling_period_s")
    sections.append("## 实体球计划事件\n\n" + markdown_table(["事件", "目标/phase", "相对phase开始frame", "持续frame", "名义持续s"],
        [[e["id"], e["waypoint_id"]+" / "+e["phase"], e["start_frame"], e["duration_frames"], number(e["duration_frames"]*period) if period is not None else "—"] for e in events])
        + f"\n\n实体球 radius={fixture.get('radius_mm', '未知')} mm，ray_fraction={fixture.get('ray_fraction', '未知')}。表格是计划 schedule；实际不可见帧和恢复控制状态见前述时间图与 CSV。")
    sections.append(f"## 设计与控制权限\n\n{targettable}\n\n眼底定义来自 `fundus_target_verification.json`：{metrics['target_evidence']['definition']['scope']}。椭球表面约束与后部帽区域是几何证明，不是实际视网膜组织或病灶测量。\n\n严格三点从零位按 P1→P2→P3 给定顺序逐个呈现目标。视觉观测构造轴向回撤、AB 方向对准、C 接近；不预计算目标 IK 路径。运行时禁止以 `target_task` / `nominal.inverse` 提供场景目标给控制；允许已知相机、RCM、尺寸、执行 Jacobian 与眼壁安全约束。场景 XYZ 仅用于放置 marker 和独立验收。当前实验没有验证同一画面中的多病灶识别与自主排序。混合对照包含场景解析转移，单独报告。\n\n遮挡 stop/recovery 配对协议检查：**{metrics['paired_protocol']['pass']}**。相同三目标、相同 reset、相同实体球 schedule；完整差异见 `metrics.json/paired_protocol`。reset 来源为 `retracted_start.json` 所指严格三点 P1 实际视觉回撤记录，不是目标 IK。实体球半径、ray fraction、事件开始和时长取各自 protocol；计划 schedule 不保证失败提前终止的 episode 实际看满全部遮挡。\n\n默认 stop 不启用记忆。opt-in 静态目标记忆需成熟的 3 次观测，受 1 s / 0.5 mm 限制，重获偏移超过 40 µm 拒绝；恢复最多 3 次，每次 4 s / 0.5 mm C 回撤。当前针尖必须始终为 fresh 观测。该设计假设相机固定、目标静态；这些限制是设计条件，非统计置信界或硬件性能。实际记忆帧数、恢复尝试/成功数从 phase report 读取。\n\n自然针遮挡的 `partial_needle/full_needle` 与实体球造成、无针反事实中仍不存在的 `not_visible` 分开计数。无针渲染只用于离线诊断，不能作为控制输入；可见像素和 event id 用于核查计划遮挡与实际曝光。")
    auditrows = [[s["label"], s["episodes"], s["frames"], s["dataset_png_count"], number(s["dataset_size_bytes"]/1024**2 if s["dataset_size_bytes"] is not None else None, 2), s["audit_pass"]] for s in summary]
    sections.append("## 数据规模与动作审计\n\n" + markdown_table(["实验", "episode", "JSONL行", "PNG", "全dataset MiB", "本脚本审计"], auditrows)
        + "\n\n名义采样 20 Hz；图像与当前 state 在 timestamp t 先于 action，action 是下一帧实际 state 的绝对 A/B/C 命令。阶段端点或失败观测可以产生短于 0.05 s 的间隔。逐 episode 审计 float32 action[:-1] 与 next-state 精确相等、terminal hold、PNG存在、policy/action NPZ 对齐和九个 perception 字段；具体误差与 dt 范围见 metrics.json。混合旧数据若无 perception 明示 LEGACY_UNAVAILABLE，不补写。\n\nPolicy 输入严格为 RGB路径、state、task；action 监督单列保存，phase/visibility/perception/场景坐标均为审计字段。RGB路径必须解码为图像，不能把文件名作为特征。paired no-image bundle 仅说明接口可做消融，当前没有训练/验证/测试 policy 对比。\n\n这是固定目标、给定顺序、少量相关 episode 的 pilot，尚非可训练数据集。不能按相邻帧随机拆分；未来按完整 episode/场景/目标布局分组拆分，并独立留出遮挡设置及失败案例，先扩大独立场景和运行数量。失败 episode 保留，禁止只筛选 PASS。")
    sections.append("## 原始图像证据\n\n![Raw RGB montage](figures/raw_frame_montage.png)\n\n截图是控制器所用 raw RGB PNG 的逐字节副本；拼图仅按显示比例缩放，未裁剪、未叠加标注或修改像素。每张原始图、SHA256、帧索引和可见性/控制状态见 rawframes.json。Viewer 的放大针/目标辅助不属于这些 raw 传感器截图。")
    sections.append(f"## 可复现、证据包与局限\n\n生成命令：`python {Path(__file__).relative_to(REPO)}`。仅写当前运行目录下 meeting/，不改源码和原始实验。`--allow-incomplete` 仅用于过程 review，材料会显著标明未完成。`metrics.json` 保存逐目标结果、动作审计、完整协议与源文件哈希，CSVs 方便复核；每张图另存 PDF。`slides.html` 为 6 页自包含 HTML，可离线打开，用方向键/空格切换并打印 PDF。\n\n`evidence_light.tar.gz`（除非 --no-archive）保留 reports/configs/CSVs、源代码快照与哈希、完整 JSONL/小型 NPZ、关键 raw PNG、脚本和组会材料；不含完整 RGB 图像库。完整数据位置见 DATASET_LINKS.md。源代码快照与采集时 hash 不一致时会单独标明，不能把当前快照冒充采集版本。\n\n没有硬件执行、物理制动/接触验证、真实 OCT 噪声、真实眼动/组织变形或临床精度结论。理想仿真深度、分割与固定标定是专家权限，RGB-only 学习能力尚未验证。单次少量 episode 无法估计泛化成功率或统计显著性。\n\n组织框架参考：[Supervisor-Skills / benchmark-paper-template]({TEMPLATE_URL})，固定 commit `{COMMIT}`。仅借用数据采集与分层评价组织方法；不是性能结果来源，不套用 LLM benchmark 结论，不引用不存在论文。")
    sections.append("## 深度传感器版本边界\n\n本报告四组多目标 run 在 OCT 包装接口接入前完成，使用相同的 MuJoCo 理想配准深度与几何分割权限。保存的传感器图像是 raw RGB；这些旧数据不含 raw OCT，不能追溯改称 OCT 闭环实验。后续 OCT 接口/传感器闭环验证须单独编号与报告，且仍需区分仿真理想深度与真实 OCT 噪声。")
    return "\n\n".join(sections) + "\n"


def html_table(headers, rows):
    esc = lambda x: html.escape("—" if x is None else str(x))
    return "<table><thead><tr>" + "".join(f"<th>{esc(x)}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>"


def make_slides(metrics, output):
    def img(relative, cls="figure"):
        path = output / relative
        if not path.is_file():
            return '<div class="missing">未完成：暂无对应最终图像证据</div>'
        return f'<img class="{cls}" alt="{html.escape(relative)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'
    summary = metrics["summary"]
    status = "最终落盘证据" if metrics["complete"] else "未完成草稿 · 不可报告最终 PASS"
    resulttable = html_table(["实验", "结果", "成功/计划", "成功/尝试", "帧"], [[s["label"], ", ".join(s["episode_results"]) or "未完成", f"{s['successful_targets']}/{s['requested_targets']}" if s["requested_targets"] else "—", f"{s['successful_targets']}/{s['attempted_targets']}" if s["attempted_targets"] else "—", s["frames"]] for s in summary])
    targets = html_table(["ID", "Robot XYZ (mm)", "Posterior Z", "Surface gap (µm)"], [[t["id"], ", ".join(f"{v:.6f}" for v in t["point_mm"]), f"{t['normalized_posterior_z']:.6f}", f"{t['surface_radial_gap_um']:.2e}"] for t in metrics["target_evidence"]["targets"]])
    pair = metrics["paired_protocol"]
    occlusionrows = []
    for name in ("strict_occlusion_stop", "strict_occlusion_recovery"):
        for ep in metrics["cases"][name]["episodes"]:
            c = ep["phase_report_counters"]
            rate = (metrics["cases"][name].get("independent_dataset_verification") or {}).get("visual_boundary_rate_limits", "NOT_RUN")
            occlusionrows.append([SHORT[name], ep["recorded_result"], ep["visibility_frame_counts"]["partial_needle"], ep["visibility_frame_counts"]["not_visible"], c["memory_frame_count"], c["recovery_attempt_count"], c["recovery_success_count"], rate])
    data = html_table(["实验", "episode", "frames", "MiB", "对齐/输入审计"], [[SHORT[s["case"]], s["episodes"], s["frames"], number(s["dataset_size_bytes"]/1024**2 if s["dataset_size_bytes"] is not None else None, 1), s["audit_pass"]] for s in summary])
    timelines = []
    for name in ("strict_occlusion_stop", "strict_occlusion_recovery"):
        ep = metrics["cases"][name]["episodes"]
        if ep:
            timelines.append(img(f"figures/timeline_{name}_ep{ep[0]['episode_index']:06d}.png", "timeline"))
    sections = [
        f'<h1>多目标严格视觉控制<br>与实体遮挡恢复</h1><p class="subtitle">2026-10-23 组会 · {status}</p>{resulttable}<p class="callout">目标成功：真实终点误差 &lt;20 µm + 至少 3 帧零命令速度。失败与未尝试目标保留。</p><p class="small">混合对照包含场景解析转移，不能称为严格视觉；当前目标按给定顺序逐个呈现。</p>',
        f'<h2>控制权限与眼底目标证据</h2><div class="flow">RGB-D 实测 → 轴向回撤 → AB 方向对准 → C 接近 → 停稳 / 独立验收</div><div class="two"><div style="width:45%">{targets}<p class="small">椭球后部表面代理，无独立视网膜层。场景 XYZ 仅放 marker 与独立验收。</p></div><div style="width:55%">{img("figures/executed_strict_path.png", "pathplot")}</div></div><p>禁止场景目标 IK 路径、目标姿态校准与 truth fallback；保留固定标定、执行 Jacobian、眼壁及运动约束。图为独立验收记录的已执行轨迹，不是规划路径。</p>',
        f'<h2>逐目标真实误差与停稳结果</h2>{img("figures/endpoint_errors.png")}<p class="small">每个数值取已完成 report；未运行目标无误差点。阶段虚拟目标误差不等于到物理标靶的真实误差。单次 pilot 不支持泛化成功率或显著性。</p>',
        '<h2>同场景遮挡对比：实际发生了什么</h2>' + html_table(["mode", "result", "partial needle帧", "not visible帧", "memory帧", "恢复尝试", "恢复成功", "独立速率"], occlusionrows) + f'<p class="small">配对协议审计：{pair["pass"]}。相同目标/reset/实体球 schedule，仅 mode 改变。提前停止使实际曝光分母不同。stop 瞬停的速率 FAIL 必须保留，不能称为有界安全制动；数据完整性与速率是不同检查。</p><div class="two">' + "".join(timelines) + '</div><p class="small">记忆依赖静态目标、固定相机与 fresh tip；记忆 1 s/0.5 mm，重获偏移 &gt;40 µm 拒绝；恢复最多3次，每次4 s/0.5 mm。</p>',
        f'<h2>Raw RGB：自然针遮挡与实体球不可见分开</h2>{img("figures/raw_frame_montage.png", "montage")}<p class="small">逐字节 raw PNG 副本；拼图仅显示缩放，无裁剪或像素覆盖。partial_needle ≠ not_visible；无针反事实仅为离线诊断。原图/帧索引/SHA256见 rawframes.json。</p>',
        f'<h2>可复现实验记录与下一阶段</h2>{data}<p>schema 2，名义20 Hz；t时刻 RGB/state → 下一绝对ABC命令。policy仅接收 RGB路径、state、task，审计标签不入输入；端点允许短间隔。</p><p>当前是相关 pilot episode，尚非可训练集。扩充独立场景/布局/运行，按episode或场景分组拆分，保留失败案例，再做RGB/no-image政策消融。</p><p class="small">没有硬件、真实OCT噪声、眼动/组织变形或临床精度结论。轻量证据包排除完整RGB库与视频，数据位置见 DATASET_LINKS.md。<a href="strict_three_preview.mp4">20fps派生预览（可选本地链接）</a>，精确时间以JSONL为准。</p><p class="small">组织框架：<a href="{TEMPLATE_URL}">Supervisor-Skills / benchmark-paper-template</a> @{COMMIT[:12]}；仅借采集/分层评价组织方法，不是性能来源。</p>',
    ]
    css = """*{box-sizing:border-box}body{margin:0;background:#182b3a;color:#172c3d;font-family:system-ui,-apple-system,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}section{display:none;width:min(1280px,100vw);height:min(720px,56.25vw);min-height:600px;margin:24px auto;background:#f8fafb;padding:40px 55px;position:relative;overflow:auto;border-top:9px solid #2c847b}section.active{display:block}h1{font-size:42px;line-height:1.18;margin:0 0 18px}h2{font-size:31px;margin:0 0 20px;color:#17384f}p{font-size:19px;line-height:1.45;margin:13px 0}.subtitle{color:#2c847b}.small{font-size:15px;color:#40576a}table{border-collapse:collapse;width:100%;font-size:16px;line-height:1.3}th,td{border-bottom:1px solid #cfdae1;padding:10px 9px;text-align:left}th{background:#e7eff3}.callout,.flow{padding:13px 17px;background:#e3efed;border-left:4px solid #2c847b}.flow{font-size:21px;margin-bottom:20px}.figure{display:block;width:100%;max-height:450px;object-fit:contain}.two{display:flex;gap:12px;align-items:flex-start}.timeline{width:calc(50% - 6px);max-height:330px;object-fit:contain}.montage{display:block;height:485px;width:100%;object-fit:contain}.footer{position:absolute;bottom:13px;right:30px;font-size:13px;color:#637b8c}.missing{color:#a3444e;padding:20px}nav{display:flex;justify-content:center;gap:16px;color:#fff;align-items:center;margin:14px}button{background:#f8fafb;color:#17384f;border:0;border-radius:5px;padding:8px 17px;cursor:pointer}a{color:#216e87}@page{size:landscape;margin:0}@media print{body{background:white}section,section.active{display:block;break-after:page;width:100vw;height:100vh;min-height:0;margin:0;border-top:7px solid #2c847b;padding:30px 40px;overflow:hidden}nav{display:none}.figure{max-height:61vh}.montage{height:66vh}p{font-size:16px}table{font-size:14px}h1{font-size:36px}h2{font-size:28px}.small{font-size:12px}}"""
    sections[-1] += '<p class="small">版本边界：四组run早于OCT包装接入，使用共同的MuJoCo理想配准深度；旧数据不含raw OCT。新OCT闭环验证单独报告。</p>'
    css += ".pathplot{display:block;width:100%;height:335px;object-fit:contain}#slide-4 .timeline{max-height:295px}"
    js = """let current=0;const pages=[...document.querySelectorAll('section')];function show(n){current=Math.max(0,Math.min(pages.length-1,n));pages.forEach((p,i)=>p.classList.toggle('active',i===current));document.getElementById('count').textContent=`${current+1} / ${pages.length}`;}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('print').onclick=()=>window.print();document.addEventListener('keydown',e=>{if(['ArrowRight','ArrowDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','ArrowUp'].includes(e.key)){e.preventDefault();show(current-1)}if(e.key==='Home')show(0);if(e.key==='End')show(pages.length-1)});show(0);"""
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2026-10-23 组会 · 严格视觉与遮挡</title><style>' + css + '</style></head><body>'
    document += "".join(f'<section id="slide-{i+1}">{body}<div class="footer">2026-10-23 · {status} · {i+1}/6</div></section>' for i, body in enumerate(sections))
    document += '<nav><button id="prev">上一页</button><span id="count"></span><button id="next">下一页</button><button id="print">打印 / PDF</button></nav><script>' + js + '</script></body></html>'
    (output / "slides.html").write_text(document, encoding="utf-8")


def export_csvs(metrics, records, output):
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(metrics["summary"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["summary"])
    endpoint_rows = []
    frame_rows = []
    for name in CASE_NAMES:
        for rec in records[name]:
            ep = rec["episode"]
            for x in ep["endpoints"]:
                endpoint_rows.append({"case": name, "episode_index": ep["episode_index"], **x})
            for row in rec["rows"]:
                frame_rows.append({"case": name, "episode_index": row["episode_index"], "frame_index": row["frame_index"], "timestamp_s": row["timestamp"], "waypoint_id": row.get("waypoint_id"), "phase": row.get("phase"),
                    **{k: row.get("visibility", {}).get(k) for k in ("target_pixels_visible", "target_occlusion_state", "target_pixels_unoccluded", "counterfactual_no_needle_rendered")},
                    **{k: row.get("perception", {}).get(k) for k in sorted(PERCEPTION_FIELDS)}})
    for filename, rows in (("endpoint_metrics.csv", endpoint_rows), ("frame_metrics.csv", frame_rows)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


def snapshot_evidence(output):
    evidence = output / "evidence"
    sources = []
    for name in CASE_NAMES:
        root = BASE / name
        for pattern in ("*.json", "*.csv", "*.md", "dataset/manifest.json", "dataset_verification.json", "visual_servo/trajectory.csv", "visual_servo/*/*/visual_servo_report.json"):
            for path in root.glob(pattern):
                if path.is_file():
                    dest = evidence / name / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
        # JSONL and small policy/action bundles retain aligned metadata without RGB.
        for pattern in ("dataset/data/*.jsonl", "dataset/policy_inputs/*.npz", "dataset/policy_inputs_no_image/*.npz", "dataset/action_labels/*.npz"):
            for path in root.glob(pattern):
                dest = evidence / name / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        hash_path = root / "source_sha256.json"
        if not hash_path.is_file():
            continue
        for relative, declared in read_json(hash_path).items():
            path = root / "source_snapshot" / relative
            origin = "run source_snapshot"
            if not path.is_file():
                path = REPO / relative
                origin = "current repository at record generation"
            if not path.is_file() or not path.resolve().is_relative_to(REPO):
                sources.append({"case": name, "source": relative, "declared_run_sha256": declared, "snapshot": "UNAVAILABLE"})
                continue
            dest = evidence / name / "source_snapshot" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            actual = digest(dest)
            sources.append({"case": name, "source": relative, "declared_run_sha256": declared,
                            "snapshot_sha256": actual, "matches_declared_run_hash": actual == declared, "origin": origin,
                            "snapshot": str(dest.relative_to(output))})
    for path in BASE.glob("*.json"):
        dest = evidence / "run_configuration" / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    # Additional verifier reports can be added by the parent without this script
    # interpreting a command's expected failure exit as an experimental success.
    for subdir in ("collection_checks", "control_checks", "occlusion_checks"):
        for path in (BASE / subdir).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".csv", ".md", ".txt") and path.stat().st_size < 4 * 1024 * 1024:
                dest = evidence / subdir / path.relative_to(BASE / subdir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-incomplete", action="store_true", help="write explicitly incomplete review materials; never infer missing PASS")
    parser.add_argument("--no-archive", action="store_true", help="omit evidence_light.tar.gz")
    args = parser.parse_args()
    cases, records = {}, {}
    for name in CASE_NAMES:
        cases[name], records[name] = load_case(name)
    incomplete = [n for n, c in cases.items() if c["status"] != "COMPLETE"]
    if incomplete and not args.allow_incomplete:
        messages = [f"{n}: {'; '.join(cases[n]['missing'])}" for n in incomplete]
        raise SystemExit("Final evidence is incomplete; nothing written. Wait for final collectors or use --allow-incomplete for review only.\n" + "\n".join(messages))
    audits_failed = [f"{n}/ep{e['episode_index']}: {'; '.join(e['audit']['issues'])}" for n, c in cases.items() for e in c["episodes"] if not e["audit"]["pass"]]
    if audits_failed:
        raise SystemExit("Dataset audit failed; nothing written.\n" + "\n".join(audits_failed))
    paired = paired_protocol(cases)
    if not paired["pass"]:
        raise SystemExit("Paired protocol mismatch; nothing written.\n" + json.dumps(paired, ensure_ascii=False, indent=2))
    output = BASE / "meeting"
    output.mkdir(parents=True, exist_ok=True)
    # Remove only this script's generated files, so a partial review cannot leave
    # old completed slides/plots or stale raw frames in a later evidence archive.
    for name in ("figures", "rawframes", "evidence"):
        path = output / name
        if path.exists():
            shutil.rmtree(path)
    (output / "figures").mkdir()
    archive = output / "evidence_light.tar.gz"
    if archive.exists():
        archive.unlink()
    metrics = {"schema_version": 1, "meeting_date": "2026-10-23", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "complete": not incomplete, "incomplete_cases": incomplete,
               "success_definition": "true endpoint error <20 um, report PASS and terminal stop validated, >=3 settled zero-command-velocity frames",
               "summary": summary_rows(cases), "cases": cases, "paired_protocol": paired,
               "target_evidence": target_evidence(),
               "framework_reference": {"url": TEMPLATE_URL, "commit": COMMIT, "role": "collection and stratified evaluation organization only; not a performance source"},
               "generation": {"script": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)), "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "argv": sys.argv},
               "limitations": ["kinematic simulation only", "ideal depth and segmentation privileges", "all four multitarget runs predate OCT wrapper integration; common MuJoCo ideal registered depth, no raw OCT", "no hardware or real OCT noise validation", "static target/fixed camera assumption", "few correlated pilot episodes", "no policy training or image ablation result", "sequential scripted target presentation, no simultaneous lesion recognition/order selection"]}
    make_endpoint_plot(cases, output / "figures")
    metrics["executed_path_evidence"] = make_executed_path(cases, output)
    for name in CASE_NAMES:
        for record in records[name]:
            make_timeline(name, record, output / "figures")
    metrics["rawframes"] = choose_raw_frames(records, output)
    export_csvs(metrics, records, output)
    metrics["source_snapshot_audit"] = snapshot_evidence(output)
    write_json(output / "metrics.json", metrics)
    (output / "report.md").write_text(report_text(metrics), encoding="utf-8")
    links = "# Full datasets (not included in the light archive)\n\nRaw PNG datasets are retained at the following absolute local paths. Metadata copies inside the archive preserve original relative image paths and require the corresponding full dataset to decode all images.\n\n"
    links += "\n".join(f"- {name}: [{case['dataset_path']}]({case['dataset_path']}) — {case['status']}" for name, case in cases.items())
    (output / "DATASET_LINKS.md").write_text(links + "\n", encoding="utf-8")
    make_slides(metrics, output)
    sha_manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("SHA256SUMS.json", "evidence_light.tar.gz") and p.suffix.lower() not in (".mp4", ".mov", ".avi")}
    write_json(output / "SHA256SUMS.json", sha_manifest)
    if not args.no_archive:
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path != archive and path.suffix.lower() not in (".mp4", ".mov", ".avi"):
                    bundle.add(path, arcname=str(Path("meeting") / path.relative_to(output)), recursive=False)
    print(json.dumps({"output": str(output), "complete": metrics["complete"], "incomplete_cases": incomplete,
                      "summary": metrics["summary"], "archive": str(archive) if not args.no_archive else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
