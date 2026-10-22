"""Aligned RGB/state/absolute-position-action episodes for offline simulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rcm_reference import save_png


TASK_REGIMES = ("registered_coordinate_prior", "visual_current_target")
POLICY_INPUT_FIELDS = ("observation.images.microscope_rgb", "observation.state", "task")
NO_IMAGE_INPUT_FIELDS = ("observation.state", "task")
VISIBILITY_CONDITIONS = ("visible", "needle_occluded", "other_visibility_issue", "unknown")


def make_policy_task(task_regime, solutions, fundus_surface=False):
    """Declare student inputs separately from the privileged expert's inputs."""
    if task_regime not in TASK_REGIMES:
        raise ValueError(f"Unknown task regime: {task_regime}")
    if task_regime == "registered_coordinate_prior":
        task = ("根据假设已配准至机器人坐标系的目标三维坐标先验和当前显微镜 RGB 观测导航至眼底目标并停稳；"
                "本数据的坐标由仿真目标输入给出，不是实测 OCT。机器人坐标（mm）："+
                "; ".join(f"{s['id']}={s['requested_robot_mm']}" for s in sorted(solutions, key=lambda s: s["id"])))
    else:
        # Identical across target sets: no goal IDs, coordinates, order or seed.
        task = ("根据当前显微镜 RGB 图像和轴位置反馈，将针尖导航至当前呈现的眼底标靶中心并停稳；"
                "持续依据视觉反馈修正动作；不提供目标三维坐标，若标靶或针尖不可观测则停止。")
    if fundus_surface:
        task = ("目标位于眼底曲面（仿真眼球后部表面）；点间必须先沿针轴回撤，再在眼内转向，"
                "最后沿固定针轴接近，禁止贴着眼底横移。"+task)
    return task


class EpisodeWriter:
    def __init__(self, root, episode_index, task, method, seed, order):
        self.root = Path(root)
        self.index = int(episode_index)
        self.task, self.method, self.seed, self.order = task, method, seed, order
        self.image_dir = self.root / "images" / f"episode_{self.index:06d}"
        self.image_dir.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.write_wall_s = 0.0
        self.policy_input_dir = self.root / "policy_inputs"
        self.policy_no_image_dir = self.root / "policy_inputs_no_image"
        self.action_label_dir = self.root / "action_labels"

    def add(self, timestamp, state, action, waypoint_id, rgb, terminal=False, phase="motion",
            visibility=None):
        import time
        started = time.perf_counter()
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        rgb = np.asarray(rgb)
        if state.shape != (3,) or action.shape != (3,) or not np.isfinite([state, action]).all():
            raise ValueError("Dataset state/action must be finite vectors of length 3")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Dataset image must be uint8 HWC RGB")
        if self.records and timestamp <= self.records[-1]["timestamp"]:
            raise ValueError("Episode timestamps must increase strictly")
        frame = len(self.records)
        image_path = self.image_dir / f"frame_{frame:06d}.png"
        save_png(image_path, rgb)
        visibility = _normalize_visibility(visibility)
        self.records.append({
            "observation.images.microscope_rgb": image_path.relative_to(self.root).as_posix(),
            "observation.state": state.tolist(), "action": action.tolist(),
            "task": self.task, "timestamp": float(timestamp),
            "episode_index": self.index, "frame_index": frame,
            "waypoint_id": waypoint_id, "is_terminal": bool(terminal),
            "phase": phase,
            "visibility": visibility,
        })
        self.write_wall_s += time.perf_counter()-started

    def finish(self, result, target_success_summary=None, stop_constraint_summary=None):
        data_dir = self.root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        stem = data_dir / f"episode_{self.index:06d}"
        with stem.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
        # Strings use Unicode arrays; loading never requires allow_pickle=True.
        np.savez_compressed(stem.with_suffix(".npz"), **{
            "observation.images.microscope_rgb": np.asarray([
                r["observation.images.microscope_rgb"] for r in self.records], dtype=str),
            "observation.state": np.asarray([r["observation.state"] for r in self.records], dtype=np.float32).reshape(-1, 3),
            "action": np.asarray([r["action"] for r in self.records], dtype=np.float32).reshape(-1, 3),
            "task": np.asarray([self.task]*len(self.records), dtype=str),
            "timestamp": np.asarray([r["timestamp"] for r in self.records], dtype=np.float64),
            "episode_index": np.full(len(self.records), self.index, dtype=np.int64),
            "frame_index": np.arange(len(self.records), dtype=np.int64),
            "waypoint_id": np.asarray([r["waypoint_id"] for r in self.records], dtype=str),
            "is_terminal": np.asarray([r["is_terminal"] for r in self.records], dtype=np.bool_),
            "phase": np.asarray([r["phase"] for r in self.records], dtype=str),
            "visibility.known": np.asarray([r["visibility"]["known"] for r in self.records], dtype=np.bool_),
            "visibility.target_pixels_visible": np.asarray([
                r["visibility"]["target_pixels_visible"] for r in self.records], dtype=np.int32),
            "visibility.target_pixels_unoccluded": np.asarray([
                r["visibility"]["target_pixels_unoccluded"] for r in self.records], dtype=np.int32),
            "visibility.target_pixels_occluded_by_needle": np.asarray([
                r["visibility"]["target_pixels_occluded_by_needle"] for r in self.records], dtype=np.int32),
            "visibility.target_occlusion_fraction_by_needle": np.asarray([
                -1. if r["visibility"]["target_occlusion_fraction_by_needle"] is None
                else r["visibility"]["target_occlusion_fraction_by_needle"] for r in self.records], dtype=np.float32),
            "visibility.target_occlusion_state": np.asarray([
                r["visibility"]["target_occlusion_state"] for r in self.records], dtype=str),
            "visibility.tip_pixels_visible": np.asarray([
                r["visibility"]["tip_pixels_visible"] for r in self.records], dtype=np.int32),
            "visibility.needle_pixels_visible": np.asarray([
                r["visibility"]["needle_pixels_visible"] for r in self.records], dtype=np.int32),
            "visibility.tip_fully_occluded": np.asarray([
                r["visibility"]["tip_fully_occluded"] for r in self.records], dtype=np.bool_),
            "visibility.counterfactual_no_needle_rendered": np.asarray([
                r["visibility"]["counterfactual_no_needle_rendered"] for r in self.records], dtype=np.bool_),
        })
        self.policy_input_dir.mkdir(parents=True, exist_ok=True)
        self.policy_no_image_dir.mkdir(parents=True, exist_ok=True)
        self.action_label_dir.mkdir(parents=True, exist_ok=True)
        policy_input_path = self.policy_input_dir/f"episode_{self.index:06d}.npz"
        policy_no_image_path = self.policy_no_image_dir/f"episode_{self.index:06d}.npz"
        action_label_path = self.action_label_dir/f"episode_{self.index:06d}.npz"
        # Keep the model-facing feature set deliberately narrow. Target IDs,
        # waypoint order, phase, visibility labels, and scene coordinates stay
        # in the audit dataset, not the policy input bundle.
        np.savez_compressed(policy_input_path, **{
            "observation.images.microscope_rgb": np.asarray([
                r["observation.images.microscope_rgb"] for r in self.records], dtype=str),
            "observation.state": np.asarray([r["observation.state"] for r in self.records], dtype=np.float32).reshape(-1, 3),
            "task": np.asarray([self.task]*len(self.records), dtype=str),
        })
        np.savez_compressed(policy_no_image_path, **{
            "observation.state": np.asarray([r["observation.state"] for r in self.records], dtype=np.float32).reshape(-1, 3),
            "task": np.asarray([self.task]*len(self.records), dtype=str),
        })
        np.savez_compressed(action_label_path, **{
            "action": np.asarray([r["action"] for r in self.records], dtype=np.float32).reshape(-1, 3),
        })
        visibility_summary = summarize_visibility(self.records)
        return {"episode_index": self.index, "method": self.method, "seed": self.seed,
                "visit_order": self.order, "result": result, "frames": len(self.records),
                "data": stem.with_suffix(".npz").relative_to(self.root).as_posix(),
                "jsonl": stem.with_suffix(".jsonl").relative_to(self.root).as_posix(),
                "policy_inputs": policy_input_path.relative_to(self.root).as_posix(),
                "policy_inputs_no_image": policy_no_image_path.relative_to(self.root).as_posix(),
                "action_labels": action_label_path.relative_to(self.root).as_posix(),
                "image_write_wall_s": self.write_wall_s,
                "visibility_summary": visibility_summary,
                "target_success_by_visibility": target_success_summary,
                "stop_constraint_summary": stop_constraint_summary}


def _normalize_visibility(visibility):
    """Use explicit sentinels when a frame has no segmentation observation."""
    if visibility is None:
        return {"known": False, "target_pixels_visible": -1, "target_pixels_unoccluded": -1,
                "target_pixels_occluded_by_needle": -1,
                "target_occlusion_fraction_by_needle": None, "target_occlusion_state": "unknown",
                "tip_pixels_visible": -1, "needle_pixels_visible": -1,
                "tip_fully_occluded": False, "counterfactual_no_needle_rendered": False}
    required = ("target_pixels_visible", "target_pixels_unoccluded",
                "target_pixels_occluded_by_needle", "target_occlusion_fraction_by_needle",
                "target_occlusion_state", "tip_pixels_visible", "needle_pixels_visible",
                "tip_fully_occluded", "counterfactual_no_needle_rendered")
    missing = [key for key in required if key not in visibility]
    if missing:
        raise ValueError(f"Incomplete visibility metadata: {missing}")
    result = {"known": True, **{key: visibility[key] for key in required}}
    for key in ("target_pixels_visible", "target_pixels_unoccluded",
                "target_pixels_occluded_by_needle", "tip_pixels_visible", "needle_pixels_visible"):
        if not isinstance(result[key], (int, np.integer)) or result[key] < 0:
            raise ValueError(f"Visibility field {key} must be a non-negative integer")
        result[key] = int(result[key])
    fraction = result["target_occlusion_fraction_by_needle"]
    if fraction is not None and (not np.isfinite(fraction) or not 0. <= float(fraction) <= 1.):
        raise ValueError("target_occlusion_fraction_by_needle must be null or in [0, 1]")
    result["target_occlusion_fraction_by_needle"] = None if fraction is None else float(fraction)
    if result["target_occlusion_state"] not in ("clear", "partial_needle", "full_needle", "not_visible"):
        raise ValueError("Unsupported target_occlusion_state")
    result["tip_fully_occluded"] = bool(result["tip_fully_occluded"])
    result["counterfactual_no_needle_rendered"] = bool(result["counterfactual_no_needle_rendered"])
    visible = result["target_pixels_visible"]
    unoccluded = result["target_pixels_unoccluded"]
    occluded = result["target_pixels_occluded_by_needle"]
    if visible > unoccluded or visible+occluded != unoccluded:
        raise ValueError("Target visibility pixel counts are inconsistent")
    return result


def summarize_visibility(records):
    known = [record["visibility"] for record in records if record["visibility"]["known"]]
    states = {state: sum(item["target_occlusion_state"] == state for item in known)
              for state in ("clear", "partial_needle", "full_needle", "not_visible")}
    fractions = [item["target_occlusion_fraction_by_needle"] for item in known
                 if item["target_occlusion_fraction_by_needle"] is not None]
    return {"known_frames": len(known), "unknown_frames": len(records)-len(known),
            "target_occlusion_state_frames": states,
            "frames_with_needle_occlusion": states["partial_needle"]+states["full_needle"],
            "maximum_target_occlusion_fraction_by_needle": max(fractions, default=None),
            "source": "MuJoCo geometry segmentation; offline no-needle counterfactual, not controller input"}


def summarize_target_success_by_visibility(records, waypoint_results):
    """Summarize target-visit success for every visibility condition observed.

    A target visit may contribute to multiple exposure groups when the target
    is visible at some times and needle-occluded at others. These are
    target-visit rates, not frame rates.
    """
    frames_by_waypoint = {}
    for record in records:
        phase = record.get("phase")
        if phase != "hold" or record.get("is_terminal", False):
            frames_by_waypoint.setdefault(record["waypoint_id"], []).append(record)

    counts = {condition: {"target_visits": 0, "successful_target_visits": 0}
              for condition in VISIBILITY_CONDITIONS}
    target_visits = []
    attempted = 0
    not_attempted = 0
    for result in waypoint_results:
        waypoint_id = result["waypoint_id"]
        did_attempt = result.get("result") != "NOT_RUN"
        if did_attempt:
            attempted += 1
        else:
            not_attempted += 1

        frames = frames_by_waypoint.get(waypoint_id, [])
        states = [record.get("visibility", {}).get("target_occlusion_state", "unknown")
                  for record in frames]
        exposures = set()
        if "clear" in states:
            exposures.add("visible")
        if any(state in ("partial_needle", "full_needle") for state in states):
            exposures.add("needle_occluded")
        if "not_visible" in states:
            exposures.add("other_visibility_issue")
        if not states or "unknown" in states:
            exposures.add("unknown")
        if states and not exposures:
            exposures.add("unknown")
        if "needle_occluded" in exposures:
            exclusive_class = "needle_occluded"
        elif "other_visibility_issue" in exposures:
            exclusive_class = "other_visibility_issue"
        elif "visible" in exposures and "unknown" not in exposures:
            exclusive_class = "visible"
        else:
            exclusive_class = "unknown"

        success = bool(did_attempt and result.get("result") == "PASS"
                       and result.get("terminal_stop_validated", False))
        if did_attempt:
            for condition in exposures:
                counts[condition]["target_visits"] += 1
                counts[condition]["successful_target_visits"] += int(success)
        frame_counts = {state: states.count(state) for state in
                        ("clear", "partial_needle", "full_needle", "not_visible", "unknown")}
        visibility_issue_counts = {
            "target_not_visible_without_needle_frames": frame_counts["not_visible"],
            "tip_fully_occluded_frames": sum(bool(record.get("visibility", {}).get("tip_fully_occluded", False))
                                              for record in frames),
            "unknown_visibility_frames": frame_counts["unknown"],
            "approach_frames": len(frames),
        }
        target_visits.append({
            "waypoint_id": waypoint_id,
            "attempted": did_attempt,
            "result": result.get("result", "UNKNOWN"),
            "reason": result.get("reason"),
            "endpoint_error_um": result.get("tip_error_um"),
            "terminal_stop_validated": bool(result.get("terminal_stop_validated", False)),
            "success_with_stop": success,
            "visibility_exposure": [name for name in VISIBILITY_CONDITIONS if name in exposures],
            "exclusive_visibility_class": exclusive_class,
            "visibility_frame_counts": frame_counts,
            "visibility_issue_frame_counts": visibility_issue_counts,
        })

    for row in counts.values():
        row["success_rate"] = (row["successful_target_visits"]/row["target_visits"]
                               if row["target_visits"] else None)
    return {
        "unit": "attempted target visit, not frame",
        "success_definition": "endpoint result PASS and terminal stop validated",
        "condition_definition": (
            "Conditions are exposure-based at target-visit level and may overlap. visible means at least "
            "one clear frame; needle_occluded means at least one partial/full needle-occlusion frame; "
            "other_visibility_issue means target absent even with the needle hidden; unknown means "
            "visibility could not be classified."),
        "exclusive_visit_class_definition": (
            "An auxiliary single-label summary prioritizes needle_occluded, then other_visibility_issue, "
            "then visible when every sampled frame is clear, otherwise unknown. It is not used for "
            "condition_exposure success rates; exposure groups may overlap."),
        "attempted_target_visits": attempted,
        "not_attempted_target_visits": not_attempted,
        "condition_exposure": counts,
        "target_visits": target_visits,
        "source": "frame visibility labels from MuJoCo segmentation plus per-waypoint endpoint and stop reports",
    }


def write_manifest(root, episodes, image_shape, sample_period_s=0.05, task_regime="visual_current_target"):
    sample_period_s = float(sample_period_s)
    if not np.isfinite(sample_period_s) or sample_period_s <= 0.:
        raise ValueError("sample_period_s must be finite and positive")
    if task_regime not in TASK_REGIMES:
        raise ValueError(f"Unknown task regime: {task_regime}")
    manifest = {
        "schema_version": 2,
        "format": "PNG RGB + JSONL metadata + NumPy NPZ (not a LeRobot container)",
        "source": "MuJoCo kinematic simulation; no hardware measurements",
        "state_semantics": "actual current MuJoCo A/B/C task position before applying action",
        "action_semantics": "next commanded absolute A/B/C task position after trajectory/rate constraints; not final mission goal",
        "units": ["degree", "degree", "millimeter"],
        "timestamp_semantics": "simulation seconds since episode start; intervals may be shorter at planner/phase endpoints",
        "nominal_sample_period_s": sample_period_s,
        "nominal_sampling_frequency_hz": 1./sample_period_s,
        "image_semantics": "raw eye_internal RGB, same render groups as visual controller; no crosshair overlay or synthetic occlusion",
        "visibility_semantics": "Per-frame counts from physical MuJoCo segmentation. Needle-caused occlusion is measured with a diagnostic render that temporarily hides needle geoms; this counterfactual is offline metadata only and is never passed to navigation/control. No synthetic noise or pasted masks are used.",
        "task_regime": task_regime,
        "policy_interface": {
            "contract_version": 1,
            "input_bundle": "policy_inputs/episode_*.npz",
            "input_fields": list(POLICY_INPUT_FIELDS),
            "no_image_ablation_bundle": "policy_inputs_no_image/episode_*.npz",
            "no_image_ablation_fields": list(NO_IMAGE_INPUT_FIELDS),
            "action_supervision": "action_labels/episode_*.npz; row-aligned with policy_inputs",
            "excluded_from_policy_inputs": ["action", "timestamp", "episode_index", "frame_index",
                                             "is_terminal", "waypoint_id", "phase", "visibility.*", "scene truth"],
            "goal_coordinates_in_task": task_regime == "registered_coordinate_prior",
            "goal_coordinate_visibility": ("registered robot-frame coordinates are present in task text; simulated, not measured OCT"
                                            if task_regime == "registered_coordinate_prior"
                                            else "target coordinates and target IDs are absent from task text"),
            "image_loading": "Decode paths relative to dataset root into RGB pixels; never tokenize paths or use filenames as features",
            "audit_bundle": "data/episode_*.npz and JSONL contain additional diagnostic labels; do not use those columns as policy features",
        },
        "alignment": "image and state at timestamp t precede action; action equals next frame state within float32 tolerance",
        "terminal_action": "hold current position; does not establish hardware braking behavior",
        "fields": {
            "observation.images.microscope_rgb": {"dtype": "uint8", "shape": image_shape, "storage": "lossless PNG path in JSONL/NPZ"},
            "observation.state": {"dtype": "float32", "shape": [3]},
            "action": {"dtype": "float32", "shape": [3]},
            "task": {"dtype": "string"}, "timestamp": {"dtype": "float64", "unit": "s"},
            "episode_index": {"dtype": "int64"}, "frame_index": {"dtype": "int64"},
            "phase": {"dtype": "string", "meaning": "phase of the outgoing action: retract, transfer, approach, visual_servo, or hold"},
            "visibility.known": {"dtype": "bool", "meaning": "false when segmentation was unavailable; remaining fields use unknown sentinels"},
            "visibility.target_pixels_visible": {"dtype": "int32", "unknown_sentinel": -1},
            "visibility.target_pixels_unoccluded": {"dtype": "int32", "unknown_sentinel": -1},
            "visibility.target_pixels_occluded_by_needle": {"dtype": "int32", "unknown_sentinel": -1},
            "visibility.target_occlusion_fraction_by_needle": {
                "dtype": "float32", "unknown_sentinel": -1.0,
                "undefined_when": "unoccluded target pixel count is zero; JSONL stores null"},
            "visibility.target_occlusion_state": {"dtype": "string", "values": ["clear", "partial_needle", "full_needle", "not_visible", "unknown"]},
            "visibility.tip_pixels_visible": {"dtype": "int32", "unknown_sentinel": -1},
            "visibility.needle_pixels_visible": {"dtype": "int32", "unknown_sentinel": -1},
            "visibility.tip_fully_occluded": {"dtype": "bool", "unknown_sentinel": False},
            "visibility.counterfactual_no_needle_rendered": {"dtype": "bool", "meaning": "true when a second diagnostic segmentation hid needle geoms"},
        },
        "episodes": episodes,
        "total_frames": sum(e["frames"] for e in episodes),
    }
    (Path(root)/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return manifest
