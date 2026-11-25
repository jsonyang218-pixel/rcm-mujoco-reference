#!/usr/bin/env python3
"""Counterfactual C-label audit and isolated dataset preparation."""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from lerobot_training.dataset_contract import valid_window_indices
from rcm.scene_v9 import RCMSceneV9
from rcm.visual_servo import VisualServoController, visual_servo_settings
from scripts.datasets.convert_plan2_training import SOURCES

UNIT = np.array([np.pi / 180.0, np.pi / 180.0, 0.001], dtype=np.float64)
DT = 0.05
EPS_MM = 1e-6
REPLAY_ATOL = np.array([0.0001, 0.0001, 0.00005])
ACTION_LIMIT = np.array([0.2897999882698059, 0.14434358477592468, 0.04271206632256508])


@lru_cache(maxsize=None)
def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_episode(source_root: Path, episode: dict) -> dict:
    current = episode
    seen = set()
    while current.get("target_robot_mm") is None and current.get("source_dataset"):
        source_dir = source_root / Path(current["source_dataset"]).name
        source_index = int(current.get("source_episode_index", -1))
        key = (source_dir, source_index)
        if key in seen or source_index < 0:
            return {}
        seen.add(key)
        metadata_path = source_dir / "metadata.json"
        if not metadata_path.is_file():
            return {}
        episodes = _json(metadata_path).get("episodes", [])
        if source_index >= len(episodes):
            return {}
        current = episodes[source_index]
    return current


def _dataset_context(dataset_dir: Path):
    metadata = _json(dataset_dir / "metadata.json")
    with np.load(dataset_dir / "data" / "dataset.npz") as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    episode_ids = np.asarray(arrays["episode_index"], dtype=np.int64)
    if not np.all(np.diff(episode_ids) >= 0):
        raise ValueError("episode_index must be ordered for source alignment")
    source_root = dataset_dir.parent
    sources = metadata["source_datasets"]
    source_cut = int(sources[1]["episode_offset"])
    source_metadata = [
        _json(source_root / Path(item["directory"]).name / "metadata.json")
        for item in sources
    ]
    episode_info = []
    for index, merged_episode in enumerate(metadata["episodes"]):
        source_slot = 0 if index < source_cut else 1
        source_index = (
            index
            if source_slot == 0
            else index - source_cut
        )
        constituent = source_metadata[source_slot]["episodes"][source_index]
        if int(merged_episode["num_frames"]) != int(constituent["num_frames"]):
            raise ValueError(f"episode {index} frame count differs from source")
        group = (
            "Original"
            if source_slot == 0
            else Path(constituent.get("source_dataset", "D4")).name
            .removeprefix("pilot_").split("_")[0].upper()
        )
        target_episode = _source_episode(source_root, constituent)
        target_robot = target_episode.get("target_robot_mm")
        episode_info.append({
            "group": group,
            "action_source": constituent.get("action_source", "executed"),
            "source_slot": source_slot,
            "source_index": int(constituent.get("source_episode_index", source_index)),
            "source_name": Path(constituent.get("source_dataset", "")).name,
            "target_robot_mm": target_robot,
        })
    if int(episode_ids.max()) + 1 != len(episode_info):
        raise ValueError("merged episode metadata does not cover all frame ids")
    return metadata, arrays, episode_info


@lru_cache(maxsize=128)
def _raw_rows(root, source_slot, source_name, index):
    if source_slot == 0:
        shard = source_name.removesuffix("_224").rsplit("_", 1)[-1]
        directory = root / "runs/20261116/online_dagger_dit_rf_round2_near_target_shards" / shard
    elif source_name == "d4_full_shard_a_unique":
        directory = root / "data/plan2_new_train_clean_runs_20261119/d4_full_remaining_shard_a_unique"
    else:
        directory = root / SOURCES[source_name]
    paths = sorted(directory.glob("episode_*"))
    if index >= len(paths):
        raise FileNotFoundError(f"raw episode {index} under {directory}")
    path = paths[index] / "dagger_samples.jsonl"
    with path.open() as stream:
        return path, [json.loads(line) for line in stream if line.strip()]


def _teacher_inputs(raw, index):
    current = raw[index]
    task = np.asarray(current["state_A_deg_B_deg_C_mm"]) * UNIT
    if not index:
        if not np.isclose(current["timestamp_s"], 0.0):
            raise ValueError("missing pre-roll velocity at first frame")
        return task, [("motor", np.zeros(3), task)]
    dt = current["timestamp_s"] - raw[index - 1]["timestamp_s"]
    if not np.isclose(dt, DT, rtol=1e-3, atol=1e-6):
        raise ValueError("missing continuous predecessor in raw data")
    previous = np.asarray(raw[index - 1]["state_A_deg_B_deg_C_mm"]) * UNIT
    candidates = [("motor", (task - previous) / dt, task)]
    # The collector advances command_q independently during a takeover. Its
    # unclipped teacher label plus state is exactly the next command_q.
    if raw[index - 1]["executor"] == "teacher" and current["executor"] == "teacher":
        delta = np.asarray(raw[index - 1]["teacher_action_deg_deg_mm"])
        if np.any(np.abs(delta) >= ACTION_LIMIT - 1e-9):
            raise ValueError("clipped predecessor hides the teacher command")
        command = previous + delta * UNIT
        previous_base = previous
        if index >= 2 and raw[index - 2]["executor"] == "teacher":
            prior = np.asarray(raw[index - 2]["teacher_action_deg_deg_mm"])
            if np.any(np.abs(prior) >= ACTION_LIMIT - 1e-9):
                raise ValueError("clipped predecessor hides the teacher velocity")
            previous_base = np.asarray(raw[index - 2]["state_A_deg_B_deg_C_mm"]) * UNIT + prior * UNIT
        candidates.append(("takeover", (command - previous_base) / dt, command))
    return task, candidates


def _target_states(scene, arrays, episode_info):
    episode_ids = arrays["episode_index"].astype(np.int64)
    states = arrays["state"]
    counts = np.bincount(episode_ids, minlength=len(episode_info))
    starts = np.r_[0, np.cumsum(counts[:-1])]
    targets = np.full((len(episode_info), 3), np.nan, dtype=np.float64)
    for episode_index, item in enumerate(episode_info):
        target_robot = item["target_robot_mm"]
        if target_robot is None:
            continue
        target_world = scene.robot_to_world(np.asarray(target_robot, dtype=np.float64))
        scene.set_target_world(target_world)
        scene.set_task(states[starts[episode_index]] * UNIT)
        target_task = scene.target_task(validate=False)
        scene.set_task(target_task)
        if np.linalg.norm(scene.site_mm("needle_tip") - target_world) >= 0.001:
            raise ValueError(f"episode {episode_index} target inverse check failed")
        targets[episode_index] = target_task / UNIT
    return targets


def _counterfactual(teacher, scene, task, observation, previous_velocity, command_task):
    actual = teacher.geometry_action_from_observation(
        task,
        observation,
        previous_velocity=previous_velocity,
        command_q=command_task,
    )
    if not actual["valid"]:
        return {"valid": False, "reason": actual.get("reason")}

    navigation = scene.config["navigation"]
    vmax = np.asarray(
        navigation.get("max_velocity_A_deg_B_deg_C_mm_per_s", [4.0, 2.0, 0.6])
    ) * UNIT
    amax = np.asarray(
        navigation.get(
            "max_acceleration_A_deg_B_deg_C_mm_per_s2", [10.0, 5.0, 1.5]
        )
    ) * UNIT
    geometry = observation["geometry"]
    measured_tip = np.asarray(geometry["tip_world_mm"], dtype=np.float64)
    measured_target = teacher._visual_goal(
        geometry["target_world_mm"], measured_tip, observation
    )
    jacobian = teacher._model_jacobian(task)
    u, singular, vt = np.linalg.svd(jacobian * vmax, full_matrices=False)
    damping = (singular[0] * 1e-4) ** 2
    solve = vmax[:, None] * (
        (vt.T * (singular / (singular**2 + damping))) @ u.T
    )
    correction = solve @ (measured_target - measured_tip)
    cap = np.minimum(vmax, 0.6 * np.sqrt(2.0 * amax * np.abs(correction)))
    wanted = np.clip(
        0.0
        if actual["in_tolerance"]
        else float(teacher.settings["geometry_gain_per_s"]) * correction,
        -cap,
        cap,
    )
    limited = np.clip(
        wanted,
        previous_velocity - amax * DT,
        previous_velocity + amax * DT,
    )
    np.testing.assert_allclose(
        limited, actual["command_velocity_task_si"], rtol=0.0, atol=1e-12
    )
    zero = teacher.geometry_action_from_observation(
        task,
        observation,
        previous_velocity=np.zeros(3, dtype=np.float64),
        command_q=command_task,
    )
    if not zero["valid"]:
        return {"valid": False, "reason": f"zero_velocity:{zero.get('reason')}"}
    return {
        "valid": True,
        "pre_accel_dC_mm": float(wanted[2] * DT / UNIT[2]),
        "zero_vel_dC_mm": float(zero["action_deg_deg_mm"][2]),
        "replayed_dC_mm": float(actual["action_deg_deg_mm"][2]),
        "replayed_action": actual["action_deg_deg_mm"],
        "accel_limited": bool(abs(limited[2] - wanted[2]) > 1e-12),
    }


def _audit(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_dir = args.dataset_dir.resolve()
    metadata, arrays, episode_info = _dataset_context(dataset_dir)
    scene = RCMSceneV9()
    teacher = None
    rows = []
    failures = []
    try:
        targets = _target_states(scene, arrays, episode_info)
        settings = visual_servo_settings(scene)
        settings.update(
            sample_period_s=DT,
            position_feedback="geometry",
            perception_backend="strict_rgbd",
            save_diagnostics=False,
            occlusion_handling="stop",
        )
        teacher = VisualServoController(scene, settings=settings)
        episode_ids = arrays["episode_index"].astype(np.int64)
        state = arrays["state"].astype(np.float64)
        action = arrays["action"].astype(np.float64)
        oct_values = arrays["oct_navigation"].astype(np.float64)
        oct_valid = arrays["oct_navigation_valid"].astype(bool)
        timestamp = arrays["timestamp_s"].astype(np.float64)
        source_frame = arrays["source_frame_index"].astype(np.int64)
        oct_distance = np.linalg.norm(oct_values, axis=1)
        target_c = targets[episode_ids, 2]
        source_teacher = np.asarray(
            [item["action_source"] == "teacher" for item in episode_info],
            dtype=bool,
        )[episode_ids]
        eligible = (
            source_teacher
            & np.isfinite(target_c)
            & oct_valid
            & np.isfinite(oct_values).all(axis=1)
            & (oct_distance < 0.2)
            & (state[:, 2] < target_c - EPS_MM)
            & (action[:, 2] < 0.0)
        )
        candidate_rows = np.flatnonzero(eligible)
        expected = {"Original": 2058, "Plan2": 9342}
        actual_counts = {
            "Original": int(sum(
                episode_info[int(episode_ids[row])]["source_slot"] == 0
                for row in candidate_rows
            )),
            "Plan2": int(sum(
                episode_info[int(episode_ids[row])]["source_slot"] == 1
                for row in candidate_rows
            )),
        }
        if args.smoke:
            rng = np.random.default_rng(42)
            sample = []
            for group in ("Original", "D1", "D2", "D3", "D4"):
                members = [row for row in candidate_rows
                           if episode_info[episode_ids[row]]["group"] == group]
                sample.extend(rng.choice(members, min(8, len(members)), replace=False))
            candidate_rows = np.asarray(sorted(sample), dtype=np.int64)
        previous_episode = None
        for audited, row in enumerate(candidate_rows, 1):
            if (audited - 1) % 100 == 0:
                progress = args.output_dir / "progress.json"
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps({
                    "attempted": audited - 1, "expected": len(candidate_rows),
                    "reproduced": sum(item["reproduced"] for item in rows),
                    "failures": len(failures),
                }) + "\n")
                temporary.replace(progress)
            episode = int(episode_ids[row])
            item = episode_info[episode]
            if episode != previous_episode:
                scene.set_target_world(scene.robot_to_world(item["target_robot_mm"]))
                previous_episode = episode
            frame = int(source_frame[row])
            try:
                raw_path, raw = _raw_rows(dataset_dir.parent.parent, item["source_slot"],
                                          item["source_name"], item["source_index"])
                source = raw[frame]
                if (not np.isclose(source["timestamp_s"], timestamp[row], atol=1e-8, rtol=0)
                        or not np.array_equal(np.asarray(source["state_A_deg_B_deg_C_mm"], np.float32), state[row])
                        or not np.array_equal(np.asarray(source["teacher_action_deg_deg_mm"], np.float32), action[row])
                        or not np.array_equal(np.asarray(source["oct_navigation_mm"], np.float32), oct_values[row])
                        or not source["teacher_valid"] or source["executor"] not in ("policy", "teacher", "beta_teacher")):
                    raise ValueError(f"raw source does not align: {raw_path}:{frame + 1}")
                task, inputs = _teacher_inputs(raw, frame)
            except (FileNotFoundError, KeyError, IndexError, ValueError) as error:
                failures.append({"row": int(row), "episode": episode, "reason": str(error)})
                continue
            scene.set_task(task)
            # A single renderer is reused across episodes; its synthetic OCT
            # clock must stay monotonic even though the source clocks reset.
            teacher.sensor.observation_time_s = audited * DT
            observation = teacher.sensor.observe()
            options = []
            for kind, previous, command_task in inputs:
                replay = _counterfactual(teacher, scene, task, observation, previous, command_task)
                if not replay["valid"]:
                    continue
                prediction = np.asarray(replay.pop("replayed_action"))
                if source["executor"] != "policy":
                    prediction = np.clip(prediction, -ACTION_LIMIT, ACTION_LIMIT)
                error = np.abs(prediction - action[row])
                options.append((replay, error, kind, previous))
            if not options:
                failures.append({
                    "row": int(row),
                    "episode": episode,
                    "reason": "invalid_teacher_geometry_or_counterfactual",
                })
                continue
            matches = [option for option in options if np.all(option[1] <= REPLAY_ATOL)]
            replay, error, kind, previous = min(options, key=lambda option: float(np.max(option[1] / REPLAY_ATOL)))
            # A matching pose is insufficient if two plausible controller
            # histories disagree about the sign counterfactual.
            signatures = {(p["pre_accel_dC_mm"] > 0, p["zero_vel_dC_mm"] >= 0) for p, *_ in matches}
            measured = observation.get("oct_navigation_depth") or {}
            delta_oct = measured.get("delta_xyz_m")
            oct_agrees = delta_oct is not None and np.allclose(
                np.asarray(delta_oct) * 1000, oct_values[row], atol=0.0005, rtol=0)
            reproduced = bool(matches and len(signatures) == 1 and oct_agrees)
            if not reproduced:
                failures.append({
                    "row": int(row), "episode": episode,
                    "reason": "original_label_mismatch" if not matches else
                    "ambiguous_teacher_history" if len(signatures) > 1 else "rgbd_observation_mismatch",
                    "absolute_error_deg_deg_mm": error.tolist(),
                })
            rows.append({
                "row": int(row),
                "episode": episode,
                "frame": int(arrays["frame_index"][row]),
                "source_frame": frame,
                "teacher_input": kind,
                "group": item["group"],
                "source": "Original" if item["source_slot"] == 0 else "Plan2",
                "action_source": item["action_source"],
                "state_A_deg": float(state[row, 0]),
                "state_B_deg": float(state[row, 1]),
                "state_C_mm": float(state[row, 2]),
                "target_C_mm": float(target_c[row]),
                "oct_dx_mm": float(oct_values[row, 0]),
                "oct_dy_mm": float(oct_values[row, 1]),
                "oct_dz_mm": float(oct_values[row, 2]),
                "velocity_A_deg_s": float(previous[0] / UNIT[0]),
                "velocity_B_deg_s": float(previous[1] / UNIT[1]),
                "velocity_C_mm_s": float(previous[2] / UNIT[2]),
                "original_dC_mm": float(action[row, 2]),
                **replay,
                "reproduced": reproduced,
                "dragged_negative": bool(
                    reproduced and replay["pre_accel_dC_mm"] > 0.0
                    and replay["zero_vel_dC_mm"] >= 0.0
                ),
            })
            if audited % 500 == 0:
                print(f"audited {audited}/{len(candidate_rows)} candidates", flush=True)
    finally:
        if teacher is not None:
            teacher.sensor.close()
        scene.close()

    if actual_counts != expected:
        failures.append({
            "reason": "candidate_count_mismatch",
            "expected": expected,
            "actual": actual_counts,
        })
    if len(rows) != len(candidate_rows):
        failures.append({
            "reason": "candidate_replay_count_mismatch",
            "candidate_count": len(candidate_rows),
            "valid_replays": len(rows),
        })
    dragged = sum(bool(row["dragged_negative"]) for row in rows)
    fraction = dragged / len(candidate_rows) if not args.smoke and not failures and rows else None
    result = {
        "status": "SMOKE" if args.smoke else "PASS" if not failures else "FAILED",
        "dataset": str(dataset_dir),
        "candidate_count": len(candidate_rows),
        "replayed_count": len(rows),
        "candidate_counts_by_source": actual_counts,
        "expected_counts_by_source": expected,
        "dragged_negative_count": dragged,
        "reproduced_count": sum(row["reproduced"] for row in rows),
        "replay_absolute_tolerance_deg_deg_mm": REPLAY_ATOL.tolist(),
        "dragged_negative_fraction": fraction,
        "threshold": 0.60,
        "threshold_reached": None if fraction is None else fraction >= 0.60,
        "definition": "pre_accel_dC_mm > 0 and zero_vel_dC_mm >= 0 among valid, near-target, C-under-target, teacher-labeled negative dC rows",
        "control_period_s": DT,
        "failure_count": len(failures),
        "replay_failure_examples": failures[:20],
        "groups": {},
    }
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        count = sum(row["dragged_negative"] for row in selected)
        result["groups"][group] = {
            "candidates": len(selected),
            "dragged_negative": int(count),
            "reproduced": sum(row["reproduced"] for row in selected),
        }
    with (args.output_dir / "contamination_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["row"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "replay_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if fraction is not None and fraction < 0.60:
        anomalies = [row for row in rows if not row["dragged_negative"]]
        picks = np.random.default_rng(42).choice(
            len(anomalies), size=min(20, len(anomalies)), replace=False
        )
        (args.output_dir / "anomalies.json").write_text(
            json.dumps([anomalies[i] for i in picks], indent=2) + "\n"
        )
    (args.output_dir / "contamination_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# C-Label Contamination Audit",
        "",
        f"Status: **{result['status']}**",
        f"Candidates replayed: {len(rows)} / {len(candidate_rows)}",
        f"By source: Original {actual_counts['Original']}, Plan2 {actual_counts['Plan2']}",
        f"Original labels reproduced: {result['reproduced_count']} / {len(candidate_rows)}",
        f"Drag fraction: {fraction:.2%}" if fraction is not None
        else "Drag fraction: not evaluated in smoke mode" if args.smoke
        else "Drag fraction: undetermined (original-label reproduction failed)",
        "",
        "This is a MuJoCo strict-RGB-D teacher replay. It is not a clinical or hardware validation.",
    ]
    (args.output_dir / "contamination_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (args.output_dir / "STATUS.md").write_text(
        "\n".join(lines + (["", f"Replay failures: {len(failures)}"] if failures else []))
        + "\n",
        encoding="utf-8",
    )
    if failures and not args.smoke:
        raise RuntimeError(
            f"label audit is not reproducible: {len(failures)} replay/contract failures"
        )


def dataset_sha256(dataset_dir):
    with (Path(dataset_dir) / "data" / "dataset.npz").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_overlay(dataset_dir, action, overlay_path):
    with np.load(overlay_path, allow_pickle=False) as archive:
        if str(archive["dataset_sha256"]) != dataset_sha256(dataset_dir):
            raise ValueError("label overlay belongs to a different dataset")
        rows = archive["rows"]
        before = archive["old_dC_mm"]
        after = archive["new_dC_mm"]
    if (
        rows.ndim != 1 or rows.dtype != np.int64 or not len(rows)
        or before.shape != rows.shape or after.shape != rows.shape
        or rows[0] < 0 or rows[-1] >= len(action) or np.any(np.diff(rows) <= 0)
        or not np.isfinite(after).all() or np.any(before >= 0)
        or np.any(after < 0) or np.any(after > 0.04271206632256508)
        or not np.array_equal(action[rows, 2], before)
    ):
        raise ValueError("invalid C-only label overlay")
    result = action.copy()
    result[rows, 2] = after
    return result


def windows(arrays, obs_steps=2):
    return valid_window_indices(
        arrays["episode_index"], arrays["source_frame_index"], arrays["timestamp_s"],
        n_obs_steps=obs_steps, n_action_steps=16,
        oct_navigation=arrays["oct_navigation"],
        oct_navigation_valid=arrays["oct_navigation_valid"],
        return_diagnostics=True,
    )


def _prepare_clean(args):
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source = args.dataset_dir.resolve()
    audit_path = args.audit_csv.resolve()
    audit_json = _json(args.audit_json)
    if (audit_json.get("status") != "PASS" or not audit_json.get("threshold_reached")
            or Path(audit_json["dataset"]) != source):
        raise ValueError("cannot clean from a failed/incomplete counterfactual audit")
    with audit_path.open(encoding="utf-8", newline="") as stream:
        changed = sorted(
            (int(row["row"]), float(row["original_dC_mm"]), float(row["zero_vel_dC_mm"]))
            for row in csv.DictReader(stream) if row["dragged_negative"] == "True"
        )
    if len(changed) != audit_json["dragged_negative_count"] or not changed:
        raise ValueError("audit CSV does not match the accepted audit summary")
    _, arrays, _ = _dataset_context(source)
    before_windows, _ = windows(arrays)
    destination.mkdir(parents=True)
    overlay = destination / "label_override.npz"
    index, before, after = zip(*changed)
    np.savez_compressed(
        overlay, rows=np.asarray(index, dtype=np.int64),
        old_dC_mm=np.asarray(before, dtype=np.float32),
        new_dC_mm=np.asarray(after, dtype=np.float32),
        dataset_sha256=dataset_sha256(source),
    )
    actions = apply_overlay(source, arrays["action"], overlay)
    np.testing.assert_array_equal(actions[:, :2], arrays["action"][:, :2])
    np.testing.assert_array_equal(np.flatnonzero(actions[:, 2] != arrays["action"][:, 2]), index)
    after_windows, _ = windows({**arrays, "action": actions})
    np.testing.assert_array_equal(before_windows, after_windows)
    (destination / "label_override.json").write_text(json.dumps({
        "source_dataset": str(source), "audit": str(args.audit_json),
        "updated_action_rows": len(changed), "executed_action_unchanged": True,
        "valid_windows_2x16": len(before_windows),
    }, indent=2) + "\n")


def _window_audit(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _, arrays, episode_info = _dataset_context(args.dataset_dir.resolve())
    records = []
    two, _ = windows(arrays, 2)
    five, _ = windows(arrays, 5)
    for group in ["Original", "D1", "D2", "D3", "D4"]:
        mask = np.asarray([item["group"] == group for item in episode_info])
        frames = mask[arrays["episode_index"]]
        before, after = int(frames[two].sum()), int(frames[five].sum())
        records.append({"group": group, "frames": int(frames.sum()),
                        "windows_2x16": before, "windows_5x16": after,
                        "retained_fraction": after / before if before else None})
    with (args.output_dir / "window_retention.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {"dataset": str(args.dataset_dir.resolve()), "records": records}
    (args.output_dir / "window_retention.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare-clean", "window-audit"):
        sub = commands.add_parser(name)
        sub.add_argument("--dataset-dir", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        if name == "audit":
            sub.add_argument("--smoke", action="store_true")
        if name == "prepare-clean":
            sub.add_argument("--audit-csv", type=Path, required=True)
            sub.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit":
        _audit(args)
    elif args.command == "prepare-clean":
        _prepare_clean(args)
    else:
        _window_audit(args)


if __name__ == "__main__":
    main()
