"""Freeze reproducible posterior-fundus scenarios and optionally collect episodes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rcm_fundus import (clearance_lower_bound_mm, make_random_fundus_targets,
                        verify_fundus_targets)
from rcm_navigation import task_si_units, write_json
from rcm_reference import HERE, Reference
from rcm_waypoint_dataset import TASK_REGIMES


def _episode_seed(base_seed, split_index, episode_index):
    # Keep split streams disjoint while remaining stable across Python versions.
    return int(base_seed) + int(split_index)*1_000_000 + int(episode_index)


def _sample_start_task(reference, seed, target_world_mm=None, minimum_clearance_mm=1.0,
                       max_attempts=10000):
    """Sample a safe reset pose where the initial target is visible in the current frame."""
    rng = np.random.default_rng(seed)
    home = np.zeros(3)
    from rcm_visual_servo import VisualServoSensor
    if target_world_mm is not None:
        reference.set_target_world(target_world_mm)
    sensor = VisualServoSensor(reference)
    try:
        for _ in range(max_attempts):
            candidate = np.array([rng.uniform(-8., 8.), rng.uniform(-8., 8.),
                                  rng.uniform(-2., 2.)])
            try:
                reference.set_task(task_si_units(candidate))
                geometry = reference.metrics()["needle_eye_geometry"]
                if (geometry["tip_inside_eye"] and
                        clearance_lower_bound_mm(reference, candidate) >= minimum_clearance_mm):
                    observation = sensor.observe()
                    if (target_world_mm is None or
                            observation["target_pixel"] is not None and observation["tip_pixel"] is not None):
                        return candidate.tolist()
            except ValueError:
                pass
            finally:
                reference.set_task(task_si_units(home))
    finally:
        sensor.close()
    raise RuntimeError("Could not sample a safe reset pose with the current target visible")


def _write_targets(reference, path, seed, count, min_separation_mm):
    rng = np.random.default_rng(seed)
    payload = make_random_fundus_targets(reference, rng, count, min_separation_mm)
    verification_targets = []
    for target in payload["targets"]:
        point = np.asarray(target["point_mm"], dtype=float)
        world = reference.robot_to_world(point) if payload.get("frame", "world") == "robot" else point
        verification_targets.append({"id": target["id"], "requested_world_mm": world.tolist()})
    verification = verify_fundus_targets(reference, verification_targets, payload["target_region"])
    if verification["result"] != "PASS":
        raise RuntimeError("Random target did not pass fundus surface verification")
    payload["generation"] = {
        "generator": "random_fundus_dataset.py",
        "random_seed": int(seed),
        "target_count": int(count),
        "minimum_separation_mm": float(min_separation_mm),
        "sampling": "uniform normalized posterior cap surface coordinates",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload, verification


def _collect_episode(target_path, episode_dir, methods, planner_seeds, profile, config_path,
                     sampling_frequency_hz, task_regime, start_task=None, strict_visual=False):
    episode_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(HERE/"run_waypoint_comparison.py"),
               "--targets", str(target_path), "--config", str(config_path),
               "--output-dir", str(episode_dir), "--profile", profile,
               "--planner-feedback", "vision", "--methods", *methods,
               "--seeds", *(str(seed) for seed in planner_seeds),
               "--sampling-frequency-hz", str(sampling_frequency_hz),
               "--task-regime", task_regime,
               "--baseline-dir", str(episode_dir.parent/"no_baseline")]
    if start_task is not None:
        command.extend(["--start-task", *(str(value) for value in start_task)])
    if strict_visual:
        command.append("--strict-visual")
    log_path = episode_dir.parent/(episode_dir.name+".stdout.log")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False)
    reports_path = episode_dir/"run_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8")) if reports_path.exists() else []
    manifest_path = episode_dir/"dataset"/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected = Counter((method, seed) for method in methods
                       for seed in ([None] if method == "visual_servo" else planner_seeds))
    manifest_episodes = manifest.get("episodes", [])
    reported = Counter((row.get("method"), row.get("seed")) for row in reports)
    captured = Counter((row.get("method"), row.get("seed")) for row in manifest_episodes)
    expected_count = sum(expected.values())
    capture_complete = (len(reports) == len(manifest_episodes) == expected_count
                        and reported == captured == expected)
    return {"exit_code": int(completed.returncode), "capture_complete": capture_complete,
            "expected_runs": expected_count, "reported_runs": len(reports),
            "manifest_episodes": len(manifest_episodes),
            "log": str(log_path), "reports": reports}


def _summarize_collection_performance(episodes):
    """Pool per-target occlusion and stop outcomes across random target sets."""
    conditions = ("visible", "needle_occluded", "other_visibility_issue", "unknown")
    methods = {}
    for episode in episodes:
        reports = episode.get("collection", {}).get("reports", [])
        for report in reports:
            method = report.get("method", "unknown")
            summary = methods.setdefault(method, {
                "episodes": 0, "successful_episodes": 0, "target_visits": 0,
                "validated_target_stops": 0, "stop_constraint_pass_episodes": 0,
                "condition_exposure": {condition: {"target_visits": 0,
                                                   "successful_target_visits": 0}
                                       for condition in conditions},
                "failure_reasons": {},
            })
            summary["episodes"] += 1
            summary["successful_episodes"] += int(report.get("result") == "PASS")
            stop = report.get("stop_constraint_summary", {})
            summary["target_visits"] += int(stop.get("attempted_targets", 0))
            summary["validated_target_stops"] += int(stop.get("targets_with_validated_stop", 0))
            summary["stop_constraint_pass_episodes"] += int(bool(stop.get("pass", False)))
            for condition in conditions:
                source = report.get("target_success_by_visibility", {}).get("condition_exposure", {}).get(condition, {})
                summary["condition_exposure"][condition]["target_visits"] += int(source.get("target_visits", 0))
                summary["condition_exposure"][condition]["successful_target_visits"] += int(
                    source.get("successful_target_visits", 0))
            for waypoint in report.get("waypoints", []):
                if waypoint.get("result") not in ("PASS", "NOT_RUN"):
                    reason = waypoint.get("reason", "UNKNOWN")
                    summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0)+1
    for summary in methods.values():
        summary["episode_success_rate"] = (summary["successful_episodes"]/summary["episodes"]
                                           if summary["episodes"] else None)
        summary["validated_stop_rate_per_attempted_target"] = (
            summary["validated_target_stops"]/summary["target_visits"]
            if summary["target_visits"] else None)
        summary["stop_constraint_pass_rate"] = (
            summary["stop_constraint_pass_episodes"]/summary["episodes"]
            if summary["episodes"] else None)
        for row in summary["condition_exposure"].values():
            row["success_rate"] = (row["successful_target_visits"]/row["target_visits"]
                                   if row["target_visits"] else None)
    return {
        "unit": "per-target visit for visibility strata; per-method run for episode and stop rates",
        "success_definition": "endpoint result PASS and terminal stop validated",
        "visibility_groups_overlap": True,
        "methods": methods,
    }


def _canonical_sha256(payload):
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_config(root, config_bytes):
    """Retain the exact configuration read before scenario generation."""
    path = root/"collection_config.json"
    with path.open("xb") as stream:
        stream.write(config_bytes)
    return path


def _freeze_scenarios(root, scenarios, args, config_path):
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_collection",
        "sampling_frequency_hz": args.sampling_frequency_hz,
        "target_count_per_episode": args.targets_per_episode,
        "task_regime": args.task_regime,
        "guidance_mode": "strict_current_frame_visual_servo" if args.strict_visual else "configured_method_set",
        "start_pose_sampling": "safe eye clearance and current-frame target/tip visibility; no target inverse kinematics"
                                if args.randomize_start else "fixed zero reset pose",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_snapshot": config_path.relative_to(root).as_posix(),
        "scenarios": [{
            key: scenario[key] for key in (
                "split", "episode_index", "random_seed", "targets", "targets_sha256",
                "start_task_A_deg_B_deg_C_mm", "target_verification")
        } for scenario in scenarios],
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    write_json(root/"scenario_manifest.json", manifest)
    return manifest["manifest_sha256"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE/"runs"/"历史记录"/"05_random_datasets"/f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--episodes", type=int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
                        default=[100, 20, 20])
    parser.add_argument("--pilot-episodes", type=int, default=0,
                        help="Number of separately labeled pilot scenarios (default: 0)")
    parser.add_argument("--pilot-only", action="store_true",
                        help="Freeze all scenarios but collect only pilot episodes")
    parser.add_argument("--targets-per-episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--min-separation-mm", type=float, default=1.0)
    parser.add_argument("--collect", action="store_true", help="Run navigation and save RGB/state/action episodes")
    parser.add_argument("--methods", nargs="+", choices=("visual_servo", "RRT", "PRM"),
                        default=["visual_servo", "RRT", "PRM"])
    parser.add_argument("--planner-seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--profile", choices=("precision", "legacy"), default="precision")
    parser.add_argument("--strict-visual", action="store_true",
                        help="Current-frame visual servo only; no equation-generated route or target-pose calibration")
    parser.add_argument("--randomize-start", action="store_true",
                        help="Assign each episode a reproducible safe randomized start pose")
    parser.add_argument("--sampling-frequency-hz", type=float, default=20.0,
                        help="Nominal RGB/state/action dataset rate (default: 20 Hz)")
    parser.add_argument("--task-regime", choices=TASK_REGIMES,
                        default="visual_current_target",
                        help="Whether simulated registered target coordinates are included in policy task text")
    args = parser.parse_args(argv)
    if any(int(value) < 0 for value in args.episodes):
        parser.error("episode counts must be nonnegative")
    if args.pilot_episodes < 0:
        parser.error("--pilot-episodes must be nonnegative")
    if args.pilot_only and args.pilot_episodes == 0:
        parser.error("--pilot-only requires --pilot-episodes greater than zero")
    if args.pilot_only and not args.collect:
        parser.error("--pilot-only requires --collect")
    if args.pilot_only and not args.strict_visual:
        parser.error("--pilot-only requires --strict-visual")
    if args.targets_per_episode < 1:
        parser.error("--targets-per-episode must be positive")
    if args.seed < 0 or any(seed < 0 for seed in args.planner_seeds):
        parser.error("seeds must be nonnegative")
    if args.min_separation_mm < 0:
        parser.error("--min-separation-mm must be nonnegative")
    if not math.isfinite(args.sampling_frequency_hz) or args.sampling_frequency_hz <= 0.:
        parser.error("--sampling-frequency-hz must be finite and positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be distinct")
    if args.strict_visual:
        if args.methods != ["visual_servo"]:
            parser.error("--strict-visual requires --methods visual_servo")
        if args.task_regime != "visual_current_target":
            parser.error("--strict-visual requires --task-regime visual_current_target")
    if args.profile != "precision":
        parser.error("Random posterior-fundus collection requires the precision profile to verify a stop at every target")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    reference = Reference(copy.deepcopy(config))
    split_names = ("train", "validation", "test")
    root = args.output_dir.resolve()
    if root.exists():
        parser.error(f"output directory already exists: {root}")
    root.mkdir(parents=True)
    frozen_config = _freeze_config(root, config_bytes)
    index = {
        "schema_version": 1,
        "dataset_type": "random_posterior_fundus_navigation",
        "target_region": "posterior_eye_surface",
        "target_count_per_episode": args.targets_per_episode,
        "target_sampling": {
            "distribution": "uniform in normalized posterior-cap coordinates on the compiled eye ellipsoid",
            "minimum_normalized_posterior_z": 0.9,
            "minimum_separation_mm": args.min_separation_mm,
            "seed_scope": "one deterministic independent seed per dataset split and episode",
        },
        "base_seed": args.seed,
        "split_counts": dict(zip(split_names, args.episodes)),
        "pilot_episode_count": args.pilot_episodes,
        "perception": "simulated RGB + segmentation + OCT-like rendered depth; no synthetic noise; physical-geometry needle occlusion is labeled per frame",
        "sampling_frequency_hz": args.sampling_frequency_hz,
        "task_regime": args.task_regime,
        "config_snapshot": frozen_config.relative_to(root).as_posix(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "research_scope": {
            "coordinate_source": "simulation targets, not measured OCT",
            "expert": ("strict current-frame visual-servo controller; target coordinates are reserved for scene setup and independent acceptance"
                       if args.strict_visual else "configured simulation controller; see each experiment_protocol.json"),
            "multi_target": ("one target per episode; no visit ordering is required"
                             if args.targets_per_episode == 1 else
                             f"one set of {args.targets_per_episode} sequentially presented targets per episode; "
                             + ("strict visual servo follows the given input order without target inverse kinematics or a precomputed scene route"
                                if args.strict_visual else
                                "methods share a script-selected visit order")),
            "image_ablation": "paired input exports only; no policy training or evaluation has been performed",
            "split_unit": "one independent target set and reset pose per episode; every method and planner seed for that set stays in the same split; pilot, train, validation, and test scenes are frozen before collection",
        },
        "methods": args.methods,
        "strict_visual": args.strict_visual,
        "randomized_start_pose": args.randomize_start,
        "episodes": [],
    }
    scenarios = []
    split_plan = [("pilot", args.pilot_episodes, 3)] + [
        (split, int(count), split_index) for split_index, (split, count)
        in enumerate(zip(split_names, args.episodes))
    ]
    for split, episode_count, split_index in split_plan:
        for episode_index in range(episode_count):
            episode_seed = _episode_seed(args.seed, split_index, episode_index)
            target_path = root/"targets"/split/f"episode_{episode_index:06d}.json"
            _, verification = _write_targets(reference, target_path, episode_seed,
                                             args.targets_per_episode, args.min_separation_mm)
            start_seed = episode_seed+500_000
            target_payload = json.loads(target_path.read_text(encoding="utf-8"))
            target_point = target_payload["targets"][0]["point_mm"]
            target_world = (reference.robot_to_world(target_point)
                            if target_payload.get("frame", "world") == "robot"
                            else target_point)
            start_task = (_sample_start_task(reference, start_seed, target_world)
                          if args.randomize_start else [0., 0., 0.])
            scenarios.append({
                "split": split, "episode_index": episode_index,
                "random_seed": episode_seed, "start_seed": start_seed if args.randomize_start else None,
                "target_count": args.targets_per_episode,
                "targets": str(target_path.relative_to(root).as_posix()),
                "targets_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                "start_task_A_deg_B_deg_C_mm": start_task,
                "target_verification": verification,
                "collection_status": "not_started",
            })
            print(json.dumps({"scenarios_frozen": len(scenarios), "split": split,
                              "episode": episode_index}, ensure_ascii=False), flush=True)
    frozen_hash = _freeze_scenarios(root, scenarios, args, frozen_config)
    index["scenario_manifest"] = "scenario_manifest.json"
    index["scenario_manifest_sha256"] = frozen_hash
    index["episodes"] = scenarios
    index["collection_status"] = "not_started" if args.collect else "targets_frozen_only"
    index["performance_summary"] = _summarize_collection_performance(index["episodes"])
    write_json(root/"dataset_index.json", index)

    selected = ([scenario for scenario in scenarios if scenario["split"] == "pilot"]
                if args.pilot_only else scenarios)
    for scenario in selected if args.collect else []:
        split = scenario["split"]
        episode_index = scenario["episode_index"]
        target_path = root/scenario["targets"]
        episode_dir = root/"episodes"/split/f"episode_{episode_index:06d}"
        result = _collect_episode(
            target_path, episode_dir, args.methods, args.planner_seeds, args.profile, frozen_config,
            args.sampling_frequency_hz, args.task_regime,
            start_task=scenario["start_task_A_deg_B_deg_C_mm"], strict_visual=args.strict_visual)
        scenario["collection"] = result
        scenario["collection_status"] = "completed" if result["capture_complete"] else "failed_process"
        index["collection_status"] = "in_progress"
        index["performance_summary"] = _summarize_collection_performance(index["episodes"])
        write_json(root/"dataset_index.json", index)
        print(json.dumps({"split": split, "episode": episode_index,
                          "seed": scenario["random_seed"], "exit_code": result["exit_code"],
                          "reported_runs": len(result["reports"])}, ensure_ascii=False), flush=True)
    index["performance_summary"] = _summarize_collection_performance(index["episodes"])
    if args.collect:
        index["collection_status"] = "complete" if all(
            scenario.get("collection_status") == "completed" for scenario in selected) else "complete_with_failures"
    write_json(root/"dataset_index.json", index)
    print(json.dumps({"output_dir": str(root.resolve()),
                      "frozen_scenarios": len(index["episodes"]),
                      "collected_scenarios": len(selected) if args.collect else 0,
                      "collection_status": index["collection_status"],
                      "completed_scenarios": sum(scenario.get("collection_status") == "completed"
                                                  for scenario in selected),
                      "navigation_successes": sum(
                          report.get("result") == "PASS"
                          for scenario in selected for report in scenario.get("collection", {}).get("reports", [])),
                      "navigation_failures": sum(
                          report.get("result") != "PASS"
                          for scenario in selected for report in scenario.get("collection", {}).get("reports", [])),
                      "scenario_manifest_sha256": frozen_hash}, ensure_ascii=False, indent=2))
    return 0 if index["collection_status"] in ("complete", "targets_frozen_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
