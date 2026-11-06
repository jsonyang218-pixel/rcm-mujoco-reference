#!/usr/bin/env python3
"""Run paired vision-servo experiments with tip-coupled self-occlusion.

This is an evaluation runner, not a replacement for collect_v9_dataset.py.
It creates one fresh simulator episode for each strategy/scenario/seed tuple.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

from rcm_fundus import make_random_fundus_targets
from rcm_occlusion_policy import LABELS, STRATEGIES, normalize_occlusion_mode
from rcm_scene_v9 import RCMSceneV9
from rcm_visual_servo import VisualServoController, visual_servo_settings


ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def write_controller_rows(path, rows):
    if not rows:
        return
    path = Path(path)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (json.dumps(value, ensure_ascii=False)
                      if isinstance(value, (dict, list, tuple)) else value)
                for key, value in row.items()})


def numeric_list(value):
    result = [float(item) for item in value]
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError("numeric grid values must be nonempty and finite")
    return result


def make_cases(args):
    seeds = [int(value) for value in args.seeds]
    if not seeds or len(set(seeds)) != len(seeds) or any(value < 0 for value in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    durations = [None] if not args.max_active_s else numeric_list(args.max_active_s)
    cases = []
    for seed, strategy, progress, coverage, max_active_s in itertools.product(
            seeds, args.strategies, numeric_list(args.progress),
            numeric_list(args.coverage), durations):
        cases.append({
            "seed": seed,
            "strategy": normalize_occlusion_mode(strategy),
            "trigger_progress": progress,
            "target_coverage": coverage,
            "max_active_s": max_active_s,
            "radius_mm": float(args.occluder_radius_mm),
        })
    return cases[:args.limit_cases] if args.limit_cases else cases


def case_identifier(case):
    duration = "open" if case["max_active_s"] is None else str(case["max_active_s"])
    return (f"seed{case['seed']:06d}__{case['strategy']}__"
            f"p{case['trigger_progress']:.2f}__c{case['target_coverage']:.2f}__"
            f"d{duration}")


def occlusion_summary(audit, dt, fallback_end_s):
    if not audit:
        return {
            "triggered": False,
            "active_frames": 0,
            "duration_s": None,
            "coverage_fraction": None,
            "release_reason": "fixture_disabled",
        }
    transitions = audit.get("transitions", [])
    show = next((item for item in transitions if item.get("transition") == "show"), None)
    hide = next((item for item in transitions if item.get("transition") == "hide"), None)
    if show is None:
        failed = next((item for item in transitions
                       if item.get("state") == "trigger_failed"), None)
        return {
            "triggered": False,
            "active_frames": 0,
            "duration_s": None,
            "coverage_fraction": None,
            "release_reason": (None if failed is None else failed.get("error")),
            "trigger_progress": None,
        }
    if hide is not None and hide.get("duration_s") is not None:
        duration = float(hide["duration_s"])
        release_reason = hide.get("release_reason")
    elif not audit.get("active"):
        duration = None
        release_reason = "released_without_duration"
    else:
        duration = max(0., float(fallback_end_s)-float(show["time_s"]))
        release_reason = "active_at_episode_end"
    return {
        "triggered": True,
        "trigger_time_s": float(show["time_s"]),
        "trigger_progress": float(show["progress"]),
        "active_frames": int(audit.get("active_frames", 0)),
        "duration_s": duration,
        "coverage_fraction": float(audit.get("max_coverage_fraction", 0.)),
        "release_reason": release_reason,
    }


def run_case(base_config, case, args):
    config = copy.deepcopy(base_config)
    config["motor_execution"] = {
        **config.get("motor_execution", {}), "enabled": False}
    config["reference_scene"]["tip_occlusion_fixture"] = {
        "enabled": True,
        "radius_mm": case["radius_mm"],
        "trigger_progress": case["trigger_progress"],
        "target_coverage": case["target_coverage"],
        "release_coverage": float(args.release_coverage),
        "front_offset_mm": float(args.front_offset_mm),
        "screen_direction_deg": float(args.screen_direction_deg),
        "max_active_s": case["max_active_s"],
        "camera_name": "eye_internal",
    }
    if args.resolution:
        config["reference_scene"]["eye_camera"]["resolution_px"] = list(args.resolution)
    visual = config["navigation"]["visual_servo"]
    visual.update({
        "perception_backend": "strict_rgbd",
        "position_feedback": "geometry",
        "use_target_pose_calibration": False,
        "occlusion_handling": case["strategy"],
        "max_iterations": int(args.max_iterations),
        "save_diagnostics": bool(args.save_frames),
        "occlusion_reacquire_frames": int(args.reacquire_frames),
        "occlusion_max_recovery_attempts": int(args.max_recovery_attempts),
        "occlusion_open_loop_max_s": float(args.open_loop_max_s),
        "occlusion_memory_max_age_s": float(args.memory_max_age_s),
        "occlusion_memory_max_travel_mm": float(args.memory_max_travel_mm),
        "occlusion_retract_mm": float(args.retract_mm),
        "occlusion_retract_speed_mm_s": float(args.retract_speed_mm_s),
        "occlusion_recovery_timeout_s": float(args.recovery_timeout_s),
    })

    directory = Path(args.output_dir) / case_identifier(case)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "scenario.json", case)
    write_json(directory / "config_snapshot.json", config)
    started = time.monotonic()
    scene = None
    controller = None
    report = None
    audit = None
    error = None
    try:
        scene = RCMSceneV9(config)
        if args.random_targets:
            targets = make_random_fundus_targets(
                scene, np.random.default_rng(case["seed"]), 1)
            scene.set_target_world(targets["targets"][0]["point_mm"])
        scene.set_task(np.asarray(args.start_task, dtype=float)
                       / np.array([180/math.pi, 180/math.pi, 1000.]))
        if args.motion_backend == "motor":
            scene.enable_motor_execution(config.get("motor_execution"))
            scene.reset_motor_time(0.)
        scene.reset_tip_occlusion()
        controller = VisualServoController(
            scene, settings=visual_servo_settings(scene),
            output_dir=(directory if args.save_frames else None))
        report = controller.run()
        audit = scene.tip_occlusion_audit()
        if args.save_frames:
            controller.write(report, directory)
    except Exception as exc:  # Experiment records must preserve failed cases.
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if scene is not None:
            scene.close()
    write_controller_rows(
        directory/"controller.csv", [] if controller is None else controller.rows)
    wall_s = time.monotonic()-started
    dt = float(visual.get("sample_period_s", .05))
    final_time = 0.
    if report is not None:
        final_time = float(report.get("iterations", 0))*dt
    occlusion = occlusion_summary(audit, dt, final_time)
    result = {
        "case": case,
        "strategy_label": LABELS.get(case["strategy"]),
        "result": "ERROR" if error else report.get("result", "ERROR"),
        "controller_reason": (None if report is None
                              else report.get("controller_reason")),
        "final_error_um": (None if report is None
                           else report.get("final_tip_target_error_um")),
        "iterations": (None if report is None else report.get("iterations")),
        "simulation_time_s": final_time,
        "wall_s": wall_s,
        "occlusion": occlusion,
        "fixture_audit": audit,
        "error": error,
        "report": report,
    }
    write_json(directory / "result.json", result)
    print(json.dumps({
        "case": case_identifier(case),
        "result": result["result"],
        "reason": result["controller_reason"],
        "error_um": result["final_error_um"],
        "occlusion_triggered": occlusion["triggered"],
        "occlusion_duration_s": occlusion["duration_s"],
        "occlusion_coverage": occlusion["coverage_fraction"],
        "wall_s": round(wall_s, 1),
        "error": error,
    }, ensure_ascii=False), flush=True)
    return result


def summarize(results):
    grouped = {}
    for item in results:
        strategy = item["case"]["strategy"]
        row = grouped.setdefault(strategy, {
            "cases": 0, "success": 0, "errors": 0, "triggered": 0,
            "final_errors_um": [], "durations_s": [],
        })
        row["cases"] += 1
        row["success"] += int(item["result"] == "PASS")
        row["errors"] += int(item["result"] == "ERROR")
        row["triggered"] += int(item["occlusion"]["triggered"])
        if item["final_error_um"] is not None:
            row["final_errors_um"].append(float(item["final_error_um"]))
        if item["occlusion"]["duration_s"] is not None:
            row["durations_s"].append(float(item["occlusion"]["duration_s"]))
    for record in grouped.values():
        errors = np.asarray(record.pop("final_errors_um"), dtype=float)
        durations = np.asarray(record.pop("durations_s"), dtype=float)
        record["success_rate"] = record["success"] / max(1, record["cases"])
        record["trigger_rate"] = record["triggered"] / max(1, record["cases"])
        record["final_error_um"] = {
            "median": None if not len(errors) else float(np.median(errors)),
            "mean": None if not len(errors) else float(np.mean(errors)),
            "p10": None if not len(errors) else float(np.quantile(errors, .1)),
            "p90": None if not len(errors) else float(np.quantile(errors, .9)),
        }
        record["occlusion_duration_s"] = {
            "median": None if not len(durations) else float(np.median(durations)),
            "mean": None if not len(durations) else float(np.mean(durations)),
            "min": None if not len(durations) else float(np.min(durations)),
            "max": None if not len(durations) else float(np.max(durations)),
        }
    return {
        "cases": len(results),
        "success": sum(item["result"] == "PASS" for item in results),
        "errors": sum(item["result"] == "ERROR" for item in results),
        "occlusion_triggered": sum(
            item["occlusion"]["triggered"] for item in results),
        "by_strategy": grouped,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT/"runs"/"occlusion_strategy_pilot")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                        choices=STRATEGIES)
    parser.add_argument("--progress", nargs="+", default=[.5, .8, .9])
    parser.add_argument("--coverage", nargs="+", default=[.5, 1.])
    parser.add_argument(
        "--max-active-s", nargs="+", default=[.5, 1., 2.],
        help="Optional upper exposure cap; actual duration may end earlier")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261102])
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--random-targets", action="store_true")
    parser.add_argument("--start-task", nargs=3, type=float, default=[0., 0., 0.],
                        metavar=("A_DEG", "B_DEG", "C_MM"))
    parser.add_argument("--motion-backend", choices=("motor", "kinematic"),
                        default="motor")
    parser.add_argument("--max-iterations", type=int, default=1600)
    parser.add_argument("--resolution", nargs=2, type=int,
                        help="Optional pilot override, e.g. --resolution 640 480")
    parser.add_argument("--occluder-radius-mm", type=float, default=.18)
    parser.add_argument("--release-coverage", type=float, default=.1)
    parser.add_argument("--front-offset-mm", type=float, default=.05)
    parser.add_argument("--screen-direction-deg", type=float, default=0.)
    parser.add_argument("--reacquire-frames", type=int, default=3)
    parser.add_argument("--max-recovery-attempts", type=int, default=3)
    parser.add_argument("--open-loop-max-s", type=float, default=2.2)
    parser.add_argument("--memory-max-age-s", type=float, default=2.2)
    parser.add_argument("--memory-max-travel-mm", type=float, default=.8)
    parser.add_argument("--retract-mm", type=float, default=1.)
    parser.add_argument("--retract-speed-mm-s", type=float, default=.3)
    parser.add_argument("--recovery-timeout-s", type=float, default=5.)
    parser.add_argument("--save-frames", action="store_true",
                        help="Save per-step diagnostics; off by default for the pilot")
    args = parser.parse_args(argv)
    if args.limit_cases is not None and args.limit_cases < 1:
        parser.error("--limit-cases must be positive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if args.max_recovery_attempts < 1:
        parser.error("--max-recovery-attempts must be positive")
    if args.resolution and any(value < 160 for value in args.resolution):
        parser.error("--resolution values must be at least 160")
    if args.occluder_radius_mm <= 0.:
        parser.error("--occluder-radius-mm must be positive")
    if not 0. <= args.release_coverage < 1.:
        parser.error("--release-coverage must be in [0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = make_cases(args)
    write_json(output/"experiment_protocol.json", {
        "case_count": len(cases),
        "cases": cases,
        "dataset_recollection_used": False,
        "scenario_scope": ("Privileged tip-coupled rendered occluder; policy-visible "
                           "input remains normal strict RGB-D only."),
        "duration_semantics": ("--max-active-s is an exposure cap; actual duration "
                               "ends earlier when the tool leaves the occlusion "
                               "configuration."),
        "paired_seed_semantics": ("Each seed reuses the same target, start pose and "
                                  "occlusion parameters across strategies."),
    })
    if args.dry_run:
        print(json.dumps({
            "case_count": len(cases),
            "first": case_identifier(cases[0]) if cases else None,
            "last": case_identifier(cases[-1]) if cases else None,
        }, ensure_ascii=False, indent=2))
        return 0
    results = [run_case(base_config, case, args) for case in cases]
    summary = summarize(results)
    write_json(output/"summary.json", summary)
    rows = []
    for item in results:
        rows.append({
            "case": case_identifier(item["case"]),
            "strategy": item["case"]["strategy"],
            "trigger_progress": item["case"]["trigger_progress"],
            "target_coverage": item["case"]["target_coverage"],
            "max_active_s": item["case"]["max_active_s"],
            "result": item["result"],
            "controller_reason": item["controller_reason"],
            "final_error_um": item["final_error_um"],
            "occlusion_triggered": item["occlusion"]["triggered"],
            "occlusion_duration_s": item["occlusion"]["duration_s"],
            "occlusion_coverage": item["occlusion"]["coverage_fraction"],
            "simulation_time_s": item["simulation_time_s"],
            "wall_s": item["wall_s"],
        })
    with (output/"metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
