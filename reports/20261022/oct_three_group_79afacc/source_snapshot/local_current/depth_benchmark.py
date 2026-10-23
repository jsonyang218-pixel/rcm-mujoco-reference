#!/usr/bin/env python3
"""Benchmark a tuned uniform retract depth against adaptive retract depth."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from multi_target_experiment import (
    METHODS,
    _annotate_transition_geometry,
    _axis_world,
    _continuous_plan,
    _path_length,
    _run_method,
    _solve_in_order,
)
from rcm_navigation import NavigationRejected, execute_plan, task_si_units, write_json
from rcm_reference import HERE, Reference


def _read_scenarios(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Scenario file must be a schema_version=1 object")
    for group in ("development", "evaluation"):
        if not isinstance(payload.get(group), list) or not payload[group]:
            raise ValueError(f"Scenario file needs a nonempty {group} list")
    if "stress" in payload and not isinstance(payload["stress"], list):
        raise ValueError("Scenario file stress must be a list when provided")
    return payload


def _scenario_targets(scenario):
    """Convert relative task poses in the scenario file to robot-frame points."""
    if not isinstance(scenario, dict) or not isinstance(scenario.get("targets"), list):
        raise ValueError("Each scenario needs a targets list")
    if len(scenario["targets"]) != 2:
        raise ValueError(f"Scenario {scenario.get('id', '<unknown>')} must contain exactly two targets")
    reference = Reference()
    targets = []
    for index, target in enumerate(scenario["targets"]):
        if not isinstance(target, dict) or "task_A_deg_B_deg_C_mm" not in target:
            raise ValueError(f"Scenario target {index} needs task_A_deg_B_deg_C_mm")
        q_user = np.asarray(target["task_A_deg_B_deg_C_mm"], dtype=float)
        if q_user.shape != (3,) or not np.isfinite(q_user).all():
            raise ValueError(f"Scenario target {index} task pose must contain three finite values")
        q = task_si_units(q_user)
        reference.validate_task(q)
        point = reference.nominal.forward(reference.state_from_task(q), enforce_limits=False)
        targets.append({
            "id": target.get("id", f"{scenario.get('id', 'scenario')}_target_{index+1}"),
            "frame": "robot",
            "point_mm": [float(value) for value in point],
            "source_task_A_deg_B_deg_C_mm": q_user.tolist(),
        })
    if len({target["id"] for target in targets}) != len(targets):
        raise ValueError(f"Scenario {scenario.get('id', '<unknown>')} target ids must be unique")
    return targets


def _evaluate_in_memory(method, targets, fixed_depth_mm, eye_margin_mm):
    """Evaluate one method without writing a candidate artifact directory."""
    planning_started = time.perf_counter()
    reference, solutions = _solve_in_order(targets)
    plan, transitions = _continuous_plan(
        reference, solutions, method, fixed_depth_mm, eye_margin_mm)
    recorder = execute_plan(reference, plan)
    _annotate_transition_geometry(recorder, transitions)
    report = recorder.report()
    report.update({
        "method": method,
        "method_description": METHODS[method],
        "target_order": [solution["id"] for solution in solutions],
        "transitions": transitions,
        "transition_metrics": transitions,
        "transition_retracts_mm": [item["retract_to_C_mm"] for item in transitions],
        "planning_time_s": time.perf_counter()-planning_started,
        "joint_path_length_deg_deg_mm": _path_length(plan),
    })
    return report


def _development_score(scenarios, parameter_mm, eye_margin_mm, method):
    results = []
    for scenario in scenarios:
        targets = _scenario_targets(scenario)
        try:
            report = _evaluate_in_memory(method, targets, parameter_mm, eye_margin_mm)
            target_depths = [float(solution["command_task_A_deg_B_deg_C_mm"][2])
                             for solution in report["target_solutions"]]
            valid_depth = method != "B" or parameter_mm <= min(target_depths) + 1e-4
            geometry_ok = all(item["geometry_feasible"] for item in report["transition_metrics"])
            valid = report["result"] == "PASS" and valid_depth and geometry_ok
            failure_reasons = []
            failure_details = []
            if not valid_depth:
                failure_reasons.append("fixed depth is deeper than a target")
                failure_details.append({
                    "code": "FIXED_DEPTH_DEEPER_THAN_TARGET",
                    "message": "uniform fixed C is deeper than the shallower target and is not a retract position",
                })
            for transition in report["transition_metrics"]:
                failure_reasons.extend(item["code"] for item in transition["infeasibility_reasons"])
                failure_details.extend(transition["infeasibility_reasons"])
            reason = None if valid else (failure_reasons or ["navigation check failed"])
            results.append({
                "scenario_id": scenario["id"],
                "result": report["result"],
                "planned_duration_s": report["planned_duration_s"],
                "planning_time_s": report["planning_time_s"],
                "geometry_feasible": geometry_ok,
                "valid_for_tuning": valid,
                "invalid_reason": reason,
                "invalid_reason_details": failure_details,
            })
        except (NavigationRejected, ValueError) as error:
            results.append({
                "scenario_id": scenario["id"],
                "result": "REJECTED",
                "planned_duration_s": None,
                "planning_time_s": None,
                "geometry_feasible": False,
                "valid_for_tuning": False,
                "invalid_reason": str(error),
                "invalid_reason_details": [{"code": "PREFLIGHT_REJECTED", "message": str(error)}],
            })
    valid = bool(results) and all(item["valid_for_tuning"] for item in results)
    durations = [item["planned_duration_s"] for item in results if item["valid_for_tuning"]]
    return {
        "parameter_mm": parameter_mm,
        "parameter_name": "fixed_depth_mm" if method == "B" else "delta_mm",
        "method": method,
        "valid_for_tuning": valid,
        "mean_planned_duration_s": None if not valid else float(np.mean(durations)),
        "max_planned_duration_s": None if not valid else float(np.max(durations)),
        "scenarios": results,
    }


def _select_parameter(scenarios, grid_mm, eye_margin_mm, method):
    scores = [_development_score(scenarios, value, eye_margin_mm, method) for value in grid_mm]
    feasible = [score for score in scores if score["valid_for_tuning"]]
    if not feasible:
        raise ValueError(f"No {method} parameter in the tuning grid is feasible for every development scenario")
    selected = min(feasible, key=lambda score: (score["mean_planned_duration_s"],
                                                -score["parameter_mm"]))
    return selected, scores


def _angle_separation_deg(report):
    reference = Reference()
    poses = [task_si_units(solution["command_task_A_deg_B_deg_C_mm"])
             for solution in report["target_solutions"]]
    dot = float(np.clip(_axis_world(reference, poses[0]) @ _axis_world(reference, poses[1]), -1., 1.))
    return math.degrees(math.acos(dot))


def _run_comparison(scenario, output_dir, fixed_depth_mm, rule_delta_mm, eye_margin_mm):
    targets = _scenario_targets(scenario)
    fixed_report, _ = _run_method(
        "B", targets, output_dir/"fixed_uniform", fixed_depth_mm, eye_margin_mm)
    fixed_report.update({
        "benchmark_role": "tuned_uniform_fixed_depth_baseline",
        "fixed_depth_mm": fixed_depth_mm,
        "fixed_depth_source": "development_scenarios_only",
    })
    write_json(output_dir/"fixed_uniform"/"experiment_report.json", fixed_report)

    rule_report, _ = _run_method(
        "R", targets, output_dir/"simple_rule", rule_delta_mm, eye_margin_mm)
    rule_report.update({
        "benchmark_role": "simple_min_target_depth_minus_delta_rule",
        "rule_delta_mm": rule_delta_mm,
        "rule_delta_source": "development_scenarios_only",
    })
    write_json(output_dir/"simple_rule"/"experiment_report.json", rule_report)

    adaptive_report, _ = _run_method(
        "C", targets, output_dir/"adaptive", 0., eye_margin_mm)
    adaptive_report.update({
        "benchmark_role": "adaptive_sampled_maximum",
        "fixed_depth_source": None,
    })
    write_json(output_dir/"adaptive"/"experiment_report.json", adaptive_report)

    fixed_transition = fixed_report["transition_metrics"][0]
    rule_transition = rule_report["transition_metrics"][0]
    adaptive_transition = adaptive_report["transition_metrics"][0]
    rule_saving = rule_report["planned_duration_s"]-adaptive_report["planned_duration_s"]
    fixed_saving = fixed_report["planned_duration_s"]-adaptive_report["planned_duration_s"]
    rule_failure_reasons = list(rule_transition["infeasibility_reasons"])
    if rule_transition.get("fallback_applied"):
        rule_failure_reasons.insert(0, {
            "code": "SIMPLE_RULE_FALLBACK",
            "message": rule_transition.get("fallback_reason"),
        })
    rule_same = (abs(rule_transition["retract_to_C_mm"]-
                     adaptive_transition["retract_to_C_mm"]) <= 1e-4 and
                 abs(rule_report["planned_duration_s"]-
                     adaptive_report["planned_duration_s"]) <= 1e-6)
    row = {
        "scenario_id": scenario["id"],
        "angle_class": scenario.get("angle_class"),
        "depth_class": scenario.get("depth_class"),
        "angle_separation_deg": _angle_separation_deg(adaptive_report),
        "start_C_mm": adaptive_transition["start_C_mm"],
        "end_C_mm": adaptive_transition["end_C_mm"],
        "adaptive_transfer_C_mm": adaptive_transition["selected_transfer_C_mm"],
        "max_feasible_transfer_C_mm": adaptive_transition["max_feasible_transfer_C_mm"],
        "target_depth_cap_C_mm": adaptive_transition["target_depth_cap_C_mm"],
        "surface_depth_cap_C_mm": adaptive_transition["surface_depth_cap_C_mm"],
        "minimum_eye_surface_clearance_mm": adaptive_transition["minimum_eye_surface_clearance_mm"],
        "adaptive_equals_shallower_target": adaptive_transition["adaptive_equals_shallower_target"],
        "adaptive_equals_max_feasible": adaptive_transition["adaptive_equals_max_feasible"],
        "limiting_reasons": adaptive_transition["limiting_reasons"],
        "fixed_depth_mm": fixed_depth_mm,
        "fixed_transfer_total_s": fixed_transition["total_time_s"],
        "fixed_planned_duration_s": fixed_report["planned_duration_s"],
        "fixed_planning_time_s": fixed_report["planning_time_s"],
        "fixed_geometry_feasible": fixed_transition["geometry_feasible"],
        "fixed_minimum_eye_surface_clearance_mm": fixed_transition.get("minimum_eye_surface_clearance_mm"),
        "rule_delta_mm": rule_delta_mm,
        "rule_transfer_C_mm": rule_transition["selected_transfer_C_mm"],
        "rule_transfer_total_s": rule_transition["total_time_s"],
        "rule_planned_duration_s": rule_report["planned_duration_s"],
        "rule_planning_time_s": rule_report["planning_time_s"],
        "rule_geometry_feasible": rule_transition["geometry_feasible"],
        "rule_minimum_eye_surface_clearance_mm": rule_transition.get("minimum_eye_surface_clearance_mm"),
        "rule_fallback_applied": rule_transition.get("fallback_applied", False),
        "rule_infeasibility_reasons": rule_failure_reasons,
        "adaptive_transfer_total_s": adaptive_transition["total_time_s"],
        "adaptive_planned_duration_s": adaptive_report["planned_duration_s"],
        "adaptive_planning_time_s": adaptive_report["planning_time_s"],
        "adaptive_geometry_feasible": adaptive_transition["geometry_feasible"],
        "adaptive_minimum_eye_surface_clearance_mm": adaptive_transition.get("minimum_eye_surface_clearance_mm"),
        "adaptive_time_saving_vs_fixed_s": fixed_saving,
        "adaptive_time_saving_vs_rule_s": rule_saving,
        "rule_and_adaptive_same": rule_same,
        "rule_failure_reasons": rule_failure_reasons,
    }
    return row, {"fixed": fixed_report, "rule": rule_report, "adaptive": adaptive_report}


def _write_results_csv(path, rows):
    fields = [
        "scenario_id", "angle_class", "depth_class", "angle_separation_deg",
        "start_C_mm", "end_C_mm", "adaptive_transfer_C_mm",
        "max_feasible_transfer_C_mm", "target_depth_cap_C_mm",
        "surface_depth_cap_C_mm", "minimum_eye_surface_clearance_mm",
        "adaptive_equals_shallower_target", "adaptive_equals_max_feasible",
        "fixed_depth_mm", "fixed_transfer_total_s", "fixed_planned_duration_s",
        "fixed_planning_time_s", "fixed_geometry_feasible",
        "rule_delta_mm", "rule_transfer_C_mm", "rule_transfer_total_s",
        "rule_planned_duration_s", "rule_planning_time_s", "rule_geometry_feasible",
        "rule_fallback_applied",
        "adaptive_transfer_total_s", "adaptive_planned_duration_s",
        "adaptive_planning_time_s", "adaptive_geometry_feasible",
        "adaptive_time_saving_vs_fixed_s", "adaptive_time_saving_vs_rule_s",
        "rule_and_adaptive_same", "limiting_reasons", "rule_failure_reasons",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for field in ("limiting_reasons", "rule_failure_reasons"):
                output[field] = json.dumps(output[field], ensure_ascii=False)
            writer.writerow(output)


def _parse_grid(value):
    try:
        grid = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("fixed depth grid must be comma-separated numbers") from error
    if not grid or any(not math.isfinite(item) or item < -4. for item in grid):
        raise argparse.ArgumentTypeError("fixed depth grid values must be finite and at least -4 mm")
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path,
                        default=HERE/"examples"/"depth_benchmark_scenarios.json")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE/"runs"/"历史记录"/"04_sessions"/"depth_benchmark"/f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--eye-margin-mm", type=float, default=.5)
    parser.add_argument("--fixed-depth-grid-mm", type=_parse_grid,
                        default=[0., .25, .5, .75, 1., 1.5, 2., 3., 4., 5.])
    parser.add_argument("--rule-delta-grid-mm", type=_parse_grid,
                        default=[0., .1, .25, .5, .75, 1., 1.5, 2.])
    args = parser.parse_args(argv)
    if not math.isfinite(args.eye_margin_mm) or args.eye_margin_mm < 0:
        parser.error("--eye-margin-mm must be finite and nonnegative")
    try:
        payload = _read_scenarios(args.scenarios)
        selected_fixed, fixed_scores = _select_parameter(
            payload["development"], args.fixed_depth_grid_mm, args.eye_margin_mm, "B")
        selected_rule, rule_scores = _select_parameter(
            payload["development"], args.rule_delta_grid_mm, args.eye_margin_mm, "R")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/"tuning_report.json", {
        "source": "development_scenarios_only",
        "fixed_depth_candidate_grid_mm": args.fixed_depth_grid_mm,
        "rule_delta_candidate_grid_mm": args.rule_delta_grid_mm,
        "eye_margin_mm": args.eye_margin_mm,
        "fixed_depth_baseline": {"selected": selected_fixed, "candidates": fixed_scores},
        "simple_rule_baseline": {"selected": selected_rule, "candidates": rule_scores},
    })

    rows = []
    for scenario in payload["evaluation"]:
        row, _ = _run_comparison(
            scenario, args.output_dir/scenario["id"], selected_fixed["parameter_mm"],
            selected_rule["parameter_mm"], args.eye_margin_mm)
        rows.append(row)

    stress_rows = []
    for scenario in payload.get("stress", []):
        row, _ = _run_comparison(
            scenario, args.output_dir/"stress"/scenario["id"],
            selected_fixed["parameter_mm"], selected_rule["parameter_mm"], args.eye_margin_mm)
        stress_rows.append(row)

    rule_failures = [row for row in [*rows, *stress_rows]
                     if row["rule_fallback_applied"] or
                     not row["rule_geometry_feasible"] or not row["rule_and_adaptive_same"]]

    summary = {
        "scope": "MuJoCo kinematic comparison; not hardware or tissue-safety validation",
        "scenario_file": str(args.scenarios),
        "eye_margin_mm": args.eye_margin_mm,
        "fixed_depth_baseline": {
            "selected_depth_mm": selected_fixed["parameter_mm"],
            "source": "independent development scenarios",
            "development_scenario_ids": [scenario["id"] for scenario in payload["development"]],
        },
        "simple_rule_baseline": {
            "rule": "C_transfer = min(C_start, C_end) - Delta",
            "selected_delta_mm": selected_rule["parameter_mm"],
            "source": "independent development scenarios",
            "development_scenario_ids": [scenario["id"] for scenario in payload["development"]],
        },
        "evaluation_scenario_ids": [scenario["id"] for scenario in payload["evaluation"]],
        "stress_scenario_ids": [scenario["id"] for scenario in payload.get("stress", [])],
        "results": rows,
        "stress_results": stress_rows,
        "rule_failure_search": {
            "found": bool(rule_failures),
            "scenario_ids": [row["scenario_id"] for row in rule_failures],
            "interpretation": (
                "simple rule diverged from adaptive or became infeasible"
                if rule_failures else
                "no rule-failure/adaptive-gain case found under the modeled ellipsoid and sampled constraints"
            ),
        },
    }
    write_json(args.output_dir/"benchmark_summary.json", summary)
    _write_results_csv(args.output_dir/"evaluation_results.csv", rows)
    print(json.dumps({
        "selected_fixed_depth_mm": selected_fixed["parameter_mm"],
        "selected_rule_delta_mm": selected_rule["parameter_mm"],
        "evaluation": [{
            "scenario_id": row["scenario_id"],
            "angle_class": row["angle_class"],
            "depth_class": row["depth_class"],
            "angle_separation_deg": row["angle_separation_deg"],
            "fixed_time_s": row["fixed_planned_duration_s"],
            "rule_time_s": row["rule_planned_duration_s"],
            "adaptive_time_s": row["adaptive_planned_duration_s"],
            "adaptive_time_saving_vs_fixed_s": row["adaptive_time_saving_vs_fixed_s"],
            "adaptive_time_saving_vs_rule_s": row["adaptive_time_saving_vs_rule_s"],
            "fixed_geometry_feasible": row["fixed_geometry_feasible"],
            "rule_geometry_feasible": row["rule_geometry_feasible"],
            "adaptive_geometry_feasible": row["adaptive_geometry_feasible"],
        } for row in rows],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
