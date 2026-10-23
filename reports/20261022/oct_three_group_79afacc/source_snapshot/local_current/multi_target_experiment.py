#!/usr/bin/env python3
"""Compare four two-target navigation strategies in the MuJoCo reference."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from rcm_navigation import (NavigationPlan, NavigationRejected, NavigationRecorder,
                            Segment, build_plan, execute_plan, load_targets,
                            navigation_settings, solve_target, task_si_units,
                            task_user_units, write_json)
from rcm_reference import HERE, Reference


DECISION_TOL_MM = 1e-4


METHODS = {
    "A": "fixed order, full reset after every target",
    "B": "fixed order, fixed retract depth",
    "R": "fixed order, min target depth minus tuned delta",
    "C": "fixed order, adaptive retract depth",
    "D": "optimized order, adaptive retract depth",
}


def _add_segment(reference, segments, state, phase, q, target_index, settings, hold=None):
    q = reference.validate_task(q).copy()
    previous, elapsed = state
    delta = np.abs(q-previous)
    if hold is None:
        vmax = task_si_units(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
        amax = task_si_units(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
        duration = max(
            settings["minimum_move_time_s"],
            float(np.max(1.875*delta/vmax)),
            float(np.max(np.sqrt((10.*math.sqrt(3.)/3.)*delta/amax))),
        )
    else:
        duration = float(hold)
    if duration > 0:
        segments.append(Segment(elapsed, duration, phase, target_index,
                                previous.copy(), q.copy()))
        elapsed += duration
    state[0], state[1] = q, elapsed


def _phase_duration(plan, phase, occurrence=0):
    """Return the duration of one named phase in a plan."""
    matches = [segment.duration_s for segment in plan.segments if segment.phase == phase]
    if occurrence >= len(matches):
        raise ValueError(f"Missing phase {phase!r} occurrence {occurrence}")
    return float(matches[occurrence])


def _phase_rows(recorder, phase, occurrence):
    rows = [row for row in recorder.rows if row["phase"] == phase]
    if occurrence >= len(rows):
        return []
    # A phase has one contiguous sample block per target/transition.
    blocks = []
    current = []
    previous_target = None
    for row in rows:
        if previous_target is not None and row["target_id"] != previous_target:
            blocks.append(current)
            current = []
        current.append(row)
        previous_target = row["target_id"]
    if current:
        blocks.append(current)
    return blocks[occurrence] if occurrence < len(blocks) else []


def _transition_geometry(recorder, phase_occurrences):
    """Summarize sampled eye geometry for one transition."""
    rows = []
    for phase, occurrence in phase_occurrences:
        rows.extend(_phase_rows(recorder, phase, occurrence))
    tip_outside = sum(not bool(row["tip_inside_eye"]) for row in rows)
    junction_outside = sum(not bool(row.get("junction_inside_eye", True)) for row in rows)
    reasons = []
    if tip_outside:
        reasons.append({
            "code": "TIP_OUTSIDE_REFERENCE_EYE",
            "message": f"needle tip is outside the reference eye ellipsoid in {tip_outside} samples",
        })
    if junction_outside:
        reasons.append({
            "code": "JUNCTION_OUTSIDE_REFERENCE_EYE",
            "message": f"needle junction is outside the reference eye ellipsoid in {junction_outside} samples",
        })
    return {
        "geometry_feasible": not reasons,
        "geometry_samples": len(rows),
        "tip_outside_eye_sample_count": tip_outside,
        "junction_outside_eye_sample_count": junction_outside,
        "infeasibility_reasons": reasons,
        "geometry_scope": "sampled reference ellipsoid and needle centerline; no tissue/contact validation",
    }


def _annotate_transition_geometry(recorder, transitions):
    """Attach measured geometry status to transition records in execution order."""
    for index, transition in enumerate(transitions):
        if transition.get("method") == "A":
            phase_occurrences = [("return_C_zero", index), ("reset", index),
                                 ("aim", index+1), ("insert", index+1)]
        else:
            phase_occurrences = [("retract_between_targets", index),
                                 ("reorient_between_targets", index),
                                 ("insert", index+1)]
        geometry = _transition_geometry(recorder, phase_occurrences)
        selected = transition.get("selected_transfer_C_mm", transition["retract_to_C_mm"])
        target_cap = transition.get("target_depth_cap_C_mm")
        surface_cap = transition.get("surface_depth_cap_C_mm")
        if target_cap is not None and selected > target_cap+DECISION_TOL_MM:
            geometry["infeasibility_reasons"].append({
                "code": "NOT_RETRACTED_TO_SHALLOWER_TARGET",
                "message": "selected transfer C is deeper than the shallower target; both endpoints are not cleared",
            })
        if surface_cap is not None and selected > surface_cap+DECISION_TOL_MM:
            geometry["infeasibility_reasons"].append({
                "code": "INSUFFICIENT_REFERENCE_EYE_MARGIN",
                "message": "selected transfer C exceeds the sampled eye-surface cap after margin",
            })
        geometry["geometry_feasible"] = not geometry["infeasibility_reasons"]
        transition.update(geometry)
        transition["transition_index"] = index + 1
    return transitions


def _write_transition_csv(path, transitions):
    fields = [
        "transition_index", "from", "to", "retract_to_C_mm",
        "start_C_mm", "end_C_mm", "max_feasible_transfer_C_mm",
        "target_depth_cap_C_mm", "surface_depth_cap_C_mm", "selection_strategy",
        "retraction_amount_mm", "retract_time_s", "reorient_time_s",
        "insert_time_s", "total_time_s", "geometry_feasible",
        "geometry_samples", "tip_outside_eye_sample_count",
        "junction_outside_eye_sample_count", "limiting_reasons", "infeasibility_reasons",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, transition in enumerate(transitions, start=1):
            row = {field: transition.get(field, index if field == "transition_index" else None)
                   for field in fields}
            for field in ("limiting_reasons", "infeasibility_reasons"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)


def _solve_in_order(targets):
    """Solve with the prior target as the B-branch hint, then restore home."""
    reference = Reference()
    solutions = []
    for target in targets:
        solution = solve_target(reference, target["point_mm"],
                                target.get("frame", "world"), target["id"])
        if not solution["reachable"]:
            raise NavigationRejected([solution])
        solutions.append(solution)
        reference.set_task(task_si_units(solution["command_task_A_deg_B_deg_C_mm"]))
    reference.set_task(reference.home)
    return reference, solutions


def _eye_radial_limit_mm(reference, axis_world):
    """Return the positive ray/ellipsoid intersection from the RCM."""
    gid = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
    rotation = reference.data.geom_xmat[gid].reshape(3, 3)
    center = reference.data.geom_xpos[gid]*1000.
    radii = reference.model.geom_size[gid]*1000.
    rcm = reference.site_mm("rcm")
    origin = rotation.T@(rcm-center)/radii
    direction = rotation.T@np.asarray(axis_world, dtype=float)/radii
    a = float(direction@direction)
    b = float(2.*origin@direction)
    c = float(origin@origin-1.)
    discriminant = b*b-4.*a*c
    if discriminant < -1e-10 or a <= 0:
        raise ValueError("Target transition has no usable eye-surface intersection")
    roots = sorted(((-b-math.sqrt(max(0., discriminant)))/(2.*a),
                    (-b+math.sqrt(max(0., discriminant)))/(2.*a)))
    positive = [root for root in roots if root >= -1e-9]
    if not positive:
        raise ValueError("Target transition points away from the modeled eye")
    return max(0., positive[-1])


def _axis_world(reference, q):
    state = reference.state_from_task(q)
    axis = reference.rotation_robot@np.asarray(reference.nominal.needle_axis(state))
    return axis/np.linalg.norm(axis)


def _slerp(a, b, fraction):
    dot = float(np.clip(a@b, -1., 1.))
    if dot > .999999:
        result = (1.-fraction)*a+fraction*b
        return result/np.linalg.norm(result)
    angle = math.acos(dot)
    result = (math.sin((1.-fraction)*angle)*a + math.sin(fraction*angle)*b)/math.sin(angle)
    return result/np.linalg.norm(result)


def adaptive_retract_analysis(reference, first_q, second_q, margin_mm):
    """Explain the sampled maximum feasible C coordinate for a target pair."""
    first_axis, second_axis = _axis_world(reference, first_q), _axis_world(reference, second_q)
    first_c_mm, second_c_mm = float(first_q[2]*1000.), float(second_q[2]*1000.)
    first_length = reference.tip_reference_mm+first_c_mm
    second_length = reference.tip_reference_mm+second_c_mm
    limits = [_eye_radial_limit_mm(reference, _slerp(first_axis, second_axis, i/64.))
              for i in range(65)]
    target_cap = min(first_c_mm, second_c_mm)
    surface_cap = min(limits)-margin_mm-reference.tip_reference_mm
    lower_bound = -reference.tip_reference_mm
    max_feasible = min(target_cap, surface_cap)
    selected = max(lower_bound, max_feasible)
    reasons = []
    if target_cap <= surface_cap+1e-9:
        reasons.append("shallower_target_depth_cap")
    if surface_cap <= target_cap+1e-9:
        reasons.append("reference_eye_surface_margin_cap")
    if max_feasible < lower_bound:
        reasons.append("minimum_recoverable_C_bound")
    return {
        "target_depth_cap_C_mm": float(target_cap),
        "surface_depth_cap_C_mm": float(surface_cap),
        "max_feasible_transfer_C_mm": float(max_feasible),
        "selected_adaptive_C_mm": float(selected),
        "adaptive_equals_shallower_target": abs(selected-target_cap) <= 1e-6,
        "adaptive_equals_max_feasible": abs(selected-max_feasible) <= 1e-6,
        "limiting_reasons": reasons,
        "surface_margin_mm": float(margin_mm),
        "surface_search_samples": len(limits),
        "geometry_constraints_checked": [
            "reference_eye_ellipsoid_radial_arc",
            "needle_tip_and_junction_samples",
        ],
        "geometry_constraints_not_modeled": [
            "needle_shaft_clearance",
            "target_tissue_exit",
            "entry_aperture",
            "force_feedback",
        ],
    }


def adaptive_retract_mm(reference, first_q, second_q, margin_mm):
    """Choose the deepest common radius that stays inside the eye arc."""
    return adaptive_retract_analysis(reference, first_q, second_q, margin_mm)["selected_adaptive_C_mm"]


def _transition_decision(reference, first_q, second_q, selected_c_mm, margin_mm, strategy):
    analysis = adaptive_retract_analysis(reference, first_q, second_q, margin_mm)
    selected = float(selected_c_mm)
    reasons = list(analysis["limiting_reasons"])
    if strategy == "uniform_fixed_depth":
        reasons = ["uniform_fixed_depth_baseline", *reasons]
    elif strategy == "simple_min_depth_rule":
        reasons = ["min_target_depth_minus_delta_rule", *reasons]
    if selected > analysis["max_feasible_transfer_C_mm"]+DECISION_TOL_MM:
        reasons.append("selected_C_exceeds_sampled_maximum")
    analysis.update({
        "selected_transfer_C_mm": selected,
        "selection_strategy": strategy,
        "minimum_eye_surface_clearance_mm": float(
            analysis["surface_depth_cap_C_mm"]+margin_mm-selected),
        "limiting_reasons": list(dict.fromkeys(reasons)),
    })
    return analysis


def _continuous_plan(reference, solutions, mode, fixed_retract_mm=0., eye_margin_mm=.5):
    settings = navigation_settings(reference)
    segments = []
    state = [reference.home.copy(), 0.]
    transitions = []
    zero = reference.home[2]
    for index, solution in enumerate(solutions):
        goal = task_si_units(solution["command_task_A_deg_B_deg_C_mm"])
        aimed = goal.copy()
        aimed[2] = zero
        if index == 0:
            _add_segment(reference, segments, state, "aim", aimed, index, settings)
        else:
            previous_goal = task_si_units(solutions[index-1]["command_task_A_deg_B_deg_C_mm"])
            if mode == "B":
                retract_mm = fixed_retract_mm
                strategy = "uniform_fixed_depth"
                requested_retract_mm = retract_mm
                fallback_applied = False
                fallback_reason = None
            elif mode == "R":
                target_cap = min(previous_goal[2], goal[2])*1000.
                requested_retract_mm = target_cap-fixed_retract_mm
                analysis = adaptive_retract_analysis(reference, previous_goal, goal, eye_margin_mm)
                retract_mm = min(requested_retract_mm, analysis["max_feasible_transfer_C_mm"])
                retract_mm = max(-reference.tip_reference_mm, retract_mm)
                strategy = "simple_min_depth_rule"
                fallback_applied = bool(abs(retract_mm-requested_retract_mm) > DECISION_TOL_MM)
                fallback_reason = "simple_rule_requested_C_exceeds_feasible_cap" if fallback_applied else None
            else:
                retract_mm = adaptive_retract_mm(reference, previous_goal, goal, eye_margin_mm)
                strategy = "adaptive_sampled_maximum"
                requested_retract_mm = retract_mm
                fallback_applied = False
                fallback_reason = None
            retract = previous_goal.copy()
            retract[2] = retract_mm*.001
            transition = {
                "method": mode,
                "from": solutions[index-1]["id"],
                "to": solution["id"],
                "retract_to_C_mm": float(retract_mm),
                "retraction_amount_mm": float(max(0., previous_goal[2]*1000.-retract_mm)),
                "start_C_mm": float(previous_goal[2]*1000.),
                "end_C_mm": float(goal[2]*1000.),
                "requested_transfer_C_mm": float(requested_retract_mm),
                "fallback_applied": fallback_applied,
                "fallback_reason": fallback_reason,
            }
            transition.update(_transition_decision(
                reference, previous_goal, goal, retract_mm, eye_margin_mm, strategy))
            if mode == "R":
                transition["rule_delta_mm"] = float(fixed_retract_mm)
            transition_start = state[1]
            retract_start = state[1]
            _add_segment(reference, segments, state, "retract_between_targets",
                         retract, index-1, settings)
            transition["retract_time_s"] = float(state[1]-retract_start)
            reorient_start = state[1]
            reoriented = goal.copy()
            reoriented[2] = retract[2]
            _add_segment(reference, segments, state, "reorient_between_targets",
                         reoriented, index, settings)
            transition["reorient_time_s"] = float(state[1]-reorient_start)
            aimed = reoriented.copy()
            insert_start = state[1]
        _add_segment(reference, segments, state, "insert", goal, index, settings)
        if index > 0:
            transition["insert_time_s"] = float(state[1]-insert_start)
            transition["total_time_s"] = float(state[1]-transition_start)
            transitions.append(transition)
        _add_segment(reference, segments, state, "dwell", goal, index, settings,
                     hold=settings["dwell_time_s"])
    final_aim = task_si_units(solutions[-1]["command_task_A_deg_B_deg_C_mm"])
    final_aim[2] = zero
    _add_segment(reference, segments, state, "final_retract", final_aim,
                 len(solutions)-1, settings)
    _add_segment(reference, segments, state, "final_reset", reference.home,
                 len(solutions)-1, settings)
    return NavigationPlan(solutions, segments, settings), transitions


def _path_length(plan):
    return float(sum(np.sum(np.abs(np.asarray(task_user_units(s.q_end))-
                                   np.asarray(task_user_units(s.q_start))))
                     for s in plan.segments))


def _reset_transitions(plan):
    """Describe A's full reset as a comparable target-to-target transition."""
    transitions = []
    for index in range(len(plan.solutions)-1):
        first = plan.solutions[index]
        second = plan.solutions[index+1]
        first_goal = task_si_units(first["command_task_A_deg_B_deg_C_mm"])
        retract_time = _phase_duration(plan, "return_C_zero", index)
        reset_time = _phase_duration(plan, "reset", index)
        aim_time = _phase_duration(plan, "aim", index+1)
        insert_time = _phase_duration(plan, "insert", index+1)
        transitions.append({
            "method": "A",
            "from": first["id"],
            "to": second["id"],
            "retract_to_C_mm": 0.0,
            "retraction_amount_mm": float(max(0., first_goal[2]*1000.)),
            "retract_time_s": retract_time,
            "reorient_time_s": reset_time + aim_time,
            "insert_time_s": insert_time,
            "total_time_s": retract_time + reset_time + aim_time + insert_time,
        })
    return transitions


def _run_method(name, targets, output_dir, fixed_retract_mm, eye_margin_mm):
    planning_started = time.perf_counter()
    if name == "A":
        reference = Reference()
        plan = build_plan(reference, targets)
        transitions = _reset_transitions(plan)
        order = [target["id"] for target in targets]
    else:
        order_targets = list(targets)
        reference, solutions = _solve_in_order(order_targets)
        plan, transitions = _continuous_plan(
            reference, solutions, name, fixed_retract_mm, eye_margin_mm)
        order = [solution["id"] for solution in solutions]
    planning_time_s = time.perf_counter()-planning_started
    recorder = execute_plan(reference, plan)
    _annotate_transition_geometry(recorder, transitions)
    recorder.write(output_dir)
    report = recorder.report()
    report.update({"method": name, "method_description": METHODS[name],
                   "target_order": order,
                   "transitions": transitions,
                   "transition_metrics": transitions,
                   "transition_retracts_mm": [item["retract_to_C_mm"] for item in transitions],
                   "planning_time_s": planning_time_s,
                   "joint_path_length_deg_deg_mm": _path_length(plan)})
    write_json(Path(output_dir)/"experiment_report.json", report)
    _write_transition_csv(Path(output_dir)/"transition_metrics.csv", transitions)
    return report, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=HERE/"examples"/"two_targets.json")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE/"runs"/"历史记录"/"04_sessions"/"multi_target"/f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--fixed-retract-mm", type=float, default=0.,
                        help="Method B C-axis depth between targets, relative to zero")
    parser.add_argument("--eye-margin-mm", type=float, default=.5,
                        help="Method C/D margin from the modeled eye surface")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fixed_retract_mm) or args.fixed_retract_mm < -4.:
        parser.error("--fixed-retract-mm must be finite and at least -4 mm")
    if not math.isfinite(args.eye_margin_mm) or args.eye_margin_mm < 0:
        parser.error("--eye-margin-mm must be finite and nonnegative")
    try:
        targets = load_targets(args.targets)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if len(targets) != 2:
        parser.error("This experiment currently requires exactly two targets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ("A", "B", "C"):
        report, _ = _run_method(name, targets, args.output_dir/name,
                                args.fixed_retract_mm, args.eye_margin_mm)
        reports[name] = report

    candidates = []
    for permutation in itertools.permutations(targets):
        reference, solutions = _solve_in_order(list(permutation))
        plan, transitions = _continuous_plan(reference, solutions, "C", 0., args.eye_margin_mm)
        candidates.append({"order": [s["id"] for s in solutions],
                           "duration_s": plan.duration_s,
                           "transitions": transitions})
    best_duration = min(item["duration_s"] for item in candidates)
    optimal_orders = [item["order"] for item in candidates
                      if abs(item["duration_s"]-best_duration) <= 1e-9]
    selected = next(item for item in candidates if item["order"] == optimal_orders[0])
    selected_targets = sorted(targets, key=lambda target: selected["order"].index(target["id"]))
    report, _ = _run_method("D", selected_targets, args.output_dir/"D",
                            args.fixed_retract_mm, args.eye_margin_mm)
    report["order_candidates"] = candidates
    report["optimal_orders"] = optimal_orders
    report["order_optimization_gain_s"] = reports["C"]["planned_duration_s"]-report["planned_duration_s"]
    write_json(args.output_dir/"D"/"experiment_report.json", report)
    reports["D"] = report
    summary = {"target_count": 2, "targets": [target["id"] for target in targets],
               "fixed_retract_mm": args.fixed_retract_mm,
               "eye_margin_mm": args.eye_margin_mm,
               "methods": reports,
               "transition_metrics_by_method": {
                   name: report["transition_metrics"] for name, report in reports.items()
               },
               "order_candidates_for_D": candidates,
               "optimal_orders_for_D": optimal_orders,
               "selected_order_for_D": selected["order"],
               "order_optimization_gain_s": reports["C"]["planned_duration_s"]-report["planned_duration_s"]}
    write_json(args.output_dir/"summary.json", summary)
    print(json.dumps({name: {"result": report["result"],
                             "duration_s": report["planned_duration_s"],
                             "target_order": report["target_order"],
                             "transition_metrics": report["transition_metrics"],
                             "transition_retracts_mm": report["transition_retracts_mm"],
                             "joint_path_length_deg_deg_mm": report["joint_path_length_deg_deg_mm"]}
                     for name, report in reports.items()},
                  ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
