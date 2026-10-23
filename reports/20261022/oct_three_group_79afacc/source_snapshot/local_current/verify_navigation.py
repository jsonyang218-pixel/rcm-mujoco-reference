#!/usr/bin/env python3
"""Eye-space sampling, B-branch regression and complete-cycle acceptance."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from rcm_reference import HERE, Reference, TASK_JOINTS
from rcm_navigation import (NavigationRejected, build_plan, execute_plan, solve_target,
                            task_si_units, task_user_units, write_json)


def verify_navigation(output_dir):
    reference = Reference()
    assert reference.encoder_bounds is None and reference.joint_bounds is None
    task_ids = [mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in TASK_JOINTS]
    assert not np.any(reference.model.jnt_limited[task_ids])
    rng = np.random.default_rng(20261008)
    gid = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
    center = reference.data.geom_xpos[gid]*1000
    rotation = reference.data.geom_xmat[gid].reshape(3, 3)
    radii = reference.model.geom_size[gid]*1000

    def eye_target(local_unit, name):
        # Select Cartesian eye points before solving. The test must not build
        # its targets from the same joint-window FK being checked.
        return {"id": name, "frame": "world",
                "point_mm": (center+rotation@(radii*np.asarray(local_unit))).tolist()}

    volume = []
    for i in range(512):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        volume.append(eye_target(direction*rng.uniform()**(1./3.)*.999, f"eye_volume_{i+1:03d}"))
    surface = []
    for i in range(64):
        z = 1.-2.*(i+.5)/64
        azimuth = i*math.pi*(3.-math.sqrt(5.))
        radius = math.sqrt(1.-z*z)
        surface.append(eye_target([radius*math.cos(azimuth), radius*math.sin(azimuth), z], f"eye_surface_{i+1:02d}"))
    poles = [eye_target(sign*np.eye(3)[axis], f"eye_pole_{axis}_{sign:+d}")
             for axis in range(3) for sign in (-1, 1)]
    regressions = [
        {"id": "original_deep", "frame": "robot", "point_mm": reference.config["reference_scene"]["original_deep_target_robot_mm"]},
        {"id": "local_demo", "frame": "robot", "point_mm": [.18705473377046364, 1.2338628825119726, 5.356546734019345]},
        {"id": "previous_B_branch_failure_1", "frame": "robot", "point_mm": [-.9885545725377721, 6.227669237226585, 2.4839317548239466]},
        {"id": "previous_B_branch_failure_2", "frame": "robot", "point_mm": [1.6850599984005847, 10.105723900879985, 2.941518733231527]},
        {"id": "Y_axis", "frame": "robot", "point_mm": [0., 5., 0.]},
        {"id": "at_RCM", "frame": "robot", "point_mm": [0., 0., 0.]},
        eye_target([0., 0., -.999], "near_entry"),
        eye_target([0., 0., 0.], "eye_center"),
    ]
    # Quantization near the B turning point must retain a real triangle.
    fold_alpha = math.degrees(math.asin(reference.nominal.geometry.offset_mm / (
        reference.nominal.geometry.link1_mm+reference.nominal.geometry.link2_mm)))
    for i, alpha in enumerate((fold_alpha-.1, fold_alpha, fold_alpha+.1)):
        theta = math.radians(alpha+reference.nominal.geometry.tilt_offset_deg)
        regressions.append({"id": f"B_turning_point_{i}", "frame": "robot",
                            "point_mm": [0., 12.*math.cos(theta), 12.*math.sin(theta)]})
    all_targets = volume+surface+poles+regressions
    original_qpos = reference.data.qpos.copy()
    original_sites = reference.model.site_pos.copy()
    original_geoms = reference.model.geom_pos.copy()
    solutions = [solve_target(reference, t["point_mm"], t["frame"], t["id"]) for t in all_targets]
    assert all(s["reachable"] for s in solutions), [s for s in solutions if not s["reachable"]]
    assert not any(s["position_limits_enabled"] for s in solutions)
    assert all(all(isinstance(v, int) for v in s["encoder_target_counts_ABC"]) for s in solutions)
    assert np.array_equal(original_qpos, reference.data.qpos)
    assert np.array_equal(original_sites, reference.model.site_pos)
    assert np.array_equal(original_geoms, reference.model.geom_pos)
    by_id = {s["id"]: s for s in solutions}
    assert by_id["local_demo"]["encoder_target_counts_ABC"][2] == int(round(893518+1.5*63822.71488))
    assert by_id["original_deep"]["requested_task_A_deg_B_deg_C_mm"][2] > 14.
    assert by_id["at_RCM"]["orientation_resolution"] == "retained_at_rcm"
    assert by_id["Y_axis"]["orientation_resolution"] == "azimuth_retained_on_Y_axis"

    # Check the independently compiled MuJoCo tree for every solved target.
    max_static_error_um = 0.
    for solution in solutions:
        q = task_si_units(solution["command_task_A_deg_B_deg_C_mm"])
        reference.set_task(q)
        error = float(np.linalg.norm(reference.site_mm("needle_tip")-solution["requested_world_mm"])*1000)
        max_static_error_um = max(max_static_error_um, error)
        assert error <= reference.config["navigation"]["endpoint_tolerance_um"]
        decoded = reference.task_from_encoders(solution["encoder_target_counts_ABC"], reference_task=q)
        assert np.allclose(decoded, q, atol=1e-10, rtol=0)
    reference.set_task(reference.home)

    # Run complete cycles for representative volume/surface points, every
    # pole and every explicit regression; all other points had static checks.
    targets = volume[:24]+surface[::8]+poles+regressions
    plan = build_plan(reference, targets)
    assert np.array_equal(original_qpos, reference.data.qpos)
    assert np.array_equal(original_sites, reference.model.site_pos)
    assert np.array_equal(original_geoms, reference.model.geom_pos)
    rate_limits = plan.settings["max_velocity_A_deg_B_deg_C_mm_per_s"]
    accel_limits = plan.settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"]
    for segment in plan.segments:
        for u in (0., .5, 1., (3.-math.sqrt(3.))/6., (3.+math.sqrt(3.))/6.):
            _, qd, qdd = segment.sample(segment.start_s+u*segment.duration_s)
            assert np.all(np.abs(task_user_units(qd)) <= np.asarray(rate_limits)+1e-9)
            assert np.all(np.abs(task_user_units(qdd)) <= np.asarray(accel_limits)+1e-9)
        for u in (0., 1.):
            _, qd, qdd = segment.sample(segment.start_s+u*segment.duration_s)
            assert np.max(np.abs(qd)) < 1e-10 and np.max(np.abs(qdd)) < 1e-10
    recorder = execute_plan(reference, plan)
    report = recorder.report()
    assert report["result"] == "PASS", report
    assert len(recorder.reached) == len(targets)
    assert np.allclose(reference.encoders(), reference.reset_encoders, atol=1e-6, rtol=0)
    planned = {s["id"]: s for s in plan.solutions}
    for row in recorder.rows:
        if row["phase"] in ("aim", "reset"):
            assert abs(row["C_mm"]) < 1e-10
        if row["phase"] in ("insert", "dwell", "return_C_zero"):
            command = planned[row["target_id"]]["command_task_A_deg_B_deg_C_mm"]
            assert abs(row["A_deg"]-command[0]) < 1e-9 and abs(row["B_deg"]-command[1]) < 1e-9
    output_dir = Path(output_dir)
    recorder.write(output_dir/"reachable_targets")
    write_json(output_dir/"eye_space_solutions.json", solutions)

    # Invalid input and targets outside the eye still get explicit rejection.
    # Rejecting a sampled encoder window is not acceptance for this task.
    bad_cases = [
        (eye_target([0., 0., 1.1], "outside_eye"), "OUTSIDE_EYE"),
        ({"id": "nonfinite", "frame": "robot", "point_mm": [math.nan, 0., 5.]}, "INVALID_POINT"),
        ({"id": "wrong_dimension", "frame": "robot", "point_mm": [0., 1.]}, "INVALID_POINT"),
    ]
    rejected = []
    before_qpos, before_site, before_geom = reference.data.qpos.copy(), reference.model.site_pos.copy(), reference.model.geom_pos.copy()
    for target, code in bad_cases:
        solution = solve_target(reference, target["point_mm"], target["frame"], target["id"])
        assert not solution["reachable"] and code in [i["code"] for i in solution["issues"]], solution
        rejected.append(solution)
    try:
        build_plan(reference, [targets[0], bad_cases[0][0]])
    except NavigationRejected as error:
        assert len(error.solutions) == 2
    else:
        raise AssertionError("Mixed valid/invalid batch was executed")
    assert np.array_equal(before_qpos, reference.data.qpos)
    assert np.array_equal(before_site, reference.model.site_pos)
    assert np.array_equal(before_geom, reference.model.geom_pos)
    write_json(output_dir/"rejected_targets.json", rejected)

    changed_config = copy.deepcopy(reference.config)
    changed_config["installation"]["euler_xyz_deg"] = [23., -17., 31.]
    changed_config["installation"]["rcm_world_mm"] = [31., -8., 14.]
    changed = Reference(changed_config)
    for name in ("original_deep", "previous_B_branch_failure_1"):
        robot_target = by_id[name]["requested_robot_mm"]
        solution = solve_target(changed, changed.robot_to_world(robot_target), "world", "transformed")
        assert solution["reachable"] and solution["encoder_target_counts_ABC"] == by_id[name]["encoder_target_counts_ABC"]

    # Preserve the prior azimuth at nonunique orientations.
    hint = task_si_units([17., -5., 1.])
    reference.set_task(hint)
    at_rcm = solve_target(reference, [0., 0., 0.], "robot")
    on_y = solve_target(reference, [0., 5., 0.], "robot")
    assert np.allclose(at_rcm["requested_task_A_deg_B_deg_C_mm"][:2], [17., -5.], atol=1e-10, rtol=0)
    assert abs(on_y["requested_task_A_deg_B_deg_C_mm"][0]-17.) < 1e-10
    recovery = build_plan(reference, [regressions[0]])
    assert [s.phase for s in recovery.segments[:2]] == ["prepare_C_zero", "prepare_reset"]
    for t in np.linspace(0., recovery.segments[0].end_s, 31):
        q, _, _, _ = recovery.sample(float(t))
        assert np.allclose(q[:2], hint[:2], atol=1e-12, rtol=0)
    recovery_recorder = execute_plan(reference, recovery)
    assert recovery_recorder.report()["result"] == "PASS"
    recovery_recorder.write(output_dir/"nonzero_start")
    result = {
        "result": "PASS", "scope": "Unlimited MuJoCo kinematic navigation; sampled eye volume and reference surface, not measured robot reach",
        "mujoco_version": mujoco.__version__, "position_limits_enabled": False,
        "target_sampling": "Cartesian points from the compiled eye ellipsoid, independent of joint-window FK",
        "eye_volume_target_count": len(volume), "eye_surface_target_count": len(surface)+len(poles),
        "regression_target_count": len(regressions), "solved_target_count": len(solutions),
        "completed_cycle_target_count": len(targets), "rejected_target_count": len(rejected),
        "trajectory_sample_count": report["sample_count"],
        "max_static_tip_error_um": max_static_error_um,
        "max_reached_tip_error_um": report["max_reached_tip_error_um"],
        "max_geometry_errors_um": report["max_geometry_errors_um"],
        "original_deep_target": by_id["original_deep"],
        "B_other_inverse_branch_and_turning_point": "PASS",
        "RCM_and_Y_axis_orientation_retention": "PASS",
        "integer_encoder_target_with_4mm_tip_offset": "PASS",
        "continuous_velocity_and_acceleration_extrema": "PASS",
        "preflight_nonmutation_and_atomic_batch_rejection": "PASS",
        "world_robot_coordinate_conversion": "PASS", "nonzero_start_C_first_recovery": "PASS",
        "cycle_order_and_final_zero": "PASS", "feedback_controller_used": False,
    }
    write_json(output_dir/"acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"acceptance")
    args = parser.parse_args()
    print(json.dumps(verify_navigation(args.output_dir), ensure_ascii=False, indent=2, allow_nan=False))
