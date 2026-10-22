"""Checks run against the real MuJoCo engine, with deterministic sampling."""
from __future__ import annotations

import ast
import copy
import json
import math
import platform
import sys
from itertools import product

import mujoco
import numpy as np

from rcm_reference import HERE, Reference, TASK_JOINTS, demo_motion, navigation_motion


def verify(reference):
    saved = reference.current_task()
    reference.set_task(reference.home)
    reset_metrics = reference.metrics()
    if reference.reference_enabled:
        assert np.allclose(reference.encoders(), reference.reset_encoders, atol=1e-6, rtol=0)
        assert np.linalg.norm(reference.current_task()) < 1e-12
    # Check the actual MuJoCo eye geometry, independently of the RCM marker.
    # The ellipsoid orientation must align its entry pole with the reset axis.
    eye_id = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
    eye_local_rcm = reference.data.geom_xmat[eye_id].reshape(3, 3).T @ (
        reference.site_mm("rcm")*.001-reference.data.geom_xpos[eye_id])
    eye_surface_value = float(np.sum((eye_local_rcm/reference.model.geom_size[eye_id])**2))
    assert abs(eye_surface_value-1.) < 1e-12, "RCM is not on the eye surface"
    eye_radial_gap_um = float(np.linalg.norm(eye_local_rcm)*(1.-1./math.sqrt(eye_surface_value))*1e6)
    reset_axis = reference.site_mm("needle_tip")-reference.site_mm("needle_axis_back")
    reset_axis /= np.linalg.norm(reset_axis)
    eye_from_rcm = reference.data.geom_xpos[eye_id]*1000-reference.site_mm("rcm")
    reset_axis_eye_center_um = float(np.linalg.norm(np.cross(eye_from_rcm, reset_axis))*1000)
    maxima = {name: 0.0 for name in ("rcm_drift_um", "needle_axis_rcm_distance_um", "fk_agreement_um",
                                   "upper_link_closure_um", "lower_link_closure_um", "distal_length_error_um",
                                   "guide_parallel_error", "equality_residual_rad", "fixed_base_error_m", "fixed_camera_error_m")}
    base_id = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_BODY, "fixed_inclined_mount")
    base_position, base_rotation = reference.data.xpos[base_id].copy(), reference.data.xmat[base_id].copy()
    cam_position, cam_rotation = reference.data.cam_xpos.copy(), reference.data.cam_xmat.copy()
    samples = 0
    if reference.encoder_bounds is None:
        task_ids = [mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in TASK_JOINTS]
        assert not np.any(reference.model.jnt_limited[task_ids]), "Unmeasured joint limits were enabled"
        reference.set_task(np.zeros(3))
        if reference.tip_reference_mm == reference.distal_mm:
            assert np.linalg.norm(reference.site_mm("needle_junction")-reference.site_mm("rcm")) < 1e-10
    for geom_name, parameter in (("needle_proximal", "proximal_radius_mm"), ("needle_distal", "distal_radius_mm")):
        gid = mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert abs(reference.model.geom_size[gid, 0]*1000-reference.config["needle"][parameter]) < 1e-12
    assert reference.model.geom("needle_distal_tip").type == mujoco.mjtGeom.mjGEOM_MESH
    assert abs(reference.model.geom("needle_distal").size[1]*2000-
               (reference.distal_mm-reference.distal_tip_mm)) < 1e-9
    for name in ("needle_tip", "needle_junction"):
        assert reference.model.site_rgba[reference.sites[name], 3] == 0, "Tip/junction marker should be hidden"

    def check(q):
        nonlocal samples
        reference.set_task(q)
        metrics = reference.metrics()
        for name in ("rcm_drift_um", "needle_axis_rcm_distance_um", "fk_agreement_um", "upper_link_closure_um", "lower_link_closure_um"):
            maxima[name] = max(maxima[name], metrics[name])
        maxima["distal_length_error_um"] = max(maxima["distal_length_error_um"], abs(metrics["distal_length_mm"]-reference.distal_mm)*1000)
        tipdir = reference.site_mm("needle_tip")-reference.site_mm("needle_axis_back")
        guide = reference.site_mm("guide_top")-reference.site_mm("guide_bottom")
        cross = np.linalg.norm(np.cross(tipdir/np.linalg.norm(tipdir), guide/np.linalg.norm(guide)))
        maxima["guide_parallel_error"] = max(maxima["guide_parallel_error"], float(cross))
        equality = np.asarray(reference.data.efc_pos)[np.asarray(reference.data.efc_type) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]
        if equality.size:
            maxima["equality_residual_rad"] = max(maxima["equality_residual_rad"], float(np.max(np.abs(equality))))
        maxima["fixed_base_error_m"] = max(maxima["fixed_base_error_m"], float(np.linalg.norm(reference.data.xpos[base_id]-base_position)),
                                           float(np.linalg.norm(reference.data.xmat[base_id]-base_rotation)))
        maxima["fixed_camera_error_m"] = max(maxima["fixed_camera_error_m"], float(np.linalg.norm(reference.data.cam_xpos-cam_position)),
                                             float(np.linalg.norm(reference.data.cam_xmat-cam_rotation)))
        samples += 1

    rng = np.random.default_rng(20261008)
    bounds = reference.joint_bounds
    if bounds is None:
        # Sampling intervals only, including both B inverses and retraction.
        bounds = np.array([np.radians([-180, 180]),
                           np.radians(np.array([-20., 150.])-reference.alpha_home),
                           np.array([-reference.tip_reference_mm, 24.-reference.tip_reference_mm])*.001])
    for q in rng.uniform(bounds[:, 0], bounds[:, 1], size=(1000, 3)):
        check(q)
    demo_status = {}
    for mode in ("a", "b", "c", "abc"):
        motion = demo_motion(reference, mode)
        for t in np.linspace(0, motion.duration, 181):
            q = motion.sample(t)
            if mode != "abc":
                others = [i for i in range(3) if i != "abc".index(mode)]
                assert np.allclose(q[others], reference.home[others], atol=1e-12, rtol=0), "Other task axes moved"
            check(q)
        demo_status[mode.upper()] = "PASS"
    reference.set_task(reference.home)
    c_initial = reference.current_task()[2]
    aim = navigation_motion(reference, reach=False)
    for t in np.linspace(0, aim.duration, 121):
        q = aim.sample(t)
        assert abs(q[2]-c_initial) < 1e-12, "Aim-only changed C"
        check(q)
    reference.set_task(reference.home)
    reach = navigation_motion(reference, reach=True)
    for t in np.linspace(0, reach.duration, 181):
        check(reach.sample(t))
    reached = reference.metrics()
    assert reached["tip_target_error_um"] < 1e-6
    assert all(v < 1e-6 for name, v in maxima.items() if name.endswith("_um"))
    assert maxima["guide_parallel_error"] < 1e-12
    assert maxima["equality_residual_rad"] < 1e-12
    assert maxima["fixed_base_error_m"] < 1e-12 and maxima["fixed_camera_error_m"] < 1e-12

    reference.set_task(reference.home)
    before_tip = reference.site_mm("needle_tip").copy()
    before_encoder = reference.encoders()[2]
    step_pose = reference.home.copy()
    step_pose[2] += reference.manual_steps["C_mm"]*.001
    reference.set_task(step_pose)
    manual_c_step_um = float(np.linalg.norm(reference.site_mm("needle_tip")-before_tip)*1000)
    assert abs(manual_c_step_um-reference.manual_steps["C_mm"]*1000) < 1e-6
    manual_c_encoder_delta = reference.encoders()[2]-before_encoder

    # Test source encoder roundtrips without reusing MuJoCo forward results.
    worst_encoder_roundtrip_um = 0.
    for q in rng.uniform(bounds[:, 0], bounds[:, 1], size=(100, 3)):
        reference.set_task(q)
        encoder = tuple(int(v) for v in reference.encoders(q))
        back = reference.task_from_encoders(encoder)
        p_before = reference.site_mm("needle_tip").copy()
        reference.set_task(back)
        worst_encoder_roundtrip_um = max(worst_encoder_roundtrip_um, float(np.linalg.norm(reference.site_mm("needle_tip")-p_before)*1000))
    assert worst_encoder_roundtrip_um < 2.0
    recorded_endpoint_cases = 0
    if reference.recorded_encoder_bounds is not None:
        for raw_encoder in product(*reference.recorded_encoder_bounds):
            reference.set_task(reference.task_from_encoders(raw_encoder))
            assert np.allclose(reference.encoders(), raw_encoder, atol=1e-6, rtol=0)
            assert reference.metrics()["fk_agreement_um"] < 1e-6
            recorded_endpoint_cases += 1

    # Changing installation affects only the fixed rigid transform, not the
    # robot-frame kinematics or tip-to-junction distance.
    changed_config = copy.deepcopy(reference.config)
    changed_config["installation"]["euler_xyz_deg"] = [23., -17., 31.]
    changed_config["installation"]["rcm_world_mm"] = [31., -8., 14.]
    changed = Reference(changed_config)
    for q in rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, 3)):
        changed.set_task(q)
        metrics = changed.metrics()
        assert metrics["fk_agreement_um"] < 1e-6
        assert metrics["needle_axis_rcm_distance_um"] < 1e-6
        assert metrics["upper_link_closure_um"] < 1e-6

    # Synthetic encoder bounds test enforcement logic; these values are NOT
    # inserted into the delivered calibration config and are NOT measured.
    limits_config = copy.deepcopy(reference.config)
    limits_config["encoder_reference"] = {"enabled": False}
    limits_config["encoders"]["calibrated"] = True
    limits_config["encoders"]["origin"] = [128., -270., 810.]
    limits_config["encoders"]["direction"] = [-1., -1., 1.]
    limits_config["measured_limits"]["enabled"] = False
    synthetic = Reference(limits_config)
    qlo, qhi = np.array([-.3, -.1, 0.]), np.array([.3, .1, .01])
    elo, ehi = synthetic.encoders(qlo), synthetic.encoders(qhi)
    limits_config["measured_limits"] = {
        "enabled": True,
        "encoder_counts": {axis: sorted([elo[i], ehi[i]]) for i, axis in enumerate("ABC")},
        "reserve_counts": {axis: 1. for axis in "ABC"}}
    limited = Reference(limits_config)
    assert np.all(limited.model.jnt_limited[[mujoco.mj_name2id(limited.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in TASK_JOINTS]])
    for axis in range(3):
        q = limited.home.copy()
        q[axis] = limited.joint_bounds[axis, 1]+1e-5
        original = limited.data.qpos.copy()
        try:
            limited.set_task(q)
            raise AssertionError("Out-of-range pose was accepted")
        except ValueError:
            assert np.array_equal(original, limited.data.qpos), "Rejected command changed pose"
    incomplete = copy.deepcopy(limits_config)
    incomplete["measured_limits"]["encoder_counts"]["B"] = None
    try:
        Reference(incomplete)
        raise AssertionError("Incomplete enabled limits were accepted")
    except ValueError:
        pass

    # Verify an explicit, nonzero reset pose and a different needle offset.
    # This exercises reference-frame rotations rather than only zero origins.
    alternate_config = copy.deepcopy(reference.config)
    if reference.reference_enabled:
        e = reference.nominal.encoders
        alternate_config["encoder_reference"]["robot_counts_ABC"] = [
            e.origin[0]+e.direction[0]*1.25*e.a_counts_per_degree,
            e.origin[1]+e.direction[1]*reference.nominal.b_displacement(55.)*e.bc_counts_per_mm,
            900000.]
        alternate_config["encoder_reference"]["needle_tip_distance_at_reference_mm"] = 6.
        alternate = Reference(alternate_config)
        assert np.allclose(alternate.encoders(), alternate.reset_encoders, atol=1e-6, rtol=0)
        alternate_bounds = alternate.joint_bounds
        if alternate_bounds is None:
            alternate_bounds = np.array([np.radians([-180, 180]),
                                        np.radians(np.array([-20., 150.])-alternate.alpha_home),
                                        np.array([-alternate.tip_reference_mm, 24.-alternate.tip_reference_mm])*.001])
        for q in rng.uniform(alternate_bounds[:, 0], alternate_bounds[:, 1], size=(50, 3)):
            alternate.set_task(q)
            metrics = alternate.metrics()
            assert metrics["fk_agreement_um"] < 1e-6
            assert metrics["needle_axis_rcm_distance_um"] < 1e-6
            assert metrics["upper_link_closure_um"] < 1e-6
            assert abs(metrics["distal_length_mm"]-reference.distal_mm) < 1e-9
    deep_target_check = "not applicable"
    if reference.config["reference_scene"].get("original_deep_target_robot_mm") is not None:
        original_target = reference.nominal.inverse(reference.config["reference_scene"]["original_deep_target_robot_mm"], enforce_limits=False)
        try:
            reference.validate_task(reference.task_from_state(original_target))
        except ValueError:
            deep_target_check = "rejected by configured limits or geometry"
        else:
            deep_target_check = "reachable with position limits disabled" if reference.joint_bounds is None else "within current configured bounds"
    reference.set_task(saved)

    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name, feature_version=(3, 11))
    return {
        "result": "PASS", "scope": "engine kinematic consistency, not real robot accuracy or identified dynamics",
        "runtime": {"python": sys.version.split()[0], "mujoco": mujoco.__version__, "platform": platform.system(),
                    "python_3_11_syntax": "PASS", "gui_keyboard_live_test": "not run: no desktop display"},
        "sampled_poses": samples, "independent_axes": demo_status, "aim_without_insertion": "PASS",
        "recorded_encoder_endpoint_combinations_checked": recorded_endpoint_cases,
        "manual_C_step_um": manual_c_step_um,
        "manual_C_model_encoder_delta_counts": manual_c_encoder_delta,
        "rigid_installation_transform": "PASS", "encoder_sign_origin_and_limits_logic": "PASS",
        "max_errors": maxima, "max_integer_encoder_roundtrip_um": worst_encoder_roundtrip_um,
        "reset_state": reset_metrics,
        "RCM_eye_entry_check": {"result": "PASS", "ellipsoid_surface_value": eye_surface_value,
                                "radial_surface_gap_um": eye_radial_gap_um,
                                "reset_axis_eye_center_distance_um": reset_axis_eye_center_um},
        "original_deep_target": deep_target_check,
        "target_terminal_state": reached, "default_limits_enabled": reference.encoder_bounds is not None,
        "coordinate_note": "45 deg illustrative installation; default is the original deep target; sampled windows are not position limits; physical needle reference unconfirmed",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Reference()), ensure_ascii=False, indent=2))
