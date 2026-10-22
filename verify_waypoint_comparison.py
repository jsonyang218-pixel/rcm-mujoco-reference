"""Check shared ordering, sampled planners, target translation and dataset alignment."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import mujoco
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np

from rcm_navigation import load_targets, navigation_settings, solve_target, task_si_units, write_json
from rcm_fundus import make_random_fundus_targets, verify_fundus_targets
from rcm_reference import Reference
from rcm_waypoint_dataset import summarize_target_success_by_visibility
from run_waypoint_comparison import summarize_stop_constraints
from rcm_sampling_planners import PRM, TrapezoidMotion, rrt, select_visit_order, shortcut_path
from rcm_visual_servo import VisualServoSensor
from rcm_waypoint_dataset import (NO_IMAGE_INPUT_FIELDS, POLICY_INPUT_FIELDS, TASK_REGIMES,
                                  EpisodeWriter, make_policy_task, write_manifest)


class ObstacleSpace:
    """Independent planner fixture: a unit cube with a forbidden central ball."""
    @staticmethod
    def normalize(q):
        return np.asarray(q, dtype=float)

    denormalize = normalize

    @staticmethod
    def valid(q):
        return bool(np.all(q >= 0) and np.all(q <= 1) and np.linalg.norm(q-.5) >= .2)

    def edge_valid(self, start, end):
        return all(self.valid(q) for q in np.linspace(start, end, 101))


def read_png(path):
    """Decode the lossless RGB/filter-0 PNG files written by save_png."""
    content = Path(path).read_bytes()
    assert content[:8] == b"\x89PNG\r\n\x1a\n"
    offset, payload, size = 8, bytearray(), None
    while offset < len(content):
        length = struct.unpack(">I", content[offset:offset+4])[0]
        kind = content[offset+4:offset+8]
        data = content[offset+8:offset+8+length]
        crc = struct.unpack(">I", content[offset+8+length:offset+12+length])[0]
        assert zlib.crc32(kind+data) & 0xffffffff == crc
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
            size = (height, width)
        elif kind == b"IDAT":
            payload.extend(data)
        offset += length+12
    height, width = size
    raw = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).reshape(height, 1+3*width)
    assert np.all(raw[:, 0] == 0)
    return raw[:, 1:].reshape(height, width, 3)


def verify_algorithms(config, targets):
    reference = Reference(copy.deepcopy(config))
    solutions = [solve_target(reference, t["point_mm"], t["frame"], t["id"]) for t in targets]
    choices = []
    if len(solutions) <= 8:
        # Exact ordering is deterministic and independent of the input order;
        # the candidate count scales with the number of targets.
        expected_candidates = math.factorial(len(solutions))
        for permutation in itertools.permutations(solutions):
            _, report = select_visit_order([0., 0., 0.], permutation, navigation_settings(reference))
            assert len(report["candidates"]) == expected_candidates
            assert report["selected_estimated_motion_time_s"] == min(
                c["estimated_motion_time_s"] for c in report["candidates"])
            choices.append(report["selected_order"])
        assert all(choice == choices[0] for choice in choices)
    else:
        # Larger sets use the documented deterministic nearest-target heuristic;
        # avoid factorial enumeration while still checking its output contract.
        _, report = select_visit_order([0., 0., 0.], solutions, navigation_settings(reference))
        assert report["objective"] == "deterministic nearest-next surrogate"
        assert sorted(report["selected_order"]) == sorted(s["id"] for s in solutions)
    original_pose, camera = reference.current_task().copy(), reference.camera_pose()
    target_geom_ids = [int(reference.model.geom("lesion_marker").id),
                       int(reference.model.geom("lesion_display_marker").id)]
    absent_marker_names = [f"lesion_reticle_{i}" for i in range(4)] + [
        f"lesion_display_reticle_{i}" for i in range(4)]
    assert all(mujoco.mj_name2id(reference.model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
               for name in absent_marker_names)
    reference.set_target_world(solutions[0]["requested_world_mm"])
    requested_robot_m = reference.world_to_robot(solutions[0]["requested_world_mm"])*.001
    assert all(np.allclose(reference.model.geom_pos[geom_id], requested_robot_m, atol=1e-14, rtol=0)
               for geom_id in target_geom_ids)
    assert np.allclose(reference.model.site_pos[reference.sites["lesion"]], requested_robot_m, atol=1e-14, rtol=0)
    assert np.array_equal(reference.current_task(), original_pose)
    assert all(np.allclose(a, b) for a, b in zip(camera, reference.camera_pose()))
    space = ObstacleSpace()
    start, goal = np.array([.1, .5, .5]), np.array([.9, .5, .5])
    assert not space.edge_valid(start, goal)
    tree_path, _ = rrt(space, start, goal, np.random.default_rng(7))
    graph = PRM(space, [start, goal], np.random.default_rng(7))
    graph_path = graph.query(0, 1)
    for path in (tree_path, graph_path):
        assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
        assert len(path) > 2
        assert all(space.edge_valid(a, b) for a, b in zip(path, path[1:]))
        shortened = shortcut_path(space, path)
        assert 2 < len(shortened) <= len(path)
        assert np.allclose(shortened[0], start) and np.allclose(shortened[-1], goal)
        assert all(space.edge_valid(a, b) for a, b in zip(shortened, shortened[1:]))
    settings = navigation_settings(reference)
    vmax = np.asarray(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
    amax = np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
    for goal in ([0., 0., 0.], [.001, -.002, .001], [30., -18., 15.]):
        motion = TrapezoidMotion(np.zeros(3), goal, settings)
        assert np.allclose(motion.sample(0)[0], 0)
        assert np.allclose(motion.sample(motion.duration_s)[0], goal)
        assert np.allclose(motion.sample(0)[1], 0)
        assert np.allclose(motion.sample(motion.duration_s)[1], 0)
        for t in np.linspace(0, motion.duration_s, 1001):
            _, velocity, acceleration = motion.sample(t)
            assert np.all(np.abs(velocity) <= vmax+1e-10)
            assert np.all(np.abs(acceleration) <= amax+1e-10)
        if motion.duration_s > 0:
            epsilon = min(1e-6, motion.accel_time_s/100)
            for t in (motion.accel_time_s, motion.accel_time_s+motion.cruise_time_s):
                left, right = motion.sample(t-epsilon), motion.sample(t+epsilon)
                assert np.all(np.abs(right[1]-left[1]) <= 2*epsilon*amax+1e-9)
            t = motion.accel_time_s/2
            position_rate = (motion.sample(t+epsilon)[0]-motion.sample(t-epsilon)[0])/(2*epsilon)
            assert np.allclose(position_rate, motion.sample(t)[1], atol=1e-7)
    return {"order_independent_of_input_permutation": "PASS", "single_target_marker_translation": "PASS",
            "peripheral_target_markers_absent": "PASS",
            "RRT_synthetic_obstacle_detour": "PASS", "PRM_synthetic_obstacle_detour": "PASS",
            "shortcut_preserves_obstacle_detour": "PASS", "trapezoid_motion_limits_and_continuity": "PASS"}


def verify_random_targets_and_reporting(config):
    reference = Reference(copy.deepcopy(config))
    targets_a = make_random_fundus_targets(reference, np.random.default_rng(20261020), 3, 1.0)
    targets_repeat = make_random_fundus_targets(reference, np.random.default_rng(20261020), 3, 1.0)
    targets_b = make_random_fundus_targets(reference, np.random.default_rng(20261021), 3, 1.0)
    points_a = np.asarray([target["point_mm"] for target in targets_a["targets"]])
    points_repeat = np.asarray([target["point_mm"] for target in targets_repeat["targets"]])
    points_b = np.asarray([target["point_mm"] for target in targets_b["targets"]])
    assert np.array_equal(points_a, points_repeat)
    assert not np.array_equal(points_a, points_b)
    solutions = [solve_target(reference, target["point_mm"], targets_a["frame"], target["id"])
                 for target in targets_a["targets"]]
    surface = verify_fundus_targets(reference, solutions, targets_a["target_region"])
    assert surface["result"] == "PASS"

    def frame(waypoint_id, state):
        return {"waypoint_id": waypoint_id, "phase": "visual_approach", "is_terminal": False,
                "visibility": {"target_occlusion_state": state}}

    records = [frame("P1", "clear"), frame("P1", "full_needle"),
               frame("P2", "clear"), frame("P3", "not_visible"), frame("P4", "unknown")]
    waypoints = [
        {"waypoint_id": "P1", "result": "PASS", "terminal_stop_validated": True},
        {"waypoint_id": "P2", "result": "FAIL", "terminal_stop_validated": False},
        {"waypoint_id": "P3", "result": "FAIL", "terminal_stop_validated": True},
        {"waypoint_id": "P4", "result": "FAIL", "terminal_stop_validated": False},
        {"waypoint_id": "P5", "result": "NOT_RUN", "terminal_stop_validated": False},
    ]
    stratified = summarize_target_success_by_visibility(records, waypoints)
    conditions = stratified["condition_exposure"]
    assert conditions["needle_occluded"] == {
        "target_visits": 1, "successful_target_visits": 1, "success_rate": 1.0}
    assert conditions["visible"] == {
        "target_visits": 2, "successful_target_visits": 1, "success_rate": 0.5}
    assert conditions["other_visibility_issue"]["target_visits"] == 1
    assert conditions["unknown"]["target_visits"] == 1
    assert stratified["attempted_target_visits"] == 4
    assert stratified["not_attempted_target_visits"] == 1
    assert sum(row["target_visits"] for row in conditions.values()) > stratified["attempted_target_visits"]
    assert all(row["target_visits"] <= stratified["attempted_target_visits"]
               for row in conditions.values())

    method_info = {"terminal_stop_validated": True, "terminal_velocity": [0., 0., 0.]}
    stopped_waypoints = [
        {"waypoint_id": "P1", "result": "PASS", "terminal_stop_validated": True},
        {"waypoint_id": "P2", "result": "PASS", "terminal_stop_validated": True},
    ]
    stop_pass = summarize_stop_constraints(stopped_waypoints, method_info, 2)
    assert stop_pass["pass"] and stop_pass["terminal_velocity_zero"]
    stopped_waypoints[1]["terminal_stop_validated"] = False
    assert not summarize_stop_constraints(stopped_waypoints, method_info, 2)["pass"]
    method_info["terminal_velocity"] = [0., 0., .1]
    assert not summarize_stop_constraints(stopped_waypoints, method_info, 2)["terminal_velocity_zero"]
    return {"random_fundus_targets_reproducible_and_seed_varied": "PASS",
            "random_fundus_targets_verified_on_posterior_surface": "PASS",
            "overlapping_visibility_exposure_success_strata": "PASS",
            "stop_constraint_rejects_missing_stop_and_nonzero_terminal_velocity": "PASS"}


def verify_policy_bundles(root, episode, task, task_regime, solutions, fundus_surface):
    """Enforce the exact model interface and row alignment with action labels."""
    assert task_regime in TASK_REGIMES
    expected_task = make_policy_task(task_regime, solutions, fundus_surface)
    assert task == expected_task
    if task_regime == "visual_current_target":
        # The task interface is intentionally coordinate-, ID-, order-, and seed-free.
        assert not any(character.isdigit() for character in task)
        assert all(f"{solution['id']}=" not in task for solution in solutions)
    else:
        assert all(f"{solution['id']}=" in task for solution in solutions)
        assert all(str(solution["requested_robot_mm"]) in task for solution in solutions)

    full_path = root/episode["policy_inputs"]
    no_image_path = root/episode["policy_inputs_no_image"]
    labels_path = root/episode["action_labels"]
    audit_path = root/episode["data"]
    with (np.load(full_path, allow_pickle=False) as full,
          np.load(no_image_path, allow_pickle=False) as no_image,
          np.load(labels_path, allow_pickle=False) as labels,
          np.load(audit_path, allow_pickle=False) as audit):
        count = int(episode["frames"])
        assert set(full.files) == set(POLICY_INPUT_FIELDS)
        assert set(no_image.files) == set(NO_IMAGE_INPUT_FIELDS)
        assert set(labels.files) == {"action"}
        assert full["observation.images.microscope_rgb"].shape == (count,)
        assert full["observation.images.microscope_rgb"].dtype.kind in "US"
        assert full["observation.state"].dtype == no_image["observation.state"].dtype == np.float32
        assert full["observation.state"].shape == no_image["observation.state"].shape == (count, 3)
        assert full["task"].shape == no_image["task"].shape == (count,)
        assert np.array_equal(full["observation.images.microscope_rgb"], audit["observation.images.microscope_rgb"])
        assert np.array_equal(full["observation.state"], no_image["observation.state"])
        assert np.array_equal(full["observation.state"], audit["observation.state"])
        assert np.array_equal(full["task"], no_image["task"])
        assert np.array_equal(full["task"], np.asarray([expected_task]*count, dtype=str))
        assert labels["action"].dtype == audit["action"].dtype == np.float32
        assert labels["action"].shape == audit["action"].shape == (count, 3)
        assert np.array_equal(labels["action"], audit["action"])
        assert np.array_equal(labels["action"][:-1], audit["observation.state"][1:])
    return "PASS"


def verify_policy_export_contract():
    """Exercise both task regimes and reject a coordinate leak into visual-only text."""
    solutions = [
        {"id": "P1", "requested_robot_mm": [1.2345, -2.3456, 23.4567]},
        {"id": "P2", "requested_robot_mm": [-4.5678, 5.6789, 22.3456]},
    ]
    initial = np.zeros(3, dtype=np.float32)
    moved = np.array([.1, -.2, .3], dtype=np.float32)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="rcm_policy_contract_") as temporary:
        temporary_root = Path(temporary)
        for regime in TASK_REGIMES:
            root = temporary_root/regime/"dataset"
            task = make_policy_task(regime, solutions, fundus_surface=True)
            writer = EpisodeWriter(root, 0, task, "visual_servo", None, ["P1", "P2"])
            writer.add(0., initial, moved, "P1", rgb, phase="visual_servo")
            writer.add(.05, moved, moved, "P1", rgb, terminal=True, phase="hold")
            episode = writer.finish("PASS")
            write_manifest(root, [episode], [2, 3, 3], .05, regime)
            assert verify_policy_bundles(root, episode, task, regime, solutions, True) == "PASS"

            if regime == "visual_current_target":
                policy_path = root/episode["policy_inputs"]
                with np.load(policy_path, allow_pickle=False) as data:
                    corrupted = {name: data[name] for name in data.files}
                corrupted["task"] = np.asarray(
                    [str(item)+" P1=[1.2345, -2.3456, 23.4567]" if index == 0 else str(item)
                     for index, item in enumerate(corrupted["task"])], dtype=str)
                np.savez_compressed(policy_path, **corrupted)
                try:
                    verify_policy_bundles(root, episode, task, regime, solutions, True)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Visual-only contract accepted a task-text coordinate leak")
    return {"both_task_regimes_exported_and_aligned": "PASS",
            "visual_current_target_rejects_coordinate_leak": "PASS",
            "exact_policy_fields_and_action_labels": "PASS"}


def verify(directory):
    directory = Path(directory)
    config = json.loads((directory/"config_snapshot.json").read_text(encoding="utf-8"))
    targets = load_targets(directory/"targets.json")
    checks = verify_algorithms(config, targets)
    checks.update(verify_random_targets_and_reporting(config))
    root = directory/"dataset"
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert "offline metadata only" in manifest["visibility_semantics"]
    reports = json.loads((directory/"run_reports.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory/"experiment_protocol.json").read_text(encoding="utf-8"))
    if protocol.get("guidance_mode") == "strict_current_frame_visual_servo":
        assert protocol["planner_feedback"] == "vision"
        assert protocol["target_pose_usage"].startswith("target world coordinates place the virtual marker")
        assert protocol["strict_visual_control_contract"]["target_detection"].startswith("current-frame RGB")
        assert "cached target after full occlusion" in protocol["strict_visual_control_contract"]["disabled"]
        assert protocol["expert_perception_privileges"]["visual_servo_known_target_pose_feature_calibration"] is False
        assert protocol["expert_perception_privileges"]["visual_servo_fundus_transfer_uses_scene_goal_pose"] is False
        checks["strict_current_frame_visual_only_protocol"] = "PASS"
    task_regime = manifest.get("task_regime")
    if task_regime is not None:
        assert task_regime in TASK_REGIMES and protocol["task_regime"] == task_regime
        interface = manifest["policy_interface"]
        assert interface["contract_version"] == 1
        assert interface["input_fields"] == list(POLICY_INPUT_FIELDS)
        assert interface["no_image_ablation_fields"] == list(NO_IMAGE_INPUT_FIELDS)
        assert interface["goal_coordinates_in_task"] == (task_regime == "registered_coordinate_prior")
    else:
        # Do not relabel old coordinate-bearing episodes as visual-only.
        checks["policy_input_contract"] = "NOT_AVAILABLE_LEGACY_UNSPECIFIED"
    nominal_period = float(protocol["dataset_sampling_period_s"])
    assert np.isclose(manifest["nominal_sample_period_s"], nominal_period, atol=1e-12, rtol=0)
    assert np.isclose(manifest["nominal_sampling_frequency_hz"], 1./nominal_period, atol=1e-10, rtol=0)
    strict_visual = protocol.get("guidance_mode") == "strict_current_frame_visual_servo"
    vision_planning = protocol.get("planner_feedback") == "vision" and not strict_visual
    if vision_planning:
        visual_map = json.loads((directory/"visual_target_map.json").read_text(encoding="utf-8"))
        order = json.loads((directory/"order_selection.json").read_text(encoding="utf-8"))
        measured_ids = {s["id"] for s in visual_map["measured_solutions"]}
        failed_ids = {s["id"] for s in visual_map["observations"] if s["result"] != "PASS"}
        assert measured_ids.isdisjoint(failed_ids)
        if failed_ids:
            assert order["selected_order"] == []
            assert all(r["execution_mode"] == "visual_survey_rejected" and not r["planner_executed"] for r in reports)
        else:
            _, expected = select_visit_order([0., 0., 0.], visual_map["measured_solutions"], navigation_settings(Reference(copy.deepcopy(config))))
            assert order["selected_order"] == expected["selected_order"]
        checks["visual_map_without_ground_truth_fallback"] = "PASS"
    elif strict_visual:
        visual_map = json.loads((directory/"visual_target_map.json").read_text(encoding="utf-8"))
        order = json.loads((directory/"order_selection.json").read_text(encoding="utf-8"))
        assert visual_map["measured_solutions"] == [] and visual_map["observations"] == []
        assert order["selected_order"] == [target["id"] for target in targets]
        assert all(report.get("execution_mode") == "strict_current_frame_visual_servo"
                   for report in reports)
        checks["strict_visual_no_survey_or_route_planning"] = "PASS"
    precision = config["navigation"].get("waypoint_trajectory") == "trapezoid"
    solutions = {s["id"]: s for s in json.loads((directory/"target_solutions.json").read_text(encoding="utf-8"))}
    region = json.loads((directory/"targets.json").read_text(encoding="utf-8")).get("target_region")
    if region:
        from rcm_fundus import verify_fundus_targets
        reference = Reference(copy.deepcopy(config))
        geom = reference.model.geom("eye_reference").id
        for solution in solutions.values():
            local = reference.data.geom_xmat[geom].reshape(3, 3).T @ (
                np.asarray(solution["requested_world_mm"])*.001-reference.data.geom_xpos[geom])
            unit = local/reference.model.geom_size[geom]
            assert abs(float(unit@unit)-1.) < 1e-9
            assert unit[2] >= region["minimum_normalized_posterior_z"]
        for bad_world in (reference.data.geom_xpos[geom]*1000.,
                          (reference.data.geom_xpos[geom]-reference.data.geom_xmat[geom].reshape(3, 3)[:, 2]*reference.model.geom_size[geom, 2])*1000.):
            bad = [{"id": "negative_test", "requested_world_mm": bad_world.tolist()}]
            assert verify_fundus_targets(reference, bad, region)["result"] == "FAIL"
        checks["fundus_surface_and_posterior_location"] = "PASS"
    shape = manifest["fields"]["observation.images.microscope_rgb"]["shape"]
    policy = config["navigation"].get("fundus_transition")
    if policy:
        from rcm_fundus import no_surface_slide_edge, standoff_task
        reference = Reference(copy.deepcopy(config))
        goals = [np.asarray(s["requested_task_A_deg_B_deg_C_mm"]) for s in solutions.values()]
        assert not no_surface_slide_edge(reference, goals[0], goals[1], policy["minimum_transfer_clearance_mm"])
        near = standoff_task(goals[0], .1)
        lateral = near.copy()
        lateral[0] += .1
        assert not no_surface_slide_edge(reference, near, lateral, policy["minimum_transfer_clearance_mm"])
        far = standoff_task(goals[0], policy["retract_mm"])
        assert no_surface_slide_edge(reference, goals[0], far, policy["minimum_transfer_clearance_mm"])
        assert no_surface_slide_edge(reference, far, goals[0], policy["minimum_transfer_clearance_mm"])
        checks["reject_surface_slide_and_allow_axial_motion"] = "PASS"
    total, render_checks, max_different_pixels = 0, 0, 0
    max_observation_interval = 0.
    for episode, report in zip(manifest["episodes"], reports, strict=True):
        if "stop_constraint_summary" in report:
            stop_summary = report["stop_constraint_summary"]
            assert episode.get("stop_constraint_summary") == stop_summary
            assert stop_summary["required_at_every_target"]
            assert stop_summary["terminal_velocity_zero"] == bool(
                np.max(np.abs(stop_summary["terminal_velocity_A_deg_B_deg_C_mm_per_s"])) <= 1e-9)
            assert episode.get("target_success_by_visibility") == report.get("target_success_by_visibility")
            success = report["target_success_by_visibility"]
            groups = success["condition_exposure"]
            grouped_visits = sum(value["target_visits"] for value in groups.values())
            grouped_successes = sum(value["successful_target_visits"] for value in groups.values())
            assert grouped_visits >= success["attempted_target_visits"]
            assert all(value["target_visits"] <= success["attempted_target_visits"]
                       and value["successful_target_visits"] <= value["target_visits"]
                       for value in groups.values())
            assert grouped_successes <= grouped_visits
            assert report["result"] == "PASS" or not stop_summary["pass"]
            if stop_summary["pass"]:
                assert all(waypoint["success_with_stop"] for waypoint in report["waypoints"])
        with np.load(root/episode["data"], allow_pickle=False) as data:
            count = episode["frames"]
            assert count > 0
            state, action, timestamps = data["observation.state"], data["action"], data["timestamp"]
            assert state.dtype == action.dtype == np.float32
            assert state.shape == action.shape == (count, 3)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            assert timestamps.dtype == np.float64 and timestamps[0] == 0 and np.all(np.diff(timestamps) > 0)
            assert np.all(np.diff(timestamps) <= nominal_period+1e-8), "Dataset has an observation gap above the configured rate"
            if len(timestamps) > 1:
                max_observation_interval = max(max_observation_interval, float(np.max(np.diff(timestamps))))
            assert data["episode_index"].dtype == data["frame_index"].dtype == np.int64
            assert np.all(data["episode_index"] == episode["episode_index"])
            assert np.array_equal(data["frame_index"], np.arange(count))
            assert np.array_equal(action[:-1], state[1:]), "Action is not aligned with next actual state"
            assert np.array_equal(action[-1], state[-1])
            if "phase" in data:
                assert len(data["phase"]) == count and data["phase"][-1] == "hold"
            visibility_fields = (
                "visibility.known", "visibility.target_pixels_visible",
                "visibility.target_pixels_unoccluded", "visibility.target_pixels_occluded_by_needle",
                "visibility.target_occlusion_fraction_by_needle", "visibility.target_occlusion_state",
                "visibility.tip_pixels_visible", "visibility.needle_pixels_visible",
                "visibility.tip_fully_occluded", "visibility.counterfactual_no_needle_rendered")
            assert all(field in data.files for field in visibility_fields)
            known = data["visibility.known"]
            assert known.dtype == np.bool_ and known.shape == (count,)
            for field in ("visibility.target_pixels_visible", "visibility.target_pixels_unoccluded",
                          "visibility.target_pixels_occluded_by_needle", "visibility.tip_pixels_visible",
                          "visibility.needle_pixels_visible"):
                assert data[field].dtype == np.int32 and data[field].shape == (count,)
            fraction = data["visibility.target_occlusion_fraction_by_needle"]
            states = data["visibility.target_occlusion_state"]
            assert fraction.dtype == np.float32 and states.shape == (count,)
            assert data["visibility.tip_fully_occluded"].dtype == np.bool_
            assert data["visibility.counterfactual_no_needle_rendered"].dtype == np.bool_
            assert np.all(data["visibility.counterfactual_no_needle_rendered"][known])
            assert np.all(data["visibility.target_pixels_visible"][known] +
                          data["visibility.target_pixels_occluded_by_needle"][known] ==
                          data["visibility.target_pixels_unoccluded"][known])
            assert np.all(data["visibility.target_pixels_visible"][known] <=
                          data["visibility.target_pixels_unoccluded"][known])
            measurable_fraction = known & (data["visibility.target_pixels_unoccluded"] > 0)
            undefined_fraction = known & ~measurable_fraction
            assert np.all(np.isfinite(fraction[measurable_fraction]))
            assert np.all((fraction[measurable_fraction] >= 0.) & (fraction[measurable_fraction] <= 1.))
            assert np.all(fraction[undefined_fraction] == -1.)
            assert np.all(fraction[~known] == -1.)
            assert np.all(data["visibility.target_pixels_visible"][~known] == -1)
            assert np.all(data["visibility.target_pixels_unoccluded"][~known] == -1)
            assert np.all(data["visibility.target_pixels_occluded_by_needle"][~known] == -1)
            assert np.all(data["visibility.target_occlusion_state"][~known] == "unknown")
            for state_name, expected in (("clear", 0), ("partial_needle", 1),
                                         ("full_needle", 2), ("not_visible", 3)):
                selected = known & (states == state_name)
                if state_name == "clear":
                    assert np.all(data["visibility.target_pixels_occluded_by_needle"][selected] == 0)
                    assert np.all(fraction[selected] == 0.)
                elif state_name in ("partial_needle", "full_needle"):
                    assert np.all(data["visibility.target_pixels_occluded_by_needle"][selected] > 0)
                    assert np.all(fraction[selected] > 0.)
                elif state_name == "not_visible":
                    assert np.all(data["visibility.target_pixels_unoccluded"][selected] == 0)
            assert np.all(np.isin(states[known], ["clear", "partial_needle", "full_needle", "not_visible"]))
            expected_start = protocol.get("start_task_A_deg_B_deg_C_mm", [0., 0., 0.])
            assert np.allclose(state[0], expected_start)
            if report.get("execution_mode") == "visual_survey_rejected":
                assert np.all(state == 0.) and np.array_equal(state, action)
                assert report["result"] == "FAIL" and episode["visit_order"] == []
            if report.get("execution_mode") == "vision_guided_sampling_planner":
                assert report["feedback_controller_used"] and not report["ground_truth_used_for_control"]
                assert not report["known_target_pose_used_for_feature_calibration"]
                feedback = json.loads((directory/f"{episode['method']}_seed_{episode['seed']}"/"visual_feedback.json").read_text(encoding="utf-8"))
                assert len(feedback) == count-1 == report["visual_feedback_samples"]
                for sample, command in zip(feedback, action[:-1]):
                    assert sample["target_pixels"] > 0
                    assert np.allclose(sample["command_A_deg_B_deg_C_mm"], command, atol=2e-6, rtol=0)
            assert data["is_terminal"][-1] and not data["is_terminal"][:-1].any()
            assert all(isinstance(task, str) and len(task) > 0 for task in data["task"].tolist())
            with (root/episode["jsonl"]).open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
                assert len(records) == count
            if task_regime is not None:
                assert manifest["policy_interface"]["image_loading"].startswith("Decode paths")
                task_texts = set(data["task"].tolist())
                assert len(task_texts) == 1
                assert verify_policy_bundles(root, episode, next(iter(task_texts)), task_regime,
                                             list(solutions.values()), bool(region)) == "PASS"
            for index, (name, record) in enumerate(zip(data["observation.images.microscope_rgb"], records)):
                assert name == record["observation.images.microscope_rgb"]
                assert np.array_equal(np.asarray(record["observation.state"], dtype=np.float32), state[index])
                assert np.array_equal(np.asarray(record["action"], dtype=np.float32), action[index])
                assert record["frame_index"] == index and record["timestamp"] == timestamps[index]
                metadata = record["visibility"]
                assert metadata["known"] == bool(known[index])
                assert metadata["target_occlusion_state"] == str(states[index])
                for key, field in (("target_pixels_visible", "visibility.target_pixels_visible"),
                                   ("target_pixels_unoccluded", "visibility.target_pixels_unoccluded"),
                                   ("target_pixels_occluded_by_needle", "visibility.target_pixels_occluded_by_needle"),
                                   ("tip_pixels_visible", "visibility.tip_pixels_visible"),
                                   ("needle_pixels_visible", "visibility.needle_pixels_visible")):
                    assert metadata[key] == int(data[field][index])
                stored_fraction = metadata["target_occlusion_fraction_by_needle"]
                assert (stored_fraction is None and fraction[index] == -1.) or np.isclose(
                    stored_fraction, fraction[index], atol=1e-7, rtol=0)
                assert metadata["tip_fully_occluded"] == bool(data["visibility.tip_fully_occluded"][index])
                assert metadata["counterfactual_no_needle_rendered"] == bool(
                    data["visibility.counterfactual_no_needle_rendered"][index])
                with (root/str(name)).open("rb") as handle:
                    header = handle.read(29)
                assert header[:8] == b"\x89PNG\r\n\x1a\n"
                width, height = struct.unpack(">II", header[16:24])
                assert [height, width, 3] == shape and header[24:26] == bytes([8, 2])
            if episode["result"] == "PASS":
                observed_order = [key for key, _ in itertools.groupby(data["waypoint_id"].tolist())]
                assert observed_order == episode["visit_order"]
                assert len(report["waypoints"]) == len(episode["visit_order"])
                assert all(w["result"] == "PASS" for w in report["waypoints"])
                assert report["trajectory_rate_limits_pass"] and report["tip_outside_eye_samples"] == 0
                if policy:
                    assert report["fundus_transition_constraints_pass"]
                    if episode["method"] in ("RRT", "PRM"):
                        for target_index, waypoint in enumerate(episode["visit_order"]):
                            phases = [p for p, _ in itertools.groupby(data["phase"][data["waypoint_id"] == waypoint].tolist()) if p != "hold"]
                            assert phases == (["transfer", "approach"] if target_index == 0 else ["retract", "transfer", "approach"]), phases
                if precision:
                    assert report["terminal_stop_validated"]
                    assert all(w["tip_error_um"] <= 20. for w in report["waypoints"])
                    if episode["method"] == "visual_servo":
                        assert all(d["settled_frames"] >= 3 and d["terminal_stop_validated"]
                                   and not d["ground_truth_used_for_stop"] for d in report["control_details"])
                        for waypoint in episode["visit_order"]:
                            indices = np.flatnonzero(data["waypoint_id"] == waypoint)
                            # The last observation at intermediate targets is shared with the next leg.
                            assert np.array_equal(state[indices[-1]], state[indices[-2]])
                            assert np.array_equal(state[indices[-1]], action[indices[-1]])
            # Independently check carried command velocities across all visual target boundaries.
            if episode["method"] == "visual_servo":
                dt = np.diff(timestamps)
                velocity = (action[:-1].astype(float)-state[:-1])/dt[:, None]
                visual_indices = (np.flatnonzero(data["phase"][:-1] == "visual_servo")
                                  if "phase" in data else np.arange(len(velocity)))
                if len(visual_indices):
                    # Retraction is analytic rest-to-rest motion; its last interval
                    # average is not the velocity at the start of visual control.
                    accelerations = []
                    for i in visual_indices:
                        previous = velocity[i-1] if i > 0 and i-1 in visual_indices else np.zeros(3)
                        accelerations.append((velocity[i]-previous)/dt[i])
                    assert np.all(np.max(np.abs(velocity[visual_indices]), axis=0) <= np.asarray(config["navigation"]["max_velocity_A_deg_B_deg_C_mm_per_s"])+.001)
                    assert np.all(np.max(np.abs(accelerations), axis=0) <= np.asarray(config["navigation"]["max_acceleration_A_deg_B_deg_C_mm_per_s2"])+.005)
            reference = Reference(copy.deepcopy(config))
            if policy:
                # Independent check against compiled MuJoCo tip positions, not planner FK.
                geom = reference.model.geom("eye_reference").id
                center = reference.data.geom_xpos[geom].copy()
                rotation = reference.data.geom_xmat[geom].reshape(3, 3).copy()
                radii = reference.model.geom_size[geom].copy()
                for q0, q1, phase in zip(state[:-1], action[:-1], data["phase"][:-1]):
                    turning = np.max(np.abs(q1[:2]-q0[:2])) > 1e-6
                    if phase in ("retract", "approach"):
                        assert not turning
                        assert q1[2] <= q0[2]+1e-6 if phase == "retract" else q1[2] >= q0[2]-1e-6
                    for interpolation_fraction in np.linspace(0., 1., 5):
                        reference.set_task(task_si_units(q0.astype(float)+interpolation_fraction*(q1.astype(float)-q0)))
                        tip = np.asarray(reference.metrics()["tip_world_mm"])*.001
                        unit = rotation.T@(tip-center)/radii
                        clearance = float(np.min(radii)*(1.-np.linalg.norm(unit))*1000.)
                        assert clearance >= -2e-6
                        if turning or phase == "transfer":
                            assert clearance >= policy["minimum_transfer_clearance_mm"]-2e-6
            sensor = VisualServoSensor(reference)
            try:
                for index in sorted({0, count//2, count-1}):
                    stored = read_png(root/str(data["observation.images.microscope_rgb"][index]))
                    assert list(stored.shape) == shape and stored.dtype == np.uint8
                    solution = solutions[str(data["waypoint_id"][index])]
                    reference.set_target_world(solution["requested_world_mm"])
                    reference.set_task(task_si_units(state[index]))
                    observation = sensor.observe()
                    rendered = observation["rgb"]
                    changed = int(np.any(stored != rendered, axis=2).sum())
                    # Dataset float32 axis rounding can change a few boundary pixels.
                    assert changed <= 10, (episode["episode_index"], index, changed)
                    max_different_pixels = max(max_different_pixels, changed)
                    render_checks += 1
                    expected_visibility = sensor.visibility_metrics(observation["segmentation"])
                    assert known[index]
                    assert expected_visibility["target_pixels_visible"] == data["visibility.target_pixels_visible"][index]
                    assert expected_visibility["target_pixels_unoccluded"] == data["visibility.target_pixels_unoccluded"][index]
                    assert expected_visibility["target_pixels_occluded_by_needle"] == data["visibility.target_pixels_occluded_by_needle"][index]
                    assert expected_visibility["target_occlusion_state"] == states[index]
                    expected_fraction = expected_visibility["target_occlusion_fraction_by_needle"]
                    if expected_fraction is None:
                        assert fraction[index] == -1.
                    else:
                        assert np.isclose(expected_fraction, fraction[index], atol=1e-7, rtol=0)
                    checks["frame_specific_geometric_occlusion_rerender"] = "PASS"
            finally:
                sensor.close()
            total += count
    assert total == manifest["total_frames"]
    result = {"result": "PASS", **checks, "episodes": len(manifest["episodes"]), "frames": total,
              "all_image_headers_checked": total, "RGB_state_rerender_checks": render_checks,
              "max_rerender_different_pixels": max_different_pixels,
              "action_next_state_alignment": "PASS", "visual_boundary_rate_limits": "PASS",
              "schema_dtypes_and_shapes": "PASS",
              "visibility_metadata_integrity": "PASS",
              "nominal_sampling_frequency_hz": manifest["nominal_sampling_frequency_hz"],
              "maximum_observation_interval_s": max_observation_interval}
    if precision:
        result["successful_episodes_precision_20um_and_waypoint_stops"] = "PASS"
    if policy:
        result["independent_phase_and_wall_clearance_checks"] = "PASS"
    if task_regime is not None:
        checks["policy_input_fields_task_regime_and_action_alignment"] = "PASS"
        checks.update(verify_policy_export_contract())
        result.update(checks)
    result["navigation_successful_episodes"] = sum(r["result"] == "PASS" for r in reports)
    result["navigation_failed_episodes"] = sum(r["result"] != "PASS" for r in reports)
    result["scope"] = "Dataset integrity and successful-episode acceptance; does not imply every navigation succeeded"
    write_json(directory/"dataset_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--policy-contract-only", action="store_true",
                       help="Run isolated policy input/export contract fixtures")
    args = parser.parse_args()
    result = verify_policy_export_contract() if args.policy_contract_only else verify(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
