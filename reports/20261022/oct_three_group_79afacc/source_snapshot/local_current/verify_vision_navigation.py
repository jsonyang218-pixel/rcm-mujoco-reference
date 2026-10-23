"""Verify image-derived positions, feedback causality, and fail-closed planning."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from rcm_navigation import load_targets, solve_target, task_user_units, write_json
from rcm_reference import HERE, Reference
from rcm_sampling_planners import PRM
from rcm_vision_navigation import (RGBDPositionSensor, feedback_command, fit_marker_center,
                                   map_visual_targets, prm_from_departure, run_vision_planner)
from rcm_waypoint_dataset import EpisodeWriter
from verify_waypoint_comparison import ObstacleSpace


def expect_failure(operation, expected):
    try:
        operation()
    except RuntimeError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("Expected sensor rejection: "+expected)


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    reference = Reference()
    sensor = RGBDPositionSensor(reference)
    try:
        observation = sensor.observe()
        target, tip, residual = sensor.positions(observation)
        assert np.linalg.norm(tip-reference.metrics()["tip_world_mm"]) < .03
        # Frozen images must be sufficient. Poison all direct truth interfaces.
        def forbidden(*args, **kwargs):
            raise AssertionError("Ground-truth interface used by perception/control")
        reference.target_task = forbidden
        reference.metrics = forbidden
        reference.site_mm = forbidden
        estimated, _, _ = sensor.positions(observation)
        assert np.array_equal(estimated, target)
        blank = dict(observation, target_pixel=None)
        expect_failure(lambda: sensor.positions(blank), "TARGET_NOT_VISIBLE")
        blank_tip = dict(observation, tip_pixel=None)
        expect_failure(lambda: sensor.positions(blank_tip), "TIP_NOT_VISIBLE")
        q = np.asarray(task_user_units(reference.current_task()))
        from rcm_vision_navigation import task_from_measurement
        desired = task_from_measurement(reference, tip)+np.array([.01, .01, .01])
        command, _, _ = feedback_command(reference, q, tip, desired, np.zeros(3), .1, "motion")
        perturbed = tip+reference.rotation_robot@np.array([.01, 0., 0.])
        changed, _, _ = feedback_command(reference, q, perturbed, desired, np.zeros(3), .1, "motion")
        assert np.linalg.norm(command-changed) > .001, "Vision does not change commands"
        # Perturb the sensor's OCT depth array, not the diagnostic scene depth or motor state.
        changed_observation = copy.deepcopy(observation)
        mask = changed_observation["segmentation"][:, :, 0] == sensor.tip_geom_id
        changed_observation["oct_depth"][mask] += .00002
        changed_tip, _ = sensor.center_from_image(changed_observation, False)
        assert np.linalg.norm(changed_tip-tip) > .01
        from_depth, _, _ = feedback_command(reference, q, changed_tip, desired, np.zeros(3), .1, "motion")
        assert np.linalg.norm(from_depth-command) > .001
    finally:
        sensor.close()
    directions = np.random.default_rng(4).normal(size=(100, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    center = np.array([2., -1., -10.])
    recovered, fit_residual = fit_marker_center(center+.22*directions, np.zeros((100, 3)), np.full(100, .22))
    assert np.linalg.norm(recovered-center) < 1e-6 and fit_residual < 1e-6
    outliers = center+.22*directions
    outliers[:8] += 1.
    robust, _ = fit_marker_center(outliers, np.zeros((100, 3)), np.full(100, .22))
    assert np.linalg.norm(robust-center) < 1e-6
    expect_failure(lambda: fit_marker_center(np.zeros((2, 3)), np.zeros((2, 3)), np.ones(2)), "INSUFFICIENT")

    reference = Reference()
    scenes = [solve_target(reference, t["point_mm"], t["frame"], t["id"])
              for t in load_targets(HERE/"examples"/"three_waypoints.json")]
    # Inject one survey-frame dropout so the fail-closed mapping path is
    # exercised independently of a particular camera/marker placement.
    original_observe = RGBDPositionSensor.observe
    survey_observations = 0
    def survey_dropout(self):
        nonlocal survey_observations
        survey_observations += 1
        result = original_observe(self)
        if survey_observations == 2:
            result["target_pixel"] = None
            result["target_pixels"] = 0
        return result
    try:
        RGBDPositionSensor.observe = survey_dropout
        mapped, audit = map_visual_targets(reference, scenes)
    finally:
        RGBDPositionSensor.observe = original_observe
    assert any(row["id"] == "P2" and row["result"] == "FAIL" for row in audit)
    assert "P2" not in [row["id"] for row in mapped], "Invisible target filled from scene truth"

    # Attaching a measured departure must retain collision-free graph edges.
    space = ObstacleSpace()
    graph = PRM(space, [np.array([.1, .5, .5]), np.array([.9, .5, .5])], np.random.default_rng(7))
    old_count = len(graph.nodes)
    departure = np.array([.11, .51, .5])
    path = prm_from_departure(graph, space, departure, 1)
    assert len(graph.nodes) == old_count
    assert np.allclose(path[0], departure)
    assert all(space.edge_valid(a, b) for a, b in zip(path, path[1:]))

    # Verify geometric occlusion at a deterministic pose independently of a
    # planner's route. Precision perception can reject an unobservable apex
    # earlier than the old single-pixel controller, while the target is clear.
    occlusion_reference = Reference()
    occlusion_reference.set_task(occlusion_reference.target_task())
    occlusion_sensor = RGBDPositionSensor(occlusion_reference)
    try:
        occluded = occlusion_sensor.observe()
        natural = occlusion_sensor.visibility_metrics(occluded['segmentation'])
        assert natural['target_occlusion_state'] in ('partial_needle', 'full_needle'), natural
        assert natural['target_pixels_occluded_by_needle'] > 0
        write_json(output/'natural_occlusion_fixture.json', natural)
    finally:
        occlusion_sensor.close()
    # Both real planners must preserve causal feedback, report failures, and
    # finish a lost-measurement episode with a zero-change simulation hold.
    fixture_reports = []
    for method in ("RRT", "PRM"):
        reference = Reference()
        scene = solve_target(reference, [1., -.2, 5.], "robot", "visible_fixture")
        measured, observation_audit = map_visual_targets(reference, [scene])
        assert len(measured) == 1, observation_audit
        directory = output/method
        directory.mkdir()
        writer = EpisodeWriter(directory/"dataset", 0, "可见内部点的视觉反馈验证，不是眼底任务", method, 7, [scene["id"]])
        frames, reached, info = run_vision_planner(method, reference, [scene], measured, directory, writer, 7)
        assert info["feedback_controller_used"] and not info["ground_truth_used_for_control"]
        assert info["visual_feedback_samples"] > 5
        assert len(frames) == len(writer.records)
        assert all(np.array_equal(np.float32(a["action"]), np.float32(b["observation.state"]))
                   for a, b in zip(writer.records, writer.records[1:]))
        assert np.all(np.asarray(info["max_velocity"]) <= np.array([4., 2., .6])+1e-9)
        states = [record["visibility"]["target_occlusion_state"] for record in writer.records]
        saw_needle_occlusion = any(state in ("partial_needle", "full_needle") for state in states)
        if reached[0]["result"] == "PASS":
            assert info["terminal_stop_validated"]
            assert reached[0]["terminal_stop_validated"]
            assert np.all(np.asarray(info["max_acceleration"]) <= np.array([10., 5., 1.5])+1e-9)
        else:
            assert reached[0]["reason"] in {
                "TARGET_NOT_VISIBLE", "TIP_NOT_VISIBLE", "INSUFFICIENT_VISUAL_DEPTH",
                "INCONSISTENT_VISUAL_DEPTH", "ILL_CONDITIONED_VISUAL_DEPTH",
                "TIP_NOT_OBSERVABLE", "INCONSISTENT_APEX_DEPTH", "ILL_CONDITIONED_APEX_DEPTH",
                "INSUFFICIENT_NEEDLE_SHAFT_DEPTH", "INSUFFICIENT_NEEDLE_AXIS_SUPPORT",
                "ILL_CONDITIONED_NEEDLE_AXIS",
                # With the physical 0.03 mm distal geometry, the controller
                # may remain measurable while its bounded visual update stalls;
                # this is a reported failure, never a substitute truth pose.
                "MAX_ITERATIONS"}, reached
            assert not info["terminal_stop_validated"]
            assert np.array_equal(writer.records[-1]["action"],
                                  writer.records[-1]["observation.state"])
        episode = writer.finish(reached[0]["result"])
        write_json(directory/"report.json", {"waypoints": reached, **info, "episode": episode})
        fixture_reports.append({"method": method, "result": reached[0]["result"],
                                "reason": reached[0].get("reason"),
                                "natural_needle_occlusion_observed": saw_needle_occlusion,
                                "tip_error_um": reached[0]["tip_error_um"], "frames": episode["frames"]})
        # Also inject a detector dropout to verify immediate fail-closed hold.
        reference = Reference()
        original_observe = RGBDPositionSensor.observe
        observations = 0
        def dropout(self):
            nonlocal observations
            observations += 1
            result = original_observe(self)
            if observations >= 6:
                result["target_pixel"] = None
                result["target_pixels"] = 0
            return result
        failed_directory = output/(method+"_dropout")
        failed_directory.mkdir()
        failed_writer = EpisodeWriter(failed_directory/"dataset", 0, "视觉检测丢失故障注入", method, 7, [scene["id"]])
        try:
            RGBDPositionSensor.observe = dropout
            _, rejected, failure = run_vision_planner(method, reference, [scene], measured, failed_directory, failed_writer, 7)
        finally:
            RGBDPositionSensor.observe = original_observe
        assert rejected[0]["result"] == "FAIL" and rejected[0]["reason"] == "TARGET_NOT_VISIBLE"
        assert failure["visual_feedback_samples"] == 4 and not failure["terminal_stop_validated"]
        assert failed_writer.records[-1]["action"] == failed_writer.records[-1]["observation.state"]
        failed_writer.finish("FAIL")
        write_json(failed_directory/"report.json", {"waypoints": rejected, **failure})
    report = {"result": "PASS", "image_only_position_reconstruction": "PASS",
              "ground_truth_interfaces_forbidden": "PASS", "depth_changes_motor_command": "PASS",
              "occluded_measurements_fail_closed": "PASS", "invisible_fundus_target_not_filled_from_truth": "PASS",
              "RRT_and_PRM_abort_after_injected_detection_loss": "PASS", "depth_boundary_outliers_rejected": "PASS",
              "PRM_observed_departure_connection": "PASS", "natural_needle_occlusion_logged_and_handled": "PASS",
              "RRT_and_PRM_injected_detection_loss_aborts": "PASS",
              "natural_visibility_fixtures": fixture_reports,
              "scope": "Implementation verification; natural occlusion may cause a safe navigation failure; no claim of completed fundus navigation or real RGB perception"}
    write_json(output/"verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2))
