"""Actual EGL sensor checks for the ideal registered OCT simulation adapter.

This is static sensor acceptance, not closed-loop navigation. Privileged
target_task() poses the scene only; measurement must use current rendered data.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rcm_navigation import load_targets
from rcm_occlusion_fixture import PhysicalOcclusionFixture
from rcm_oct import DEPTH_CONVENTION, RegisteredOCTFrame, SIMULATION_SOURCE, SyntheticOCTProvider
from rcm_reference import Reference
from rcm_visual_servo import VisualServoSensor


def forbidden_truth(*args, **kwargs):
    raise AssertionError('Dynamic truth must not supply OCT sensor measurements')


def verify(output, config_path=Path('config.json'), targets_path=Path('examples/three_waypoints.json')):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    # An actual opaque sphere changes all three sensor passes consistently.
    # Its privileged placement is scene intervention, never a measurement.
    config['reference_scene']['occlusion_fixture'] = {
        'enabled': True, 'radius_mm': .3, 'ray_fraction': .75,
        'events': [{'id': 'oct_sensor_occlusion', 'waypoint_id': '*',
                    'start_frame': 1, 'duration_frames': 1}],
    }
    reference = Reference(config=config)
    sensor = VisualServoSensor(reference)
    fixture = PhysicalOcclusionFixture(reference)
    checks, rows = [], []
    raw = {}
    original_render = sensor._render

    def record_render(mode, include_fundus=None):
        array = original_render(mode, include_fundus)
        if mode == 'depth':
            raw['scene' if include_fundus else 'feature'] = array.copy()
            if not include_fundus:
                raw['feature_segmentation'] = original_render('segmentation', include_fundus=False)
        return array

    sensor._render = record_render
    camera_id = reference.camera_id(sensor.camera_name)
    camera_m = reference.data.cam_xpos[camera_id].copy()
    rotation = reference.data.cam_xmat[camera_id].reshape(3, 3).copy()
    # Independently construct right/down/forward from MuJoCo's camera basis.
    optical_basis = np.column_stack((rotation[:, 0], -rotation[:, 1], -rotation[:, 2]))
    near = reference.eye_camera['znear_mm']*.001
    far = reference.eye_camera['zfar_mm']*.001

    def observe(timestamp):
        sensor.observation_time_s = timestamp
        with ExitStack() as stack:
            for name in ('metrics', 'site_mm', 'target_task', 'current_task'):
                stack.enter_context(patch.object(reference, name, forbidden_truth))
            observation = sensor.observe()
        frame = observation['oct_frame']
        assert observation['depth'] is frame.scene_surface_depth_m
        assert observation['oct_depth'] is frame.registered_feature_depth_m
        assert observation['oct_depth_valid_mask'] is frame.feature_valid_mask
        assert sensor.last_oct_frame is frame
        assert np.array_equal(observation['depth'], raw['scene'], equal_nan=True)
        assert np.array_equal(observation['oct_depth'], raw['feature'], equal_nan=True)
        assert frame.scene_surface_depth_m.dtype == frame.registered_feature_depth_m.dtype == np.float32
        for depth, mask in ((observation['depth'], frame.scene_valid_mask),
                            (observation['oct_depth'], frame.feature_valid_mask)):
            assert np.array_equal(mask, np.isfinite(depth) & (depth > frame.min_valid_depth_m)
                                  & (depth < frame.max_valid_depth_m))
        background = raw['feature_segmentation'][:, :, 0] < 0
        assert background.any()
        assert not frame.feature_valid_mask[background].any()
        metadata = observation['oct_metadata']
        assert metadata == frame.metadata()
        assert metadata['source'] == SIMULATION_SOURCE
        assert metadata['depth_unit'] == 'm' and metadata['depth_convention'] == DEPTH_CONVENTION
        assert metadata['camera_name'] == metadata['registration_frame'] == sensor.camera_name
        assert metadata['clock_domain'] == 'simulation'
        assert metadata['extra']['timestamp_policy'] == 'controller_simulation_clock'
        assert metadata['is_simulation'] and not metadata['raw_oct_available'] and not metadata['realtime_validated']
        assert metadata['min_valid_depth_m'] == sensor.oct_provider.min_valid_depth_m
        assert metadata['max_valid_depth_m'] == sensor.oct_provider.max_valid_depth_m
        assert metadata['extra']['renderer_znear_m'] == near
        assert metadata['extra']['renderer_zfar_m'] == far
        assert frame.timestamp_s == timestamp
        assert tuple(metadata['shape_hw']) == (sensor.height, sensor.width)
        json.dumps(metadata, allow_nan=False)
        return observation

    def relative_check(observation):
        measured = observation['geometry']
        navigation = observation['oct_navigation_depth']
        assert measured['target_world_mm'] is not None, measured
        assert measured['tip_world_mm'] is not None, measured
        assert navigation is not None
        tip_m = np.asarray(measured['tip_world_mm'])*.001
        target_m = np.asarray(measured['target_world_mm'])*.001
        tip_optical = (tip_m-camera_m)@optical_basis
        target_optical = (target_m-camera_m)@optical_basis
        delta = (target_m-tip_m)@optical_basis
        np.testing.assert_allclose(navigation['delta_xyz_m'], delta, atol=1e-16, rtol=1e-12)
        assert np.isclose(navigation['axial_depth_difference_m'], delta[2], atol=1e-16)
        assert np.isclose(observation['tip_depth_m'], tip_optical[2], atol=1e-16)
        assert near < tip_optical[2] < far and near < target_optical[2] < far
        assert np.isclose(navigation['euclidean_distance_m'], np.linalg.norm(target_m-tip_m), atol=1e-16)
        axis = np.asarray(measured['tip_quality']['axis_world'])
        axis /= np.linalg.norm(axis)
        assert np.isclose(navigation['along_needle_difference_m'], (target_m-tip_m)@axis, atol=1e-16)
        assert navigation['definition'] == 'target_minus_tip'
        assert navigation['coordinate_frame'] == 'camera_optical' and navigation['unit'] == 'm'
        assert navigation['source_flags']['frame_id'] == observation['oct_frame'].frame_id
        assert navigation['source_flags']['target_source'] == 'current_visible_measurement'
        # Off-axis point norm differs from optical z: range is not substituted.
        assert abs(np.linalg.norm(target_optical)-target_optical[2]) > 1e-5
        return navigation

    try:
        for index, target in enumerate(load_targets(targets_path)):
            world = (reference.robot_to_world(target['point_mm'])
                     if target['frame'] == 'robot' else np.asarray(target['point_mm']))
            reference.set_target_world(world)
            goal = reference.target_task()
            pose = goal.copy()
            pose[2] *= .8
            reference.set_task(pose)
            observation = observe(4.+index*.05)
            navigation = relative_check(observation)
            assert navigation['axial_depth_difference_m'] > .001
            truth_delta = (world-reference.site_mm('needle_tip'))*.001@optical_basis
            assert abs(navigation['axial_depth_difference_m']-truth_delta[2]) < 10e-6
            rows.append({'target_id': target['id'], 'pose': '80_percent_insertion',
                         'sequence_id': observation['oct_frame'].sequence_id,
                         'timestamp_s': observation['oct_frame'].timestamp_s,
                         'target_pixels': observation['target_pixels'],
                         'tip_pixels': observation['tip_pixels'],
                         'axial_depth_difference_m': navigation['axial_depth_difference_m'],
                         'along_needle_difference_m': navigation['along_needle_difference_m'],
                         'truth_axial_error_um': abs(navigation['axial_depth_difference_m']-truth_delta[2])*1e6})
            # Keep one real frame for independent reload; later checks mutate a copy.
            if index == 2:
                frozen = copy.deepcopy(observation)
                frozen_tip = np.asarray(observation['geometry']['tip_world_mm'])
                saved_path = observation['oct_frame'].save(output/'visible_P3.npz')
        checks.extend(['render_depth_values_unchanged_float32_metre_maps_and_identity_aliases',
                       'configured_near_far_valid_masks_and_truthful_simulation_provenance',
                       'reverse_z_far_background_roundoff_excluded_without_changing_raw_values',
                       'three_waypoint_current_visible_target_minus_tip_optical_coordinates',
                       'dynamic_truth_interfaces_forbidden_during_all_observations'])

        repeated = observe(4.10)
        assert repeated['oct_frame'].timestamp_repeated
        assert repeated['oct_metadata']['fresh_simulation_timestamp'] is False
        assert repeated['oct_frame'].sequence_id == frozen['oct_frame'].sequence_id+1
        assert repeated['oct_frame'].frame_id != frozen['oct_frame'].frame_id
        assert repeated['oct_frame'].acquired_monotonic_s >= frozen['oct_frame'].acquired_monotonic_s
        assert np.array_equal(repeated['oct_depth'], frozen['oct_depth'])
        relative_check(repeated)
        checks.append('explicit_nonzero_simulation_clock_and_repeated_capture_marked_nonfresh')

        # A perturbation through the historical alias must still affect the
        # estimator. Frozen dataclass attributes do not freeze NumPy buffers.
        mask = frozen['segmentation'][:, :, 0] == sensor.geometry.proximal_id
        frozen['oct_depth'][mask] += .00002
        assert frozen['oct_depth'] is frozen['oct_frame'].registered_feature_depth_m
        try:
            changed_tip, _ = sensor.geometry.tip(frozen)
            changed_um = float(np.linalg.norm(changed_tip-frozen_tip)*1000.)
            assert changed_um > 5., changed_um
            causal_outcome = {'outcome': 'estimate_changed', 'tip_change_um': changed_um}
        except RuntimeError as exc:
            causal_outcome = {'outcome': 'inconsistent_depth_rejected', 'reason': str(exc)}
        checks.append('legacy_depth_alias_perturbation_remains_causal')

        # Privileged static posing beyond the point verifies the sign reverses.
        overshoot = goal.copy()
        overshoot[2] += .0003
        reference.set_task(overshoot)
        beyond = observe(4.15)
        reverse = relative_check(beyond)
        assert reverse['axial_depth_difference_m'] < -.0001
        rows.append({'target_id': 'P3', 'pose': '300_um_beyond',
                     'axial_depth_difference_m': reverse['axial_depth_difference_m'],
                     'along_needle_difference_m': reverse['along_needle_difference_m']})
        checks.append('axial_and_along_needle_signs_reverse_beyond_target')

        # Restore a clear scene, then occlude the target with actual geometry.
        pose = goal.copy()
        pose[2] *= .8
        reference.set_task(pose)
        assert fixture.update('P3', 'sensor_check', 1)['active']
        occluded = observe(4.20)
        assert occluded['target_pixels'] == 0 and occluded['target_pixel'] is None
        assert occluded['target_depth_m'] is None
        assert occluded['geometry']['target_world_mm'] is None
        assert occluded['oct_navigation_depth'] is None
        assert occluded['oct_frame'].feature_valid_mask.any()
        occluded['oct_frame'].save(output/'occluded_P3.npz')
        fixture.hide('sensor_check_complete')
        checks.append('actual_opaque_sphere_occlusion_never_invents_current_target_or_relative_depth')

        # Remove shaft support from the actual render, preserving target and
        # distal pixels; a visible taper alone cannot supply a physical tip.
        proximal = sensor.geometry.proximal_id
        original_group = int(reference.model.geom_group[proximal])
        reference.model.geom_group[proximal] = 2
        try:
            unsupported = observe(4.25)
        finally:
            reference.model.geom_group[proximal] = original_group
        assert unsupported['target_pixels'] > 0
        assert unsupported['geometry']['tip_world_mm'] is None
        assert unsupported['tip_depth_m'] is None and unsupported['tip_pixel'] is None
        assert unsupported['oct_navigation_depth'] is None
        checks.append('missing_visible_shaft_support_never_invents_relative_depth')

        # A provider may invalidate depths while retaining their numeric
        # values. Both the legacy depth fields and geometry must honor it.
        sensor.oct_provider = SyntheticOCTProvider(
            nominal_period_s=sensor.settings['sample_period_s'],
            min_valid_depth_m=near, max_valid_depth_m=.001)
        sensor.last_oct_frame = None
        invalid = observe(4.30)
        assert invalid['target_pixels'] > 0
        assert not invalid['oct_frame'].feature_valid_mask.any()
        assert invalid['target_depth_m'] is None and invalid['tip_depth_m'] is None
        assert invalid['geometry']['target_world_mm'] is None
        assert invalid['geometry']['tip_world_mm'] is None
        assert invalid['oct_navigation_depth'] is None
        checks.append('custom_provider_invalid_depth_masks_are_consumed_by_legacy_fields_and_geometry')

        loaded = RegisteredOCTFrame.load(saved_path)
        saved_metadata = json.loads(saved_path.with_suffix('.json').read_text(encoding='utf-8'))
        assert loaded.metadata() == saved_metadata
        with np.load(saved_path, allow_pickle=False) as archive:
            assert all(archive[key].dtype.kind != 'O' for key in archive.files)
            for name in ('scene_surface_depth_m', 'registered_feature_depth_m',
                         'scene_valid_mask', 'feature_valid_mask'):
                assert np.array_equal(getattr(loaded, name), archive[name], equal_nan=True)
        checks.append('actual_render_npz_json_roundtrip_without_pickle')
        result = {'result': 'PASS', 'check_groups': len(checks), 'checks': checks,
                  'static_poses': rows, 'depth_alias_perturbation': causal_outcome,
                  'saved_frames': ['visible_P3.npz', 'occluded_P3.npz'],
                  'scope': 'Static actual EGL-rendered sensor acceptance only; no closed-loop navigation, physical OCT scan, hardware registration or real-time validation.'}
        (output/'oct_sensor_verification.json').write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
        return result
    finally:
        sensor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs/multitarget_occlusion_20261022/oct_sensor_checks'))
    parser.add_argument('--config', type=Path, default=Path('config.json'))
    parser.add_argument('--targets', type=Path, default=Path('examples/three_waypoints.json'))
    args = parser.parse_args()
    result = verify(args.output_dir, args.config, args.targets)
    print(json.dumps({'result': result['result'], 'check_groups': result['check_groups'],
                      'scope': result['scope'], 'report': str(args.output_dir/'oct_sensor_verification.json')},
                     ensure_ascii=False, indent=2))
