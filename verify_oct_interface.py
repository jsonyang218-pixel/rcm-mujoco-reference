"""Fast numerical checks for the explicitly simulated registered OCT interface."""
from __future__ import annotations

import argparse
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np

from rcm_oct import OCTFrame, OCTSource, RegisteredOCTFrame, SyntheticOCTProvider, relative_depth


def _reject(exception, operation):
    try:
        operation()
    except exception:
        return
    raise AssertionError(f"expected {exception.__name__}")


def verify(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    scene = np.array([[.020, .021, np.nan], [0., -1., np.inf]], dtype=np.float64)
    feature = np.array([[.018, .025, np.nan], [0., -1., np.inf]], dtype=np.float64)
    original_scene, original_feature = scene.copy(), feature.copy()
    provider = SyntheticOCTProvider(nominal_period_s=.05)
    assert isinstance(provider, OCTSource)
    frame = provider.capture(scene, feature, timestamp_s=0., acquisition_wall_s=10.)
    assert isinstance(frame, OCTFrame) and OCTFrame is RegisteredOCTFrame
    assert frame.sequence_id == 0 and frame.timestamp_s == 0.
    assert np.array_equal(scene, original_scene, equal_nan=True)
    assert np.array_equal(feature, original_feature, equal_nan=True)
    assert frame.scene_surface_depth_m.dtype == frame.registered_feature_depth_m.dtype == np.float32
    assert not np.shares_memory(scene, frame.scene_surface_depth_m)
    assert not np.shares_memory(feature, frame.registered_feature_depth_m)
    expected_valid = np.array([[True, True, False], [False, False, False]])
    assert np.array_equal(frame.scene_valid_mask, expected_valid)
    assert np.array_equal(frame.feature_valid_mask, expected_valid)
    scene[0, 0], feature[0, 0] = 5., 6.
    assert np.isclose(frame.sample((0, 0), map='scene'), .020)
    assert np.isclose(frame.sample((0, 0)), .018)
    checks.append('copy_inputs_float32_maps_and_finite_positive_validity')

    # Units are declared metres: even a large numeric value is never guessed to
    # be millimetres and silently divided by 1000. Plausible limits are explicit.
    metre_frame = SyntheticOCTProvider().capture([[20.]], [[20.]], timestamp_s=0.)
    assert metre_frame.sample((0, 0)) == 20.
    assert metre_frame.metadata()['depth_unit'] == 'm'
    checks.append('metres_only_without_implicit_mm_conversion')

    for pixel in ((2, 0), (0, 1), (1, 1), (2, 1)):
        _reject(ValueError, lambda pixel=pixel: frame.sample(pixel))
    for pixel in ((-1, 0), (3, 0), (0, 2)):
        _reject(IndexError, lambda pixel=pixel: frame.sample(pixel))
    for pixel in ((.5, 0), (np.nan, 0), ('0', '0'), (0,), (True, False)):
        _reject(ValueError, lambda pixel=pixel: frame.sample(pixel))
    _reject(ValueError, lambda: frame.sample((0, 0), map='bscan'))
    checks.append('invalid_outside_subpixel_and_wrong_map_samples_rejected')

    bounded = SyntheticOCTProvider(min_valid_depth_m=.01, max_valid_depth_m=.04)
    limits = np.array([[.005, .02, .05]], dtype=np.float32)
    limited = bounded.capture(limits, limits, timestamp_s=0.)
    assert np.array_equal(limited.feature_valid_mask, [[False, True, False]])
    assert np.array_equal(limited.registered_feature_depth_m, limits)
    assert limited.metadata()['max_valid_depth_m'] == .04
    _reject(ValueError, lambda: limited.sample((2, 0)))
    checks.append('explicit_near_far_bounds_filter_masks_without_changing_values')

    invalid_provider = SyntheticOCTProvider()
    for bad in (np.ones((2,)), np.ones((0, 2)), np.ones((1, 1, 1)), [[True]], [['depth']], [[1j]]):
        _reject(ValueError, lambda bad=bad: invalid_provider.capture(bad, bad, timestamp_s=0.))
    _reject(ValueError, lambda: invalid_provider.capture([[.02]], [[.02, .03]], timestamp_s=0.))
    for kwargs in ({'nominal_period_s': 0}, {'nominal_period_s': float('inf')},
                   {'min_valid_depth_m': -.1}, {'max_valid_depth_m': 0},
                   {'min_valid_depth_m': .02, 'max_valid_depth_m': .01}):
        _reject(ValueError, lambda kwargs=kwargs: SyntheticOCTProvider(**kwargs))
    assert invalid_provider.capture([[.02]], [[.02]], timestamp_s=0.).sequence_id == 0
    checks.append('shape_numeric_and_bounds_validation_without_consuming_sequence')

    for bad in (-.1, np.nan, np.inf, True):
        _reject(ValueError, lambda bad=bad: provider.capture([[.02]], [[.02]], timestamp_s=bad))
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=0., acquisition_wall_s=11.))
    repeated = provider.capture(
        [[.02]], [[.02]], timestamp_s=0., acquisition_wall_s=11.,
        allow_repeated_timestamp=True, metadata={'acquisition_reason': 'frozen_time_survey'})
    assert repeated.sequence_id == 1
    assert repeated.timestamp_repeated
    assert repeated.metadata()['fresh_simulation_timestamp'] is False
    next_frame = provider.capture([[.02]], [[.02]], timestamp_s=.05, acquisition_wall_s=12.)
    assert next_frame.sequence_id == 2 and not next_frame.timestamp_repeated
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=.04, acquisition_wall_s=13.))
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=.1, acquisition_wall_s=11.))
    fresh = provider.capture([[.02]], [[.02]], timestamp_s=.1, acquisition_wall_s=13.)
    assert fresh.sequence_id == 3
    checks.append('simulation_clock_order_repetition_flags_and_sequence_order')

    other_stream = SyntheticOCTProvider().capture([[.02]], [[.02]], timestamp_s=0.)
    assert other_stream.stream_id != frame.stream_id
    assert other_stream.frame_id != frame.frame_id
    assert other_stream.sequence_id == frame.sequence_id == 0
    assert frame.frame_id.startswith(frame.stream_id)
    checks.append('frame_identifiers_unique_across_provider_streams')

    declared = provider.capture(
        [[.02]], [[.02]], timestamp_s=.15, acquisition_wall_s=14.,
        metadata={'source': 'untrusted_device_label', 'note': 'test metadata stays nested'})
    metadata = declared.metadata()
    json.dumps(metadata, allow_nan=False)
    assert metadata['source'] == 'mujoco_ideal_registered_oct_simulation'
    assert metadata['is_simulation'] is True
    assert metadata['raw_oct_available'] is metadata['realtime_validated'] is False
    assert metadata['clock_domain'] == 'simulation'
    assert metadata['depth_convention'] == 'positive_camera_optical_axis_z_not_ray_range'
    assert metadata['registration_frame'] == metadata['camera_name'] == 'eye_internal'
    assert metadata['nominal_period_s'] == .05 and metadata['nominal_frequency_hz'] == 20.
    assert metadata['nominal_rate_is_wall_clock_guarantee'] is False
    assert 'scene_surface_depth_m' not in metadata and 'registered_feature_depth_m' not in metadata
    _reject(FrozenInstanceError, lambda: setattr(declared, 'raw_oct_available', True))
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=.2, metadata={'array': np.ones((2, 2))}))
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=.2, metadata={'nan': np.nan}))
    _reject(ValueError, lambda: provider.capture([[.02]], [[.02]], timestamp_s=.2, camera_name=''))
    checks.append('truthful_simulation_provenance_json_metadata_and_nominal_rate_only')

    # Saving does not pickle arrays or objects; metadata is a Unicode scalar.
    saved_path = frame.save(output / 'numeric_fixture.npz')
    repeated_path = repeated.save(output / 'repeated_time_fixture.npz')
    loaded = RegisteredOCTFrame.load(saved_path)
    loaded_repeated = RegisteredOCTFrame.load(repeated_path)
    assert loaded.metadata() == frame.metadata()
    assert loaded_repeated.metadata() == repeated.metadata()
    for name in ('scene_surface_depth_m', 'registered_feature_depth_m', 'scene_valid_mask', 'feature_valid_mask'):
        assert np.array_equal(getattr(loaded, name), getattr(frame, name), equal_nan=True)
    with np.load(saved_path, allow_pickle=False) as saved:
        assert all(saved[key].dtype.kind != 'O' for key in saved.files)
    assert json.loads(saved_path.with_suffix('.json').read_text()) == frame.metadata()
    checks.append('npz_json_roundtrip_without_pickle_preserves_masks_maps_and_clock')

    tip = np.array([.001, .002, .020])
    target = np.array([.004, .006, .018])
    axis = np.array([0., 2., 0.])
    original_tip, original_target, original_axis = tip.copy(), target.copy(), axis.copy()
    differences = relative_depth(tip, target, needle_axis_camera=axis,
                                 source_flags={'is_simulation': True, 'registered': True})
    assert np.allclose(differences['delta_xyz_m'], [.003, .004, -.002])
    assert np.isclose(differences['axial_depth_difference_m'], -.002)
    assert np.isclose(differences['along_needle_difference_m'], .004)
    assert np.isclose(differences['euclidean_distance_m'], np.sqrt(29) * .001)
    assert differences['definition'] == 'target_minus_tip'
    assert np.array_equal(tip, original_tip) and np.array_equal(target, original_target)
    assert np.array_equal(axis, original_axis)
    reverse = relative_depth(target, tip)
    assert np.isclose(reverse['axial_depth_difference_m'], .002)
    assert reverse['along_needle_difference_m'] is None
    _reject(ValueError, lambda: relative_depth([0, 0], [0, 0, 1]))
    _reject(ValueError, lambda: relative_depth([0, 0, 0], [0, 0, np.nan]))
    _reject(ValueError, lambda: relative_depth(tip, target, needle_axis_camera=[0, 0, 0]))
    checks.append('signed_axial_and_along_needle_differences_distinct_without_input_mutation')

    result = {
        'result': 'PASS', 'check_groups': len(checks), 'checks': checks,
        'scope': 'Numerical registered-depth adapter only; no hardware, physical OCT image formation, raw B-scan, or real-time validation.',
        'fixture_frame_metadata': frame.metadata(),
    }
    (output / 'oct_interface_verification.json').write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('runs/multitarget_occlusion_20261022/oct_interface_checks'))
    print(json.dumps(verify(parser.parse_args().output_dir), ensure_ascii=False, indent=2))
