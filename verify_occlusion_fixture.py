"""Short scheduler regressions and real RGB-D sphere-occlusion checks.

Scene truth places the fixtures and independently scores estimates. It never
supplies missing geometry to perception or target memory. No navigation runs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rcm_occlusion_fixture import (
    GEOM_NAME, HIDDEN_POSITION_M, PhysicalOcclusionFixture,
    add_occluder_to_xml, fixture_settings,
)


OUTPUT = Path('runs/multitarget_occlusion_20261022/occlusion_checks')


class SmallReference:
    """Minimal real MuJoCo scene for deterministic scheduling checks."""
    def __init__(self, events=(), enabled=True):
        self.config = {'reference_scene': {'occlusion_fixture': {
            'enabled': enabled, 'events': list(events)}}}
        root = ET.Element('mujoco')
        asset = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'camera', name='eye_internal', pos='0 0 1')
        ET.SubElement(world, 'geom', name='lesion_marker', type='sphere', size='.001')
        add_occluder_to_xml(asset, world, self.config)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def camera_id(self, name):
        return int(self.model.camera(name).id)


class SchedulerChecks(unittest.TestCase):
    def test_opt_in_and_default_sphere(self):
        reference = SmallReference(enabled=False)
        fixture = PhysicalOcclusionFixture(reference)
        self.assertFalse(fixture.update('P3', 'approach', 20)['active'])
        self.assertEqual(fixture.geom_id, -1)
        settings = fixture_settings({})
        self.assertFalse(settings['enabled'])
        self.assertEqual(settings['radius_mm'], .3)
        self.assertEqual(settings['ray_fraction'], .75)
        reference = SmallReference()
        fixture = PhysicalOcclusionFixture(reference)
        geom = reference.model.geom(GEOM_NAME)
        self.assertEqual(int(geom.type[0]), mujoco.mjtGeom.mjGEOM_SPHERE)
        self.assertAlmostEqual(float(geom.size[0]), .0003)
        self.assertEqual(float(reference.model.mat_rgba[int(geom.matid[0]), 3]), 1.)

    def test_absolute_default_does_not_restart_at_phase_boundary(self):
        fixture = PhysicalOcclusionFixture(SmallReference([{
            'id': 'absolute', 'waypoint_id': 'P3', 'start_frame': 5,
            'duration_frames': 2, 'phase': 'approach'}]))
        self.assertFalse(fixture.update('P3', 'transfer', 0)['active'])
        self.assertTrue(fixture.update('P3', 'approach', 5)['active'])
        self.assertTrue(fixture.update('P3', 'approach', 6)['active'])
        self.assertFalse(fixture.update('P3', 'approach', 7)['active'])
        self.assertFalse(fixture.update('P3', 'approach', 8)['active'])

    def test_phase_relative_late_phase_and_recovery_keep_origin(self):
        fixture = PhysicalOcclusionFixture(SmallReference([{
            'id': 'relative', 'waypoint_id': 'P3', 'start_frame': 1,
            'duration_frames': 2, 'phase': 'approach', 'relative_to_phase': True}]))
        fixture.update('P3', 'transfer', 0)
        first = fixture.update('P3', 'approach', 40)
        self.assertFalse(first['active'])
        self.assertEqual(first['phase_origin_frame'], 40)
        self.assertTrue(fixture.update('P3', 'approach', 41)['active'])
        # Repeated current frame and a new recovery observation in the same
        # phase do not restart the event or reset the global clock.
        self.assertTrue(fixture.update('P3', 'approach', 41)['active'])
        self.assertTrue(fixture.update('P3', 'approach', 42)['active'])
        fixture.hide('bounded_reobservation')
        last = fixture.update('P3', 'approach', 43)
        self.assertFalse(last['active'])
        self.assertEqual(last['phase_frame_index'], 3)
        fixture.update('P3', 'other_phase', 44)
        self.assertFalse(fixture.update('P3', 'approach', 45)['active'])

    def test_phase_and_waypoint_origins_are_independent(self):
        fixture = PhysicalOcclusionFixture(SmallReference([{
            'id': 'each_phase', 'start_frame': 1, 'duration_frames': 1,
            'relative_to_phase': True}]))
        self.assertFalse(fixture.update('P3', 'align', 10)['active'])
        self.assertTrue(fixture.update('P3', 'align', 11)['active'])
        self.assertFalse(fixture.update('P3', 'approach', 12)['active'])
        self.assertTrue(fixture.update('P3', 'approach', 13)['active'])
        self.assertFalse(fixture.update('P4', 'approach', 0)['active'])
        self.assertTrue(fixture.update('P4', 'approach', 1)['active'])
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            fixture.update('P3', 'align', 2)

    def test_reset_restarts_clock_but_hide_does_not(self):
        fixture = PhysicalOcclusionFixture(SmallReference([{
            'start_frame': 0, 'duration_frames': 1, 'relative_to_phase': True}]))
        self.assertTrue(fixture.update('P3', 'approach', 10)['active'])
        fixture.hide('recovery')
        self.assertFalse(fixture.update('P3', 'approach', 11)['active'])
        fixture.reset()
        self.assertTrue(fixture.update('P3', 'approach', 0)['active'])

    def test_overlap_fails_closed_and_input_and_audit_are_isolated(self):
        events = [{'id': 'a', 'start_frame': 0, 'duration_frames': 2},
                  {'id': 'b', 'start_frame': 1, 'duration_frames': 2}]
        reference = SmallReference(events)
        original_config = copy.deepcopy(reference.config)
        fixture = PhysicalOcclusionFixture(reference)
        state = fixture.update('P3', 'approach', 0)
        self.assertTrue(state['active'])
        self.assertFalse(any('world_mm' in name for name in state))
        self.assertGreater(len(fixture.audit['transitions']), 0)
        state['active'] = False
        audit = fixture.audit
        audit['current']['active'] = False
        self.assertTrue(fixture.audit['current']['active'])
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            fixture.update('P3', 'approach', 1)
        self.assertFalse(fixture.audit['current']['active'])
        np.testing.assert_array_equal(reference.model.geom_pos[fixture.geom_id], HIDDEN_POSITION_M)
        self.assertEqual(reference.config, original_config)

    def test_invalid_configuration_and_clock_are_rejected(self):
        for bad in ['true', 1, None]:
            with self.subTest(relative_to_phase=bad):
                with self.assertRaisesRegex(ValueError, 'relative_to_phase'):
                    SmallReference([{'relative_to_phase': bad}])
        fixture = PhysicalOcclusionFixture(SmallReference())
        for bad in [-1, 1.5, True]:
            with self.subTest(frame=bad):
                with self.assertRaises(ValueError):
                    fixture.update('P3', 'approach', bad)


def array_hashes(observation):
    return {name: hashlib.sha256(observation[name].tobytes()).hexdigest()
            for name in ('rgb', 'depth', 'oct_depth', 'segmentation')}


def render_checks(output):
    from rcm_reference import Reference, save_png
    from rcm_visual_servo import VisualServoSensor

    reference = Reference()
    config = copy.deepcopy(reference.config)
    p3 = next(target for target in json.loads(
        (Path(__file__).parent/'examples/three_waypoints.json').read_text())['targets']
              if target['id'] == 'P3')
    config['reference_scene']['target_robot_mm'] = p3['point_mm']
    config['reference_scene']['occlusion_fixture'] = {
        'enabled': True,
        'events': [{'id': 'p3_physical_sphere', 'waypoint_id': 'P3',
                    'phase': 'approach', 'start_frame': 1,
                    'duration_frames': 1, 'relative_to_phase': True}],
    }
    reference = Reference(config)
    fixture = PhysicalOcclusionFixture(reference)
    sensor = VisualServoSensor(reference)
    goal = reference.target_task().copy()  # Privileged static test placement only.
    rows = []
    try:
        for pose, standoff_mm in [('standoff_2mm', 2.), ('goal', 0.)]:
            fixture.reset()
            q = goal.copy()
            q[2] -= standoff_mm*.001
            reference.set_task(q)
            before_state = fixture.update('P3', 'approach', 20)
            before = sensor.observe()
            before_hashes = array_hashes(before)
            natural = sensor.visibility_metrics(before['segmentation'])
            assert array_hashes(before) == before_hashes
            assert not before_state['active']
            assert before['target_pixels'] > 0 and before['target_pixel'] is not None
            assert before['geometry']['target_world_mm'] is not None
            assert natural['target_occlusion_state'] in ('clear', 'partial_needle')

            during_state = fixture.update('P3', 'approach', 21)
            during = sensor.observe()
            during_hashes = array_hashes(during)
            controlled = sensor.visibility_metrics(during['segmentation'])
            assert array_hashes(during) == during_hashes
            assert during_state['active']
            assert during['target_pixels'] == 0 and during['target_pixel'] is None
            assert during['geometry']['target_world_mm'] is None
            assert during['geometry']['target_error'] == 'TARGET_NOT_VISIBLE'
            # Removing the needle leaves the physical sphere in place. This
            # deliberately labels experimental blockage as not_visible.
            assert controlled['target_occlusion_state'] == 'not_visible', controlled
            assert controlled['target_pixels_unoccluded'] == 0
            assert controlled['target_pixels_occluded_by_needle'] == 0
            quality = during['geometry']['tip_quality']
            assert during['geometry']['tip_world_mm'] is not None
            assert quality['source'] == 'shaft_end_plane', quality
            assert quality['fit_quality_mm'] <= .01, quality
            assert np.any(during['segmentation'][:, :, 0] == fixture.geom_id)

            after_state = fixture.update('P3', 'approach', 22)
            after = sensor.observe()
            recovered = sensor.visibility_metrics(after['segmentation'])
            assert not after_state['active']
            assert after['target_pixels'] == before['target_pixels'] > 0
            assert after['geometry']['target_world_mm'] is not None
            assert array_hashes(before) == before_hashes
            assert array_hashes(during) == during_hashes
            assert array_hashes(after) == before_hashes
            truth_tip = reference.site_mm('needle_tip')  # Independent scoring only.
            error_um = float(np.linalg.norm(
                np.asarray(during['geometry']['tip_world_mm'])-truth_tip)*1000.)
            assert error_um < 10., error_um
            for name, observation in [('before', before), ('sphere_active', during), ('restored', after)]:
                save_png(output/f'P3_{pose}_{name}.png', observation['rgb'])
            rows.append({
                'pose': pose, 'standoff_mm': standoff_mm,
                'resolution_px': [sensor.width, sensor.height],
                'natural_needle_visibility': natural,
                'controlled_sphere_visibility': controlled,
                'restored_visibility': recovered,
                'controlled_target_geometry_error': during['geometry']['target_error'],
                'controlled_tip_geometry': {
                    'tip_world_mm': during['geometry']['tip_world_mm'],
                    'tip_quality': quality, 'independent_tip_error_um': error_um},
                'sphere_pixels': int(np.sum(during['segmentation'][:, :, 0] == fixture.geom_id)),
                'changed_rgb_pixels': int(np.any(before['rgb'] != during['rgb'], axis=2).sum()),
                'raw_arrays_unchanged_after_capture': True,
                'sphere_removal_restores_arrays_byte_for_byte': True,
                'raw_array_hashes': {'natural': before_hashes, 'controlled_sphere': during_hashes},
                'privileged_scene_intervention_audit': fixture.audit,
            })
    finally:
        sensor.close()
    return rows


def verify(output=OUTPUT, skip_render=False):
    output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerChecks)
    run = unittest.TextTestRunner(verbosity=2).run(suite)
    result = {'result': 'PASS' if run.wasSuccessful() else 'FAIL',
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'scheduler_tests_run': run.testsRun,
              'scheduler_failures': len(run.failures), 'scheduler_errors': len(run.errors),
              'renderer_executed': False,
              'scope': 'Static fixture placement and short scheduler tests; no navigation or natural full-needle-occlusion claim.'}
    if run.wasSuccessful() and not skip_render:
        try:
            result['render_checks'] = render_checks(output)
            result['renderer_executed'] = True
        except Exception as exc:
            result.update(result='FAIL', renderer_error=f'{type(exc).__name__}: {exc}')
            (output/'occlusion_fixture_checks.json').write_text(json.dumps(result, indent=2)+'\n')
            raise
    (output/'occlusion_fixture_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    if not run.wasSuccessful():
        raise AssertionError('Occlusion fixture scheduler checks failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--skip-render', action='store_true', help='Run scheduler checks only')
    args = parser.parse_args()
    result = verify(args.output_dir, args.skip_render)
    print(json.dumps({key: value for key, value in result.items() if key != 'render_checks'}, indent=2))
