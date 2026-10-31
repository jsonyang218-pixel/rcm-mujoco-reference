"""Exercise the real v9 viewer filtering without opening a GUI.

The passive viewer is replaced only at the window boundary. Its real setup
and update callback operate on the injected v9 model; actual MuJoCo renders
then verify the resulting visibility. Optical checks use the real strict
RGB/depth provider, not viewer images or simulator-ID masks for control.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import visual_servo_settings
from run_visual_servo import run_viewer


class WindowBoundary:
    """Record the options used by run_viewer without creating a window."""

    def __init__(self):
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.viewport = mujoco.MjrRect(0, 0, 640, 480)
        self.running = True
        self.images = []
        self.texts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.running = False

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.running

    def set_images(self, value):
        self.images.append(value)

    def set_texts(self, value):
        self.texts.append(value)

    def sync(self):
        pass


def viewer_options(reference):
    """Capture initial and callback-enforced options from the real entrypoint."""
    import mujoco.viewer

    window = WindowBoundary()
    captured = {}
    controller = SimpleNamespace(rows=[{
        'pixel_error_px': 0., 'depth_error_mm': 0.,
        'A_deg': 0., 'B_deg': 0., 'C_mm': 0.,
    }])

    def run():
        captured['initial_groups'] = window.opt.geomgroup.copy()
        # A UI change must not re-enable the environment on the next update.
        window.opt.geomgroup[:] = 1
        controller.iteration_callback(0, {
            'rgb': np.zeros((12, 16, 3), dtype=np.uint8),
            'target_pixel': None, 'tip_pixel': None,
        })
        captured['callback_groups'] = window.opt.geomgroup.copy()
        captured['site_groups'] = window.opt.sitegroup.copy()
        window.running = False
        return {'result': 'PASS', 'scope': 'Viewer callback probe only'}

    controller.run = run
    with patch('mujoco.viewer.launch_passive', return_value=window) as launch, \
            patch('run_visual_servo.time.sleep', return_value=None):
        run_viewer(reference, reference, controller, 'overview')
    captured['launched_model'] = launch.call_args.args[0]
    captured['launched_data'] = launch.call_args.args[1]
    captured['inset_count'] = len(window.images)
    return captured


class V9ViewerTests(unittest.TestCase):
    options = {'callback_groups': np.empty(0, dtype=np.uint8)}
    photo_ids = np.empty(0, dtype=np.int32)
    render_counts = {}
    optical_checks = []

    @classmethod
    def setUpClass(cls):
        cls.reference = RCMSceneV9()
        cls.options = viewer_options(cls.reference)
        cls.photo_ids = np.asarray([
            i for i in range(cls.reference.model.ngeom)
            if (name := mujoco.mj_id2name(
                cls.reference.model, mujoco.mjtObj.mjOBJ_GEOM, i))
            and (name == 'scan_C' or name.startswith(
                ('tool_', 'side_cover_', 'side_base_', 'side_lower_')))
        ], dtype=np.int32)
        cls.environment_ids = np.flatnonzero(
            cls.reference.model.geom_group == 1).astype(np.int32)
        cls.render_counts = {}
        cls.optical_checks = []

    @classmethod
    def tearDownClass(cls):
        cls.reference.close()

    def test_entrypoint_uses_injected_model_and_enforces_filters(self):
        self.assertIs(self.options['launched_model'], self.reference.model)
        self.assertIs(self.options['launched_data'], self.reference.data)
        for name in ('initial_groups', 'callback_groups'):
            np.testing.assert_array_equal(
                self.options[name], [1, 0, 1, 1, 1, 1])
        self.assertEqual(int(self.options['site_groups'].sum()), 0)
        self.assertEqual(self.options['inset_count'], 1)

    def test_photo_parts_pass_viewer_filter_and_environment_does_not(self):
        model = self.reference.model
        groups = self.options['callback_groups']
        self.assertGreaterEqual(len(self.photo_ids), 50)
        self.assertTrue(np.all(groups[model.geom_group[self.photo_ids]] == 1))
        self.assertTrue(np.all(model.geom_group[self.photo_ids] == 0))
        for name in ('scan_C', 'tool_metal_sleeve', 'tool_taper',
                     'tool_hex_nut', 'tool_bottom_bridge', 'tool_white_socket',
                     'side_lower_skirt', 'side_cover_fastener_0'):
            with self.subTest(part=name):
                self.assertIn(model.geom(name).id, self.photo_ids)
        case_mesh = int(model.geom_dataid[model.geom('scan_C').id])
        self.assertEqual(mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_MESH, case_mesh), 'photo_tool_case_mesh')
        self.assertGreater(len(self.environment_ids), 0)
        self.assertTrue(np.all(groups[model.geom_group[self.environment_ids]] == 0))
        for name in ('floor', 'support', 'phantom', 'guide_link', 'rear_link',
                     'upper_link', 'lower_link'):
            with self.subTest(environment=name):
                self.assertIn(model.geom(name).id, self.environment_ids)

    def test_photo_case_sleeve_and_connector_have_visible_pixels(self):
        reference = self.reference
        renderer = mujoco.Renderer(reference.model, height=480, width=640)
        renderer.enable_segmentation_rendering()
        option = mujoco.MjvOption()
        option.geomgroup[:] = self.options['callback_groups']
        option.sitegroup[:] = 0
        parts = ('scan_C', 'tool_metal_sleeve', 'tool_hex_nut')
        try:
            cameras = [name for name in ('overview', 'side', 'needle_close')
                       if mujoco.mj_name2id(reference.model,
                                           mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0]
            self.assertIn('overview', cameras)
            for camera in cameras:
                renderer.update_scene(reference.data, camera=camera, scene_option=option)
                image = renderer.render()
                ids = np.where(
                    image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
                    image[:, :, 0], -1)
                counts = {name: int((ids == reference.model.geom(name).id).sum())
                          for name in parts}
                counts['environment_and_linkage'] = int(
                    np.isin(ids, self.environment_ids).sum())
                self.render_counts[camera] = counts
                self.assertEqual(counts['environment_and_linkage'], 0)
            for part in parts:
                with self.subTest(part=part):
                    self.assertGreater(max(row[part] for row in self.render_counts.values()), 0)
        finally:
            renderer.close()

    def test_photo_display_change_leaves_strict_rgb_and_depth_identical(self):
        reference = self.reference
        sensor = StrictRGBDSensor(reference, visual_servo_settings(reference))
        original_groups = reference.model.geom_group.copy()
        original_task = reference.current_task().copy()
        try:
            for pose, task in (
                    ('home', np.zeros(3)),
                    ('screenshot_pose', np.array([
                        np.deg2rad(4.97), np.deg2rad(-7.22), .01929]))):
                reference.set_task(task)
                reference.model.geom_group[:] = original_groups
                corrected = {mode: sensor._render(mode) for mode in ('rgb', 'depth')}
                # Recreate only the pre-fix photo grouping. Both display groups
                # are excluded from the actual optical passes by construction.
                reference.model.geom_group[self.photo_ids] = 1
                old_groups = {mode: sensor._render(mode) for mode in ('rgb', 'depth')}
                for mode in ('rgb', 'depth'):
                    with self.subTest(pose=pose, mode=mode):
                        np.testing.assert_array_equal(corrected[mode], old_groups[mode])
                self.optical_checks.append({
                    'pose': pose,
                    'rgb_equal': True,
                    'depth_equal': True,
                    'rgb_sha256': hashlib.sha256(corrected['rgb'].tobytes()).hexdigest(),
                    'depth_sha256': hashlib.sha256(corrected['depth'].tobytes()).hexdigest(),
                })
        finally:
            reference.model.geom_group[:] = original_groups
            reference.set_task(original_task)
            sensor.close()


def verify(output_dir=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(V9ViewerTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'result': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'scope': ('Real injected v9 model and real viewer setup/update filtering; '
                  'headless window boundary, rendered visibility and strict RGB/depth invariance; '
                  'not a complete navigation or physical/OCT validation'),
        'viewer_groups': V9ViewerTests.options['callback_groups'].tolist(),
        'photo_display_geom_count': len(V9ViewerTests.photo_ids),
        'visible_photo_pixels_by_camera': V9ViewerTests.render_counts,
        'strict_optical_comparison': V9ViewerTests.optical_checks,
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'viewer_verification.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    report = verify(parser.parse_args().output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['result'] == 'PASS' else 1)
