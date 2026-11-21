"""Pinned G15 default, G10 rollback, isolation and protocol count checks."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DriverSelectionTests(unittest.TestCase):
    def probe(self, code, variant=None):
        env = dict(os.environ)
        env.pop('RCM_DRIVER_VARIANT', None)
        if variant is not None:
            env['RCM_DRIVER_VARIANT'] = variant
        prefix = 'import sys; sys.path.insert(0, "mujoco_model"); '
        return subprocess.run([sys.executable, '-E', '-B', '-c', prefix + code],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=30)

    def test_default_is_pinned_g15(self):
        result = self.probe('import json; from driver_loader import driver_identity; '
                            'print(json.dumps(driver_identity()))')
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual(info['variant'], 'g15')
        self.assertEqual(info['control_gear'], 15)
        self.assertEqual(info['kinematics_sha256'],
                         '3465daf19fd3269466f90fd77c6626149ab4da8ddde612c8f3d9a12c77d06088')

    def test_g10_rollback_in_fresh_process(self):
        result = self.probe('from driver_loader import load_driver; '
                            'print(load_driver().CONTROL_GEAR)', 'g10')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '10')

    def test_unknown_variant_is_rejected(self):
        result = self.probe('from driver_loader import load_driver; load_driver()', 'g99')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('RCM_DRIVER_VARIANT must be g10 or g15', result.stderr)

    def test_process_cannot_silently_switch_cached_version(self):
        result = self.probe('import os; from driver_loader import load_driver; '
                            'load_driver(); os.environ["RCM_DRIVER_VARIANT"]="g10"; '
                            'load_driver()')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fresh process', result.stderr)

    def test_preloaded_external_package_is_rejected(self):
        result = self.probe('import types; from driver_loader import load_driver; '
                            'sys.modules["robot_control"]=types.ModuleType("robot_control"); '
                            'load_driver()')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fresh process', result.stderr)

    def test_both_source_manifests_match(self):
        for manifest_name, directory in (
            ('source_manifest.json', 'simulation_driver'),
            ('source_manifest_g15.json', 'simulation_driver_g15'),
        ):
            manifest = json.loads((ROOT / 'mujoco_model' / manifest_name).read_text())
            self.assertEqual(len(manifest['files_sha256']), manifest['file_count'])
            for name, digest in manifest['files_sha256'].items():
                path = ROOT / 'mujoco_model/vendor' / directory / name
                with self.subTest(path=str(path)):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_single_handle_targets_and_duplicate_packets(self):
        sys.path.insert(0, str(ROOT / 'mujoco_model'))
        from motor_backend import MotorBackend
        motor = MotorBackend(needle_profile='legacy')
        kin = motor.kin
        gear = kin.CONTROL_GEAR
        for count in (1, 2, -1):
            with self.subTest(count=count):
                plant = MotorBackend(needle_profile='legacy')
                home = plant.robot
                expected = (round(home[0] + count * gear * 3. / 2000. * 200000. / 360.),
                            round(kin.ENCODER_COUNTS_PER_MM * kin.b_distance_from_alpha(
                                kin.b_alpha_from_robot_encoder(home[1]) + count*gear*3./4000.)),
                            round(home[2] + count * 3. * gear * 10000. / 4000.), 0)
                command = kin.make_control_message((count, count, count, 0), 'hold')
                receipt = plant.consume(command, 0.)
                self.assertEqual(receipt['target'], expected)
                self.assertTrue(plant.consume(command, .01)['duplicate'])
                self.assertEqual(plant.target, expected)

    def test_scene_and_motor_share_driver_and_keep_four_mm_home(self):
        from rcm.scene_v9 import RCMSceneV9
        scene = RCMSceneV9()
        try:
            scene.enable_motor_execution()
            motor = scene.motor_execution
            self.assertIs(scene.scene.kin, motor.backend.kin)
            self.assertEqual(motor.backend.needle_profile, 'legacy')
            self.assertAlmostEqual(scene._ref.state_from_task([0., 0., 0.]).length_mm, 4.)
            self.assertAlmostEqual(math.dist(scene.site_mm('needle_tip'),
                                             scene.site_mm('rcm_site')), 4., delta=.001)
            scene.set_task([.001, .001, .00001], motor_dt=.05)
            info = motor.summary()['driver']
            self.assertEqual(info['control_gear'], scene.scene.kin.CONTROL_GEAR)
            self.assertEqual(motor.last_record['driver'], info)
        finally:
            scene.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
