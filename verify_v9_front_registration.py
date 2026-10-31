"""Verify the photo front's real geometry and first-hit visibility.

Simulation geometry is used only for this independent audit. These ray tests
and mesh identities are never supplied to the visual navigation controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from rcm_scene_v9 import RCMSceneV9


ROOT = Path(__file__).resolve().parent
ORIGINAL_GUIDE_SHA256 = "56e0a44314af3b66371a54c11f048db3331b134e26fa44d17483bb1d030d6385"


class FrontRegistrationTests(unittest.TestCase):
    checks = {}

    @classmethod
    def setUpClass(cls):
        cls.scene = RCMSceneV9()
        cls.model, cls.data = cls.scene.model, cls.scene.data
        cls.mount_id = cls.model.body("needle_mount").id
        cls.rotation = cls.data.xmat[cls.mount_id].reshape(3, 3).copy()
        cls.origin = cls.data.xpos[cls.mount_id].copy()
        cls.checks = {}

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    def mesh_in_tool_frame(self, name):
        geom = self.model.geom(name)
        mesh = int(self.model.geom_dataid[geom.id])
        begin = self.model.mesh_vertadr[mesh]
        vertices = self.model.mesh_vert[begin:begin + self.model.mesh_vertnum[mesh]]
        world = (vertices @ self.data.geom_xmat[geom.id].reshape(3, 3).T
                 + self.data.geom_xpos[geom.id])
        return (world - self.origin) @ self.rotation

    def front_ray(self, y, z):
        origin = self.origin + self.rotation @ np.array([.25, y, z])
        direction = -self.rotation[:, 0]
        hit = np.full(1, -1, dtype=np.int32)
        # Audit display geometry only; the environment, sensor-only geoms and
        # optical markers cannot manufacture a passing front-visibility test.
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        distance = mujoco.mj_ray(
            self.model, self.data, origin, direction, groups, 1, -1, hit)
        name = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                int(hit[0])) if hit[0] >= 0 else None)
        return name, float(distance)

    def test_registration_offset_matches_original_60mm_and_current_22mm(self):
        mount = self.scene.tool_mount
        original = ET.parse(ROOT / "mujoco_model/assets/scan_template.xml").getroot()
        old_shaft = np.fromstring(
            original.find(".//site[@name='needle_base']").get("pos"), sep=" ")[2]
        current_shaft = self.scene.config["needle"]["proximal_length_mm"] * .001
        expected_offset = old_shaft - current_shaft
        self.assertAlmostEqual(old_shaft, .060)
        self.assertAlmostEqual(current_shaft, .022)
        self.assertAlmostEqual(expected_offset, .038)
        registered_axis = np.asarray(mount["mount_rotation_registered"])[:, 2]
        old_plane = np.asarray(mount["rcm_fit_plane_registered_m"])
        new_origin = np.asarray(mount["rcm_registered_m"])
        np.testing.assert_allclose(
            new_origin, old_plane + registered_axis * expected_offset,
            rtol=0, atol=1e-12)
        geometry = self.scene.needle_eye_geometry()
        self.assertTrue(geometry["rcm_on_eye_surface"])
        self.assertTrue(geometry["tip_inside_eye"])
        self.assertLess(geometry["entry_rcm_distance_mm"], 1e-5)
        self.assertAlmostEqual(self.scene.metrics()["distal_length_mm"], 4., places=7)
        self.checks["axial_registration_offset_mm"] = expected_offset * 1000
        self.checks["physical_proximal_distal_lengths_mm"] = [22., 4.]

    def test_original_scan_unchanged_and_delivered_upper_guide_closed(self):
        assets = ROOT / "mujoco_model/assets"
        source_path = assets / "distal_guide_smooth.obj"
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.assertEqual(source_hash, ORIGINAL_GUIDE_SHA256)
        report = json.loads((ROOT / "mujoco_model/v9_front_display_build.json").read_text())
        self.assertEqual(report["source_sha256"], source_hash)
        output_path = assets / self.scene.side_profile["scan_guide_mesh"]
        self.assertEqual(hashlib.sha256(output_path.read_bytes()).hexdigest(),
                         report["output_sha256"])
        source = trimesh.load_mesh(source_path, process=False)
        output = trimesh.load_mesh(output_path, process=False)
        self.assertTrue(output.is_watertight)
        self.assertTrue(output.is_volume)
        self.assertGreater(output.volume, 0.)
        plane = np.asarray(report["plane_origin_in_source_frame_m"])
        normal = np.asarray(report["plane_normal_in_source_frame"])
        # Original vertices strictly above the cut remain at their original
        # source coordinates. The new lower cap is the inferred display patch.
        kept = source.vertices[(source.vertices - plane) @ normal > 1e-5]
        self.assertGreater(len(kept), 1000)
        distance = cKDTree(output.vertices).query(kept)[0]
        self.assertLess(float(distance.max()), 1e-10)
        guide = self.mesh_in_tool_frame("scan_distal")
        minimum_z = float(guide[:, 2].min())
        self.assertGreaterEqual(minimum_z, .117 - 2e-8)
        self.assertLess(float(abs(minimum_z - .117)), 2e-8)
        self.checks["source_scan_unchanged"] = True
        self.checks["retained_original_vertex_max_error_m"] = float(distance.max())
        self.checks["derived_guide_closed_positive_volume"] = True
        self.checks["guide_minimum_tool_z_mm"] = minimum_z * 1000

    def test_centered_case_matches_cap_width_and_axial_connection(self):
        case = self.mesh_in_tool_frame("scan_C")
        guide = self.mesh_in_tool_frame("scan_distal")
        low, high = case.min(axis=0), case.max(axis=0)
        dimensions = high - low
        np.testing.assert_allclose(dimensions, [.028, .076, .072], atol=3e-8, rtol=0)
        np.testing.assert_allclose((low + high)[:2] / 2, [0., 0.], atol=2e-8)
        cap_width = float(np.ptp(guide[:, 1]))
        ratio = float(dimensions[1] / cap_width)
        self.assertGreater(ratio, .85)
        self.assertLess(ratio, 1.1)
        # The case hides the lower rear guide; a floating gap must not be
        # introduced between the replacement front and retained upper scan.
        overlap = float(high[2] - guide[:, 2].min())
        self.assertGreaterEqual(overlap, -2e-8)
        self.assertLess(overlap, .03)
        self.checks["case_depth_width_height_mm"] = (dimensions * 1000).tolist()
        self.checks["case_to_cap_width_ratio"] = ratio
        self.checks["case_upper_overlap_with_rear_guide_mm"] = overlap * 1000

    def test_sleeve_needle_and_C_axis_are_coaxial(self):
        rcm = self.scene.site_mm("driver_junction") * .001
        tip = self.scene.site_mm("needle_tip") * .001
        needle_axis = (tip - rcm) / np.linalg.norm(tip - rcm)
        sleeve = self.model.geom("tool_metal_sleeve")
        sleeve_axis = self.data.geom_xmat[sleeve.id].reshape(3, 3)[:, 2]
        self.assertGreater(abs(float(sleeve_axis @ needle_axis)), 1 - 1e-10)
        center_delta = self.data.geom_xpos[sleeve.id] - rcm
        lateral_error = float(np.linalg.norm(
            center_delta - needle_axis * (center_delta @ needle_axis)))
        self.assertLess(lateral_error, 1e-10)
        joint = self.model.joint("C")
        body_rotation = self.data.xmat[int(self.model.jnt_bodyid[joint.id])].reshape(3, 3)
        c_axis = body_rotation @ joint.axis
        self.assertGreater(abs(float(c_axis @ needle_axis)), 1 - 1e-10)
        self.checks["sleeve_needle_C_coaxial"] = True
        self.checks["sleeve_axis_lateral_error_mm"] = lateral_error * 1000

    def test_guide_rails_symmetric_and_clear_clip_is_on_photo_left(self):
        centers = []
        for sign in (-1, 1):
            geom = self.model.geom(f"tool_guide_rail_{sign}")
            centers.append((self.data.geom_xpos[geom.id] - self.origin) @ self.rotation)
        negative, positive = centers
        np.testing.assert_allclose(negative[[0, 2]], positive[[0, 2]], atol=1e-12)
        self.assertAlmostEqual(float(negative[1]), -float(positive[1]), places=11)
        self.assertAlmostEqual(float(positive[1] - negative[1]), .058, places=10)
        clip = self.model.geom("tool_clear_clip")
        clip_center = (self.data.geom_xpos[clip.id] - self.origin) @ self.rotation
        self.assertLess(float(clip_center[1]), -.038)
        self.checks["symmetric_rail_spacing_mm"] = float(positive[1] - negative[1]) * 1000
        self.checks["clear_clip_left_of_center"] = True

    def test_real_front_rays_hit_white_case_before_old_scan(self):
        case = self.mesh_in_tool_frame("scan_C")
        base = float(case[:, 2].min())
        rays = []
        for y, z in [(-.028, base + .025), (.028, base + .025),
                     (-.028, base + .058), (.028, base + .058),
                     (0., base + .060), (-.020, base + .010),
                     (.020, base + .010)]:
            name, distance = self.front_ray(y, z)
            with self.subTest(y=y, z=z):
                self.assertEqual(name, "scan_C")
                self.assertGreater(distance, 0)
            rays.append({"transverse_y_mm": y * 1000,
                         "axial_z_mm": z * 1000, "first_hit": name,
                         "distance_m": distance})
        opening_hit, opening_distance = self.front_ray(0., base + .010)
        self.assertNotEqual(opening_hit, "scan_C")
        self.assertIsNotNone(opening_hit)
        self.assertGreater(opening_distance, 0.)
        # Near the circle centre, the centred 12 mm notch is open to at least
        # 11.5 mm and has white-case material on both sides by 12.5 mm. Offset
        # the height by 0.2 mm to avoid a ray exactly on triangulation vertices.
        for sign in (-1, 1):
            with self.subTest(notch_side=sign):
                inside, _ = self.front_ray(sign * .0115, base + .0062)
                outside, _ = self.front_ray(sign * .0125, base + .0062)
                self.assertNotEqual(inside, "scan_C")
                self.assertEqual(outside, "scan_C")
        self.checks["front_case_first_hit_rays"] = rays
        self.checks["center_notch_ray_first_hit"] = opening_hit
        self.checks["center_notch_symmetric_half_width_bound_mm"] = [11.5, 12.5]
        self.checks["ray_scope"] = "Display group 0 geometry audit only; not control input"


def verify(output_dir=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrontRegistrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"result": "PASS" if result.wasSuccessful() else "FAIL",
              "tests": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors),
              "scope": ("Real compiled front geometry, retained source scan, "
                        "photo-frame first-hit rays and RCM registration; "
                        "not navigation or metrological hardware validation"),
              "checks": FrontRegistrationTests.checks}
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "front_registration_verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    report = verify(parser.parse_args().output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["result"] == "PASS" else 1)
