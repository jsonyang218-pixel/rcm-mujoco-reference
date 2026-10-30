"""Check the photo tool, outward needle, and v9 logical-task mapping."""
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from rcm_scene_v9 import RCMSceneV9


def verify():
    scene = RCMSceneV9()
    try:
        m = scene.model
        cid = m.body("C_translation").id
        mount = m.body("needle_mount").id
        assert m.body_parentid[mount] == cid
        assert m.geom("scan_C").bodyid == mount
        assert m.geom("tool_metal_sleeve").bodyid == mount
        np.testing.assert_array_equal(scene.tool_mount["case_center_xy_tool_m"], [0.,0.])
        # Use actual compiled geometry rather than only trusting the config:
        # the rounded case's XY bounds must center on the physical needle axis.
        case = m.geom("scan_C")
        mesh_id = int(case.dataid[0])
        start = m.mesh_vertadr[mesh_id]
        vertices = m.mesh_vert[start:start+m.mesh_vertnum[mesh_id]]
        r = np.empty(9);mujoco.mju_quat2Mat(r,case.quat)
        local = vertices@r.reshape(3,3).T+case.pos
        case_center = (local.min(axis=0)+local.max(axis=0))*.5
        np.testing.assert_allclose(case_center[:2], [0,0], atol=2e-8)
        dimensions = local.max(axis=0)-local.min(axis=0)
        profile = scene.side_profile
        expected_dimensions = np.array([profile["case_depth_side_mm"],profile["case_width_transverse_mm"],
                                        profile["case_height_axial_mm"]])*.001
        np.testing.assert_allclose(dimensions,expected_dimensions,atol=2e-8)
        np.testing.assert_allclose(m.geom("tool_metal_sleeve").pos[:2], [0,0], atol=1e-12)
        assert m.geom("tool_clear_clip").bodyid == mount
        for i in range(m.ngeom):
            name = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ""
            assert not name.startswith("tool_fluid_line_")
            if name.startswith("side_cover_"):
                assert m.geom_group[i] == 1
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("tool_"):
                assert m.geom_group[g] == 1
                assert m.geom_bodyid[g] == mount
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "axis_aid") < 0
        assert np.isclose(m.geom("needle_proximal").size[0], .00025)
        assert np.isclose(m.geom("needle_distal").size[0], .00003)
        assert m.geom("needle_proximal").type == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert m.geom("needle_distal").type == mujoco.mjtGeom.mjGEOM_CYLINDER
        for name in ("A", "C"):
            actuator = m.actuator(name+"_servo")
            np.testing.assert_allclose(actuator.ctrlrange, m.joint(name).range)
        assert m.site_pos[scene.sites["needle_base"], 2] > 0
        assert m.site_pos[scene.sites["needle_tip"], 2] < 0
        for name in ("needle_proximal", "needle_distal", "needle_distal_tip"):
            p = m.geom(name)
            d = m.geom(name.replace("needle_", "needle_display_", 1))
            np.testing.assert_array_equal(p.size, d.size)
            np.testing.assert_array_equal(p.pos, d.pos)
            assert d.group == 2 and p.group == 3

        worst_mm = 0.
        samples = 0
        for a, b, c in itertools.product((-10., 0., 10.), (-5., 0., 5.), (0., 4., 8.)):
            q = np.array([np.radians(a), np.radians(b), c*.001])
            scene.set_task(q)
            expected = scene.robot_to_world(scene._ref.forward_kinematics(q))
            worst_mm = max(worst_mm, float(np.linalg.norm(scene.site_mm("needle_tip")-expected)))
            assert scene.needle_eye_geometry()["entry_rcm_distance_mm"] < 1e-5
            np.testing.assert_allclose(scene.current_task(), q, atol=1e-12)
            samples += 1
        assert worst_mm < .001
        scene.set_task(np.zeros(3))
        geometry = scene.needle_eye_geometry()
        assert geometry["rcm_on_eye_surface"] and geometry["tip_inside_eye"]
        assert geometry["entry_needle_part"] == "junction"

        # Check the actual deliverable, not only the runtime adapter.
        exported = mujoco.MjModel.from_xml_path(str(Path(__file__).parent/"v9_eye_scene.xml"))
        data = mujoco.MjData(exported)
        mujoco.mj_forward(exported, data)
        eye = exported.geom("eye_reference")
        rotation = data.geom_xmat[eye.id].reshape(3, 3)
        for name, on_surface in (("rcm_site", True), ("needle_tip", False)):
            p = data.site_xpos[exported.site(name).id]
            value = float(np.sum((rotation.T@(p-data.geom_xpos[eye.id])/eye.size)**2))
            assert abs(value-1) < 1e-8 if on_surface else value < 1
        return {"result": "PASS", "poses": samples, "max_fk_error_mm": worst_mm,
                "case_and_needle_center_offset_mm": (case_center[:2]*1000).tolist(),
                "case_depth_width_height_mm": (dimensions*1000).tolist(),
                "outward_needle_and_attached_head": True,
                "exported_xml_eye_entry": True,
                "photo_dimensions_calibrated": False}
    finally:
        scene.close()


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
