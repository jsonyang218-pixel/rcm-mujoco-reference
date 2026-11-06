"""Validate smoothed display skins and preservation of source scans/frames."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent


def verify():
    folder = ROOT/"mujoco_model"
    report = json.loads((folder/"smooth_shells_report.json").read_text())
    xml = ET.parse(ROOT/"v9_eye_scene.xml").getroot()
    result = {}
    for name, info in report["parts"].items():
        for key in ("source", "output"):
            file = folder/"assets"/info[key]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == info[key+"_sha256"]
        assert xml.find(f"./asset/mesh[@name='{name}']").get("file") == info["output"]
        mesh = trimesh.load_mesh(folder/"assets"/info["output"], process=False)
        assert mesh.is_watertight and mesh.is_volume and mesh.is_winding_consistent
        assert np.isfinite(mesh.vertices).all()
        assert np.isfinite(mesh.vertex_normals).all()
        source = trimesh.load_mesh(folder/"assets"/info["source"], process=False)
        # Compare sharp joins between adjacent triangles, in degrees. A smooth
        # reconstruction should remove the high-angle creases of scan burrs.
        old_sharp = np.mean(source.face_adjacency_angles > np.radians(45))
        new_sharp = np.mean(mesh.face_adjacency_angles > np.radians(45))
        assert new_sharp < old_sharp, (name, old_sharp, new_sharp)
        result[name] = {"watertight": True, "original_preserved": True,
                        "sharp_edge_fraction_before": float(old_sharp),
                        "sharp_edge_fraction_after": float(new_sharp)}
    # Visual mesh swaps must not change the tool kinematic sites or eye pose.
    smooth = mujoco.MjModel.from_xml_path(str(ROOT/"v9_eye_scene.xml"))
    for name, info in report["parts"].items():
        xml.find(f"./asset/mesh[@name='{name}']").set("file", info["source"])
    xml.find("compiler").set("meshdir", str(folder/"assets"))
    original = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    for q in (np.zeros(smooth.nq), np.array([.1, .05, -.002, .05, -.05, -.05, .001])):
        a, b = mujoco.MjData(smooth), mujoco.MjData(original)
        a.qpos[:] = q; b.qpos[:] = q
        mujoco.mj_forward(smooth, a); mujoco.mj_forward(original, b)
        np.testing.assert_array_equal(a.site_xpos, b.site_xpos)
        np.testing.assert_array_equal(a.cam_xpos, b.cam_xpos)
    return {"result": "PASS", "parts": result, "kinematic_sites_unchanged": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
