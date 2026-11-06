"""Check real exported geometry, world-axis views and rigid tool/eye alignment."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parent


def verify():
    model = mujoco.MjModel.from_xml_path(str(ROOT/"v9_eye_scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geom = model.geom("scan_parallel_cover")
    mesh_id = int(geom.dataid[0])
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start:start+model.mesh_vertnum[mesh_id]]
    rotation = data.geom_xmat[geom.id].reshape(3,3)
    world = vertices@rotation.T+data.geom_xpos[geom.id]
    rectangle = np.asarray(MultiPoint(world[:,:2]).minimum_rotated_rectangle.exterior.coords)
    edges = np.diff(rectangle, axis=0)
    axis = edges[np.argmax(np.linalg.norm(edges,axis=1))]
    if axis[0] < 0:
        axis = -axis
    angle = float(np.degrees(np.arctan2(axis[1],axis[0])))
    assert abs(angle) < .01, angle
    for name in ("horizontal_mount_plate", "phantom_pad", "target_table_top"):
        geom = model.geom(name)
        np.testing.assert_allclose(data.geom_xmat[geom.id].reshape(3,3)[:,2], [0,0,1], atol=1e-10)
    eye = model.geom("eye_reference")
    eye_rotation = data.geom_xmat[eye.id].reshape(3,3)
    rcm = data.site_xpos[model.site("rcm_site").id]
    tip = data.site_xpos[model.site("needle_tip").id]
    junction = data.site_xpos[model.site("driver_junction").id]
    needle_axis = tip-junction
    needle_angle = float(np.degrees(np.arctan2(needle_axis[1],needle_axis[0])))
    assert abs(needle_angle-angle) < .01, (needle_angle, angle)
    # The holder and C translation must follow the physical needle direction,
    # not only a viewer-only display copy.
    sleeve = model.geom("tool_metal_sleeve")
    sleeve_axis = data.geom_xmat[sleeve.id].reshape(3,3)[:,2]
    needle_axis /= np.linalg.norm(needle_axis)
    side_profile = json.loads((ROOT/"v9_side_profile.json").read_text())
    side_pitch = float(np.degrees(np.arctan2(-needle_axis[2],np.linalg.norm(needle_axis[:2]))))
    assert abs(side_pitch-side_profile["needle_pitch_from_horizontal_deg"]) < .01
    assert abs(float(sleeve_axis@needle_axis)) > 1-1e-10
    c_joint = model.joint("C")
    c_body = int(model.jnt_bodyid[c_joint.id])
    c_axis = data.xmat[c_body].reshape(3,3)@c_joint.axis
    assert abs(float(c_axis@needle_axis)) > 1-1e-10
    center = data.geom_xpos[eye.id]
    value = float(np.sum((eye_rotation.T@(rcm-center)/eye.size)**2))
    assert abs(value-1) < 1e-8
    assert np.sum((eye_rotation.T@(tip-center)/eye.size)**2) < 1
    # Re-register the tool pivot without translating the original scan shells
    # at zero pose. Compare raw mesh coordinate frames, not their differing
    # principal-inertia centers after smoothing.
    sys.path.insert(0,str(ROOT/"mujoco_model"))
    from model_sync import DriverModel
    source = DriverModel(needle_profile="legacy")
    source_xml = ET.fromstring(source.xml)
    exported_xml = ET.parse(ROOT/"v9_eye_scene.xml").getroot()
    config = json.loads((ROOT/"v9_alignment.json").read_text())
    yaw = np.radians(config["yaw_world_deg"])
    q = np.array([[np.cos(yaw),-np.sin(yaw),0],
                  [np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    pivot = np.asarray(config["pivot_world_m"])
    translation = pivot-q@pivot
    for name in ("scan_fixed_housing","scan_parallel_cover","scan_distal"):
        old_body = int(source.model.geom(name).bodyid[0])
        new_body = int(model.geom(name).bodyid[0])
        old_pos = np.fromstring(source_xml.find(f".//geom[@name='{name}']").get("pos","0 0 0"),sep=" ")
        new_pos = np.fromstring(exported_xml.find(f".//geom[@name='{name}']").get("pos","0 0 0"),sep=" ")
        points = np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]])
        old_points = (points+old_pos)@source.data.xmat[old_body].reshape(3,3).T+source.data.xpos[old_body]
        expected = old_points@q.T+translation
        actual = (points+new_pos)@data.xmat[new_body].reshape(3,3).T+data.xpos[new_body]
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
    return {"result": "PASS", "top_view_long_axis_angle_deg": angle,
            "top_view_needle_axis_angle_deg": needle_angle,
            "side_view_needle_pitch_deg": side_pitch,
            "holder_and_C_axis_coaxial": True,
            "registered_scan_pose_preserved_at_zero": True,
            "support_faces_horizontal": True, "rcm_on_eye_surface": True,
            "tip_inside_eye": True}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
