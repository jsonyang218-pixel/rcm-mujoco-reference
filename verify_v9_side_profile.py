"""Verify the side-photo solid geometry, real openings and part attachment."""
from pathlib import Path
import json

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent


def verify():
    profile = json.loads((ROOT/"v9_side_profile.json").read_text())
    assets = ROOT/"mujoco_model/assets"
    case = trimesh.load_mesh(assets/profile["case_mesh"],process=False)
    bracket = trimesh.load_mesh(assets/"photo_tool_bracket_side_r3.obj",process=False)
    skirt = trimesh.load_mesh(assets/profile["base_skirt_mesh"],process=False)
    for mesh in (case,bracket,skirt):
        assert mesh.is_watertight and mesh.is_volume and mesh.is_winding_consistent
    # The notch and shaft bore must be actual empty geometry, not black decals.
    origins = np.array([[-.1,0,.004],[-.1,.020,.004]])
    hit = case.ray.intersects_any(origins,np.array([[1.,0,0],[1.,0,0]]))
    assert hit.tolist() == [False,True]
    origins = np.array([[0,0,-.01],[.007,.020,-.01]])
    hit = bracket.ray.intersects_any(origins,np.array([[0.,0,1],[0.,0,1]]))
    assert hit.tolist() == [False,True]
    model = mujoco.MjModel.from_xml_path(str(ROOT/"v9_eye_scene.xml"))
    assert model.geom("side_lower_skirt").bodyid == model.body("A_stage").id
    panel = model.body("photo_cover_side_panel").id
    assert model.body_parentid[panel] == model.body("silver_linear_stage").id
    mount = model.body("needle_mount").id
    assert model.geom("tool_clear_clip").bodyid == mount
    assert profile["fluid_tube_enabled"] is False
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ""
        assert not name.startswith("tool_fluid_line_")
        if name.startswith(("side_","tool_")):
            assert model.geom_group[i] == 1
    return {"result":"PASS","closed_case_bracket_and_skirt":True,
            "center_notch_and_shaft_bore_open":True,
            "decorations_attached_and_sensor_excluded":True,
            "deleted_fluid_tube_still_absent":True}


if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
