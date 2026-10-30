"""Photo-estimated front tool for the v9 eye scene (metres).

The housing, ferrule and connector are illustrative, uncalibrated
display geometry. The configured proximal/distal needle dimensions remain the
measurement geometry in rcm_scene_v9. All parts move with C_translation.
"""
import math
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def rebuild_front_tool(root, body, shaft_base, case_center_xy):
    profile = json.loads(Path(__file__).with_name("v9_side_profile.json").read_text())
    asset = root.find("asset")
    case_x, case_y = map(float, case_center_xy)
    for geom in list(body.findall("geom")):
        if geom.get("name") in ("scan_C", "needle_holder"):
            body.remove(geom)

    def geom(name, kind, material, **attributes):
        return ET.SubElement(body, "geom", name=name, type=kind,
                             material=material, group="1", **attributes)

    def cylinder(name, z0, z1, radius, material, x=0, y=0):
        geom(name, "cylinder", material,
             fromto=f"{x} {y} {z0} {x} {y} {z1}", size=str(radius))

    def turned(name, rings, material, sides=48):
        vertices = []
        for z, radius in rings:
            vertices.extend((radius*math.cos(2*math.pi*k/sides),
                             radius*math.sin(2*math.pi*k/sides), z)
                            for k in range(sides))
        vertices.extend(((0, 0, rings[0][0]), (0, 0, rings[-1][0])))
        faces = []
        for j in range(len(rings)-1):
            for k in range(sides):
                a, b = j*sides+k, j*sides+(k+1)%sides
                faces.extend(((a, b, b+sides), (a, b+sides, a+sides)))
        for k in range(sides):
            faces.extend(((len(vertices)-2, (k+1)%sides, k),
                          (len(vertices)-1, (len(rings)-1)*sides+k,
                           (len(rings)-1)*sides+(k+1)%sides)))
        ET.SubElement(asset, "mesh", name=name+"_mesh",
                      vertex=" ".join(str(v) for xyz in vertices for v in xyz),
                      face=" ".join(str(v) for tri in faces for v in tri))
        geom(name, "mesh", material, mesh=name+"_mesh")

    def housing(z0, z1):
        assert abs((z1-z0)-profile["case_height_axial_mm"]*.001) < 1e-12
        ET.SubElement(asset, "mesh", name="photo_tool_case_mesh", file=profile["case_mesh"])
        geom("scan_C", "mesh", "tool_case", mesh="photo_tool_case_mesh",
             pos=f"{case_x} {case_y} {z0}")

    # Work backwards from the measured shaft attachment: the visual head stays
    # connected even if the configured proximal shaft length is changed.
    z = shaft_base
    turned("tool_taper", [(z, .00025), (z+.0015, .0018),
                           (z+.003, .0021)], "tool_steel")
    turned("tool_hex_nut", [(z+.003, .0023), (z+.006, .0023)],
           "tool_steel", sides=6)
    cylinder("tool_luer_neck", z+.006, z+.008, .0018, "tool_polymer")
    cylinder("tool_luer_collar", z+.008, z+.014, .0030, "tool_polymer")
    for k in range(6):
        cylinder(f"tool_collar_rib_{k}", z+.008+k*.0009,
                 z+.0083+k*.0009, .00325, "tool_polymer")
    cylinder("tool_metal_sleeve", z+.014, z+.032, .0048, "tool_steel")
    cylinder("tool_sleeve_top", z+.031, z+.034, .0052, "tool_steel")
    cylinder("tool_white_socket", z+.034, z+.054, .0058, "tool_case")
    housing(z+.040, z+.040+profile["case_height_axial_mm"]*.001)
    for sign in (-1, 1):
        cylinder(f"tool_guide_rail_{sign}", z+.031, z+.047, .0018,
                 "tool_steel", x=case_x-.006, y=case_y+sign*.020)
    ET.SubElement(asset, "mesh", name="photo_tool_bracket_mesh", file="photo_tool_bracket_side_r3.obj")
    geom("tool_bottom_bridge", "mesh", "tool_steel", mesh="photo_tool_bracket_mesh",
         pos=f"0 0 {z+.032}")
    # The side attachment is a rigid transparent clamp, not the deleted tube.
    clip_y = case_y-profile["case_width_transverse_mm"]*.0005-.007
    for sign in (-1,1):
        geom(f"tool_clip_post_{sign}", "box", "tool_clear",
             pos=f"{case_x+sign*.013} {clip_y} {z+.067}", size=".0015 .008 .018")
        geom(f"tool_clip_end_{sign}", "box", "tool_clear",
             pos=f"{case_x} {clip_y} {z+.067+sign*.018}", size=".014 .008 .0015")
    geom("tool_clear_clip", "box", "tool_clear",
         pos=f"{case_x+.014} {clip_y} {z+.067}", size=".0005 .007 .016", rgba=".74 .84 .89 .08")
    for k, z_offset in enumerate((.052,.082)):
        geom(f"tool_clip_screw_{k}", "cylinder", "tool_fastener",
             fromto=f"{case_x+.014} {clip_y} {z+z_offset} {case_x+.017} {clip_y} {z+z_offset}",
             size=".0023")
    for k, z_offset in enumerate((.052,.067,.082)):
        geom(f"tool_clip_rim_{k}", "box", "tool_clear", pos=f"{case_x+.012} {clip_y} {z+z_offset}",
             size=".002 .009 .0012")


def add_side_cover_details(root):
    import numpy as np
    import mujoco
    profile = json.loads(Path(__file__).with_name("v9_side_profile.json").read_text())
    parent = root.find(".//body[@name='silver_linear_stage']")
    rotation = np.asarray(profile["cover_panel_basis_local"])
    quat = np.empty(4); mujoco.mju_mat2Quat(quat, rotation.ravel())
    panel = ET.SubElement(parent, "body", name="photo_cover_side_panel")
    def geom(name, kind, material, **kw):
        ET.SubElement(panel, "geom", name=name, type=kind, group="1", material=material, **kw)
    for k, endpoints in enumerate(profile["cover_etch_endpoints_local_m"]):
        geom(f"side_cover_etch_{k}", "capsule", "tool_etch",
             fromto=" ".join(map(str, endpoints[0]+endpoints[1])), size=".00023")
    for k, position in enumerate(profile["cover_fastener_positions_local_m"]):
        geom(f"side_cover_fastener_{k}", "cylinder", "tool_fastener",
             pos=" ".join(map(str,position)),quat=" ".join(map(str,quat)),size=".0018 .0005")
    asset = root.find("asset")
    ET.SubElement(asset,"mesh",name="photo_lower_skirt_mesh",file=profile["base_skirt_mesh"])
    base = root.find(".//body[@name='A_stage']")
    ET.SubElement(base,"geom",name="side_lower_skirt",type="mesh",mesh="photo_lower_skirt_mesh",
                  material="tool_base_gray",group="1")
    normal = np.asarray(profile["base_front_normal_local"])
    for k, position in enumerate(profile["base_ports_local_m"]):
        p = np.asarray(position)
        for suffix, radius, material, depth in (("rim",.006,"tool_steel",.003),
                                                ("socket",.0045,"tool_fastener",.004)):
            endpoints = np.r_[p-normal*.002,p+normal*depth]
            ET.SubElement(base,"geom",name=f"side_base_port_{k}_{suffix}",type="cylinder",
                          fromto=" ".join(map(str,endpoints)),size=str(radius),material=material,group="1")
    tri = np.asarray(profile["base_warning_triangle_local"])
    vertices = np.vstack([tri,tri.mean(axis=0)-normal*.0001])
    ET.SubElement(asset,"mesh",name="photo_base_warning_mesh",
                  vertex=" ".join(str(v) for xyz in vertices for v in xyz),face="0 1 2 0 3 1 1 3 2 2 3 0")
    ET.SubElement(base,"geom",name="side_base_warning",type="mesh",mesh="photo_base_warning_mesh",
                  rgba=".96 .76 .16 1",group="1")
    bolt = profile["base_warning_bolt_local"]
    for k, (a,b) in enumerate(zip(bolt,bolt[1:])):
        ET.SubElement(base,"geom",name=f"side_base_warning_bolt_{k}",type="capsule",
                      fromto=" ".join(map(str,a+b)),size=".00055",material="tool_fastener",group="1")
