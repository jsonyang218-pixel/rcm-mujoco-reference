"""Build the centered, thin side-profile case and bored lower bracket."""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box
import trimesh

HERE = Path(__file__).resolve().parent


def build():
    profile = json.loads((HERE.parent/"v9_side_profile.json").read_text())
    width = profile["case_width_transverse_mm"]*.001
    depth = profile["case_depth_side_mm"]*.001
    height = profile["case_height_axial_mm"]*.001
    corner = profile["case_corner_radius_mm"]*.001
    radius = profile["case_notch_radius_mm"]*.001
    stem = profile["case_notch_stem_height_mm"]*.001
    outline = box(-width/2+corner,corner,width/2-corner,height-corner).buffer(corner,quad_segs=16)
    notch = box(-radius,-.01,radius,stem).union(Point(0,stem).buffer(radius,quad_segs=24))
    outline = outline.difference(notch)
    case = trimesh.creation.extrude_polygon(outline,height=depth,engine="earcut")
    # Polygon coordinates are transverse Y / axial Z; extrusion is depth X.
    v = case.vertices.copy()
    case.vertices = v[:,[2,0,1]]
    case.vertices[:,0] -= depth/2
    case.fix_normals()
    assert case.is_watertight and case.is_volume
    output = HERE/"assets"/profile["case_mesh"]
    case.export(output)
    bracket_width = profile.get("bracket_width_mm", 68.0)*.001
    bracket_depth = profile.get("bracket_depth_mm", 20.0)*.001
    bracket_thickness = profile.get("bracket_thickness_mm", 4.0)*.001
    bore_radius = profile.get("bracket_bore_radius_mm", 8.2)*.001
    if not (0 < bore_radius < min(bracket_width,bracket_depth)/2
            and bracket_thickness > 0):
        raise ValueError("Bracket dimensions must leave material around the sleeve bore")
    ring = box(-bracket_depth/2,-bracket_width/2,
               bracket_depth/2,bracket_width/2).difference(
                   Point(0,0).buffer(bore_radius,quad_segs=32))
    bracket = trimesh.creation.extrude_polygon(ring,height=bracket_thickness,engine="earcut")
    bracket.vertices[:,2] -= bracket_thickness/2
    assert bracket.is_watertight and bracket.is_volume
    bracket.export(HERE/"assets/photo_tool_bracket_side_r3.obj")
    # Conform etch marks and screw heads to the retained scan surface instead
    # of putting a floating rectangular skin over its rounded cover.
    cover = trimesh.load_mesh(HERE/"assets/parallel_cover_smooth.obj",process=False)
    center = np.asarray(profile["cover_panel_center_local_m"])
    basis = np.asarray(profile["cover_panel_basis_local"])
    normal = basis[:,2]

    def on_surface(u,v,offset):
        origin = center+basis@np.array([u,v,.04])
        locations,_,_ = cover.ray.intersects_location(origin[None,:],(-normal)[None,:])
        if not len(locations):
            raise ValueError(f"Side marking outside scan cover: {u}, {v}")
        point = locations[np.argmin(np.linalg.norm(locations-origin,axis=1))]
        return (point+normal*offset).tolist()

    marks = []
    for k in range(23):
        u = (k-11)*.0034
        height = (.006+.009*(1-abs(k-11)/12))*(.48 if k%2 else 1)
        marks.append([on_surface(u,-height/2,.00035), on_surface(u,height/2,.00035)])
    profile["cover_etch_endpoints_local_m"] = marks
    profile["cover_fastener_positions_local_m"] = [on_surface(u,.018,.0008) for u in (-.055,.055)]
    # Photo-estimated lower apron, recessed behind the retained white scan.
    # Express the new solid in A_stage coordinates so it follows that stage.
    from installation import transform
    _, install_r, install_t = transform(HERE)
    alignment = json.loads((HERE.parent/"v9_alignment.json").read_text())
    yaw = np.radians(alignment["yaw_world_deg"])
    yaw_r = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    pivot = np.asarray(alignment["pivot_world_m"])
    yaw_t = pivot-yaw_r@pivot
    mount = json.loads((HERE.parent/"v9_tool_mount.json").read_text())
    stage_r = yaw_r@install_r
    stage_p = yaw_r@(install_r@np.asarray(mount["rcm_registered_m"])+install_t)+yaw_t
    lower, upper = profile["base_skirt_world_y_bounds_m"]
    outline = Polygon(profile["base_skirt_outline_world_xz_m"]).buffer(-.004,join_style="round").buffer(.004,quad_segs=12)
    skirt = trimesh.creation.extrude_polygon(outline,height=upper-lower,engine="earcut")
    v = skirt.vertices.copy()
    world = np.column_stack([v[:,0],v[:,2]+lower,v[:,1]])
    skirt.vertices = (world-stage_p)@stage_r
    skirt.fix_normals()
    assert skirt.is_watertight and skirt.is_volume
    skirt.export(HERE/"assets"/profile["base_skirt_mesh"])
    front_y = lower-.003
    profile["base_ports_local_m"] = [((np.array([x,front_y,z])-stage_p)@stage_r).tolist()
                                       for x,z in profile["base_ports_world_xz_m"]]
    profile["base_front_normal_local"] = (stage_r.T@np.array([0,-1.,0])).tolist()
    x,z = profile["base_warning_world_xz_m"]
    triangle = np.array([[x-.010,front_y,z-.008],[x+.010,front_y,z-.008],[x,front_y,z+.010]])
    profile["base_warning_triangle_local"] = ((triangle-stage_p)@stage_r).tolist()
    bolt = np.array([[x+.001,front_y-.0004,z+.006],[x-.003,front_y-.0004,z],
                     [x+.002,front_y-.0004,z],[x-.001,front_y-.0004,z-.005]])
    profile["base_warning_bolt_local"] = ((bolt-stage_p)@stage_r).tolist()
    (HERE.parent/"v9_side_profile.json").write_text(json.dumps(profile,indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    build()
