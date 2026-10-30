"""Estimate a v9 tool frame from the uploaded binary STL's metal sleeve.

The sparse sleeve scan does not determine a calibrated shaft axis. Fit its
radial surface subject to the user's front/side photograph constraints, keeping
the nominal RCM reference height. No source mesh or hardware driver is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def fit(path):
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = json.loads((HERE/"enclosure_reference.json").read_text())["sha256"]
    if source_hash != expected:
        raise ValueError("STL does not match the registered scan; registration must be checked first")
    count = int(np.fromfile(path, dtype="<u4", count=1, offset=80)[0])
    if path.stat().st_size != 84+count*50:
        raise ValueError("Expected an intact binary STL")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3,3)), ("attr", "<u2")])
    triangles = np.memmap(path, dtype=dtype, offset=84, shape=(count,))
    reg = json.loads((HERE/"registration.json").read_text())["new_raw_mm_to_existing_world_m"]
    matrix, translation = np.asarray(reg["A"]), np.asarray(reg["b_m"])
    points, areas = [], []
    for start in range(0,count,100000):
        vertices = np.asarray(triangles["vertices"][start:start+100000], dtype=float)
        center = vertices.mean(axis=1)@matrix.T+translation
        keep = ((center[:,0]>.452)&(center[:,0]<.48)
                &(center[:,1]>.055)&(center[:,1]<.09)
                &(center[:,2]>.145)&(center[:,2]<.165))
        vertices = vertices[keep]
        points.append(center[keep])
        areas.append(np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],
                                              vertices[:,2]-vertices[:,0]), axis=1))
    points, weights = np.concatenate(points), np.concatenate(areas)
    weights /= np.median(weights)
    alignment = json.loads((ROOT/"v9_alignment.json").read_text())
    installation = json.loads((HERE/"installation_config.json").read_text())
    yaw = np.radians(float(alignment["yaw_world_deg"]))
    pitch = np.radians(float(installation["pitch_y_deg"]))
    c, s = np.cos(pitch), np.sin(pitch)
    side = json.loads((ROOT/"v9_side_profile.json").read_text())
    side_pitch = np.radians(float(side["needle_pitch_from_horizontal_deg"]))
    rz = np.array([[np.cos(yaw),-np.sin(yaw),0],
                   [np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    ry = np.array([[c,0,s],[0,1.,0],[-s,0,c]])
    direction = (rz@ry).T@np.array([-np.cos(side_pitch),0,np.sin(side_pitch)])

    def residual(parameters):
        delta = points-np.array([parameters[0],parameters[1],.16])
        radial = delta-np.outer(delta@direction,direction)
        return (np.linalg.norm(radial,axis=1)-parameters[2])*np.sqrt(weights)

    fitted = least_squares(residual, [.46,.066,.005],
                           bounds=([.45,.05,.001],[.48,.08,.012]),
                           loss="soft_l1", f_scale=.0004)
    if not fitted.success:
        raise ValueError(f"Sleeve fit failed: {fitted.message}")
    point = np.array([fitted.x[0],fitted.x[1],.16])
    rcm = point+direction*((.125-point[2])/direction[2])
    y_axis = (rz@ry).T@np.array([0,1.,0])
    x_axis = np.cross(y_axis,direction);x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(direction,x_axis)
    rotation = np.column_stack([x_axis,y_axis,direction])
    # Keep the scan's area center as diagnostic evidence only. Its incomplete,
    # asymmetric surface is not an installation datum. The user's frontal
    # photograph establishes that the case and sleeve are centered together.
    case_path = HERE/"assets/C_carriage.obj"
    case = trimesh.load_mesh(case_path, process=False)
    case_points = (case.triangles_center+np.array([.456,.07,.125])-rcm)@rotation
    case_center = np.average(case_points,axis=0,weights=case.area_faces)
    assert abs(((rz@ry)@direction)[1]) < 1e-12
    return {"source_stl_sha256": source_hash, "source_stl_triangles": count,
            "fit_region_registered_bounds_m": {"x":[.452,.48],"y":[.055,.09],"z":[.145,.165]},
            "fit_surface_points": len(points), "fit_point_registered_m": point.tolist(),
            "fit_axis_registered": direction.tolist(), "fit_radius_m": float(fitted.x[2]),
            "weighted_radial_residual_p50_p90_mm": (np.percentile(abs(residual(fitted.x)),[50,90])*1000).tolist(),
            "constraint": "User side-photo pitch estimate and zero-pose XY parallelism; sleeve position/radius fitted to STL",
            "tool_side_pitch_deg": float(side["needle_pitch_from_horizontal_deg"]),
            "mount_rotation_registered": rotation.tolist(), "rcm_registered_m": rcm.tolist(),
            "rcm_z_reference_m": .125, "installation_pitch_deg": float(installation["pitch_y_deg"]),
            "alignment_yaw_deg": float(alignment["yaw_world_deg"]),
            "case_reference_mesh_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
            "case_scan_area_center_tool_m": case_center.tolist(),
            "case_center_xy_tool_m": [0.,0.],
            "case_center_constraint": "User frontal photograph: centered case, sleeve and needle; zero XY offset",
            "calibrated_to_hardware": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"v9_tool_mount.json")
    args = parser.parse_args()
    report = fit(args.stl)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
