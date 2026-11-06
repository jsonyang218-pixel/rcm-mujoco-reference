"""Keep the source scan's upper cap/guide as a closed v9 display asset.

The front photograph supplies the replacement-envelope boundary. The source
scan, virtual joints and calibrated needle geometry are not edited. All mesh
coordinates and plane distances in this script are in metres.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    assets = HERE / "assets"
    profile = json.loads((ROOT / "v9_side_profile.json").read_text())
    mount = json.loads((ROOT / "v9_tool_mount.json").read_text())
    source_path = assets / "distal_guide_smooth.obj"
    output_path = assets / profile["scan_guide_mesh"]
    if output_path.parent.resolve() != assets.resolve() or output_path.suffix != ".obj":
        raise ValueError("scan_guide_mesh must name an OBJ in the assets directory")
    if output_path.resolve() == source_path.resolve():
        raise ValueError("The original smooth scan must remain unchanged")

    source_hash = sha256(source_path)
    source = trimesh.load_mesh(source_path, process=False)
    rotation = np.asarray(mount["mount_rotation_registered"], dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("The registered tool frame must be orthonormal")
    normal = rotation[:, 2]
    rcm = np.asarray(mount["rcm_registered_m"], dtype=float)
    fit_plane = np.asarray(mount["rcm_fit_plane_registered_m"], dtype=float)
    registration_offset = float(mount["front_axial_registration_offset_mm"]) * .001
    np.testing.assert_allclose(
        rcm - fit_plane, normal * registration_offset, atol=1e-12, rtol=0,
        err_msg="The front registration must be an axial shift of the fit plane")
    template = ET.parse(assets / "scan_template.xml").getroot()
    original_stage = template.find(".//body[@name='A_stage']")
    original_origin = np.fromstring(original_stage.get("pos"), sep=" ")
    cut_height = float(profile["scan_guide_cut_height_tool_mm"]) * .001
    if not np.isfinite(cut_height) or cut_height <= 0:
        raise ValueError("The display cut height must be finite and positive")

    # A source vertex v is displayed at v + original_origin - new_rcm in
    # the rebased A-stage. Preserve the part above tool-local cut_height.
    plane_origin = rcm - original_origin + normal * cut_height
    signed_source = (source.vertices - plane_origin) @ normal
    if not (signed_source.min() < 0 < signed_source.max()):
        raise ValueError("The photo replacement plane must cross the source scan")
    result = trimesh.intersections.slice_mesh_plane(
        source, plane_normal=normal, plane_origin=plane_origin, cap=True)
    result.fix_normals(multibody=True)
    if not (result.is_watertight and result.is_volume):
        raise ValueError("The retained cap and guide must be a closed positive volume")
    if float(((result.vertices - plane_origin) @ normal).min()) < -1e-10:
        raise ValueError("The retained asset crosses below the replacement plane")

    # Keep the original source-coordinate frame; MuJoCo applies its existing
    # stage/installation transforms. Extra decimal places avoid export drift.
    output_path.write_text(trimesh.exchange.obj.export_obj(result, digits=12))
    exported = trimesh.load_mesh(output_path, process=False)
    if not (exported.is_watertight and exported.is_volume):
        raise ValueError("The delivered OBJ is not a closed positive volume")
    np.testing.assert_allclose(exported.bounds, result.bounds, atol=1e-11, rtol=0)
    if sha256(source_path) != source_hash:
        raise RuntimeError("The original smooth scan was modified")

    report = {
        "result": "PASS",
        "scope": ("Photo-estimated display-envelope replacement only; no source "
                  "scan overwrite, physical needle change or hardware calibration"),
        "units": "metres",
        "source": source_path.relative_to(ROOT).as_posix(),
        "source_sha256": source_hash,
        "source_faces": len(source.faces),
        "source_bounds_m": source.bounds.tolist(),
        "source_unchanged": True,
        "output": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(output_path),
        "output_faces": len(exported.faces),
        "output_bounds_m": exported.bounds.tolist(),
        "output_watertight": bool(exported.is_watertight),
        "output_positive_volume": bool(exported.is_volume),
        "output_volume_m3": float(exported.volume),
        "source_coordinate_frame_preserved": True,
        "original_A_stage_origin_registered_m": original_origin.tolist(),
        "rcm_fit_plane_registered_m": fit_plane.tolist(),
        "new_rcm_registered_m": rcm.tolist(),
        "front_axial_registration_offset_mm": registration_offset * 1000,
        "tool_cut_height_mm": cut_height * 1000,
        "cut_height_relative_to_original_fit_plane_mm": (
            registration_offset + cut_height) * 1000,
        "plane_normal_in_source_frame": normal.tolist(),
        "plane_origin_in_source_frame_m": plane_origin.tolist(),
        "kept_halfspace": "dot(vertex - plane_origin, plane_normal) >= 0",
        "retained_min_signed_distance_m": float(
            ((exported.vertices - plane_origin) @ normal).min()),
        "cap_enabled": True,
        "photo_dimensions_calibrated": False,
    }
    report_path = HERE / "v9_front_display_build.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
