"""Deburr closed scan skins for display; preserve source assets and frames.

Uses a 0.8 mm voxel grid, 1.6 mm closing and sub-voxel fairing. This
reconstructs small scan features; it is not calibrated manufacturing geometry.
Run from any directory after installing requirements-mesh.txt.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

HERE = Path(__file__).resolve().parent
# C_carriage has already been replaced by the clean photo-built tool case in
# this scene. Reconstruct only the three scan skins that are actually visible.
PARTS = ("fixed_housing", "parallel_cover", "distal_guide")
PITCH = .0008


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface_distances(points, mesh):
    """Upper-bound closest-surface distances using nearby triangle candidates."""
    tree = cKDTree(mesh.triangles_center)
    result = []
    for chunk in np.array_split(points, max(1, len(points)//1000)):
        _, indices = tree.query(chunk, k=24)
        triangles = mesh.triangles[indices.ravel()]
        query = np.repeat(chunk, 24, axis=0)
        closest = trimesh.triangles.closest_point(triangles, query)
        result.extend(np.linalg.norm(closest-query, axis=1).reshape(-1,24).min(axis=1))
    return np.asarray(result)*1000


def build():
    report = {"scope": "Deburred display reconstruction; not measured CAD or collision geometry",
              "voxel_pitch_mm": PITCH*1000, "morphology_radius_mm": 1.6,
              "gaussian_sigma_voxels": .5, "taubin_iterations": 80,
              "humphrey_iterations": 120, "humphrey_alpha": .01, "humphrey_beta": .5,
              "joint_frames_changed": False, "parts": {}}
    for name in PARTS:
        source = HERE/"assets"/(name+"_enclosed.obj")
        source_hash = digest(source)
        mesh = trimesh.load_mesh(source, process=False)
        main_component = max(mesh.split(only_watertight=False), key=lambda p: abs(p.volume))
        voxels = mesh.voxelized(PITCH).fill()
        # Padding keeps the solid away from array boundaries during filtering.
        solid = np.pad(voxels.matrix, 5)
        xyz = np.indices((5, 5, 5))-2
        ball = np.sum(xyz*xyz, axis=0) <= 4
        solid = ndimage.binary_closing(solid, structure=ball)
        labels, count = ndimage.label(solid)
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        # Eliminate isolated scan flakes, retaining substantial separate parts.
        keep = sizes >= max(64, sizes.max()*.001)
        keep[0] = False
        solid = keep[labels]
        field = ndimage.gaussian_filter(solid.astype(np.float32), sigma=.5)
        vertices, faces, _, _ = marching_cubes(field, level=.5, spacing=(PITCH,)*3)
        vertices += voxels.transform[:3, 3]-5*PITCH
        smooth = trimesh.Trimesh(vertices, faces, process=True)
        smooth.fix_normals(multibody=True)
        trimesh.smoothing.filter_taubin(smooth, lamb=.5, nu=.53, iterations=80)
        face_budget = 120000 if name == "fixed_housing" else 60000
        if len(smooth.faces) > face_budget:
            full_resolution = smooth
            reduced = smooth.simplify_quadric_decimation(face_count=face_budget)
            # QEM can collapse tiny disconnected voxel remnants to coincident
            # double faces. Remove zero-volume remnants, not real shell parts.
            components = reduced.split(only_watertight=False)
            reduced = trimesh.util.concatenate([
                part for part in components
                if len(part.faces) >= 20 and abs(part.volume) > 1e-12])
            reduced.fix_normals(multibody=True)
            # Preserve topology on thin covers if QEM cannot reduce them
            # safely; a lighter display mesh must not introduce cracks.
            smooth = reduced if reduced.is_watertight and reduced.is_volume else full_resolution
        # Fair the final triangulation as well: doing this after QEM removes
        # residual grid ripples without introducing global Laplacian shrinkage.
        trimesh.smoothing.filter_humphrey(smooth, alpha=.01, beta=.5, iterations=120)
        assert smooth.is_watertight and smooth.is_volume, name
        # Deterministic sparse surface comparison; nearest-vertex distances
        # bound local reconstruction changes, not exact surface Hausdorff error.
        samples = mesh.vertices[::max(1, len(mesh.vertices)//20000)]
        distances = cKDTree(smooth.vertices).query(samples)[0]*1000
        bounds_change = float(np.max(np.abs(mesh.bounds-smooth.bounds))*1000)
        main_bounds_change = float(np.max(np.abs(main_component.bounds-smooth.bounds))*1000)
        print(name, "source_main_bounds", main_component.bounds.tolist(),
              "smoothed_bounds", smooth.bounds.tolist(), flush=True)
        volume_ratio = abs(smooth.volume/mesh.volume)
        p95_mm = float(np.percentile(distances, 95))
        outward_distances = surface_distances(
            smooth.vertices[::max(1,len(smooth.vertices)//20000)], mesh)
        outward_p95_mm = float(np.percentile(outward_distances, 95))
        print(name, "volume_ratio", volume_ratio, "source_distance_p95_mm", p95_mm, flush=True)
        # Closing removes inaccessible narrow scan recesses: source-to-output
        # distances can be large there. Require the reconstructed outer skin
        # to remain close to source surfaces and preserve total solid volume.
        assert .85 < volume_ratio < 1.15, (name, volume_ratio)
        assert outward_p95_mm < 2, (name, outward_p95_mm)
        output = HERE/"assets"/(name+"_smooth.obj")
        smooth.export(output)
        actual = trimesh.load_mesh(output, process=False)
        assert actual.is_watertight and actual.is_volume, name
        assert digest(source) == source_hash
        report["parts"][name] = {
            "source": source.name, "source_sha256": source_hash,
            "output": output.name, "output_sha256": digest(output),
            "faces": len(actual.faces), "watertight": True,
            "face_budget": face_budget,
            "consistent_winding": bool(actual.is_winding_consistent),
            "max_bounds_change_mm": bounds_change,
            "max_main_component_bounds_change_mm": main_bounds_change,
            "retained_volume_ratio": volume_ratio,
            "output_to_source_surface_distance_mm": {
                "p95": outward_p95_mm, "max": float(outward_distances.max())},
            "sampled_source_to_output_vertex_distance_mm": {
                "median": float(np.median(distances)),
                "p95": float(np.percentile(distances, 95)),
                "max": float(distances.max())},
            "bounds_m": actual.bounds.tolist()}
        print(name, report["parts"][name], flush=True)
    (HERE/"smooth_shells_report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    build()
