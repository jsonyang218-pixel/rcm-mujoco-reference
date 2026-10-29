"""Close segmented display skins without changing source scans or joint frames.

Install requirements-mesh.txt before rebuilding. Inferred patches are visual
reconstruction, not measured CAD, wall thickness, or collision calibration.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pymeshlab
import trimesh

HERE = Path(__file__).resolve().parent
PARTS = {
    'fixed_housing': 'fixed_housing_clean',
    'parallel_cover': 'parallel_cover',
    'distal_guide': 'distal_guide',
    'C_carriage': 'C_carriage',
}


def build():
    report = {'scope': 'Closed visual skins, with inferred missing surfaces',
              'hardware_geometry_calibrated': False,
              'wall_thickness_modeled': False,
              'joint_frames_changed': False, 'collision_enabled': False,
              'reference': 'enclosure_reference.json', 'parts': {}}
    for name, source in PARTS.items():
        path = HERE / 'assets' / f'{source}.obj'
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        original_faces = ms.current_mesh().face_number()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=100)
        ms.meshing_repair_non_manifold_edges(method='Remove Faces')
        ms.meshing_repair_non_manifold_vertices()
        # Ear clipping follows each existing boundary. Refine the new patches
        # rather than spanning the missing surface with a few large triangles.
        ms.meshing_close_holes(maxholesize=20000, selfintersection=False,
                              refinehole=True,
                              refineholeedgelen=pymeshlab.PercentageValue(.8))
        ms.meshing_re_orient_faces_coherently()
        mesh = trimesh.Trimesh(ms.current_mesh().vertex_matrix().copy(),
                               ms.current_mesh().face_matrix().copy(), process=False)
        components = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), engine='scipy')
        retained = [c for c in components if len(c) >= 100]
        mesh = mesh.submesh([np.concatenate(retained)], append=True, repair=False)
        mesh.fix_normals(multibody=True)
        assert mesh.is_watertight and mesh.is_volume, name
        output = HERE / 'assets' / f'{name}_enclosed.obj'
        mesh.export(output)
        # Check the actual exported file, not just the in-memory mesh.
        exported = trimesh.load_mesh(output, process=False)
        assert exported.is_watertight and exported.is_volume, name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
        report['parts'][name] = {
            'source': path.name, 'source_sha256': source_hash,
            'source_faces': original_faces, 'output': output.name,
            'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'faces': len(exported.faces), 'watertight': True,
            'consistent_winding': bool(exported.is_winding_consistent),
            'enclosed_volume_m3': float(exported.volume),
            'bounds_m': exported.bounds.tolist(),
        }
        print(name, 'closed:', len(exported.faces), 'faces', flush=True)
    (HERE / 'enclosed_shells_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    build()
