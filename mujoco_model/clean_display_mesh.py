"""Remove scan pedestal and disconnected debris; retain the source mesh."""
from pathlib import Path
import json
import numpy as np
import trimesh
from installation import transform

HERE = Path(__file__).resolve().parent


def main():
    mesh = trimesh.load_mesh(HERE / 'assets/fixed_housing.obj', process=True)
    components = sorted(trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), engine='scipy'),
        key=len, reverse=True)
    # The second component is the detached scan support sheet. Retain the
    # housing and the three substantial mechanical patches, omit tiny debris.
    retained = [c for i, c in enumerate(components) if i != 1 and len(c) >= 1000]
    clean = mesh.submesh([np.concatenate(retained)], append=True, repair=False)
    _, rotation, translation = transform(HERE)
    # Horizontal mounting edge in the default photo-estimated installation.
    normal = rotation.T @ np.array([0., 0., 1.])
    origin = rotation.T @ (np.array([0., 0., .235]) - translation)
    clean = trimesh.intersections.slice_mesh_plane(clean, normal, origin, cap=False)
    clean.export(HERE / 'assets/fixed_housing_clean.obj')
    report = {'source_faces': len(mesh.faces), 'display_faces': len(clean.faces),
              'source_preserved': True, 'watertight': bool(clean.is_watertight),
              'mounting_cut_world_z_m': .235, 'solid_volume_created': False}
    (HERE / 'display_mesh_cleanup.json').write_text(json.dumps(report, indent=2) + '\n')
    print(report)


if __name__ == '__main__':
    main()
