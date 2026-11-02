"""Round only the lower 3 mm depth-side edges of the existing white case.

The original case asset and upper scan are never overwritten. Only existing
cap triangles are reused; no Boolean or external triangulator is required.
"""
from pathlib import Path
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent


def build():
    source = trimesh.load_mesh(HERE/'assets/photo_tool_case_side_r3.obj', process=False)
    cap = source.faces[np.all(np.isclose(source.vertices[source.faces, 0], -.014), axis=1)]
    ids, inverse = np.unique(cap, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    yz = source.vertices[ids, 1:].copy()
    edges = np.sort(np.vstack([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    # Loft identical upper contours and quarter-round lower side corners.
    # The original central arch and its radius remain unchanged.
    radius = .003
    side = .011 + radius*np.sin(np.linspace(0, np.pi/2, 13))
    levels = np.r_[-side[::-1], side]
    vertices, faces = [], []
    n = len(yz)
    for x in levels:
        points = np.column_stack([np.full(n, x), yz])
        edge = max(abs(x)-.011, 0.)
        lift = radius - np.sqrt(max(radius**2-edge**2, 0.))
        low = points[:, 2] < radius
        points[low, 2] += lift*(1-points[low, 2]/radius)
        vertices.extend(points)
    faces.extend(triangles[:, ::-1])
    faces.extend(triangles+(len(levels)-1)*n)
    for k in range(len(levels)-1):
        for a, b in boundary:
            faces.extend(((k*n+a,k*n+b,(k+1)*n+b),
                          (k*n+a,(k+1)*n+b,(k+1)*n+a)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    assert mesh.is_watertight and mesh.is_volume
    target = HERE/'assets/photo_tool_case_redframe.obj'
    target.write_text(trimesh.exchange.obj.export_obj(mesh, digits=12))
    print(target)


if __name__ == '__main__':
    build()
