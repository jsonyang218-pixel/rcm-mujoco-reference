"""Audit the local display edit against an explicitly captured pre-edit model."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from rcm_scene_v9 import RCMSceneV9

ROOT = Path(__file__).resolve().parent
ALLOWED = {'scan_C', 'photo_tool_case_mesh'}


def xml_records(xml):
    root = ET.fromstring(xml)
    records = {}
    def walk(node, path):
        name = node.get('name', '')
        if name in ALLOWED or name.startswith('redframe_'):
            return
        if node.tag == 'inertial' and path.endswith('/body:needle_mount'):
            return  # The newly explicit inertia is compared after compilation.
        key = path + '/' + node.tag + ':' + name
        records[key] = dict(sorted(node.attrib.items()))
        for child in node:
            walk(child, key)
    walk(root, '')
    return records


def snapshot(scene):
    root = ET.fromstring(scene.xml)
    records = xml_records(scene.xml)
    meshes = {}
    meshdir = Path(root.find('compiler').get('meshdir'))
    for mesh in root.findall('./asset/mesh[@file]'):
        if mesh.get('name') not in ALLOWED:
            path = meshdir / mesh.get('file')
            meshes[mesh.get('name')] = hashlib.sha256(path.read_bytes()).hexdigest()
    m = scene.model
    dynamics = {m.body(i).name: {k: np.asarray(getattr(m, k)[i]).tolist()
                for k in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat')}
                for i in range(m.nbody) if not m.body(i).name.startswith('redframe_')}
    return {'xml_records': records, 'mesh_hashes': meshes, 'body_dynamics': dynamics,
            'config': scene.config, 'mount': scene.tool_mount, 'alignment': scene.alignment}


def verify(baseline):
    scene = RCMSceneV9()
    try:
        current = snapshot(scene)
        for key in current.keys() - {'body_dynamics'}:
            assert current[key] == baseline[key], f'Outside allowlist changed: {key}'
        assert current['body_dynamics'].keys() == baseline['body_dynamics'].keys()
        for body, values in current['body_dynamics'].items():
            for key, value in values.items():
                np.testing.assert_allclose(value, baseline['body_dynamics'][body][key], rtol=1e-12, atol=1e-15)
        original_xml = subprocess.check_output(['git', 'show', 'ffd7b15:v9_eye_scene.xml'], cwd=ROOT, text=True)
        assert xml_records((ROOT/'v9_eye_scene.xml').read_text()) == xml_records(original_xml)
        m, d = scene.model, scene.data
        guide = m.geom('redframe_guide_plate')
        assert m.body_parentid[int(guide.bodyid[0])] == m.body('distal_orientation').id
        for name in ('scan_C', 'tool_clear_clip', 'tool_bottom_bridge', 'redframe_clip_crossbar_0',
                     'redframe_socket_shoulder'):
            assert m.geom(name).bodyid == m.body('needle_mount').id
        case = trimesh.load_mesh(ROOT/'mujoco_model/assets/photo_tool_case_redframe.obj', process=False)
        assert case.is_watertight and case.is_volume and case.is_winding_consistent
        source = trimesh.load_mesh(ROOT/'mujoco_model/assets/photo_tool_case_side_r3.obj', process=False)
        # Upper case, front width and central arch are retained; only lower
        # depth-side edges (within 3 mm of the bottom) are rounded.
        from scipy.spatial import cKDTree
        upper = source.vertices[source.vertices[:, 2] > .0031]
        assert cKDTree(case.vertices).query(upper)[0].max() < 1e-9
        from render_v9_three_views import GeometryBounds
        cap = m.geom('scan_distal').id
        mid = m.geom_dataid[cap]
        begin = m.mesh_faceadr[mid]
        faces = m.mesh_face[begin:begin+m.mesh_facenum[mid]]
        mount = m.body('needle_mount').id
        rotation = d.xmat[mount].reshape(3, 3)
        vertices = (GeometryBounds(m, d).mesh_vertices(cap)-d.xpos[mount]) @ rotation
        cap_mesh = trimesh.Trimesh(vertices, faces, process=False)
        probes = np.array([[x,y,z] for x in (-.021,-.020,-.019)
                           for y in (-.020,0.,.020) for z in (.118,.120,.121)])
        assert cap_mesh.contains(probes).all(), 'The lower plate must join the retained silver guide'
        minimum_clearance = float('inf')
        samples = 0
        # Actual compiled bounds in the shared B tool frame, across the full
        # configured B/C joint ranges, with MuJoCo forward kinematics.
        for b in np.linspace(*m.joint('B').range, 5):
            for c in np.linspace(*m.joint('C').range, 9):
                d.qpos[m.joint('B').qposadr[0]] = b
                d.qpos[m.joint('C').qposadr[0]] = c
                mujoco.mj_forward(m, d)
                gid = guide.id
                origin = d.geom_xpos[gid]
                normal = d.geom_xmat[gid].reshape(3, 3)[:, 0]
                front = float(guide.size[0])
                from render_v9_three_views import GeometryBounds
                bounds = GeometryBounds(m, d)
                for i in range(m.ngeom):
                    if m.geom_bodyid[i] != m.body('needle_mount').id:
                        continue
                    lo, _ = bounds.interval(i, normal)
                    gap = lo - float(origin @ normal) - front
                    minimum_clearance = min(minimum_clearance, gap)
                    assert gap > .0005, (m.geom(i).name, gap)
                samples += 1
        return {'result': 'PASS', 'baseline_commit': 'ffd7b15',
                'outside_allowlist_exactly_unchanged': True,
                'exported_MJCF_outside_allowlist_exactly_unchanged': True,
                'unchanged_mesh_assets': len(current['mesh_hashes']),
                'body_mass_inertia_and_frames_unchanged': True,
                'registration_38mm_not_reapplied': True,
                'case_closed_positive_volume': True,
                'upper_case_vertices_preserved': True,
                'upper_plate_connection_internal_probes': len(probes),
                'B_C_full_range_samples': samples,
                'minimum_backplate_to_C_geometry_clearance_mm': minimum_clearance*1000,
                'scope': 'Display geometry and geometric clearance only; collisions remain disabled; no hardware calibration'}
    finally:
        scene.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if args.capture:
        s = RCMSceneV9()
        try:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            args.capture.write_text(json.dumps(snapshot(s), indent=2)+'\n')
        finally:
            s.close()
    else:
        result = verify(json.loads(args.baseline.read_text()))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
