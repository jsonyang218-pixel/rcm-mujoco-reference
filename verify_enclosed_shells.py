"""Verify closed exported skins, source preservation, and needle articulation."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'mujoco_model'))
from model_sync import DriverModel, HOME


def main():
    folder = HERE / 'mujoco_model'
    report = json.loads((folder / 'enclosed_shells_report.json').read_text())
    result = {}
    for name, info in report['parts'].items():
        source = folder / 'assets' / info['source']
        output = folder / 'assets' / info['output']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == info['source_sha256']
        assert hashlib.sha256(output.read_bytes()).hexdigest() == info['output_sha256']
        mesh = trimesh.load_mesh(output, process=False)
        _, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
        assert np.all(counts == 2), name
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume, name
        result[name] = {'boundary_edges': int(sum(counts == 1)),
                        'nonmanifold_edges': int(sum(counts > 2)),
                        'watertight': True, 'faces': len(mesh.faces)}
    scene = DriverModel()
    root = ET.fromstring(scene.xml)
    assert scene.model.nq == 7 and scene.model.nu == 3 and scene.model.neq == 4
    for name, info in report['parts'].items():
        assert root.find(f"./asset/mesh[@name='{name}']").get('file') == info['output']
    assert root.find(".//geom[@name='scan_C']").get('group', '0') == '0'
    assert root.find(".//geom[@name='scan_distal']").get('group', '0') == '0'
    assert not root.findall(".//geom[@name='front_guard']")
    needle = root.find(".//geom[@name='needle_distal']")
    assert needle.get('group') == '2' and needle.get('size') == '3e-05'
    shell = root.find(".//body[@name='A_stage']/geom[@name='scan_fixed_housing']")
    assert shell is not None
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, 'scan_fixed_housing')
    home_position = scene.data.geom_xpos[geom].copy()
    home_rotation = scene.data.geom_xmat[geom].copy()
    for axis in ('A', 'B', 'C'):
        scene.sync(HOME)
        joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, axis)
        scene.data.qpos[scene.model.jnt_qposadr[joint]] += .01
        mujoco.mj_forward(scene.model, scene.data)
        if axis == 'A':
            assert not np.allclose(scene.data.geom_xmat[geom], home_rotation)
        else:
            assert np.allclose(scene.data.geom_xpos[geom], home_position)
            assert np.allclose(scene.data.geom_xmat[geom], home_rotation)
    print(json.dumps({'status': 'PASS', 'parts': result,
                      'source_meshes_unchanged': True,
                      'needle_preserved': True, 'shell_follows_A_only': True,
                      'scope': 'Visual topology and articulation; not collision or hardware validation'},
                     indent=2))


if __name__ == '__main__':
    main()
