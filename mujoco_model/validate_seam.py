"""Independent local-shell repair regression against the portable v7 snapshot.

Checks source-face multiplicities, ownership, fixed-shell motion and controller
coordinates. This is not a proof of globally watertight or calibrated geometry.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

from model_sync import DriverModel, HOME
from driver_loader import get_message_types

HERE = Path(__file__).resolve().parent
NAMES = ('fixed_housing', 'parallel_cover', 'distal_guide', 'C_carriage')
ORIGINS = np.array([[0, 0, 0], [.266, .070, .325],
                    [.456, .070, .125], [.456, .070, .125]])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triangle_counts(triangles):
    # OBJ has eight-decimal metre precision. Canonical vertex ordering makes
    # this insensitive to vertex renumbering, winding and face order.
    q = np.rint(np.asarray(triangles) * 1e8).astype(np.int64)
    order = np.lexsort((q[:, :, 2], q[:, :, 1], q[:, :, 0]), axis=1)
    canonical = np.take_along_axis(q, order[:, :, None], axis=1).reshape(-1, 9)
    return Counter(row.tobytes() for row in canonical)


def ident(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    assert value >= 0, name
    return value


def main():
    source = trimesh.load_mesh(HERE / 'assets/registered_scan_world.obj', process=False)
    old_owner = np.load(HERE / 'evidence/v7_segmentation.npz')['owner']
    new_owner = np.load(HERE / 'assets/segmentation.npz')['owner']
    returned = np.load(HERE / 'assets/returned_shell_faces.npy')
    assert len(source.faces) == len(old_owner) == len(new_owner) == 240000
    assert set(np.unique(new_owner)) == {0, 1, 2, 3}
    changed = np.flatnonzero((old_owner != new_owner) & (old_owner == 1))
    assert np.array_equal(changed, returned) and len(changed) == 537
    assert np.all(old_owner[changed] == 1) and np.all(new_owner[changed] == 0)
    cap = np.load(HERE / 'assets/silver_cap_faces.npy')
    expected_owner = old_owner.copy()
    expected_owner[returned] = 0
    assert np.all(expected_owner[cap] == 0)
    expected_owner[cap] = 1
    assert np.array_equal(expected_owner, new_owner)
    c = source.triangles_center
    distance = (c - [.351482, -.025692, .305064]) @ [-.22892878, .95569462, .18503894]
    expected_patch = ((old_owner == 1) & (c[:, 0] > .26) & (c[:, 0] < .33) &
                      (c[:, 2] > .25) & (c[:, 2] < .35) & (distance < -.0025))
    assert np.array_equal(changed, np.flatnonzero(expected_patch))
    part_faces = {}
    for i, (name, origin) in enumerate(zip(NAMES, ORIGINS)):
        part = trimesh.load_mesh(HERE / f'assets/{name}.obj', process=False)
        actual = triangle_counts(part.triangles + origin)
        expected = triangle_counts(source.triangles[new_owner == i])
        assert actual == expected, f'Source-face multiplicity mismatch: {name}'
        part_faces[name] = len(part.faces)
    assert sum(part_faces.values()) == 240000
    for name in ('distal_guide', 'C_carriage'):
        assert digest(HERE / f'assets/{name}.obj') == digest(HERE / f'evidence/v7_meshes/{name}.obj')
    config = HERE / 'installation_config.json'
    original_config = HERE / 'evidence/v7_installation_config.json'
    assert original_config.exists(), 'Pinned installation configuration snapshot missing'
    assert digest(config) == digest(original_config)
    assert json.loads(config.read_text())['pitch_y_deg'] == -40
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for name, value in manifest['files_sha256'].items():
        assert digest(HERE / 'vendor/simulation_driver' / name) == value, name

    spec = importlib.util.spec_from_file_location('seam_baseline_v7', HERE / 'evidence/v7_model_sync.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_xml = ET.parse(HERE / 'evidence/v7_model.xml').getroot()
    original_install = original_xml.find(".//body[@name='static_installation']")
    _, Encoder = get_message_types()
    profiles = []
    max_body = max_axis = max_feedback = max_fk = 0.0
    with tempfile.TemporaryDirectory(prefix='seam-v7-baseline-') as tmp:
        root = Path(tmp)
        (root / 'assets').mkdir()
        for name in NAMES:
            shutil.copyfile(HERE / f'evidence/v7_meshes/{name}.obj', root / f'assets/{name}.obj')
        shutil.copyfile(HERE / 'assets/v6_template.xml', root / 'assets/v6_template.xml')
        shutil.copyfile(original_config, root / 'installation_config.json')
        shutil.copyfile(HERE / 'evidence/v6_silver_motion_config.json', root / 'silver_motion_config.json')
        for profile in ('new', 'legacy'):
            before = module.DriverModel(root, needle_profile=profile)
            after = DriverModel(needle_profile=profile)
            current_install = ET.fromstring(after.xml).find(".//body[@name='static_installation']")
            assert original_install.attrib == current_install.attrib
            assert after.model.nq == before.model.nq == 7
            assert after.model.nu == before.model.nu == 3
            assert after.model.njnt == before.model.njnt == 7
            assert after.model.neq == before.model.neq == 4
            for field in ('actuator_ctrlrange', 'actuator_gainprm', 'actuator_biasprm',
                          'jnt_range', 'jnt_axis', 'jnt_type', 'eq_data'):
                assert np.array_equal(getattr(after.model, field), getattr(before.model, field)), field
            assert np.array_equal(after.robot_to_world, before.robot_to_world)
            assert np.array_equal(after.rcm_world, before.rcm_world)
            poses = [after.kin.predicted_robot_encoders(HOME, (a, b, c, 0))
                     for a, b, c in itertools.product((-300, 0, 300), (-700, 0, 700), (-900, 0, 900))]
            limits = after.kin.ROBOT_ENCODER_LIMITS
            poses.extend((*values, 0) for values in itertools.product(*(limits[a] for a in 'ABC')))
            fixed = ident(after.model, mujoco.mjtObj.mjOBJ_GEOM, 'scan_fixed_housing')
            fixed_body = int(after.model.geom_bodyid[fixed])
            installation = ident(after.model, mujoco.mjtObj.mjOBJ_BODY, 'static_installation')
            assert fixed_body == installation
            initial_fixed_position = after.data.xpos[fixed_body].copy()
            initial_fixed_rotation = after.data.xmat[fixed_body].copy()
            for robot in poses:
                before.sync(robot)
                after.sync(robot)
                assert np.array_equal(after.data.qpos, before.data.qpos)
                for i in range(before.model.nbody):
                    name = mujoco.mj_id2name(before.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    j = ident(after.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    error = max(float(np.max(np.abs(after.data.xpos[j] - before.data.xpos[i]))),
                                float(np.max(np.abs(after.data.xmat[j] - before.data.xmat[i]))))
                    assert error < 1e-12, (profile, name, error)
                    max_body = max(max_body, error)
                error = float(np.max(np.abs(after.data.xaxis - before.data.xaxis)))
                assert error < 1e-12
                max_axis = max(max_axis, error)
                assert np.array_equal(after.data.xpos[fixed_body], initial_fixed_position)
                assert np.array_equal(after.data.xmat[fixed_body], initial_fixed_rotation)
                old_feedback = before.feedback_from_model(Encoder())
                new_feedback = after.feedback_from_model(Encoder())
                error = max(abs(getattr(new_feedback, 'pos' + a) - getattr(old_feedback, 'pos' + a)) for a in 'XYZ')
                assert error < 1e-7
                max_feedback = max(max_feedback, error)
                max_fk = max(max_fk, after.diagnostics['max_fk_error_m'])
                assert after.diagnostics['max_fk_error_m'] < 1e-9
            profiles.append({'profile': profile, 'pose_samples': len(poses),
                             'source_limit_corners_included': True,
                             'fixed_shell_remains_on_installation': True})
            # Any optional future local backing must remain a static visual.
            for i in range(after.model.ngeom):
                name = mujoco.mj_id2name(after.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
                if name.startswith('seam_backing'):
                    assert after.model.geom_bodyid[i] == installation
                    assert after.model.geom_contype[i] == after.model.geom_conaffinity[i] == 0
    result = {'status': 'PASS', 'version': 9,
              'scope': 'Local visual shell ownership repair; not global watertightness or hardware calibration',
              'source_faces_verified': len(source.faces), 'part_faces': part_faces,
              'source_face_multiplicities_preserved': True, 'source_faces_duplicated_by_repair': 0,
              'reassigned_white_shell_faces': len(returned), 'only_declared_owner_changes': True,
              'also_reassigned_silver_cap_faces': len(cap),
              'distal_and_C_mesh_bytes_unchanged': True, 'installation_config_bytes_unchanged': True,
              'verified_original_driver_files': len(manifest['files_sha256']),
              'profiles': profiles, 'new_controlled_axes': 0,
              'joint_axes_limits_and_actuator_parameters_unchanged': True,
              'max_body_frame_difference': max_body, 'max_world_joint_axis_difference': max_axis,
              'max_feedback_difference_mm': max_feedback, 'max_driver_fk_error_m': max_fk,
              'white_shell_skin_fixed_across_all_test_poses': True}
    (HERE / 'outputs').mkdir(exist_ok=True)
    (HERE / 'outputs/seam_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
