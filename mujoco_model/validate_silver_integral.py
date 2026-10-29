"""Verify that the scanned silver cap and cover move as one rigid body.

Regression is against the portable v8 assets. Source geometry is checked by
triangle multiplicities, separately from the deliberately changed ownership.
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
    q = np.rint(np.asarray(triangles) * 1e8).astype(np.int64)
    order = np.lexsort((q[:, :, 2], q[:, :, 1], q[:, :, 0]), axis=1)
    canonical = np.take_along_axis(q, order[:, :, None], axis=1).reshape(-1, 9)
    return Counter(row.tobytes() for row in canonical)


def ident(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    assert value >= 0, name
    return value


def main():
    cap_file = HERE / 'assets/silver_cap_faces.npy'
    report_file = HERE / 'silver_cap_repair.json'
    assert cap_file.exists() and report_file.exists(), 'Run silver-cap repair before validation'
    repair = json.loads(report_file.read_text())
    source = trimesh.load_mesh(HERE / 'assets/registered_scan_world.obj', process=False)
    old_owner = np.load(HERE / 'evidence/v8_segmentation.npz')['owner']
    new_owner = np.load(HERE / 'assets/segmentation.npz')['owner']
    cap = np.load(cap_file)
    white = np.load(HERE / 'assets/returned_shell_faces.npy')
    assert len(source.faces) == len(old_owner) == len(new_owner) == 240000
    assert set(np.unique(new_owner)) == {0, 1, 2, 3}
    assert cap.ndim == 1 and np.issubdtype(cap.dtype, np.integer)
    assert len(cap) > 0 and np.array_equal(cap, np.unique(cap))
    assert cap.min() >= 0 and cap.max() < len(source.faces)
    changed = np.flatnonzero(old_owner != new_owner)
    assert np.array_equal(changed, cap), 'Unexpected changes outside the declared silver cap'
    assert np.all(old_owner[cap] == 0) and np.all(new_owner[cap] == 1)
    assert len(white) == 537 and np.all(old_owner[white] == 0) and np.all(new_owner[white] == 0)
    assert not np.intersect1d(cap, white).size, 'White-shell fix was reverted'
    assert repair['reassigned_cap_faces'] == len(cap)
    assert repair['protected_white_shell_faces'] == len(white)
    part_faces = {}
    for i, (name, origin) in enumerate(zip(NAMES, ORIGINS)):
        part = trimesh.load_mesh(HERE / f'assets/{name}.obj', process=False)
        assert triangle_counts(part.triangles + origin) == triangle_counts(source.triangles[new_owner == i]), name
        part_faces[name] = len(part.faces)
    assert sum(part_faces.values()) == len(source.faces)
    for name in ('distal_guide', 'C_carriage'):
        current = digest(HERE / f'assets/{name}.obj')
        assert current == digest(HERE / f'evidence/v8_meshes/{name}.obj')
        assert current == digest(HERE / f'evidence/v7_meshes/{name}.obj')
    original_config = HERE / 'evidence/v7_installation_config.json'
    assert digest(HERE / 'installation_config.json') == digest(original_config)
    assert json.loads(original_config.read_text())['pitch_y_deg'] == -40
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        assert digest(HERE / 'vendor/simulation_driver' / name) == expected, name

    # The v7 snapshot uses the same seven-joint skeleton as v8; supply v8
    # meshes and settings in an isolated directory to reconstruct the baseline.
    spec = importlib.util.spec_from_file_location('silver_cap_v8_baseline', HERE / 'evidence/v7_model_sync.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_xml = ET.parse(HERE / 'evidence/v8_model.xml').getroot()
    original_install = original_xml.find(".//body[@name='static_installation']")
    _, Encoder = get_message_types()
    profiles = []
    max_body = max_axis = max_feedback = max_fk = max_rigidity_error = 0.0
    cap_points = source.triangles_center[cap][::max(1, len(cap) // 25)] - ORIGINS[1]
    silver_anchor = source.triangles_center[old_owner == 1].mean(axis=0) - ORIGINS[1]
    local_distances = np.linalg.norm(cap_points - silver_anchor, axis=1)
    with tempfile.TemporaryDirectory(prefix='silver-cap-v8-baseline-') as tmp:
        root = Path(tmp)
        (root / 'assets').mkdir()
        for name in NAMES:
            shutil.copyfile(HERE / f'evidence/v8_meshes/{name}.obj', root / f'assets/{name}.obj')
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
            installation = ident(after.model, mujoco.mjtObj.mjOBJ_BODY, 'static_installation')
            silver_body = ident(after.model, mujoco.mjtObj.mjOBJ_BODY, 'silver_linear_stage')
            silver_geom = ident(after.model, mujoco.mjtObj.mjOBJ_GEOM, 'scan_parallel_cover')
            fixed_geom = ident(after.model, mujoco.mjtObj.mjOBJ_GEOM, 'scan_fixed_housing')
            assert after.model.geom_bodyid[silver_geom] == silver_body
            # The white shell follows A only; the installation stays fixed.
            a_stage = ident(after.model, mujoco.mjtObj.mjOBJ_BODY, 'A_stage')
            assert after.model.geom_bodyid[fixed_geom] == a_stage
            slider = ident(after.model, mujoco.mjtObj.mjOBJ_JOINT, 'silver_B_linear')
            assert after.model.jnt_bodyid[slider] == silver_body
            assert after.model.jnt_type[slider] == mujoco.mjtJoint.mjJNT_SLIDE
            after_xml = ET.fromstring(after.xml)
            assert after_xml.find(".//body[@name='silver_linear_stage']/geom[@name='scan_parallel_cover']").get('mesh') == 'parallel_cover'
            fixed_position = after.data.xpos[installation].copy()
            fixed_rotation = after.data.xmat[installation].copy()
            observed_cap_positions = []
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
                assert np.array_equal(after.data.xpos[installation], fixed_position)
                assert np.array_equal(after.data.xmat[installation], fixed_rotation)
                rotation = after.data.xmat[silver_body].reshape(3, 3)
                translation = after.data.xpos[silver_body]
                world_points = cap_points @ rotation.T + translation
                world_anchor = rotation @ silver_anchor + translation
                rigidity = float(np.max(np.abs(np.linalg.norm(world_points - world_anchor, axis=1) - local_distances)))
                assert rigidity < 1e-12
                max_rigidity_error = max(max_rigidity_error, rigidity)
                observed_cap_positions.append(world_points.mean(axis=0))
                old_feedback = before.feedback_from_model(Encoder())
                new_feedback = after.feedback_from_model(Encoder())
                error = max(abs(getattr(new_feedback, 'pos' + a) - getattr(old_feedback, 'pos' + a)) for a in 'XYZ')
                assert error < 1e-7
                max_feedback = max(max_feedback, error)
                max_fk = max(max_fk, after.diagnostics['max_fk_error_m'])
                assert after.diagnostics['max_fk_error_m'] < 1e-9
            motion = float(np.linalg.norm(np.ptp(observed_cap_positions, axis=0)))
            assert motion > .001, 'Silver cap failed to follow the moving cover'
            profiles.append({'profile': profile, 'pose_samples': len(poses),
                             'source_limit_corners_included': True,
                             'cap_and_cover_share_silver_body': True,
                             'observed_cap_motion_range_m': motion})
    result = {'status': 'PASS', 'version': 9,
              'scope': 'Silver end cap and main cover share one visual rigid body; no hardware calibration',
              'source_faces_verified': len(source.faces), 'part_faces': part_faces,
              'source_face_multiplicities_preserved': True, 'source_faces_duplicated_by_repair': 0,
              'reassigned_silver_cap_faces': len(cap), 'only_declared_owner_changes': True,
              'previous_white_shell_537_faces_still_fixed': True,
              'distal_and_C_mesh_bytes_unchanged': True, 'installation_config_bytes_unchanged': True,
              'verified_original_driver_files': len(manifest['files_sha256']),
              'profiles': profiles, 'new_controlled_axes': 0,
              'joint_axes_limits_and_actuator_parameters_unchanged': True,
              'max_body_frame_difference': max_body, 'max_world_joint_axis_difference': max_axis,
              'max_feedback_difference_mm': max_feedback, 'max_driver_fk_error_m': max_fk,
              'max_cap_to_cover_anchor_distance_error_m': max_rigidity_error}
    (HERE / 'outputs').mkdir(exist_ok=True)
    (HERE / 'outputs/silver_integral_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
