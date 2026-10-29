"""Check a static rigid installation change against the portable v6 baseline.

This validates coordinate consistency and preservation of relative robot motion,
not the physical accuracy of the photo-derived installation angle.
"""
from pathlib import Path
import importlib.util
import json
import hashlib
import shutil
import tempfile

import numpy as np
import mujoco

from model_sync import DriverModel, HOME
from driver_loader import get_message_types

HERE = Path(__file__).resolve().parent
V6_MESH_SHA256 = {
    'fixed_housing.obj': 'd916d87fef7f70542db529748748edc6561841b5894df831c1999c9c63f71d67',
    'parallel_cover.obj': '7384fa1facd9cfaaeee8b6232e08b816ab49a5f39ee178218e8f37ea0025a8fa',
    'distal_guide.obj': '7d0e16dfe66f687581476b1df9bb2f5fc27cd6c0b0c240368b92ec0cfee0fc11',
    'C_carriage.obj': 'a236d19e2bdd61efe356a215efecdec14b3a1b7d6db943ac326789d10c5c47a2',
}


def object_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f'Required object absent: {name}')
    return result


def main():
    snapshot = HERE / 'evidence/v6_model_sync.py'
    spec = importlib.util.spec_from_file_location('installation_baseline_v6', snapshot)
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    geom_names = ('scan_fixed_housing', 'scan_parallel_cover', 'scan_distal',
                  'scan_C', 'needle_holder', 'needle_shaft', 'needle_distal')
    site_names = ('rcm_site', 'needle_base', 'needle_tip', 'driver_junction',
                  'front_upper', 'front_lower')
    mesh_files = ('fixed_housing.obj', 'parallel_cover.obj',
                  'distal_guide.obj', 'C_carriage.obj')
    _, Encoder = get_message_types()
    profiles = []
    max_position = max_orientation = max_axis = max_feedback = max_q = 0.0
    with tempfile.TemporaryDirectory(prefix='v6-installation-baseline-') as temp:
        base_dir = Path(temp)
        shutil.copyfile(HERE / 'assets/v6_template.xml', base_dir / 'model.xml')
        shutil.copyfile(HERE / 'evidence/v6_silver_motion_config.json',
                        base_dir / 'silver_motion_config.json')
        shutil.copytree(HERE / 'assets', base_dir / 'assets')
        for profile in ('new', 'legacy'):
            before = baseline_module.DriverModel(base_dir, needle_profile=profile)
            after = DriverModel(needle_profile=profile)
            rotation = np.array(after.install_R)
            translation = np.array(after.install_t)
            assert rotation.shape == (3, 3) and translation.shape == (3,)
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(rotation) - 1) < 1e-12
            assert np.allclose(after.robot_to_world,
                               rotation @ before.robot_to_world, atol=1e-12)
            assert np.allclose(after.rcm_world,
                               rotation @ baseline_module.RCM + translation, atol=1e-12)
            assert after.model.nq == before.model.nq
            assert after.model.nu == before.model.nu == 3
            assert after.model.neq == before.model.neq
            assert np.array_equal(after.model.actuator_ctrlrange, before.model.actuator_ctrlrange)
            assert np.array_equal(after.model.actuator_gainprm, before.model.actuator_gainprm)
            assert np.array_equal(after.model.actuator_biasprm, before.model.actuator_biasprm)
            joint_names = [mujoco.mj_id2name(before.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                           for i in range(before.model.njnt)]
            samples = 0
            profile_position = profile_orientation = profile_axis = profile_feedback = 0.
            for da in (-300, 0, 300):
                for db in (-700, 0, 700):
                    for dc in (-900, 0, 900):
                        robot = after.kin.predicted_robot_encoders(HOME, (da, db, dc, 0))
                        before.sync(robot)
                        after.sync(robot)
                        for name in geom_names:
                            bi = object_id(before.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                            ai = object_id(after.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                            expected = rotation @ before.data.geom_xpos[bi] + translation
                            error = float(np.linalg.norm(after.data.geom_xpos[ai] - expected))
                            expected_R = rotation @ before.data.geom_xmat[bi].reshape(3, 3)
                            attitude = float(np.max(np.abs(after.data.geom_xmat[ai].reshape(3, 3) - expected_R)))
                            assert error < 1e-9 and attitude < 1e-10, (name, error, attitude)
                            profile_position = max(profile_position, error)
                            profile_orientation = max(profile_orientation, attitude)
                        for name in site_names:
                            bi = object_id(before.model, mujoco.mjtObj.mjOBJ_SITE, name)
                            ai = object_id(after.model, mujoco.mjtObj.mjOBJ_SITE, name)
                            error = float(np.linalg.norm(after.data.site_xpos[ai] -
                                          (rotation @ before.data.site_xpos[bi] + translation)))
                            assert error < 1e-9, (name, error)
                            profile_position = max(profile_position, error)
                        for name in joint_names:
                            bi = object_id(before.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                            ai = object_id(after.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                            q_error = abs(float(after.data.qpos[after.model.jnt_qposadr[ai]] -
                                                before.data.qpos[before.model.jnt_qposadr[bi]]))
                            axis_error = float(np.linalg.norm(after.data.xaxis[ai] -
                                                             rotation @ before.data.xaxis[bi]))
                            assert q_error < 1e-12 and axis_error < 1e-10, (name, q_error, axis_error)
                            assert np.array_equal(after.model.jnt_range[ai], before.model.jnt_range[bi])
                            max_q = max(max_q, q_error)
                            profile_axis = max(profile_axis, axis_error)
                        old_feedback = before.feedback_from_model(Encoder())
                        new_feedback = after.feedback_from_model(Encoder())
                        xyz_old = np.array([getattr(old_feedback, 'pos' + a) for a in 'XYZ'])
                        xyz_new = np.array([getattr(new_feedback, 'pos' + a) for a in 'XYZ'])
                        error = float(np.max(np.abs(xyz_new - xyz_old)))
                        assert error < 2e-6, error
                        profile_feedback = max(profile_feedback, error)
                        assert after.diagnostics['max_fk_error_m'] < 1e-9
                        samples += 1
            # World fixture/support orientations must not be pitched with robot.
            level_geoms = []
            for name in ('floor', 'support', 'phantom_pad'):
                bi = object_id(before.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                ai = object_id(after.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                assert np.allclose(after.data.geom_xmat[ai], before.data.geom_xmat[bi], atol=1e-12), name
                level_geoms.append(name)
            for i in range(after.model.ngeom):
                if after.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    normal = after.data.geom_xmat[i].reshape(3, 3)[:, 2]
                    assert np.allclose(normal, [0, 0, 1], atol=1e-12)
            mount = object_id(after.model, mujoco.mjtObj.mjOBJ_GEOM, 'horizontal_mount_plate')
            assert np.allclose(after.data.geom_xmat[mount].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-12)
            mount_top = float(after.data.geom_xpos[mount, 2] + after.model.geom_size[mount, 2])
            assert abs(mount_top - .10) < 1e-9
            for index in range(4):
                edge = object_id(after.model, mujoco.mjtObj.mjOBJ_GEOM, f'horizontal_reference_edge_{index}')
                assert abs(after.data.geom_xpos[edge, 2] - .1004) < 1e-9
                assert abs(after.data.geom_xmat[edge].reshape(3, 3)[2, 2]) < 1e-10
            profiles.append({'profile': profile, 'pose_samples': samples,
                'max_transformed_position_error_m': profile_position,
                'max_transformed_orientation_matrix_error': profile_orientation,
                'max_world_joint_axis_error': profile_axis,
                'max_driver_feedback_difference_mm': profile_feedback,
                'world_level_fixture_geoms': level_geoms,
                'horizontal_mount_plate_top_z_m': mount_top,
                'red_reference_edges_horizontal': True,
                'relative_joint_coordinates_and_ranges_unchanged': True,
                'silver_guide_rotates_with_installation': True})
            max_position = max(max_position, profile_position)
            max_orientation = max(max_orientation, profile_orientation)
            max_axis = max(max_axis, profile_axis)
            max_feedback = max(max_feedback, profile_feedback)
    hashes = {name: hashlib.sha256((HERE / 'assets' / name).read_bytes()).hexdigest()
              for name in mesh_files}
    baseline_hashes = {name: hashlib.sha256((HERE / 'evidence/v7_meshes' / name).read_bytes()).hexdigest()
                       for name in mesh_files}
    assert baseline_hashes == V6_MESH_SHA256, 'Archived original scan meshes changed'
    for name in ('distal_guide.obj', 'C_carriage.obj'):
        assert hashes[name] == baseline_hashes[name], name
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((HERE / 'vendor/simulation_driver' / name).read_bytes()).hexdigest() == digest
    result = {'status': 'PASS', 'scope': 'Rigid static installation adjustment only; photograph angle approximate',
        'profiles': profiles, 'max_transformed_position_error_m': max_position,
        'max_transformed_orientation_matrix_error': max_orientation,
        'max_world_joint_axis_error': max_axis, 'max_joint_q_difference': max_q,
        'max_driver_feedback_difference_mm': max_feedback,
        'mesh_sha256': hashes, 'scan_mesh_bytes_identical_to_v6': hashes == baseline_hashes,
        'archived_scan_mesh_bytes_identical_to_v6': True,
        'local_white_shell_ownership_change_checked_by': 'validate_seam.py',
        'actuator_parameters_identical_to_v6': True,
        'original_driver_files_verified': len(manifest['files_sha256']),
        'new_controlled_axes': 0, 'hardware_installation_calibrated': False}
    (HERE / 'outputs').mkdir(exist_ok=True)
    (HERE / 'outputs/installation_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
