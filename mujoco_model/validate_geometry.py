"""Independent source-coverage and visual ownership checks for the completed scan."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import trimesh,mujoco
from scipy.spatial import cKDTree
from model_sync import DriverModel,HOME
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    args=parser.parse_args()
    registered=trimesh.load_mesh(HERE/'assets/registered_scan_world.obj',process=False)
    owner=np.load(HERE/'assets/segmentation.npz')['owner']
    assert len(owner)==len(registered.faces)==240000
    assert set(np.unique(owner))=={0,1,2,3}
    names=('fixed_housing','parallel_cover','distal_guide','C_carriage')
    origins=np.array([[0,0,0],[.266,.070,.325],[.456,.070,.125],[.456,.070,.125]])
    parts=[];counts={}
    for i,(name,origin) in enumerate(zip(names,origins)):
        part=trimesh.load_mesh(HERE/f'assets/{name}.obj',process=False)
        part.vertices+=origin;parts.append(part);counts[name]=len(part.faces)
        assert len(part.faces)==int((owner==i).sum())
    union=trimesh.util.concatenate(parts)
    assert len(union.faces)==len(registered.faces)
    before=registered.triangles_center;after=union.triangles_center
    max_centroid_error=float(max(cKDTree(before).query(after)[0].max(),cKDTree(after).query(before)[0].max()))
    assert max_centroid_error<2e-8
    before=registered.triangles.reshape(-1,3);after=union.triangles.reshape(-1,3)
    max_point_error=float(max(cKDTree(before).query(after)[0].max(),cKDTree(after).query(before)[0].max()))
    assert max_point_error<2e-8
    reg=json.loads((HERE/'registration.json').read_text());rotation=np.array(reg['R'])
    assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-9)
    assert abs(np.linalg.det(rotation)-1)<1e-9 and reg['scale']==1
    # Validate the current cap/white ownership directly against source indices;
    # no earlier model implementation or mesh snapshot is needed.
    cap=np.load(HERE/'assets/silver_cap_faces.npy')
    white=np.load(HERE/'assets/returned_shell_faces.npy')
    ownership=json.loads((HERE/'silver_cap_repair.json').read_text())
    assert len(cap)==ownership['reassigned_cap_faces'] and np.all(owner[cap]==1)
    assert len(white)==ownership['protected_white_shell_faces'] and np.all(owner[white]==0)
    assert not np.intersect1d(cap,white).size
    model=DriverModel();geom=mujoco.mj_name2id(model.model,mujoco.mjtObj.mjOBJ_GEOM,'scan_fixed_housing')
    a_body=model.model.body('A_stage').id
    assert model.model.geom_bodyid[geom]==a_body
    cover=model.model.geom('scan_parallel_cover')
    silver=model.model.body('silver_linear_stage').id
    assert int(cover.bodyid[0])==silver
    assert model.model.jnt_bodyid[model.model.joint('silver_B_linear').id]==silver
    rotation=model.data.xmat[a_body].reshape(3,3)
    fixed_pos=rotation.T@(model.data.geom_xpos[geom]-model.data.xpos[a_body])
    fixed_rot=rotation.T@model.data.geom_xmat[geom].reshape(3,3)
    samples=0;max_fk=0.
    for da in (-300,0,300):
        for db in (-700,0,700):
            for dc in (-900,0,900):
                robot=model.kin.predicted_robot_encoders(HOME,(da,db,dc,0));d=model.sync(robot)
                rotation=model.data.xmat[a_body].reshape(3,3)
                assert np.allclose(rotation.T@(model.data.geom_xpos[geom]-model.data.xpos[a_body]),fixed_pos,atol=1e-12,rtol=0)
                assert np.allclose(rotation.T@model.data.geom_xmat[geom].reshape(3,3),fixed_rot,atol=1e-12,rtol=0)
                max_fk=max(max_fk,d['max_fk_error_m']);samples+=1
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((HERE/'vendor/simulation_driver'/name).read_bytes()).hexdigest()==digest
    result={'status':'PASS','retained_visual_triangles':len(registered.faces),'part_faces':counts,
      'all_source_face_owners_accounted_for':True,'max_triangle_centroid_error_m':max_centroid_error,
      'max_source_point_error_m':max_point_error,'registration_is_rigid_without_scale':True,
      'shell_follows_A_only':True,'shell_pose_samples':samples,'max_fk_error_m':max_fk,'original_driver_files_verified':len(manifest['files_sha256']),
      'silver_cap_faces_on_single_slider':len(cap),'protected_white_faces_on_shell':len(white),
      'source_globally_watertight':False,'scope':'Complete coverage of simplified uploaded scan; not CAD or metrological calibration.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'geometry_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
