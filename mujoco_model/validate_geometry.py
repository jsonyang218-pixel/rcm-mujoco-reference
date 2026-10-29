"""Independent source-coverage and visual ownership checks for the completed scan."""
from pathlib import Path
import json,hashlib
import numpy as np
import trimesh,mujoco
from scipy.spatial import cKDTree
from model_sync import DriverModel,HOME
HERE=Path(__file__).resolve().parent

def main():
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
    model=DriverModel();geom=mujoco.mj_name2id(model.model,mujoco.mjtObj.mjOBJ_GEOM,'scan_fixed_housing')
    fixed_pos=model.data.geom_xpos[geom].copy();fixed_rot=model.data.geom_xmat[geom].copy()
    samples=0;max_fk=0.
    for da in (-300,0,300):
        for db in (-700,0,700):
            for dc in (-900,0,900):
                robot=model.kin.predicted_robot_encoders(HOME,(da,db,dc,0));d=model.sync(robot)
                assert np.allclose(model.data.geom_xpos[geom],fixed_pos,atol=1e-12)
                assert np.allclose(model.data.geom_xmat[geom],fixed_rot,atol=1e-12)
                max_fk=max(max_fk,d['max_fk_error_m']);samples+=1
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        assert hashlib.sha256((HERE/'vendor/simulation_driver'/name).read_bytes()).hexdigest()==digest
    result={'status':'PASS','retained_visual_triangles':len(registered.faces),'part_faces':counts,
      'all_source_face_owners_accounted_for':True,'max_triangle_centroid_error_m':max_centroid_error,
      'max_source_point_error_m':max_point_error,'registration_is_rigid_without_scale':True,
      'fixed_body_pose_samples':samples,'max_fk_error_m':max_fk,'original_driver_files_verified':len(manifest['files_sha256']),
      'source_globally_watertight':False,'scope':'Complete coverage of simplified uploaded scan; not CAD or metrological calibration.'}
    (HERE/'outputs/geometry_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
