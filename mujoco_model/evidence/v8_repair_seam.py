"""Local scan ownership repair; keep every source triangle and the v7 pose."""
from pathlib import Path
import json
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
ORIGIN=np.array([.266,.070,.325])

def main():
    mesh=trimesh.load_mesh(HERE/'assets/registered_scan_world.obj',process=False)
    data=np.load(HERE/'evidence/v7_segmentation.npz')
    owner=data['owner'].copy()
    centre=mesh.triangles_center
    plane_centre=np.array([.351482,-.025692,.305064])
    plane_normal=np.array([-.22892878,.95569462,.18503894])
    distance=(centre-plane_centre)@plane_normal
    patch=((owner==1)&(centre[:,0]>.26)&(centre[:,0]<.33)&
        (centre[:,2]>.25)&(centre[:,2]<.35)&(distance<-.0025))
    assert patch.any()
    owner[patch]=0
    counts={}
    for i,(name,origin) in enumerate((('fixed_housing',np.zeros(3)),('parallel_cover',ORIGIN))):
        part=mesh.submesh([np.flatnonzero(owner==i)],append=True,repair=False)
        part.vertices-=origin
        part.export(HERE/f'assets/{name}.obj')
        counts[name]=len(part.faces)
    np.save(HERE/'assets/returned_shell_faces.npy',np.flatnonzero(patch))
    np.savez_compressed(HERE/'assets/segmentation.npz',owner=owner,reference_distance_m=data['reference_distance_m'])
    report={'version':8,'scope':'Local white shell skin reassigned from moving silver cover to fixed shell',
        'returned_source_faces':int(patch.sum()),'returned_bounds_reference_m':mesh.triangles[patch].reshape(-1,3).min(axis=0).tolist()+mesh.triangles[patch].reshape(-1,3).max(axis=0).tolist(),
        'original_triangles_retained':True,'source_triangle_count':len(mesh.faces),
        'joint_topology_changed':False,'installation_pitch_changed':False,
        'source_mesh_has_open_boundaries':True,'physical_transmission_calibrated':False}
    (HERE/'seam_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    seg=json.loads((HERE/'segmentation_report.json').read_text())
    seg['part_faces'].update(counts);seg['local_ownership_repair']=report
    (HERE/'segmentation_report.json').write_text(json.dumps(seg,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
