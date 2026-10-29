"""Reunite the silver cover endcap, preserving the v8 white-shell repair."""
from pathlib import Path
import json
import numpy as np
import trimesh

HERE=Path(__file__).resolve().parent
ORIGIN=np.array([.266,.070,.325])

def main():
    source=trimesh.load_mesh(HERE/'assets/registered_scan_world.obj',process=False)
    original=np.load(HERE/'evidence/v8_segmentation.npz')
    owner=original['owner'].copy()
    definition=json.loads((HERE/'silver_cap_definition.json').read_text())
    uvn=(source.triangles_center-np.array(definition['origin_m']))@np.array(definition['basis_columns_uvn'])
    u,v,n=uvn.T
    main=((owner==0)&(u>-.106)&(u<-.080)&(v>.005)&(v<.036)&(n>-.0015)&(n<.0715))
    normal_u=np.abs(source.face_normals@np.array(definition['basis_columns_uvn'])[:,0])
    # The endcap's lower rim is a vertical continuation of its end plane.
    # Exclude the white roof by its normal, rather than slicing it by height.
    bottom=((owner==0)&(u>-.0925)&(u<-.088)&(v>-.0015)&(v<.005)&
        (n>-.0015)&(n<.0715)&(normal_u>.55))
    cap=main|bottom
    white=np.load(HERE/'assets/returned_shell_faces.npy')
    cap[white]=False
    assert main.sum()==definition['count']
    assert cap.sum()==1797
    assert np.all(owner[white]==0)
    owner[cap]=1
    counts={}
    for i,(name,origin) in enumerate((('fixed_housing',np.zeros(3)),('parallel_cover',ORIGIN))):
        part=source.submesh([np.flatnonzero(owner==i)],append=True,repair=False)
        part.vertices-=origin
        part.export(HERE/f'assets/{name}.obj')
        counts[name]=len(part.faces)
    np.save(HERE/'assets/silver_cap_faces.npy',np.flatnonzero(cap))
    np.savez_compressed(HERE/'assets/segmentation.npz',owner=owner,reference_distance_m=original['reference_distance_m'])
    report={'version':9,'scope':'Round silver endcap and long cover share the same rigid slider body',
        'reassigned_cap_faces':int(cap.sum()),'main_cap_faces':int(main.sum()),'bottom_rim_faces':int(bottom.sum()),
        'bottom_rim_rule':'-.0925<u<-.088; -.0015<v<.005; -.0015<n<.0715; abs(normal dot u)>.55; exclude protected white faces',
        'direction':'fixed_housing -> parallel_cover',
        'protected_white_shell_faces':len(white),'white_shell_repair_preserved':True,
        'source_triangle_count':len(source.faces),'all_source_triangle_positions_preserved':True,
        'new_geometry_added':False,'new_joints_added':False,'installation_changed':False,
        'driver_changed':False,'source_scan_globally_watertight':False,
        'video_calibrated_silver_travel':False,'definition_file':'silver_cap_definition.json'}
    (HERE/'silver_cap_repair.json').write_text(json.dumps(report,indent=2)+'\n')
    segmentation=json.loads((HERE/'segmentation_report.json').read_text())
    segmentation['part_faces'].update(counts)
    segmentation['silver_cap_ownership_repair']=report
    (HERE/'segmentation_report.json').write_text(json.dumps(segmentation,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
