"""Export the installed model without changing encoder zero or mesh data."""
from pathlib import Path
import json
from model_sync import DriverModel
HERE=Path(__file__).resolve().parent

def main():
    for profile in ('new','legacy'):
        scene=DriverModel(needle_profile=profile)
        scene.export_xml(HERE/f'model_{profile}.xml')
        if profile=='new':
            scene.export_xml(HERE/'model.xml')
            values={'installation':scene.installation,'rotation':scene.install_R.tolist(),'translation_m':scene.install_t.tolist(),
                'rcm_world_m':scene.rcm_world.tolist(),'robot_to_world':scene.robot_to_world.tolist(),
                'joint_bounds_unchanged':scene.joint_bounds,'encoder_zero_changed':False,
                'mesh_geometry_changed':False,'visual_face_ownership_changed':True,'all_robot_parts_share_rigid_installation':True,
                'laboratory_planes_remain_horizontal':True,'photo_estimate_not_metrological_calibration':True}
            (HERE/'static_pose_parameters.json').write_text(json.dumps(values,indent=2))
            parameters=json.loads((HERE/'model_parameters.json').read_text())
            parameters.update(source_version=9,
                common_robot_to_world_rotation=scene.robot_to_world.tolist(),
                rcm_world_m=scene.rcm_world.tolist(),
                installation=scene.installation,
                installation_rotation=scene.install_R.tolist(),
                installation_translation_m=scene.install_t.tolist(),
                mesh_coordinate_frame='v6 installation reference; rigid transform applied by static_installation body',
                local_white_shell_repair=json.loads((HERE/'seam_repair.json').read_text()),
                silver_endcap_repair=json.loads((HERE/'silver_cap_repair.json').read_text()))
            (HERE/'model_parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
            registration=json.loads((HERE/'registration.json').read_text())
            old=registration['new_raw_mm_to_existing_world_m']
            import numpy as np
            registration['new_raw_mm_to_installed_world_m']={
                'formula':'world_m = A @ new_raw_mm + b_m',
                'A':(scene.install_R@np.array(old['A'])).tolist(),
                'b_m':(scene.install_R@np.array(old['b_m'])+scene.install_t).tolist(),
                'note':'The existing_world transform above is the uninstalled v6 reference frame.'}
            (HERE/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
            segmentation=json.loads((HERE/'segmentation_report.json').read_text())
            if 'registered_bounds_world_m' in segmentation:
                segmentation['registered_bounds_installation_reference_m']=segmentation.pop('registered_bounds_world_m')
            segmentation['mesh_frame']='uninstalled v6 reference; world transform is in installation_config.json and static_pose_parameters.json'
            (HERE/'segmentation_report.json').write_text(json.dumps(segmentation,indent=2)+'\n')
    print(f'Exported {scene.installation["pitch_y_deg"]:+g} degree photo-estimated correction with unchanged source joint coordinates.')

if __name__=='__main__':main()
