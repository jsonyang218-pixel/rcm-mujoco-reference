#!/usr/bin/env python3
"""Needle/entry regressions against actual MuJoCo scene geometry."""
import copy
import json
from rcm_reference import HERE, Reference
from rcm_navigation import build_plan, execute_plan


def verify():
    reference = Reference()
    cases = {}
    for geom_name, radius_key in [('needle_proximal', 'proximal_radius_mm'),
                                  ('needle_distal', 'distal_radius_mm')]:
        assert abs(reference.model.geom(geom_name).size[0]*1000.-reference.config['needle'][radius_key]) < 1e-12
    assert abs(reference.model.geom('needle_tip_marker').size[0]*1000.-reference.tip_apex_radius_mm) < 1e-12
    length = reference.distal_mm
    offset = reference.tip_reference_mm
    assert abs(reference.model.geom('needle_distal').size[1]*2000.-(length-reference.distal_tip_mm)) < 1e-12
    tip_geom_id = int(reference.model.geom('needle_distal_tip').id)
    assert reference.model.geom('needle_distal_tip').type == 7
    mesh_id = int(reference.model.geom_dataid[tip_geom_id])
    mesh_start = int(reference.model.mesh_vertadr[mesh_id])
    mesh_count = int(reference.model.mesh_vertnum[mesh_id])
    mesh_z = reference.model.mesh_vert[mesh_start:mesh_start+mesh_count, 2]
    compiled_tip_z = reference.model.geom_pos[tip_geom_id, 2] + mesh_z
    tube_end_m = -(length-reference.distal_tip_mm)*.001
    assert abs(float(compiled_tip_z.max())-tube_end_m) < 1e-8
    assert abs(float(compiled_tip_z.min())-(-length*.001)) < 1e-8
    for label, c_mm in [('zero', 0.), ('C_2mm', 2.), ('tip_at_RCM', -offset),
                        ('junction_at_RCM', length-offset), ('proximal_at_RCM', length-offset+2.)]:
        reference.set_task([0., 0., c_mm*.001])
        geometry = reference.needle_eye_geometry()
        assert geometry['rcm_on_eye_surface']
        assert geometry['entry_rcm_distance_mm'] < 1e-9
        assert abs(geometry['junction_rcm_axial_offset_mm']-(offset+c_mm-length)) < 1e-9
        assert abs(geometry['distal_centerline_inside_mm']-min(length, max(0., offset+c_mm))) < 1e-9
        assert abs(geometry['proximal_centerline_inside_mm']-max(0., offset+c_mm-length)) < 1e-9
        cases[label] = geometry
    assert cases['junction_at_RCM']['entry_needle_part'] == 'junction'
    assert cases['proximal_at_RCM']['entry_needle_part'] == 'proximal'
    assert abs(cases['proximal_at_RCM']['entry_needle_diameter_mm']-2*reference.config['needle']['proximal_radius_mm']) < 1e-12
    reference.set_task(reference.home)
    target = reference.config['reference_scene']['original_deep_target_robot_mm']
    plan = build_plan(reference, [{'id':'deep_target','frame':'robot','point_mm':target}])
    recorder = execute_plan(reference, plan)
    report = recorder.report()
    assert report['result'] == 'PASS'
    deep = report['target_results'][0]['needle_eye_geometry']
    assert abs(deep['distal_centerline_inside_mm']-length) < 1e-8
    assert abs(deep['proximal_centerline_inside_mm']-(18.668147567212985-length)) < 1e-8
    assert deep['entry_needle_part'] == 'proximal'
    assert deep['entry_rcm_distance_mm'] < 1e-8
    cases['deep_target'] = deep
    changed_config = copy.deepcopy(reference.config)
    changed_config['reference_scene']['eye_center_robot_mm'][0] += 1.
    changed = Reference(changed_config)
    shifted = changed.needle_eye_geometry()
    assert not shifted['rcm_on_eye_surface']
    assert shifted['entry_rcm_distance_mm'] > .001
    cases['synthetic_misaligned_eye'] = shifted
    changed_config = copy.deepcopy(reference.config)
    changed_config['installation']['euler_xyz_deg'] = [23., -17., 31.]
    changed_config['installation']['rcm_world_mm'] = [31., -8., 14.]
    changed = Reference(changed_config)
    changed.set_task([0.,0.,.002])
    transformed = changed.needle_eye_geometry()
    for name in ('distal_centerline_inside_mm','proximal_centerline_inside_mm','entry_rcm_distance_mm'):
        assert abs(transformed[name]-cases['C_2mm'][name]) < 1e-9
    recorder.write(HERE/'runs/历史记录/04_sessions/checks/needle_geometry_deep')
    result = {'result':'PASS','scope':'reference needle centerline and eye ellipsoid; no aperture or tissue contact validation',
              'cases':cases,'rigid_installation_invariance':'PASS','position_limits_enabled':False}
    (HERE/'needle_geometry_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
