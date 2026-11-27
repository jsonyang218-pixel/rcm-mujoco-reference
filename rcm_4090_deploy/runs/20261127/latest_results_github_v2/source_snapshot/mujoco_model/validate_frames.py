"""Verify one mechanical installation for both needle profiles and source limits.

These checks validate encoder/FK and model-frame consistency. They do not
identify the real installation or real motor dynamics.
"""
from pathlib import Path
import argparse
import itertools
import json
import math
import numpy as np
import mujoco
from model_sync import DriverModel, HOME

HERE=Path(__file__).resolve().parent

def sid(scene,name):
    value=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_SITE,name)
    if value<0:raise ValueError(f'Missing required site: {name}')
    return value

def bid(scene,name):
    value=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,name)
    if value<0:raise ValueError(f'Missing mechanical body: {name}')
    return value

def reject_atomically(scene,counts):
    before=scene.data.qpos.copy()
    try:
        scene.sync(counts)
    except (ValueError,TypeError,OverflowError) as exc:
        assert np.array_equal(scene.data.qpos,before),'Rejected encoder state mutated qpos'
        return str(exc)
    raise AssertionError(f'Invalid encoder state accepted: {counts}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'outputs')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    new=DriverModel(needle_profile='new');legacy=DriverModel(needle_profile='legacy')
    assert np.array_equal(new.robot_to_world,legacy.robot_to_world)
    assert new.c_direction==legacy.c_direction==1
    assert np.allclose(new.robot_to_world.T@new.robot_to_world,np.eye(3),atol=1e-14)
    assert math.isclose(np.linalg.det(new.robot_to_world),1,abs_tol=1e-14)
    assert np.array_equal(new.model.jnt_axis,legacy.model.jnt_axis)
    assert np.array_equal(new.model.jnt_range,legacy.model.jnt_range)
    assert np.array_equal(new.model.actuator_ctrlrange,legacy.model.actuator_ctrlrange)
    assert new.joint_bounds==legacy.joint_bounds
    for scene in (new,legacy):
        for joint in ('A','B','C'):
            j=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_JOINT,joint)
            assert np.allclose(scene.model.jnt_range[j],scene.joint_bounds[joint],atol=1e-14)
            actuator=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_ACTUATOR,joint+'_servo')
            assert np.allclose(scene.model.actuator_ctrlrange[actuator],scene.joint_bounds[joint],atol=1e-14)
    limits=new.kin.ROBOT_ENCODER_LIMITS
    levels=[(limits[a][0],HOME[i],limits[a][1]) for i,a in enumerate('ABC')]
    counts=[tuple(v)+(0,) for v in itertools.product(*levels)]
    rng=np.random.default_rng(20261028)
    counts += [tuple(int(rng.integers(limits[a][0],limits[a][1]+1)) for a in 'ABC')+(0,) for _ in range(64)]
    bodies=('A_stage','distal_orientation','C_translation','rear_crank','upper_bridge','lower_bridge')
    nbody=[bid(new,name) for name in bodies];lbody=[bid(legacy,name) for name in bodies]
    nj=sid(new,'driver_junction');lj=sid(legacy,'driver_junction')
    nt=sid(new,'needle_tip');lt=sid(legacy,'needle_tip')
    maximum_fk=0.;maximum_junction=0.;maximum_body=0.;maximum_tip_difference=0.;minimum_offset=1.;maximum_offset=0.
    for robot in counts:
        nr=new.sync(robot);lr=legacy.sync(robot)
        maximum_fk=max(maximum_fk,nr['max_fk_error_m'],lr['max_fk_error_m'])
        junction_error=float(np.linalg.norm(new.data.site_xpos[nj]-legacy.data.site_xpos[lj]))
        maximum_junction=max(maximum_junction,junction_error)
        assert junction_error<1e-12
        body_error=max(float(np.max(np.abs(new.data.xpos[nbody]-legacy.data.xpos[lbody]))),
                       float(np.max(np.abs(new.data.xmat[nbody]-legacy.data.xmat[lbody]))))
        maximum_body=max(maximum_body,body_error);assert body_error<1e-12
        tip_delta=legacy.data.site_xpos[lt]-new.data.site_xpos[nt]
        expected=.008*(new.robot_to_world@np.array(new.kin.needle_direction_from_robot_encoders(robot)))
        difference=float(np.linalg.norm(tip_delta-expected));maximum_tip_difference=max(maximum_tip_difference,difference)
        assert difference<1e-12
        length=float(np.linalg.norm(tip_delta));minimum_offset=min(minimum_offset,length);maximum_offset=max(maximum_offset,length)
        assert math.isclose(length,.008,abs_tol=1e-12)
    assert maximum_fk<1e-9
    rejected={}
    for i,axis in enumerate('ABC'):
        for edge,label in ((limits[axis][0]-1,'below_min'),(limits[axis][1]+1,'above_max')):
            robot=list(HOME);robot[i]=edge
            rejected[axis+'_'+label]=reject_atomically(new,robot)
    rejected['noninteger_count']=reject_atomically(new,(.5,HOME[1],HOME[2],0))
    rejected['bool_count']=reject_atomically(new,(True,HOME[1],HOME[2],0))
    result={'status':'PASS','scope':'Shared installation and exact source-encoder FK consistency, not hardware calibration',
       'profile_names':['new','legacy'],'poses_per_profile':len(counts),'source_limits_corners_tested':True,
       'installation_transform_identical':True,'mechanical_joint_axes_identical':True,
       'mechanical_joint_and_actuator_ranges_identical':True,
       'max_mujoco_vs_source_fk_error_m':maximum_fk,'max_profile_junction_difference_m':maximum_junction,
       'max_profile_mechanical_body_pose_difference':maximum_body,'max_signed_tip_delta_formula_error_m':maximum_tip_difference,
       'tip_profile_separation_mm':[minimum_offset*1000,maximum_offset*1000],
       'robot_to_world_shared':new.robot_to_world.tolist(),
       'source_encoder_bounds':limits,'joint_ranges_model_units':{k:list(v) for k,v in new.joint_bounds.items()},
       'rejected_without_state_mutation':rejected,
       'physical_installation_calibrated':False,'motor_dynamics_identified':False}
    path=args.output_dir/'frame_consistency_validation.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
