"""Verify raw image/state/action alignment by replaying the motor protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from rcm_reference import Reference

sys.path.insert(0,str(Path(__file__).parent/'mujoco_model'))
from motor_backend import MotorBackend

UNITS=np.array([180/np.pi,180/np.pi,1000.])


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def inside(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Dataset path escapes root')
    return path


def verify(directory):
    out=Path(directory).resolve();dataset=out/'dataset'
    manifest=json.loads((dataset/'manifest.json').read_text())
    if manifest.get('execution_backend')!='MotorBackend':raise ValueError('Not a v9 motor dataset')
    config=json.loads((out/'config_snapshot.json').read_text())
    config['motor_execution']={'enabled':False}
    decoder=Reference(config)
    total=lag_frames=successes=0;episode_results=[]
    period=float(manifest['nominal_sample_period_s'])
    for episode in manifest['episodes']:
        rows=read_rows(inside(dataset,episode['jsonl']))
        audits=read_rows(inside(out,episode['frame_audit']))
        steps=read_rows(inside(out,episode['motor_steps']))
        assert len(rows)==len(audits)==episode['frames'] and len(rows)>0
        assert len(steps)==len(rows)-1
        assert rows[-1]['is_terminal'] and not any(r['is_terminal'] for r in rows[:-1])
        settings=episode['motor_execution']
        motor=MotorBackend(initial_robot=tuple(audits[0]['robot_ABCE']),needle_profile='legacy',
            tau_s=settings['tau_s'],timeout_s=settings['timeout_s'],
            speed_limits=settings['speed_limits_A_deg_B_deg_C_mm_per_s'])
        with np.load(inside(dataset,episode['policy_inputs']),allow_pickle=False) as policy:
            assert set(policy.files)=={'observation.images.microscope_rgb','observation.state','task'}
            assert len(policy['task'])==len(rows)
            policy_states=policy['observation.state'].copy()
            policy_images=policy['observation.images.microscope_rgb'].copy()
            policy_tasks=policy['task'].copy()
        with np.load(inside(dataset,episode['policy_inputs_no_image']),allow_pickle=False) as no_image:
            assert set(no_image.files)=={'observation.state','task'}
            np.testing.assert_array_equal(no_image['observation.state'],policy_states)
            np.testing.assert_array_equal(no_image['task'],policy_tasks)
        with np.load(inside(dataset,episode['action_labels']),allow_pickle=False) as labels:
            assert labels.files==['action'];actions=labels['action'].copy()
        with np.load(inside(dataset,episode['data']),allow_pickle=False) as full:
            np.testing.assert_array_equal(full['observation.state'],policy_states)
            np.testing.assert_array_equal(full['action'],actions)
            np.testing.assert_array_equal(full['observation.images.microscope_rgb'],policy_images)
            np.testing.assert_allclose(full['timestamp'],[r['timestamp'] for r in rows],atol=1e-10,rtol=0)
        max_velocity=np.zeros(3);max_acceleration=np.zeros(3);previous_velocity=np.zeros(3)
        for i,(row,audit) in enumerate(zip(rows,audits)):
            assert row['frame_index']==audit['frame_index']==i
            assert np.isclose(row['timestamp'],audit['timestamp_s'],atol=1e-10,rtol=0)
            assert np.isclose(row['timestamp'],i*period,atol=1e-8,rtol=0)
            state=np.asarray(row['observation.state'])
            np.testing.assert_allclose(policy_states[i],state,atol=1e-7,rtol=0)
            np.testing.assert_array_equal(actions[i],np.asarray(row['action'],dtype=np.float32))
            assert policy_images[i]==row['observation.images.microscope_rgb']
            assert policy_tasks[i]==row['task']
            np.testing.assert_allclose(decoder.task_from_encoders(audit['robot_ABCE'][:3],reference_task=state/UNITS)*UNITS,state,atol=2e-6,rtol=0)
            image=np.asarray(Image.open(inside(dataset,row['observation.images.microscope_rgb'])).convert('RGB'))
            assert list(image.shape)==[1440,1920,3]
            assert hashlib.sha256(image.tobytes()).hexdigest()==audit['rgb_pixels_sha256']
            assert tuple(motor.robot)==tuple(audit['robot_ABCE'])
            if i<len(steps):
                step=steps[i]
                assert audit['motor_transition_index']==i==step['step_index']
                assert np.isclose(step['start_time_s'],row['timestamp'],atol=1e-8,rtol=0)
                assert np.isclose(step['end_time_s'],rows[i+1]['timestamp'],atol=1e-8,rtol=0)
                assert step['actual_before_robot_ABCE']==list(motor.robot)
                command=motor.kin.make_control_message(step['handles_ABCE'],'hold')
                receipt=motor.consume(command,step['start_time_s'])
                assert list(receipt['target'])==step['quantized_target_robot_ABCE']
                commanded=decoder.task_from_encoders(receipt['target'][:3],reference_task=state/UNITS)*UNITS
                np.testing.assert_allclose(row['action'],commanded,atol=2e-6,rtol=0)
                n=step['integration_steps'];dt=step['duration_s']/n
                assert 0<dt<=settings['integration_step_s']+1e-12
                for substep in range(n):motor.step(dt,step['start_time_s']+(substep+1)*dt)
                assert step['actual_after_robot_ABCE']==list(motor.robot)==audits[i+1]['robot_ABCE']
                assert not step['fault']
                next_state=np.asarray(rows[i+1]['observation.state'])
                velocity=(next_state-state)/period
                max_velocity=np.maximum(max_velocity,np.abs(velocity))
                max_acceleration=np.maximum(max_acceleration,np.abs(velocity-previous_velocity)/period)
                previous_velocity=velocity
                lag_frames+=int(np.linalg.norm(next_state-np.asarray(row['action']))>1e-5)
            else:
                assert audit['motor_transition_index'] is None
                np.testing.assert_array_equal(row['action'],row['observation.state'])
        report=json.loads(inside(out,episode['servo_report']).read_text())
        assert report['result']==episode['result']
        if manifest.get('perception_backend')=='strict_rgbd':
            assert report['perception_backend']=='strict_rgbd'
            assert report['simulator_ids_used_for_control'] is False
            assert report['hidden_geometry_render_used_for_control'] is False
            metadata=report['final_oct_metadata']
            assert metadata['extra']['rgb_and_depth_share_visibility'] is True
            assert metadata['extra']['feature_pass_excludes_fundus_backdrop'] is False
            assert metadata['extra']['simulator_object_ids_used'] is False
        assert len(steps)==settings['control_steps']
        assert np.isclose(settings['simulation_time_s'],rows[-1]['timestamp'],atol=1e-8,rtol=0)
        if episode['result']=='PASS':
            assert report['result']=='PASS' and report['terminal_stop_validated']
            assert report['motor_execution']['encoder_settled']
            assert report['final_tip_target_error_um']<20.
            assert motor.robot==motor.target
            successes+=1
        total+=len(rows)
        episode_results.append({'episode_index':episode['episode_index'],'frames':len(rows),
            'navigation_result':episode['result'],'final_error_um':episode['final_error_um'],
            'max_actual_velocity_A_deg_B_deg_C_mm_per_s':max_velocity.tolist(),
            'max_actual_acceleration_A_deg_B_deg_C_mm_per_s2':max_acceleration.tolist(),
            'actual_acceleration_is_explicitly_limited_by_motor_model':False,
            'actual_velocity_within_nominal_controller_limits':bool(np.all(max_velocity<=np.asarray(config['navigation']['max_velocity_A_deg_B_deg_C_mm_per_s'])+1e-6)),
            'actual_acceleration_within_nominal_controller_limits':bool(np.all(max_acceleration<=np.asarray(config['navigation']['max_acceleration_A_deg_B_deg_C_mm_per_s2'])+1e-6))})
    assert total==manifest['total_frames']
    return {'result':'PASS','scope':'Dataset alignment and exact offline motor replay; not physical motor validation',
        'perception_backend':manifest.get('perception_backend','historical_simulation_ids'),
        'episodes':len(episode_results),'navigation_successes':successes,'total_frames':total,
        'transitions_with_measured_motor_lag':lag_frames,
        'raw_rgb_hashes_match':True,'state_matches_actual_encoders':True,
        'actions_are_protocol_targets_not_next_states':True,'policy_inputs_exclude_audit_fields':True,
        'episode_results':episode_results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.run_dir),ensure_ascii=False,indent=2))
