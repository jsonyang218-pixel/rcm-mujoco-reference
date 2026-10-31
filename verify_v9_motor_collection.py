"""Reject motor dataset corruption and incorrect next-state action labels."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from rcm_reference import Reference
from rcm_waypoint_dataset import EpisodeWriter,write_manifest
from verify_v9_dataset import verify,UNITS


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False))


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))


def fixture(out):
    ref=Reference();ref.enable_motor_execution()
    state0=ref.current_task()*UNITS;robot0=list(ref.motor_robot())
    ref.set_task([.01,-.005,.001],motor_dt=.05)
    step=ref.motor_execution.last_record
    state1=ref.current_task()*UNITS
    image=np.zeros((1440,1920,3),dtype=np.uint8)
    writer=EpisodeWriter(out/'dataset',0,'靠近可见目标并停稳','visual_servo_motor',123,['P1'])
    writer.add(0.,state0,np.asarray(step['commanded_task_rad_rad_m'])*UNITS,'P1',image)
    writer.add(.05,state1,state1,'P1',image,terminal=True,phase='hold')
    episode=writer.finish('FAIL')
    episode.update(motor_steps='motor_steps.jsonl',frame_audit='frame_audit.jsonl',
        servo_report='report.json',motor_execution=ref.motor_execution.summary(),final_error_um=999.)
    manifest=write_manifest(out/'dataset',[episode],[1440,1920,3])
    manifest['execution_backend']='MotorBackend'
    write(out/'dataset/manifest.json',manifest)
    write(out/'config_snapshot.json',ref.config)
    write(out/'report.json',{'result':'FAIL'})
    jsonl(out/'motor_steps.jsonl',[step])
    audits=[]
    for i,(state,robot) in enumerate([(state0,robot0),(state1,list(ref.motor_robot()))]):
        audits.append({'frame_index':i,'timestamp_s':i*.05,'robot_ABCE':robot,
            'rgb_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'motor_transition_index':0 if i==0 else None})
    jsonl(out/'frame_audit.jsonl',audits)
    return out


class MotorDatasetTests(unittest.TestCase):
    def test_strict_claim_rejects_simulator_identity_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=fixture(Path(tmp));path=out/'dataset/manifest.json'
            manifest=json.loads(path.read_text());manifest['perception_backend']='strict_rgbd';write(path,manifest)
            write(out/'report.json',{'result':'FAIL','perception_backend':'strict_rgbd',
                'simulator_ids_used_for_control':True,'hidden_geometry_render_used_for_control':False,
                'final_oct_metadata':{'extra':{'rgb_and_depth_share_visibility':True,
                    'feature_pass_excludes_fundus_backdrop':False,'simulator_object_ids_used':False}}})
            with self.assertRaises(AssertionError):verify(out)

    def test_exact_replay_and_failed_navigation_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=verify(fixture(Path(tmp)))
            self.assertEqual(report['result'],'PASS')
            self.assertEqual(report['navigation_successes'],0)
            self.assertEqual(report['transitions_with_measured_motor_lag'],1)

    def test_actual_next_state_is_not_a_valid_action_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=fixture(Path(tmp));dataset=out/'dataset'
            path=dataset/'data/episode_000000.jsonl'
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            rows[0]['action']=rows[1]['observation.state'];jsonl(path,rows)
            actions=np.asarray([r['action'] for r in rows],dtype=np.float32)
            np.savez_compressed(dataset/'action_labels/episode_000000.npz',action=actions)
            full=dataset/'data/episode_000000.npz'
            with np.load(full,allow_pickle=False) as source:values={k:source[k] for k in source.files}
            values['action']=actions;np.savez_compressed(full,**values)
            with self.assertRaises(AssertionError):verify(out)

    def test_encoder_image_time_and_policy_corruptions_are_rejected(self):
        for kind in ('encoder','image','time','policy'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                out=fixture(Path(tmp))
                if kind=='policy':
                    p=out/'dataset/policy_inputs/episode_000000.npz'
                    with np.load(p,allow_pickle=False) as source:values={k:source[k] for k in source.files}
                    values['target_world_mm']=np.zeros((2,3));np.savez_compressed(p,**values)
                else:
                    p=out/'frame_audit.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                    if kind=='encoder':rows[1]['robot_ABCE'][0]+=100
                    elif kind=='image':rows[0]['rgb_pixels_sha256']='0'*64
                    else:rows[1]['timestamp_s']+=.01
                    jsonl(p,rows)
                with self.assertRaises(AssertionError):verify(out)


if __name__=='__main__':unittest.main(verbosity=2)
