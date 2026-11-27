"""Check geometry-oracle numerical parity and axis transition indexing."""
import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from rcm.scene_v9 import RCMSceneV9
from rcm.visual_servo import VisualServoController, visual_servo_settings
from scripts.research.diagnose_step680_axes import UNIT
from scripts.research.motor_perception_gates import oracle_command, axis_metrics, resolve_training_target


class GeometryOracleTests(unittest.TestCase):
    def test_oracle_matches_public_geometry_teacher_equation(self):
        scene=RCMSceneV9()
        try:
            teacher=VisualServoController.__new__(VisualServoController)
            teacher.reference=scene
            teacher.settings=visual_servo_settings(scene)
            teacher.settings['position_tolerance_um']=.35
            teacher.measurement_goal=teacher.target_transform=None
            teacher.control_axes=teacher.completion_axes=np.ones(3,dtype=bool)
            q=np.array([-5.,-2.,10.])*UNIT
            scene.set_task(q)
            tip=scene.site_mm('needle_tip')
            for error in ([.03,-.02,.01],[.0001,0,0]):
                for velocity in (np.zeros(3),np.array([.03,-.01,.005])*UNIT):
                    target=tip+error
                    command_q=q+np.array([.001,-.002,.0003])*UNIT
                    # This is a numerical parity fixture, explicitly oracle
                    # truth. It is never passed to a perception estimator.
                    observation={'target_pixel':[0,0],'target_depth_m':1.,'tip_depth_m':1.,
                        'geometry':{'target_world_mm':target,'tip_world_mm':tip,
                                    'tip_quality':{'fit_quality_mm':0},
                                    'target_quality':{'fit_quality_mm':0}}}
                    expected=teacher.geometry_action_from_observation(q,observation,velocity,command_q)
                    actual,next_velocity,_=oracle_command(scene,q,tip,target,velocity,command_q)
                    self.assertTrue(expected['valid'])
                    np.testing.assert_allclose(actual,expected['candidate_task_si'],atol=1e-14,rtol=0)
                    np.testing.assert_allclose(next_velocity,expected['command_velocity_task_si'],atol=1e-14,rtol=0)
        finally:
            scene.close()

    def test_axis_window_uses_command_paired_with_its_transition(self):
        trajectory=[{'step':699,'command_delta_deg_deg_mm':[1,1,1],
                     'jacobian_column_um_per_deg_deg_mm':[10,20,1000]},
                    {'step':700,'command_delta_deg_deg_mm':[1,1,1],
                     'jacobian_column_um_per_deg_deg_mm':[10,20,1000]}]
        records=[{'actual_before_task_rad_rad_m':[0,0,0],
                  'actual_after_task_rad_rad_m':[0,0,0]} for _ in range(701)]
        records[700]['actual_after_task_rad_rad_m']=(np.array([0,.01,.001])*UNIT).tolist()
        rows=axis_metrics(trajectory,records)
        self.assertEqual(rows['A']['frames'],1)
        self.assertEqual(rows['A']['nonzero_command_no_motion_fraction'],1.)
        self.assertEqual(rows['B']['nonzero_command_no_motion_fraction'],0.)
        self.assertAlmostEqual(rows['C']['minimum_feedback_tip_step_um'],1.)

    def test_converted_raw_episode_id_does_not_override_merge_offset(self):
        with tempfile.TemporaryDirectory(prefix='rcm_target_ancestry_') as directory:
            root=Path(directory)
            merged=root/'merged';converted=root/'converted'
            merged.mkdir();(converted/'data').mkdir(parents=True)
            (merged/'metadata.json').write_text(json.dumps({
                'episodes':[{'num_frames':12,'source_dataset':str(converted),'source_episode_index':99}],
                'source_datasets':[{'directory':str(converted),'episode_offset':0,'episodes':1}]}))
            (converted/'metadata.json').write_text(json.dumps({'episodes':[
                {'num_frames':12,'source_episode_index':99,'terminal_audit_index':0}]}))
            (converted/'data/terminal_audits.json').write_text(json.dumps([
                {'frames':12,'terminal_target_world_m':[.1,.2,.3]}]))
            target=resolve_training_target(merged,0,{})
            self.assertEqual(target['frame'],'world_mm')
            self.assertEqual(target['point'],[100.,200.,300.])
            self.assertEqual(target['source_episode'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
