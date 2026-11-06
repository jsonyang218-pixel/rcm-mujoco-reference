"""Motor lag, protocol conversion, resets and feedback/render consistency."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from rcm_reference import Reference
from rcm_scene_v9 import RCMSceneV9

sys.path.insert(0, str(Path(__file__).parent/'mujoco_model'))
from motor_backend import MotorBackend


class MotorExecutionTests(unittest.TestCase):
    def make_reference(self):
        ref = Reference()
        ref.enable_motor_execution()
        return ref

    def test_protocol_target_conversion_is_not_encoder_subtraction(self):
        motor = MotorBackend(initial_handles=(147,-13,-30901,0))
        desired = (1200,1500000,950000,0)
        command = motor.command_for_robot_target(desired)
        expected = motor.kin.virtual_handles_from_robot_encoders(desired,motor.initial_robot)
        self.assertEqual(tuple(getattr(command,'handleEncoder'+a) for a in 'ABCE'),
                         tuple(a+b for a,b in zip(motor.initial_handles,expected)))
        receipt = motor.consume(command,0.)
        # Protocol quantization tolerances: < one handle on each axis.
        self.assertLess(abs(receipt['target'][0]-desired[0]),9)
        self.assertLess(abs(receipt['target'][2]-desired[2]),39)
        self.assertLess(abs(motor.kin.b_alpha_from_robot_encoder(receipt['target'][1])-
                            motor.kin.b_alpha_from_robot_encoder(desired[1])),.004)

    def test_lag_and_feedback_decoding_with_substeps(self):
        ref=self.make_reference();goal=np.array([.02,-.01,.001])
        ref.set_task(goal,motor_dt=.05)
        q=ref.current_task()
        self.assertTrue(np.all(np.abs(q)>0))
        self.assertTrue(np.all(np.abs(q)<np.abs(goal)))
        np.testing.assert_allclose(q,ref.task_from_encoders(ref.motor_robot()[:3]),atol=1e-12)
        self.assertEqual(ref.motor_execution.last_record['integration_steps'],25)
        self.assertAlmostEqual(ref.motor_execution.time_s,.05)
        for _ in range(150):ref.set_task(goal,motor_dt=.05)
        self.assertTrue(ref.motor_execution.settled())

    def test_invalid_commands_never_teleport_or_advance_clock(self):
        ref=self.make_reference()
        before=ref.data.qpos.copy();robot=ref.motor_robot()
        with self.assertRaises(ValueError):ref.set_task([100.,0.,0.])
        with self.assertRaises(ValueError):ref.set_task([0.,0.,0.],motor_dt=float('nan'))
        self.assertEqual(ref.motor_execution.time_s,0.)
        self.assertEqual(ref.motor_robot(),robot)
        np.testing.assert_array_equal(ref.data.qpos,before)

    def test_reset_clock_reanchors_and_allows_next_tick(self):
        ref=self.make_reference();ref.set_task([.01,0.,.0001],motor_dt=.05)
        held=ref.motor_robot();ref.reset_motor_time()
        self.assertEqual(ref.motor_execution.time_s,0.)
        self.assertEqual(ref.motor_execution.backend.target,held)
        ref.set_task([.02,.01,.0002],motor_dt=.05)
        self.assertAlmostEqual(ref.motor_execution.time_s,.05)
        self.assertNotEqual(ref.motor_robot(),held)

    def test_kinematic_queries_do_not_advance_motor(self):
        ref=self.make_reference()
        for _ in range(5):
            ref.encoders([.01,0.,.001]);ref.validate_task([.01,0.,.001])
            ref.forward_kinematics([.01,0.,.001])
        self.assertEqual(ref.motor_execution.time_s,0.)
        self.assertEqual(ref.motor_execution.records,[])

    def test_v9_has_one_motor_and_display_matches_feedback(self):
        ref=RCMSceneV9()
        try:
            ref.enable_motor_execution()
            ref.set_task([.02,-.01,.001],motor_dt=.05)
            self.assertIsNone(ref._ref.motor_execution)
            self.assertEqual(len(ref.motor_execution.records),1)
            self.assertEqual(tuple(ref.scene.last_robot),ref.motor_robot())
            np.testing.assert_allclose(ref.current_task(),ref.task_from_encoders(ref.motor_robot()[:3]),atol=1e-12)
            self.assertLess(ref.metrics()['fk_agreement_um'],.5)
            self.assertLess(ref.metrics()['rcm_drift_um'],1e-5)
        finally:ref.close()

    def test_outer_loop_accumulates_sub_handle_corrections_and_settles(self):
        import rcm_visual_servo as servo
        ref=Reference()
        ref.set_task(np.array([5.,-7.,18.])/np.array([180/np.pi,180/np.pi,1000.]))
        goal=np.array([4.97804035,-7.20088258,19.69507786])/np.array([180/np.pi,180/np.pi,1000.])
        ref.set_target_world(ref.robot_to_world(ref.forward_kinematics(goal)))
        ref.enable_motor_execution()
        class Sensor:
            camera_name='eye_internal';width=height=2
            def __init__(self,reference,settings):self.reference=reference;self.geometry=self
            def project(self,point):return np.asarray(point)[:2],float(point[2])*.001
            def observe(self):
                metrics=self.reference.metrics()
                tip,target=metrics['tip_world_mm'],metrics['target_world_mm']
                quality={'source':'independent_test_measurement','support':100,'residual_mm':0.,'fit_quality_mm':.001}
                return {'geometry':{'tip_world_mm':tip,'target_world_mm':target,'tip_quality':quality,'target_quality':quality},
                    'tip_pixel':tip[:2],'target_pixel':target[:2],
                    'tip_depth_m':tip[2]*.001,'target_depth_m':target[2]*.001,
                    'tip_pixels':100,'target_pixels':100,'rgb':np.zeros((2,2,3),dtype=np.uint8)}
            def close(self):pass
        settings=servo.visual_servo_settings(ref)
        settings.update(max_iterations=300,save_diagnostics=False)
        with patch.object(servo,'VisualServoSensor',Sensor):
            controller=servo.VisualServoController(ref,settings=settings)
            report=controller.run()
        self.assertEqual(report['result'],'PASS',report)
        self.assertTrue(report['motor_execution']['encoder_settled'])
        self.assertTrue(report['terminal_stop_validated'])
        self.assertGreater(len(ref.motor_execution.records),1)


if __name__=='__main__':unittest.main(verbosity=2)
