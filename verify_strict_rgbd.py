"""Current-frame RGB/visible-depth checks without simulator identity feedback.

The four pose checks use simulation truth only for independent evaluation.
Frozen-frame checks give the estimator fixed calibration plus arrays, while
all model/state/target truth APIs are deliberately unavailable. The benchmark
uses visible orange/green artificial fiducials; this is not lesion detection
or a validation of real OCT accuracy.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from rcm_scene_v9 import RCMSceneV9
from rcm_strict_rgbd import StrictRGBDSensor
from rcm_visual_servo import VisualServoController, visual_servo_settings


def forbidden(*args, **kwargs):
    raise AssertionError('Strict perception consulted simulator identity or dynamic truth')


class ForbiddenTruth:
    def __getattr__(self, name):
        forbidden()

    def __getitem__(self, key):
        forbidden()


class FrozenCalibration:
    """Only camera/RCM/needle calibration is exposed, with no target pose."""
    def __init__(self, reference):
        self.eye_camera=copy.deepcopy(reference.eye_camera)
        self.config={'needle':copy.deepcopy(reference.config['needle'])}
        self.rcm_mm=reference.rcm_mm.copy()
        self._pose=tuple(value.copy() for value in reference.camera_pose('eye_internal'))
        self.model=self.data=self.sites=self.nominal=ForbiddenTruth()
        self.metrics=self.site_mm=self.current_task=self.target_task=forbidden

    def camera_pose(self, name):
        if name!='eye_internal':
            raise AssertionError('Unexpected camera')
        return tuple(value.copy() for value in self._pose)


class StrictRGBDTests(unittest.TestCase):
    pose_results=[]

    @classmethod
    def setUpClass(cls):
        cls.reference=RCMSceneV9()
        cls.settings=visual_servo_settings(cls.reference)
        cls.settings.update(perception_backend='strict_rgbd',position_feedback='geometry',
                            occlusion_handling='stop',save_diagnostics=False)
        cls.calibration=FrozenCalibration(cls.reference)
        cls.frames={}
        cls.pose_results=[]
        sensor=StrictRGBDSensor(cls.reference,cls.settings)
        try:
            for i,(name,pose) in enumerate([
                    ('home',[0.,0.,0.]),('C8',[0.,0.,8.]),
                    ('near',[5.,-7.,18.]),('goal',[4.97804,-7.20088,19.695])]):
                cls.reference.set_task(np.asarray(pose)/[180/np.pi,180/np.pi,1000.])
                sensor.observation_time_s=i*.05
                observation=sensor.observe()
                truth=cls.reference.metrics()
                cls.frames[name]={'rgb':observation['rgb'].copy(),
                    'depth':observation['depth'].copy(),
                    'truth_tip':np.asarray(truth['tip_world_mm']),
                    'truth_target':np.asarray(truth['target_world_mm'])}
        finally:
            sensor.close()

    @classmethod
    def tearDownClass(cls):
        cls.reference.close()
        cls.frames.clear()

    def setUp(self):
        # Supplying a provider suppresses all simulator renderer construction.
        # These tests call measure() with frozen registered arrays instead.
        self.sensor=StrictRGBDSensor(self.calibration,self.settings,frame_provider=object())
        self.timestamp=0.

    def tearDown(self):
        self.sensor.close()

    def measure(self, name='home', rgb=None, depth=None):
        frame=self.frames[name]
        result=self.sensor.measure(frame['rgb'] if rgb is None else rgb,
            frame['depth'] if depth is None else depth,self.timestamp,
            {'source':'frozen_visible_simulation_fixture','is_simulation':True})
        self.timestamp+=.05
        return result

    def test_representative_pose_accuracy_without_ids(self):
        results=[]
        for name,frame in self.frames.items():
            with self.subTest(pose=name):
                observation=self.measure(name)
                geometry=observation['geometry']
                self.assertIsNotNone(geometry['tip_world_mm'],geometry)
                self.assertIsNotNone(geometry['target_world_mm'],geometry)
                tip_error=float(np.linalg.norm(np.asarray(geometry['tip_world_mm'])-frame['truth_tip'])*1000.)
                target_error=float(np.linalg.norm(np.asarray(geometry['target_world_mm'])-frame['truth_target'])*1000.)
                self.assertLess(tip_error,10.)
                self.assertLess(target_error,5.)
                self.assertIsNone(observation['segmentation'])
                self.assertFalse(observation['simulator_ids_used_for_control'])
                self.assertFalse(observation['hidden_geometry_render_used_for_control'])
                results.append({'pose':name,'tip_error_um':tip_error,
                    'target_error_um':target_error,
                    'tip_source':geometry['tip_quality']['source'],
                    'apex_rgb_pixels':int(observation['feature_masks']['apex'].sum())})
        type(self).pose_results=results

    def test_frozen_arrays_suffice_with_all_dynamic_truth_forbidden(self):
        observation=self.measure('near')
        self.assertNotIn('reference_scene',self.calibration.config)
        self.assertEqual(self.sensor.renderers,{})
        self.assertIsNone(self.sensor.geometry.proximal_id)
        self.assertIsNone(self.sensor.geometry.target_id)
        # The same estimator must produce the same result from the same frame,
        # despite having no live model, target inverse, qpos or site access.
        again=self.measure('near')
        for key in ('tip_world_mm','target_world_mm'):
            np.testing.assert_allclose(observation['geometry'][key],again['geometry'][key],atol=1e-9,rtol=0)

    def test_segmentation_field_is_unused_even_when_poisoned(self):
        observation=self.measure('near')
        expected_tip=np.asarray(observation['geometry']['tip_world_mm'])
        expected_target=np.asarray(observation['geometry']['target_world_mm'])
        for mode in ('missing','poisoned'):
            with self.subTest(mode=mode):
                value=dict(observation)
                if mode=='missing':value.pop('segmentation')
                else:value['segmentation']=ForbiddenTruth()
                tip,_=self.sensor.geometry.tip(value)
                target,_=self.sensor.geometry.target(value)
                np.testing.assert_allclose(tip,expected_tip,atol=1e-9,rtol=0)
                np.testing.assert_allclose(target,expected_target,atol=1e-9,rtol=0)

    def test_removed_rgb_target_cannot_be_recovered_from_unchanged_depth(self):
        good=self.measure('near')
        rgb=self.frames['near']['rgb'].copy()
        mask=ndimage.binary_dilation(good['feature_masks']['target'],iterations=3)
        rgb[mask]=[180,65,80]
        missing=self.measure('near',rgb=rgb)
        self.assertIsNone(missing['target_pixel'])
        self.assertIsNone(missing['geometry']['target_world_mm'])
        self.assertIn('TARGET_NOT_VISIBLE',missing['geometry']['target_error'])
        self.assertIsNone(missing['oct_navigation_depth'])
        self.assertIsNone(self.sensor.visibility_metrics(ForbiddenTruth()))

    def test_empty_depth_fails_observability_without_pose_fallback(self):
        self.measure('home')
        for value in (0.,float('nan')):
            with self.subTest(depth=value):
                depth=np.full_like(self.frames['home']['depth'],value)
                observation=self.measure('home',depth=depth)
                self.assertIsNone(observation['geometry']['tip_world_mm'])
                self.assertIsNone(observation['geometry']['target_world_mm'])
                self.assertIsNone(observation['oct_navigation_depth'])

    def test_depth_change_changes_estimate_or_rejects_measurement(self):
        original=self.measure('home')
        changed=self.frames['home']['depth'].copy()+.0001
        observation=self.measure('home',depth=changed)
        for key in ('tip_world_mm','target_world_mm'):
            point=observation['geometry'][key]
            if point is not None:
                change=float(np.linalg.norm(np.asarray(point)-original['geometry'][key])*1000.)
                self.assertGreater(change,5.)

    def test_rgb_and_depth_render_once_with_identical_visible_scene(self):
        sensor=StrictRGBDSensor(self.reference,self.settings)
        calls=[];options=[]
        groups_before=self.reference.model.geom_group.copy()
        actual=sensor._render
        def track(mode):
            calls.append(mode)
            return actual(mode)
        try:
            with patch.object(sensor,'_render',side_effect=track):
                patches=[]
                for mode in ('rgb','depth'):
                    renderer=sensor.renderers[mode]
                    original=renderer.update_scene
                    def update(*args,_original=original,**kwargs):
                        options.append(kwargs['scene_option'].geomgroup.copy())
                        return _original(*args,**kwargs)
                    patcher=patch.object(renderer,'update_scene',side_effect=update)
                    patches.append(patcher);patcher.start()
                try:observation=sensor.observe()
                finally:
                    for patcher in patches:patcher.stop()
            self.assertEqual(calls,['rgb','depth'])
            self.assertEqual(set(sensor.renderers),{'rgb','depth'})
            np.testing.assert_array_equal(options[0],options[1])
            self.assertEqual(int(options[0][5]),1,'Retinal backdrop was removed from measurement')
            np.testing.assert_array_equal(groups_before,self.reference.model.geom_group)
            np.testing.assert_array_equal(observation['oct_frame'].scene_surface_depth_m,
                                          observation['oct_frame'].registered_feature_depth_m)
            self.assertIs(observation['depth'],observation['oct_depth'])
            self.assertFalse(observation['oct_metadata']['extra']['feature_pass_excludes_fundus_backdrop'])
            with self.assertRaises(ValueError):sensor._render('segmentation')
        finally:sensor.close()

    def test_external_registered_frame_provider_needs_no_simulator_render(self):
        frozen=self.frames['near'];calls=[]
        class Provider:
            def capture(self,*,timestamp_s):
                calls.append(timestamp_s)
                return frozen['rgb'],frozen['depth'],{
                    'source':'test_external_registered_rgbd','is_simulation':False,
                    'raw_oct_available':False,'clock_domain':'device'}
        sensor=StrictRGBDSensor(self.calibration,self.settings,frame_provider=Provider())
        try:
            with patch.object(sensor,'_render',side_effect=forbidden):observation=sensor.observe()
            self.assertEqual(calls,[0.])
            self.assertFalse(observation['oct_metadata']['is_simulation'])
            self.assertEqual(observation['oct_metadata']['clock_domain'],'device')
            self.assertEqual(observation['oct_metadata']['source'],'test_external_registered_rgbd')
        finally:sensor.close()

    def test_controller_selects_strict_sensor_and_rejects_unsafe_modes(self):
        controller=VisualServoController(self.reference,settings=dict(self.settings))
        try:self.assertIsInstance(controller.sensor,StrictRGBDSensor)
        finally:controller.sensor.close()
        for override in ({'position_feedback':'image_depth'},
                         {'occlusion_handling':'static_memory_reobserve'}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                VisualServoController(self.reference,settings={**self.settings,**override})

    def test_strict_controller_rejects_injected_world_goals(self):
        for kwargs in ({'measurement_goal':np.zeros(3)},
                       {'target_transform':lambda target,tip,observation:target}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                controller=VisualServoController(self.reference,settings=dict(self.settings),**kwargs)
                controller.sensor.close()


def verify(output_dir=None):
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StrictRGBDTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'result':'PASS' if result.wasSuccessful() else 'FAIL',
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'scope':'Current visible RGB plus ideal registered simulation depth; fixed colored fiducials and calibration, not real OCT or clinical perception',
        'simulator_object_identity_needed':False,
        'hidden_object_or_fundus_removed_measurement_pass_needed':False,
        'pose_accuracy':StrictRGBDTests.pose_results}
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        (path/'strict_rgbd_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    report=verify(parser.parse_args().output_dir)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['result']=='PASS' else 1)
