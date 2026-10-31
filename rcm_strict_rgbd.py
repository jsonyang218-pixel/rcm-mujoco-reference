"""Current-frame RGB + visible depth perception, without simulator object IDs.

The simulator is only a frame provider. Detection/geometry can also consume
externally registered RGB/depth arrays through measure(). Colored fiducials
and known dimensions are explicit calibration assumptions, not lesion AI.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import mujoco
import numpy as np

from rcm_oct import SyntheticOCTProvider, relative_depth
from rcm_visual_geometry import NeedleGeometry


class StrictRGBDSensor:
    def __init__(self, reference, settings, frame_provider=None):
        from rcm_rgb_features import RGBFeatureDetector
        self.reference=reference;self.settings=settings
        self.camera_name=settings['camera_name']
        self.width,self.height=reference.eye_camera['resolution_px']
        pos,rotation=reference.camera_pose(self.camera_name)
        # Camera/RCM/shape values are fixed calibration; geom IDs and moving
        # target/needle transforms are never read by this estimator.
        self.geometry=NeedleGeometry(camera_position_mm=pos*1000.,camera_rotation=rotation,
            width=self.width,height=self.height,
            focal_px=.5*self.height/math.tan(math.radians(reference.eye_camera['fovy_deg'])/2),
            rcm_mm=reference.rcm_mm,
            proximal_radius_mm=reference.config['needle']['proximal_radius_mm'],
            proximal_length_mm=reference.config['needle']['proximal_length_mm'],
            distal_length_mm=reference.config['needle']['distal_length_mm'],
            apex_radius_mm=reference.config['needle']['tip_marker_radius_mm'],
            target_radius_mm=float(settings.get('calibrated_target_radius_mm',.15)),
            proximal_id=None,apex_id=None,target_id=None,distal_ids=(),
            target_rgb_min=settings['target_rgb_min'],target_rgb_max=settings['target_rgb_max'])
        self.detector=RGBFeatureDetector(settings)
        self.frame_provider=frame_provider
        self.renderers={}
        if frame_provider is None:
            for mode in ('rgb','depth'):
                renderer=mujoco.Renderer(reference.model,height=self.height,width=self.width)
                if mode=='depth':renderer.enable_depth_rendering()
                self.renderers[mode]=renderer
        self.provider=SyntheticOCTProvider(nominal_period_s=settings['sample_period_s'],
            min_valid_depth_m=reference.eye_camera['znear_mm']*.001,
            max_valid_depth_m=reference.eye_camera['zfar_mm']*.001*(1.-1e-4))
        self.observation_time_s=None;self.observation_time_origin_s=0.
        self.last_oct_frame=None;self.sequence=0

    def close(self):
        for renderer in self.renderers.values():renderer.close()

    def _render(self,mode):
        if mode not in ('rgb','depth'):raise ValueError('Strict perception only renders RGB and depth')
        reference=self.reference
        before=(reference.model.vis.map.znear,reference.model.vis.map.zfar)
        reference.model.vis.map.znear=reference.eye_camera['znear_mm']*.001/reference.model.stat.extent
        reference.model.vis.map.zfar=reference.eye_camera['zfar_mm']*.001/reference.model.stat.extent
        try:
            option=mujoco.MjvOption();option.sitegroup[:]=0
            option.geomgroup[:]=[0,0,0,1,1,1]
            # RGB and depth share the same optical scene, including retina.
            # Viewer-only shells/aids are excluded from BOTH sensor passes.
            self.renderers[mode].update_scene(reference.data,camera=self.camera_name,scene_option=option)
            return self.renderers[mode].render().copy()
        finally:reference.model.vis.map.znear,reference.model.vis.map.zfar=before

    def observe(self):
        timestamp=(self.sequence*float(self.settings['sample_period_s'])
                   if self.observation_time_s is None else float(self.observation_time_s))
        if self.frame_provider is not None:
            rgb,depth,metadata=self.frame_provider.capture(timestamp_s=timestamp)
        else:
            rgb=self._render('rgb');depth=self._render('depth')
            metadata={'source':'mujoco_visible_scene_depth_simulation','is_simulation':True,
                      'raw_oct_available':False,'realtime_validated':False}
        return self.measure(rgb,depth,timestamp,metadata)

    def measure(self,rgb,depth,timestamp,metadata=None):
        rgb=np.asarray(rgb);depth=np.asarray(depth)
        if rgb.dtype!=np.uint8 or rgb.shape!=(self.height,self.width,3) or depth.shape!=rgb.shape[:2]:
            raise ValueError('Expected registered uint8 RGB and metric depth at calibrated resolution')
        repeated=self.last_oct_frame is not None and timestamp==self.last_oct_frame.timestamp_s
        frame=self.provider.capture(depth,depth,timestamp_s=timestamp,camera_name=self.camera_name,
            acquisition_wall_s=time.monotonic(),allow_repeated_timestamp=repeated,
            metadata={**(metadata or {}),'feature_pass_excludes_fundus_backdrop':False,
                'visibility_gated_by_normal_scene_segmentation':False,
                'rgb_and_depth_share_visibility':True,'simulator_object_ids_used':False})
        provenance=metadata or {}
        frame=replace(frame,source=provenance.get('source','external_registered_rgbd'),
            is_simulation=bool(provenance.get('is_simulation',False)),
            raw_oct_available=bool(provenance.get('raw_oct_available',False)),
            realtime_validated=bool(provenance.get('realtime_validated',False)),
            clock_domain=provenance.get('clock_domain','simulation' if provenance.get('is_simulation',False) else 'external'),
            registration_description='One normally visible, registered metric depth map; no hidden-feature render or object-ID segmentation.')
        self.last_oct_frame=frame;self.sequence+=1
        masks=self.detector.detect(rgb)
        def centroid(mask):
            rows,cols=np.where(mask)
            return None if not len(rows) else [float(cols.mean()),float(rows.mean())]
        observation={'rgb':rgb.copy(),'depth':frame.scene_surface_depth_m,
            'oct_depth':frame.scene_surface_depth_m,'oct_depth_valid_mask':frame.scene_valid_mask,
            'feature_masks':masks,'needle_mask':masks['proximal']|masks['distal']|masks['apex'],
            'target_pixel':centroid(masks['target']),'tip_pixel':None,
            'target_depth_m':None,'tip_depth_m':None,
            'target_pixels':int(masks['target'].sum()),'tip_pixels':int(masks['apex'].sum()),
            'segmentation':None,'oct_frame':frame,'oct_metadata':frame.metadata(),
            'oct_depth_source':frame.source,'oct_depth_unit':'m','oct_navigation_depth':None,
            'perception_backend':'strict_rgbd','mask_source':'current_rgb',
            'simulator_ids_used_for_control':False,'hidden_geometry_render_used_for_control':False}
        measured={}
        for name in ('target','tip'):
            try:
                point,quality=getattr(self.geometry,name)(observation)
                measured[name+'_world_mm']=point.tolist();measured[name+'_quality']=quality
            except RuntimeError as error:
                measured[name+'_world_mm']=None;measured[name+'_error']=str(error)
        observation['geometry']=measured
        for name in ('target','tip'):
            point=measured[name+'_world_mm']
            if point is not None:observation[name+'_pixel'],observation[name+'_depth_m']=self.geometry.project(point)
            else:observation[name+'_pixel']=observation[name+'_depth_m']=None
        if measured['tip_world_mm'] is not None and measured['target_world_mm'] is not None:
            def optical(point):
                p=self.geometry.camera_rotation.T@(np.asarray(point)-self.geometry.camera_position_mm)*.001
                return p*[1.,-1.,-1.]
            observation['oct_navigation_depth']=relative_depth(optical(measured['tip_world_mm']),
                optical(measured['target_world_mm']),source_flags={'source':frame.source,
                    'frame_id':frame.frame_id,'target_source':'current_rgb_visible_depth','is_simulation':frame.is_simulation})
        return observation

    def visibility_metrics(self,segmentation=None):
        # RGB detection cannot attribute an absent target specifically to the
        # needle. Keep causal occlusion labels unknown instead of consulting IDs.
        return None
