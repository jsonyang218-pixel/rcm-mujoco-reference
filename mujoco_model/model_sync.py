"""Exact encoder-FK mapping to the scan model; no ROS or hardware I/O."""
from pathlib import Path
import math
import numbers
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from driver_loader import load_driver
from silver_motion import settings,native_polynomial
from installation import transform,apply as apply_installation

HERE=Path(__file__).resolve().parent
HOME=(0,1466018,893518,0)
RCM=np.array([.456,.070,.125])

def rx(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rz(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

class DriverModel:
    def __init__(self,model_dir=None,needle_profile='new',installation_pitch_deg=None):
        if needle_profile not in ('new','legacy'):raise ValueError('Unknown needle profile')
        self.root=Path(model_dir) if model_dir is not None else HERE
        self.kin=load_driver();self.profile=needle_profile
        self.offset_sign=-1 if needle_profile=='new' else 1
        self.alpha0=self.kin.b_alpha_from_robot_encoder(HOME[1])
        theta0=math.radians(self.alpha0+22)
        # One installation for the mechanical arm, independent of which needle
        # is attached. Changing needle profile changes its signed tip offset,
        # never the installation, A bearing axis or C motor direction.
        # This is still a visual registration, not a measured installation.
        self.robot_to_installation=rz(math.pi/2)@rx(math.pi/2-theta0)
        self.installation,self.install_R,self.install_t=transform(self.root,installation_pitch_deg)
        self.robot_to_world=self.install_R@self.robot_to_installation
        self.rcm_world=self.install_R@RCM+self.install_t
        self.c_direction=1
        encoder_bounds=self.kin.ROBOT_ENCODER_LIMITS
        self.joint_bounds={
            'A':tuple(math.radians(value*360/200000) for value in encoder_bounds['A']),
            'B':tuple(math.radians(self.kin.b_alpha_from_robot_encoder(value)-self.alpha0)
                      for value in encoder_bounds['B']),
            'C':tuple((value/self.kin.ENCODER_COUNTS_PER_MM-14)*.001
                      for value in encoder_bounds['C']),
        }
        self.joint_bounds['rear_pitch']=self.joint_bounds['B']
        self.joint_bounds['upper_counter_pitch']=tuple(-v for v in reversed(self.joint_bounds['B']))
        self.joint_bounds['lower_counter_pitch']=self.joint_bounds['upper_counter_pitch']
        cfg,self.silver_axis,self.silver_gain,self.silver_zero,silver_bounds=settings(self.root,self.kin)
        self.joint_bounds['silver_B_linear']=silver_bounds
        # Build from the current uninstalled template so angle edits are idempotent.
        tree=ET.parse(self.root/'assets/scan_template.xml');root=tree.getroot()
        root.set('model','driver_integrated_'+needle_profile)
        root.find('compiler').set('meshdir',str((self.root/'assets').resolve()))
        # Separate closed display skins retain the original part frames and
        # articulation; raw scan meshes remain untouched as source evidence.
        for name in ('fixed_housing', 'parallel_cover', 'distal_guide', 'C_carriage'):
            root.find(f"./asset/mesh[@name='{name}']").set('file', f'{name}_enclosed.obj')
        # Opaque white plastic shell follows A only. Mesh vertices are in the
        # installation frame; compensate for A_stage's origin at encoder zero.
        shell = root.find("./worldbody/geom[@name='scan_fixed_housing']")
        a_body = root.find(".//body[@name='A_stage']")
        root.find('worldbody').remove(shell)
        shell.set('pos', ' '.join(map(str, -RCM)))
        a_body.append(shell)
        material = root.find("./asset/material[@name='mat_scan_fixed_housing']")
        material.set('rgba', '.84 .86 .89 1')
        material.set('specular', '.4')
        material.set('shininess', '.55')
        # MuJoCo's default visible geom groups are 0, 1, 2. Keep needle
        # geometry in its original group while hiding internal visual links.
        for name in ('guide_link', 'rear_link', 'upper_link', 'lower_link'):
            root.find(f".//geom[@name='{name}']").set('group', '3')
        axis=self.robot_to_installation@np.array([0,1,0])
        for joint in root.findall('.//joint'):
            name=joint.get('name')
            if name=='A':joint.set('axis',' '.join(map(str,axis)))
            if name in ('B','rear_pitch','upper_counter_pitch','lower_counter_pitch'):
                joint.set('axis','0 1 0')
            if name=='C':joint.set('axis','0 0 1')
            if name=='silver_B_linear':joint.set('axis',' '.join(map(str,self.silver_axis)))
            if name in self.joint_bounds:
                joint.set('range',' '.join(map(str,self.joint_bounds[name])))
        silver_equal=root.find("./equality/joint[@name='couple_silver_B_linear']")
        if silver_equal is not None:
            coefficients,_=native_polynomial(self.kin,self.silver_gain,self.silver_zero)
            silver_equal.set('polycoef',' '.join(map(str,coefficients)))
        for act in root.findall('./actuator/position'):
            name=act.get('joint')
            if name in self.joint_bounds:
                act.set('ctrlrange',' '.join(map(str,self.joint_bounds[name])))
        cbody=root.find(".//body[@name='C_translation']")
        # The signed source offset is along the shared driver direction (+Z at
        # home). Retain the needle base; move only profile-specific geometry.
        tip_z=self.offset_sign*.004
        tip=cbody.find("site[@name='needle_tip']")
        if tip is None:raise ValueError('Needle-tip site absent from model')
        tip.set('pos',f'0 0 {tip_z}')
        base=cbody.find("site[@name='needle_base']")
        if base is not None:base.set('pos','0 0 .060')
        junction=cbody.find("site[@name='driver_junction']")
        if junction is not None:junction.set('pos','0 0 0')
        shaft_end=.008 if needle_profile=='legacy' else 0.0
        for name,start,end in (
            ('needle_shaft',.060,shaft_end),
            ('needle_distal',shaft_end,tip_z),
        ):
            geom=cbody.find(f"geom[@name='{name}']")
            if geom is None:raise ValueError(f'{name} geom absent from model')
            geom.attrib.pop('pos',None);geom.attrib.pop('quat',None)
            geom.set('fromto',f'0 0 {start} 0 0 {end}')
        if cbody.find("site[@name='driver_junction']") is None:
            ET.SubElement(cbody,'site',name='driver_junction',pos='0 0 0',size='.0004',rgba='0 .9 .3 1')
        apply_installation(root,self.installation,self.install_R,self.install_t)
        self.xml=ET.tostring(root,encoding='unicode')
        self.model=mujoco.MjModel.from_xml_string(self.xml)
        self.data=mujoco.MjData(self.model)
        self.sites={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,name)
            for name in ('driver_junction','needle_tip','needle_base','rcm_site')}
        self.sync(HOME)

    def sync(self,robot):
        robot=tuple(robot)
        if len(robot)!=4:raise ValueError('Four encoder counts required')
        if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or
               not -(2**31)<=int(v)<2**31 for v in robot):
            raise ValueError('Robot encoder values must be int32 integers')
        robot=tuple(int(v) for v in robot)
        self.kin.validate_manual_jog(robot,(0,0,0,0))
        phi=math.radians(robot[0]*360/200000)
        b=math.radians(self.kin.b_alpha_from_robot_encoder(robot[1])-self.alpha0)
        c=(robot[2]/self.kin.ENCODER_COUNTS_PER_MM-14)*.001
        silver=(robot[1]-self.silver_zero)/self.kin.ENCODER_COUNTS_PER_MM*.001*self.silver_gain
        positions={'A':phi,'B':b,'C':c,'rear_pitch':b,'upper_counter_pitch':-b,'lower_counter_pitch':-b,'silver_B_linear':silver}
        for name,value in positions.items():
            j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            lo,hi=self.model.jnt_range[j]
            if not lo-1e-10<=value<=hi+1e-10:raise ValueError(f'{name} outside source-derived software range')
        for name,value in positions.items():
            j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            self.data.qpos[self.model.jnt_qposadr[j]]=value
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)
        tip=self.data.site_xpos[self.sites['needle_tip']].copy()
        junction=self.data.site_xpos[self.sites['driver_junction']].copy()
        expected_tip=self.rcm_world+self.robot_to_world@np.array(self.kin.model_tip_from_robot_encoders(robot,4,self.offset_sign))*.001
        expected_junction=self.rcm_world+self.robot_to_world@np.array(self.kin.model_tip_from_robot_encoders(robot,0))*.001
        self.last_robot=robot
        self.diagnostics={'profile':self.profile,'robot':robot,'A_deg':math.degrees(phi),
            'B_relative_deg':math.degrees(b),'B_alpha_deg':self.alpha0+math.degrees(b),'C_axial_mm':c*1000,
            'max_fk_error_m':max(float(np.linalg.norm(tip-expected_tip)),float(np.linalg.norm(junction-expected_junction))),
            'tip_world_m':tip.tolist(),'junction_world_m':junction.tolist(),
            'silver_linear_travel_mm':silver*1000,'silver_motion_mapping_calibrated':False,
            'installation_pitch_y_deg':self.installation['pitch_y_deg'],'installation_photo_estimate':True,
            'shared_installation_across_needle_profiles':True,
            'alignment_status':'exact driver algebra in one visually aligned, uncalibrated world frame'}
        return self.diagnostics

    def feedback_from_model(self,message):
        """Replace XYZ with MuJoCo junction transformed back into driver mm."""
        junction=self.data.site_xpos[self.sites['driver_junction']]
        xyz=(self.robot_to_world.T@(junction-self.rcm_world))*1000
        for name,value in zip(('posX','posY','posZ'),xyz):
            setattr(message,name,float(np.float32(value)))
        return message

    def export_xml(self,destination):
        """Portable profile-specific MJCF next to assets, with relative mesh path."""
        root=ET.fromstring(self.xml);root.find('compiler').set('meshdir','assets')
        ET.indent(root,space='  ');ET.ElementTree(root).write(destination,encoding='unicode')
