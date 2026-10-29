"""Rigid static installation transform; lab surfaces remain world-horizontal."""
from pathlib import Path
import math,json,xml.etree.ElementTree as ET
import numpy as np

def transform(root_dir,pitch_override=None):
    cfg=json.loads((Path(root_dir)/'installation_config.json').read_text())
    angle=float(cfg['pitch_y_deg'] if pitch_override is None else pitch_override)
    if not math.isfinite(angle) or not -80<=angle<=80:raise ValueError('Installation pitch outside display range')
    cfg['pitch_y_deg']=angle
    a=math.radians(angle);c,s=math.cos(a),math.sin(a)
    R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    pivot=np.array(cfg['pivot_reference_m'],float)
    t=pivot-R@pivot+np.array(cfg['additional_translation_m'],float)
    return cfg,R,t

def text(values):return ' '.join(f'{float(x):.12g}' for x in values)

def apply(root,cfg,R,t):
    world=root.find('worldbody')
    angle=math.radians(cfg['pitch_y_deg'])/2
    installation=ET.SubElement(world,'body',name='static_installation',pos=text(t),quat=text([math.cos(angle),0,math.sin(angle),0]))
    robot_bodies={'A_stage','silver_linear_stage'}
    robot_sites={'base_reference','rcm_site'}
    for node in list(world):
        name=node.get('name','')
        if ((node.tag=='body' and name in robot_bodies) or
            (node.tag=='site' and name in robot_sites) or
            (node.tag=='geom' and (name=='scan_fixed_housing' or name.startswith('rcm_cross_')))):
            world.remove(node);installation.append(node)
    old_rcm=np.array([.456,.070,.125]);new_rcm=R@old_rcm+t;delta=new_rcm-old_rcm
    for geom in world.findall('geom'):
        if geom.get('name') in ('phantom','phantom_pad') or geom.get('name','').startswith('target_tape_'):
            p=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            geom.set('pos',text(p+delta))
    support=world.find("geom[@name='support']")
    p=np.array(cfg['pivot_reference_m']);z=float(cfg['reference_plane_z_m'])
    support.set('pos',text([p[0],p[1],z/2]));support.set('size',text([.065,z/2]))
    # Horizontal mounting plate corresponds to the red reference rectangle.
    ET.SubElement(world,'geom',name='horizontal_mount_plate',type='box',pos=text([p[0],p[1],z-.004]),
        size='.15 .12 .004',rgba='.67 .71 .75 1')
    for i,(start,end) in enumerate([
        ([p[0]-.15,p[1]-.12,z+.0004],[p[0]+.15,p[1]-.12,z+.0004]),
        ([p[0]+.15,p[1]-.12,z+.0004],[p[0]+.15,p[1]+.12,z+.0004]),
        ([p[0]+.15,p[1]+.12,z+.0004],[p[0]-.15,p[1]+.12,z+.0004]),
        ([p[0]-.15,p[1]+.12,z+.0004],[p[0]-.15,p[1]-.12,z+.0004])]):
        ET.SubElement(world,'geom',name=f'horizontal_reference_edge_{i}',type='capsule',fromto=text([*start,*end]),
            size='.0007',rgba='.95 .09 .04 1')
    # Translate the target, but keep its pad, box and table horizontal.
    pad=world.find("geom[@name='phantom_pad']");padpos=np.fromstring(pad.get('pos'),sep=' ')
    half=np.fromstring(pad.get('size'),sep=' ');tabletop=float(padpos[2]-half[2])
    if tabletop>.025:
        center=[padpos[0]+.045,padpos[1],tabletop-.008]
        ET.SubElement(world,'geom',name='target_table_top',type='box',pos=text(center),size='.15 .14 .008',rgba='.16 .19 .23 1')
        for i,(dx,dy) in enumerate([(-.12,-.11),(.12,-.11),(-.12,.11),(.12,.11)]):
            h=(tabletop-.016)/2
            ET.SubElement(world,'geom',name=f'target_table_leg_{i}',type='box',pos=text([center[0]+dx,center[1]+dy,h]),size=text([.008,.008,h]),rgba='.22 .25 .28 1')
    # Camera aimed at the raised robot rather than the former uninstalled pose.
    center=np.array([.21,.03,.25]);center=R@center+t
    camera=world.find("camera[@name='overview']")
    look=center+np.array([.10,0,.015]);pos=look+np.array([.62,-.85,.30])
    forward=(look-pos);forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    camera.set('pos',text(pos));camera.set('xyaxes',text([*right,*up]));camera.set('fovy','40')
    root.find('statistic').set('center',text(center));root.find('statistic').set('extent','.65')
    return new_rcm
