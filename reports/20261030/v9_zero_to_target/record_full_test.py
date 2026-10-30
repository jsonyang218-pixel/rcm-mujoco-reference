"""Record a direct v9 visual servo run from zero without editing the model."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from rcm_scene_v9 import RCMSceneV9
from rcm_visual_servo import VisualServoController, visual_servo_settings, _dilate_mask

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
OUT=parser.parse_args().output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=False)
scene=RCMSceneV9()
assert np.allclose(scene.current_task(),np.zeros(3),atol=1e-12)
settings=visual_servo_settings(scene)
settings.update(max_iterations=1000,save_diagnostics=True)
controller=VisualServoController(scene,settings=settings,output_dir=OUT)
root=ET.fromstring(scene.xml)
root.find('visual/headlight').set('ambient','.4 .4 .4')
root.find('visual/headlight').set('diffuse','.6 .6 .6')
ET.SubElement(root.find('asset'),'texture',name='video_dark_sky',type='skybox',builtin='gradient',rgb1='.045 .06 .08',rgb2='.045 .06 .08',width='512',height='3072')
world=root.find('worldbody')
for geom in list(world.findall('geom')):
    name=geom.get('name','')
    if name in {'floor','support','phantom','phantom_pad','horizontal_mount_plate'} or name.startswith(('horizontal_reference_edge_','target_tape_','target_table_')):
        world.remove(geom)
center=scene.data.geom_xpos[scene.model.geom('eye_reference').id].copy()
for name,c,span in [('video_full',np.array([.16,.03,.30]),.66),('video_eye',center,.08)]:
    ET.SubElement(world,'camera',name=name,pos=' '.join(map(str,c+[0,-1,0])),xyaxes='1 0 0 0 0 1',orthographic='true',fovy=str(span))
display=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
data=mujoco.MjData(display)
for i in range(display.ngeom):
    name=mujoco.mj_id2name(display,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
    if name in {'rear_link','upper_link','lower_link','guide_link','axis_aid'} or name.startswith('rcm_cross_'):
        display.geom_rgba[i,3]=0
eye_id=display.geom('eye_reference').id
eye_mat=int(display.geom_matid[eye_id])
display.mat_rgba[eye_mat,3]=.13
option=mujoco.MjvOption();option.sitegroup[:]=0;option.geomgroup[3]=0
renderer=mujoco.Renderer(display,width=600,height=400)
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={s:ImageFont.truetype(font_path,s) for s in (18,20,22,26,34)}
log=(OUT/'ffmpeg.log').open('w')
video=OUT/'v9_zero_to_target_full_test.mp4'
encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x1080','-r','20','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=log)
started=time.monotonic()
frame_count=0
last_index=-1
max_fk=max_rcm=0.
crop=None
last_frame=None

def emit(frame,count):
    global frame_count
    payload=frame.tobytes()
    for _ in range(count):encoder.stdin.write(payload)
    frame_count+=count

def add_sphere(pos,radius,rgba):
    idx=renderer.scene.ngeom
    mujoco.mjv_initGeom(renderer.scene.geoms[idx],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),np.asarray(pos),np.eye(3).ravel(),np.array(rgba,dtype=np.float32))
    renderer.scene.ngeom+=1

def compose(index,observation,final_report=None):
    global crop,max_fk,max_rcm,last_frame
    row=controller.rows[-1]
    geometry=observation.get('geometry',{})
    metrics=scene.metrics()
    max_fk=max(max_fk,metrics['fk_agreement_um']);max_rcm=max(max_rcm,metrics['rcm_drift_um'])
    raw=observation['rgb'].copy()
    if observation.get('needle_mask') is not None:raw[_dilate_mask(observation['needle_mask'],2)]=(255,225,35)
    im=Image.fromarray(raw);d=ImageDraw.Draw(im)
    projected=[]
    for key,color in [('tip_world_mm','#32ff50'),('target_world_mm','#00dcff')]:
        point=geometry.get(key)
        if point is None:continue
        pixel,_=controller.sensor.geometry.project(point);projected.append(pixel)
        x,y=pixel
        d.line((x-10,y,x+10,y),fill=color,width=3);d.line((x,y-10,x,y+10),fill=color,width=3)
    if len(projected)==2:d.line(tuple(projected[0])+tuple(projected[1]),fill='white',width=2)
    if crop is None:
        pts=np.array(projected)
        width=max(800.,np.ptp(pts[:,0])+300.,(np.ptp(pts[:,1])+240.)*4/3)
        width=min(width,1920.)
        c=pts.mean(axis=0)
        crop=tuple(map(int,[c[0]-width/2,c[1]-width*3/8,c[0]+width/2,c[1]+width*3/8]))
    eye=im.crop(crop).resize((928,696),Image.Resampling.LANCZOS)
    data.qpos[:]=scene.data.qpos;mujoco.mj_forward(display,data)
    panels=[]
    for camera in ('video_full','video_eye'):
        renderer.update_scene(data,camera=camera,scene_option=option)
        if camera=='video_eye':
            # These enlarged colored spheres and the transparent shell are display aids only.
            add_sphere(np.array(metrics['tip_world_mm'])*.001,.00055,[.2,1,.3,1])
            add_sphere(np.array(metrics['target_world_mm'])*.001,.00065,[0,.85,1,1])
            add_sphere(np.array(metrics['rcm_world_mm'])*.001,.0005,[1,.45,.1,1])
        panels.append(Image.fromarray(renderer.render().copy()))
    frame=Image.new('RGB',(1600,1080),'#101720');frame.paste(panels[0],(24,160));frame.paste(panels[1],(24,600));frame.paste(eye,(648,160))
    d=ImageDraw.Draw(frame)
    def text(x,y,value,size=22,color='#d6e1ef'):d.text((x,y),value,font=fonts[size],fill=color)
    text(24,18,'v9 从零位到目标 · 完整视觉闭环测试',34,'white')
    text(24,70,'起点 A=0° / B=0° / C=0 mm · 使用当前 v9 相机反馈 · 1× 仿真速度',22)
    text(24,120,'整机运动',26,'white');text(24,560,'眼球透明示意：针尖 → 目标',26,'white')
    text(648,120,'眼内相机 · 固定视野（白线为测量误差）',26,'white')
    done=final_report is not None
    label=('PASS · 连续 3 帧停稳' if final_report['result']=='PASS' else '未通过 · '+str(final_report['reason'])) if done else ('零位起点' if index==0 else '视觉反馈控制中')
    text(1030,24,label,26,'#6cf4b2' if done and final_report['result']=='PASS' else '#72caff')
    error=float(row['tip_error_um'])
    text(1030,70,f'距目标 {error/1000:.3f} mm / {error:.1f} μm',22,'#f2d089')
    text(24,1007,f"A {float(row['A_deg']):+.2f}°   B {float(row['B_deg']):+.2f}°   C {float(row['C_mm']):.3f} mm",20)
    text(648,865,'青：目标   绿：针尖   橙：RCM；球体 / 黄色针体为显示辅助',18)
    est=row.get('estimated_position_error_um')
    text(648,897,f"仿真时间 {float(row['time_s']):.2f} s    迭代 {index+1}    视觉误差 {float(est):.2f} μm" if est is not None else '视觉观测丢失',20)
    x0,y0,x1,y1=665,943,1548,1012
    d.line((x0,y1,x1,y1),fill='#44566b')
    points=[]
    for r in controller.rows:
        e=r.get('estimated_position_error_um')
        if e is not None:points.append((x0+(x1-x0)*float(r['time_s'])/40.,y1-(y1-y0)*np.log10(max(float(e),1))/np.log10(25000)))
    if len(points)>1:d.line(points,fill='#66d8ff',width=3)
    ty=y1-(y1-y0)*np.log10(20)/np.log10(25000);d.line((x0,ty,x1,ty),fill='#6cf4b2')
    text(648,1018,'视觉误差（对数尺度）；绿线：20 μm                  0 → 40 s',18)
    text(24,1047,'直接 v9 单目标仿真；理想 RGB-D / 分割。透明眼球和放大标记仅用于显示；不代表真实硬件性能。',18,'#96a9bd')
    last_frame=frame
    return frame

def callback(index,observation):
    global last_index
    if index%40==0:
        row=controller.rows[-1]
        print(json.dumps({'iteration':index,'wall_s':round(time.monotonic()-started,1),'C_mm':row['C_mm'],'error_um':row['tip_error_um'],'tip_pixels':row['tip_pixels'],'target_pixels':row['target_pixels']}),flush=True)
    if index%4==0:
        frame=compose(index,observation)
        if index==0:
            frame.save(OUT/'start.png');emit(frame,60)
        emit(frame,4);last_index=index
    return True

controller.iteration_callback=callback
try:
    report=controller.run()
    report.update(audit_backend='RCMSceneV9 direct sensor and control',audit_initial_task_A_deg_B_deg_C_mm=[0.,0.,0.],audit_scope='Complete zero-to-target single-goal visual servo simulation; no tissue contact or hardware acceptance',audit_wall_time_s=time.monotonic()-started)
    controller.write(report,OUT)
    final_index=len(controller.rows)-1
    final=compose(final_index,controller.final_observation,report)
    emit(final,max(1,(final_index-last_index)))
    emit(final,80);final.save(OUT/'final.png')
    encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
    if report['result']=='PASS':
        assert report['terminal_stop_validated'] and report['trajectory_rate_limits_pass']
        assert report['final_tip_target_error_um']<20.
    metadata={'video':video.name,'result':report['result'],'reason':report['reason'],'fps':20,'frames':frame_count,'duration_s':frame_count/20,'playback_speed':1.,'simulation_duration_s':controller.rows[-1]['time_s'],'initial_task_A_deg_B_deg_C_mm':[0,0,0],'initial_error_um':controller.rows[0]['tip_error_um'],'final_error_um':report['final_tip_target_error_um'],'iterations':report['iterations'],'max_rcm_drift_um':max_rcm,'max_fk_error_um':max_fk,'control_feedback':'Current RCMSceneV9 RGB-D geometry; no legacy camera feedback','recording':'Sensor frames captured during the actual closed-loop run; every fourth step sampled','display_only':['Transparent eye shell','Enlarged tip/target/RCM spheres in side view','Needle highlight and geometry projection crosshairs','Fixed camera crop'],'report':'visual_servo_report.json'}
    (OUT/'video_validation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
finally:
    if encoder.poll() is None:
        encoder.stdin.close();encoder.wait()
    log.close();renderer.close();scene.close()
