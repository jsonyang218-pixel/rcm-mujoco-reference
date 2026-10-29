"""Render the photo-estimated installation against an explicit horizontal datum."""
from pathlib import Path
import os,sys,json,argparse,math
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MUJOCO_GL','egl' if sys.platform=='linux' else 'glfw')
os.environ.setdefault('MESA_SHADER_CACHE_DIR',str(HERE/'.cache'))
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from model_sync import DriverModel

def font(size):
    for p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','C:/Windows/Fonts/msyh.ttc','/System/Library/Fonts/PingFang.ttc'):
        if Path(p).is_file():return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def capture(pitch,width=1100,height=850):
    scene=DriverModel(installation_pitch_deg=pitch)
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[.15,.025,.30];cam.distance=.92;cam.azimuth=90;cam.elevation=0
    # Fixed front-side orthographic projection makes the ground datum horizontal.
    from xml.etree import ElementTree as ET
    root=ET.fromstring(scene.xml);world=root.find('worldbody')
    ET.SubElement(world,'camera',name='reference_side',pos='.15 -1 .30',xyaxes='1 0 0 0 0 1',orthographic='true',fovy='.64')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m);d.qpos[:]=scene.data.qpos;mujoco.mj_forward(m,d)
    with mujoco.Renderer(m,width=width,height=height) as r:
        options=mujoco.MjvOption();options.sitegroup[:]=0
        r.update_scene(d,camera='reference_side',scene_option=options);im=Image.fromarray(r.render().copy())
    dr=ImageDraw.Draw(im);dr.rectangle((0,0,width,76),fill=(20,27,36))
    dr.text((18,10),f'相对旧模型修正 {pitch:+.0f}°｜前端抬高约 {-pitch:.0f}°',font=font(29 if width>=1000 else 25),fill='white')
    dr.text((18,48),'红边为水平基准；单张照片估计，各轴相对关系保留',font=font(20),fill='#c9d6de')
    # Add a clearly visible datum line in the side projection at world z=0.1m.
    y=height/2-(.10-.30)*height/.64
    dr.line((45,round(y),width-45,round(y)),fill='#ff442d',width=3)
    dr.text((48,round(y)+10),'水平基准',font=font(23),fill='#ff6f52')
    return im

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pitch',type=float,default=None);args=p.parse_args()
    default=DriverModel();angle=default.installation['pitch_y_deg'] if args.pitch is None else args.pitch
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    capture(angle).save(out/'static_side_reference.png')
    samples=(-35.,-40.,-45.)
    sheet=Image.new('RGB',(2400,650))
    for i,a in enumerate(samples):sheet.paste(capture(a,800,650),(800*i,0))
    sheet.save(out/'pitch_options.png')
    with mujoco.Renderer(default.model,width=1100,height=850) as r:
        r.update_scene(default.data,camera='overview');Image.fromarray(r.render()).save(out/'static_overview.png')
    print(out/'static_side_reference.png')

if __name__=='__main__':main()
