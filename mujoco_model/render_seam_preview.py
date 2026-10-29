"""Render v8/v9 comparison of the integral silver cover, preserving white skin."""
from pathlib import Path
import os,sys,json,math,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MUJOCO_GL','egl' if sys.platform=='linux' else 'glfw')
os.environ.setdefault('MESA_SHADER_CACHE_DIR',str(HERE/'.cache'))
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from model_sync import DriverModel,HOME
from preview_static_pose import font

def visual_model(scene,before=False,view='side'):
    root=ET.fromstring(scene.xml);world=root.find('worldbody')
    if before:
        for name in ('fixed_housing','parallel_cover','distal_guide','C_carriage'):
            root.find(f"./asset/mesh[@name='{name}']").set('file',str(HERE/f'evidence/v8_meshes/{name}.obj'))
        for parent in root.iter():
            for node in list(parent):
                if node.get('name','').startswith('seam_backing'):parent.remove(node)
    # Fixed side camera covers the skin, curved lip and real linkage opening.
    if view=='side':
        ET.SubElement(world,'camera',name='seam_detail',pos='.105 -1 .415',xyaxes='1 0 0 0 0 1',orthographic='true',fovy='.17')
    else:
        look=np.array([.12,0,.37]);pos=look+np.array([.2,-.65,.15]);f=(look-pos);f/=np.linalg.norm(f)
        x=np.cross(f,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(x,f)
        ET.SubElement(world,'camera',name='seam_detail',pos=' '.join(map(str,pos)),xyaxes=' '.join(map(str,[*x,*y])),fovy='24')
    for geom in root.findall('.//geom'):
        name=geom.get('name','')
        if name in ('floor','support','horizontal_mount_plate','phantom','phantom_pad') or name.startswith(('target_','horizontal_reference_','rcm_cross_')):
            geom.set('rgba','0 0 0 0')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    return m,mujoco.MjData(m)

def label(im,title,subtitle):
    draw=ImageDraw.Draw(im);draw.rectangle((0,0,im.width,72),fill=(20,27,36))
    draw.text((16,8),title,font=font(27),fill='white')
    draw.text((16,43),subtitle,font=font(19),fill='#c9d6de')
    return im

def save_gif(frames,destination,width):
    # Use one palette throughout so the fixed shell does not change colour
    # between frames, and keep the downloadable animated preview compact.
    resized=[im.resize((width,round(im.height*width/im.width)),Image.Resampling.LANCZOS) for im in frames]
    palette=resized[len(resized)//4].quantize(colors=128)
    encoded=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in resized]
    encoded[0].save(destination,save_all=True,append_images=encoded[1:],duration=80,loop=0,optimize=True)

def main():
    out=HERE/'outputs';out.mkdir(exist_ok=True)
    scene=DriverModel();kin=scene.kin
    before,bd=visual_model(scene,True);after,ad=visual_model(scene,False)
    option=mujoco.MjvOption();option.sitegroup[:]=0;option.geomgroup[2]=0
    frames=[];after_frames=[]
    with mujoco.Renderer(before,width=800,height=650) as rb,mujoco.Renderer(after,width=800,height=650) as ra:
        for i in range(81):
            # +/- ~5 degree B demonstration; silver stage remains independent.
            scalar=math.sin(2*math.pi*i/80)
            robot=kin.predicted_robot_encoders(HOME,(0,round(700*scalar),0,0))
            diag=scene.sync(robot)
            images=[]
            for m,d,r,old in ((before,bd,rb,True),(after,ad,ra,False)):
                d.qpos[:]=scene.data.qpos;mujoco.mj_forward(m,d)
                r.update_scene(d,camera='seam_detail',scene_option=option)
                im=Image.fromarray(r.render().copy())
                images.append(label(im,'修正前：端盖未随银罩移动' if old else '修正后：端盖与银罩是同一整体',f'B 仿真角 {diag["B_relative_deg"]:+.2f}°｜白壳修复和静态角度保留'))
            sheet=Image.new('RGB',(1600,650));sheet.paste(images[0],(0,0));sheet.paste(images[1],(800,0))
            frames.append(sheet)
            after_frames.append(images[1])
            if i==0:sheet.save(out/'silver_before_after_home.png')
            if i==20:sheet.save(out/'silver_before_after_raised.png')
        save_gif(frames,out/'silver_integral_comparison.gif',1280)
        save_gif(after_frames,out/'silver_integral_motion.gif',640)
    print(out/'silver_before_after_home.png')

if __name__=='__main__':main()
