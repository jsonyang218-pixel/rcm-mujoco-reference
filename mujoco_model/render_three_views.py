"""Orthographic front, right and top views of the existing repaired scan model."""
from pathlib import Path
import os,sys,json,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MUJOCO_GL','egl' if sys.platform=='linux' else 'glfw')
os.environ.setdefault('MESA_SHADER_CACHE_DIR',str(HERE/'.cache'))
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from model_sync import DriverModel,HOME

def main():
    folder=HERE/'outputs/three_views';folder.mkdir(parents=True,exist_ok=True)
    original=DriverModel();root=ET.fromstring(original.xml)
    root.find('visual/headlight').set('ambient','.55 .55 .55')
    root.find('visual/headlight').set('diffuse','.55 .55 .55')
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='white_background',type='skybox',builtin='gradient',
        rgb1='1 1 1',rgb2='1 1 1',width='512',height='3072')
    world=root.find('worldbody')
    for geom in list(world.findall('geom')):
        if geom.get('name') in ('floor','support','phantom','phantom_pad') or geom.get('name','').startswith(('target_tape_','target_table_')):
            world.remove(geom)
    span=.68
    views=[('front','正视图','−Y → +Y','X','Z',[.15,-1.,.30],'1 0 0 0 0 1'),
           ('right','右侧视图','+X → −X','Y','Z',[1.15,.025,.30],'0 1 0 0 0 1'),
           ('top','俯视图','+Z → −Z','X','Y',[.15,.025,1.30],'1 0 0 0 1 0')]
    for name,label,direction,h,v,pos,axes in views:
        ET.SubElement(world,'camera',name='ortho_'+name,pos=' '.join(map(str,pos)),xyaxes=axes,
            orthographic='true',fovy=str(span))
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);data.qpos[:]=original.data.qpos;mujoco.mj_forward(model,data)
    candidates=[os.environ.get('MODEL_FONT_PATH',''),
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/msyh.ttc','/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font_path=next((p for p in candidates if p and Path(p).is_file()),None)
    if font_path is None:raise FileNotFoundError('Set MODEL_FONT_PATH to a local .ttf/.ttc font for view labels')
    large=ImageFont.truetype(font_path,30);small=ImageFont.truetype(font_path,17);title=ImageFont.truetype(font_path,35)
    size=850;pad=40
    panel_images=[];metadata={}
    options=mujoco.MjvOption()
    # Hide display-only linkage aids individually, keeping needle geoms even
    # though the original model uses the same group for both.
    auxiliary={'rear_link','upper_link','lower_link','guide_link','axis_aid'}
    for i in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name in auxiliary or name.startswith('rcm_cross_'):model.geom_group[i]=5
    options.geomgroup[5]=0
    options.sitegroup[:]=0 # omit metrology sites; physical needle geometry stays
    with mujoco.Renderer(model,height=size,width=size) as renderer:
        for name,label,direction,h,v,pos,axes in views:
            renderer.update_scene(data,camera='ortho_'+name,scene_option=options)
            rendered=Image.fromarray(renderer.render().copy())
            assert np.var(np.asarray(rendered).astype(float))>10
            panel=Image.new('RGB',(size,size+115),'white');panel.paste(rendered,(0,65));d=ImageDraw.Draw(panel)
            d.text((pad,7),label,font=large,fill='#182533')
            d.text((pad+190,23),f'观看方向 {direction}',font=small,fill='#536373')
            # Orthographic world height .55m means 850/.55 pixels per metre.
            length=round(size*.100/span);x0=pad;y0=size+84
            d.line((x0,y0,x0+length,y0),fill='#283b4d',width=3)
            for x in (x0,x0+length):d.line((x,y0-6,x,y0+6),fill='#283b4d',width=2)
            d.text((x0+length+12,y0-14),'100 mm',font=small,fill='#283b4d')
            # Orientation glyphs match the camera's image plane.
            ox,oy=size-115,size+74
            d.line((ox,oy,ox+52,oy),fill='#52697b',width=2);d.polygon([(ox+52,oy),(ox+43,oy-4),(ox+43,oy+4)],fill='#52697b')
            d.line((ox,oy,ox,oy-40),fill='#52697b',width=2);d.polygon([(ox,oy-40),(ox-4,oy-31),(ox+4,oy-31)],fill='#52697b')
            d.text((ox+57,oy-12),h,font=small,fill='#283b4d');d.text((ox-8,oy-65),v,font=small,fill='#283b4d')
            panel.save(folder/f'{name}.png');panel_images.append(panel)
            metadata[name]={'orthographic':True,'view_direction':direction,'image_axes':[h,v],
                            'vertical_span_m':span,'pose':'same encoder home; static installation pitch -40deg'}
    sheet=Image.new('RGB',(3*size+40,size+235),'#edf1f5');d=ImageDraw.Draw(sheet)
    d.text((35,15),'水平基准安装角修正版 · 三视图',font=title,fill='#142431')
    d.text((36,65),'整机前端抬高约40°（照片估计）｜同一编码器零位｜红边为水平安装基准',font=small,fill='#536373')
    for k,panel in enumerate(panel_images):sheet.paste(panel,(20+k*size,110))
    sheet.save(folder/'three_views.png')
    sheet.save(folder/'three_views.pdf','PDF',resolution=150.)
    (folder/'view_parameters.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(folder/'three_views.png')

if __name__=='__main__':main()
