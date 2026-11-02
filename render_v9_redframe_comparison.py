"""Render matched side-detail cameras from retained baseline and current MJCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/rcm-mesa')
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from rcm_scene_v9 import RCMSceneV9
from render_v9_three_views import GeometryBounds, camera, font

ROOT = Path(__file__).resolve().parent


def render(baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    scene = RCMSceneV9()
    try:
        qpos = scene.data.qpos.copy()
        models = []
        for xml in (baseline.read_text(), scene.xml):
            root = ET.fromstring(xml)
            root.find('compiler').set('meshdir', str(ROOT/'mujoco_model/assets'))
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
            data = mujoco.MjData(model); data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            ids = [i for i in range(model.ngeom) if model.geom(i).name == 'scan_distal'
                   or model.geom(i).name == 'scan_C'
                   or model.geom(i).name.startswith(('tool_', 'needle_display_', 'redframe_'))]
            models.append((root, model, data, ids))
        right, up, normal = np.eye(3)[0], np.eye(3)[2], -np.eye(3)[1]
        lower, upper = [], []
        for _, model, data, ids in models:
            lo, hi = GeometryBounds(model, data).bounds(ids, np.eye(3))
            lower.append(lo); upper.append(hi)
        lo, hi = np.min(lower, axis=0), np.max(upper, axis=0)
        center = (lo+hi)/2
        span = float(max((hi-lo)[[0,2]])*1.12)
        images = []
        metadata = {}
        for label, (root, _, _, ids) in zip(('before', 'after'), models):
            metadata[label] = camera(root, 'redframe_comparison', center, right, up, normal, span)
            root.find('visual/global').set('offwidth', '1200')
            root.find('visual/global').set('offheight', '1200')
            model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
            data = mujoco.MjData(model);data.qpos[:]=qpos;mujoco.mj_forward(model,data)
            model.geom_group[:] = 1;model.geom_group[ids] = 0
            opt = mujoco.MjvOption();opt.geomgroup[:]=[1,0,0,0,0,0];opt.sitegroup[:]=0
            with mujoco.Renderer(model, height=1200, width=1200) as renderer:
                renderer.update_scene(data, camera='redframe_comparison', scene_option=opt)
                image = Image.fromarray(renderer.render().copy())
            image.save(output/(label+'_side_detail.png'))
            images.append(image)
        sheet = Image.new('RGB',(2400,1280),'#10161d');draw=ImageDraw.Draw(sheet)
        for i, (image, title) in enumerate(zip(images, ('Before: ffd7b15', 'After: local red-frame geometry'))):
            sheet.paste(image,(1200*i,80));draw.text((1200*i+24,20),title,font=font(32),fill='white')
        sheet.save(output/'local_before_after.png')
        metadata.update(scope='Matched world side view; isolated front assembly, unscaled physical needle',
                        mm_per_pixel=span*1000/1200,
                        baseline_xml_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest())
        (output/'comparison_camera.json').write_text(json.dumps(metadata,indent=2)+'\n')
    finally:
        scene.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-xml',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();render(args.baseline_xml,args.output_dir)
