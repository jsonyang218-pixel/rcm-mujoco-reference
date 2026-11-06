"""Orthographic close-ups of the current scanned needle assembly."""
from pathlib import Path
import argparse
import os
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', str(Path(__file__).parent/'.cache'))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model_sync import DriverModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--front-mesh-dir', type=Path, help='Preview candidate *_symmetric.obj meshes')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = DriverModel()
    root = ET.fromstring(scene.xml)
    if args.front_mesh_dir:
        for name in ('distal_guide', 'C_carriage'):
            root.find(f"./asset/mesh[@name='{name}']").set(
                'file', str((args.front_mesh_dir/f'{name}_symmetric.obj').resolve()))
    ET.SubElement(root.find('asset'), 'texture', name='detail_black', type='skybox',
                  builtin='gradient', rgb1='0 0 0', rgb2='0 0 0', width='512', height='3072')
    root.find('visual/headlight').set('ambient', '.45 .45 .45')
    body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, 'distal_orientation')
    rotation = scene.data.xmat[body].reshape(3, 3)
    translation = scene.data.xpos[body]
    normal = np.array([.953, .302, 0.]); normal /= np.linalg.norm(normal)
    across = np.array([-normal[1], normal[0], 0.])
    center = rotation @ np.array([0., -.025, .082]) + translation
    up = rotation @ np.array([0., 0., 1.])
    views = [('front', '针座正面', -across), ('side', '针座侧面', normal),
             ('oblique', '斜侧近景', normal*.65-across*.76)]
    for name, _, direction in views:
        direction = rotation @ direction
        forward = -direction/np.linalg.norm(direction)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        ET.SubElement(root.find('worldbody'), 'camera', name='detail_'+name,
                      pos=' '.join(map(str, center+direction*.6)),
                      xyaxes=' '.join(map(str, np.r_[right, vertical])),
                      orthographic='true', fovy='.16')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model); data.qpos[:] = scene.data.qpos
    mujoco.mj_forward(model, data)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
        if name not in ('scan_distal', 'scan_C') and not name.startswith('needle_'):
            model.geom_group[i] = 5
    options = mujoco.MjvOption(); options.sitegroup[:] = 0; options.geomgroup[5] = 0
    fonts = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
             '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    font = ImageFont.truetype(next(p for p in fonts if Path(p).exists()), 26)
    size = 850
    sheet = Image.new('RGB', (size*3, size+55), 'black'); draw = ImageDraw.Draw(sheet)
    with mujoco.Renderer(model, height=size, width=size) as renderer:
        for k, (name, label, _) in enumerate(views):
            renderer.update_scene(data, camera='detail_'+name, scene_option=options)
            im = Image.fromarray(renderer.render().copy())
            im.save(args.output_dir/f'{name}.png')
            sheet.paste(im, (k*size, 55))
            draw.text((k*size+25, 12), label, font=font, fill='white')
    sheet.save(args.output_dir/'needle_holder_detail.png')
    print(args.output_dir/'needle_holder_detail.png')


if __name__ == '__main__':
    main()
