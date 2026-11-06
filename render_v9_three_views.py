#!/usr/bin/env python3
"""Render the current injected v9 model in three world-axis orthographic views.

The separate front detail looks along the photo tool's local +X direction.
It is a labelled part view, not one of the world-coordinate engineering views.
The renderer never changes the model's installation, geometry or task pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/rcm-mesa')

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rcm_scene_v9 import RCMSceneV9

ROOT = Path(__file__).resolve().parent
VIEWER_GROUPS = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)
INK = '#eef2f5'
MUTED = '#a9b8c8'
BACKGROUND = '#10161d'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    candidates = [os.environ.get('MODEL_FONT_PATH', ''),
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  'C:/Windows/Fonts/msyh.ttc',
                  '/System/Library/Fonts/PingFang.ttc',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    path = next((p for p in candidates if p and Path(p).is_file()), None)
    if path is None:
        raise FileNotFoundError('Set MODEL_FONT_PATH to a local Chinese .ttf/.ttc font')
    return ImageFont.truetype(path, size)


class GeometryBounds:
    """Support intervals from actual compiled mesh vertices and primitives."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.mesh_world = {}

    def mesh_vertices(self, gid):
        if gid not in self.mesh_world:
            mid = int(self.model.geom_dataid[gid])
            start = int(self.model.mesh_vertadr[mid])
            vertices = self.model.mesh_vert[start:start + self.model.mesh_vertnum[mid]]
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            self.mesh_world[gid] = vertices @ rotation.T + self.data.geom_xpos[gid]
        return self.mesh_world[gid]

    def interval(self, gid, direction):
        model, data = self.model, self.data
        kind = model.geom_type[gid]
        center = data.geom_xpos[gid]
        rotation = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            projection = self.mesh_vertices(gid) @ direction
            return float(projection.min()), float(projection.max())
        local_direction = rotation.T @ direction
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            radius = np.abs(local_direction) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radius = np.linalg.norm(local_direction * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius = size[0] * np.linalg.norm(local_direction[:2]) + size[1] * abs(local_direction[2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = size[0] + size[1] * abs(local_direction[2])
        else:
            raise ValueError(f'Unsupported visible bounding geometry type {kind}, id={gid}')
        projection = float(center @ direction)
        return projection - float(radius), projection + float(radius)

    def bounds(self, ids, basis):
        intervals = np.array([[self.interval(gid, direction) for gid in ids]
                              for direction in np.asarray(basis).T])
        return intervals[:, :, 0].min(axis=1), intervals[:, :, 1].max(axis=1)


def long_axis_xy(vertices):
    """Minimum-area rectangle axis, using a 2-D convex hull without shapely."""
    points = sorted(set(map(tuple, np.asarray(vertices[:, :2], dtype=float))))
    if len(points) < 3:
        raise ValueError('Cover has no nondegenerate XY projection')

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    best = None
    for edge in np.roll(hull, -1, axis=0) - hull:
        along = edge / np.linalg.norm(edge)
        across = np.array([-along[1], along[0]])
        extent = np.ptp(hull @ np.column_stack([along, across]), axis=0)
        area = float(np.prod(extent))
        if best is None or area < best[0]:
            best = (area, along if extent[0] >= extent[1] else across)
    axis = best[1]
    if axis[0] < 0:
        axis = -axis
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def camera(root, name, center, right, up, normal, span):
    rotation = np.column_stack([right, up, normal])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(rotation) - 1) < 1e-10
    position = center + normal * max(1., span * 3)
    ET.SubElement(root.find('worldbody'), 'camera', name=name,
                  pos=' '.join(map(str, position)),
                  xyaxes=' '.join(map(str, np.r_[right, up])),
                  orthographic='true', fovy=str(span))
    return {'position_world_m': position.tolist(), 'center_world_m': center.tolist(),
            'image_right_world': right.tolist(), 'image_up_world': up.tolist(),
            'view_direction_world': (-normal).tolist(),
            'rotation_determinant': float(np.linalg.det(rotation)),
            'orthographic': True, 'vertical_span_m': span}


def panel(image, label, direction, axes, span, size):
    scale = size / 1000.
    top = max(52, round(68 * scale))
    bottom = max(58, round(72 * scale))
    pad = max(16, round(26 * scale))
    result = Image.new('RGB', (size, size + top + bottom), BACKGROUND)
    result.paste(image, (0, top))
    draw = ImageDraw.Draw(result)
    draw.text((pad, 4), label, font=font(max(20, round(32 * scale))), fill=INK)
    draw.text((pad, max(29, round(39 * scale))), direction,
              font=font(max(13, round(19 * scale))), fill=MUTED)
    bar_mm = next(value for value in (100, 50, 20, 10, 5)
                  if value / (span * 1000) < .30)
    length = size * bar_mm / (span * 1000)
    y = top + size + bottom // 2
    draw.line((pad, y, pad + length, y), fill=INK, width=3)
    for x in (pad, pad + length):
        draw.line((x, y - 5, x, y + 5), fill=INK, width=2)
    small = font(max(13, round(19 * scale)))
    draw.text((pad + length + 10, y - 14), f'{bar_mm} mm', font=small, fill=INK)
    draw.text((size - max(200, round(285 * scale)), y - 14),
              f'横 {axes[0]} / 纵 {axes[1]}', font=small, fill=MUTED)
    return result


def render(output_dir, size=1000):
    if size < 400:
        raise ValueError('--size must be at least 400')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = RCMSceneV9()
    try:
        np.testing.assert_allclose(scene.current_task(), np.zeros(3), atol=1e-12)
        root = ET.fromstring(scene.xml)
        root.find('visual/global').set('offwidth', str(max(size, int(root.find('visual/global').get('offwidth', '0')))))
        root.find('visual/global').set('offheight', str(max(size, int(root.find('visual/global').get('offheight', '0')))))
        headlight = root.find('visual/headlight')
        if headlight is not None:
            headlight.set('ambient', '.45 .45 .45')
            headlight.set('diffuse', '.55 .55 .55')
        asset = root.find('asset')
        for texture in list(asset.findall("texture[@type='skybox']")):
            asset.remove(texture)
        ET.SubElement(asset, 'texture', name='three_views_background', type='skybox',
                      builtin='gradient', rgb1='.04 .055 .07', rgb2='.04 .055 .07',
                      width='512', height='3072')

        names = [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
                 for i in range(scene.model.ngeom)]
        hidden_reference = [i for i, name in enumerate(names) if name.startswith('rcm_cross_')]
        visible = [i for i in range(scene.model.ngeom)
                   if VIEWER_GROUPS[int(scene.model.geom_group[i])] and i not in hidden_reference]
        bounds = GeometryBounds(scene.model, scene.data)
        lower, upper = bounds.bounds(visible, np.eye(3))
        center = (lower + upper) * .5
        span = float((upper - lower).max() * 1.12)
        if not np.isfinite(span) or span <= 0:
            raise ValueError('Invalid visible model extent')
        specifications = [
            ('front', '正视图', '沿世界 +X 看 −X', '+Y', '+Z',
             np.array([0., 1., 0.]), np.array([0., 0., 1.]), np.array([1., 0., 0.])),
            ('side', '侧视图', '沿世界 −Y 看 +Y', '+X', '+Z',
             np.array([1., 0., 0.]), np.array([0., 0., 1.]), np.array([0., -1., 0.])),
            ('top', '俯视图', '沿世界 +Z 看 −Z', '+X', '+Y',
             np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])),
        ]
        normals = np.column_stack([item[-1] for item in specifications])
        np.testing.assert_allclose(normals.T @ normals, np.eye(3), atol=1e-12)
        views = {}
        for name, label, direction, h, v, right, up, normal in specifications:
            views[name] = camera(root, 'three_views_' + name, center, right, up, normal, span)
            lo, hi = bounds.bounds(visible, np.column_stack([right, up, normal]))
            projected_center = np.column_stack([right, up, normal]).T @ center
            margin = min(float((lo[:2] - projected_center[:2] + span / 2).min()),
                         float((span / 2 - hi[:2] + projected_center[:2]).min()))
            assert margin > 0, (name, margin)
            views[name].update(image_axes=[h, v], mm_per_pixel=span * 1000 / size,
                               framing_margin_px=margin * size / span,
                               all_visible_geometry_inside_frame=True)

        mount = scene.model.body('needle_mount').id
        tool_rotation = scene.data.xmat[mount].reshape(3, 3).copy()
        detail_ids = [i for i, name in enumerate(names)
                      if name == 'scan_C' or name.startswith(('tool_', 'needle_'))]
        detail_lower, detail_upper = bounds.bounds(detail_ids, tool_rotation)
        detail_center = tool_rotation @ ((detail_lower + detail_upper) * .5)
        detail_span = float(max((detail_upper - detail_lower)[1:]) * 1.12)
        views['front_detail'] = camera(root, 'three_views_front_detail', detail_center,
                                       tool_rotation[:, 1], tool_rotation[:, 2],
                                       tool_rotation[:, 0], detail_span)
        views['front_detail'].update(
            scope='Labelled front-tool part view, not a world-axis engineering view',
            image_axes=['tool +Y', 'tool +Z'], mm_per_pixel=detail_span * 1000 / size,
            included_geoms=[names[i] for i in detail_ids],
            whole_robot=False)
        assembly_ids = sorted(set(detail_ids) | {scene.model.geom('scan_distal').id})
        assembly_lower, assembly_upper = bounds.bounds(assembly_ids, tool_rotation)
        assembly_center = tool_rotation @ ((assembly_lower + assembly_upper) * .5)
        assembly_span = float(max((assembly_upper - assembly_lower)[1:]) * 1.12)
        views['front_assembly'] = camera(root, 'three_views_front_assembly', assembly_center,
                                         tool_rotation[:, 1], tool_rotation[:, 2],
                                         tool_rotation[:, 0], assembly_span)
        views['front_assembly'].update(
            scope='Labelled front-assembly part view including the silver cap, not a complete robot view',
            image_axes=['tool +Y', 'tool +Z'], mm_per_pixel=assembly_span * 1000 / size,
            included_geoms=[names[i] for i in assembly_ids],
            whole_robot=False)

        display = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        data = mujoco.MjData(display)
        data.qpos[:] = scene.data.qpos
        mujoco.mj_forward(display, data)
        for i in hidden_reference:
            display.geom_group[i] = 1
        option = mujoco.MjvOption()
        option.geomgroup[:] = VIEWER_GROUPS
        option.sitegroup[:] = 0
        panels = []
        with mujoco.Renderer(display, height=size, width=size) as renderer:
            for name, label, direction, h, v, *_ in specifications:
                renderer.update_scene(data, camera='three_views_' + name, scene_option=option)
                image = Image.fromarray(renderer.render().copy())
                part = panel(image, label, direction, [h, v], span, size)
                part.save(output_dir / (name + '.png'))
                panels.append(part)
            # A separate, clearly labelled part view can omit the surrounding
            # scan and base. The complete world-axis views above omit neither.
            display.geom_group[:] = 1
            display.geom_group[detail_ids] = 0
            option.geomgroup[:] = [1, 0, 0, 0, 0, 0]
            renderer.update_scene(data, camera='three_views_front_detail', scene_option=option)
            detail = Image.fromarray(renderer.render().copy())
            panel(detail, '前端局部正面核对', '沿针座局部 +X；仅显示白壳、夹具、针座和针具',
                  ['局部 +Y', '局部 +Z'], detail_span, size).save(output_dir / 'front_detail.png')
            display.geom_group[:] = 1
            display.geom_group[assembly_ids] = 0
            renderer.update_scene(data, camera='three_views_front_assembly', scene_option=option)
            assembly = Image.fromarray(renderer.render().copy())
            panel(assembly, '前端组件局部正面', '沿针座局部 +X；含银帽、白壳、导轨、套管和针具',
                  ['局部 +Y', '局部 +Z'], assembly_span, size).save(output_dir / 'front_assembly.png')

        pad = max(12, round(size * .02))
        header = max(105, round(size * .13))
        sheet = Image.new('RGB', (size * 3 + pad * 4, panels[0].height + header + pad), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text((pad, 10), '当前 v9 模型 · 完整世界坐标三视图',
                  font=font(max(25, round(size * .04))), fill=INK)
        draw.text((pad, max(49, round(size * .063))),
                  f'同一零位 A=0° / B=0° / C=0 mm · 正交投影 · 统一比例 {span * 1000 / size:.3f} mm/px',
                  font=font(max(16, round(size * .022))), fill=MUTED)
        draw.text((pad, max(76, round(size * .096))),
                  '整机保留安装姿态；针具未放大；前端照片方向另见 front_detail.png',
                  font=font(max(13, round(size * .018))), fill=MUTED)
        for i, image in enumerate(panels):
            sheet.paste(image, (pad + i * (size + pad), header))
        sheet.save(output_dir / 'three_views.png')
        sheet.save(output_dir / 'three_views.pdf', 'PDF', resolution=150.)

        needle = scene.site_mm('needle_tip') - scene.site_mm('driver_junction')
        needle_top_angle = float(np.degrees(np.arctan2(needle[1], needle[0])))
        needle_side_pitch = float(np.degrees(np.arctan2(-needle[2], np.linalg.norm(needle[:2]))))
        cover_angle = long_axis_xy(bounds.mesh_vertices(scene.model.geom('scan_parallel_cover').id))
        expected_pitch = float(scene.side_profile['needle_pitch_from_horizontal_deg'])
        assert abs(needle_top_angle) < .01, needle_top_angle
        assert abs(cover_angle) < .01, cover_angle
        assert abs(needle_side_pitch - expected_pitch) < .01, needle_side_pitch
        for part in ('needle_proximal', 'needle_distal', 'needle_distal_tip'):
            physical = scene.model.geom(part)
            visual = scene.model.geom(part.replace('needle_', 'needle_display_', 1))
            np.testing.assert_array_equal(physical.size, visual.size)
            np.testing.assert_array_equal(physical.pos, visual.pos)

        source_names = ['render_v9_three_views.py', 'rcm_scene_v9.py', 'v9_front_tool.py',
                        'v9_alignment.json', 'v9_tool_mount.json', 'v9_side_profile.json', 'config.json']
        compiler = ET.fromstring(scene.xml).find('compiler')
        mesh_dir = Path(compiler.get('meshdir'))
        assets = {}
        for node in ET.fromstring(scene.xml).findall('./asset/mesh[@file]'):
            path = mesh_dir / node.get('file')
            assets[str(path.relative_to(ROOT))] = sha256(path)
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                             stderr=subprocess.DEVNULL, text=True).strip()
            tree_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                                  stderr=subprocess.DEVNULL, text=True)
            source_tree_dirty = bool(tree_status.strip())
        except (OSError, subprocess.CalledProcessError):
            commit = None
            tree_status = None
            source_tree_dirty = None
        metadata = {
            'result': 'PASS', 'scene': 'RCMSceneV9', 'source_commit': commit,
            'source_tree_dirty': source_tree_dirty,
            'source_commit_is_baseline': source_tree_dirty if commit is not None else None,
            'source_tree_status_porcelain': tree_status.splitlines() if tree_status is not None else None,
            'source_files_sha256': {name: sha256(ROOT / name) for name in source_names},
            'mesh_assets_sha256': assets,
            'runtime_xml_sha256': hashlib.sha256(scene.xml.encode()).hexdigest(),
            'task_A_deg_B_deg_C_mm': [0., 0., 0.], 'qpos': scene.data.qpos.tolist(),
            'whole_robot_geoms_not_rearranged_or_hidden': True,
            'viewer_geom_groups': VIEWER_GROUPS.tolist(),
            'hidden_reference_geoms': [names[i] for i in hidden_reference],
            'metrology_sites_visible': False,
            'render_copy_headlight': {'ambient': [.45, .45, .45], 'diffuse': [.55, .55, .55]},
            'render_copy_skybox_rgb': [.04, .055, .07],
            'visible_geom_count': len(visible), 'visible_geom_names': [names[i] for i in visible],
            'world_aabb_m': {'lower': lower.tolist(), 'upper': upper.tolist()},
            'shared_main_view_span_m': span, 'shared_main_view_mm_per_pixel': span * 1000 / size,
            'main_camera_normals_orthogonal': True, 'views': views,
            'needle_display_geometry_same_size_as_physical': True,
            'needle_dimensions_mm': {key: scene.config['needle'][key] for key in
                                     ('proximal_length_mm', 'proximal_radius_mm',
                                      'distal_length_mm', 'distal_radius_mm')},
            'axis_checks': {'result': 'PASS', 'top_needle_angle_deg': needle_top_angle,
                            'top_cover_long_axis_angle_deg': cover_angle,
                            'side_needle_downward_pitch_deg': needle_side_pitch,
                            'expected_side_needle_downward_pitch_deg': expected_pitch},
            'scope': 'Geometric orthographic rendering; dimensions other than configured needle remain photo estimates',
        }
        (output_dir / 'view_parameters.json').write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(output_dir / 'three_views.png')
        print(json.dumps({'result': 'PASS', 'main_view_mm_per_pixel': metadata['shared_main_view_mm_per_pixel'],
                          'axis_checks': metadata['axis_checks']}, ensure_ascii=False, indent=2))
        return metadata
    finally:
        scene.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=1000,
                        help='Square render size per engineering view; all main panels share scale')
    args = parser.parse_args(argv)
    render(args.output_dir, args.size)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
