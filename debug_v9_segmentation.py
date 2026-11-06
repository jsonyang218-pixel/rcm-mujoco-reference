"""Diagnostic: what does the v9 eye_internal segmentation actually see?

Renders with the exact geomgroup mask that VisualServoSensor uses for the
segmentation pass (group 2=0, group 3=1, group 4=1, group 5=0), then
reports every geom that appears in the image, the theoretical lesion pixel
from space.project(), and the depth at that pixel.

Run from the repo root:
    python debug_v9_segmentation.py [--save-png]
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rcm_scene_v9 import RCMSceneV9  # noqa: E402


def _geom_name(model, gid):
    try:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"<geom {gid}>"
    except Exception:
        return f"<geom {gid}>"


def _xml_geom_attrs(xml, name):
    root = ET.fromstring(xml)
    node = next((geom for geom in root.iter("geom") if geom.get("name") == name), None)
    if node is None:
        return None
    return {key: node.get(key) for key in ("fromto", "pos", "size", "quat", "group")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-png", action="store_true",
                        help="Save rgb/seg images to debug_v9_*.png")
    args = parser.parse_args()

    scene = RCMSceneV9()
    mujoco.mj_step(scene.model, scene.data)

    width, height = scene.eye_camera["resolution_px"]
    fovy_deg = float(scene.model.cam_fovy[scene.camera_id("eye_internal")])
    focal_px = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)

    def project_point(point_mm):
        cam_pos, cam_mat = scene.camera_pose("eye_internal")
        camera_point = cam_mat.T @ (np.asarray(point_mm) - cam_pos * 1000.0)
        if abs(camera_point[2]) <= 1e-12:
            return None
        return {
            "pixel": [width / 2.0 - 0.5 + focal_px * camera_point[0] / -camera_point[2],
                      height / 2.0 - 0.5 - focal_px * camera_point[1] / -camera_point[2]],
            "depth_mm": float(-camera_point[2]),
        }

    # --- needle chain source/compiled geometry ---
    print("=" * 60)
    print("Needle chain geometry")
    print("=" * 60)
    body_id = int(scene.model.body("C_translation").id)
    c_world_mm = scene.data.xpos[body_id] * 1000.0
    print(f"C_translation world: {c_world_mm.tolist()}")
    for name in ("needle_shaft", "needle_distal", "needle_distal_tip", "needle_tip_marker"):
        raw = _xml_geom_attrs(scene.scene.xml, name)
        injected = _xml_geom_attrs(scene.xml, name)
        model_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        print(f"\n{name}")
        print(f"  raw DriverModel XML: {raw}")
        print(f"  final injected XML:  {injected}")
        if model_id < 0:
            print("  compiled: MISSING")
            continue
        model_pos_mm = scene.model.geom_pos[model_id] * 1000.0
        world_pos_mm = scene.data.geom_xpos[model_id] * 1000.0
        model_size_mm = scene.model.geom_size[model_id] * 1000.0
        print(f"  compiled local pos mm: {model_pos_mm.tolist()}")
        print(f"  compiled world pos mm: {world_pos_mm.tolist()}")
        print(f"  compiled size mm:      {model_size_mm.tolist()}")
        print(f"  camera projection:     {project_point(world_pos_mm)}")

    # --- render with exact VisualServoSensor segmentation mask ---
    old_znear = scene.model.vis.map.znear
    old_zfar = scene.model.vis.map.zfar
    scene.model.vis.map.znear = scene.eye_camera["znear_mm"] * 0.001 / scene.model.stat.extent
    scene.model.vis.map.zfar  = scene.eye_camera["zfar_mm"]  * 0.001 / scene.model.stat.extent

    option = mujoco.MjvOption()
    option.geomgroup[0] = 0  # mechanical
    option.geomgroup[1] = 0  # housing
    option.geomgroup[2] = 0  # eye shell (now off)
    option.geomgroup[3] = 1  # needle + lesion
    option.geomgroup[4] = 1  # tip marker
    option.geomgroup[5] = 0  # retina backdrop (excluded from seg)
    option.sitegroup[:] = 0  # site markers are not physical image features

    seg_renderer = mujoco.Renderer(scene.model, height=height, width=width)
    seg_renderer.enable_segmentation_rendering()
    seg_renderer.update_scene(scene.data, camera="eye_internal", scene_option=option)
    seg = np.asarray(seg_renderer.render()).copy()

    dep_renderer = mujoco.Renderer(scene.model, height=height, width=width)
    dep_renderer.enable_depth_rendering()
    dep_renderer.update_scene(scene.data, camera="eye_internal", scene_option=option)
    raw_depth = np.asarray(dep_renderer.render()).copy()

    rgb_renderer = mujoco.Renderer(scene.model, height=height, width=width)
    rgb_renderer.update_scene(scene.data, camera="eye_internal", scene_option=option)
    rgb = np.asarray(rgb_renderer.render()).copy()

    scene.model.vis.map.znear = old_znear
    scene.model.vis.map.zfar  = old_zfar

    # --- decode segmentation ---
    obj_id   = seg[:, :, 0].astype(np.int32)
    obj_type = seg[:, :, 1].astype(np.int32)
    geom_mask = obj_type == int(mujoco.mjtObj.mjOBJ_GEOM)

    print("=" * 60)
    print("Geoms visible in segmentation (group 2=off, 3=on, 4=on, 5=off)")
    print("=" * 60)
    geom_pixels: dict[int, int] = {}
    for gid in np.unique(obj_id[geom_mask]):
        px = int((geom_mask & (obj_id == gid)).sum())
        geom_pixels[int(gid)] = px

    for gid, px in sorted(geom_pixels.items(), key=lambda kv: -kv[1]):
        name  = _geom_name(scene.model, gid)
        group = int(scene.model.geom_group[gid]) if gid < scene.model.ngeom else -1
        print(f"  geom {gid:4d}  group {group}  {px:7d} px  {name}")

    # --- lesion_marker specifically ---
    lesion_id = int(mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_marker"))
    lesion_px = geom_pixels.get(lesion_id, 0)
    print(f"\nlesion_marker  id={lesion_id}  group={int(scene.model.geom_group[lesion_id])}  "
          f"pixels={lesion_px}")

    # --- theoretical projection ---
    lesion_world_mm = scene.site_mm("lesion")
    cam_pos_m, cam_mat = scene.camera_pose("eye_internal")
    cam_pos_mm = cam_pos_m * 1000.0
    p = cam_mat.T @ (lesion_world_mm - cam_pos_mm)
    if abs(p[2]) > 1e-9:
        proj_u = width  / 2.0 - 0.5 + focal_px * p[0] / -p[2]
        proj_v = height / 2.0 - 0.5 - focal_px * p[1] / -p[2]
        proj_depth_mm = float(-p[2])
    else:
        proj_u = proj_v = proj_depth_mm = float("nan")

    print(f"\nTheoretical lesion projection:  u={proj_u:.2f}  v={proj_v:.2f}  depth={proj_depth_mm:.3f} mm")

    pu, pv = int(round(proj_u)), int(round(proj_v))
    if 0 <= pv < height and 0 <= pu < width:
        hit_id   = int(obj_id[pv, pu])
        hit_type = int(obj_type[pv, pu])
        hit_name = (_geom_name(scene.model, hit_id) if hit_type == int(mujoco.mjtObj.mjOBJ_GEOM)
                    else f"type={hit_type}")
        raw = float(raw_depth[pv, pu])
        # Renderer depth output is already metric (metres), as consumed by
        # NeedleGeometry.points(). Applying the reverse-Z decode again would
        # turn ~40.7 mm into a bogus sub-millimetre value.
        actual_depth_m = raw
        print(f"Pixel ({pu},{pv}) in seg image:  id={hit_id}  {hit_name}")
        print(f"Depth at ({pu},{pv}):  raw={raw:.6f}  actual={actual_depth_m*1000:.3f} mm  "
              f"(expected {proj_depth_mm:.3f} mm)")
    else:
        print(f"Projected pixel ({pu},{pv}) is OUTSIDE the {width}x{height} image")

    # --- check the 9x9 neighbourhood around the projected pixel ---
    print(f"\nNeighbourhood ({max(0,pv-4)}:{min(height,pv+5)}, {max(0,pu-4)}:{min(width,pu+5)}):")
    region_ids = obj_id[max(0, pv-4):min(height, pv+5), max(0, pu-4):min(width, pu+5)]
    region_types = obj_type[max(0, pv-4):min(height, pv+5), max(0, pu-4):min(width, pu+5)]
    uniq, counts = np.unique(region_ids[region_types == int(mujoco.mjtObj.mjOBJ_GEOM)], return_counts=True)
    for uid, cnt in zip(uniq.tolist(), counts.tolist()):
        print(f"  geom {uid:4d}  {cnt:3d} px  {_geom_name(scene.model, uid)}  "
              f"group={int(scene.model.geom_group[uid]) if uid < scene.model.ngeom else -1}")

    # --- per-group total ---
    print("\nTotal pixels per geomgroup in segmentation:")
    for g in range(6):
        ids = np.flatnonzero(scene.model.geom_group == g).astype(np.int32)
        px = int((geom_mask & np.isin(obj_id, ids)).sum())
        print(f"  group {g}: {px} px")

    # --- model extent and znear/zfar sanity ---
    print(f"\nmodel.stat.extent = {scene.model.stat.extent:.6f} m")
    print(f"eye znear={scene.eye_camera['znear_mm']} mm  zfar={scene.eye_camera['zfar_mm']} mm")
    print(f"normalised znear={scene.eye_camera['znear_mm']*0.001/scene.model.stat.extent:.6g}  "
          f"zfar={scene.eye_camera['zfar_mm']*0.001/scene.model.stat.extent:.6g}")

    if args.save_png:
        try:
            from PIL import Image
            Image.fromarray(rgb.astype(np.uint8)).save("debug_v9_rgb.png")
            # colour-code segmentation by geom group
            seg_vis = np.zeros((*obj_id.shape, 3), dtype=np.uint8)
            colours = [(220,220,220),(100,180,255),(255,200,80),(255,80,80),(80,255,80),(180,80,255)]
            for g, col in enumerate(colours):
                ids = np.flatnonzero(scene.model.geom_group == g).astype(np.int32)
                mask = geom_mask & np.isin(obj_id, ids)
                seg_vis[mask] = col
            if 0 <= pv < height and 0 <= pu < width:
                seg_vis[max(0,pv-3):min(height,pv+4), max(0,pu-3):min(width,pu+4)] = (255, 255, 0)
            Image.fromarray(seg_vis).save("debug_v9_seg.png")
            print("\nSaved debug_v9_rgb.png and debug_v9_seg.png")
            print("  Yellow cross = theoretical lesion projection")
        except ImportError:
            print("\n(Pillow not installed — skipping PNG save)")

    seg_renderer.close()
    dep_renderer.close()
    rgb_renderer.close()


if __name__ == "__main__":
    main()
