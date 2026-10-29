"""Reference-compatible adapter backed by the v9 scanned model.

The legacy research modules speak to ``rcm_reference.Reference``.  This class
keeps that public surface while driving ``mujoco_model.model_sync.DriverModel``.
Kinematic/encoder math is shared between the two models, so task<->encoder
conversion is delegated to a hidden legacy ``Reference``; the v9 world frame and
its compiled scene (with the eye scene injected) are the rendering backend.
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from rcm_reference import Reference, _vec

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mujoco_model"))


def _qvec(values, length):
    values = np.asarray(values, dtype=float)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"expected {length} finite numbers")
    return values


def _s(values):
    return " ".join(f"{float(v):.15g}" for v in values)


def _quat_from_matrix(matrix):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=float).reshape(9))
    return quat


def _camera_axes(pos, target):
    z = np.asarray(pos, dtype=float) - np.asarray(target, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(up @ z)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.r_[x, np.cross(z, x)]


class RCMSceneV9:
    def __init__(self, config=None, model_dir=None):
        self.config = config if config is not None else json.loads(
            (ROOT / "config.json").read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported config schema")
        self._ref = Reference(self.config)
        from model_sync import DriverModel
        self.scene = DriverModel(model_dir=model_dir, needle_profile="legacy")
        self.eye_camera = self._ref.eye_camera
        self.rcm_mm = self.scene.rcm_world * 1000.0
        self.R_robot_world = self.scene.robot_to_world
        self._build_injected_model()
        self.set_task(np.zeros(3))

    def _build_injected_model(self):
        root = ET.fromstring(self.scene.xml)
        self._inject_assets(root)
        self._inject_eye_scene(root)
        self._inject_needle_sensor_geoms(root)
        ET.indent(root, space="  ")
        self.xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.sites = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                      for name in ("rcm_site", "needle_tip", "driver_junction",
                                   "needle_base", "lesion", "eye_camera_site",
                                   "front_upper", "front_lower",
                                   "upper_bridge_end", "lower_bridge_end")}
        for name, sid in self.sites.items():
            if sid == -1:
                raise ValueError(f"site '{name}' not found in injected v9 model")
        self._renderers = {}
        width, height = self.eye_camera["resolution_px"]
        for mode in ("rgb", "depth", "segmentation"):
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            if mode == "depth":
                renderer.enable_depth_rendering()
            elif mode == "segmentation":
                renderer.enable_segmentation_rendering()
            self._renderers[mode] = renderer

    def _inject_assets(self, root):
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        existing = {material.get("name") for material in asset.findall("material")}
        for name, rgba in {
            "eye": ".65 .80 .90 .10", "pupil": ".05 .08 .12 .85",
            "lens": ".70 .85 1.0 .16", "retina": ".78 .30 .34 1",
            "lesion": ".95 .37 .12 1", "tip_marker": ".15 1 .25 1",
            "display_target": "1 .88 .05 1", "camera_marker": ".05 .95 .95 1",
            "needle": ".80 .84 .90 1",
        }.items():
            if name not in existing:
                ET.SubElement(asset, "material", name=name, rgba=rgba,
                              specular="0.4", shininess="0.45")

    def _to_world_m(self, point_robot_mm):
        return self.scene.rcm_world + self.R_robot_world @ (np.asarray(point_robot_mm, dtype=float) * 0.001)

    def _inject_eye_scene(self, root):
        global_vis = root.find("./visual/global")
        if global_vis is not None:
            width, height = self.eye_camera["resolution_px"]
            global_vis.set("offwidth", str(max(int(global_vis.get("offwidth", "1280")), width)))
            global_vis.set("offheight", str(max(int(global_vis.get("offheight", "960")), height)))
        world = root.find("worldbody")
        eye = self.config["reference_scene"]
        eye_center = self._to_world_m(eye["eye_center_robot_mm"])
        radii = np.asarray(eye["eye_radii_robot_mm"], dtype=float) * 0.001
        eye_quat = _qvec(eye.get("eye_quat_robot_wxyz", [1, 0, 0, 0]), 4)
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, eye_quat)
        world_quat = _quat_from_matrix(self.R_robot_world @ mat.reshape(3, 3))
        ET.SubElement(world, "geom", name="eye_reference", type="ellipsoid",
                      pos=_s(eye_center), size=_s(radii), quat=_s(world_quat),
                      material="eye", group="2")

        target = self._to_world_m(eye["target_robot_mm"])
        ET.SubElement(world, "geom", name="lesion_marker", type="sphere",
                      pos=_s(target), size=".00015", material="lesion", group="3")
        ET.SubElement(world, "site", name="lesion", pos=_s(target))
        display = float(eye.get("display_target_radius_mm", 0.5))
        ET.SubElement(world, "geom", name="lesion_display_marker", type="sphere",
                      pos=_s(target), size=str(display * 0.001),
                      material="display_target", group="2")

        camera = self.eye_camera
        camera_pos = self._to_world_m(camera["position_robot_mm"])
        camera_target_robot = np.asarray(camera["target_robot_mm"], dtype=float) * 0.001
        camera_pos_robot = np.asarray(camera["position_robot_mm"], dtype=float) * 0.001
        local_xy = _camera_axes(camera_pos_robot, camera_target_robot)
        xyaxes = np.r_[self.R_robot_world @ local_xy[:3],
                       self.R_robot_world @ local_xy[3:]]
        ET.SubElement(world, "camera", name="eye_internal", pos=_s(camera_pos),
                      xyaxes=_s(xyaxes),
                      fovy=str(camera["fovy_deg"]))
        ET.SubElement(world, "site", name="eye_camera_site", pos=_s(camera_pos),
                      size="0.0009", rgba="0.05 0.95 0.95 1", group="5")

    def _inject_needle_sensor_geoms(self, root):
        body = next((b for b in root.iter("body") if b.get("name") == "C_translation"), None)
        if body is None:
            raise ValueError("v9 model XML missing body 'C_translation'; cannot inject needle sensor geoms")
        needle = self.config["needle"]
        tip_z = needle["distal_length_mm"] * 0.001
        taper_start = tip_z + needle["distal_tip_length_mm"] * 0.001
        for geom in body.findall("geom"):
            name = geom.get("name")
            if name == "needle_shaft":
                geom.set("name", "needle_proximal")
                geom.set("group", "3")
            elif name == "needle_distal":
                geom.set("group", "3")
        ET.SubElement(body, "geom", name="needle_distal_tip", type="capsule",
                      fromto=f"0 0 {taper_start:.15g} 0 0 {tip_z:.15g}",
                      size=f"{needle['distal_radius_mm'] * 0.001:.15g}",
                      material="needle", group="3")
        ET.SubElement(body, "geom", name="needle_tip_marker", type="sphere",
                      pos=f"0 0 {tip_z:.15g}",
                      size=f"{needle['tip_marker_radius_mm'] * 0.001:.15g}",
                      material="tip_marker", group="4")

    def build_xml(self):
        return self.xml

    def state_from_task(self, q):
        return self._ref.state_from_task(q)

    def task_from_state(self, state):
        return self._ref.task_from_state(state)

    def validate_task(self, q):
        return self._ref.validate_task(q)

    def set_task(self, q):
        q = self._ref.validate_task(q)
        self._ref.set_task(q)
        robot = tuple(int(round(float(v))) for v in self._ref.encoders(q)) + (0,)
        self.scene.sync(robot)
        self.data.qpos[:] = self.scene.data.qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def current_task(self):
        return self._ref.current_task()

    def encoders(self, q=None):
        return self._ref.encoders(q)

    def task_from_encoders(self, values, reference_task=None):
        return self._ref.task_from_encoders(values, reference_task)

    def robot_to_world(self, p_mm):
        return self.rcm_mm + self.R_robot_world @ _vec(p_mm)

    def world_to_robot(self, p_mm):
        return self.R_robot_world.T @ (_vec(p_mm) - self.rcm_mm)

    def set_target_world(self, point_mm):
        robot_mm = self.world_to_robot(point_mm)
        target_m = _vec(point_mm) * 0.001
        marker = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_marker")
        display = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_display_marker")
        self.model.geom_pos[marker] = target_m
        self.model.geom_pos[display] = target_m
        self.model.site_pos[self.sites["lesion"]] = target_m
        mujoco.mj_forward(self.model, self.data)
        # _ref.target_task() reads legacy-world target, so re-project the shared
        # robot-frame point through the legacy install transform rather than v9's.
        self._ref.set_target_world(self._ref.robot_to_world(robot_mm))

    def target_task(self, validate=True):
        return self._ref.target_task(validate)

    def site_mm(self, name):
        return self.data.site_xpos[self.sites[name]] * 1000.0

    def camera_id(self, name="eye_internal"):
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name))

    def render_camera(self, name="eye_internal", mode="rgb"):
        if mode not in ("rgb", "depth", "segmentation"):
            raise ValueError("mode must be rgb, depth, or segmentation")
        width, height = self.eye_camera["resolution_px"]
        old_znear, old_zfar = self.model.vis.map.znear, self.model.vis.map.zfar
        self.model.vis.map.znear = self.eye_camera["znear_mm"] * 0.001 / self.model.stat.extent
        self.model.vis.map.zfar = self.eye_camera["zfar_mm"] * 0.001 / self.model.stat.extent
        try:
            option = mujoco.MjvOption()
            option.geomgroup[0] = 0
            option.geomgroup[1] = 0
            option.geomgroup[2] = 0
            option.geomgroup[3] = 1
            option.geomgroup[4] = 1
            option.geomgroup[5] = 1
            self._renderers[mode].update_scene(self.data, camera=name, scene_option=option)
            return np.asarray(self._renderers[mode].render()).copy()
        finally:
            self.model.vis.map.znear, self.model.vis.map.zfar = old_znear, old_zfar

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()

    def camera_pose(self, name="eye_internal"):
        cid = self.camera_id(name)
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def needle_eye_geometry(self):
        tip, junction, rcm = [self.site_mm(name) for name in
                              ("needle_tip", "driver_junction", "rcm_site")]
        back = self.site_mm("needle_base")
        axis = tip - back
        axis /= np.linalg.norm(axis)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
        rotation = self.data.geom_xmat[gid].reshape(3, 3)
        center = self.data.geom_xpos[gid] * 1000.0
        radii = self.model.geom_size[gid] * 1000.0
        origin = rotation.T @ (rcm - center) / radii
        direction = rotation.T @ axis / radii
        a, b, c = float(direction @ direction), float(2 * origin @ direction), float(origin @ origin - 1)
        discriminant = b * b - 4 * a * c
        result = {"tip_inside_eye": bool(np.sum((rotation.T @ (tip - center) / radii) ** 2) <= 1 + 1e-9),
                  "junction_inside_eye": bool(np.sum((rotation.T @ (junction - center) / radii) ** 2) <= 1 + 1e-9),
                  "junction_rcm_axial_offset_mm": float((junction - rcm) @ axis),
                  "tip_rcm_distance_mm": float(np.linalg.norm(tip - rcm)),
                  "entry_rcm_distance_mm": None, "entry_world_mm": None,
                  "entry_needle_part": "no_eye_intersection",
                  "distal_centerline_inside_mm": 0.0, "proximal_centerline_inside_mm": 0.0,
                  "entry_needle_diameter_mm": None, "entry_aperture_validated": False,
                  "rcm_eye_surface_value": float(origin @ origin),
                  "rcm_on_eye_surface": abs(c) < 1e-8}
        if discriminant < -1e-12:
            return result
        roots = sorted([(-b - math.sqrt(max(0.0, discriminant))) / (2 * a),
                        (-b + math.sqrt(max(0.0, discriminant))) / (2 * a)])
        entry, exit_ = roots
        locations = [float((p - rcm) @ axis) for p in (back, junction, tip)]
        def overlap(lo, hi):
            return max(0.0, min(hi, exit_) - max(lo, entry))
        result.update(entry_world_mm=(rcm + entry * axis).tolist(),
                      entry_rcm_distance_mm=abs(entry),
                      distal_centerline_inside_mm=overlap(locations[1], locations[2]),
                      proximal_centerline_inside_mm=overlap(locations[0], locations[1]))
        if abs(entry - locations[1]) < 1e-8:
            part = "junction"
        elif locations[1] < entry < locations[2] + 1e-8:
            part = "distal"
        elif locations[0] - 1e-8 < entry < locations[1]:
            part = "proximal"
        else:
            part = "outside_modeled_needle"
        result["entry_needle_part"] = part
        if part in ("distal", "proximal", "junction"):
            radius = self.config["needle"]["distal_radius_mm"] if part == "distal" else self.config["needle"]["proximal_radius_mm"]
            result["entry_needle_diameter_mm"] = 2 * radius
        return result

    def metrics(self):
        tip, junction, rcm = [self.site_mm(name) for name in
                              ("needle_tip", "driver_junction", "rcm_site")]
        target = self.site_mm("lesion")
        back = self.site_mm("needle_base")
        direction = tip - back
        direction /= np.linalg.norm(direction)
        predicted = self.robot_to_world(self._ref.forward_kinematics(self.current_task()))
        return {
            "task_A_deg_B_deg_C_mm": [math.degrees(self.current_task()[0]),
                                      math.degrees(self.current_task()[1]),
                                      float(self.current_task()[2] * 1000)],
            "tip_world_mm": tip.tolist(), "target_world_mm": target.tolist(),
            "rcm_world_mm": rcm.tolist(), "junction_world_mm": junction.tolist(),
            "tip_target_error_um": float(np.linalg.norm(tip - target) * 1000),
            "rcm_drift_um": float(np.linalg.norm(rcm - self.rcm_mm) * 1000),
            "needle_axis_rcm_distance_um": float(np.linalg.norm(np.cross(rcm - back, direction)) * 1000),
            "fk_agreement_um": float(np.linalg.norm(tip - predicted) * 1000),
            "distal_length_mm": float(np.linalg.norm(tip - junction)),
            "upper_link_closure_um": float(np.linalg.norm(
                self.site_mm("upper_bridge_end") - self.site_mm("front_upper")) * 1000),
            "lower_link_closure_um": float(np.linalg.norm(
                self.site_mm("lower_bridge_end") - self.site_mm("front_lower")) * 1000),
            "needle_eye_geometry": self.needle_eye_geometry(),
        }
