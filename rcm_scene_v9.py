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
    visual_perception_backend = 'strict_rgbd'

    def __init__(self, config=None, model_dir=None):
        # The eye shell is display-only for this adapter. Sensor renders must
        # exclude it so the shell cannot occlude intraocular geometry.
        self.sensor_eye_group_visible = False
        self.config = config if config is not None else json.loads(
            (ROOT / "config.json").read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported config schema")
        # The hidden reference is a coordinate/encoder calculator. Only this
        # adapter advances a motor; otherwise a command would be applied twice.
        self._ref = Reference({**self.config, 'motor_execution': {'enabled': False}})
        # Preserve the legacy kinematic model surface used by controller
        # Jacobian construction.
        self.nominal = self._ref.nominal
        from model_sync import DriverModel
        self.scene = DriverModel(model_dir=model_dir, needle_profile="legacy")
        self.eye_camera = self._ref.eye_camera
        self.alignment = json.loads((ROOT/"v9_alignment.json").read_text())
        yaw = math.radians(float(self.alignment["yaw_world_deg"]))
        if not math.isfinite(yaw):
            raise ValueError("v9 alignment yaw must be finite")
        c, s = math.cos(yaw), math.sin(yaw)
        self.alignment_R = np.array([[c,-s,0.], [s,c,0.], [0.,0.,1.]])
        pivot = _qvec(self.alignment["pivot_world_m"], 3)
        self.alignment_t = pivot-self.alignment_R@pivot
        self.tool_mount = json.loads((ROOT/"v9_tool_mount.json").read_text())
        self.side_profile = json.loads((ROOT/"v9_side_profile.json").read_text())
        expected_axial_offset = (float(self.side_profile["reference_shaft_attachment_from_rcm_mm"])
                                 - float(self.config["needle"]["proximal_length_mm"]))
        if not np.isclose(self.tool_mount["front_axial_registration_offset_mm"], expected_axial_offset):
            raise ValueError("Tool mount is stale; refit after changing the proximal needle attachment")
        if (not np.isclose(self.tool_mount["alignment_yaw_deg"], self.alignment["yaw_world_deg"])
                or not np.isclose(self.tool_mount["installation_pitch_deg"], self.scene.installation["pitch_y_deg"])):
            raise ValueError("Tool mount is stale; refit the registered STL for the current installation")
        if not np.isclose(self.tool_mount["tool_side_pitch_deg"], self.side_profile["needle_pitch_from_horizontal_deg"]):
            raise ValueError("Tool mount is stale; refit after changing the side profile")
        self.mount_R = np.asarray(self.tool_mount["mount_rotation_registered"], dtype=float)
        if (self.mount_R.shape != (3,3)
                or not np.allclose(self.mount_R.T@self.mount_R, np.eye(3), atol=1e-10)
                or not np.isclose(np.linalg.det(self.mount_R), 1.)):
            raise ValueError("Invalid STL tool-mount rotation")
        self.mount_rcm = _qvec(self.tool_mount["rcm_registered_m"], 3)
        installed_rcm = self.scene.install_R@self.mount_rcm+self.scene.install_t
        self.rcm_mm = (self.alignment_R@installed_rcm+self.alignment_t)*1000.
        # Match the scan's outward (-local Z) tool direction. A proper half
        # turn, with mirrored A/C presentation below, preserves the legacy task
        # frame while putting the eye in front of the instrument, not its case.
        self.R_robot_world = (self.alignment_R @ self.scene.install_R @ self.mount_R
                              @ self.scene.robot_to_installation @ np.diag([1., -1., -1.]))
        self._build_injected_model()
        self.set_task(np.zeros(3))
        self.motor_execution = None
        if self.config.get('motor_execution', {}).get('enabled', False):
            self.enable_motor_execution()

    def _build_injected_model(self):
        root = ET.fromstring(self.scene.xml)
        self._apply_stl_mount_frame(root)
        self._align_installation(root)
        # Keep the registered scan assets intact. Use deburred display skins
        # only in this injected eye scene; all part/joint frames stay fixed.
        for name in ("fixed_housing", "parallel_cover", "distal_guide"):
            mesh = root.find(f"./asset/mesh[@name='{name}']")
            mesh.set("file", self.side_profile["scan_guide_mesh"] if name == "distal_guide"
                     else name+"_smooth.obj")
        self._inject_assets(root)
        cap = root.find(".//geom[@name='scan_distal']")
        cap.attrib.pop("rgba", None)
        cap.set("material", "tool_panel_silver")
        self._inject_eye_scene(root)
        self._inject_needle_sensor_geoms(root)
        from v9_front_tool import add_side_cover_details
        add_side_cover_details(root)
        self._classify_viewer_geometry(root)
        self._inject_position_sensors(root)
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

    def _apply_stl_mount_frame(self, root):
        """Fit the virtual RCM chain to the STL tool axis, preserving scan pose.

        Conjugating A/B/C axes by one mount rotation keeps the shared logical
        FK exact. Visual mesh offsets compensate the estimated RCM-origin
        correction, so the registered scan itself does not move at zero pose.
        This is a v9 simulation registration, never a hardware calibration.
        """
        stage = root.find(".//body[@name='A_stage']")
        distal = stage.find("body[@name='distal_orientation']")
        old_origin = np.fromstring(stage.get("pos"), sep=" ")
        delta = self.mount_rcm-old_origin
        stage.set("pos", _s(self.mount_rcm))

        def compensate(node):
            if node.tag not in ("geom", "site", "body"):
                return
            if node.get("fromto") is not None:
                p = np.fromstring(node.get("fromto"), sep=" ").reshape(2,3)
                node.set("fromto", _s((p-delta).ravel()))
            else:
                p = np.fromstring(node.get("pos", "0 0 0"), sep=" ")
                node.set("pos", _s(p-delta))

        for node in stage:
            if node is not distal:
                compensate(node)
        for node in distal:
            if node.get("name") != "C_translation":
                compensate(node)
        root.find(".//site[@name='rcm_site']").set("pos", _s(self.mount_rcm))
        for joint in stage.iter("joint"):
            axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
            joint.set("axis", _s(self.mount_R@axis))

    def _align_installation(self, root):
        """Align robot geometry, task frame and cameras, keeping the lab level."""
        world = root.find("worldbody")
        installation = world.find("body[@name='static_installation']")
        if installation is None:
            raise ValueError("v9 model missing static installation")
        world.remove(installation)
        wrapper = ET.SubElement(world, "body", name="v9_horizontal_alignment",
                                pos=_s(self.alignment_t),
                                quat=_s(_quat_from_matrix(self.alignment_R)))
        wrapper.append(installation)
        # Reposition lab fixtures with the assembly. A pure world-Z yaw keeps
        # all support faces horizontal; gravity is never rotated.
        for geom in world.findall("geom"):
            if geom.get("name") == "floor":
                continue
            if geom.get("pos") is not None:
                p = np.fromstring(geom.get("pos"), sep=" ")
                geom.set("pos", _s(self.alignment_R@p+self.alignment_t))
                old_rotation = np.empty(9)
                quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
                mujoco.mju_quat2Mat(old_rotation, quat)
                geom.set("quat", _s(_quat_from_matrix(
                    self.alignment_R@old_rotation.reshape(3,3))))
            if geom.get("fromto") is not None:
                points = np.fromstring(geom.get("fromto"), sep=" ").reshape(2,3)
                geom.set("fromto", _s((points@self.alignment_R.T+self.alignment_t).ravel()))
        for camera in world.findall("camera"):
            p = np.fromstring(camera.get("pos"), sep=" ")
            camera.set("pos", _s(self.alignment_R@p+self.alignment_t))
            axes = np.fromstring(camera.get("xyaxes"), sep=" ").reshape(2,3)
            camera.set("xyaxes", _s((axes@self.alignment_R.T).ravel()))
        statistic = root.find("statistic")
        center = np.fromstring(statistic.get("center"), sep=" ")
        statistic.set("center", _s(self.alignment_R@center+self.alignment_t))

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
            "needle_proximal_metal": ".45 .50 .58 1",
            "tool_steel": ".72 .73 .70 1",
            "tool_polymer": ".88 .90 .88 .72",
            "tool_case": ".91 .92 .90 1",
            "tool_clear": ".74 .84 .89 .22",
            "tool_fastener": ".18 .19 .20 1",
            "tool_panel_silver": ".61 .64 .66 1",
            "tool_etch": ".87 .90 .92 1",
            "tool_base_gray": ".43 .45 .45 1",
        }.items():
            if name not in existing:
                ET.SubElement(asset, "material", name=name, rgba=rgba,
                              specular="0.4", shininess="0.45")

    def _to_world_m(self, point_robot_mm):
        return self.rcm_mm*.001 + self.R_robot_world @ (np.asarray(point_robot_mm, dtype=float) * 0.001)

    def _inject_eye_scene(self, root):
        global_vis = root.find("./visual/global")
        if global_vis is not None:
            width, height = self.eye_camera["resolution_px"]
            global_vis.set("offwidth", str(max(int(global_vis.get("offwidth", "1280")), width)))
            global_vis.set("offheight", str(max(int(global_vis.get("offheight", "960")), height)))
        world = root.find("worldbody")
        asset = root.find("asset")
        if asset is None:
            raise ValueError("v9 model XML missing 'asset' element")
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

        camera = self.eye_camera
        camera_pos = self._to_world_m(camera["position_robot_mm"])
        camera_target = self._to_world_m(camera["target_robot_mm"])
        raw_eye_camera = self.config["reference_scene"].get("eye_camera", {})
        pupil_robot_mm = raw_eye_camera.get("pupil_robot_mm",
                                            camera["position_robot_mm"])
        pupil = self._to_world_m(pupil_robot_mm)
        optical_axis = camera_target - pupil
        optical_axis /= np.linalg.norm(optical_axis)
        pupil_axes = _camera_axes(pupil, camera_target).reshape(2, 3)
        for index, offset in enumerate((pupil_axes[0], -pupil_axes[0],
                                        pupil_axes[1], -pupil_axes[1])):
            ET.SubElement(world, "geom", name=f"pupil_ring_{index}", type="sphere",
                          pos=_s(pupil + offset * 0.0025), size=".00025",
                          material="pupil", group="2")
        ET.SubElement(world, "geom", name="lens_visual", type="ellipsoid",
                      pos=_s(pupil + optical_axis * 0.006),
                      size=".003 .003 .0015", quat=_s(world_quat),
                      material="lens", group="2")

        # Match the legacy RGB-only posterior fundus backdrop. It is offset
        # outward along the ellipsoid normal so the needle and lesion remain in
        # front, and stays in group 5 so depth/segmentation callers can opt out.
        retina_body = ET.SubElement(world, "body", name="retina_visual_frame",
                                    pos=_s(eye_center), quat=_s(world_quat))
        retina_offset_m = 0.008
        retina_levels = np.linspace(0.55, 1.0, 10)
        retina_azimuths = 32
        retina_rings = [
            [(math.sqrt(max(0.0, 1.0 - float(level) ** 2))
              * math.cos(2.0 * math.pi * index / retina_azimuths),
              math.sqrt(max(0.0, 1.0 - float(level) ** 2))
              * math.sin(2.0 * math.pi * index / retina_azimuths),
              float(level))
             for index in range(retina_azimuths)]
            for level in retina_levels
        ]

        def retina_vertex(unit):
            point = radii * np.asarray(unit, dtype=float)
            normal = np.asarray(unit, dtype=float) / radii
            normal /= np.linalg.norm(normal)
            return tuple(point + retina_offset_m * normal)

        mesh_vertices = [retina_vertex(value) for ring in retina_rings for value in ring]
        mesh_faces = []
        for ring_index in range(len(retina_rings) - 1):
            for index in range(retina_azimuths):
                next_index = (index + 1) % retina_azimuths
                a = ring_index * retina_azimuths + index
                b = ring_index * retina_azimuths + next_index
                c = (ring_index + 1) * retina_azimuths + index
                d = (ring_index + 1) * retina_azimuths + next_index
                mesh_faces.extend(((a, d, b), (a, c, d)))
        apex_vertex = len(mesh_vertices)
        mesh_vertices.append(retina_vertex((0.0, 0.0, 1.0)))
        last_ring_start = (len(retina_rings) - 1) * retina_azimuths
        for index in range(retina_azimuths):
            mesh_faces.append((last_ring_start + index, apex_vertex,
                               last_ring_start + (index + 1) % retina_azimuths))
        ET.SubElement(asset, "mesh", name="retina_visual_mesh",
                      vertex=_s(value for vertex in mesh_vertices for value in vertex),
                      face=" ".join(" ".join(str(value) for value in face)
                                    for face in mesh_faces))
        ET.SubElement(retina_body, "geom", name="retina_visual", type="mesh",
                      mesh="retina_visual_mesh", material="retina", group="5")

        target = self._to_world_m(eye["target_robot_mm"])
        ET.SubElement(world, "geom", name="lesion_marker", type="sphere",
                      pos=_s(target), size=".00015", material="lesion", group="3")
        ET.SubElement(world, "site", name="lesion", pos=_s(target))
        display = float(eye.get("display_target_radius_mm", 0.5))
        ET.SubElement(world, "geom", name="lesion_display_marker", type="sphere",
                      pos=_s(target), size=str(display * 0.001),
                      material="display_target", group="2")

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
        # The source model's axis aid is wider than the physical needle and
        # extends through the eye. Omit it from this injected eye scene so it
        # cannot cover the shaft/tip or look like an extra needle segment.
        for parent in root.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("name") == "axis_aid":
                    parent.remove(geom)
        proximal_z = float(needle["proximal_length_mm"]) * 0.001
        if not math.isfinite(proximal_z) or proximal_z <= 0:
            raise ValueError("needle.proximal_length_mm must be finite and positive")
        tip_z = -needle["distal_length_mm"] * 0.001
        distal_tip_length = needle["distal_tip_length_mm"] * 0.001
        tube_end = tip_z + distal_tip_length
        if tube_end > 0:
            raise ValueError("needle.distal_tip_length_mm cannot exceed distal_length_mm")
        for geom in body.findall("geom"):
            name = geom.get("name")
            if name == "needle_shaft":
                geom.set("name", "needle_proximal")
                geom.set("type", "cylinder")
                geom.set("group", "3")
                geom.attrib.pop("pos", None)
                geom.attrib.pop("quat", None)
                geom.set("fromto", f"0 0 {proximal_z:.15g} 0 0 0")
                geom.set("size", f"{needle['proximal_radius_mm'] * 0.001:.15g}")
                geom.attrib.pop("rgba", None)
                geom.set("material", "needle_proximal_metal")
            elif name == "needle_distal":
                geom.set("type", "cylinder")
                geom.set("group", "3")
                geom.attrib.pop("pos", None)
                geom.attrib.pop("quat", None)
                geom.set("fromto", f"0 0 0 0 0 {tube_end:.15g}")
                geom.set("size", f"{needle['distal_radius_mm'] * 0.001:.15g}")
                geom.attrib.pop("rgba", None)
                geom.set("material", "needle")
        base = next((site for site in body.findall("site") if site.get("name") == "needle_base"), None)
        if base is None:
            raise ValueError("v9 model XML missing site 'needle_base'")
        base.set("pos", f"0 0 {proximal_z:.15g}")
        body.find("site[@name='needle_tip']").set("pos", f"0 0 {tip_z:.15g}")
        from v9_front_tool import rebuild_front_tool
        rebuild_front_tool(root, body, proximal_z, self.tool_mount["case_center_xy_tool_m"])
        for joint in root.iter("joint"):
            if joint.get("name") in ("A", "C") and joint.get("range"):
                low, high = map(float, joint.get("range").split())
                joint.set("range", _s([-high, -low]))
        for actuator in root.findall("./actuator/position"):
            if actuator.get("joint") in ("A", "C") and actuator.get("ctrlrange"):
                low, high = map(float, actuator.get("ctrlrange").split())
                actuator.set("ctrlrange", _s([-high, -low]))

        # Add a tapered tip mesh matching the legacy model. A capsule ends in a
        # hemisphere larger than the calibrated apex marker and hides it.
        asset = root.find("asset")
        if asset is None:
            raise ValueError("v9 model XML missing 'asset' element")
        base_radius = float(needle["distal_radius_mm"]) * 0.001
        apex_radius = float(needle["distal_tip_apex_radius_mm"]) * 0.001
        poly_sides = 8
        vertices = []
        for z, radius in ((0.0, base_radius), (-distal_tip_length, apex_radius)):
            for index in range(poly_sides):
                angle = 2.0 * math.pi * index / poly_sides
                vertices.extend((radius * math.cos(angle),
                                 radius * math.sin(angle), z))
        vertices.extend((0.0, 0.0, 0.0))
        vertices.extend((0.0, 0.0, -distal_tip_length))
        faces = []
        base_center = 2 * poly_sides
        apex_center = base_center + 1
        for index in range(poly_sides):
            next_index = (index + 1) % poly_sides
            faces.extend(((index, next_index, poly_sides + next_index),
                          (index, poly_sides + next_index, poly_sides + index),
                          (base_center, next_index, index),
                          (apex_center, poly_sides + index,
                           poly_sides + next_index)))
        ET.SubElement(asset, "mesh", name="needle_distal_tip_mesh",
                      vertex=_s(vertices),
                      face=" ".join(" ".join(str(value) for value in face)
                                    for face in faces))
        ET.SubElement(body, "geom", name="needle_distal_tip", type="mesh",
                      mesh="needle_distal_tip_mesh", pos=f"0 0 {tube_end:.15g}",
                      material="needle", group="3")
        # MuJoCo's default viewer enables groups 0..2, while sensor passes
        # enable group 3 and exclude group 2. Same-size display copies make the
        # actual stepped needle visible when opening the exported XML directly.
        # Never enlarge these copies: the junction must remain at the eye port.
        for name in ("needle_proximal", "needle_distal", "needle_distal_tip"):
            physical = body.find(f"geom[@name='{name}']")
            display = ET.fromstring(ET.tostring(physical))
            display.set("name", name.replace("needle_", "needle_display_", 1))
            display.set("group", "2")
            body.append(display)
        ET.SubElement(body, "geom", name="needle_tip_marker", type="sphere",
                      pos=f"0 0 {tip_z:.15g}",
                      size=f"{needle['tip_marker_radius_mm'] * 0.001:.15g}",
                      material="tip_marker", group="4")
        # Holder, both needle segments, metrology sites and display copies use
        # one STL-derived frame. Rotating only their colors/visual copies would
        # leave the C translation and optical geometry inconsistent.
        mount = ET.SubElement(body, "body", name="needle_mount",
                              quat=_s(_quat_from_matrix(self.mount_R)))
        for node in list(body):
            if node.tag in ("geom", "site"):
                body.remove(node)
                mount.append(node)

    def _inject_position_sensors(self, root):
        sensor = root.find("sensor")
        if sensor is None:
            sensor = ET.SubElement(root, "sensor")
        mapping = {
            "rcm_world_position": "rcm_site",
            "needle_tip_world_position": "needle_tip",
            "needle_junction_world_position": "driver_junction",
            "lesion_world_position": "lesion",
        }
        existing = {node.get("name"): node for node in sensor.findall("framepos")}
        for name, site_name in mapping.items():
            node = existing.get(name)
            if node is None:
                node = ET.SubElement(sensor, "framepos", name=name)
            node.set("objtype", "site")
            node.set("objname", site_name)

    def _classify_viewer_geometry(self, root):
        """Separate display-only environment and linkage aids from CAD."""
        environment_names = {
            "floor", "support", "phantom", "phantom_pad",
            "horizontal_mount_plate", "target_table_top",
        }
        environment_prefixes = (
            "target_tape_", "horizontal_reference_edge_",
            "target_table_leg_",
        )
        linkage_aids = {"guide_link", "rear_link", "upper_link", "lower_link"}
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if (name in environment_names or name.startswith(environment_prefixes)
                    or name in linkage_aids):
                geom.set("group", "1")

    def build_xml(self):
        return self.xml

    def state_from_task(self, q):
        return self._ref.state_from_task(q)

    def task_from_state(self, state):
        return self._ref.task_from_state(state)

    def validate_task(self, q):
        return self._ref.validate_task(q)

    def set_task(self, q, motor_dt=None):
        motor = getattr(self, 'motor_execution', None)
        if motor is None:
            if motor_dt is not None:
                raise ValueError('Enable motor execution before passing motor_dt')
            self._apply_task_pose(q)
            return
        previous = self.data.qpos.copy()
        actual = motor.advance(q, motor_dt)
        self._apply_task_pose(actual)
        self.data.qvel[:] = (self.data.qpos - previous) / motor.last_record['duration_s']
        mujoco.mj_forward(self.model, self.data)

    def enable_motor_execution(self, settings=None):
        if getattr(self, 'motor_execution', None) is not None:
            raise ValueError('Motor execution is already enabled')
        from rcm_motor_execution import MotorExecution
        self.motor_execution = MotorExecution(self, settings or self.config.get('motor_execution'))
        self._apply_task_pose(self.task_from_encoders(self.motor_execution.backend.robot[:3]))

    def motor_robot(self):
        if self.motor_execution is None:
            raise RuntimeError('Motor execution is not enabled')
        return self.motor_execution.backend.robot

    def reset_motor_time(self, t=0.):
        if self.motor_execution is None:
            raise RuntimeError('Motor execution is not enabled')
        self.motor_execution.reset_clock(t)

    def _apply_task_pose(self, q):
        q = self._ref.validate_task(q)
        self._ref.set_task(q)
        robot = tuple(int(round(float(v))) for v in self._ref.encoders(q)) + (0,)
        self.scene.sync(robot)
        self.data.qpos[:] = self.scene.data.qpos
        # Presentation conversion only; encoders()/task_from_encoders() retain
        # the shared logical task convention, never hardware motor commands.
        for name in ("A", "C"):
            address = self.model.jnt_qposadr[self.model.joint(name).id]
            self.data.qpos[address] *= -1
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
            option.geomgroup[2] = 1
            option.geomgroup[3] = 1
            option.geomgroup[4] = 1
            option.geomgroup[5] = 1
            option.sitegroup[:] = 0
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
        tip, junction, back, rcm = [self.site_mm(name) for name in
                                    ("needle_tip", "driver_junction", "needle_base", "rcm_site")]
        axis = tip - junction
        axis /= np.linalg.norm(axis)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
        rotation = self.data.geom_xmat[gid].reshape(3, 3)
        center = self.data.geom_xpos[gid] * 1000.0
        radii = self.model.geom_size[gid] * 1000.0
        # The junction is the modeled eye-entry point. Intersect from there
        # toward the tip and use the root nearest the junction, which tolerates
        # sub-micrometre surface residuals without borrowing direction from the
        # proximal end.
        origin = rotation.T @ (junction - center) / radii
        direction = rotation.T @ axis / radii
        a, b, c = float(direction @ direction), float(2 * origin @ direction), float(origin @ origin - 1)
        discriminant = b * b - 4 * a * c
        rcm_origin = rotation.T @ (rcm - center) / radii
        result = {"tip_inside_eye": bool(np.sum((rotation.T @ (tip - center) / radii) ** 2) <= 1 + 1e-9),
                  "junction_inside_eye": bool(np.sum((rotation.T @ (junction - center) / radii) ** 2) <= 1 + 1e-9),
                  "junction_rcm_axial_offset_mm": float((junction - rcm) @ axis),
                  "tip_rcm_distance_mm": float(np.linalg.norm(tip - rcm)),
                  "entry_rcm_distance_mm": None, "entry_world_mm": None,
                  "entry_needle_part": "no_eye_intersection",
                  "distal_centerline_inside_mm": 0.0, "proximal_centerline_inside_mm": 0.0,
                  "entry_needle_diameter_mm": None, "entry_aperture_validated": False,
                  "rcm_eye_surface_value": float(rcm_origin @ rcm_origin),
                  "rcm_on_eye_surface": bool(abs(float(rcm_origin @ rcm_origin) - 1) < 1e-8)}
        if discriminant < -1e-12:
            return result
        roots = sorted([(-b - math.sqrt(max(0.0, discriminant))) / (2 * a),
                        (-b + math.sqrt(max(0.0, discriminant))) / (2 * a)])
        entry, exit_ = sorted(roots, key=abs)
        if entry > exit_:
            entry, exit_ = exit_, entry
        locations = [float((p - junction) @ axis) for p in (back, junction, tip)]
        def overlap(lo, hi):
            return max(0.0, min(hi, exit_) - max(lo, entry))
        entry_world = junction + entry * axis
        result.update(entry_world_mm=entry_world.tolist(),
                      entry_rcm_distance_mm=float(np.linalg.norm(entry_world - rcm)),
                      distal_centerline_inside_mm=overlap(locations[1], locations[2]),
                      proximal_centerline_inside_mm=overlap(locations[0], locations[1]))
        if abs(entry - locations[1]) < 1e-5:
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
