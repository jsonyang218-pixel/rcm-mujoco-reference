"""MuJoCo kinematic reference, SI units in MJCF, mm in calibration inputs.

Three independent task joints; four dependent visual joints. No physical motor
plant, ROS connection, measured inertia, tissue contact, or hardware commands.
"""
from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import mujoco
import numpy as np

from rcm_recovered_model import EncoderMap, Geometry, JointState, RCMModel

HERE = Path(__file__).resolve().parent
TASK_JOINTS = ("A_azimuth", "B_pitch", "C_insertion")
DEPENDENT_JOINTS = {
    "visual_B_rear": 1.0,
    "visual_B_bridge": -1.0,
    "visual_B_lower_bridge": -1.0,
    "visual_B_aux": 1.0,
}


def _vec(values, length=3):
    v = np.asarray(values, dtype=float)
    if v.shape != (length,) or not np.isfinite(v).all():
        raise ValueError(f"Expected {length} finite numbers")
    return v


def installation_rotation(degrees):
    """R = Rz(z) Ry(y) Rx(x); fixed, extrinsic XYZ installation angles."""
    x, y, z = np.radians(_vec(degrees))
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_quaternion(rotation):
    # MuJoCo stores unit quaternions as w,x,y,z.
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
    return quat


def _s(values):
    return " ".join(f"{float(v):.15g}" for v in values)


class Reference:
    def __init__(self, config=None):
        self.config = config if config is not None else json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("Unsupported config schema")
        enc = self.config["encoders"]
        self.nominal = RCMModel(Geometry(**self.config["kinematics"]), EncoderMap(
            a_counts_per_degree=enc["a_counts_per_degree"], bc_counts_per_mm=enc["bc_counts_per_mm"],
            origin=tuple(enc["origin"]), direction=tuple(enc["direction"])))
        self.rotation_mount = installation_rotation(self.config["installation"]["euler_xyz_deg"])
        # Fixed basis conversion from source robot frame to the earlier diagram.
        # Rx(180) is a coordinate convention, NOT another physical installation.
        self.rotation_robot = self.rotation_mount @ np.diag([1.0, -1.0, -1.0])
        self.rcm_mm = _vec(self.config["installation"]["rcm_world_mm"])
        self.alpha_nominal_zero = 90.0 - self.nominal.geometry.tilt_offset_deg
        self.distal_mm = float(self.config["needle"]["distal_length_mm"])
        self.distal_tip_mm = float(self.config["needle"].get("distal_tip_length_mm", 0.))
        self.tip_apex_radius_mm = float(self.config["needle"].get("distal_tip_apex_radius_mm", 0.))
        for name in ("distal_length_mm", "distal_tip_length_mm", "distal_tip_apex_radius_mm",
                     "proximal_length_mm", "proximal_radius_mm", "distal_radius_mm",
                     "tip_marker_radius_mm", "tip_visual_marker_radius_mm", "display_scale"):
            value = self.config["needle"].get(name)
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"needle.{name} must be positive")
        if self.distal_tip_mm >= self.distal_mm:
            raise ValueError("needle.distal_tip_length_mm must be shorter than distal_length_mm")
        if self.tip_apex_radius_mm > float(self.config["needle"]["distal_radius_mm"]):
            raise ValueError("needle.distal_tip_apex_radius_mm cannot exceed distal_radius_mm")
        display_scale = float(self.config["needle"].get("display_scale", 1.0))
        if not math.isfinite(display_scale) or display_scale <= 0:
            raise ValueError("needle.display_scale must be positive")
        self.reference_spec = self.config.get("encoder_reference", {"enabled": False})
        self.reference_enabled = bool(self.reference_spec["enabled"])
        self.reset_encoders = None
        self.phi_home, self.alpha_home, self.tip_reference_mm = 0., self.alpha_nominal_zero, self.distal_mm
        if self.reference_enabled:
            self.reset_encoders = _vec(self.reference_spec["robot_counts_ABC"])
            e = self.nominal.encoders
            nominal_reset = (self.reset_encoders-np.array(e.origin))/np.array(e.direction)
            self.phi_home = nominal_reset[0]/e.a_counts_per_degree
            self.alpha_home = self.nominal.alpha_from_b(nominal_reset[1]/e.bc_counts_per_mm)
            self.tip_reference_mm = float(self.reference_spec["needle_tip_distance_at_reference_mm"])
            if not math.isfinite(self.tip_reference_mm) or self.tip_reference_mm < 0:
                raise ValueError("Reference tip distance must be finite and nonnegative")
        self.b_initial_deg = self.alpha_home-self.alpha_nominal_zero
        _vec(self.config["reference_scene"]["eye_center_robot_mm"])
        if (_vec(self.config["reference_scene"]["eye_radii_robot_mm"]) <= 0).any():
            raise ValueError("Eye radii must be positive")
        _vec(self.config["reference_scene"]["target_robot_mm"])
        eye_camera = self.config["reference_scene"].get("eye_camera", {})
        camera_position = _vec(eye_camera.get("position_robot_mm", self.config["reference_scene"]["eye_center_robot_mm"]))
        camera_target = _vec(eye_camera.get("target_robot_mm", self.config["reference_scene"]["target_robot_mm"]))
        if not math.isfinite(float(eye_camera.get("fovy_deg", 70.0))) or not 1.0 < float(eye_camera.get("fovy_deg", 70.0)) < 179.0:
            raise ValueError("reference_scene.eye_camera.fovy_deg must be between 1 and 179")
        resolution = eye_camera.get("resolution_px", [320, 240])
        if len(resolution) != 2 or any(int(v) <= 0 for v in resolution):
            raise ValueError("reference_scene.eye_camera.resolution_px must contain two positive integers")
        if np.linalg.norm(camera_target-camera_position) <= 1e-9:
            raise ValueError("reference_scene.eye_camera must point at a distinct target")
        self.eye_camera = {
            "position_robot_mm": camera_position.tolist(),
            "target_robot_mm": camera_target.tolist(),
            "fovy_deg": float(eye_camera.get("fovy_deg", 70.0)),
            "resolution_px": [int(resolution[0]), int(resolution[1])],
            "znear_mm": float(eye_camera.get("znear_mm", 0.1)),
            "zfar_mm": float(eye_camera.get("zfar_mm", 60.0)),
        }
        if self.eye_camera["znear_mm"] <= 0 or self.eye_camera["zfar_mm"] <= self.eye_camera["znear_mm"]:
            raise ValueError("reference_scene.eye_camera clipping distances are invalid")
        link = self.config["illustrative_linkage"]
        for name in ("rear_distance_mm", "rear_pivot_spacing_mm", "upper_crank_mm", "lower_crank_mm", "guide_offset_mm"):
            if not math.isfinite(link[name]) or link[name] <= 0:
                raise ValueError(f"illustrative_linkage.{name} must be positive")
        visual_scale = float(link.get("visual_scale", 1.0))
        if not math.isfinite(visual_scale) or visual_scale <= 0:
            raise ValueError("illustrative_linkage.visual_scale must be positive")
        if not link["lower_crank_mm"] < link["upper_crank_mm"]:
            raise ValueError("Lower crank must be shorter than upper crank")
        self.encoder_bounds = self._read_limits()
        self.joint_bounds = self._joint_limits()
        self.manual_steps = self.config.get("manual_steps", {"A_deg": .5, "B_deg": .5, "C_mm": .25})
        for axis, unit in (("A", "deg"), ("B", "deg"), ("C", "mm")):
            value = self.manual_steps[f"{axis}_{unit}"]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"manual_steps.{axis}_{unit} must be positive")
        self.xml = self.build_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.qaddr = {name: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                      for name in (*TASK_JOINTS, *DEPENDENT_JOINTS)}
        self.sites = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                      for name in ("rcm", "needle_tip", "needle_junction", "needle_axis_back", "lesion",
                                   "eye_camera_site",
                                   "link_O1", "link_O2", "link_P", "link_Q", "link_P2", "link_F", "link_F2",
                                   "bridge_end", "lower_bridge_end", "guide_top", "guide_bottom")}
        # Preserve an explicit reported reset; only legacy configs may fall back.
        self.home = np.zeros(3)
        if self.joint_bounds is not None and not self._in_joint_bounds(self.home):
            if self.reference_enabled:
                raise ValueError("Enabled limits exclude reported reset after reserves; adjust the explicit reset inclusion policy")
            self.home = self.joint_bounds.mean(axis=1)
        self.set_task(self.home)

    def _read_limits(self):
        spec = self.config["measured_limits"]
        self.recorded_encoder_bounds = None
        if not spec["enabled"]:
            return None
        scope = spec.get("scope", "calibrated_workspace")
        if not self.config["encoders"]["calibrated"] and scope != "simulation_recorded_window":
            raise ValueError("Uncalibrated encoder mapping may enforce recorded windows only in explicit simulation scope")
        bounds, recorded = [], []
        include_reset = bool(spec.get("include_reported_reset_for_simulation", False))
        if include_reset and (scope != "simulation_recorded_window" or self.reset_encoders is None):
            raise ValueError("Reset inclusion requires a supplied reference and explicit simulation scope")
        for axis in "ABC":
            pair = _vec(spec["encoder_counts"][axis], 2)
            reserve = float(spec["reserve_counts"][axis])
            if not math.isfinite(reserve) or reserve < 0 or pair[0] >= pair[1]:
                raise ValueError(f"Invalid measured {axis} bounds or reserve")
            recorded.append(pair.copy())
            if include_reset:
                reset = self.reset_encoders["ABC".index(axis)]
                pair[0], pair[1] = min(pair[0], reset), max(pair[1], reset)
            pair += np.array([reserve, -reserve])
            if pair[0] >= pair[1]:
                raise ValueError(f"{axis} reserve leaves an empty range")
            bounds.append(pair)
        self.recorded_encoder_bounds = np.array(recorded)
        return np.array(bounds)

    def _joint_limits(self):
        if self.encoder_bounds is None:
            return None
        e, g = self.nominal.encoders, self.nominal.geometry
        a = (self.encoder_bounds[0] - e.origin[0]) / e.direction[0] / e.a_counts_per_degree - self.phi_home
        b = [self.nominal.alpha_from_b((v-e.origin[1])/e.direction[1]/e.bc_counts_per_mm) - self.alpha_home
             for v in self.encoder_bounds[1]]
        if self.reference_enabled:
            c = (self.encoder_bounds[2]-self.reset_encoders[2])/e.direction[2]/e.bc_counts_per_mm
        else:
            c = (self.encoder_bounds[2]-e.origin[2])/e.direction[2]/e.bc_counts_per_mm-g.c_reference_mm-self.tip_reference_mm
        result = np.array([np.radians(np.sort(a)), np.radians(np.sort(b)), np.sort(c)*.001])
        # Check the same invertible B branch over the whole converted interval.
        for b_rad in np.linspace(*result[1], 101):
            alpha = self.alpha_home + math.degrees(b_rad)
            d = self.nominal.b_displacement(alpha)
            if abs(self.nominal.alpha_from_b(d)-alpha) > 1e-7:
                raise ValueError("Measured B range crosses a non-invertible assembly branch")
        if result[2, 0]*1000+self.tip_reference_mm < -1e-9:
            raise ValueError("Measured C range implies a negative RCM-to-tip length; check its zero")
        return result

    def _in_joint_bounds(self, q):
        return bool(np.all(q >= self.joint_bounds[:, 0]-1e-12) and np.all(q <= self.joint_bounds[:, 1]+1e-12))

    def state_from_task(self, q):
        a, b, c = _vec(q)
        return JointState(self.phi_home+math.degrees(a), self.alpha_home+math.degrees(b), self.tip_reference_mm+c*1000)

    def task_from_state(self, state):
        return np.array([math.radians(state.phi_deg-self.phi_home), math.radians(state.alpha_deg-self.alpha_home),
                         (state.length_mm-self.tip_reference_mm)*.001])

    def forward_kinematics(self, q):
        """Analytic needle tip in the shared robot frame (mm)."""
        return self.nominal.forward(self.state_from_task(q), enforce_limits=False)

    def validate_task(self, q):
        q = _vec(q)
        state = self.state_from_task(q)
        self.nominal.forward(state, enforce_limits=False)
        d = self.nominal.b_displacement(state.alpha_deg)
        if abs(self.nominal.alpha_from_b(d, reference_alpha_deg=state.alpha_deg)-state.alpha_deg) > 1e-7:
            raise ValueError("B angle cannot be recovered on a valid linkage branch")
        if self.joint_bounds is not None and not self._in_joint_bounds(q):
            raise ValueError("Requested pose exceeds measured encoder limits (including reserves)")
        return q

    def set_task(self, q):
        q = self.validate_task(q)
        for name, value in zip(TASK_JOINTS, q):
            self.data.qpos[self.qaddr[name]] = value
        for name, multiplier in DEPENDENT_JOINTS.items():
            self.data.qpos[self.qaddr[name]] = multiplier*q[1]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def current_task(self):
        return np.array([self.data.qpos[self.qaddr[name]] for name in TASK_JOINTS])

    def encoders(self, q=None):
        q = self.current_task() if q is None else _vec(q)
        values = list(self.nominal.to_encoders(self.state_from_task(q), integer=False, enforce_limits=False))
        if self.reference_enabled:
            e = self.nominal.encoders
            values[0] = self.reset_encoders[0]+e.direction[0]*math.degrees(q[0])*e.a_counts_per_degree
            values[1] = self.reset_encoders[1]+e.direction[1]*e.bc_counts_per_mm*(
                self.nominal.b_displacement(self.alpha_home+math.degrees(q[1]))-self.nominal.b_displacement(self.alpha_home))
            values[2] = self.reset_encoders[2]+e.direction[2]*q[2]*1000*e.bc_counts_per_mm
        return tuple(values)

    def task_from_encoders(self, values, reference_task=None):
        """Decode B using the supplied target pose, or the current pose as hint.

        Counts alone do not determine the B angle across its turning point.
        The hint preserves the branch; it is not an additional encoder.
        """
        values = _vec(values)
        hint = self.current_task() if reference_task is None else _vec(reference_task)
        alpha_hint = self.state_from_task(hint).alpha_deg
        if self.encoder_bounds is not None and not np.all((values >= self.encoder_bounds[:, 0]) & (values <= self.encoder_bounds[:, 1])):
            raise ValueError("Encoder input exceeds measured limits")
        if self.reference_enabled:
            e = self.nominal.encoders
            normalized = (values-np.array(e.origin))/np.array(e.direction)
            phi = normalized[0]/e.a_counts_per_degree
            alpha = self.nominal.alpha_from_b(normalized[1]/e.bc_counts_per_mm,
                                             reference_alpha_deg=alpha_hint)
            insertion = (values[2]-self.reset_encoders[2])/e.direction[2]/e.bc_counts_per_mm
            q = np.array([math.radians(phi-self.phi_home), math.radians(alpha-self.alpha_home), insertion*.001])
        else:
            q = self.task_from_state(self.nominal.from_encoders(values, enforce_limits=False,
                                                               reference_alpha_deg=alpha_hint))
        return self.validate_task(q)

    def robot_to_world(self, p_mm):
        return self.rcm_mm + self.rotation_robot @ _vec(p_mm)

    def world_to_robot(self, p_mm):
        return self.rotation_robot.T @ (_vec(p_mm)-self.rcm_mm)

    def set_target_world(self, point_mm):
        """Translate the target markers and measurement site; retain pose."""
        robot_mm = self.world_to_robot(point_mm)
        marker = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_marker")
        display_marker = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lesion_display_marker")
        target_m = robot_mm*.001
        self.model.geom_pos[marker] = target_m
        self.model.geom_pos[display_marker] = target_m
        self.model.site_pos[self.sites["lesion"]] = target_m
        self.config["reference_scene"]["target_robot_mm"] = robot_mm.tolist()
        mujoco.mj_forward(self.model, self.data)

    def target_task(self, validate=True):
        point = self.config["reference_scene"]["target_robot_mm"]
        q = self.task_from_state(self.nominal.inverse(point, enforce_limits=False,
                                                     reference_state=self.state_from_task(self.current_task())))
        return self.validate_task(q) if validate else q

    def site_mm(self, name):
        return self.data.site_xpos[self.sites[name]]*1000

    def camera_id(self, name="eye_internal"):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id < 0:
            raise ValueError(f"Unknown camera: {name}")
        return int(camera_id)

    def render_camera(self, name="eye_internal", mode="rgb"):
        """Render a configured camera as RGB, metric depth, or geom segmentation."""
        if mode not in ("rgb", "depth", "segmentation"):
            raise ValueError("mode must be rgb, depth, or segmentation")
        width, height = self.eye_camera["resolution_px"]
        old_znear, old_zfar = self.model.vis.map.znear, self.model.vis.map.zfar
        self.model.vis.map.znear = self.eye_camera["znear_mm"]*.001/self.model.stat.extent
        self.model.vis.map.zfar = self.eye_camera["zfar_mm"]*.001/self.model.stat.extent
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            try:
                if mode == "depth":
                    renderer.enable_depth_rendering()
                elif mode == "segmentation":
                    renderer.enable_segmentation_rendering()
                option = mujoco.MjvOption()
                option.geomgroup[2] = 0  # viewer-only geometry is never a sensor input
                option.sitegroup[:] = 0
                renderer.update_scene(self.data, camera=name, scene_option=option)
                return np.asarray(renderer.render()).copy()
            finally:
                self.model.vis.map.znear, self.model.vis.map.zfar = old_znear, old_zfar

    def camera_pose(self, name="eye_internal"):
        camera_id = self.camera_id(name)
        return self.data.cam_xpos[camera_id].copy(), self.data.cam_xmat[camera_id].reshape(3, 3).copy()

    def needle_eye_geometry(self):
        """Intersect the measured model centerline with its compiled eye.

        This diagnoses geometry, not shaft-volume contact or port suitability.
        No insertion or position limit is imposed by this calculation.
        """
        tip, junction, back, rcm = [self.site_mm(name) for name in
                                   ("needle_tip", "needle_junction", "needle_axis_back", "rcm")]
        axis = (tip-back)/np.linalg.norm(tip-back)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "eye_reference")
        rotation = self.data.geom_xmat[gid].reshape(3, 3)
        center = self.data.geom_xpos[gid]*1000
        radii = self.model.geom_size[gid]*1000
        origin = rotation.T@(rcm-center)/radii
        direction = rotation.T@axis/radii
        a, b, c = float(direction@direction), float(2*origin@direction), float(origin@origin-1)
        discriminant = b*b-4*a*c
        result = {"scope": "needle centerline vs reference ellipsoid; not tissue contact",
                  "rcm_eye_surface_value": float(origin@origin),
                  "rcm_on_eye_surface": abs(c) < 1e-8,
                  "tip_inside_eye": bool(np.sum((rotation.T@(tip-center)/radii)**2) <= 1+1e-9),
                  "junction_inside_eye": bool(np.sum((rotation.T@(junction-center)/radii)**2) <= 1+1e-9),
                  "junction_rcm_axial_offset_mm": float((junction-rcm)@axis),
                  "tip_rcm_distance_mm": float(np.linalg.norm(tip-rcm)),
                  "entry_rcm_distance_mm": None, "entry_world_mm": None,
                  "entry_needle_part": "no_eye_intersection",
                  "distal_centerline_inside_mm": 0., "proximal_centerline_inside_mm": 0.,
                  "entry_needle_diameter_mm": None, "entry_aperture_validated": False}
        if discriminant < -1e-12:
            return result
        roots = sorted(((-b-math.sqrt(max(0., discriminant)))/(2*a),
                        (-b+math.sqrt(max(0., discriminant)))/(2*a)))
        entry, exit_ = roots
        locations = [float((p-rcm)@axis) for p in (back, junction, tip)]
        def overlap(lo, hi):
            return max(0., min(hi, exit_)-max(lo, entry))
        result.update(entry_world_mm=(rcm+entry*axis).tolist(),
                      entry_rcm_distance_mm=abs(entry),
                      distal_centerline_inside_mm=overlap(locations[1], locations[2]),
                      proximal_centerline_inside_mm=overlap(locations[0], locations[1]))
        if abs(entry-locations[1]) < 1e-8:
            part = "junction"
        elif locations[1] < entry < locations[2]+1e-8:
            part = "distal"
        elif locations[0]-1e-8 < entry < locations[1]:
            part = "proximal"
        else:
            part = "outside_modeled_needle"
        result["entry_needle_part"] = part
        if part in ("distal", "proximal", "junction"):
            radius = self.config["needle"]["distal_radius_mm"] if part == "distal" else self.config["needle"]["proximal_radius_mm"]
            result["entry_needle_diameter_mm"] = 2*radius
        return result

    def metrics(self):
        q = self.current_task()
        tip, junction, rcm, back = [self.site_mm(n) for n in ("needle_tip", "needle_junction", "rcm", "needle_axis_back")]
        direction = tip-back
        direction /= np.linalg.norm(direction)
        predicted = self.robot_to_world(self.nominal.forward(self.state_from_task(q), enforce_limits=False))
        robot_encoders = self.encoders(q)
        e = self.nominal.encoders
        normalized = (np.array(robot_encoders)-np.array(e.origin))/np.array(e.direction)
        b_mm, c_mm = normalized[1:]/e.bc_counts_per_mm
        nominal_c_offset = float(c_mm-self.nominal.geometry.c_reference_mm)
        source_length = nominal_c_offset
        if -1/e.bc_counts_per_mm < source_length < 0:
            source_length = 0.  # source one-count L=0 quantization clamp
        controller_point = source_length*np.array(self.nominal.needle_axis(self.state_from_task(q)))
        return {
            "task_A_deg_B_deg_C_mm": [math.degrees(q[0]), math.degrees(q[1]), float(q[2]*1000)],
            "model_robot_encoder_counts_float": list(robot_encoders),
            "encoder_zero_counts_ABC": None if self.reset_encoders is None else self.reset_encoders.tolist(),
            "zero_pose_confirmed_by_user": self.reference_enabled and self.reference_spec.get("zero_pose_confirmed_by_user", False),
            "encoder_physical_units": {
                "A_absolute_deg": float(normalized[0]/e.a_counts_per_degree),
                "B_displacement_mm": float(b_mm),
                "B_absolute_alpha_deg": self.nominal.alpha_from_b(
                    float(b_mm), reference_alpha_deg=self.state_from_task(q).alpha_deg),
                "B_inverse_candidates_deg": list(self.nominal.alpha_candidates_from_b(float(b_mm))),
                "C_absolute_position_mm": float(c_mm),
                "C_offset_from_nominal_reference_mm": nominal_c_offset,
            },
            "controller_model_point_robot_mm": controller_point.tolist(),
            "needle_tip_distance_at_reset_mm": self.tip_reference_mm,
            "physical_needle_reference_confirmed": self.reference_spec.get("physical_needle_reference_confirmed", False),
            "needle_eye_geometry": self.needle_eye_geometry(),
            "distal_tube_length_mm": float(self.distal_mm-self.distal_tip_mm),
            "distal_tip_length_mm": float(self.distal_tip_mm),
            "distal_tip_apex_radius_mm": float(self.tip_apex_radius_mm),
            "distal_tip_shape": self.config.get("needle_dimension_reference", {}).get("tip_shape", "tapered_mesh"),
            "rcm_world_mm": rcm.tolist(), "junction_world_mm": junction.tolist(), "tip_world_mm": tip.tolist(),
            "target_world_mm": self.site_mm("lesion").tolist(),
            "rcm_drift_um": float(np.linalg.norm(rcm-self.rcm_mm)*1000),
            "needle_axis_rcm_distance_um": float(np.linalg.norm(np.cross(rcm-back, direction))*1000),
            "distal_length_mm": float(np.linalg.norm(tip-junction)),
            "fk_agreement_um": float(np.linalg.norm(tip-predicted)*1000),
            "tip_target_error_um": float(np.linalg.norm(tip-self.site_mm("lesion"))*1000),
            "upper_link_closure_um": float(np.linalg.norm(self.site_mm("bridge_end")-self.site_mm("link_F"))*1000),
            "lower_link_closure_um": float(np.linalg.norm(self.site_mm("lower_bridge_end")-self.site_mm("link_F2"))*1000),
            "limits_enabled": self.encoder_bounds is not None,
            "limits_scope": self.config["measured_limits"].get("scope", "calibrated_workspace"),
            "recorded_encoder_bounds": None if self.recorded_encoder_bounds is None else self.recorded_encoder_bounds.tolist(),
            "effective_simulation_encoder_bounds": None if self.encoder_bounds is None else self.encoder_bounds.tolist(),
        }

    def build_xml(self):
        root = ET.Element("mujoco", model="inclined_ABC_RCM_kinematic_reference")
        root.append(ET.Comment(" KINEMATIC REFERENCE. Installation, visible linkage and inertias are not measured. "))
        ET.SubElement(root, "compiler", angle="radian", autolimits="false")
        option = ET.SubElement(root, "option", timestep="0.005", gravity="0 0 0")
        ET.SubElement(option, "flag", contact="disable")
        visual = ET.SubElement(root, "visual")
        sensor_width, sensor_height = self.eye_camera['resolution_px']
        ET.SubElement(visual, "global", offwidth=str(max(1280, sensor_width)),
                      offheight=str(max(960, sensor_height)))
        ET.SubElement(visual, "map", znear="0.002", zfar="6", fogstart="3", fogend="6")
        ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.7 0.7 0.7", specular="0.3 0.3 0.3")
        ET.SubElement(visual, "rgba", haze="0.09 0.11 0.15 1")
        ET.SubElement(root, "statistic", center=_s(self.rcm_mm*.001+np.array([0, -.065, -.025])), extent="0.24")
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", contype="0", conaffinity="0", material="metal", group="0")
        ET.SubElement(default, "site", rgba="0 0 0 0", size="0.00015", group="5")
        ET.SubElement(default, "joint", limited="false", damping="0", armature="0")
        asset = ET.SubElement(root, "asset")
        for name, rgba in {"metal": ".45 .50 .58 1", "shell": ".88 .90 .94 1", "b_link": ".24 .53 .79 1",
                           "c_guide": ".22 .66 .58 1", "needle": ".80 .84 .90 1", "rcm_red": "1 .07 .08 1",
                           "camera_marker": ".05 .95 .95 1", "tip_marker": ".15 1 .25 1",
                           "display_needle": "1 .78 .05 1", "display_target": "1 .88 .05 1",
                           "eye": ".65 .80 .90 .10", "pupil": ".05 .08 .12 .85",
                           "lens": ".70 .85 1.0 .16", "retina": ".78 .30 .34 1",
                           "lesion": ".95 .37 .12 1", "floor": ".12 .15 .20 1"}.items():
            ET.SubElement(asset, "material", name=name, rgba=rgba, specular="0.4", shininess="0.45")
        custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "text", name="scope", data="kinematic_only_not_identified_dynamics")
        recorded = self.config["measured_limits"]["encoder_counts"]
        status = self.config["measured_limits"].get("scope", "measured_enabled") if self.encoder_bounds is not None else (
            "recorded_window_disabled" if all(recorded[axis] is not None for axis in "ABC") else "unmeasured_disabled")
        ET.SubElement(custom, "text", name="limits_status", data=status)
        ET.SubElement(custom, "numeric", name="needle_distal_length_m", data=str(self.distal_mm*.001))
        if self.reset_encoders is not None:
            ET.SubElement(custom, "numeric", name="reported_reset_robot_encoder_ABC", data=_s(self.reset_encoders))
        world = ET.SubElement(root, "worldbody")
        # Optional experimental geometry; default scenes and sensors are unchanged.
        from rcm_occlusion_fixture import add_occluder_to_xml
        add_occluder_to_xml(asset, world, self.config)
        rcm_m = self.rcm_mm*.001
        ET.SubElement(world, "light", pos=_s(rcm_m+[.1, -.1, .3]), dir="-0.3 0.2 -1", directional="true")
        ET.SubElement(world, "geom", name="reference_floor", type="plane", pos=_s(rcm_m+[0, 0, -.175]), size=".6 .6 .01", material="floor")
        ET.SubElement(world, "site", name="rcm", pos=_s(rcm_m))
        ET.SubElement(world, "geom", name="rcm_marker", type="sphere", pos=_s(rcm_m), size=".00065", material="rcm_red")

        def camera_axes(pos, target):
            z = np.asarray(pos, dtype=float)-np.asarray(target, dtype=float)
            z /= np.linalg.norm(z)
            up = np.array([0., 0., 1.])
            if abs(float(up@z)) > .95:
                up = np.array([0., 1., 0.])
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            return np.r_[x, y]

        def camera(name, offset, target, fovy):
            pos, target = rcm_m+np.array(offset), rcm_m+np.array(target)
            ET.SubElement(world, "camera", name=name, pos=_s(pos), xyaxes=_s(camera_axes(pos, target)), fovy=str(fovy))
        camera("overview", [.27, -.19, .18], [0, -.055, -.027], 42)
        camera("side", [.34, -.06, -.035], [0, -.06, -.035], 40)
        camera("needle_close", [.075, .075, .050], [0, .002, -.002], 28)

        reference = ET.SubElement(world, "body", name="robot_reference_frame", pos=_s(rcm_m), quat=_s(rotation_quaternion(self.rotation_robot)))
        eye = self.config["reference_scene"]
        ET.SubElement(reference, "geom", name="eye_reference", type="ellipsoid", pos=_s(_vec(eye["eye_center_robot_mm"])*.001),
                      size=_s(_vec(eye["eye_radii_robot_mm"])*.001), quat=_s(_vec(eye.get("eye_quat_robot_wxyz", [1, 0, 0, 0]), 4)),
                      material="eye", group="2")
        target = _vec(eye["target_robot_mm"])*.001
        eye_camera = self.eye_camera
        camera_position = np.asarray(eye_camera["position_robot_mm"])*.001
        camera_target = np.asarray(eye_camera["target_robot_mm"])*.001
        pupil_robot = np.asarray(eye_camera.get("pupil_robot_mm", eye_camera["position_robot_mm"]), dtype=float)*.001
        optical_axis = camera_target-pupil_robot
        optical_axis /= np.linalg.norm(optical_axis)
        pupil_axes = camera_axes(pupil_robot, camera_target).reshape(2, 3)
        # Four small markers show the anterior pupil aperture used by the
        # external monocular microscope. The actual optical aperture remains
        # conceptual; these markers do not participate in control.
        for index, offset in enumerate((pupil_axes[0], -pupil_axes[0], pupil_axes[1], -pupil_axes[1])):
            ET.SubElement(reference, "geom", name=f"pupil_ring_{index}", type="sphere",
                          pos=_s(pupil_robot+offset*.0025), size=".00025", material="pupil", group="2")
        ET.SubElement(reference, "geom", name="lens_visual", type="ellipsoid",
                      pos=_s(pupil_robot+optical_axis*.006), size=".003 .003 .0015",
                      material="lens", group="2")
        # The physical target is a single sphere with no auxiliary reticle points.
        # It may disappear when the needle is directly in front of it.
        ET.SubElement(reference, "geom", name="lesion_marker", type="sphere", pos=_s(target), size=".00015", material="lesion", group="3")
        ET.SubElement(reference, "site", name="lesion", pos=_s(target))
        # Viewer-only target aid. Group 2 is disabled in sensor renders, so
        # the enlarged marker cannot change segmentation, depth or control.
        display_target_radius_mm = float(eye.get("display_target_radius_mm", 0.5))
        ET.SubElement(reference, "geom", name="lesion_display_marker", type="sphere",
                      pos=_s(target), size=str(display_target_radius_mm*.001),
                      material="display_target", group="2")
        # The RGB-only fundus backdrop follows the curved posterior eye cap.
        # Offset it outward along each surface normal so targets and the needle
        # tip remain in front. This visual backdrop is not a tissue/contact model.
        retina_body = ET.SubElement(reference, "body", name="retina_visual_frame",
                                    pos=_s(_vec(eye["eye_center_robot_mm"])*.001),
                                    quat=_s(_vec(eye.get("eye_quat_robot_wxyz", [1, 0, 0, 0]), 4)))
        retina_radii = _vec(eye["eye_radii_robot_mm"])*.001
        # The offset separates the optical backdrop from the target plane
        # without changing target positions on the actual eye ellipsoid.
        retina_offset_m = .008
        retina_levels, retina_azimuths = np.linspace(.55, 1., 10), 32
        retina_rings = [[(math.sqrt(max(0., 1.-float(level)**2))*math.cos(2.*math.pi*index/retina_azimuths),
                          math.sqrt(max(0., 1.-float(level)**2))*math.sin(2.*math.pi*index/retina_azimuths),
                          float(level))
                         for index in range(retina_azimuths)]
                        for level in retina_levels]
        def retina_vertex(unit):
            point = retina_radii*np.asarray(unit, dtype=float)
            normal = np.asarray(unit, dtype=float)/retina_radii
            normal /= np.linalg.norm(normal)
            return tuple(point+retina_offset_m*normal)

        mesh_vertices = [retina_vertex(value) for ring in retina_rings for value in ring]
        mesh_faces = []
        for ring_index in range(len(retina_rings)-1):
            for index in range(retina_azimuths):
                nxt_index = (index+1) % retina_azimuths
                a, b = ring_index*retina_azimuths+index, ring_index*retina_azimuths+nxt_index
                c, d = (ring_index+1)*retina_azimuths+index, (ring_index+1)*retina_azimuths+nxt_index
                # The external camera views the posterior cap from its inner
                # side, so reverse the outward triangle winding.
                mesh_faces.extend(((a, d, b), (a, c, d)))
        apex_vertex = len(mesh_vertices)
        mesh_vertices.append(retina_vertex((0., 0., 1.)))
        last_ring_start = (len(retina_rings)-1)*retina_azimuths
        for index in range(retina_azimuths):
            mesh_faces.append((last_ring_start+index, apex_vertex,
                               last_ring_start+(index+1) % retina_azimuths))
        ET.SubElement(asset, "mesh", name="retina_visual_mesh",
                      vertex=_s([coordinate for vertex in mesh_vertices for coordinate in vertex]),
                      face=" ".join(" ".join(str(value) for value in face) for face in mesh_faces))
        ET.SubElement(retina_body, "geom", name="retina_visual", type="mesh",
                      mesh="retina_visual_mesh",
                      # Group 5 is reserved for the RGB-only fundus backdrop.
                      # The visual-servo depth/segmentation passes disable it,
                      # so this backdrop cannot occlude the needle tip.
                      material="retina", group="5")
        camera_axis = camera_target-camera_position
        camera_axis /= np.linalg.norm(camera_axis)
        ET.SubElement(reference, "camera", name="eye_internal",
                      pos=_s(camera_position), xyaxes=_s(camera_axes(camera_position, camera_target)),
                      fovy=str(eye_camera["fovy_deg"]))
        ET.SubElement(reference, "site", name="eye_camera_site", pos=_s(camera_position), size="0.0009",
                      rgba="0.05 0.95 0.95 1", group="5")
        ET.SubElement(reference, "geom", name="eye_camera_marker", type="sphere", pos=_s(camera_position),
                      size="0.00075", material="camera_marker", group="2")
        ET.SubElement(reference, "geom", name="eye_camera_axis", type="cylinder",
                      fromto=_s(np.r_[camera_position, camera_position+camera_axis*.004]),
                      size="0.00012", material="camera_marker", group="2")
        mount = ET.SubElement(world, "body", name="fixed_inclined_mount", pos=_s(rcm_m), quat=_s(rotation_quaternion(self.rotation_mount)))
        visual_scale = float(self.config["illustrative_linkage"].get("visual_scale", 1.0))

        def box(parent, name, center, full_size, material="metal", group="0", scale=1.0):
            return ET.SubElement(parent, "geom", name=name, type="box", pos=_s(np.array(center)*scale*.001),
                                 size=_s(np.array(full_size)*scale*.0005), material=material, group=group)

        def rod(parent, name, start, end, radius, material="metal", group="0", scale=1.0):
            return ET.SubElement(parent, "geom", name=name, type="cylinder", fromto=_s(np.r_[start, end]*scale*.001),
                                 size=str(radius*scale*.001), material=material, group=group)

        def tapered_tip_mesh(name, base_radius_mm, apex_radius_mm, length_mm, sides=8):
            """Create a local tapered tip mesh along -Z; apex remains at -length."""
            vertices = []
            for z_mm, radius_mm in ((0., base_radius_mm), (-length_mm, apex_radius_mm)):
                for index in range(sides):
                    angle = 2.*math.pi*index/sides
                    vertices.extend((radius_mm*math.cos(angle)*.001,
                                     radius_mm*math.sin(angle)*.001,
                                     z_mm*.001))
            vertices.extend((0., 0., 0.))
            vertices.extend((0., 0., -length_mm*.001))
            faces = []
            for index in range(sides):
                nxt = (index+1) % sides
                faces.extend(((index, nxt, sides+nxt), (index, sides+nxt, sides+index)))
                faces.append((2*sides, nxt, index))
                faces.append((2*sides+1, sides+index, sides+nxt))
            return ET.SubElement(asset, "mesh", name=name, vertex=_s(vertices),
                                 face=" ".join(" ".join(str(value) for value in face) for face in faces))

        tapered_tip_mesh("needle_distal_tip_mesh", self.config["needle"]["distal_radius_mm"],
                         self.tip_apex_radius_mm, self.distal_tip_mm)

        def site(parent, name, pos):
            ET.SubElement(parent, "site", name=name, pos=_s(np.array(pos)*.001))

        def inertial(parent, mass):
            ET.SubElement(parent, "inertial", pos="0 0 0", mass=str(mass), diaginertia="0.0001 0.0001 0.0001")

        # Positive placeholder inertias only satisfy the MJCF compiler; they
        # have no measured physical interpretation and are not used for motion.
        box(mount, "fixed_base_plate", [0, -138, -39], [66, 72, 8], scale=visual_scale)
        box(mount, "fixed_base_case", [0, -137, -20], [48, 56, 31], "shell", "1", visual_scale)
        box(mount, "fixed_base_top", [0, -120, -4], [44, 42, 15], "shell", "1", visual_scale)
        rod(mount, "A_bearing", [0, -148, 0], [0, -102, 0], 9, scale=visual_scale)
        a_quat = rotation_quaternion(installation_rotation([0, -self.phi_home, 0]))
        b_quat = rotation_quaternion(installation_rotation([self.b_initial_deg, 0, 0]))
        counter_b_quat = rotation_quaternion(installation_rotation([-self.b_initial_deg, 0, 0]))
        a_body = ET.SubElement(mount, "body", name="A_stage", quat=_s(a_quat))
        inertial(a_body, .4)

        ui = self.config["demonstration"]["ui_ranges"]
        joint_ranges = self.joint_bounds if self.joint_bounds is not None else np.array([
            np.radians(_vec(ui["A_deg"], 2)), np.radians(_vec(ui["B_deg"], 2)), _vec(ui["C_mm"], 2)*.001])
        for pair in joint_ranges:
            if pair[0] >= pair[1]:
                raise ValueError("UI joint ranges must increase")

        def task_joint(parent, index, kind, axis):
            ET.SubElement(parent, "joint", name=TASK_JOINTS[index], type=kind, axis=axis,
                          limited="true" if self.joint_bounds is not None else "false", range=_s(joint_ranges[index]), group="0")

        task_joint(a_body, 0, "hinge", "0 -1 0")
        box(a_body, "A_rotating_hub", [0, -121, 2], [37, 25, 11], scale=visual_scale)
        l = self.config["illustrative_linkage"]
        rear, spacing, upper, lower, offset = [l[n] for n in ("rear_distance_mm", "rear_pivot_spacing_mm", "upper_crank_mm", "lower_crank_mm", "guide_offset_mm")]
        rear_v, spacing_v, upper_v, lower_v, offset_v = [v*visual_scale for v in (rear, spacing, upper, lower, offset)]
        site(a_body, "link_O1", [0, -rear_v, 0])
        site(a_body, "link_O2", [0, -rear_v+spacing_v, 0])
        # Exact RCM task chain, with both orientation hinges at the RCM.
        b_body = ET.SubElement(a_body, "body", name="B_RCM_task_frame", quat=_s(b_quat))
        inertial(b_body, .25)
        task_joint(b_body, 1, "hinge", "1 0 0")
        site(b_body, "link_F", [0, 0, upper_v])
        site(b_body, "link_F2", [0, 0, lower_v])
        rod(b_body, "fore_parallel_link", [0, 0, lower], [0, 0, upper], 3.5, scale=visual_scale)
        for name, z in (("upper", upper), ("lower", lower)):
            rod(b_body, f"guide_support_{name}", [0, 0, z], [0, -offset, z], 3, scale=visual_scale)
        box(b_body, "C_fixed_spine", [0, -offset, (lower+88)/2], [18, 7, 88-lower], scale=visual_scale)
        box(b_body, "C_motor_case", [0, -offset, 83], [29, 23, 28], "shell", "1", visual_scale)
        box(b_body, "C_motor_cap", [0, -offset, 103], [27, 23, 12], group="1", scale=visual_scale)
        rod(b_body, "C_motor_end", [0, -offset, 108], [0, -offset, 111], 12, group="1", scale=visual_scale)
        for x, label in ((-11, "left"), (11, "right")):
            rod(b_body, f"C_guide_{label}", [x, -offset, 26], [x, -offset, 79], 1.6, "c_guide", scale=visual_scale)
            box(b_body, f"C_guide_mount_{label}", [x, -offset, 77], [5, 6, 5], scale=visual_scale)
        site(b_body, "guide_top", [0, -offset_v, 79*visual_scale])
        site(b_body, "guide_bottom", [0, -offset_v, 26*visual_scale])

        c_body = ET.SubElement(b_body, "body", name="C_carriage_and_needle", pos=_s([0, 0, -(self.tip_reference_mm-self.distal_mm)*.001]))
        inertial(c_body, .05)
        task_joint(c_body, 2, "slide", "0 0 -1")
        d, prox = self.distal_mm, self.config["needle"]["proximal_length_mm"]
        display_scale = float(self.config["needle"].get("display_scale", 1.0))
        box(c_body, "C_moving_carriage", [0, -offset/2, 46-d], [27, offset+6, 8], scale=visual_scale)
        rod(c_body, "needle_holder", [0, 0, 42-d], [0, 0, 30-d], 4.3)
        rod(c_body, "needle_adapter", [0, 0, 30-d], [0, 0, prox], 2.4)
        rod(c_body, "needle_proximal", [0, 0, prox], [0, 0, 0], self.config["needle"]["proximal_radius_mm"], "needle", "3")
        tube_end = -(d-self.distal_tip_mm)
        rod(c_body, "needle_distal", [0, 0, 0], [0, 0, tube_end], self.config["needle"]["distal_radius_mm"], "needle", "3")
        # Group 2 is enabled only by the human-facing MuJoCo viewer. The
        # visual aid is therefore absent from RGB/depth/segmentation sensors,
        # while making the complete needle easy to follow in the large model.
        rod(c_body, "needle_display_proximal", [0, 0, prox], [0, 0, 0],
            self.config["needle"]["proximal_radius_mm"]*display_scale,
            "display_needle", "2")
        rod(c_body, "needle_display_distal", [0, 0, 0], [0, 0, tube_end],
            self.config["needle"]["distal_radius_mm"]*display_scale,
            "display_needle", "2")
        ET.SubElement(c_body, "geom", name="needle_display_tip", type="sphere",
                      pos=_s([0, 0, -d*.001]), size=str(max(self.tip_apex_radius_mm,
                      self.config["needle"]["tip_marker_radius_mm"])*display_scale*.001),
                      material="display_needle", group="2")
        ET.SubElement(c_body, "geom", name="needle_distal_tip", type="mesh", mesh="needle_distal_tip_mesh",
                      pos=_s([0, 0, tube_end*.001]), material="needle", group="3")
        tip_marker_radius_mm = float(self.config["needle"].get(
            "tip_marker_radius_mm", max(self.tip_apex_radius_mm, .01)))
        ET.SubElement(c_body, "geom", name="needle_tip_marker", type="sphere", pos=_s([0, 0, -d*.001]),
                      size=str(tip_marker_radius_mm*.001), material="tip_marker", group="4")
        # Viewer-only enlarged marker. Sensor render passes disable group 2;
        # control uses the physical distal segment and calibrated tip geometry.
        tip_visual_radius_mm = float(self.config["needle"].get(
            "tip_visual_marker_radius_mm", max(tip_marker_radius_mm, .08)))
        ET.SubElement(c_body, "geom", name="needle_tip_visual_marker", type="sphere", pos=_s([0, 0, -d*.001]),
                      size=str(tip_visual_radius_mm*.001), material="tip_marker", group="2")
        site(c_body, "needle_junction", [0, 0, 0])
        site(c_body, "needle_tip", [0, 0, -d])
        site(c_body, "needle_axis_back", [0, 0, prox])

        # Dependent visual chain: the 120 mm bridges translate with B but
        # remain parallel to their home orientations. This is not hardware CAD.
        rear_body = ET.SubElement(a_body, "body", name="equivalent_rear_crank", pos=_s([0, -rear_v*.001, 0]), quat=_s(b_quat))
        inertial(rear_body, .1)
        ET.SubElement(rear_body, "joint", name="visual_B_rear", type="hinge", axis="1 0 0", group="3")
        rod(rear_body, "B_rear_crank", [0, 0, 0], [0, 0, upper], 3.2, "b_link", scale=visual_scale)
        site(rear_body, "link_P", [0, 0, upper_v])
        site(rear_body, "link_P2", [0, 0, lower_v])
        for name, z, width, height in (("bridge", upper, 22, 9), ("lower_bridge", lower, 12, 7)):
            bridge = ET.SubElement(rear_body, "body", name=f"equivalent_{name}", pos=_s([0, 0, z*visual_scale*.001]), quat=_s(counter_b_quat))
            inertial(bridge, .15)
            ET.SubElement(bridge, "joint", name=f"visual_B_{name}", type="hinge", axis="1 0 0", group="3")
            box(bridge, f"B_{name}", [0, rear/2, 0], [width, rear, height], "metal" if name=="bridge" else "b_link", scale=visual_scale)
            site(bridge, f"{name}_end", [0, rear_v, 0])
            if name == "bridge":
                box(bridge, "moving_middle_housing", [0, rear-65, 0], [29, 89, 19], group="1", scale=visual_scale)
        aux_body = ET.SubElement(a_body, "body", name="equivalent_aux_crank", pos=_s([0, (-rear_v+spacing_v)*.001, 0]), quat=_s(b_quat))
        inertial(aux_body, .08)
        ET.SubElement(aux_body, "joint", name="visual_B_aux", type="hinge", axis="1 0 0", group="3")
        rod(aux_body, "B_aux_crank", [0, 0, 0], [0, 0, upper], 2.4, scale=visual_scale)
        site(aux_body, "link_Q", [0, 0, upper_v])
        equality = ET.SubElement(root, "equality")
        for name, factor in DEPENDENT_JOINTS.items():
            ET.SubElement(equality, "joint", name=f"couple_{name}", joint1=name, joint2="B_pitch", polycoef=f"0 {factor:g} 0 0 0")
        sensor = ET.SubElement(root, "sensor")
        for name in ("rcm", "needle_tip", "needle_junction", "lesion"):
            ET.SubElement(sensor, "framepos", name=f"{name}_world_position", objtype="site", objname=name)
        for name in TASK_JOINTS:
            ET.SubElement(sensor, "jointpos", name=f"{name}_position", joint=name)
        ET.indent(root, space="  ")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


class Motion:
    """Finite, smooth task-space keyframes; evaluated without physics stepping."""
    def __init__(self, start, segments):
        self.start = _vec(start).copy()
        self.segments = [(float(duration), _vec(target).copy()) for duration, target in segments]
        if not self.segments or any(duration <= 0 for duration, _ in self.segments):
            raise ValueError("Motion needs positive segment durations")
        self.duration = sum(duration for duration, _ in self.segments)

    def sample(self, t):
        t, previous = max(0.0, float(t)), self.start
        for duration, target in self.segments:
            if t <= duration:
                w = .5-.5*math.cos(math.pi*t/duration)
                return (1-w)*previous+w*target
            t -= duration
            previous = target
        return self.segments[-1][1].copy()


def demo_motion(reference, mode):
    home = reference.home
    spec = reference.config["demonstration"]
    segments = []
    for axis in ("abc" if mode == "abc" else mode):
        index = "abc".index(axis)
        amplitude = math.radians(spec[f"{axis}_amplitude_deg"]) if axis != "c" else spec["c_stroke_mm"]*.001
        lo, hi = home.copy(), home.copy()
        hi[index] += amplitude
        if axis == "c":
            segments.extend([(2.5, hi), (1., hi), (2.5, home)])
        else:
            lo[index] -= amplitude
            segments.extend([(1.5, lo), (3., hi), (1.5, home)])
    for _, target in segments:
        reference.validate_task(target)
    return Motion(home, segments)


def navigation_motion(reference, reach):
    start, target = reference.current_task(), reference.target_task(validate=reach)
    aimed = start.copy()
    aimed[:2] = target[:2]
    reference.validate_task(aimed)
    segments = [(1.5, aimed)]
    if reach:
        segments.append((2., target))
    return Motion(start, segments)


def save_png(path, rgb):
    """Write MuJoCo's uint8 RGB render using the standard library."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("Expected RGB image")
    def chunk(kind, data):
        return struct.pack(">I", len(data))+kind+data+struct.pack(">I", zlib.crc32(kind+data) & 0xffffffff)
    payload = b"".join(b"\0"+row.tobytes() for row in rgb)
    result = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    result += chunk(b"IDAT", zlib.compress(payload)) + chunk(b"IEND", b"")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(result)


def render(reference, path, camera="overview"):
    with mujoco.Renderer(reference.model, height=960, width=1280) as renderer:
        option = mujoco.MjvOption()
        option.sitegroup[:] = 0
        if camera == "eye_internal":
            # The microscope view is an optical interior view through the
            # pupil. Hide the external eye shell and robot housing so the
            # retina/task region and needle remain visible in the FOV.
            option.geomgroup[0] = 0
            option.geomgroup[1] = 0
            option.geomgroup[2] = 0
            option.geomgroup[3] = 1
            option.geomgroup[4] = 1
            option.geomgroup[5] = 1  # fundus backdrop for the displayed eye view
        renderer.update_scene(reference.data, camera=camera, scene_option=option)
        save_png(path, renderer.render())
