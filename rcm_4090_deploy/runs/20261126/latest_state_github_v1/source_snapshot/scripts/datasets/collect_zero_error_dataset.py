"""Rebuild the 608 training episodes with bit-exact v12 terminal targets.

The source trajectory is used only for its first and last A/B/C states.  Each
v12 episode starts at that source start through an explicit simulator reset,
uses a current-frame strict RGB-D teacher when possible, and always finishes
through MotorBackend at the integer-encoder pose reproduced from the source
terminal state.  Generated RGB is downsampled to 224x224 immediately; strict
perception remains at the configured calibrated rendering resolution.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from rcm.reference import HERE
from rcm.scene_v9 import RCMSceneV9
from rcm.strict_rgbd import StrictRGBDSensor
from rcm.visual_servo import VisualServoController, visual_servo_settings
from rcm.waypoint_dataset import EpisodeWriter, make_policy_task, write_manifest
from scripts.applications.run_waypoint_comparison import (
    guard_scene_goal_inverse,
    perception_record,
)


UNITS = np.array([180.0 / np.pi, 180.0 / np.pi, 1000.0], dtype=np.float64)
SI_UNITS = np.array([np.pi / 180.0, np.pi / 180.0, 0.001], dtype=np.float64)
DEFAULT_SOURCE = HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224"
DEFAULT_TRAINING_OUTPUT = HERE / "data" / "lerobot_v12_zero_error_train_224"
IMAGE_SHAPE = (224, 224, 3)
EXECUTION_SOURCES = {
    "strict_rgbd_teacher",
    "exact_terminal_lock",
    "privileged_joint_fallback",
    "hold",
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceEpisode:
    episode_index: int
    source_dataset: str
    source_episode_index: int
    num_frames: int
    start_state_A_deg_B_deg_C_mm: np.ndarray
    terminal_state_A_deg_B_deg_C_mm: np.ndarray


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes source root: {relative}")
    return path


def load_source_episodes(source_dataset):
    """Load and validate the ordered 608 source episode starts/terminals."""
    source_root = Path(source_dataset).resolve()
    metadata = json.loads(
        (source_root / "metadata.json").read_text(encoding="utf-8")
    )
    source_metadata = {}
    source_occurrence = {}
    episodes = []
    with np.load(source_root / "data" / "dataset.npz", allow_pickle=False) as data:
        required = {"episode_index", "frame_index", "state"}
        if not required.issubset(data.files):
            raise ValueError(
                f"v12 source arrays changed; expected at least {sorted(required)}, "
                f"got {sorted(data.files)}"
            )
        episode_indices = data["episode_index"].astype(np.int64)
        frame_indices = data["frame_index"].astype(np.int64)
        states = data["state"].astype(np.float64)
        if len(episode_indices) != len(frame_indices) or len(states) != len(episode_indices):
            raise ValueError("v12 source arrays are not row aligned")
        expected_episodes = list(range(int(metadata["num_episodes"])))
        actual_episodes = sorted(int(value) for value in np.unique(episode_indices))
        if actual_episodes != expected_episodes:
            raise ValueError("v12 source episode indexes are not contiguous")
        if len(metadata["episodes"]) != metadata["num_episodes"]:
            raise ValueError("v12 source episode metadata count is inconsistent")
        if int(data["episode_index"].size) != int(metadata["num_frames"]):
            raise ValueError("v12 source frame count is inconsistent")
        for episode_index, manifest_row in enumerate(metadata["episodes"]):
            rows = np.flatnonzero(episode_indices == episode_index)
            expected_frames = np.arange(len(rows), dtype=np.int64)
            if len(rows) == 0 or not np.array_equal(frame_indices[rows], expected_frames):
                raise ValueError(f"v12 source episode {episode_index} frame indexes are invalid")
            if int(manifest_row["episode_index"]) != episode_index:
                raise ValueError(f"v12 source episode {episode_index} metadata index changed")
            if int(manifest_row["num_frames"]) != len(rows):
                raise ValueError(f"v12 source episode {episode_index} frame count changed")
            source_name = str(manifest_row["source_dataset"])
            if source_name not in source_metadata:
                source_path = _inside(source_root.parent.parent, source_name)
                source_payload = json.loads(
                    (source_path / "metadata.json").read_text(encoding="utf-8")
                )
                source_metadata[source_name] = source_payload["episodes"]
                source_occurrence[source_name] = 0
            local_index = source_occurrence[source_name]
            local_rows = source_metadata[source_name]
            if local_index >= len(local_rows):
                raise ValueError(f"source episode mapping exceeds {source_name}")
            local_row = local_rows[local_index]
            if int(local_row["episode_index"]) != local_index:
                raise ValueError(f"source episode indexes changed in {source_name}")
            if int(local_row["num_frames"]) != len(rows):
                raise ValueError(
                    f"source frame count mismatch for episode {episode_index}: "
                    f"{local_row['num_frames']} != {len(rows)}"
                )
            start = states[rows[0]].copy()
            terminal = states[rows[-1]].copy()
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(terminal)):
                raise ValueError(f"source episode {episode_index} has non-finite states")
            episodes.append(SourceEpisode(
                episode_index=episode_index,
                source_dataset=source_name,
                source_episode_index=local_index,
                num_frames=len(rows),
                start_state_A_deg_B_deg_C_mm=start,
                terminal_state_A_deg_B_deg_C_mm=terminal,
            ))
            source_occurrence[source_name] += 1
    return episodes


def _scene_config(config):
    value = copy.deepcopy(config)
    value["motor_execution"] = {
        **value.get("motor_execution", {}),
        "enabled": False,
    }
    return value


def stabilize_terminal_pose(config, source_terminal_state, source_start_state=None,
                            max_steps=512):
    """Reach the source terminal through the same protocol anchor as collection.

    The vendor driver's handle conversion can quantize the old terminal B/C
    counts by a few units.  Planning therefore starts at the v12 episode start,
    commands the old terminal encoder through MotorBackend, and defines the
    exact target from the protocol-reachable pose that the collection scene will
    reproduce with the identical anchor.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = np.asarray(source_terminal_state, dtype=np.float64)
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError("source terminal state must be three finite A/B/C values")
    start_state = (
        state.copy()
        if source_start_state is None
        else np.asarray(source_start_state, dtype=np.float64)
    )
    if start_state.shape != (3,) or not np.all(np.isfinite(start_state)):
        raise ValueError("source start state must be three finite A/B/C values")
    scene = RCMSceneV9(_scene_config(config))
    try:
        scene.set_task(start_state * SI_UNITS)
        source_nominal_target_world_m = scene.site_mm("lesion").copy() * 0.001
        requested_terminal_robot_ABCE = (
            tuple(int(round(float(value))) for value in scene.encoders(state * SI_UNITS))
            + (0,)
        )
        terminal_reference_task = scene.task_from_encoders(
            requested_terminal_robot_ABCE[:3],
            reference_task=state * SI_UNITS,
        )
        scene.enable_motor_execution({"enabled": True})
        scene.reset_motor_time(0.0)
        collection_start_actual = scene.current_task().copy()
        dt = float(scene.motor_execution.control_period_s)
        steps = []
        for _ in range(max_steps):
            before = list(scene.motor_robot())
            actual = scene.motor_execution.advance_robot(
                requested_terminal_robot_ABCE,
                dt,
                requested_task=terminal_reference_task,
            )
            scene._apply_task_pose(actual)
            step = scene.motor_execution.last_record
            steps.append({
                "before_robot_ABCE": before,
                "after_robot_ABCE": list(scene.motor_robot()),
                "target_robot_ABCE": list(scene.motor_execution.backend.target),
                "step_index": int(step["step_index"]),
            })
            if scene.motor_execution.settled():
                break
        else:
            raise RuntimeError("terminal encoder did not settle")
        endpoint_robot_ABCE = tuple(scene.motor_execution.backend.target)
        if tuple(scene.motor_robot()) != endpoint_robot_ABCE:
            raise RuntimeError("terminal encoder replay did not reach its target")
        endpoint_task = scene.task_from_encoders(
            endpoint_robot_ABCE[:3],
            reference_task=scene.current_task(),
        )
        source_terminal_tip_world_m = (
            scene.data.site_xpos[scene.sites["needle_tip"]].copy()
        )
        scene.set_target_from_needle_tip()
        terminal_tip_world_m = (
            scene.data.site_xpos[scene.sites["needle_tip"]].copy()
        )
        terminal_target_world_m = (
            scene.data.site_xpos[scene.sites["lesion"]].copy()
        )
        if not np.array_equal(terminal_tip_world_m, terminal_target_world_m):
            raise RuntimeError("strict target helper did not preserve exact coordinates")
        error_um = float(np.linalg.norm(
            (scene.site_mm("needle_tip") - scene.site_mm("lesion"))
        ) * 1000.0)
        if error_um != 0.0:
            raise RuntimeError(f"strict target helper produced non-zero error: {error_um}")
        return {
            "endpoint_robot_ABCE": list(endpoint_robot_ABCE),
            "requested_terminal_robot_ABCE": list(requested_terminal_robot_ABCE),
            "endpoint_task_rad_rad_m": endpoint_task.tolist(),
            "endpoint_settle_steps": steps,
            "collection_start_state_A_deg_B_deg_C_mm": (
                collection_start_actual * UNITS
            ).tolist(),
            "source_terminal_tip_world_m": source_terminal_tip_world_m.tolist(),
            "source_nominal_target_world_m": source_nominal_target_world_m.tolist(),
            "terminal_tip_world_m": terminal_tip_world_m.tolist(),
            "terminal_target_world_m": terminal_target_world_m.tolist(),
            "target_minus_source_nominal_target_m": (
                terminal_target_world_m - source_nominal_target_world_m
            ).tolist(),
            "target_minus_source_terminal_tip_m": (
                terminal_target_world_m - source_terminal_tip_world_m
            ).tolist(),
            "final_error_um": error_um,
        }
    finally:
        scene.close()


def limited_joint_path(start_task, terminal_task, maximum_velocity, maximum_acceleration,
                       control_period_s):
    """Return a straight A/B/C path with explicit velocity/acceleration limits."""
    start = np.asarray(start_task, dtype=np.float64)
    terminal = np.asarray(terminal_task, dtype=np.float64)
    vmax = np.asarray(maximum_velocity, dtype=np.float64)
    amax = np.asarray(maximum_acceleration, dtype=np.float64)
    if start.shape != (3,) or terminal.shape != (3,):
        raise ValueError("start and terminal tasks must be length 3")
    if vmax.shape != (3,) or amax.shape != (3,):
        raise ValueError("velocity and acceleration limits must be length 3")
    if (not np.all(np.isfinite(start)) or not np.all(np.isfinite(terminal))
            or np.any(vmax <= 0) or np.any(amax <= 0)):
        raise ValueError("joint path inputs must be finite with positive limits")
    dt = float(control_period_s)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("control period must be positive")
    delta = terminal - start
    distance = float(np.max(np.abs(delta)))
    if distance == 0.0:
        return [terminal.copy()]
    active = np.abs(delta) > 0.0
    # ``distance`` is measured on the largest-delta axis.  Convert each
    # per-axis bound into the equivalent scalar bound for this line.
    scalar_vmax = float(np.min(
        vmax[active] * distance / np.abs(delta[active])
    ))
    scalar_amax = float(np.min(
        amax[active] * distance / np.abs(delta[active])
    ))
    if distance <= scalar_vmax * scalar_vmax / scalar_amax:
        peak_velocity = math.sqrt(scalar_amax * distance)
        acceleration_time = peak_velocity / scalar_amax
        duration = 2.0 * acceleration_time
    else:
        peak_velocity = scalar_vmax
        acceleration_time = peak_velocity / scalar_amax
        duration = distance / peak_velocity + acceleration_time

    def progress(t):
        if t <= acceleration_time:
            return 0.5 * scalar_amax * t * t
        if t <= duration - acceleration_time:
            return (0.5 * scalar_amax * acceleration_time * acceleration_time
                    + peak_velocity * (t - acceleration_time))
        remaining = max(0.0, duration - t)
        return distance - 0.5 * scalar_amax * remaining * remaining

    # Collector commands consume one full control period each.  Sample the
    # profile on that same clock so a shortened final interval cannot exceed
    # the declared velocity limit.
    sample_count = max(1, int(math.ceil(duration / dt - 1e-12)))
    times = np.arange(1, sample_count + 1, dtype=np.float64) * dt
    commands = []
    for timestamp in times:
        fraction = min(1.0, max(0.0, progress(float(timestamp)) / distance))
        commands.append(start + delta * fraction)
    commands[-1] = terminal.copy()
    return commands


def _downsample_rgb(rgb, shape=IMAGE_SHAPE):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    if image.size != (int(shape[1]), int(shape[0])):
        image = image.resize(
            (int(shape[1]), int(shape[0])),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(image, dtype=np.uint8).copy()


class _EpisodeRecorder:
    def __init__(self, scene, writer, motor, episode_index, source_episode_index):
        self.scene = scene
        self.writer = writer
        self.motor = motor
        self.episode_index = int(episode_index)
        self.source_episode_index = int(source_episode_index)
        self.pending = None
        self.audits = []
        self.source_counts = {name: 0 for name in EXECUTION_SOURCES}

    def _metrics(self):
        return self.scene.metrics()

    def _store(self, pending, action, execution_source, phase, terminal):
        if execution_source not in EXECUTION_SOURCES:
            raise ValueError(f"unknown execution source: {execution_source}")
        image = pending["image_224"]
        transition = None if terminal else pending["motor_transition_index"]
        self.writer.add(
            pending["time_s"],
            pending["state"],
            action,
            "P1",
            image,
            terminal=terminal,
            phase=phase,
            visibility=None,
            perception=pending["perception"],
        )
        self.audits.append({
            "frame_index": len(self.audits),
            "source_episode_index": self.source_episode_index,
            "timestamp_s": pending["time_s"],
            "robot_ABCE": pending["robot"],
            "state_A_deg_B_deg_C_mm": pending["state"].tolist(),
            "rgb_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "motor_transition_index": transition,
            "execution_source": execution_source,
            "phase": phase,
            "terminal": bool(terminal),
            "strict_zero_error": True,
            "tip_world_m": pending["tip_world_m"],
            "target_world_m": pending["target_world_m"],
            "tip_target_error_um": pending["tip_target_error_um"],
            "control_frame_index": pending["control_frame_index"],
        })
        self.source_counts[execution_source] += 1

    def capture(self, observation, execution_source, phase, control_frame_index=None):
        if self.pending is not None:
            step = self.motor.last_record
            if step is None or not np.isclose(
                    step["start_time_s"], self.pending["time_s"], atol=1e-10, rtol=0):
                raise RuntimeError("motor/image transition timing mismatch")
            action = np.asarray(step["commanded_task_rad_rad_m"]) * UNITS
            pending = dict(self.pending)
            pending["motor_transition_index"] = int(step["step_index"])
            self._store(
                pending,
                action,
                execution_source,
                phase,
                terminal=False,
            )
        state_si = self.scene.current_task().copy()
        perception = perception_record(self.scene, observation, phase)
        perception["control_frame_index"] = (
            None if control_frame_index is None else int(control_frame_index)
        )
        self.pending = {
            "time_s": float(self.motor.time_s),
            "state": state_si * UNITS,
            "robot": list(self.scene.motor_robot()),
            "image_224": _downsample_rgb(observation["rgb"]),
            "perception": perception,
            "motor_transition_index": (
                None if self.motor.last_record is None
                else int(self.motor.last_record["step_index"])
            ),
            "control_frame_index": (
                None if control_frame_index is None else int(control_frame_index)
            ),
            "tip_world_m": self.scene.data.site_xpos[
                self.scene.sites["needle_tip"]
            ].tolist(),
            "target_world_m": self.scene.data.site_xpos[
                self.scene.sites["lesion"]
            ].tolist(),
            "tip_target_error_um": float(self._metrics()["tip_target_error_um"]),
        }

    def finalize_terminal(self):
        if self.pending is None:
            raise RuntimeError("cannot finalize an episode without a pending frame")
        self._store(
            self.pending,
            self.pending["state"],
            "hold",
            "hold",
            terminal=True,
        )
        self.pending = None


def _expected_control_shape(config):
    width, height = config["reference_scene"]["eye_camera"]["resolution_px"]
    return int(height), int(width), 3


def collect_episode(config, output_dir, source, seed, max_visual_iterations):
    episode_dir = Path(output_dir) / f"episode_{source.episode_index:06d}"
    episode_dir.mkdir(exist_ok=False)
    planning = stabilize_terminal_pose(
        config,
        source.terminal_state_A_deg_B_deg_C_mm,
        source_start_state=source.start_state_A_deg_B_deg_C_mm,
    )
    target_world_m = np.asarray(planning["terminal_target_world_m"], dtype=np.float64)
    scene = RCMSceneV9(_scene_config(config))
    controller = None
    sensor = None
    started = time.monotonic()
    try:
        scene.set_task(
            source.start_state_A_deg_B_deg_C_mm * SI_UNITS
        )
        scene.set_target_world_m(target_world_m)
        # Confirm the exact target copy before enabling any motion.
        if not np.array_equal(
                scene.data.site_xpos[scene.sites["lesion"]], target_world_m):
            raise RuntimeError("target scene coordinates changed after assignment")
        scene.enable_motor_execution({"enabled": True})
        scene.reset_motor_time(0.0)
        source_start_actual = scene.current_task().copy()
        planned_start = np.asarray(
            planning["collection_start_state_A_deg_B_deg_C_mm"],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            source_start_actual * UNITS,
            planned_start,
            atol=1e-12,
            rtol=0,
        )
        motor = scene.motor_execution
        settings = visual_servo_settings(scene)
        settings.update(
            max_iterations=int(max_visual_iterations),
            save_diagnostics=True,
            position_feedback="geometry",
            occlusion_handling="stop",
            perception_backend="strict_rgbd",
            failure_hold_on_motor=False,
        )
        if not np.isclose(settings["sample_period_s"], motor.control_period_s):
            raise ValueError("strict RGB-D and motor periods differ")
        endpoint_robot_ABCE = tuple(planning["endpoint_robot_ABCE"])
        requested_terminal_robot_ABCE = tuple(
            planning["requested_terminal_robot_ABCE"]
        )
        endpoint_task = scene.task_from_encoders(
            endpoint_robot_ABCE[:3],
            reference_task=scene.current_task(),
        )
        writer = EpisodeWriter(
            Path(output_dir) / "dataset",
            source.episode_index,
            make_policy_task("visual_current_target", []),
            "strict_zero_error_v12",
            seed,
            ["P1"],
        )
        recorder = _EpisodeRecorder(
            scene,
            writer,
            motor,
            source.episode_index,
            source.source_episode_index,
        )
        guard_scene_goal_inverse(scene)
        controller = VisualServoController(
            scene,
            settings=settings,
            output_dir=episode_dir,
        )

        def capture_visual(iteration, observation):
            recorder.capture(
                observation,
                "strict_rgbd_teacher",
                "visual_servo",
                control_frame_index=iteration,
            )
            if iteration % 50 == 0:
                print(json.dumps({
                    "episode": source.episode_index,
                    "iteration": iteration,
                    "simulation_s": motor.time_s,
                    "wall_s": round(time.monotonic() - started, 1),
                    "error_um": scene.metrics()["tip_target_error_um"],
                }, ensure_ascii=False), flush=True)
            return True

        controller.iteration_callback = capture_visual
        visual_report = controller.run()
        visual_result = visual_report["result"]
        fallback_used = visual_result != "PASS"
        sensor = StrictRGBDSensor(scene, settings)

        def close_outstanding_visual_transition():
            """Record a final controller action before a different handoff.

            Some failure exits issue one last teacher hold and return without a
            callback for the resulting state.  Closing that transition with a
            current observation keeps image/state/action rows aligned before a
            fallback or exact-terminal command starts.
            """
            if recorder.pending is None:
                return
            step = motor.last_record
            if step is None or not np.isclose(
                    step["start_time_s"],
                    recorder.pending["time_s"],
                    atol=1e-10,
                    rtol=0):
                return
            observation = sensor.observe()
            recorder.capture(
                observation,
                "strict_rgbd_teacher",
                "visual_servo",
                control_frame_index=None,
            )

        def command_and_observe(terminal_command, execution_source, phase):
            scene.set_task(terminal_command, motor_dt=motor.control_period_s)
            observation = sensor.observe()
            recorder.capture(
                observation,
                execution_source,
                phase,
                control_frame_index=None,
            )

        def command_exact_terminal_and_observe():
            actual = motor.advance_robot(
                requested_terminal_robot_ABCE,
                motor.control_period_s,
                requested_task=endpoint_task,
            )
            scene._apply_task_pose(actual)
            observation = sensor.observe()
            recorder.capture(
                observation,
                "exact_terminal_lock",
                "exact_terminal_lock",
                control_frame_index=None,
            )

        close_outstanding_visual_transition()
        if fallback_used:
            vmax = (
                np.asarray(
                    config["navigation"]["max_velocity_A_deg_B_deg_C_mm_per_s"],
                    dtype=np.float64,
                )
                * SI_UNITS
            )
            amax = (
                np.asarray(
                    config["navigation"]["max_acceleration_A_deg_B_deg_C_mm_per_s2"],
                    dtype=np.float64,
                )
                * SI_UNITS
            )
            # Leave headroom for integer encoder quantization in the driver.
            # The actual feedback velocity must remain inside the nominal
            # controller bounds, not merely the continuous requested profile.
            vmax *= 0.8
            amax *= 0.6
            fallback_path = limited_joint_path(
                scene.current_task(),
                endpoint_task,
                vmax,
                amax,
                motor.control_period_s,
            )
            for command in fallback_path:
                command_and_observe(
                    command,
                    "privileged_joint_fallback",
                    "fallback",
                )
            # A candidate path can arrive through a quantized branch whose
            # cumulative handles already equal the endpoint handles. The
            # driver then treats the endpoint retransmit as a duplicate and
            # retains the branch's neighbouring target. Nudge C by roughly one
            # handle (10 encoder counts, about 0.16 um) through MotorBackend so
            # the following exact endpoint packet is a non-duplicate command.
            prime_robot = list(endpoint_robot_ABCE)
            prime_robot[2] += 10
            prime_task = scene.task_from_encoders(
                prime_robot[:3],
                reference_task=scene.current_task(),
            )
            command_and_observe(
                prime_task,
                "privileged_joint_fallback",
                "fallback_prime",
            )

        # Visual success or the fallback path is followed by an explicit
        # terminal lock.  Even a zero-distance lock consumes one protocol
        # interval so its outgoing action and settled hold are auditable.
        reached = False
        for _ in range(32):
            command_exact_terminal_and_observe()
            if motor.settled():
                reached = True
                break
        if not reached:
            raise RuntimeError("exact terminal encoder lock did not settle")
        if tuple(motor.backend.robot) != endpoint_robot_ABCE:
            last = motor.last_record or {}
            raise RuntimeError(
                "exact terminal lock did not reach source terminal encoder: "
                f"robot={motor.backend.robot}, target={motor.backend.target}, "
                f"endpoint={endpoint_robot_ABCE}, "
                f"requested={requested_terminal_robot_ABCE}, "
                f"last_quantized_target={last.get('quantized_target_robot_ABCE')}"
            )
        if tuple(motor.backend.target) != endpoint_robot_ABCE:
            raise RuntimeError("exact terminal lock target differs from source terminal encoder")

        recorder.finalize_terminal()
        tip_world_m = scene.data.site_xpos[scene.sites["needle_tip"]].copy()
        stored_target_world_m = scene.data.site_xpos[scene.sites["lesion"]].copy()
        tip_world_mm = tip_world_m * 1000.0
        target_world_mm = stored_target_world_m * 1000.0
        final_error_um = float(np.linalg.norm(tip_world_mm - target_world_mm) * 1000.0)
        if not np.array_equal(tip_world_m, stored_target_world_m):
            raise RuntimeError("terminal tip and target are not component-wise bit-equal")
        if final_error_um != 0.0:
            raise RuntimeError(f"terminal error is not exactly zero: {final_error_um}")
        if not motor.settled() or motor.backend.robot != motor.backend.target:
            raise RuntimeError("motor is not settled at the exact terminal target")

        metrics = scene.metrics()
        report = dict(visual_report)
        report.update({
            "result": "PASS",
            "controller_result": visual_result,
            "visual_servo_result": visual_result,
            "fallback_used": fallback_used,
            "fallback_mode": (
                "privileged_joint_space_velocity_acceleration_limited"
                if fallback_used else None
            ),
            "strict_zero_error": True,
            "terminal_stop_validated": bool(motor.settled()),
            "final_tip_world_m": tip_world_m.tolist(),
            "terminal_target_world_m": stored_target_world_m.tolist(),
            "final_tip_world_mm": tip_world_mm.tolist(),
            "target_world_mm": target_world_mm.tolist(),
            "final_tip_target_error_um": final_error_um,
            "final_error_um": final_error_um,
            "terminal_robot_ABCE": list(motor.backend.robot),
            "terminal_target_robot_ABCE": list(motor.backend.target),
            "source_episode_index": source.source_episode_index,
            "source_dataset": source.source_dataset,
            "source_start_state_A_deg_B_deg_C_mm": (
                source.start_state_A_deg_B_deg_C_mm.tolist()
            ),
            "source_terminal_state_A_deg_B_deg_C_mm": (
                source.terminal_state_A_deg_B_deg_C_mm.tolist()
            ),
            "collection_start_state_A_deg_B_deg_C_mm": (
                source_start_actual * UNITS
            ).tolist(),
            "endpoint_robot_ABCE": list(endpoint_robot_ABCE),
            "requested_terminal_robot_ABCE": list(
                requested_terminal_robot_ABCE
            ),
            "endpoint_settle_steps": planning["endpoint_settle_steps"],
            "source_terminal_tip_world_m": planning["source_terminal_tip_world_m"],
            "source_nominal_target_world_m": (
                planning["source_nominal_target_world_m"]
            ),
            "target_minus_source_nominal_target_m": (
                planning["target_minus_source_nominal_target_m"]
            ),
            "target_minus_source_terminal_tip_m": (
                planning["target_minus_source_terminal_tip_m"]
            ),
            "collection_wall_s": time.monotonic() - started,
            "scene_backend": "RCMSceneV9",
            "target_source": "stabilized_source_terminal_needle_tip",
            "target_coordinates_are_privileged_audit_only": True,
            "policy_inputs_exclude_target_truth": True,
            "motor_execution": motor.summary(),
            "metrics": metrics,
        })
        controller.write(report, episode_dir)
        write_jsonl(episode_dir / "motor_steps.jsonl", motor.records)
        write_jsonl(episode_dir / "frame_audit.jsonl", recorder.audits)
        episode = writer.finish(
            "PASS",
            stop_constraint_summary={
                "pass": True,
                "attempted_targets": 1,
                "targets_with_validated_stop": 1,
            },
        )
        episode.update({
            "motor_steps": (episode_dir / "motor_steps.jsonl").relative_to(
                output_dir
            ).as_posix(),
            "frame_audit": (episode_dir / "frame_audit.jsonl").relative_to(
                output_dir
            ).as_posix(),
            "servo_report": (episode_dir / "visual_servo_report.json").relative_to(
                output_dir
            ).as_posix(),
            "strict_zero_error": True,
            "final_error_um": 0.0,
            "visual_servo_result": visual_result,
            "fallback_used": fallback_used,
            "fallback_reason": (
                None if not fallback_used
                else report.get("reason", report.get("controller_reason"))
            ),
            "fallback_mode": (
                "privileged_joint_space_velocity_acceleration_limited"
                if fallback_used else None
            ),
            "source_episode_index": source.source_episode_index,
            "source_dataset": source.source_dataset,
            "source_start_state_A_deg_B_deg_C_mm": (
                source.start_state_A_deg_B_deg_C_mm.tolist()
            ),
            "source_terminal_state_A_deg_B_deg_C_mm": (
                source.terminal_state_A_deg_B_deg_C_mm.tolist()
            ),
            "collection_start_state_A_deg_B_deg_C_mm": (
                source_start_actual * UNITS
            ).tolist(),
            "endpoint_robot_ABCE": list(endpoint_robot_ABCE),
            "requested_terminal_robot_ABCE": list(
                requested_terminal_robot_ABCE
            ),
            "terminal_robot_ABCE": list(motor.backend.robot),
            "terminal_target_robot_ABCE": list(motor.backend.target),
            "terminal_tip_world_m": tip_world_m.tolist(),
            "terminal_target_world_m": stored_target_world_m.tolist(),
            "source_terminal_tip_world_m": planning["source_terminal_tip_world_m"],
            "source_nominal_target_world_m": (
                planning["source_nominal_target_world_m"]
            ),
            "target_minus_source_nominal_target_m": (
                planning["target_minus_source_nominal_target_m"]
            ),
            "target_minus_source_terminal_tip_m": (
                planning["target_minus_source_terminal_tip_m"]
            ),
            "execution_source_counts": recorder.source_counts,
            "motor_execution": motor.summary(),
        })
        write_json(episode_dir / "episode_result.json", episode)
        return episode
    finally:
        if sensor is not None:
            sensor.close()
        if controller is not None:
            try:
                controller.sensor.close()
            except Exception:
                pass
        scene.close()


def _source_hashes(source_dir):
    source_root = Path(source_dir).resolve()
    paths = {
        "source_dataset.npz": source_root / "data" / "dataset.npz",
        "source_metadata.json": source_root / "metadata.json",
        "collector": Path(__file__).resolve(),
        "verifier": HERE / "scripts" / "verification" / "verify_zero_error_dataset.py",
        "scene_v9": HERE / "rcm" / "scene_v9.py",
        "motor_execution": HERE / "rcm" / "motor_execution.py",
        "strict_rgbd": HERE / "rcm" / "strict_rgbd.py",
        "visual_servo": HERE / "rcm" / "visual_servo.py",
        "waypoint_dataset": HERE / "rcm" / "waypoint_dataset.py",
        "motor_backend": HERE / "mujoco_model" / "motor_backend.py",
        "config": HERE / "config.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _resume_episodes(dataset, output_dir, protocol, next_expected):
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        uncommitted = sorted(Path(output_dir).glob("episode_*"))
        if uncommitted:
            raise ValueError(
                "cannot resume without dataset/manifest.json after episodes started"
            )
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = list(manifest.get("episodes", []))
    if [row.get("episode_index") for row in episodes] != list(range(len(episodes))):
        raise ValueError("resume manifest episode indexes are not contiguous")
    if len(episodes) > next_expected:
        raise ValueError("resume manifest has more episodes than requested")
    committed = len(episodes)
    uncommitted = output_dir / f"episode_{committed:06d}" / "episode_result.json"
    if uncommitted.is_file():
        row = json.loads(uncommitted.read_text(encoding="utf-8"))
        if int(row.get("episode_index", -1)) != committed:
            raise ValueError("uncommitted episode result has the wrong index")
        episodes.append(row)
        committed += 1
    return episodes


def _protocol(config, source_dir, requested_episodes, max_visual_iterations):
    return {
        "dataset_version": "v12",
        "purpose": "strict_zero_error",
        "source_dataset": str(Path(source_dir).resolve()),
        "source_sha256": _source_hashes(source_dir),
        "requested_episodes": int(requested_episodes),
        "max_visual_iterations": int(max_visual_iterations),
        "image_shape": list(IMAGE_SHAPE),
        "control_resolution_px": config["reference_scene"]["eye_camera"]["resolution_px"],
        "action_semantics": (
            "quantized MotorBackend absolute A/B/C target for the following "
            "control interval; never the next actual state"
        ),
        "target_semantics": (
            "bit-exact stabilized source-terminal needle-tip MuJoCo world position"
        ),
        "fallback_semantics": (
            "privileged A/B/C waypoint path with explicit velocity and "
            "acceleration limits, still executed through MotorBackend"
        ),
        "clock": "simulation; not realtime acquisition",
    }


def _validate_pilot_gate(output_dir):
    path = Path(output_dir) / "pilot_gate.json"
    if not path.is_file():
        raise ValueError(
            "bulk collection requires --episodes 1 pilot completion and "
            "pilot_gate.json PASS"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    raw = report.get("raw_pilot", {})
    training = report.get("pilot_training_verification", {})
    if (
        report.get("result") != "PASS"
        or raw.get("result") != "PASS"
        or raw.get("episodes") != 1
        or raw.get("all_terminal_errors_exactly_zero") is not True
        or training.get("result") != "PASS"
    ):
        raise ValueError("pilot gate did not pass strict raw and 224 checks")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--episodes", type=int, default=608)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--max-visual-iterations", type=int, default=1600)
    parser.add_argument(
        "--control-resolution",
        nargs=2,
        type=int,
        default=[1920, 1440],
        metavar=("WIDTH", "HEIGHT"),
        help="strict RGB-D perception resolution; stored images remain 224x224",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--convert-when-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build the 224 LeRobot-style training dataset after all 608 episodes pass",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.max_visual_iterations < 1:
        parser.error("--episodes and --max-visual-iterations must be positive")
    if any(value < 1 for value in args.control_resolution):
        parser.error("--control-resolution values must be positive")

    source_episodes = load_source_episodes(args.source_dataset)
    if args.episodes > len(source_episodes):
        parser.error(
            f"--episodes exceeds source count {len(source_episodes)}"
        )
    output_dir = args.output_dir.resolve()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config["motor_execution"] = {
        **config.get("motor_execution", {}),
        "enabled": True,
        "control_period_s": float(
            config["navigation"]["visual_servo"]["sample_period_s"]
        ),
    }
    config["navigation"]["visual_servo"]["perception_backend"] = "strict_rgbd"
    config["reference_scene"]["eye_camera"]["resolution_px"] = list(
        args.control_resolution
    )
    protocol = _protocol(
        config,
        args.source_dataset,
        len(source_episodes),
        args.max_visual_iterations,
    )
    dataset = output_dir / "dataset"
    if output_dir.exists():
        if not args.resume:
            parser.error("output directory exists; pass --resume to continue it")
        previous_protocol = json.loads(
            (output_dir / "collection_protocol.json").read_text(encoding="utf-8")
        )
        if previous_protocol != protocol:
            parser.error("cannot resume: collection protocol or source hashes changed")
        previous_config = json.loads(
            (output_dir / "config_snapshot.json").read_text(encoding="utf-8")
        )
        if previous_config != config:
            parser.error("cannot resume: effective config changed")
    else:
        if args.episodes > 1:
            parser.error(
                "bulk collection requires a pilot run; first run with "
                "--episodes 1, then resume with --episodes 608"
            )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "config_snapshot.json", config)
        write_json(output_dir / "collection_protocol.json", protocol)
    if args.episodes > 1:
        try:
            _validate_pilot_gate(output_dir)
        except ValueError as error:
            parser.error(str(error))

    episodes = _resume_episodes(
        dataset,
        output_dir,
        protocol,
        args.episodes,
    )
    if episodes:
        write_manifest(
            dataset,
            episodes,
            list(IMAGE_SHAPE),
            float(config["navigation"]["visual_servo"]["sample_period_s"]),
            "visual_current_target",
        )
        manifest_path = dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "dataset_version": "v12",
            "strict_zero_error": True,
            "execution_backend": "MotorBackend",
            "perception_backend": "strict_rgbd",
            "action_semantics": protocol["action_semantics"],
            "target_semantics": protocol["target_semantics"],
            "audit_path_base": "Parent of this dataset directory",
        })
        write_json(manifest_path, manifest)

    failed = []
    for source in source_episodes[len(episodes):args.episodes]:
        episode_dir = output_dir / f"episode_{source.episode_index:06d}"
        try:
            episode = collect_episode(
                config,
                output_dir,
                source,
                args.seed + source.episode_index,
                args.max_visual_iterations,
            )
        except Exception as error:
            failed_row = {
                "episode_index": source.episode_index,
                "source_episode_index": source.source_episode_index,
                "result": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp_s": time.time(),
            }
            failed.append(failed_row)
            with (output_dir / "failed_episodes.jsonl").open(
                    "a", encoding="utf-8") as stream:
                stream.write(json.dumps(
                    failed_row, ensure_ascii=False, allow_nan=False
                ) + "\n")
            if episode_dir.exists():
                destination = output_dir / (
                    f"failed_episode_{source.episode_index:06d}_"
                    f"{int(time.time())}"
                )
                episode_dir.replace(destination)
            print(json.dumps(failed_row, ensure_ascii=False), file=sys.stderr, flush=True)
            break
        episodes.append(episode)
        manifest = write_manifest(
            dataset,
            episodes,
            list(IMAGE_SHAPE),
            float(config["navigation"]["visual_servo"]["sample_period_s"]),
            "visual_current_target",
        )
        manifest.update({
            "dataset_version": "v12",
            "strict_zero_error": True,
            "execution_backend": "MotorBackend",
            "perception_backend": "strict_rgbd",
            "action_semantics": protocol["action_semantics"],
            "target_semantics": protocol["target_semantics"],
            "audit_path_base": "Parent of this dataset directory",
        })
        write_json(dataset / "manifest.json", manifest)
        print(json.dumps({
            "episode_complete": source.episode_index,
            "result": episode["result"],
            "frames": episode["frames"],
            "visual_result": episode["visual_servo_result"],
            "fallback_used": episode["fallback_used"],
        }, ensure_ascii=False), flush=True)

    if failed:
        write_json(output_dir / "partial_collection.json", {
            "result": "FAIL",
            "completed_episodes": len(episodes),
            "requested_episodes": args.episodes,
            "failure": failed[-1],
        })
        return 1
    if not episodes:
        parser.error("no episodes collected")

    from scripts.verification.verify_zero_error_dataset import verify_run
    verification = verify_run(
        output_dir,
        source_dataset=args.source_dataset,
        expected_episodes=args.episodes,
        require_complete=(args.episodes == len(source_episodes)),
    )
    verification["collection_source_sha256"] = protocol["source_sha256"]
    write_json(output_dir / "dataset_verification.json", verification)

    training_result = None
    if verification["result"] == "PASS" and args.episodes == 1:
        from scripts.datasets.convert_zero_error_to_lerobot import convert
        from scripts.verification.verify_zero_error_dataset import (
            verify_training_dataset,
        )

        pilot_dir = output_dir / "pilot_224"
        if pilot_dir.exists():
            parser.error("pilot 224 output already exists")
        pilot_conversion = convert(
            output_dir,
            pilot_dir,
            expected_episodes=1,
            require_complete=False,
        )
        pilot_training_verification = verify_training_dataset(
            output_dir,
            pilot_dir,
            expected_episodes=1,
        )
        pilot_gate = {
            "result": (
                "PASS"
                if pilot_training_verification["result"] == "PASS"
                else "FAIL"
            ),
            "scope": "episode_000000 only; prerequisite for bulk collection",
            "raw_pilot": verification,
            "pilot_conversion": pilot_conversion,
            "pilot_training_verification": pilot_training_verification,
            "pilot_224_dir": pilot_dir.relative_to(output_dir).as_posix(),
            "recorded_unix_s": time.time(),
        }
        write_json(output_dir / "pilot_gate.json", pilot_gate)
        if pilot_gate["result"] != "PASS":
            return 1
    if (
        verification["result"] == "PASS"
        and args.convert_when_complete
        and args.episodes == len(source_episodes)
        and len(episodes) == len(source_episodes)
    ):
        from scripts.datasets.convert_zero_error_to_lerobot import convert
        training_result = convert(output_dir, args.training_output_dir)
        write_json(output_dir / "training_conversion.json", training_result)
        from scripts.verification.verify_zero_error_dataset import verify_training_dataset
        training_verification = verify_training_dataset(
            output_dir,
            args.training_output_dir,
            expected_episodes=len(source_episodes),
        )
        write_json(output_dir / "training_verification.json", training_verification)
        if training_verification["result"] != "PASS":
            return 1
    print(json.dumps({
        "result": verification["result"],
        "episodes": len(episodes),
        "training": training_result,
        "verification": verification,
    }, ensure_ascii=False, indent=2))
    return 0 if verification["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
