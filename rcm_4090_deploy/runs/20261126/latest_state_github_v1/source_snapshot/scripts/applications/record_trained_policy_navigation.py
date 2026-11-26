"""Run one trained-policy navigation episode and record an annotated video.

The policy, RGB-D sensor, and MotorBackend are the same objects used by
``lerobot_training.eval_policy``.  The observer only renders and records the
post-command state; it does not provide ground-truth control inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot_training.eval_policy import (
    DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
    PolicyEvaluator,
)
from rcm.scene_v9 import RCMSceneV9
from rcm.strict_rgbd import StrictRGBDSensor
from rcm.visual_servo import visual_servo_settings


CHECKPOINT = (
    "outputs/homog_label_fix_20261124_v5/"
    "homog_label_fix_20261124_v5_phase1_B/best_model.pth"
)


def _font(size: int):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _annotate(frame, lines):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(18, image.height // 70))
    line_height = int(font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 8
    pad = 18
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle((0, 0, image.width, box_height), fill=(0, 0, 0, 185))
    for index, line in enumerate(lines):
        draw.text((pad, pad + index * line_height), line,
                  fill=(240, 248, 255, 255), font=font)
    return np.asarray(image)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(CHECKPOINT))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-action-samples", type=int, default=7)
    parser.add_argument("--target", type=float, nargs=3,
                        default=(1.92, 8.478, 22.043),
                        metavar=("A", "B", "C"),
                        help="reachable target in robot coordinates [mm]")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    model_path = args.model.resolve()
    if not model_path.is_file():
        parser.error(f"checkpoint not found: {model_path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        parser.error(f"output directory is not empty: {output_dir}; use --overwrite only for an explicit rerun")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = RCMSceneV9()
    sensor = None
    writer = None
    trace_stream = None
    try:
        target_robot_mm = np.asarray(args.target, dtype=np.float64)
        target_world_mm = scene.robot_to_world(target_robot_mm)
        scene.set_target_world(target_world_mm)

        settings = visual_servo_settings(scene)
        settings["perception_backend"] = "strict_rgbd"
        settings["rgbd_render_quality"] = {
            "offsamples": 0, "numslices": 112, "numstacks": 64,
        }
        sensor = StrictRGBDSensor(scene, settings)
        evaluator = PolicyEvaluator(
            str(model_path), device=args.device,
            num_action_samples=args.num_action_samples,
            n_action_execute_steps=1,
            flow_integration_steps=4,
            measure_inference=True,
            controller_variant="feedback",
            rgbd_render_quality=settings["rgbd_render_quality"],
        )

        video_path = output_dir / "trained_policy_navigation.mp4"
        trace_path = output_dir / "step_trace.jsonl"
        results_path = output_dir / "evaluation_results.json"
        metadata_path = output_dir / "run_metadata.json"
        writer = imageio.get_writer(
            str(video_path), fps=20, codec="libx264", quality=8,
            macro_block_size=1,
        )
        trace_stream = trace_path.open("w", encoding="utf-8")
        frame_count = 0

        def observe(event):
            nonlocal frame_count
            current_tip = scene.site_mm("needle_tip")
            error_mm = float(np.linalg.norm(current_tip - target_world_mm))
            accepted = np.asarray(event["accepted_action_deg_deg_mm"], dtype=float)
            raw = np.asarray(event["raw_action_deg_deg_mm"], dtype=float)
            lines = [
                "TRAINED POLICY NAVIGATION | DIT rectified-flow",
                f"checkpoint: {model_path.name} | step: {event['step'] + 1:04d} | time: {scene.motor_execution.time_s:.2f}s",
                f"error: {error_mm * 1000.0:8.3f} um | safety-limited: {event['safety_limited']} (scale {event['safety_scale']:.4g})",
                "raw action [deg, deg, mm]: " + np.array2string(raw, precision=5, suppress_small=False),
                "accepted action [deg, deg, mm]: " + np.array2string(accepted, precision=5, suppress_small=False),
            ]
            writer.append_data(_annotate(scene.render_camera(), lines))
            record = dict(event)
            record.update({
                "error_after_mm": error_mm,
                "simulation_time_s": float(scene.motor_execution.time_s),
                "frame_index": frame_count,
            })
            trace_stream.write(json.dumps(record, allow_nan=False) + "\n")
            trace_stream.flush()
            frame_count += 1

        evaluator.step_observer = observe
        result = evaluator.evaluate_episode(
            scene, target_world_mm, max_steps=args.max_steps,
            error_threshold_mm=0.020, control_period_s=0.05,
            action_limit_deg_deg_mm=DEFAULT_ACTION_LIMIT_DEG_DEG_MM,
            settled_velocity_deg_deg_mm_per_s=(0.05, 0.05, 0.01),
            early_stop_error_mm=30.0, max_consecutive_safety_limited=20,
            seed=args.seed, sensor=sensor, progress_every=100,
        )
        payload = {
            "result": result,
            "video": str(video_path),
            "trace": str(trace_path),
            "frame_count": frame_count,
            "checkpoint": str(model_path),
            "target_robot_mm": target_robot_mm.tolist(),
            "target_world_mm": target_world_mm.tolist(),
            "policy_type": evaluator.policy_type,
            "architecture": evaluator.checkpoint.get("architecture"),
        }
        results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps({
            "checkpoint": str(model_path),
            "policy_type": evaluator.policy_type,
            "architecture": evaluator.checkpoint.get("architecture"),
            "epoch": evaluator.checkpoint.get("epoch"),
            "device": str(evaluator.device),
            "execution_backend": "MotorBackend",
            "perception_backend": "strict_rgbd",
            "ground_truth_used_for_control": False,
            "video_fps": 20,
            "frame_count": frame_count,
        }, indent=2), encoding="utf-8")
        print(json.dumps({
            "success": bool(result["success"]),
            "steps": result["steps"],
            "final_error_um": result["final_error_mm"] * 1000.0,
            "early_stop_reason": result["early_stop_reason"],
            "video": str(video_path),
            "results": str(results_path),
        }, indent=2))
    finally:
        if trace_stream is not None:
            trace_stream.close()
        if writer is not None:
            writer.close()
        if sensor is not None:
            sensor.close()
        scene.close()


if __name__ == "__main__":
    main()
