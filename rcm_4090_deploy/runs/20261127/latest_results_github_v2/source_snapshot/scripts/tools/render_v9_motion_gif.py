#!/usr/bin/env python3
"""Render a compact, human-viewable GIF of the current v9 MuJoCo motion.

The animation uses the same RCMSceneV9 task interface as the simulation.  It
smoothly cycles the demonstration amplitudes recorded in config.json, unless
explicit amplitudes are supplied on the command line.  The overview and
intraocular camera are rendered from the same MuJoCo state for every frame.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def _smooth_cycle(phase: float) -> float:
    """One gentle out-and-back cycle with zero velocity at both ends."""
    phase = float(phase) % 1.0
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


def _font(size: int, bold: bool = False):
    names = (
        ("msyhbd.ttc", "simhei.ttf", "arialbd.ttf")
        if bold else
        ("msyh.ttc", "simhei.ttf", "arial.ttf")
    )
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _render_panel(renderer, data, camera, option, width, height):
    renderer.update_scene(data, camera=camera, scene_option=option)
    image = Image.fromarray(renderer.render().copy()).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def _draw_header(draw, title_font, label_font, subtitle, width, height):
    header_height = round(height * 0.13)
    margin = round(width * 0.0167)
    draw.rectangle((0, 0, width, header_height), fill="#101923")
    draw.text(
        (margin, round(height * 0.015)),
        "MuJoCo v9 机械臂运动演示",
        font=title_font,
        fill="#f4f7fb",
    )
    draw.text(
        (margin + 1, round(height * 0.078)),
        subtitle,
        font=label_font,
        fill="#91a7bc",
    )


def _draw_panel_frame(draw, box, title, accent, title_font, scale=1.0):
    x0, y0, x1, y1 = box
    radius = max(4, round(8 * scale))
    title_height = max(18, round(25 * scale))
    draw.rounded_rectangle(
        (x0 - 4, y0 - 4, x1 + 4, y1 + 4),
        radius=radius, outline="#344657", width=2,
    )
    draw.rectangle(
        (x0, y0 - title_height, x0 + min(round(190 * scale), x1 - x0),
         y0 - 4),
        fill="#101923",
    )
    draw.text(
        (x0 + max(4, round(8 * scale)), y0 - title_height + 2),
        title, font=title_font, fill=accent,
    )


def _write_metadata(path, *, gif_path, frames, fps, duration_s, amplitudes,
                    preview, dimensions):
    payload = {
        "status": "PASS",
        "animation": gif_path.name,
        "preview": preview.name,
        "model_backend": "RCMSceneV9",
        "source": "current eye-injected scanned MuJoCo model",
        "dimensions_px": list(dimensions),
        "aspect_ratio": "16:9",
        "ppt_canvas_basis": "half of 1280x720",
        "frames": frames,
        "fps": fps,
        "duration_s": duration_s,
        "motion_cycle_s": duration_s,
        "amplitude_A_deg": amplitudes["A_deg"],
        "amplitude_B_deg": amplitudes["B_deg"],
        "stroke_C_mm": amplitudes["C_mm"],
        "views": ["overview", "eye_internal"],
        "overview_fovy_deg": 27.0,
        "trajectory": "smooth zero-velocity out-and-back demonstration cycle",
        "physical_hardware_claim": False,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_gif(output, *, frames=48, fps=20, size=(640, 360),
               a_amplitude=5.0, b_amplitude=2.0, c_stroke=2.0):
    from rcm.scene_v9 import RCMSceneV9

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output.with_name(output.stem + "_preview.png")
    metadata_path = output.with_name(output.stem + "_metadata.json")
    width, height = (int(value) for value in size)
    if width < 640 or height < 360:
        raise ValueError("GIF canvas must be at least 640x360")
    if abs(width / height - 16.0 / 9.0) > 0.002:
        raise ValueError("GIF canvas must use a 16:9 aspect ratio")
    if frames < 24:
        raise ValueError("frames must be at least 24")
    if not 1 <= fps <= 30:
        raise ValueError("fps must be between 1 and 30")

    scene = RCMSceneV9()
    render_images = []
    preview_index = frames // 2
    overview_camera = scene.camera_id("overview")
    original_overview_fovy = float(scene.model.cam_fovy[overview_camera])
    try:
        # The stock overview intentionally includes a wide working envelope.
        # A temporary optical zoom makes the arm and joint motion legible in a
        # small GIF without changing any pose, geometry or joint limits.
        scene.model.cam_fovy[overview_camera] = 27.0
        option_overview = mujoco.MjvOption()
        option_overview.sitegroup[:] = 0
        option_overview.geomgroup[:] = np.array(
            [1, 0, 1, 1, 1, 1], dtype=np.uint8
        )
        option_eye = mujoco.MjvOption()
        option_eye.sitegroup[:] = 0
        option_eye.geomgroup[:] = np.array(
            [0, 0, 1, 1, 1, 1], dtype=np.uint8
        )

        margin = round(width * 0.0167)
        gap = round(width * 0.0125)
        top = round(height * 0.185)
        left_w = round(width * 0.61)
        left_h = round(height * 0.74)
        right_x = margin + left_w + gap
        right_w = width - margin - right_x
        eye_size = min(right_w, round(height * 0.53))
        overview_w, overview_h = left_w, left_h
        eye_w = eye_h = eye_size
        scale = width / 960.0
        title_font = _font(max(9, round(26 * scale)), True)
        label_font = _font(max(8, round(14 * scale)))
        panel_font = _font(max(9, round(17 * scale)), True)
        value_font = _font(max(9, round(15 * scale)), True)
        small_font = _font(max(8, round(13 * scale)))

        with mujoco.Renderer(
            scene.model, width=overview_w, height=overview_h
        ) as overview_renderer, mujoco.Renderer(
            scene.model, width=eye_w, height=eye_h
        ) as eye_renderer:
            history_a, history_b, history_c = [], [], []
            for index in range(frames):
                amount = _smooth_cycle(index / (frames - 1))
                # A and B are out of phase so the two rotational motions are
                # independently visible instead of collapsing into one gesture.
                a_deg = a_amplitude * amount
                b_deg = -b_amplitude * amount
                c_mm = c_stroke * amount
                scene.set_task(
                    np.array([
                        math.radians(a_deg),
                        math.radians(b_deg),
                        c_mm * 0.001,
                    ])
                )
                metrics = scene.metrics()
                history_a.append(float(metrics["task_A_deg_B_deg_C_mm"][0]))
                history_b.append(float(metrics["task_A_deg_B_deg_C_mm"][1]))
                history_c.append(float(metrics["task_A_deg_B_deg_C_mm"][2]))

                overview = _render_panel(
                    overview_renderer, scene.data, "overview",
                    option_overview, overview_w, overview_h,
                )
                eye = _render_panel(
                    eye_renderer, scene.data, "eye_internal",
                    option_eye, eye_w, eye_h,
                )
                frame = Image.new("RGB", (width, height), "#0b1118")
                draw = ImageDraw.Draw(frame)
                _draw_header(
                    draw, title_font, label_font,
                    "A/B/C 平滑运动 · 同一物理状态同步渲染整机与眼内视角",
                    width, height,
                )

                overview_xy = (margin, top)
                eye_xy = (right_x + (right_w - eye_w) // 2, top)
                frame.paste(overview, overview_xy)
                frame.paste(eye, eye_xy)
                _draw_panel_frame(
                    draw,
                    (margin, top, margin + overview_w, top + overview_h),
                    "整机运动", "#74d4ff", panel_font, scale,
                )
                _draw_panel_frame(
                    draw,
                    (eye_xy[0], top, eye_xy[0] + eye_w, top + eye_h),
                    "眼内相机", "#6cf4b2", panel_font, scale,
                )

                # Joint read-outs and progress markers.
                status_top = top + eye_h + round(height * 0.025)
                status_bottom = top + left_h
                draw.rounded_rectangle(
                    (right_x, status_top, right_x + right_w, status_bottom), radius=8,
                    fill="#111b25", outline="#2b3e50", width=2,
                )
                draw.text(
                    (right_x + round(10 * scale), status_top + round(7 * scale)),
                    "当前关节状态", font=panel_font, fill="#e8eef5",
                )
                draw.text(
                    (right_x + round(10 * scale), status_top + round(35 * scale)),
                    f"A {a_deg:+.2f}°   B {b_deg:+.2f}°   C {c_mm:+.2f} mm",
                    font=value_font, fill="#dce6ef",
                )
                draw.text(
                    (right_x + round(10 * scale), status_top + round(59 * scale)),
                    f"帧 {index + 1:02d}/{frames}   t={index / fps:.2f}s",
                    font=small_font,
                    fill="#9fb2c3",
                )
                draw.text(
                    (right_x + round(10 * scale), status_top + round(80 * scale)),
                    "tip-RCM "
                    f"{metrics['needle_eye_geometry']['tip_rcm_distance_mm']:.3f} mm",
                    font=small_font, fill="#9fb2c3",
                )

                # Compact timeline keeps direction and playback progress clear.
                footer_y = height - round(height * 0.052)
                progress_x0 = margin
                progress_x1 = width - margin
                progress_y = height - round(height * 0.012)
                draw.text(
                    (margin, footer_y),
                    "零位 → A+/B−/C+ → 回零 · 循环播放",
                    font=small_font, fill="#91a7bc",
                )
                draw.line(
                    (progress_x0, progress_y, progress_x1, progress_y),
                    fill="#314455", width=2,
                )
                progress_x = progress_x0 + (progress_x1 - progress_x0) * index / (frames - 1)
                draw.line(
                    (progress_x0, progress_y, progress_x, progress_y),
                    fill="#74d4ff", width=3,
                )
                if index == preview_index:
                    frame.save(preview_path)
                render_images.append(
                    frame.quantize(
                        colors=192,
                        method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.NONE,
                    )
                )
    finally:
        scene.model.cam_fovy[overview_camera] = original_overview_fovy
        scene.close()

    duration_ms = max(20, round(1000 / fps))
    render_images[0].save(
        output,
        save_all=True,
        append_images=render_images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    _write_metadata(
        metadata_path,
        gif_path=output,
        frames=frames,
        fps=fps,
        duration_s=frames / fps,
        amplitudes={
            "A_deg": a_amplitude,
            "B_deg": b_amplitude,
            "C_mm": c_stroke,
        },
        preview=preview_path,
        dimensions=(width, height),
    )
    return output, preview_path, metadata_path


def main(argv=None):
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    demo = config["demonstration"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/current_v9_motion.gif"
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--a-amplitude-deg", type=float,
                        default=float(demo["a_amplitude_deg"]))
    parser.add_argument("--b-amplitude-deg", type=float,
                        default=float(demo["b_amplitude_deg"]))
    parser.add_argument("--c-stroke-mm", type=float,
                        default=float(demo["c_stroke_mm"]))
    args = parser.parse_args(argv)

    for name in ("a_amplitude_deg", "b_amplitude_deg", "c_stroke_mm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    gif, preview, metadata = render_gif(
        args.output,
        frames=args.frames,
        fps=args.fps,
        a_amplitude=args.a_amplitude_deg,
        b_amplitude=args.b_amplitude_deg,
        c_stroke=args.c_stroke_mm,
    )
    print(json.dumps({
        "status": "PASS",
        "gif": str(gif),
        "preview": str(preview),
        "metadata": str(metadata),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
