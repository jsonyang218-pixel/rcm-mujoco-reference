"""Render measured navigation paths, error curves and before/after comparisons."""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mujoco
import numpy as np

from rcm_reference import Reference, save_png


COLORS = {"visual_servo": "#0072B2", "RRT": "#D55E00", "PRM": "#009E73"}
LABELS = {"visual_servo": "视觉伺服", "RRT": "RRT", "PRM": "PRM"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace(root, report):
    folder = report["method"] if report["seed"] is None else f"{report['method']}_seed_{report['seed']}"
    with (root/folder/"trajectory.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    world = np.array([[float(row[f"tip_world_{axis}_mm"]) for axis in "xyz"] for row in rows])
    return rows, world


def save_figure(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mujoco_scene(reference, world_path, solutions, color, fundus=False):
    """Use the actual compiled eye and measured world-space trace in MuJoCo."""
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[2] = 1
    option.sitegroup[:] = 0
    reference.model.geom("eye_camera_marker").rgba[3] = 0.
    reference.model.geom("eye_camera_axis").rgba[3] = 0.
    eye_id = reference.model.geom("eye_reference").id
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = reference.data.geom_xpos[eye_id]
    camera.distance, camera.azimuth, camera.elevation = .040, 125., -12.
    with mujoco.Renderer(reference.model, width=960, height=720) as renderer:
        renderer.update_scene(reference.data, camera=camera, scene_option=option)

        def add_segment(a, b, rgba, width=.00008, arrow=False):
            if np.linalg.norm(np.asarray(a)-b) < 1e-10:
                return
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
            mujoco.mjv_initGeom(geom, kind, np.ones(3)*width, np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(geom, kind, width, np.asarray(a)*.001, np.asarray(b)*.001)
            renderer.scene.ngeom += 1

        if fundus:
            rotation = reference.data.geom_xmat[eye_id].reshape(3, 3)
            center = reference.data.geom_xpos[eye_id]*1000.
            radii = reference.model.geom_size[eye_id]*1000.
            for z in (.9, .95, .99):
                theta = np.linspace(0, 2*np.pi, 65)
                radius = np.sqrt(1-z*z)
                unit = np.column_stack([radius*np.cos(theta), radius*np.sin(theta), np.full_like(theta, z)])
                ring = (unit*radii)@rotation.T+center
                for a, b in zip(ring, ring[1:]):
                    add_segment(a, b, [.85, .55, .85, 1.], width=.000025)
        indices = np.unique(np.linspace(0, len(world_path)-1, min(240, len(world_path))).astype(int))
        rgba = [*matplotlib.colors.to_rgb(color), 1.]
        for a, b in zip(world_path[indices[:-1]], world_path[indices[1:]]):
            add_segment(a, b, rgba)
        for fraction in (.2, .55, .8):
            i = min(len(world_path)-2, int(fraction*len(world_path)))
            end = min(len(world_path)-1, i+max(2, len(world_path)//35))
            add_segment(world_path[i], world_path[end], rgba, width=.00020, arrow=True)
        points = [world_path[0], *[s["requested_world_mm"] for s in solutions], reference.rcm_mm]
        labels = ["Start", *[s["id"] for s in solutions], "RCM"]
        for index, (point, label) in enumerate(zip(points, labels)):
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            rgba = [1., .77, .15, 1.] if index in (1, 2, 3) else [.8, .85, .95, 1.]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.ones(3)*.00027,
                               np.asarray(point)*.001, np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
            geom.label = label
            renderer.scene.ngeom += 1
        return renderer.render().copy()


def render_results(run_dir, baseline_dir=None):
    root = Path(run_dir)
    output = root/"figures"
    output.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    config = load_json(root/"config_snapshot.json")
    reference = Reference(copy.deepcopy(config))
    reports = load_json(root/"run_reports.json")
    fundus = (root/"fundus_target_verification.json").exists()
    by_id = {s["id"]: s for s in load_json(root/"target_solutions.json")}
    selected_order = load_json(root/"order_selection.json")["selected_order"]
    order = selected_order or list(by_id)  # Marker labels only when survey failed.
    order_label = " → ".join(selected_order) if selected_order else "视觉建图失败，未生成访问顺序；机器人未运动"
    solutions = [by_id[key] for key in order]
    representatives = {}
    for report in reports:
        representatives.setdefault(report["method"], report)
    traces = {}
    for method, report in representatives.items():
        rows, world = load_trace(root, report)
        robot = (world-reference.rcm_mm)@reference.rotation_robot
        traces[method] = (rows, world, robot)
    points = np.asarray([s["requested_robot_mm"] for s in solutions])
    methods = list(representatives)
    fig = plt.figure(figsize=(5.4*len(methods), 5.4), layout="constrained")
    eye = reference.model.geom("eye_reference").id
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 21), np.linspace(0, np.pi, 13))
    local = np.stack([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)], axis=-1)
    eye_world = (local*(reference.model.geom_size[eye]*1000))@reference.data.geom_xmat[eye].reshape(3, 3).T+reference.data.geom_xpos[eye]*1000
    eye_robot = (eye_world-reference.rcm_mm)@reference.rotation_robot
    for index, method in enumerate(methods, 1):
        ax = fig.add_subplot(1, len(methods), index, projection="3d")
        robot = traces[method][2]
        ax.plot_wireframe(*[eye_robot[:, :, i] for i in range(3)], color="#A8BAC4", linewidth=.45, alpha=.35)
        if fundus:
            angles, polar = np.meshgrid(np.linspace(0, 2*np.pi, 33), np.linspace(0, np.arccos(.9), 12))
            cap = np.stack([np.cos(angles)*np.sin(polar), np.sin(angles)*np.sin(polar), np.cos(polar)], axis=-1)
            cap_world = (cap*(reference.model.geom_size[eye]*1000))@reference.data.geom_xmat[eye].reshape(3, 3).T+reference.data.geom_xpos[eye]*1000
            cap_robot = (cap_world-reference.rcm_mm)@reference.rotation_robot
            ax.plot_surface(*[cap_robot[:, :, i] for i in range(3)], color="#C477C8", alpha=.24, linewidth=0)
        ax.plot(*robot.T, color=COLORS[method], linewidth=2.3)
        ax.scatter(*robot[0], color="#263238", marker="*", s=70)
        ax.scatter(*points.T, color="#263238", s=24)
        ax.scatter(0, 0, 0, color="#808080", marker="x", s=30)
        for name, point in zip(order, points):
            ax.text(*point, "  "+name, fontsize=10)
        ax.text(*robot[0], "  起点", fontsize=9)
        ax.text(0, 0, 0, " RCM", fontsize=9)
        ax.set(xlabel="机器人 X / mm", ylabel="机器人 Y / mm", zlabel="机器人 Z / mm",
               xlim=(-12, 13), ylim=(-10, 15), zlim=(-1, 25),
               title=LABELS[method]+("（未完成）" if representatives[method]["result"] != "PASS" else ""))
        ax.set_box_aspect((25, 25, 26))
        ax.view_init(elev=22, azim=-65)
    fig.suptitle("仿真针尖三维路径 · "+order_label+"\n"+
                 ("紫色区域：眼底曲面；三个目标均在曲面上" if fundus else "相同坐标范围；采样规划展示首个种子，重合路径分面显示"), fontsize=13)
    save_figure(fig, output/"paths_3d")

    # One combined scene view makes the reference geometry explicit.  The
    # regular comparison figure uses one panel per method; this additional
    # figure overlays the representative traces on the compiled eye model so
    # the reader can see where the RCM, initial tip, and fundus targets sit.
    fig = plt.figure(figsize=(8.8, 7.2), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot_surface(*[eye_robot[:, :, i] for i in range(3)], color="#B8C4CE",
                    alpha=.16, linewidth=.25, edgecolor="#94A3B8", rstride=1, cstride=1)
    if fundus:
        ax.plot_surface(*[cap_robot[:, :, i] for i in range(3)], color="#C477C8",
                        alpha=.34, linewidth=0, label="眼底代理曲面")
    for method in methods:
        robot = traces[method][2]
        ax.plot(*robot.T, color=COLORS[method], linewidth=2.4, label=LABELS[method])
    ax.scatter(0, 0, 0, color="#344054", marker="x", s=70, label="RCM 原点")
    ax.scatter(*traces[methods[0]][2][0], color="#7F56D9", marker="*", s=90, label="初始针尖")
    ax.scatter(*points.T, color="#111827", s=35, depthshade=False)
    for name, point in zip(order, points):
        ax.text(*point, "  "+name, fontsize=10, color="#111827")
    ax.text(0, 0, 0, " RCM (0,0,0)", fontsize=10, color="#344054")
    ax.set(xlabel="RCM-X / mm", ylabel="RCM-Y / mm", zlabel="RCM-Z / mm",
           title="眼球模型中的三点视觉导航完整轨迹")
    ax.set_box_aspect((25, 25, 26))
    ax.set_xlim(-12, 13)
    ax.set_ylim(-10, 15)
    ax.set_zlim(-1, 25)
    ax.view_init(elev=22, azim=-65)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("眼球模型 + 针尖轨迹\n灰色：眼球椭球；紫色：眼底代理曲面；坐标原点：RCM", fontsize=13)
    save_figure(fig, output/"eye_model_path")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, (i, j) in zip(axes, [(0, 2), (0, 1), (1, 2)]):
        for method in methods:
            robot = traces[method][2]
            ax.plot(robot[:, i], robot[:, j], label=LABELS[method], color=COLORS[method],
                    linestyle={"visual_servo": "-", "RRT": "--", "PRM": ":"}[method], linewidth=2)
        ax.scatter(points[:, i], points[:, j], color="#263238", s=24, zorder=5)
        for name, point in zip(order, points):
            ax.annotate(name, (point[i], point[j]), xytext=(5, 7), textcoords="offset points")
        ax.set(xlabel=f"机器人 {'XYZ'[i]} / mm", ylabel=f"机器人 {'XYZ'[j]} / mm", aspect="equal")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("路径投影 · 根据 MuJoCo 实际针尖坐标绘制")
    save_figure(fig, output/"path_projections")

    policy = config["navigation"].get("fundus_transition")
    if policy:
        planned = [m for m in methods if m in ("RRT", "PRM") and representatives[m]["tip_path_length_mm"] > 0]
        if planned:
            fig = plt.figure(figsize=(6.5*len(planned), 6), layout="constrained")
            phase_styles = {"retract": ("#D55E00", "回撤"), "transfer": ("#0072B2", "眼内转向"),
                            "approach": ("#009E73", "接近眼底")}
            for panel, method in enumerate(planned, 1):
                ax = fig.add_subplot(1, len(planned), panel, projection="3d")
                rows, _, robot = traces[method]
                used = set()
                # The trace stores the actual needle-tip position.  The RCM is
                # the origin of this frame, so the initial tip is about 4 mm
                # away from (0, 0, 0).  Keep the full RCM-to-fundus context in
                # this figure instead of cropping the origin out of the view.
                rcm = np.zeros(3)
                initial_tip = robot[0]
                local_first = None
                for phase, entries in itertools.groupby(enumerate(rows), lambda item: item[1].get("phase")):
                    indices = [index for index, _ in entries]
                    if phase not in phase_styles:
                        continue
                    segment = robot[max(0, indices[0]-1):indices[-1]+1]
                    segment = segment[segment[:, 2] >= 18.]
                    if len(segment) < 2:
                        continue
                    if local_first is None:
                        local_first = segment[0]
                    color, label = phase_styles[phase]
                    ax.plot(*segment.T, color=color, linewidth=2.5,
                            linestyle="--" if phase == "retract" else "-", label=label if phase not in used else None)
                    middle = len(segment)//2
                    direction = segment[min(middle+1, len(segment)-1)]-segment[max(0, middle-1)]
                    if np.linalg.norm(direction) > 1e-12:
                        ax.quiver(*segment[middle], *(direction/np.linalg.norm(direction)*.5),
                                  color=color, arrow_length_ratio=.4, linewidth=1.5)
                    used.add(phase)
                if local_first is not None:
                    ax.plot(*np.vstack([rcm, local_first]).T, color="#667085", linewidth=1.5,
                            linestyle=(0, (3, 3)), label="RCM 到眼底局部路径" if "rcm_link" not in used else None)
                    used.add("rcm_link")
                ax.scatter(*rcm, color="#344054", marker="x", s=55, zorder=8)
                ax.text(*rcm, " RCM (0,0,0)", fontsize=9, color="#344054")
                ax.scatter(*initial_tip, color="#7F56D9", marker="*", s=70, zorder=8)
                ax.text(*initial_tip, " 初始针尖", fontsize=9, color="#6941C6")
                if fundus:
                    ax.plot_surface(*[cap_robot[:, :, i] for i in range(3)], color="#C477C8", alpha=.16, linewidth=0)
                ax.scatter(*points.T, color="#222222", s=30)
                for name, point in zip(order, points):
                    ax.text(*point, " "+name)
                ax.set(xlim=(-6, 6), ylim=(0, 11), zlim=(0, 25),
                       xlabel="X / mm", ylabel="Y / mm", zlabel="Z / mm", title=method)
                ax.view_init(elev=23, azim=-65)
                if used:
                    ax.legend(loc="upper left")
            fig.suptitle("RCM 到眼底路径：回撤 → 眼内转向 → 接近\n灰色虚线：RCM 到眼底局部路径；紫色星号：初始针尖；彩色线：眼底附近运动阶段", fontsize=13)
            save_figure(fig, output/"fundus_transitions")

    curve_count = max(1, len(order))
    fig, axes = plt.subplots(1, curve_count, figsize=(5.0*curve_count, 4.8),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    for ax, waypoint in zip(axes, order):
        for method in methods:
            rows = [r for r in traces[method][0] if r["waypoint_id"] == waypoint]
            if not rows:
                continue
            times = np.array([float(r["time_s"]) for r in rows])
            errors = np.array([float(r["tip_target_error_um"]) for r in rows])
            ax.semilogy(times-times[0], np.maximum(errors, 1e-6), color=COLORS[method], label=LABELS[method],
                        linestyle={"visual_servo": "-", "RRT": "--", "PRM": ":"}[method])
        ax.axhline(config["navigation"]["endpoint_tolerance_um"], color="#777777", linestyle="--", linewidth=1, label="验收阈值")
        ax.set(title=waypoint, xlabel="该段运动时间 / s", ylabel="三维针尖误差 / μm（对数）", ylim=(1e-6, 3e4))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("各目标误差收敛 · 误差仅用于评估；小于 1e-6 μm 的模型数值误差按绘图下限显示")
    save_figure(fig, output/"error_curves")

    fig, axes = plt.subplots(1, len(methods), figsize=(5.4*len(methods), 4.8), squeeze=False, layout="constrained")
    for ax, method in zip(axes[0], methods):
        rgb = mujoco_scene(Reference(copy.deepcopy(config)), traces[method][1], solutions, COLORS[method], fundus)
        save_png(output/f"mujoco_{method}.png", rgb)
        ax.imshow(rgb)
        elapsed = representatives[method]["motion_time_s"]
        status = ("未启动：视觉建图失败" if representatives[method].get("execution_mode") == "visual_survey_rejected" else
                  f"完成 · {elapsed:.2f} s" if representatives[method]["result"] == "PASS" else f"未完成（{elapsed:.2f} s 停止）")
        ax.set_title(LABELS[method]+" · "+status)
        ax.axis("off")
    fig.suptitle("MuJoCo 路径渲染 · 黄色球：目标点；浅色球：起点 / RCM；箭头：运动方向"+
                 ("\n紫色网格：眼底曲面；"+("点间先回撤，再转向、接近" if policy else "目标均位于该曲面") if fundus else ""))
    save_figure(fig, output/"mujoco_paths")

    comparable = False
    if baseline_dir is not None and (Path(baseline_dir)/"target_solutions.json").exists():
        old_goals = {s["id"]: s["requested_world_mm"] for s in load_json(Path(baseline_dir)/"target_solutions.json")}
        comparable = (set(old_goals) == set(by_id) and
                      all(np.allclose(old_goals[key], by_id[key]["requested_world_mm"], atol=1e-9, rtol=0) for key in by_id))
    protocol = load_json(root/"experiment_protocol.json") if (root/"experiment_protocol.json").exists() else {}
    if comparable and protocol.get("planner_feedback") != "vision" and (Path(baseline_dir)/"run_reports.json").exists():
        old = load_json(Path(baseline_dir)/"run_reports.json")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        positions = np.arange(len(methods))
        for offset, source, label, color in [(-.18, old, "改进前", "#ADB8C4"), (.18, reports, "改进后", "#0072B2")]:
            means = [np.mean([r["motion_time_s"] for r in source if r["method"] == m]) for m in methods]
            bars = axes[0].bar(positions+offset, means, .34, color=color, label=label)
            axes[0].bar_label(bars, fmt="%.2f", fontsize=9)
        axes[0].set(xticks=positions, xticklabels=[LABELS[m] for m in methods], ylabel="完整访问时间 / s", title="运动时间（原速度上限）")
        axes[0].legend()
        for offset, source, label, color in [(-.18, old, "改进前", "#ADB8C4"), (.18, reports, "改进后", "#0072B2")]:
            visual = next((r for r in source if r["method"] == "visual_servo"), None)
            if visual:
                errors = {w["waypoint_id"]: w["tip_error_um"] for w in visual["waypoints"]}
                bars = axes[1].bar(np.arange(3)+offset, [errors[k] for k in order], .34, color=color, label=label)
                axes[1].bar_label(bars, fmt="%.2f", fontsize=9)
        axes[1].axhline(20, color="#A53636", linestyle="--", label="20 μm 验收线")
        axes[1].set(xticks=np.arange(3), xticklabels=order, ylabel="到达误差 / μm", title="视觉伺服精度")
        axes[1].legend()
        fig.suptitle("前后对比 · 新版各指定路径点均停稳；旧版视觉轨迹不包含完整制动")
        save_figure(fig, output/"before_after")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    print(render_results(args.run_dir, args.baseline_dir))
