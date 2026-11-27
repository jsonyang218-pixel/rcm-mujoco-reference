"""Offline axis-floor diagnostics from frozen I0 trajectories; no rollout."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UNIT = np.array([np.pi / 180.0, np.pi / 180.0, 0.001])
AXES = ("A", "B", "C")
INDICES = (0, 2, 6, 8)
DEFAULT_SOURCE = Path("runs/20261120/overnight_fast5_20261120_v1/task34/task4_dev20")
HISTORICAL = Path("runs/20261119/overnight_c_fix_symlog_081234/task2_step4000")
DEADZONE_HIGH_FRACTION = 0.25


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def zero_velocity_teacher(jacobian, error_world_mm, navigation, tolerance_um):
    """Geometry teacher's numerical law at v_prev=0, command_q=q.

    The caller supplies a saved OCT displacement or an oracle displacement.
    This does not replay the missing RGB-D geometry/quality validity checks.
    """
    vmax = np.asarray(navigation["max_velocity_A_deg_B_deg_C_mm_per_s"]) * UNIT
    amax = np.asarray(navigation["max_acceleration_A_deg_B_deg_C_mm_per_s2"]) * UNIT
    dt = float(navigation["sample_period_s"])
    u, singular, vt = np.linalg.svd(jacobian * vmax, full_matrices=False)
    if singular[-1] < max(1e-10, singular[0] * 1e-6):
        raise ValueError("singular actuation Jacobian")
    correction = vmax * (vt.T @ (singular / (singular**2 + (singular[0] * 1e-4)**2) * (u.T @ error_world_mm)))
    if np.linalg.norm(error_world_mm) * 1000 < tolerance_um:
        wanted = np.zeros(3)
    else:
        wanted = float(navigation["visual_servo"]["geometry_gain_per_s"]) * correction
    cap = np.minimum(vmax, 0.6 * np.sqrt(2 * amax * np.abs(correction)))
    velocity = np.clip(np.clip(wanted, -cap, cap), -amax * dt, amax * dt)
    return velocity * dt / UNIT


def crossings(values):
    signs = np.sign(values)
    signs = signs[signs != 0]
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def finite_array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"expected finite {shape}, got {result.shape}")
    return result


def metrics(states, final_state, commands, target, gains, teacher, teacher_oracle):
    movement = np.diff(np.vstack((states, final_state)), axis=0)
    residual = states - target
    rows = {}
    for i, name in enumerate(AXES):
        samples = slice(700, 1600)
        steady = slice(1000, 1600)
        active = np.abs(commands[samples, i]) > 0
        stalled = np.abs(movement[samples, i]) < 1e-6
        nonzero = np.abs(movement[samples, i]) >= 1e-6
        residual_um = np.abs(residual[samples, i]) * gains[samples, i]
        movement_um = np.abs(movement[samples, i]) * gains[samples, i]
        command_median = float(np.median(np.abs(commands[samples, i])))
        teacher_median = float(np.median(np.abs(teacher[samples, i])))
        oracle_median = float(np.median(np.abs(teacher_oracle[samples, i])))
        rows[name] = {
            "unit": "mm" if i == 2 else "deg",
            "residual_contribution_mean_um": float(residual_um.mean()),
            "residual_contribution_median_um": float(np.median(residual_um)),
            "motion_contribution_mean_um": float(movement_um.mean()),
            "motion_contribution_median_um": float(np.median(movement_um)),
            "steady_bias": float(residual[steady, i].mean()),
            "steady_std": float(residual[steady, i].std()),
            "steady_bias_equivalent_um": float(np.mean(residual[steady, i] * gains[steady, i])),
            "steady_std_equivalent_um": float(np.std(residual[steady, i] * gains[steady, i])),
            "steady_zero_crossings": crossings(residual[steady, i]),
            "active_command_frames": int(active.sum()),
            "deadzone_frames": int(np.count_nonzero(active & stalled)),
            "deadzone_fraction_all_frames": float(np.mean(active & stalled)),
            "deadzone_fraction_active_commands": float(np.mean(stalled[active])) if active.any() else None,
            "minimum_nonzero_step": float(np.min(np.abs(movement[samples, i][nonzero]))) if nonzero.any() else None,
            "minimum_nonzero_step_tip_um": float(np.min(movement_um[nonzero])) if nonzero.any() else None,
            "policy_abs_median": command_median,
            "zero_vel_teacher_oct_abs_median": teacher_median,
            "zero_vel_teacher_oracle_abs_median": oracle_median,
            "policy_teacher_oct_median_ratio": command_median / teacher_median if teacher_median else None,
            "policy_teacher_oracle_median_ratio": command_median / oracle_median if oracle_median else None,
        }
    return rows, residual, movement


def choose_branch(targets):
    triggers = []
    for target, rows in targets.items():
        for axis, row in rows["axes"].items():
            if (row["deadzone_fraction_active_commands"] is not None
                    and row["deadzone_fraction_active_commands"] >= DEADZONE_HIGH_FRACTION):
                triggers.append({"episode": target, "axis": axis, "criterion": "high_nonzero_command_no_motion",
                                 "value": row["deadzone_fraction_active_commands"]})
            if (row["minimum_nonzero_step_tip_um"] is not None
                    and row["minimum_nonzero_step_tip_um"] >= 5):
                triggers.append({"episode": target, "axis": axis, "criterion": "minimum_step_ge_5um",
                                 "value": row["minimum_nonzero_step_tip_um"]})
    if triggers:
        return {"case": 1, "status": "selected_for_investigation", "triggers": triggers,
                "next_action": "Audit MotorBackend and run the authorized five-target pure-expert physical reachability check; suspend fine-tuning.",
                "causal_limit": "High no-motion fraction is a branch trigger, not proof that a 20-50 um physical floor exists."}
    return {"case": None, "status": "requires_case2_case3_evidence_review", "triggers": [],
            "next_action": "Do not select or train Fine DiT unless the case-2 or case-3 criteria are actually met."}


def complete_summary(folder):
    summary = read(folder / "summary.json")
    if summary.get("status") != "completed" or summary["target_count"] != 20:
        raise ValueError(f"not a complete dev20: {folder}")
    episodes = []
    for index in range(20):
        path = folder / f"E{index:02d}/evaluation_results.json"
        row = read(path)["episodes"][0]
        if row["episode_index"] != index or row["seed"] != 42 + index or row["early_stop_reason"]:
            raise ValueError(f"invalid target or seed: {path}")
        if not row["success"] and row["steps"] != 1600:
            raise ValueError(f"incomplete episode: {path}")
        episodes.append(row)
    computed = float(np.mean([row["final_error_mm"] * 1000 for row in episodes]))
    if abs(computed - summary["final_mean_um"]) > 1e-8:
        raise ValueError("summary differs from episodes")
    return summary, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    result = {"status": "running", "started_at": started, "command": "python -B -m scripts.research.diagnose_step680_axes --source-dir " + str(args.source_dir) + " --output-dir " + str(args.output_dir)}
    try:
        execute(args, result)
    except BaseException as exc:
        result.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write(args.output_dir / "axis_floor_summary.json", result)
        with (ROOT / "TEST_LOG.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n- {result['finished_at']}: step680_axes_v1 offline diagnostic **{result['status']}**; command `{result['command']}`; source `{args.source_dir}`; checkpoint `{result.get('checkpoint', 'see source summary')}`; E00/E02/E06/E08 seeds42/44/48/50 I0 1600-step frozen trajectories, no rollout/training; case={result.get('recommendation', {}).get('case')}; evidence `{args.output_dir}/axis_floor_summary.json`; failure={result.get('failure', 'none')}. CPU NumPy analysis (small/I/O-bound); NVIDIA EGL scene initialization. Simulation only.\n")


def execute(args, result):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from rcm.scene_v9 import RCMSceneV9
    from rcm.visual_servo import VisualServoController
    from scripts.research.overnight_cfix import continuation_identity
    from scripts.research.overnight_cfix_fast import identity
    from scripts.research.overnight_cfix_eval import validate_contract, validate_episode, targets

    summary, episodes = complete_summary(args.source_dir)
    historical, previous = complete_summary(HISTORICAL)
    provenance = read(args.source_dir / "provenance.json")
    for key, value in identity().items():
        if provenance[key] != value:
            raise ValueError(f"source provenance mismatch: {key}")
    if digest(summary["checkpoint"]) != provenance["checkpoint_sha256"]:
        raise ValueError("checkpoint digest mismatch")
    manifest = Path("configs/v12_reachable_development_targets_0p35um.json")
    target_rows = targets(manifest)
    pairs = []
    for index, (old, new) in enumerate(zip(previous, episodes)):
        payload = read(args.source_dir / f"E{index:02d}/evaluation_results.json")
        validate_contract(payload["evaluation_contract"], "I0")
        validate_episode(new, target_rows[index], index, True)
        if old["target_id"] != new["target_id"] or old["seed"] != new["seed"]:
            raise ValueError("historical target order/seed mismatch")
        pairs.append({"episode": f"E{index:02d}", "target_id": new["target_id"], "seed": new["seed"],
                      "historical_final_um": old["final_error_mm"] * 1000,
                      "step2000_final_um": new["final_error_mm"] * 1000})
    result.update(checkpoint=summary["checkpoint"], checkpoint_sha256=provenance["checkpoint_sha256"],
                  source_summary=summary, source_provenance=provenance,
                  analysis_code_sha256=digest(__file__),
                  windows={"transition": "640 <= state step <= 800 (161 states)",
                           "contribution": "700 <= pre-action step < 1600 (900 transitions)",
                           "steady": "1000 <= pre-action step < 1600 (600 states)"},
                  definitions={"delta_for_floor": "state - decoded frozen reachable target; norm(J_column * residual_SI) in um",
                               "delta_for_motion": "state[k+1]-state[k], with final_state appended for action[1599]",
                               "contributions": "Column-vector norms are not additive Euclidean error components; stacked bar is a sum-of-norms diagnostic.",
                               "teacher": "Numerical geometry-teacher law; zero previous velocity and command_q=q; recorded OCT camera vector mapped to world. Original RGB-D geometry/quality not saved, so not an exact sensory teacher replay.",
                               "oracle_teacher": "Same numerical law using reconstructed true target-minus-tip, for diagnostic comparison only.",
                               "deadzone": "abs(applied_cmd)>0 and abs(next_state-state)<1e-6 in deg/deg/mm; high means >=25% of active commands (explicit interpretation of qualitative plan criterion).",
                               "physical_limit": "Minimum observed feedback step is not the handle-command quantum and does not establish strict physical attainability."},
                  coarse_comparison={"historical": historical, "step2000": summary,
                                     "selected_checkpoint": historical["checkpoint"],
                                     "selection_basis": "lower historical dev20 final mean and median; historical reference only, missing code/scene provenance; no fresh matched-baseline claim",
                                     "historical_provenance_matched": False, "paired_targets": pairs},
                  targets={})
    scene = RCMSceneV9()
    try:
        if scene.motor_execution is not None:
            raise ValueError("offline reconstruction must not have motor execution enabled")
        teacher_context = SimpleNamespace(reference=scene)
        navigation = scene.config["navigation"]
        tolerance_um = float(navigation["visual_servo"]["position_tolerance_um"])
        for index in INDICES:
            episode = episodes[index]
            trace = episode["trajectory"]
            states = finite_array(trace["states"], (1600, 3))
            final_state = finite_array(episode["final_state_A_deg_B_deg_C_mm"], (3,))
            commands = finite_array(trace["actions"], (1600, 3))
            times = finite_array(trace["simulation_time_s"], (1600,))
            np.testing.assert_allclose(np.diff(times), 0.05, rtol=0, atol=1e-10)
            target_task = scene.task_from_encoders(episode["reachable_encoder_counts_ABC"], reference_task=states[-1] * UNIT)
            target = target_task / UNIT
            target_world = np.asarray(episode["target_world_mm"])
            scene.set_task(target_task)
            target_fk_error_um = float(np.linalg.norm(scene.site_mm("needle_tip") - target_world) * 1000)
            if target_fk_error_um > 1e-5:
                raise ValueError("target encoder/FK mismatch")
            gains = np.full((1600, 3), np.nan)
            teacher = np.full((1600, 3), np.nan)
            teacher_oracle = np.full((1600, 3), np.nan)
            observed_error = np.asarray(trace["oct_navigation_mm"])
            valid = np.asarray(trace["oct_navigation_valid"])
            if not valid[640:].all():
                raise ValueError("missing OCT vector in diagnostic window; do not substitute truth")
            discrepancy = []
            sensory_discrepancy = []
            signed_sensory = []
            for step in range(640, 1600):
                q = states[step] * UNIT
                jacobian = VisualServoController._model_jacobian(teacher_context, q)
                gains[step] = np.linalg.norm(jacobian * UNIT, axis=0) * 1000
                scene.set_task(q)
                tip = scene.site_mm("needle_tip")
                world_error = target_world - tip
                rotation = scene.data.cam_xmat[scene.camera_id("eye_internal")].reshape(3, 3)
                oct_world_error = rotation @ (observed_error[step] * np.array([1, -1, -1]))
                teacher[step] = zero_velocity_teacher(jacobian, oct_world_error, navigation, tolerance_um)
                teacher_oracle[step] = zero_velocity_teacher(jacobian, world_error, navigation, tolerance_um)
                discrepancy.append(abs(np.linalg.norm(world_error) - trace["errors"][step]) * 1000)
                if step >= 1000:
                    sensory_discrepancy.append(np.linalg.norm(oct_world_error - world_error) * 1000)
                    signed_sensory.append((oct_world_error - world_error) * 1000)
            if max(discrepancy) > 1e-5:
                raise ValueError("recorded error/FK reconstruction mismatch")
            axis_rows, residual, movement = metrics(states, final_state, commands, target, gains, teacher, teacher_oracle)
            name = f"E{index:02d}"
            result["targets"][name] = {"target_id": episode["target_id"], "seed": episode["seed"],
                                       "target_state_A_deg_B_deg_C_mm": target.tolist(), "axes": axis_rows,
                                       "source_sha256": digest(args.source_dir / name / "evaluation_results.json"),
                                       "target_fk_error_um": target_fk_error_um,
                                       "maximum_error_reconstruction_delta_um": max(discrepancy),
                                       "steady_sensor_vector_bias_world_um": np.mean(signed_sensory, axis=0).tolist(),
                                       "steady_sensor_vector_error_median_um": float(np.median(sensory_discrepancy))}
            np.savez_compressed(args.output_dir / f"{name}_axis_diagnostics.npz", step=np.arange(640, 1600),
                                residual=residual[640:], actual_change=movement[640:], applied_cmd=commands[640:],
                                zero_vel_teacher=teacher[640:], zero_vel_teacher_oracle=teacher_oracle[640:],
                                tip_um_per_action_unit=gains[640:])
            for start, stop, suffix in ((640, 801, "step640_800"), (1000, 1600, "step1000_1600")):
                fig, axes = plt.subplots(3, 2, figsize=(15, 10), layout="constrained")
                x = np.arange(start, stop)
                for axis, label in enumerate(AXES):
                    unit = "mm" if axis == 2 else "deg"
                    axes[axis, 0].plot(x, residual[start:stop, axis], color="#0072b2")
                    axes[axis, 0].axhline(0, color="black", linewidth=0.7)
                    axes[axis, 0].set(xlabel="Recorded pre-action step (zero based)", ylabel=f"{label}: state-target ({unit})")
                    for values, title in ((commands, "Applied policy command"), (movement, "Actual next-state change"),
                                          (teacher, "Zero-v teacher (saved OCT)"), (teacher_oracle, "Zero-v teacher (oracle)")):
                        axes[axis, 1].plot(x, values[start:stop, axis], label=title, alpha=0.85, linewidth=0.8)
                    axes[axis, 1].set(xlabel="Recorded pre-action step", ylabel=f"{label}: increment ({unit})")
                    for ax in axes[axis]:
                        ax.grid(alpha=0.2)
                axes[0, 1].legend(fontsize=8)
                fig.suptitle(f"{name} / step2000 I0: frozen trajectory diagnostic; no policy rollout")
                fig.savefig(args.output_dir / f"{name}_{suffix}.png", dpi=150)
                plt.close(fig)
            print(name, json.dumps(axis_rows, ensure_ascii=False), flush=True)
    finally:
        scene.close()
    result["recommendation"] = choose_branch(result["targets"])
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    bottom = np.zeros(len(INDICES))
    names = list(result["targets"])
    for axis, color in zip(AXES, ("#0072b2", "#d55e00", "#009e73")):
        values = np.array([result["targets"][name]["axes"][axis]["residual_contribution_mean_um"] for name in names])
        ax.bar(names, values, bottom=bottom, label=axis, color=color)
        bottom += values
    ax.set(ylabel="Mean ||J_axis (q_axis-target_axis)|| (um)", title="Step 700-1599: residual contribution (sum of norms, not Euclidean error)")
    ax.legend()
    fig.savefig(args.output_dir / "axis_contribution_bar.png", dpi=150)
    plt.close(fig)
    result["status"] = "completed"
    lines = ["# 680 步三轴误差地板诊断", "", f"生成时间：{result['started_at']}。仅离线重构，无训练、无新策略 rollout。", "",
             "## 完整 dev20 与 Coarse 历史参照", "",
             f"- step2000：20/20，终值均值 {summary['final_mean_um']:.6f} µm，中位数 {summary['final_median_um']:.6f} µm，最低值均值 {summary['minimum_mean_um']:.6f} µm，严格成功 {summary['canonical_success_count']}/20。",
             f"- task2_step4000 历史参照：终值均值 {historical['final_mean_um']:.6f} µm，中位数 {historical['final_median_um']:.6f} µm，最低值均值 {historical['minimum_mean_um']:.6f} µm，严格成功 {historical['canonical_success_count']}/20。",
             f"- 按终值稳定性暂选 `{historical['checkpoint']}` 作后续 Coarse；历史缺代码/场景指纹，仅描述性参照，不能认定本轮严格配对增益。",
             "- 原计划 27.852 µm、0/10 是 2026-11-20 15:01 的阶段快照；完整终审于 15:24 完成。", "",
             "## 指标与口径", "",
             "窗口：切档 640–800（含端点）；贡献/死区/幅值比 700–1599（900 帧）；稳态 1000–1599（600 帧）。",
             "state[k] 是 action[k] 前状态；实际变化为 state[k+1]-state[k]，最后一步使用单独保存的 final_state。",
             "误差贡献用 state-target；另列运动贡献以消除 ΔA/ΔB/ΔC 的歧义。三列范数之和不是欧氏误差。", "",
             "|目标|轴|误差贡献均值/中位 µm|运动贡献均值/中位 µm|稳态 Bias / Std（deg,deg,mm）|等效 Bias / Std µm|过零|非零指令无位移 %|最小实际步进 µm|策略/教师 OCT / oracle|",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, target in result["targets"].items():
        for axis, row in target["axes"].items():
            ratio = lambda x: "undefined (teacher median=0)" if x is None else f"{x:.3f}"
            lines.append(f"|{name}|{axis}|{row['residual_contribution_mean_um']:.3f} / {row['residual_contribution_median_um']:.3f}|{row['motion_contribution_mean_um']:.3f} / {row['motion_contribution_median_um']:.3f}|{row['steady_bias']:.7g} / {row['steady_std']:.7g}|{row['steady_bias_equivalent_um']:.3f} / {row['steady_std_equivalent_um']:.3f}|{row['steady_zero_crossings']}|{row['deadzone_fraction_active_commands']*100:.2f}|{row['minimum_nonzero_step_tip_um']:.5f}|{ratio(row['policy_teacher_oct_median_ratio'])} / {ratio(row['policy_teacher_oracle_median_ratio'])}|")
    lines += ["", "## 自动分流", "", f"选择 **情况 {result['recommendation']['case']}：先核查 MotorBackend、暂停模型微调**。",
              "“死区占比高”在本诊断显式定义为非零指令帧中无位移比例 ≥25%；该门槛是对计划定性条件的操作化解释，不是已验证物理定律。",
              "高无位移率只触发情况 1 核查；最小反馈步进、协议 handle 量化、闭环物理可达性需要分别验证，不能直接断言 20–50 µm 是物理硬下界。", "",
              "## 教师重算限制与感知核对", "",
              "日志仅保存 OCT navigation 向量，未保存原始 RGB-D geometry 与拟合质量。主教师列使用保存的 OCT 向量转世界坐标，重算现有几何教师的零初速数值控制律（当前 8 µm 停止区、3/s 增益、原速度/加速度限制），不是完整感知教师精确重放。",
              "另列 oracle 教师使用真值 target-minus-tip，仅作诊断，不进入策略控制，不伪造感知质量。教师中位数为 0 时幅值比记 null，不除零或写无穷大。", ""]
    for name, target in result["targets"].items():
        lines.append(f"- {name}：稳态 OCT 向量与真值向量差的中位范数 {target['steady_sensor_vector_error_median_um']:.3f} µm；FK/日志误差最大差 {target['maximum_error_reconstruction_delta_um']:.3g} µm。")
    lines += ["", "## 复现", "", f"`{result['command']}`", "", "证据：axis_floor_summary.json、逐目标 NPZ、切档与稳态三轴图、axis_contribution_bar.png；源码/权重/轨迹指纹见 JSON。仅验证 MuJoCo 软件仿真，不代表临床精度或硬件就绪。"]
    (args.output_dir / "AXIS_FLOOR_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "case": result["recommendation"]["case"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
