"""Audit protocol quantization and check five-target oracle motor reachability."""
from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.research.diagnose_step680_axes import UNIT, digest, read, write

ROOT = Path(__file__).resolve().parents[2]
INDICES = (0, 2, 6, 8, 10)
MANIFEST = Path("configs/v12_reachable_development_targets_0p35um.json")


def protocol_candidates(scene, backend, row):
    target_counts = tuple(row["reachable_encoder_counts_ABC"]) + (0,)
    center_message = backend.command_for_robot_target(target_counts)
    center = tuple(getattr(center_message, "handleEncoder" + axis) for axis in "ABCE")
    target_world = np.asarray(row["reachable_target_world_mm"])
    candidates = []
    for offset in product(range(-2, 3), repeat=3):
        handles = tuple(center[i] + offset[i] for i in range(3)) + (0,)
        robot = backend.kin.predicted_robot_encoders(backend.initial_robot, handles)
        try:
            q = scene.task_from_encoders(robot[:3], reference_task=np.zeros(3))
            scene.validate_task(q)
            scene.set_task(q)
        except ValueError:
            continue
        candidates.append({"handles_ABCE": list(handles), "robot_ABCE": list(robot),
                           "task_rad_rad_m": q.tolist(),
                           "error_um": float(np.linalg.norm(scene.site_mm("needle_tip") - target_world) * 1000)})
    if not candidates:
        raise ValueError("no valid protocol candidates")
    candidates.sort(key=lambda item: item["error_um"])
    return {"target_id": row["id"], "manifest_encoder_counts_ABC": list(target_counts[:3]),
            "center_handles_ABCE": list(center), "searched_candidates": len(candidates),
            "search_radius_handles": 2, "best": candidates[0]}


def audit(args, result):
    from rcm.scene_v9 import RCMSceneV9
    from rcm.motor_execution import MotorExecution
    from rcm.visual_servo import VisualServoController
    from types import SimpleNamespace
    from scripts.tools.experiment_workflow import scene_and_code
    diagnostic = read(args.diagnostic_summary)
    if diagnostic["status"] != "completed" or diagnostic["recommendation"]["case"] != 1:
        raise ValueError("case-1 diagnostic gate not met")
    scene = RCMSceneV9()
    try:
        if scene.motor_execution is not None:
            raise ValueError("audit requires a kinematic scene")
        scene.set_task(np.zeros(3))
        detached_motor = MotorExecution(scene)
        backend = detached_motor.backend
        result.update(source_diagnostic=str(args.diagnostic_summary),
                      diagnostic_sha256=digest(args.diagnostic_summary),
                      target_manifest=str(MANIFEST), target_manifest_sha256=digest(MANIFEST),
                      code_scene_identity=scene_and_code(), initial_robot_ABCE=list(backend.initial_robot),
                      execution_settings={"tau_s": backend.tau_s, "timeout_s": backend.timeout_s,
                                          "speed_limits_deg_deg_mm_per_s": list(backend.speed_limits),
                                          "control_period_s": detached_motor.control_period_s,
                                          "integration_step_s": detached_motor.integration_step_s,
                                          "friction_or_configurable_deadband": "No such parameter in MotorBackend; first-order lag plus integer encoder and handle quantization."},
                      controller="Exact-target oracle with original MotorBackend; no learned policy or perception. Physical feasibility only, not a deployable expert or policy comparison.",
                      candidates={}, local_handle_quantum={})
        for index in INDICES:
            row = read(MANIFEST)["targets"][index]
            entry = protocol_candidates(scene, backend, row)
            result["candidates"][f"E{index:02d}"] = entry
            best = entry["best"]
            q = np.asarray(best["task_rad_rad_m"])
            jacobian = VisualServoController._model_jacobian(SimpleNamespace(reference=scene), q)
            quantums = {}
            for axis, label in enumerate("ABC"):
                adjacent = best["handles_ABCE"].copy()
                adjacent[axis] += 1
                counts = backend.kin.predicted_robot_encoders(backend.initial_robot, adjacent)
                next_q = scene.task_from_encoders(counts[:3], reference_task=q)
                quantums[label] = {"task_increment_deg_deg_mm": ((next_q - q) / UNIT).tolist(),
                                   "linearized_tip_um": float(np.linalg.norm(jacobian @ (next_q - q)) * 1000)}
            result["local_handle_quantum"][f"E{index:02d}"] = quantums
        result.update(status="completed", attainable_in_local_search_count=sum(
            row["best"]["error_um"] < 0.35 for row in result["candidates"].values()),
                      limitation="Search is over +/-2 protocol handles around the inverse target, not a proof over all joint branches. The manifest freezes encoder-lattice endpoints, not protocol-handle endpoints. Do not change the frozen manifest.")
        lines = ["# MotorBackend 协议量化审计", "", "离线检查；不修改仿真器、驱动或冻结目标，不执行 motor.step。", "",
                 "原可达清单生成器只在整数 encoder counts 上搜索；MotorBackend 还会把目标转换为整数 handle，二者不是同一格点。",
                 "MotorBackend 没有独立摩擦或可配置死区参数；当前模型为一阶滞后、速度限制与双层整数协议量化。", "",
                 "|目标|最邻近协议端点误差 µm|A 单 handle µm|B 单 handle µm|C 单 handle µm|", "|---|---|---|---|---|"]
        for name, entry in result["candidates"].items():
            quantum = result["local_handle_quantum"][name]
            lines.append(f"|{name}|{entry['best']['error_um']:.6f}|{quantum['A']['linearized_tip_um']:.6f}|{quantum['B']['linearized_tip_um']:.6f}|{quantum['C']['linearized_tip_um']:.6f}|")
        lines += ["", "搜索范围为逆解附近每轴 ±2 handle（每目标 125 候选）；这不是跨全部运动学分支的全局不可达证明。",
                  "下一步以同一 home/session anchor、同一目标、50 ms、最多 1600 步、0.35 µm 且连续 3 帧 MotorExecution.settled，执行 5 目标编码器 oracle 专家验证。",
                  "专家已知真值目标，仅检验执行链物理可行性；不能算作网络、感知教师或临床精度结果。"]
        (args.output_dir / "MOTOR_PROTOCOL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"local_best_errors_um": {name: row["best"]["error_um"] for name, row in result["candidates"].items()},
                          "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)
    finally:
        scene.close()


def expert(args, result):
    from lerobot_training.eval_policy import DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    from rcm.scene_v9 import RCMSceneV9
    from scripts.tools.experiment_workflow import scene_and_code
    action_limit = np.asarray(DEFAULT_ACTION_LIMIT_DEG_DEG_MM, dtype=np.float64)
    audit_result = read(args.audit_summary)
    if audit_result["status"] != "completed" or audit_result["code_scene_identity"] != scene_and_code():
        raise ValueError("missing audit or changed code/scene")
    if digest(MANIFEST) != audit_result["target_manifest_sha256"]:
        raise ValueError("target manifest changed")
    result.update(audit_summary=str(args.audit_summary), audit_sha256=digest(args.audit_summary),
                  target_manifest=str(MANIFEST), target_manifest_sha256=digest(MANIFEST),
                  code_scene_identity=scene_and_code(), controller=audit_result["controller"],
                  protocol={"episode_indices": list(INDICES), "seeds": [42 + i for i in INDICES],
                            "max_steps": 1600, "control_period_s": 0.05, "error_threshold_um": 0.35,
                            "settled_consecutive_frames": 3, "action_limits_deg_deg_mm": action_limit.tolist(),
                            "uses_ground_truth_target_for_control": True, "neural_inference": False,
                            "cpu_reason": "Small scalar kinematics and protocol simulation; no tensor inference/training. NVIDIA EGL used for scene initialization."}, episodes=[])
    for index in INDICES:
        name = f"E{index:02d}"
        target_row = read(MANIFEST)["targets"][index]
        selected = audit_result["candidates"][name]["best"]
        scene = RCMSceneV9()
        try:
            scene.set_task(np.zeros(3))
            scene.set_target_world(target_row["reachable_target_world_mm"])
            scene.enable_motor_execution()
            motor = scene.motor_execution
            if list(motor.backend.initial_robot) != audit_result["initial_robot_ABCE"]:
                raise ValueError("motor session anchor differs from audit")
            goal = np.asarray(selected["task_rad_rad_m"])
            target = np.asarray(target_row["reachable_target_world_mm"])
            trajectory = []
            strict_streak = 0
            max_settled_streak = 0
            settled_streak = 0
            success = False
            for step in range(1600):
                q = scene.current_task().copy()
                error_um = float(np.linalg.norm(scene.site_mm("needle_tip") - target) * 1000)
                settled = bool(motor.settled())
                strict_streak = strict_streak + 1 if error_um < 0.35 and settled else 0
                settled_streak = settled_streak + 1 if settled else 0
                max_settled_streak = max(max_settled_streak, settled_streak)
                trajectory.append({"step": step, "error_um": error_um,
                                   "state_A_deg_B_deg_C_mm": (q / UNIT).tolist(),
                                   "encoder_settled": settled, "robot_ABCE": list(motor.backend.robot)})
                if strict_streak >= 3:
                    success = True
                    break
                delta = np.clip((goal - q) / UNIT, -action_limit, action_limit)
                desired = q + delta * UNIT
                scene.validate_task(desired)
                requested = (selected["robot_ABCE"] if np.all(np.abs((goal-q)/UNIT) <= action_limit)
                             else [int(round(v)) for v in scene.encoders(desired)] + [0])
                actual = motor.advance_robot(requested, 0.05, requested_task=desired)
                scene._apply_task_pose(actual)
                trajectory[-1]["command_delta_A_deg_B_deg_C_mm"] = delta.tolist()
                if step == 0 or (step + 1) % 200 == 0:
                    print(f"{name} step {step+1}/1600 error={error_um:.6f} um settled={settled}", flush=True)
            final_error = float(np.linalg.norm(scene.site_mm("needle_tip") - target) * 1000)
            episode = {"episode": name, "target_id": target_row["id"], "seed": 42 + index,
                       "status": "completed", "steps": len(trajectory), "strict_success": success,
                       "final_error_um": final_error, "minimum_error_um": min(row["error_um"] for row in trajectory),
                       "encoder_settled_at_exit": bool(motor.settled()),
                       "max_settled_consecutive_frames": max_settled_streak,
                       "selected_protocol_endpoint": selected, "motor_summary": motor.summary(),
                       "trajectory": trajectory, "motor_records": motor.records}
            write(args.output_dir / f"{name}_expert.json", episode)
            result["episodes"].append({key: value for key, value in episode.items() if key not in ("trajectory", "motor_records")})
            print(f"{name} finished: {final_error:.6f} um; strict={success}; {args.output_dir / (name + '_expert.json')}", flush=True)
        finally:
            scene.close()
    result.update(status="completed", completed_targets=len(result["episodes"]),
                  strict_success_count=sum(row["strict_success"] for row in result["episodes"]),
                  final_mean_um=float(np.mean([row["final_error_um"] for row in result["episodes"]])),
                  dependent_fine_dit_training="skipped: case-1 investigation only; no training authorized for another branch")
    result["physical_gate_passed"] = result["strict_success_count"] == 5
    lines = ["# 五目标纯专家物理可达性验证", "", result["controller"], "",
             f"流程完成：{result['completed_targets']}/5；严格成功：{result['strict_success_count']}/5；终值均值：{result['final_mean_um']:.6f} µm。",
             "同一 home/session anchor；E00/E02/E06/E08/E10；seed=42+index；50 ms；每目标最多 1600 步；0.35 µm 且连续 3 帧编码器停稳。", "",
             "|目标|终值 µm|最小值 µm|停稳|严格成功|", "|---|---|---|---|---|"]
    for row in result["episodes"]:
        lines.append(f"|{row['episode']}|{row['final_error_um']:.6f}|{row['minimum_error_um']:.6f}|{row['encoder_settled_at_exit']}|{row['strict_success']}|")
    lines += ["", "这是使用真值目标的编码器 oracle 上界诊断，未使用网络或感知反馈，不得与 learned policy 成功率混淆。",
              "无参数更改、无重试、无目标清单改写。物理门禁失败则不启动 Fine DiT 或数据回放训练。仅验证 MuJoCo 软件仿真。"]
    (args.output_dir / "EXPERT_REACHABILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "strict_success_count": result["strict_success_count"],
                      "physical_gate_passed": result["physical_gate_passed"], "output_dir": str(args.output_dir)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "expert"))
    parser.add_argument("--diagnostic-summary", type=Path)
    parser.add_argument("--audit-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "audit" and args.diagnostic_summary is None:
        parser.error("audit requires --diagnostic-summary")
    if args.mode == "expert" and args.audit_summary is None:
        parser.error("expert requires --audit-summary")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "mode": args.mode, "output_dir": str(args.output_dir), "analysis_code_sha256": digest(__file__)}
    try:
        (audit if args.mode == "audit" else expert)(args, result)
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        filename = "motor_protocol_audit.json" if args.mode == "audit" else "expert_summary.json"
        write(args.output_dir / filename, result)
        with (ROOT / "TEST_LOG.md").open("a", encoding="utf-8") as stream:
            stream.write(f"\n- {result['finished_at']}: step680_case1_{args.mode}_v1 **{result['status']}**; command `python -B -m scripts.research.step680_motor_reachability {args.mode} --output-dir {args.output_dir}`; config `configs/experiment_card_step680_case1_expert_20261120_v1.json`; frozen target manifest `{MANIFEST}` sha256 `{result.get('target_manifest_sha256', 'see source')}`; dataset/checkpoint N/A (oracle/no training); seeds42/44/48/50/52; strict={result.get('strict_success_count', 'N/A')}/5; final_mean_um={result.get('final_mean_um', 'N/A')}; failure={result.get('failure', 'none')}; evidence `{args.output_dir / filename}`. CPU scalar kinematics/protocol checks, NVIDIA EGL. Simulation only.\n")


if __name__ == "__main__":
    main()
