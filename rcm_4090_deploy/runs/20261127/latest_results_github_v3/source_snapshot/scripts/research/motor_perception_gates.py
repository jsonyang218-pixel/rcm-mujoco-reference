"""Execute the motor/perception recovery plan without neural training.

Oracle feedback is explicitly privileged. Perception uses only registered
RGB/depth and static calibration; truth is read separately for evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.research.diagnose_step680_axes import UNIT, digest, read, write

ROOT = Path(__file__).resolve().parents[2]
INDICES = (0, 2, 6, 8, 10)
MANIFEST = Path("configs/v12_reachable_development_targets_0p35um.json")
SOURCE = Path("runs/20261120/overnight_fast5_20261120_v1/task34/task4_dev20")
AUDIT = Path("github_reports/diag_20261120_step680_axes/motor_protocol_audit/motor_protocol_audit.json")
ENDPOINT = Path("runs/20261120/step680_case1_expert_v1/physical_check/expert_summary.json")
DATASET = Path("data/plan2_plus_original_train_224_20261119")
BASELINE_QUALITY = {"offsamples": 4, "numslices": 28, "numstacks": 16}
SINGLE_QUALITY = {"offsamples": 0, "numslices": 28, "numstacks": 16}
FIXED_QUALITY = {"offsamples": 0, "numslices": 112, "numstacks": 64}
MOTOR_FILES = ("rcm/motor_execution.py", "mujoco_model/motor_backend.py",
               "mujoco_model/vendor/simulation_driver/ros_msg/src/robot_control/kinematics.py")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def oracle_command(reference, q, tip_mm, target_mm, velocity, command_q):
    """Geometry teacher's SVD/gain/braking/acceleration law, with true points.

    This mirrors VisualServoController.geometry_action_from_observation's
    numerical law, with all three axes, strict 0.35 um tolerance, and no
    sensory visibility/quality inputs. Accumulated command_q is retained.
    """
    from types import SimpleNamespace
    from rcm.visual_servo import VisualServoController, visual_servo_settings
    nav = reference.config["navigation"]
    settings = visual_servo_settings(reference)
    dt = float(settings["sample_period_s"])
    vmax = np.asarray(nav["max_velocity_A_deg_B_deg_C_mm_per_s"]) * UNIT
    amax = np.asarray(nav["max_acceleration_A_deg_B_deg_C_mm_per_s2"]) * UNIT
    error = np.asarray(target_mm) - np.asarray(tip_mm)
    jacobian = VisualServoController._model_jacobian(SimpleNamespace(reference=reference), q)
    u, singular, vt = np.linalg.svd(jacobian * vmax, full_matrices=False)
    if singular[-1] < max(1e-10, singular[0] * 1e-6):
        raise ValueError("singular geometry teacher Jacobian")
    correction = vmax * (vt.T @ (singular / (singular**2 + (singular[0] * 1e-4)**2) * (u.T @ error)))
    projected = jacobian @ np.linalg.lstsq(jacobian, error, rcond=None)[0]
    wanted = (np.zeros(3) if np.linalg.norm(projected) * 1000 <= .35
              else float(settings["geometry_gain_per_s"]) * correction)
    cap = np.minimum(vmax, .6 * np.sqrt(2 * amax * np.abs(correction)))
    wanted = np.clip(wanted, -cap, cap)
    next_velocity = np.clip(wanted, velocity - amax * dt, velocity + amax * dt)
    return reference.validate_task(command_q + next_velocity * dt).copy(), next_velocity, jacobian


def axis_metrics(trajectory, records):
    """Post-command transitions, indexed by the corresponding pre-action step."""
    selected = [row for row in trajectory if 700 <= row["step"] < 1600 and row["step"] < len(records)]
    axes = {}
    for axis, label in enumerate("ABC"):
        commands = np.array([row["command_delta_deg_deg_mm"][axis] for row in selected])
        movement = np.array([(records[row["step"]]["actual_after_task_rad_rad_m"][axis]
                              - records[row["step"]]["actual_before_task_rad_rad_m"][axis]) / UNIT[axis]
                             for row in selected])
        gains = np.array([row["jacobian_column_um_per_deg_deg_mm"][axis] for row in selected])
        active = np.abs(commands) > 0
        moving = np.abs(movement) >= 1e-6
        axes[label] = {"unit": "mm" if axis == 2 else "deg", "frames": len(selected),
                       "active_command_frames": int(active.sum()),
                       "nonzero_command_no_motion_frames": int(np.count_nonzero(active & ~moving)),
                       "nonzero_command_no_motion_fraction": float(np.mean(~moving[active])) if active.any() else None,
                       "minimum_feedback_step": float(np.min(np.abs(movement[moving]))) if moving.any() else None,
                       "minimum_feedback_tip_step_um": float(np.min(np.abs(movement[moving]) * gains[moving])) if moving.any() else None}
    return axes


def run_oracle(args, result):
    from rcm.scene_v9 import RCMSceneV9
    from rcm.visual_servo import visual_servo_settings
    from scripts.tools.experiment_workflow import scene_and_code
    manifest, bounds = read(MANIFEST), read(AUDIT)
    result.update(target_manifest=str(MANIFEST), target_manifest_sha256=digest(MANIFEST),
                  motor_source_sha256={path: digest(path) for path in MOTOR_FILES},
                  code_scene_identity=scene_and_code(), episodes=[],
                  controller="True-position feedback geometry teacher; no RGB-D/OCT or network; original MotorBackend, accumulated absolute command_q and previous command velocity.",
                  protocol={"indices": list(INDICES), "seeds": [42+i for i in INDICES], "max_steps": 1600,
                            "control_period_s": .05, "threshold_um": .35, "consecutive_settled": 3,
                            "no_motion_threshold_deg_deg_mm": 1e-6, "axis_window": "700 <= step < 1600",
                            "initial_task_rad_rad_m": [0, 0, 0]})
    for index in INDICES:
        name = f"E{index:02d}"
        scene = RCMSceneV9()
        episode = {"episode": name, "seed": 42+index, "status": "running", "trajectory": []}
        try:
            np.random.seed(42+index)
            scene.set_task(np.zeros(3))
            target = np.asarray(manifest["targets"][index]["reachable_target_world_mm"])
            scene.set_target_world(target)
            scene.enable_motor_execution()
            motor = scene.motor_execution
            if visual_servo_settings(scene)["sample_period_s"] != .05:
                raise ValueError("control period changed")
            command_q, velocity = np.zeros(3), np.zeros(3)
            strict_streak = settled_streak = max_settled_streak = 0
            success = False
            for step in range(1600):
                q = scene.current_task().copy()
                tip = scene.site_mm("needle_tip")
                error_um = float(np.linalg.norm(target-tip) * 1000)
                settled = bool(motor.settled())
                strict_streak = strict_streak+1 if error_um <= .35 and settled else 0
                settled_streak = settled_streak+1 if settled else 0
                max_settled_streak = max(max_settled_streak, settled_streak)
                row = {"step": step, "error_um": error_um, "encoder_settled": settled,
                       "state_deg_deg_mm": (q/UNIT).tolist()}
                episode["trajectory"].append(row)
                if strict_streak >= 3:
                    success = True
                    break
                candidate, velocity, jacobian = oracle_command(scene, q, tip, target, velocity, command_q)
                row.update(command_delta_deg_deg_mm=((candidate-q)/UNIT).tolist(),
                           command_increment_deg_deg_mm=((candidate-command_q)/UNIT).tolist(),
                           jacobian_column_um_per_deg_deg_mm=(np.linalg.norm(jacobian*UNIT, axis=0)*1000).tolist())
                scene.set_task(candidate, motor_dt=.05)
                command_q = candidate.copy()
                if step == 0 or (step+1) % 200 == 0:
                    print(f"{name} true geometry teacher {step+1}/1600 error={error_um:.6f}um settled={settled}", flush=True)
            final_error = float(np.linalg.norm(target-scene.site_mm("needle_tip"))*1000)
            episode.update(status="completed", strict_success=success, steps=len(episode["trajectory"]),
                           final_error_um=final_error,
                           minimum_error_um=min(final_error, min(row["error_um"] for row in episode["trajectory"])),
                           encoder_settled_frames=sum(row["encoder_settled"] for row in episode["trajectory"]),
                           encoder_settled_at_exit=bool(motor.settled()), max_settled_consecutive_frames=max_settled_streak,
                           target_world_mm=target.tolist(), final_state_deg_deg_mm=(scene.current_task()/UNIT).tolist(),
                           local_endpoint_bound_um=bounds["candidates"][name]["best"]["error_um"],
                           axes=axis_metrics(episode["trajectory"], motor.records), motor_records=motor.records)
        except BaseException as exc:
            episode.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed", failure=repr(exc))
            raise
        finally:
            scene.close()
            write(args.output_dir/f"{name}_oracle.json", episode)
            result["episodes"].append({k:v for k,v in episode.items() if k not in ("trajectory", "motor_records")})
            record(f"Gate1a {name}", episode["status"], args.output_dir/f"{name}_oracle.json",
                   f"seed={42+index}, original manifest, geometry oracle, max1600 x50ms; "
                   f"final={episode.get('final_error_um')}um, strict={episode.get('strict_success')}; failure={episode.get('failure')}")
        print(f"{name}: {episode['final_error_um']:.6f}um; {args.output_dir/f'{name}_oracle.json'}", flush=True)
    result.update(completed_targets=len(result["episodes"]),
                  strict_success_count=sum(row["strict_success"] for row in result["episodes"]),
                  final_mean_um=float(np.mean([row["final_error_um"] for row in result["episodes"]])))
    result["strict_gate_passed"] = result["strict_success_count"] == 5
    result["gate1b"] = {
        "decision": "hardware_calibration_required; neither 1A nor hardware-conditioned 1B established",
        "absolute_handle_semantics": "Cumulative absolute int32 handles anchored to original session home; duplicate commands are idempotent, not increments.",
        "geometry_teacher_residual": "command_q and command velocity accumulate across ticks; no per-tick subhandle loss in this teacher.",
        "policy_execution_difference": "eval_policy forms desired=actual+action each tick; small learned increments can be lost. Changing this controller contract does not remove the handle endpoint lattice and is not an established backend bug.",
        "hardware_resolution_verified": False,
        "branch_1b_not_taken": "Source explicitly marks motor dynamics uncalibrated. Need physical handle calibration or authorization for a simulation-only protocol branch before replacing targets/thresholds.",
        "frozen_manifest_limitation": "Encoder-lattice reachability alone does not guarantee integer-handle reachability; +/-2-handle bounds are local, not a global proof.",
        "backend_mutated": False, "manifest_mutated": False, "threshold_mutated": False}
    result["gate2b_status"] = "eligible_pending_gate1b_closure" if result["strict_gate_passed"] else "skipped: Gate1 strict execution gate failed"
    result["phase3_status"] = "skipped: Gate1b and Gate2b not closed; all neural training paused"


def summarize_vectors(vectors):
    values = np.asarray(vectors, dtype=float)
    if len(values) == 0:
        return {"count": 0}
    return {"count": len(values), "mean_xyz_um": values.mean(axis=0).tolist(),
            "std_xyz_um": values.std(axis=0).tolist(),
            "median_norm_um": float(np.median(np.linalg.norm(values, axis=1))),
            "max_norm_um": float(np.max(np.linalg.norm(values, axis=1)))}


def resolve_training_target(directory, index, cache, seen=()):
    """Follow merge offsets and exact terminal audits before raw ancestry.

    source_episode_index in converted zero-error metadata names the *raw*
    pre-conversion episode, not its position in the converted dataset. Merge
    source_datasets[].episode_offset and terminal_audit_index are authoritative.
    Never guess a missing target from the final float32 policy state.
    """
    directory=Path(directory)
    key=(str(directory.resolve()),int(index))
    if key in seen:return None
    def load(path):
        path=Path(path)
        if path not in cache:cache[path]=read(path)
        return cache[path]
    metadata_path=directory/"metadata.json"
    if not metadata_path.is_file():return None
    metadata=load(metadata_path)
    if not 0 <= index < len(metadata["episodes"]):return None
    row=metadata["episodes"][index]
    for field,frame in (("target_robot_mm","robot_mm"),("target_world_mm","world_mm")):
        if row.get(field) is not None:
            return {"frame":frame,"point":row[field],"source":str(metadata_path),"source_episode":index}
    audit_path=directory/"data/terminal_audits.json"
    if audit_path.is_file():
        audits=load(audit_path)
        audit_index=int(row.get("terminal_audit_index",index))
        if 0 <= audit_index < len(audits):
            audit=audits[audit_index]
            if (audit.get("terminal_target_world_m") is not None
                    and audit.get("frames")==row["num_frames"]):
                return {"frame":"world_mm","point":(np.asarray(audit["terminal_target_world_m"])*1000).tolist(),
                        "source":str(audit_path),"source_episode":audit_index}
    for source in metadata.get("source_datasets",[]):
        start=int(source["episode_offset"])
        if start <= index < start+int(source["episodes"]):
            source_dir=directory.parent/Path(source["directory"]).name
            return resolve_training_target(source_dir,index-start,cache,seen+(key,))
    if row.get("source_dataset") and "source_episode_index" in row:
        source_dir=directory.parent/Path(row["source_dataset"]).name
        return resolve_training_target(source_dir,int(row["source_episode_index"]),cache,seen+(key,))
    return None


def dataset_audit(scene, output_dir):
    from scripts.research.overnight_cfix_data import _dataset_context
    metadata, arrays, info = _dataset_context(DATASET)
    _, rotation = scene.camera_pose("eye_internal")
    groups = {"Original": [], "Plan2": []}
    near_groups = {"Original": [], "Plan2": []}
    counts = {"invalid_oct": 0, "missing_target": 0}
    episode_ids = arrays["episode_index"].astype(int)
    targets,provenance,cache = {},[],{}
    for index,item in enumerate(info):
        resolved=resolve_training_target(DATASET,index,cache)
        if resolved is not None:
            targets[index]=(scene.robot_to_world(resolved["point"]) if resolved["frame"]=="robot_mm"
                            else np.asarray(resolved["point"]))
        provenance.append({"episode":index,"group":item["group"],"resolved":resolved})
    write(output_dir/"training_target_provenance.json",{
        "episodes":provenance,"source_sha256":{str(p):digest(p) for p in cache},
        "resolved_episodes":len(targets),"total_episodes":len(info)})
    ids, errors, group_ids = [], [], []
    for row in range(len(episode_ids)):
        if not arrays["oct_navigation_valid"][row]:
            counts["invalid_oct"] += 1
            continue
        index = episode_ids[row]
        if index not in targets:
            counts["missing_target"] += 1
            continue
        q = arrays["state"][row].astype(float) * UNIT
        tip = scene.robot_to_world(scene.nominal.forward(scene.state_from_task(q), enforce_limits=False))
        true_optical_mm = (rotation.T @ (targets[index]-tip)) * [1,-1,-1]
        error_um = (arrays["oct_navigation"][row].astype(float)-true_optical_mm)*1000
        group = "Original" if info[index]["group"] == "Original" else "Plan2"
        groups[group].append(error_um)
        if np.linalg.norm(true_optical_mm)*1000 < 50:near_groups[group].append(error_um)
        ids.append(row); errors.append(error_um); group_ids.append(0 if group == "Original" else 1)
    np.savez_compressed(output_dir/"stored_oct_errors.npz", row_indices=np.asarray(ids),
                        error_optical_xyz_um=np.asarray(errors), group_id=np.asarray(group_ids))
    return {"dataset": str(DATASET), "dataset_sha256": digest(DATASET/"data/dataset.npz"),
            "metadata_sha256": digest(DATASET/"metadata.json"), "total_frames": len(episode_ids),
            "skipped": counts, "groups": {key:summarize_vectors(value) for key,value in groups.items()},
            "near_50um_groups": {key:summarize_vectors(value) for key,value in near_groups.items()},
            "resolved_target_episodes":len(targets),"total_episodes":len(info),
            "method": "Every valid stored OCT vector with an independently resolved target minus FK truth. Merge offsets and exact terminal audits take precedence over raw ancestry. Unresolved targets are explicitly skipped. State stored as float32 deg/deg/mm; reconstruction precision is finite. No images regenerated or frozen data changed.",
            "action_label_scope": "Stored OCT observation bias is audited; action-label corruption is not inferred solely from observation bias."}


def run_perception(args, result):
    from rcm.scene_v9 import RCMSceneV9
    from rcm.strict_rgbd import StrictRGBDSensor
    from rcm.visual_servo import visual_servo_settings
    from scripts.verification.verify_strict_rgbd import FrozenCalibration
    result.update(source=str(SOURCE), source_summary_sha256=digest(SOURCE/"summary.json"),
                  source_checkpoint=read(SOURCE/"summary.json")["checkpoint"],
                  fixed_steps=[1000,1200,1400,1599], frames=[],
                  variants={"original":BASELINE_QUALITY,"single_sample":SINGLE_QUALITY,"corrected":FIXED_QUALITY},
                  design="Paired original/single-sample/corrected renders at identical saved states and targets. These are offline fixed-state diagnostics, not new teacher rollouts.",
                  estimator_truth_isolation="Every frame remeasured using FrozenCalibration with model/data/target truth APIs forbidden.")
    for index in INDICES:
        name = f"E{index:02d}"
        payload_path = SOURCE/f"{name}/evaluation_results.json"
        episode = read(payload_path)["episodes"][0]
        states = np.asarray(episode["trajectory"]["states"])
        target = np.asarray(episode["target_world_mm"])
        scene = RCMSceneV9()
        try:
            scene.set_target_world(target)
            for variant, quality in result["variants"].items():
                settings = visual_servo_settings(scene)
                settings["rgbd_render_quality"] = quality
                sensor = StrictRGBDSensor(scene, settings)
                frozen = StrictRGBDSensor(FrozenCalibration(scene), settings, frame_provider=object())
                try:
                    for step in result["fixed_steps"]:
                        scene.set_task(states[step]*UNIT)
                        sensor.observation_time_s = step*.05
                        observation = sensor.observe()
                        replay = frozen.measure(observation["rgb"],observation["depth"],step*.05)
                        geometry = replay["geometry"]
                        row = {"episode":name,"seed":42+index,"step":step,"variant":variant,
                               "quality":quality,"source_sha256":digest(payload_path)}
                        result["frames"].append(row)
                        if geometry.get("target_world_mm") is None or geometry.get("tip_world_mm") is None:
                            row.update(status="measurement_failed",geometry=geometry)
                            continue
                        tip = scene.site_mm("needle_tip")
                        est_target, est_tip = map(np.asarray,(geometry["target_world_mm"],geometry["tip_world_mm"]))
                        delta = est_target-target
                        camera_delta = sensor.geometry.camera_rotation.T@delta
                        nav_delta = delta-(est_tip-tip)
                        true_pixel,true_depth = sensor.geometry.project(target)
                        est_pixel,_ = sensor.geometry.project(est_target)
                        row.update(status="measured",target_error_world_um=(delta*1000).tolist(),
                                   target_error_camera_opengl_um=(camera_delta*1000).tolist(),
                                   target_pixel_error_uv=(np.asarray(est_pixel)-true_pixel).tolist(),
                                   target_optical_depth_error_um=float(-camera_delta[2]*1000),
                                   target_error_um=float(np.linalg.norm(delta)*1000),
                                   tip_error_um=float(np.linalg.norm(est_tip-tip)*1000),
                                   oct_error_world_um=(nav_delta*1000).tolist(),
                                   oct_error_um=float(np.linalg.norm(nav_delta)*1000),
                                   target_quality=geometry["target_quality"],true_target_depth_m=true_depth)
                        np.testing.assert_allclose(geometry["target_world_mm"],observation["geometry"]["target_world_mm"],atol=1e-12,rtol=0)
                        if step == 1200:
                            np.savez_compressed(args.output_dir/f"{name}_{variant}_frame1200.npz",
                                rgb=observation["rgb"],depth=observation["depth"],
                                truth_target_mm=target,truth_tip_mm=tip,state_deg_deg_mm=states[step])
                finally:
                    frozen.close();sensor.close()
            print(f"{name}: offline perception variants saved in {args.output_dir}",flush=True)
        finally:
            scene.close()
    result["statistics"] = {}
    for variant in result["variants"]:
        frames = [row for row in result["frames"] if row["variant"] == variant and row["status"] == "measured"]
        by_target = {}
        for index in INDICES:
            rows = [row for row in frames if row["episode"] == f"E{index:02d}"]
            by_target[f"E{index:02d}"] = {
                "target_world":summarize_vectors([row["target_error_world_um"] for row in rows]),
                "oct_world":summarize_vectors([row["oct_error_world_um"] for row in rows])}
        result["statistics"][variant] = {"measured_frames":len(frames),"by_target":by_target,
            "target":summarize_vectors([row["target_error_world_um"] for row in frames]),
            "oct":summarize_vectors([row["oct_error_world_um"] for row in frames])}
    scene = RCMSceneV9()
    try:
        result["training_observation_audit"] = dataset_audit(scene,args.output_dir)
    finally:
        scene.close()
    corrected = result["statistics"]["corrected"]
    result["all_sampled_corrected_oct_below_0p5um"] = (
        corrected["measured_frames"] == 20 and corrected["oct"]["max_norm_um"] < .5)
    result["attribution"] = {
        "mask": "RGB mask selects visible sphere surface samples; the estimator already fits the calibrated sphere center rather than using surface depth as center. Earlier erosion ablations did not remove the main bias.",
        "sampling": "Compare original vs single_sample: same sphere mesh/intrinsics/extrinsics and estimator, changed offscreen MSAA only. Depth resolve and RGB coverage are incompatible with the assumed single pixel-center ray at subpixel precision.",
        "tessellation": "Compare single_sample vs corrected: changed only render sphere tessellation, reducing deviation from the calibrated ideal sphere.",
        "calibration": "Same camera pose, fovy, pixel-center convention and .15mm sphere radius throughout; no fitted truth offset or access to dynamic truth in estimation.",
        "scope": "Deterministic simulation bias across four saved steady poses per target; repeated deterministic frames do not estimate real sensor random noise. This is an empirical diagnostic, not a theoretical perception limit.",
        "gate2b": "Not run here; requires a passed and closed Gate1 execution protocol."}


def record(purpose,status,path,detail):
    with (ROOT/"TEST_LOG.md").open("a",encoding="utf-8") as log:
        log.write(f"\n- {now()}: {purpose} **{status}**; `python -B -m scripts.research.motor_perception_gates`; {detail}; evidence `{path}`. CPU scalar kinematics/statistics (small/I/O-bound), NVIDIA EGL rendering; no training. MuJoCo simulation only.\n")


def reports(args,result):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    oracle = read(args.workflow_dir/"oracle/oracle_dev5_summary.json")
    perception = read(args.workflow_dir/"perception/oct_bias_breakdown.json")
    gate1 = ROOT/"github_reports/gate1_20261120_motor_oracle"
    gate2 = ROOT/"github_reports/gate2_20261120_oct_bias"
    gate1.mkdir(parents=True,exist_ok=False)
    gate2.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.workflow_dir/"oracle/oracle_dev5_summary.json",gate1/"oracle_dev5_summary.json")
    shutil.copy2(args.workflow_dir/"perception/oct_bias_breakdown.json",gate2/"oct_bias_breakdown.json")
    lines=["# Gate 1：电机执行链与 Oracle 审计","",
           f"诊断流程：{oracle.get('status')}；真值几何教师完成 {oracle.get('completed_targets',0)}/5；严格成功 {oracle.get('strict_success_count',0)}/5。",
           "固定 E00/E02/E06/E08/E10，seeds42/44/48/50/52，最多1600步，50ms，≤0.35µm且连续3帧 MotorExecution.settled。",
           "教师使用真值几何反馈并保留 command_q/速度累积；未调用 RGB-D/OCT 或网络。原 MotorBackend 和冻结目标不变。",
           "此前最近端点编码器 Oracle 是辅助可达性证据，不能冒充本轮几何教师。","",
           "|目标|终值 µm|最低 µm|停稳帧|最长连续停稳|局部端点参照 µm|","|---|---|---|---|---|---|"]
    for row in oracle["episodes"]:
        lines.append(f"|{row['episode']}|{row.get('final_error_um',float('nan')):.6f}|{row.get('minimum_error_um',float('nan')):.6f}|{row.get('encoder_settled_frames')}|{row.get('max_settled_consecutive_frames')}|{row.get('local_endpoint_bound_um',float('nan')):.6f}|")
    lines += ["","700≤step<1600 三轴非零指令无位移占比、最小反馈步长及 Jacobian 等效尖端步长见 JSON 的 episodes[].axes。",
              "指令无位移沿用每轴1e-6 deg/deg/mm阈值，反馈最小步长不等于协议 handle 量子。","",
              "## 协议分流结论","",
              "handle 是以 session home 为锚的绝对累计整数目标；重复包必须幂等。把残差加入后端会错误累加绝对命令。几何教师已有 command_q 残差累积，未证实分支1A所述的后端丢增量 bug。",
              "策略评估器按 actual+action 生成每拍目标，确有小动作可能被量化吞掉的控制器语义差别；这不能消除端点格点限制，也不能据此宣布后端修复完成。",
              "原清单只保证整数 encoder 可达，未保证整数 handle；±2 handle 搜索只给局部界，不能证明全局不可达。",
              "真实硬件 handle 步距尚无测量标定，源码标记动力学未辨识。分支1B以真实硬件约束为前提，目前不能擅自选取新目标或放宽门槛。需补充硬件标定，或明确授权仿真协议分支后另建版本。",
              "Gate 1 尚未收口。Gate 2b 和 Phase 3 暂停；不启动 Fine DiT、微调或策略复测。","",
              f"原端点实验：`{ENDPOINT}`；本轮原始记录：`{args.workflow_dir/'oracle'}`。",
              "仅验证软件/MuJoCo仿真，不代表真实硬件或临床精度。"]
    (gate1/"GATE1_MOTOR_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for column,index in enumerate((0,8)):
        episode=read(args.workflow_dir/f"oracle/E{index:02d}_oracle.json")
        rows=episode["trajectory"]
        states=np.asarray([row["state_deg_deg_mm"] for row in rows])
        for axis in range(3):
            axes[axis,column].plot([row["step"] for row in rows],states[:,axis])
            axes[axis,column].set_ylabel(f"{'ABC'[axis]} ({'mm' if axis==2 else 'deg'})")
            axes[axis,column].grid(alpha=.2)
        axes[0,column].set_title(f"E{index:02d} true geometry teacher")
        axes[-1,column].set_xlabel("pre-action step")
    fig.tight_layout();fig.savefig(gate1/"oracle_axes_E00_E08.png",dpi=150);plt.close(fig)
    lines=["# Gate 2：OCT 偏差归因（2a）","",
           f"固定五目标×4稳态帧×3渲染变体；诊断状态：{perception['status']}。未运行 Gate 2b 闭环。",
           "每帧通过禁止动态真值接口的 FrozenCalibration 重新测量；真值只用于误差评估。世界xyz、相机xyz、像素uv和光轴深度分量见 oct_bias_breakdown.json。","",
           "|渲染变体|有效帧|目标误差中位数 µm|OCT误差中位数 µm|OCT最大 µm|","|---|---|---|---|---|"]
    for name,stats in perception["statistics"].items():
        lines.append(f"|{name}|{stats['measured_frames']}|{stats['target'].get('median_norm_um')}|{stats['oct'].get('median_norm_um')}|{stats['oct'].get('max_norm_um')}|")
    lines += ["","原渲染为4重采样、28切片/16层；single_sample仅关闭多重采样；corrected再提高至112切片/64层。",
              "前一对照定位深度 resolve/覆盖与像素中心射线的不一致；后一对照定位低精度球面网格与已标定理想球面的差别。相机标定、RGB检测器和球心拟合公式不变，无经验真值偏移补偿。",
              "SyntheticOCTProvider仅提供可见表面深度；目标估计器本来就在拟合半径0.15mm球心，并非误将表面当球心。单纯掩码腐蚀未移除主偏差。",
              "跨四个确定性稳态姿态的均值/标准差记录在JSON；无法据此估计真实传感器随机噪声或证明理论上限。","",
              "## 历史存储观察","",
              json.dumps(perception["training_observation_audit"],ensure_ascii=False,indent=2),"",
              "上述检查仅覆盖目标可独立追溯的 Original 与 Plan2 有效观察；未追溯部分明确跳过，不能外推为全量审计。保留原始标签/数据；OCT观察带偏差不等同于已经证明动作标签以同样数值损坏。","",
              "## 尚未完成","",
              "Gate 2b 依赖 Gate 1 通过并冻结执行协议，目前跳过。因此没有 RGB-D 教师终值、成功率或 Oracle 配对改善结果；离线帧改善不能替代闭环验收。Phase 3 和全部训练暂停。",
              f"完整帧/NPZ证据：`{args.workflow_dir/'perception'}`。仅为仿真结果。"]
    (gate2/"GATE2_OCT_BIAS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,5))
    rows=oracle["episodes"]
    ax.bar([row["episode"] for row in rows],[row["final_error_um"] for row in rows],label="Oracle geometry teacher (measured)")
    ax.axhline(.35,color="red",linestyle="--",label="strict 0.35 um")
    ax.set_ylabel("Final error (um)");ax.legend()
    ax.set_title("Gate 2b RGB-D teacher: NOT RUN (Gate 1 unresolved)")
    fig.tight_layout();fig.savefig(gate2/"rgbd_vs_oracle_dev5.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for variant in perception["variants"]:
        vals=[perception["statistics"][variant]["by_target"][f"E{i:02d}"]["oct_world"].get("median_norm_um",np.nan) for i in INDICES]
        ax.plot([f"E{i:02d}" for i in INDICES],vals,"o-",label=variant)
    ax.axhline(.5,color="red",linestyle="--");ax.set_ylabel("Fixed-frame OCT error median (um)");ax.legend()
    fig.tight_layout();fig.savefig(gate2/"offline_oct_attribution.png",dpi=150);plt.close(fig)
    result.update(gate1_report=str(gate1),gate2_report=str(gate2),
                  workflow_scope="Gate1a and independent Gate2a diagnostics; Gate1b hardware fact unresolved; Gate2b/Phase3 skipped",
                  strict_gate_passed=oracle.get("strict_gate_passed",False),neural_training="paused")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("oracle","perception","reports"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--workflow-dir",type=Path)
    args=parser.parse_args()
    if args.mode=="reports" and args.workflow_dir is None:parser.error("reports requires --workflow-dir")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    snapshots=args.output_dir/"code_snapshots"
    for source in ("scripts/research/motor_perception_gates.py","rcm/strict_rgbd.py",*MOTOR_FILES):
        destination=snapshots/source
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/source,destination)
    result={"status":"running","started_at":now(),"mode":args.mode,"analysis_code_sha256":digest(__file__),
            "output_dir":str(args.output_dir),"checkpoint":"none: diagnostic only","no_training":True}
    filename={"oracle":"oracle_dev5_summary.json","perception":"oct_bias_breakdown.json","reports":"report_receipt.json"}[args.mode]
    try:
        {"oracle":run_oracle,"perception":run_perception,"reports":reports}[args.mode](args,result)
        result["status"]="completed"
    except BaseException as exc:
        result.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",failure=repr(exc))
        raise
    finally:
        result["finished_at"]=now()
        write(args.output_dir/filename,result)
        record(f"motor_perception_gate_v1/{args.mode}",result["status"],args.output_dir/filename,
               f"config configs/experiment_card_motor_perception_gate_20261120_v1.json; "
               f"strict={result.get('strict_success_count','N/A')}/5; final_mean={result.get('final_mean_um')}um; failure={result.get('failure')}")
        print(f"Stage {result['status']}: {args.output_dir/filename}",flush=True)


if __name__=="__main__":
    main()
