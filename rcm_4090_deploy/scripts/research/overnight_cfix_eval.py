"""Evaluate one overnight arm and publish only complete, validated summaries."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from lerobot_training.eval_policy import (
    DEFAULT_ACTION_LIMIT_DEG_DEG_MM, TASK_TO_DEG_DEG_MM, PolicyEvaluator,
)
from lerobot_training.plot_closed_loop_errors import (
    _errors_um, _plot_best_and_final, _plot_error_curves, plt,
)
from rcm.scene_v9 import RCMSceneV9

ARMS = {"I0": (4, 1), "I1": (8, 1), "I3": (8, 5)}
TARGET_COUNT = 20
BASE_SEED = 42
MAX_STEPS = 1600
ERROR_THRESHOLD_MM = 0.00035
DIAGNOSTIC_TARGETS = (8, 0, 2, 6)
FAST5_INDICES = (0, 2, 6, 8, 10)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def targets(path):
    rows = json.loads(path.read_text())["targets"][:TARGET_COUNT]
    if (len(rows) != TARGET_COUNT or len({row["id"] for row in rows}) != TARGET_COUNT
            or any("development" not in row["roles"] for row in rows)):
        raise ValueError("expected 20 distinct frozen development targets")
    return rows


def validate_episode(episode, target, index, require_latency, max_steps=MAX_STEPS):
    trace = episode["trajectory"]
    steps = episode["steps"]
    states = np.asarray(trace["states"], dtype=np.float64)
    actions = np.asarray(trace["actions"], dtype=np.float64)
    errors = np.asarray(trace["errors"], dtype=np.float64)
    final_state = np.asarray(episode["final_state_A_deg_B_deg_C_mm"], dtype=np.float64)
    terminal = episode["terminal_success"]
    if (episode["episode_index"] != index or episode["target_id"] != target["id"]
            or episode["seed"] != BASE_SEED + index or not episode["target_manifest_used"]
            or not np.allclose(episode["target_robot_mm"], target["reachable_target_robot_mm"],
                               rtol=0, atol=1e-9)
            or list(episode["reachable_encoder_counts_ABC"]) != target["reachable_encoder_counts_ABC"]):
        raise ValueError(f"E{index:02d}: target or seed mismatch")
    if (episode["early_stop_reason"] or not 1 <= steps <= max_steps
            or (not episode["success"] and steps != max_steps)
            or (episode["success"] and (not terminal["encoder_settled_at_exit"]
                                        or terminal["consecutive_frames_at_exit"] < 3
                                        or episode["final_error_mm"] > ERROR_THRESHOLD_MM))):
        raise ValueError(f"E{index:02d}: incomplete or unsafe rollout: {episode['early_stop_reason']}")
    if (states.shape != (steps, 3) or errors.shape != (steps,)
            or actions.shape != (steps - int(episode["success"]), 3)
            or len(trace["encoder_settled"]) != steps or final_state.shape != (3,)):
        raise ValueError(f"E{index:02d}: inconsistent trajectory shape")
    for values in (states, actions, errors, final_state, episode["final_error_mm"],
                   episode["final_velocity_deg_deg_mm_per_s"],
                   trace["velocities"], trace["oct_navigation_mm"], trace["simulation_time_s"]):
        if not np.isfinite(values).all():
            raise ValueError(f"E{index:02d}: nonfinite trajectory")
    if (np.any(np.abs(actions) > np.asarray(DEFAULT_ACTION_LIMIT_DEG_DEG_MM) + 1e-7)
            or np.any(np.vstack([states, final_state]) < np.array([-30, -30, 0]) - 1e-6)
            or np.any(np.vstack([states, final_state]) > np.array([30, 30, 50]) + 1e-6)):
        raise ValueError(f"E{index:02d}: command or state outside evaluation limits")
    if require_latency:
        recorded = trace["policy_inference_ms"]
        latency = np.asarray(recorded[1:], dtype=np.float64)
        if (len(recorded) != len(actions) or len(recorded) < 2 or recorded[0] is not None
                or not np.isfinite(latency).all() or np.any(latency <= 0)):
            raise ValueError(f"E{index:02d}: missing or invalid inference latency")


def validate_contract(contract, arm):
    steps, samples = ARMS[arm]
    expected = {"version": 2, "execution_backend": "MotorBackend", "control_period_s": 0.05,
                "success_error_threshold_mm": ERROR_THRESHOLD_MM,
                "required_consecutive_success_frames": 3, "flow_integration_steps": steps,
                "num_action_samples": samples, "ground_truth_used_for_control": False,
                "early_stop_error_mm": 30.0, "max_consecutive_safety_limited": 20,
                "rgb_preprocessing": "1920x1440 -> 224x224 bilinear"}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{arm}: evaluation contract mismatch")


def summarize(out, episodes, checkpoint, arm, *, indices=tuple(range(20)), max_steps=MAX_STEPS,
              tier="dev20", workers=1):
    episode_by_index = {episode["episode_index"]: episode for episode in episodes}
    if len(episodes) != len(indices) or set(episode_by_index) != set(indices):
        raise ValueError("refusing to summarize a partial evaluation")
    episodes = [episode_by_index[index] for index in indices]
    rows, jitter, latency = [], [], []
    for episode in episodes:
        trace = episode["trajectory"]
        errors = np.asarray(trace["errors"]) * 1000
        actions = np.asarray(trace["actions"])
        close = errors[:len(actions)] < 200
        changes = np.abs(np.diff(actions[:, :2], axis=0))[close[1:] & close[:-1]]
        jitter.extend(changes.tolist())
        times = trace["policy_inference_ms"][1:]
        latency.extend(times)
        rows.append({
            "episode": f"E{episode['episode_index']:02d}", "target_id": episode["target_id"],
            "seed": episode["seed"], "steps": episode["steps"],
            "final_error_um": episode["final_error_mm"] * 1000,
            "minimum_error_um": float(errors.min()), "success": episode["success"],
            "encoder_settled_at_exit": episode["terminal_success"]["encoder_settled_at_exit"],
            "encoder_settled_near_target_frames": int(np.sum(np.asarray(trace["encoder_settled"]) & (errors < 200))),
            "near_target_ab_pairs": len(changes),
            "inference_median_ms": float(np.median(times)),
        })
    c_diagnostics = {}
    figure, axes = plt.subplots(3, len(DIAGNOSTIC_TARGETS), figsize=(14, 7), layout="constrained")
    scene = RCMSceneV9()
    try:
        for column, idx in enumerate(DIAGNOSTIC_TARGETS):
            ep = episode_by_index[idx]
            trace = ep["trajectory"]
            error = np.asarray(trace["errors"]) * 1000
            target_c = scene.task_from_encoders(
                ep["reachable_encoder_counts_ABC"], reference_task=np.zeros(3)
            )[2] * TASK_TO_DEG_DEG_MM[2]
            residual = (np.asarray(trace["states"])[:, 2] - target_c) * 1000
            dc = np.asarray(trace["actions"])[:, 2] * 1000
            minimum = int(error.argmin())
            x = np.arange(max(0, minimum - 40), min(len(error), minimum + 81))
            xc = x[x < len(dc)]
            axes[0, column].set_title(f"E{idx:02d}")
            axes[0, column].plot(x - minimum, residual[x])
            axes[1, column].plot(xc - minimum, dc[xc])
            axes[2, column].plot(x - minimum, error[x])
            for axis in axes[:, column]:
                axis.axvline(0, color="gray", linestyle=":")
            post = dc[minimum:minimum + 80]
            c_diagnostics[f"E{idx:02d}"] = {
                "minimum_step": minimum, "C_residual_at_min_um": float(residual[minimum]),
                "C_retreat_after_80_steps_um": float(residual[minimum] - residual[minimum + 80])
                if minimum + 80 < len(error) else None,
                "post_min_negative_commands": int(np.sum(post < 0)),
                "post_min_under_target_negative_commands": int(np.sum(
                    (post < 0) & (residual[minimum:minimum + len(post)] < -0.001)
                )),
                "post_min_command_frames": len(post),
            }
    finally:
        scene.close()
    for axis, label in zip(axes[:, 0], ("C minus target (um)", "dC command (um)", "Tip error (um)")):
        axis.set_ylabel(label)
    for axis in axes[-1, :]:
        axis.set_xlabel("Steps from minimum")
    figure.savefig(out / "c_min_aligned.png", dpi=150)
    plt.close(figure)
    errors = _errors_um(episodes)
    _plot_error_curves(episodes, errors, 0.35, arm, out / "error_curves_all.png")
    _plot_best_and_final(episodes, errors, 0.35, out / "best_and_final_error.png")
    with (out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finals = [row["final_error_um"] for row in rows]
    payload = {
        "status": "completed", "checkpoint": str(checkpoint), "arm": arm,
        "target_count": len(rows), "base_seed": BASE_SEED, "max_steps": max_steps,
        "tier": tier, "episode_indices": list(indices), "parallel_workers": workers,
        "error_threshold_um": 0.35, "flow_steps": ARMS[arm][0], "samples": ARMS[arm][1],
        "final_mean_um": float(np.mean(finals)), "final_median_um": float(np.median(finals)),
        "final_min_um": min(finals), "minimum_mean_um": float(np.mean([r["minimum_error_um"] for r in rows])),
        "canonical_success_count": sum(r["success"] for r in rows),
        "encoder_settled_at_exit_count": sum(r["encoder_settled_at_exit"] for r in rows),
        "encoder_settled_near_target_frames": sum(r["encoder_settled_near_target_frames"] for r in rows),
        "inference_median_ms": float(np.median(latency)), "inference_p90_ms": float(np.percentile(latency, 90)),
        "inference_timing": "CUDA events: sample and aggregate, excluding initial history fill and host transfer; "
                            "observed under concurrent workers, conservative real-time gate",
        "near_target_ab_mean_change_deg": np.mean(jitter, axis=0).tolist() if jitter else None,
        "near_target_ab_pairs": len(jitter), "c_retraction": c_diagnostics,
    }
    write_json(out / "summary.json", payload)


def parallel_evaluate(args, indices, max_steps):
    """Run independent original-index shards; never publish partial final results."""
    from lerobot_training.run_model_pipeline import start_logged, stop_logged

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    selected = targets(args.target_manifest)
    pending, active, episodes = list(indices), [], []
    workers = min(args.workers, len(indices))

    def progress(status):
        write_json(out / "progress.json", {"status": status, "validated_targets": len(episodes),
                   "expected": len(indices), "episode_indices": list(indices),
                   "completed_indices": sorted(ep["episode_index"] for ep in episodes),
                   "max_steps": max_steps, "tier": "smoke" if args.smoke else args.tier})

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    progress("running")
    try:
        while pending or active:
            while pending and len(active) < workers:
                index = pending.pop(0)
                command = [sys.executable, "-B", "-u", "-m", "scripts.research.overnight_cfix_eval",
                           "--checkpoint", str(args.checkpoint), "--target-manifest", str(args.target_manifest),
                           "--output-dir", str(out / f"E{index:02d}"), "--arm", args.arm,
                           "--tier", args.tier, "--episode-index", str(index)]
                if args.smoke:
                    command += ["--smoke", "--smoke-steps", str(max_steps)]
                job = start_logged(command, out / f"E{index:02d}.log", cwd=Path.cwd(),
                                   env=os.environ.copy(), label=f"{args.arm}/E{index:02d}")
                active.append((index, job))
            for index, job in list(active):
                code = job["process"].poll()
                if code is None:
                    continue
                stop_logged([job])
                active.remove((index, job))
                if code:
                    raise RuntimeError(f"E{index:02d} failed (exit {code}); see {out / f'E{index:02d}.log'}")
                folder = out / f"E{index:02d}"
                payload = json.loads((folder / "evaluation_results.json").read_text())
                validate_contract(payload["evaluation_contract"], args.arm)
                if len(payload["episodes"]) != 1:
                    raise ValueError("expected exactly one episode per shard")
                episode = payload["episodes"][0]
                validate_episode(episode, selected[index], index, True, max_steps)
                receipt = json.loads((folder / "shard_complete.json").read_text())
                if receipt != {"status": "completed", "index": index, "max_steps": max_steps,
                               "arm": args.arm, "checkpoint": str(args.checkpoint)}:
                    raise ValueError("shard receipt mismatch")
                episodes.append(episode)
                progress("running")
                print(f"[{args.arm}] Validated shards: {len(episodes)}/{len(indices)}; {folder}", flush=True)
            if active:
                time.sleep(0.25)
        if args.smoke:
            write_json(out / "smoke.json", {"status": "passed", "indices": list(indices),
                       "seeds": [BASE_SEED + index for index in indices], "steps": max_steps,
                       "validated_targets": len(episodes), "arm": args.arm,
                       "checkpoint": str(args.checkpoint)})
        else:
            summarize(out, episodes, args.checkpoint, args.arm, indices=indices,
                      max_steps=max_steps, tier=args.tier, workers=workers)
        progress("completed")
        print(f"[{args.arm}] Results: {out}", flush=True)
    except BaseException as error:
        progress("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        raise
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        stop_logged([job for _, job in active], grace_seconds=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--tier", choices=("fast5", "dev20"), default="dev20")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episode-index", type=int, help="Internal single original-index worker")
    parser.add_argument("--smoke", action="store_true", help="Only E00; never reports a full result")
    parser.add_argument("--smoke-steps", type=int, help="E00 step limit for --smoke (default: 2)")
    args = parser.parse_args()
    if args.smoke_steps is not None and (not args.smoke or not 2 <= args.smoke_steps <= 100):
        parser.error("--smoke-steps requires --smoke and a limit between 2 and 100")
    if not 1 <= args.workers <= 5:
        parser.error("--workers must be between 1 and 5")
    indices = FAST5_INDICES if args.tier == "fast5" else tuple(range(TARGET_COUNT))
    if args.episode_index is not None and args.episode_index not in indices:
        parser.error("episode index outside tier")
    max_steps = (args.smoke_steps or 2) if args.smoke else (1000 if args.tier == "fast5" else MAX_STEPS)
    if args.workers > 1 and args.episode_index is None:
        parallel_evaluate(args, indices[:3] if args.smoke else indices, max_steps)
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    selected = targets(args.target_manifest)
    steps, samples = ARMS[args.arm]
    evaluator = PolicyEvaluator(str(args.checkpoint), device="cuda", num_action_samples=samples,
                                flow_integration_steps=steps, measure_inference=True)
    if evaluator.device.type != "cuda":
        raise RuntimeError("CUDA required")
    episodes = []
    chosen = (args.episode_index,) if args.episode_index is not None else (indices[:1] if args.smoke else indices)
    if not args.smoke and args.episode_index is None:
        write_json(out / "progress.json", {"validated_targets": 0, "expected": len(chosen)})
    for idx in chosen:
        target = selected[idx]
        payload = evaluator.evaluate(
            num_episodes=1, output_dir=out if args.episode_index is not None else out / f"E{idx:02d}", seed=BASE_SEED,
            max_steps=max_steps,
            error_threshold_mm=ERROR_THRESHOLD_MM, target_episodes=selected,
            target_start=idx, target_end=idx+1,
            progress_every=int(os.environ.get("RCM_EVAL_PROGRESS_EVERY", "100")),
        )
        validate_contract(payload["evaluation_contract"], args.arm)
        episode = payload["episodes"][0]
        validate_episode(episode, target, idx, require_latency=True, max_steps=max_steps)
        if args.episode_index is not None:
            write_json(out / "shard_complete.json", {"status": "completed", "index": idx,
                       "max_steps": max_steps, "arm": args.arm, "checkpoint": str(args.checkpoint)})
            return
        if args.smoke:
            times = episode["trajectory"]["policy_inference_ms"]
            if episode["steps"] != max_steps:
                raise ValueError("incomplete evaluation smoke")
            write_json(out / "smoke.json", {"status": "passed", "steps": max_steps, "arm": args.arm,
                                           "checkpoint": str(args.checkpoint), "target_id": target["id"],
                                           "seed": BASE_SEED + idx, "latency_ms": times})
            return
        episodes.append(episode)
        write_json(out / "progress.json", {"validated_targets": len(episodes), "expected": len(chosen)})
        print(f"[{args.arm}] Validated shards: {len(episodes)}/{len(chosen)}; {out / f'E{idx:02d}'}", flush=True)
    summarize(out, episodes, args.checkpoint, args.arm, indices=indices, max_steps=max_steps, tier=args.tier)
    print(f"[{args.arm}] Final result: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
