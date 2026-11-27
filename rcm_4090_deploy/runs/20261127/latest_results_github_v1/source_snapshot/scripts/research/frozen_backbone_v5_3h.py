"""Frozen-backbone v5 three-hour sprint and development validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_replay_common import (
    FAST5,
    TARGETS,
    VAL,
    identity as evaluation_identity,
)

BASE = "f49a3a27eb6debf2e03ba8776cb4387d407ada7f"
SOURCE_BRANCH = "experiment/overnight-label-fix-v4-20261124"
BRANCH = "experiment/frozen-backbone-v5-20261124"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = ROOT / "runs/20261123/homog_clean_git_v1/repository"
SOURCE_DATA = ROOT / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"
BASE_DATA = ROOT / "data/plan2_plus_original_train_224_20261119"
BASELINE = ROOT / "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth"
V4_OUTPUTS = ROOT / "outputs/homog_label_fix_20261124_v4"
FALLBACK_F = (
    V4_OUTPUTS
    / "homog_label_fix_20261124_v4_fallback_F"
    / "step_004000.pth"
)
V3_WEIGHTS = ROOT / "outputs/homog_label_fix_20261123_v3"
WORKFLOW = ROOT / "runs/20261124/homog_label_fix_20261124_v5_3h"
OUTPUTS = ROOT / "outputs/homog_label_fix_20261124_v5"
DATASETS = {
    "A": ROOT / "data/homog_label_fix_20261124_v5_A_t3_w025",
    "B": ROOT / "data/homog_label_fix_20261124_v5_B_executed_w025",
    "C": ROOT / "data/homog_label_fix_20261124_v5_C_t3_w010",
}
ARM_CONFIG = {
    "A": {
        "mode": "t3_only",
        "t3_stationary_weight": .25,
        "batches": 4000,
        "correction_ratio": .15,
        "lr_schedule": "constant",
        "lr_min": 0.,
        "near_target_um": None,
        "near_target_weight": 1.,
        "checkpoints": (2000, 4000),
        "train_eta_min": 12,
    },
    "B": {
        "mode": "executed_only",
        "t3_stationary_weight": .25,
        "batches": 8000,
        "correction_ratio": .20,
        "lr_schedule": "cosine",
        "lr_min": 1e-6,
        "near_target_um": 80.,
        "near_target_weight": 2.,
        "checkpoints": (4000, 6000, 8000),
        "train_eta_min": 20,
    },
    "C": {
        "mode": "t3_only",
        "t3_stationary_weight": .10,
        "batches": 8000,
        "correction_ratio": .20,
        "lr_schedule": "cosine",
        "lr_min": 1e-6,
        "near_target_um": 60.,
        "near_target_weight": 3.,
        "checkpoints": (4000, 6000, 8000),
        "train_eta_min": 20,
    },
}
DEV20 = tuple(range(20))
STEPS = 1600
BASELINE_FINAL_MAX_UM = 54.31
BASELINE_MIN_MAX_UM = 30.
DIRECTION_MARGIN = .02
FIRST_CHECKPOINT_SKIP_UM = 75.
HARD_BUDGET_MIN = 180.
NEW_TASK_CUTOFF_MIN = 165.
FINAL_RESERVE_MIN = 10.
DEV20_MIN_REMAINING_MIN = 35.
FAST5_ETA_MIN = 8
SNAPSHOT_ETA_MIN = 3
DEV20_ETA_MIN = 35
MODULE = "scripts.research.frozen_backbone_v5_3h"
MANIFEST_NAME = "homog_label_fix_20261124_v5_execution_manifest.json"
STATUS_NAME = "homog_label_fix_20261124_v5_status.json"
STAGE_NAME = "homog_label_fix_20261124_v5_stage_results.json"
REPORT_NAME = "homog_label_fix_20261124_v5_REPORT.md"
SUMMARY_NAME = "homog_label_fix_20261124_v5_summary.json"
ATTENTION_LR = 3e-5


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def record(purpose, status, output, details):
    with (ROOT / "TEST_LOG.md").open("ab") as stream:
        stream.write(
            (
                f"\n- {stamp()}: homog_label_fix_20261124_v5 {purpose} "
                f"**{status}**; command `{' '.join(sys.argv)}`; {details}; "
                f"output `{output}`. Software/MuJoCo simulation only.\n"
            ).encode()
        )


def signature(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def source_identity():
    files = [
        ROOT / "scripts/research/frozen_backbone_v5_3h.py",
        ROOT / "scripts/research/frozen_backbone_v5_publish.py",
        ROOT / "scripts/research/homog_label_fix_repair.py",
        ROOT / "scripts/research/homog_replay_evaluate.py",
        ROOT / "scripts/research/homog_replay_common.py",
        ROOT / "lerobot_training/dit_experience_replay.py",
        ROOT / "lerobot_training/dataset_contract.py",
        ROOT / "tests/test_frozen_backbone_v5.py",
        TARGETS,
    ]
    return {
        "base": BASE,
        "source_branch": SOURCE_BRANCH,
        "experiment_branch": BRANCH,
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in files
        },
        "evaluation_identity": evaluation_identity(),
    }


def _protected_files(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in (".pth", ".npz")
    )


def protected_hashes():
    roots = (
        SOURCE_DATA,
        V3_WEIGHTS,
        V4_OUTPUTS,
        BASELINE,
        BASE_DATA,
    )
    result = {}
    for root in roots:
        files = _protected_files(root)
        if root.is_file():
            result[str(root)] = file_sha256(root)
        else:
            result[str(root)] = {
                str(path.relative_to(root)): file_sha256(path)
                for path in files
            }
    return result


def _git(*args, env=None):
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args],
        text=True,
        env=env,
    ).strip()


def ensure_branch():
    if _git("cat-file", "-t", BASE) != "commit":
        raise ValueError("authorized v4 base is missing")
    exists = subprocess.run(
        [
            "git", "-C", str(SOURCE_REPO), "show-ref", "--verify",
            "--quiet", f"refs/heads/{BRANCH}",
        ],
        check=False,
    ).returncode == 0
    if not exists:
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "branch", BRANCH, BASE],
            check=True,
        )
    if _git("merge-base", BASE, BRANCH) != BASE:
        raise ValueError("v5 experiment branch does not descend from f49a3a27")
    return _git("rev-parse", BRANCH)


def assert_assets_unchanged(manifest):
    if protected_hashes() != manifest["protected_hashes"]:
        raise ValueError(
            "v3/v4/baseline/source protected assets changed during the run"
        )


def elapsed_minutes(started_epoch, now=None):
    now = time.time() if now is None else now
    return (now - float(started_epoch)) / 60.


def remaining_new_task_minutes(started_epoch, now=None):
    return NEW_TASK_CUTOFF_MIN - elapsed_minutes(started_epoch, now)


def can_start(
    started_epoch,
    eta_minutes,
    *,
    now=None,
    reserve=FINAL_RESERVE_MIN,
):
    now = time.time() if now is None else now
    return (
        elapsed_minutes(started_epoch, now) < NEW_TASK_CUTOFF_MIN
        and remaining_new_task_minutes(started_epoch, now)
        >= float(eta_minutes) + float(reserve)
    )


def hard_deadline_epoch(started_epoch):
    return float(started_epoch) + HARD_BUDGET_MIN * 60.


def terminate_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_managed(command, log_path, deadline_epoch, cwd=ROOT):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.time() >= deadline_epoch:
                terminate_process_group(process)
                raise TimeoutError(
                    "hard deadline exceeded; terminated process group "
                    f"{process.pid}"
                )
            time.sleep(.25)
    if process.returncode != 0:
        raise RuntimeError(
            f"command exited {process.returncode}; see {log_path}"
        )
    return process.returncode


def run_managed_many(specs, deadline_epoch):
    processes = []
    for command, log_path in specs:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((process, stream, log_path))
    try:
        while any(process.poll() is None for process, _, _ in processes):
            if time.time() >= deadline_epoch:
                for process, _, _ in processes:
                    terminate_process_group(process)
                raise TimeoutError(
                    "parallel task hard deadline exceeded; terminated all groups"
                )
            time.sleep(.25)
    finally:
        for _, stream, _ in processes:
            stream.close()
    failed = [
        (process.returncode, path)
        for process, _, path in processes
        if process.returncode != 0
    ]
    if failed:
        raise RuntimeError(f"parallel command failure: {failed}")


def env_command():
    return [
        "env",
        "-u", "PYTHONHOME",
        "-u", "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES=0",
        "RCM_DRIVER_VARIANT=g10",
        "MUJOCO_GL=egl",
        "MUJOCO_EGL_DEVICE_ID=0",
        "MPLBACKEND=Agg",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        str(ROOT / ".venv/bin/python"),
        "-E",
        "-B",
        "-u",
    ]


def train_command(
    arm,
    destination,
    *,
    resume=False,
    smoke=False,
):
    config = ARM_CONFIG[arm]
    command = env_command() + [
        "-m", "lerobot_training.dit_experience_replay", "train",
        "--data-dir", str(BASE_DATA),
        "--correction-data-dir", str(DATASETS[arm]),
        "--original-max-episode", "838",
        "--val-data-dir", str(VAL),
        "--init-from", str(BASELINE),
        "--output-dir", str(destination),
        "--mode", "ratio",
        "--correction-ratio", str(config["correction_ratio"]),
        "--backbone-lr", "0.0",
        "--attention-lr", str(ATTENTION_LR),
        "--require-temporal-bias-only",
        "--lr-schedule", config["lr_schedule"],
        "--lr-min", str(config["lr_min"]),
        "--total-batches", str(2 if smoke else config["batches"]),
        "--steps-per-epoch", "2" if smoke else "1000",
        "--batch-size", "2" if smoke else "16",
        "--val-batch-size", "1" if smoke else "64",
        "--num-workers", "0" if smoke else "4",
        "--checkpoint-every-steps", "2" if smoke else "1000",
        "--log-every-steps", "1" if smoke else "100",
        "--near-target-weight", str(config["near_target_weight"]),
    ]
    if smoke:
        command += ["--validation-max-batches", "1"]
    if config["near_target_um"] is not None:
        command += ["--near-target-um", str(config["near_target_um"])]
    if resume:
        command += ["--resume"]
    return command


def train_stage(
    arm,
    destination,
    log_path,
    deadline_epoch,
):
    destination = Path(destination)
    receipt = destination / "training_receipt.json"
    if receipt.is_file() and read(receipt).get("status") == "completed":
        return read(receipt)
    resume = (destination / "resume_state.pth").is_file()
    run_managed(
        train_command(arm, destination, resume=resume),
        log_path,
        deadline_epoch,
    )
    if not receipt.is_file():
        raise RuntimeError(f"training completed without receipt: {destination}")
    return read(receipt)


def evaluation_command(checkpoint, destination, indices, workers):
    return env_command() + [
        "-m", MODULE, "_evaluate",
        "--checkpoint", str(checkpoint),
        "--destination", str(destination),
        "--indices", ",".join(str(value) for value in indices),
        "--workers", str(workers),
    ]


def snapshot_command(checkpoint, destination):
    return env_command() + [
        "-m", MODULE, "_snapshot",
        "--checkpoint", str(checkpoint),
        "--destination", str(destination),
    ]


def evaluate_stage(
    checkpoint,
    destination,
    indices,
    workers,
    deadline_epoch,
    log_path,
):
    destination = Path(destination)
    summary = destination / "summary.json"
    if summary.is_file():
        return read(summary)
    run_managed(
        evaluation_command(checkpoint, destination, indices, workers),
        log_path,
        deadline_epoch,
    )
    return read(summary)


def snapshot_stage(checkpoint, destination, deadline_epoch, log_path):
    destination = Path(destination)
    summary_path = destination / "snapshot_task2_vs_cleanoct.json"
    if summary_path.is_file():
        return read(summary_path)["candidate"]
    run_managed(
        snapshot_command(checkpoint, destination),
        log_path,
        deadline_epoch,
    )
    return read(summary_path)["candidate"]


def _evaluate_worker(args):
    from scripts.research.homog_replay_evaluate import evaluate

    indices = tuple(int(value) for value in args.indices.split(",") if value)
    result = evaluate(
        args.checkpoint,
        1,
        "feedback",
        indices,
        args.destination,
        {"identity": evaluation_identity()},
        workers=args.workers,
        max_workers=6,
    )
    try:
        from scripts.research.homog_replay_pipeline import plots

        plots(result, args.destination)
    except Exception as error:
        write_json(
            args.destination / "plot_failure.json",
            {"status": "failed", "reason": repr(error)},
        )
    print(json.dumps({
        "destination": str(args.destination),
        "final_mean_um": result["final_mean_um"],
        "completed_targets": result["completed_targets"],
    }))


def _snapshot_worker(args):
    from scripts.research.a_axis_flip_audit import snapshots

    result = snapshots(
        {"candidate": args.checkpoint},
        args.destination,
        evaluation_identity(),
    )["candidate"]
    print(json.dumps({
        "destination": str(args.destination),
        "first_order_toward_fraction": result["first_order_toward_fraction"],
    }))


def evaluation_metrics(summary):
    return {
        "checkpoint": summary["protocol"]["checkpoint"],
        "completed_targets": int(summary["completed_targets"]),
        "final_mean_um": float(summary["final_mean_um"]),
        "min_mean_um": float(summary["min_mean_um"]),
        "strict_success_count": int(summary["strict_success_count"]),
        "final_hits_le_5_um": int(summary["final_hits_le_5_um"]),
        "final_hits_le_10_um": int(summary["final_hits_le_10_um"]),
        "final_hits_le_30_um": int(summary["final_hits_le_30_um"]),
        "absolute_drift_1000_to_1600_um": float(
            summary["absolute_drift_1000_to_1600_um"]
        ),
    }


def checkpoint_freeze_evidence(base_checkpoint, candidate_checkpoint):
    """Compare every checkpoint tensor except temporal_bias byte-for-byte."""
    import torch

    base = torch.load(
        base_checkpoint, map_location="cpu", weights_only=False
    )["model_state_dict"]
    candidate = torch.load(
        candidate_checkpoint, map_location="cpu", weights_only=False
    )["model_state_dict"]
    if set(base) != set(candidate):
        raise ValueError("freeze comparison checkpoints have different keys")
    frozen_changed = [
        name for name in base
        if "temporal_bias" not in name
        and not torch.equal(base[name], candidate[name])
    ]
    temporal_changed = [
        name for name in base
        if "temporal_bias" in name
        and not torch.equal(base[name], candidate[name])
    ]
    passed = not frozen_changed and bool(temporal_changed)
    if not passed:
        raise ValueError(
            "checkpoint freeze evidence failed: "
            f"{len(frozen_changed)} frozen tensors changed and "
            f"{len(temporal_changed)} temporal tensors changed"
        )
    return {
        "passed": True,
        "frozen_tensor_count": len(base) - sum(
            "temporal_bias" in name for name in base
        ),
        "frozen_tensors_changed": len(frozen_changed),
        "temporal_tensor_count": sum(
            "temporal_bias" in name for name in base
        ),
        "temporal_tensors_changed": len(temporal_changed),
        "comparison": "elementwise torch.equal on every non-temporal tensor",
    }


def phase1_candidate(
    arm,
    checkpoint,
    fast,
    snapshot,
    *,
    baseline_toward_fraction,
    step=None,
):
    metrics = evaluation_metrics(fast)
    toward = float(snapshot["first_order_toward_fraction"])
    reverse = {
        axis: int(value)
        for axis, value in snapshot["axis_reverse_snapshot_counts"].items()
    }
    return {
        "arm": arm,
        "step": int(step) if step is not None else None,
        "checkpoint": str(checkpoint),
        **metrics,
        "toward_fraction": toward,
        "axis_reverse_snapshot_counts": reverse,
        "baseline_toward_fraction": float(baseline_toward_fraction),
        "direction_gate_min": float(baseline_toward_fraction) - DIRECTION_MARGIN,
        "direction_gate_passed": bool(
            toward >= float(baseline_toward_fraction) - DIRECTION_MARGIN
        ),
    }


def should_skip_after_first_checkpoint(candidate):
    return float(candidate["final_mean_um"]) > FIRST_CHECKPOINT_SKIP_UM


def select_top1(candidates):
    eligible = [
        candidate for candidate in candidates
        if candidate.get("direction_gate_passed")
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda value: (
            -int(value["strict_success_count"]),
            float(value["final_mean_um"]),
            float(value["min_mean_um"]),
            float(value["absolute_drift_1000_to_1600_um"]),
            -float(value["toward_fraction"]),
            str(value["arm"]),
            int(value.get("step") or 0),
        ),
    )


def phase2_gate(candidate):
    return bool(
        float(candidate["final_mean_um"]) < BASELINE_FINAL_MAX_UM
        or float(candidate["min_mean_um"]) < BASELINE_MIN_MAX_UM
        or int(candidate["strict_success_count"]) > 0
    )


def _raw_rollout(summary, episode):
    path = Path(episode["result_path"])
    if file_sha256(path) != episode["result_sha256"]:
        raise ValueError(f"evaluation result changed: {path}")
    payload = read(path)
    return payload["episodes"][0]["trajectory"]


def late_rollout_metrics(summary, start=1000, end=1600):
    errors, actions = [], []
    flips_by_axis = []
    active_by_axis = []
    for episode in summary["episodes"]:
        trace = _raw_rollout(summary, episode)
        frame_errors = np.asarray(trace["errors"], dtype=np.float64)
        frame_actions = np.asarray(trace["actions"], dtype=np.float64)
        if len(frame_errors) < end or len(frame_actions) < end:
            raise ValueError("late-rollout segment is incomplete")
        errors.append(frame_errors[start:end] * 1000.)
        actions.append(frame_actions[start:end])
        segment = frame_actions[start:end]
        previous = segment[:-1]
        following = segment[1:]
        active = (np.abs(previous) > 1e-12) & (
            np.abs(following) > 1e-12
        )
        flips_by_axis.append(
            active & (np.sign(previous) * np.sign(following) < 0)
        )
        active_by_axis.append(active)
    errors = np.concatenate(errors).reshape(-1)
    actions = np.concatenate(actions, axis=0)
    action_norm = np.linalg.norm(actions, axis=1)
    flips = np.concatenate(flips_by_axis, axis=0)
    active = np.concatenate(active_by_axis, axis=0)
    axis_flip_rates = {
        axis: float(flips[:, index].sum() / max(1, active[:, index].sum()))
        for index, axis in enumerate("ABC")
    }
    return {
        "episodes": int(len(summary["episodes"])),
        "mean_action_norm_deg_deg_mm": float(action_norm.mean()),
        "mean_late_error_um": float(errors.mean()),
        "axis_sign_flip_rates": axis_flip_rates,
        "start_step": int(start),
        "end_step_exclusive": int(end),
    }


def diagnose_stall_or_oscillation(
    baseline_summary,
    candidate_summary,
):
    baseline = late_rollout_metrics(baseline_summary)
    candidate = late_rollout_metrics(candidate_summary)
    action_ratio = (
        candidate["mean_action_norm_deg_deg_mm"]
        / baseline["mean_action_norm_deg_deg_mm"]
        if baseline["mean_action_norm_deg_deg_mm"] > 0.
        else math.inf
    )
    error_ratio = (
        candidate["mean_late_error_um"] / baseline["mean_late_error_um"]
        if baseline["mean_late_error_um"] > 0.
        else math.inf
    )
    if action_ratio <= .5 and error_ratio >= 1.:
        classification = "stalling"
    elif max(candidate["axis_sign_flip_rates"].values()) >= .20:
        classification = "oscillation"
    else:
        classification = "mixed_or_unresolved"
    return {
        "classification": classification,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_to_baseline": {
            "action_norm_ratio": float(action_ratio),
            "late_error_ratio": float(error_ratio),
        },
        "rules": (
            "stalling if action_norm_ratio<=0.5 and late_error_ratio>=1; "
            "else oscillation if any axis flip rate>=0.2; else mixed"
        ),
    }


def build_phase0_datasets(output):
    from scripts.research.homog_label_fix_repair import build_fixed_dataset

    results = {}
    for arm, dataset in DATASETS.items():
        config = ARM_CONFIG[arm]
        audit_dir = (
            output / f"homog_label_fix_20261124_v5_phase0_{arm}_audit"
        )
        if dataset.exists():
            receipt_path = dataset / "dataset_receipt.json"
            audit_path = dataset / "dataset_audit.json"
            if not receipt_path.is_file() or not audit_path.is_file():
                raise FileExistsError(
                    f"unregistered v5 dataset destination: {dataset}"
                )
            receipt, audit = read(receipt_path), read(audit_path)
            if (
                receipt.get("mode") != config["mode"]
                or float(receipt.get("t3_stationary_weight", -1))
                != float(config["t3_stationary_weight"])
            ):
                raise ValueError(f"resume dataset contract mismatch: {dataset}")
        else:
            audit, receipt = build_fixed_dataset(
                audit_dir,
                dataset,
                config["mode"],
                t3_stationary_weight=config["t3_stationary_weight"],
            )
        if audit["status"] != "passed" or not all(
            gate["passed"] for gate in audit["gates"].values()
        ):
            raise ValueError(f"Phase 0 dataset audit failed: V5-{arm}")
        results[arm] = {
            "status": "passed",
            "dataset": str(dataset),
            "dataset_sha256": receipt["fixed_dataset_sha256"],
            "mode": audit["mode"],
            "t3_stationary_weight": audit["t3_stationary_weight"],
            "valid_windows": int(audit["valid_windows"]),
            "nonzero_weight_fraction": float(
                audit["nonzero_weight_fraction"]
            ),
            "empirical_t3_stationary_action_fraction": float(
                audit["empirical_t3_stationary_action_fraction"]
            ),
            "theoretical_weighted_t3_stationary_action_fraction": float(
                audit[
                    "theoretical_weighted_t3_stationary_action_fraction"
                ]
            ),
            "correction_sample_weight": audit["correction_sample_weight"],
        }
    write_json(
        output / "homog_label_fix_20261124_v5_phase0_datasets.json",
        results,
    )
    return results


def phase0(
    output,
    started_epoch,
):
    deadline = hard_deadline_epoch(started_epoch)
    result = {
        "status": "running",
        "datasets": build_phase0_datasets(output),
    }
    baseline_snapshot = snapshot_stage(
        BASELINE,
        output / "reports/homog_label_fix_20261124_v5_phase0_baseline_snapshot9",
        deadline,
        output / "homog_label_fix_20261124_v5_phase0_baseline_snapshot.log",
    )
    result["baseline_snapshot"] = {
        "toward_fraction": float(
            baseline_snapshot["first_order_toward_fraction"]
        ),
        "axis_reverse_snapshot_counts": {
            axis: int(value)
            for axis, value in baseline_snapshot[
                "axis_reverse_snapshot_counts"
            ].items()
        },
        "snapshots": int(baseline_snapshot["snapshots"]),
    }
    if not can_start(
        started_epoch,
        DEV20_ETA_MIN,
        now=time.time(),
    ):
        result.update(
            status="partial_time",
            reason="baseline and Fallback F dev20 did not fit the T+165 gate",
        )
        write_json(
            output / "homog_label_fix_20261124_v5_phase0_gates.json",
            result,
        )
        return result
    baseline_dest = (
        output / "reports/homog_label_fix_20261124_v5_phase0_baseline_dev20"
    )
    fallback_dest = (
        output / "reports/homog_label_fix_20261124_v5_phase0_fallback_F_dev20"
    )
    run_managed_many(
        [
            (
                evaluation_command(BASELINE, baseline_dest, DEV20, 3),
                output / "homog_label_fix_20261124_v5_phase0_baseline_dev20.log",
            ),
            (
                evaluation_command(FALLBACK_F, fallback_dest, DEV20, 3),
                output / "homog_label_fix_20261124_v5_phase0_fallback_F_dev20.log",
            ),
        ],
        deadline,
    )
    baseline_summary = read(baseline_dest / "summary.json")
    fallback_summary = read(fallback_dest / "summary.json")
    result["baseline_dev20"] = evaluation_metrics(baseline_summary)
    result["fallback_F_dev20"] = evaluation_metrics(fallback_summary)
    result["evaluator_identity_match"] = (
        baseline_summary["protocol"]["identity"]
        == fallback_summary["protocol"]["identity"]
    )
    result["late_diagnostic"] = diagnose_stall_or_oscillation(
        baseline_summary,
        fallback_summary,
    )
    result["status"] = "completed"
    write_json(
        output / "homog_label_fix_20261124_v5_phase0_gates.json",
        result,
    )
    record("phase0", "completed", output, {
        "baseline_toward_fraction": result["baseline_snapshot"][
            "toward_fraction"
        ],
        "diagnosis": result["late_diagnostic"]["classification"],
        "baseline_dev20_final_mean_um": result["baseline_dev20"][
            "final_mean_um"
        ],
        "fallback_dev20_final_mean_um": result["fallback_F_dev20"][
            "final_mean_um"
        ],
    })
    return result


def phase1(output, started_epoch, phase0_result):
    deadline = hard_deadline_epoch(started_epoch)
    baseline_toward = float(
        phase0_result["baseline_snapshot"]["toward_fraction"]
    )
    result = {
        "status": "running",
        "baseline_toward_fraction": baseline_toward,
        "direction_gate_min": baseline_toward - DIRECTION_MARGIN,
        "arms": {},
        "candidates": [],
    }
    for arm in ("A", "B", "C"):
        config = ARM_CONFIG[arm]
        if not can_start(started_epoch, config["train_eta_min"]):
            result["arms"][arm] = {
                "status": "skipped_time",
                "reason": (
                    f"conservative {config['train_eta_min']} min training ETA "
                    f"plus {FINAL_RESERVE_MIN} min reserve did not fit "
                    "before T+165"
                ),
            }
            continue
        destination = (
            OUTPUTS / f"homog_label_fix_20261124_v5_phase1_{arm}"
        )
        arm_result = {
            "status": "training",
            "config": config,
            "checkpoints": {},
        }
        try:
            receipt = train_stage(
                arm,
                destination,
                output / f"homog_label_fix_20261124_v5_phase1_{arm}_train.log",
                deadline,
            )
        except BaseException as error:
            arm_result.update(status="failed", reason=repr(error))
            result["arms"][arm] = arm_result
            write_json(
                output / "homog_label_fix_20261124_v5_phase1_gates.json",
                result,
            )
            if isinstance(error, (KeyboardInterrupt, TimeoutError)):
                raise
            continue
        arm_result["status"] = "training_completed"
        arm_result["training_receipt"] = receipt
        for checkpoint_index, step in enumerate(config["checkpoints"]):
            if not can_start(
                started_epoch,
                FAST5_ETA_MIN + SNAPSHOT_ETA_MIN,
            ):
                arm_result["checkpoints"][str(step)] = {
                    "status": "skipped_time",
                }
                break
            checkpoint = destination / f"step_{step:06d}.pth"
            if not checkpoint.is_file():
                arm_result["checkpoints"][str(step)] = {
                    "status": "missing",
                }
                break
            fast_dest = (
                output / "reports/homog_label_fix_20261124_v5_phase1"
                / f"{arm}_step{step}_fast5"
            )
            snapshot_dest = (
                output / "reports/homog_label_fix_20261124_v5_phase1"
                / f"{arm}_step{step}_snapshot9"
            )
            try:
                fast = evaluate_stage(
                    checkpoint,
                    fast_dest,
                    FAST5,
                    5,
                    deadline,
                    output / (
                        "homog_label_fix_20261124_v5_phase1_"
                        f"{arm}_{step}_fast5.log"
                    ),
                )
                snapshot = snapshot_stage(
                    checkpoint,
                    snapshot_dest,
                    deadline,
                    output / (
                        "homog_label_fix_20261124_v5_phase1_"
                        f"{arm}_{step}_snapshot9.log"
                    ),
                )
            except BaseException as error:
                arm_result["checkpoints"][str(step)] = {
                    "status": "failed",
                    "reason": repr(error),
                }
                if isinstance(error, (KeyboardInterrupt, TimeoutError)):
                    raise
                break
            candidate = phase1_candidate(
                arm,
                checkpoint,
                fast,
                snapshot,
                baseline_toward_fraction=baseline_toward,
                step=step,
            )
            candidate.update({
                "mode": config["mode"],
                "t3_stationary_weight": config["t3_stationary_weight"],
                "correction_ratio": config["correction_ratio"],
                "lr_schedule": config["lr_schedule"],
                "lr_min": config["lr_min"],
                "near_target_um": config["near_target_um"],
                "near_target_weight": config["near_target_weight"],
                "fast5_destination": str(fast_dest),
                "snapshot9_destination": str(snapshot_dest),
            })
            arm_result["checkpoints"][str(step)] = candidate
            result["candidates"].append(candidate)
            if (
                checkpoint_index == 0
                and should_skip_after_first_checkpoint(candidate)
            ):
                arm_result["early_stop"] = True
                arm_result["reason"] = (
                    f"first checkpoint final_mean_um>"
                    f"{FIRST_CHECKPOINT_SKIP_UM:g}; later checkpoint "
                    "evaluations skipped"
                )
                break
        result["arms"][arm] = arm_result
        write_json(
            output / "homog_label_fix_20261124_v5_phase1_gates.json",
            result,
        )
    top1 = select_top1(result["candidates"])
    result["top1"] = top1
    result["top1_mode"] = "ranked_direction_gate" if top1 else "none"
    result["status"] = "completed"
    write_json(
        output / "homog_label_fix_20261124_v5_phase1_gates.json",
        result,
    )
    record("phase1", "completed", output, {
        "candidate_count": len(result["candidates"]),
        "top1": top1,
    })
    return result


def phase2(output, started_epoch, phase0_result, phase1_result):
    deadline = hard_deadline_epoch(started_epoch)
    top1 = phase1_result.get("top1")
    result = {
        "status": "running",
        "gate": phase2_gate(top1) if top1 else False,
        "remaining_new_task_minutes": remaining_new_task_minutes(
            started_epoch
        ),
    }
    if top1 is None:
        result.update(
            status="skipped_no_direction_qualified_top1",
            reason="no Phase 1 checkpoint passed the baseline-relative gate",
        )
        write_json(
            output / "homog_label_fix_20261124_v5_phase2_dev20.json",
            result,
        )
        return result
    if not result["gate"]:
        result.update(
            status="skipped_metric_gate",
            reason=(
                f"requires final_mean_um<{BASELINE_FINAL_MAX_UM}, "
                f"min_mean_um<{BASELINE_MIN_MAX_UM:g}, or strict_success>0"
            ),
        )
        write_json(
            output / "homog_label_fix_20261124_v5_phase2_dev20.json",
            result,
        )
        return result
    if not can_start(started_epoch, DEV20_ETA_MIN):
        result.update(
            status="skipped_time",
            reason=(
                f"fewer than {DEV20_ETA_MIN + FINAL_RESERVE_MIN} minutes "
                "remain before T+165"
            ),
        )
        write_json(
            output / "homog_label_fix_20261124_v5_phase2_dev20.json",
            result,
        )
        return result
    destination = (
        output / "reports/homog_label_fix_20261124_v5_phase2_top1_dev20"
    )
    summary = evaluate_stage(
        top1["checkpoint"],
        destination,
        DEV20,
        3,
        deadline,
        output / "homog_label_fix_20261124_v5_phase2_top1_dev20.log",
    )
    baseline_fast = phase0_result["baseline_dev20"]
    metrics = evaluation_metrics(summary)
    result.update(
        status="completed",
        top1=top1,
        top1_dev20=metrics,
        baseline_dev20=baseline_fast,
        delta_final_mean_um=(
            metrics["final_mean_um"] - baseline_fast["final_mean_um"]
        ),
        delta_min_mean_um=(
            metrics["min_mean_um"] - baseline_fast["min_mean_um"]
        ),
        evaluator_identity_match=(
            summary["protocol"]["identity"]
            == read(
                output
                / "reports/homog_label_fix_20261124_v5_phase0_baseline_dev20"
                / "summary.json"
            )["protocol"]["identity"]
        ),
    )
    write_json(
        output / "homog_label_fix_20261124_v5_phase2_dev20.json",
        result,
    )
    record("phase2_dev20", "completed", output, metrics)
    return result


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def make_report(output, summary, status, reason=None):
    lines = [
        "# Frozen-Backbone v5 three-hour sprint and dev20 validation",
        "",
        f"Status: `{status}`. Base: `{BASE}`. Branch: `{BRANCH}`.",
        "",
        (
            "Scope: software and MuJoCo simulation only. This is not a "
            "clinical or hardware-readiness result."
        ),
        "",
        "## Budget and execution",
        "",
        f"- Hard budget: {HARD_BUDGET_MIN:g} minutes.",
        f"- No-new-task cutoff: T+{NEW_TASK_CUTOFF_MIN:g} minutes.",
        f"- Reserved close-out: {FINAL_RESERVE_MIN:g} minutes.",
        f"- Elapsed: {_fmt(summary.get('elapsed_minutes'), 2)} minutes.",
    ]
    if reason:
        lines.append(f"- Failure/interruption reason: `{reason}`")
    lines += ["", "## Baseline and Phase 0", ""]
    phase0 = summary.get("stages", {}).get("phase0")
    if phase0:
        lines += [
            (
                "- Baseline 9-snapshot toward fraction: "
                f"{_fmt(phase0['baseline_snapshot']['toward_fraction'])}; "
                "axis reverse counts: "
                f"`{phase0['baseline_snapshot']['axis_reverse_snapshot_counts']}`."
            ),
            (
                "- Baseline dev20 final/min mean: "
                f"{_fmt(phase0['baseline_dev20']['final_mean_um'])}/"
                f"{_fmt(phase0['baseline_dev20']['min_mean_um'])} um; "
                "<=30/10/5 hit rates: "
                f"{phase0['baseline_dev20']['final_hits_le_30_um']}/20, "
                f"{phase0['baseline_dev20']['final_hits_le_10_um']}/20, "
                f"{phase0['baseline_dev20']['final_hits_le_5_um']}/20."
            ),
            (
                "- Fallback F dev20 final/min mean: "
                f"{_fmt(phase0['fallback_F_dev20']['final_mean_um'])}/"
                f"{_fmt(phase0['fallback_F_dev20']['min_mean_um'])} um; "
                "<=30/10/5 hit rates: "
                f"{phase0['fallback_F_dev20']['final_hits_le_30_um']}/20, "
                f"{phase0['fallback_F_dev20']['final_hits_le_10_um']}/20, "
                f"{phase0['fallback_F_dev20']['final_hits_le_5_um']}/20."
            ),
            (
                "- Late 1000--1600 diagnosis: "
                f"`{phase0['late_diagnostic']['classification']}`; "
                "action-norm ratio "
                f"{_fmt(phase0['late_diagnostic']['candidate_to_baseline']['action_norm_ratio'])}, "
                "late-error ratio "
                f"{_fmt(phase0['late_diagnostic']['candidate_to_baseline']['late_error_ratio'])}, "
                "axis flip rates "
                f"`{phase0['late_diagnostic']['candidate']['axis_sign_flip_rates']}`."
            ),
        ]
    else:
        lines.append("- Phase 0 did not produce a reportable result.")
    lines += [
        "",
        "## Dataset sampling evidence",
        "",
        "| Arm | Mode | T3 weight | Nonzero weight fraction | "
        "Empirical T3 stationary fraction | Theoretical weighted fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    datasets = (phase0 or {}).get("datasets", {})
    for arm in ("A", "B", "C"):
        value = datasets.get(arm)
        if value:
            lines.append(
                f"| V5-{arm} | `{value['mode']}` | "
                f"{_fmt(value['t3_stationary_weight'])} | "
                f"{_fmt(value['nonzero_weight_fraction'])} | "
                f"{_fmt(value['empirical_t3_stationary_action_fraction'])} | "
                f"{_fmt(value['theoretical_weighted_t3_stationary_action_fraction'])} |"
            )
    lines += [
        "",
        "`nonzero_weight_fraction` only proves that stationary frames retained a "
        "nonzero weight. It is not evidence that their empirical action share "
        "was calibrated.",
        "",
        "## Phase 1 fast5",
        "",
        "| Arm/step | Direction gate | Toward fraction | Reverse counts | "
        "Final mean um | Min mean um | Strict | <=30/10/5 | Drift um |",
        "|---|---|---:|---|---:|---:|---:|---|---:|",
    ]
    phase1 = summary.get("stages", {}).get("phase1", {})
    for candidate in phase1.get("candidates", []):
        lines.append(
            f"| {candidate['arm']}/{candidate['step']} | "
            f"{candidate['direction_gate_passed']} | "
            f"{_fmt(candidate['toward_fraction'])} | "
            f"`{candidate['axis_reverse_snapshot_counts']}` | "
            f"{_fmt(candidate['final_mean_um'])} | "
            f"{_fmt(candidate['min_mean_um'])} | "
            f"{candidate['strict_success_count']}/5 | "
            f"{candidate['final_hits_le_30_um']}/"
            f"{candidate['final_hits_le_10_um']}/"
            f"{candidate['final_hits_le_5_um']} | "
            f"{_fmt(candidate['absolute_drift_1000_to_1600_um'])} |"
        )
    top1 = phase1.get("top1")
    lines += ["", "## Top-1 and dev20", ""]
    if top1:
        lines.append(
            f"- Top-1: `V5-{top1['arm']}` step {top1['step']} "
            f"(`{top1['checkpoint']}`). Ranking: strict success descending, "
            "final mean ascending, min mean ascending, absolute drift "
            "ascending, toward fraction descending."
        )
    else:
        lines.append("- Top-1: none; no checkpoint passed the direction gate.")
    phase2 = summary.get("stages", {}).get("phase2", {})
    lines.append(
        f"- dev20 status: `{phase2.get('status', 'not_started')}`. "
        f"Metric gate: `{phase2.get('gate')}`."
    )
    if phase2.get("status") == "completed":
        value = phase2["top1_dev20"]
        lines += [
            (
                "- Top-1 dev20 final/min mean: "
                f"{_fmt(value['final_mean_um'])}/"
                f"{_fmt(value['min_mean_um'])} um; "
                f"strict {value['strict_success_count']}/20; "
                "<=30/10/5 hit rates: "
                f"{value['final_hits_le_30_um']}/20, "
                f"{value['final_hits_le_10_um']}/20, "
                f"{value['final_hits_le_5_um']}/20."
            ),
            (
                "- Delta versus Phase 0 baseline: final "
                f"{phase2['delta_final_mean_um']:+.4f} um, min "
                f"{phase2['delta_min_mean_um']:+.4f} um; evaluator identity "
                f"match `{phase2['evaluator_identity_match']}`."
            ),
        ]
    elif phase2.get("reason"):
        lines.append(f"- Reason: {phase2['reason']}")
    lines += [
        "",
        "## Interpretation",
        "",
        "Only relative development-target comparisons are reported. Failed "
        "gates were not retried or retuned, and no final-test or hardware "
        "claim is made.",
        "",
    ]
    path = Path(output) / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def smoke(output):
    output = Path(output).resolve()
    smoke_root = output / "smoke"
    receipt_path = (
        smoke_root / "homog_label_fix_20261124_v5_smoke_receipt.json"
    )
    if receipt_path.is_file():
        receipt = read(receipt_path)
        if receipt.get("identity") == source_identity() and receipt.get("passed"):
            return receipt
        raise FileExistsError("existing v5 smoke has different provenance")
    smoke_root.mkdir(parents=True, exist_ok=False)
    from scripts.research.homog_label_fix_repair import build_fixed_dataset

    datasets = {}
    audits = {}
    for arm, config in ARM_CONFIG.items():
        dataset = (
            smoke_root
            / f"homog_label_fix_20261124_v5_smoke_{arm}_dataset"
        )
        audit_dir = (
            smoke_root
            / f"homog_label_fix_20261124_v5_smoke_{arm}_audit"
        )
        audit, receipt = build_fixed_dataset(
            audit_dir,
            dataset,
            config["mode"],
            t3_stationary_weight=config["t3_stationary_weight"],
        )
        datasets[arm] = dataset
        audits[arm] = {
            "status": audit["status"],
            "t3_stationary_weight": audit["t3_stationary_weight"],
            "nonzero_weight_fraction": audit["nonzero_weight_fraction"],
            "empirical_t3_stationary_action_fraction": audit[
                "empirical_t3_stationary_action_fraction"
            ],
            "theoretical_weighted_t3_stationary_action_fraction": audit[
                "theoretical_weighted_t3_stationary_action_fraction"
            ],
            "receipt": receipt,
        }
    global DATASETS
    final_datasets = DATASETS
    destination = (
        smoke_root
        / "homog_label_fix_20261124_v5_smoke_train_C"
    )
    DATASETS = {**DATASETS, "C": datasets["C"]}
    try:
        command = train_command("C", destination, smoke=True)
        started = time.monotonic()
        run_managed(
            command,
            smoke_root / "homog_label_fix_20261124_v5_smoke.log",
            time.time() + 30 * 60,
        )
    finally:
        DATASETS = final_datasets
    training_receipt = read(destination / "training_receipt.json")
    manifest = read(destination / "training_manifest.json")
    history = read(destination / "training_history.json")
    required_live_fields = {
        "correction_samples",
        "correction_action_frames",
        "t3_stationary_action_frames",
        "stationary_windows",
        "actual_t3_stationary_action_fraction",
        "theoretical_weighted_t3_stationary_action_fraction",
        "all_batch_theoretical_t3_stationary_action_fraction",
        "all_batch_actual_t3_stationary_action_fraction",
    }
    if (
        not history
        or not required_live_fields.issubset(history[-1]["t3_stationary_telemetry"])
        or not required_live_fields.issubset(
            training_receipt["t3_stationary_telemetry_cumulative"]
        )
        or "t3_stationary_telemetry" not in manifest
    ):
        raise ValueError("v5 smoke did not persist stationary telemetry")
    receipt = {
        "status": "completed",
        "passed": True,
        "identity": source_identity(),
        "timestamp": stamp(),
        "elapsed_s": time.monotonic() - started,
        "command": command,
        "datasets": {arm: str(path) for arm, path in datasets.items()},
        "audits": audits,
        "checkpoint": str(destination / "step_000002.pth"),
        "checkpoint_sha256": file_sha256(
            destination / "step_000002.pth"
        ),
        "checkpoint_freeze_evidence": checkpoint_freeze_evidence(
            BASELINE,
            destination / "step_000002.pth",
        ),
        "training_receipt": training_receipt,
        "three_hour_timer_started": False,
        "simulation_scope": "Software/MuJoCo only",
    }
    write_json(receipt_path, receipt)
    record("2_batch_smoke", "passed", smoke_root, {
        "identity": signature(source_identity()),
        "checkpoint": receipt["checkpoint"],
        "telemetry": training_receipt[
            "t3_stationary_telemetry_cumulative"
        ],
    })
    return receipt


def execution_manifest(output, require_smoke=True):
    output = Path(output).resolve()
    if output != WORKFLOW:
        raise ValueError(f"authorized workflow path is {WORKFLOW}")
    manifest_path = output / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = read(manifest_path)
        if manifest["identity"] != source_identity():
            raise ValueError("workflow source identity changed")
        if manifest["protected_hashes"] != protected_hashes():
            raise ValueError("protected assets changed")
        return manifest
    if any(path.exists() for path in (*DATASETS.values(), OUTPUTS)):
        raise FileExistsError(
            "v5 dataset/output destinations are occupied without a manifest"
        )
    smoke_path = (
        output
        / "smoke"
        / "homog_label_fix_20261124_v5_smoke_receipt.json"
    )
    if require_smoke:
        if not smoke_path.is_file():
            raise FileNotFoundError(f"successful smoke receipt required: {smoke_path}")
        smoke_value = read(smoke_path)
        if (
            not smoke_value.get("passed")
            or smoke_value.get("identity") != source_identity()
        ):
            raise ValueError("smoke receipt is failed or has changed provenance")
    ensure_branch()
    from scripts.research.a_axis_flip_common import gpu_check

    manifest = {
        "status": "preflight_passed",
        "timestamp": stamp(),
        "identity": source_identity(),
        "protected_hashes": protected_hashes(),
        "smoke_receipt": str(smoke_path),
        "branch_parent": ensure_branch(),
        "gpu": gpu_check(),
        "datasets": {arm: str(path) for arm, path in DATASETS.items()},
        "outputs": str(OUTPUTS),
        "fixed_training": {
            "backbone_lr": 0.,
            "attention_lr": ATTENTION_LR,
            "only_temporal_bias_trainable": True,
        },
        "gates": {
            "direction_margin": DIRECTION_MARGIN,
            "first_checkpoint_skip_final_mean_um": (
                FIRST_CHECKPOINT_SKIP_UM
            ),
            "phase2_final_mean_um": BASELINE_FINAL_MAX_UM,
            "phase2_min_mean_um": BASELINE_MIN_MAX_UM,
            "phase2_strict_success_count": 1,
            "dev20_min_remaining_minutes": DEV20_MIN_REMAINING_MIN,
        },
        "budget": {
            "hard_minutes": HARD_BUDGET_MIN,
            "new_task_cutoff_minutes": NEW_TASK_CUTOFF_MIN,
            "final_reserve_minutes": FINAL_RESERVE_MIN,
        },
        "simulation_scope": "Software/MuJoCo only",
    }
    write_json(manifest_path, manifest)
    return manifest


def build_summary(output, stage_results, status, reason, started_epoch):
    value = {
        "status": status,
        "reason": reason,
        "started_epoch": float(started_epoch),
        "started_at": datetime.fromtimestamp(
            started_epoch, timezone.utc
        ).astimezone().isoformat(timespec="seconds"),
        "finished_at": stamp(),
        "elapsed_minutes": elapsed_minutes(started_epoch),
        "hard_budget_minutes": HARD_BUDGET_MIN,
        "new_task_cutoff_minutes": NEW_TASK_CUTOFF_MIN,
        "stages": stage_results,
        "simulation_scope": "Software/MuJoCo only",
    }
    write_json(output / SUMMARY_NAME, value)
    return value


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "homog_label_fix_20261124_v5_workflow.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("v5 workflow is already running") from error
        manifest = execution_manifest(output)
        state_path = output / STATUS_NAME
        if state_path.is_file():
            state = read(state_path)
            started_epoch = float(state["started_epoch"])
        else:
            started_epoch = time.time()
            state = {
                "status": "running",
                "stage": "phase0",
                "started_at": stamp(),
                "started_epoch": started_epoch,
                "new_task_stop_at": datetime.fromtimestamp(
                    started_epoch + NEW_TASK_CUTOFF_MIN * 60,
                    timezone.utc,
                ).astimezone().isoformat(timespec="seconds"),
                "hard_stop_at": datetime.fromtimestamp(
                    hard_deadline_epoch(started_epoch),
                    timezone.utc,
                ).astimezone().isoformat(timespec="seconds"),
            }
            write_json(state_path, state)
            write_json(
                output / "homog_label_fix_20261124_v5_timer_started.json",
                state,
            )
        stages = {}
        status, reason = "failed", None
        try:
            state.update(status="running", stage="phase0")
            write_json(state_path, state)
            stages["phase0"] = phase0(output, started_epoch)
            state.update(status="running", stage="phase1")
            write_json(state_path, state)
            stages["phase1"] = phase1(
                output, started_epoch, stages["phase0"],
            )
            state.update(status="running", stage="phase2")
            write_json(state_path, state)
            stages["phase2"] = phase2(
                output,
                started_epoch,
                stages["phase0"],
                stages["phase1"],
            )
            status = "completed"
        except BaseException as error:
            status = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, TimeoutError))
                else "failed"
            )
            reason = repr(error)
        finally:
            try:
                assert_assets_unchanged(manifest)
            except BaseException as error:
                status = "failed"
                reason = (
                    f"{reason}; protected asset check: {error!r}"
                    if reason else f"protected asset check: {error!r}"
                )
            summary = build_summary(
                output, stages, status, reason, started_epoch,
            )
            report = make_report(
                output, summary, status, reason,
            )
            summary["report"] = str(report)
            write_json(output / SUMMARY_NAME, summary)
            write_json(
                output / STAGE_NAME,
                {
                    "status": status,
                    "reason": reason,
                    "stages": stages,
                    "summary": str(output / SUMMARY_NAME),
                    "report": str(report),
                },
            )
            state.update(
                status=status,
                stage="report",
                finished_at=stamp(),
                report=str(report),
                summary=str(output / SUMMARY_NAME),
            )
            write_json(state_path, state)
            record("workflow", status, output, {
                "reason": reason,
                "elapsed_minutes": elapsed_minutes(started_epoch),
                "report": str(report),
            })
            try:
                from scripts.research.frozen_backbone_v5_publish import publish

                publish(output, push=True)
            except BaseException as error:
                write_json(
                    output
                    / "homog_label_fix_20261124_v5_publication_failure.json",
                    {"status": "failed", "reason": repr(error)},
                )
                record("publication", "failed", output, repr(error))
            if reason is not None:
                raise RuntimeError(reason)
        return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", type=Path, default=WORKFLOW)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=WORKFLOW)
    evaluate_parser = subparsers.add_parser("_evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--destination", type=Path, required=True)
    evaluate_parser.add_argument("--indices", required=True)
    evaluate_parser.add_argument("--workers", type=int, default=3)
    snapshot_parser = subparsers.add_parser("_snapshot")
    snapshot_parser.add_argument("--checkpoint", type=Path, required=True)
    snapshot_parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "smoke":
        result = smoke(args.output_dir)
    elif args.action == "run":
        result = run(args.output_dir)
    elif args.action == "_evaluate":
        _evaluate_worker(args)
        result = None
    else:
        _snapshot_worker(args)
        result = None
    if result is not None:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
