"""Six-hour, gated homog-label-fix v4 experiment (software/MuJoCo only)."""
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

BASE = "703003bc7595318709097faa184f147d2386554f"
SOURCE_BRANCH = "feature/homog-label-fix-20261123"
BRANCH = "experiment/overnight-label-fix-v4-20261124"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = ROOT / "runs/20261123/homog_clean_git_v1/repository"
SOURCE_DATA = ROOT / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"
BASE_DATA = ROOT / "data/plan2_plus_original_train_224_20261119"
BASELINE = ROOT / "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth"
V3_WEIGHTS = ROOT / "outputs/homog_label_fix_20261123_v3"
V3_FAST5 = ROOT / "github_reports/homog_label_fix_20261123_v3_s3_fast5_1600"
WORKFLOW = ROOT / "runs/20261124/homog_label_fix_20261124_v4_6h"
OUTPUTS = ROOT / "outputs/homog_label_fix_20261124_v4"
DATASETS = {
    "executed_only": ROOT / "data/homog_label_fix_20261124_v4_executed_only",
    "dir_consistent": ROOT / "data/homog_label_fix_20261124_v4_dir_consistent",
    "t3_only": ROOT / "data/homog_label_fix_20261124_v4_t3_only",
}
ARM_NAME = {
    "executed_only": "D1",
    "dir_consistent": "D2",
    "t3_only": "D3",
}
DEV20 = tuple(range(20))
STEPS = 1600
BASELINE_FAST5_UM = 65.569288
STRICT_FINAL_UM = 65.57
NEW_TASK_DEADLINE_MIN = 335
FINAL_DEADLINE_MIN = 360
CLOSE_OUT_RESERVE_MIN = 25
PHASE1_BATCHES = 4000
PHASE2_BATCHES = 8000
FALLBACK_BATCHES = 4000
FAST5_ETA_MIN = 18
TRAIN_4000_ETA_MIN = 8
TRAIN_8000_ETA_MIN = 12
DEV20_ETA_MIN = 100
FAST5_SNAPSHOT_ETA_MIN = 6
MODULE = "scripts.research.overnight_label_fix_v4_6h"
MANIFEST_NAME = "homog_label_fix_20261124_v4_execution_manifest.json"
STATUS_NAME = "homog_label_fix_20261124_v4_status.json"
STAGE_NAME = "homog_label_fix_20261124_v4_stage_results.json"
REPORT_NAME = "homog_label_fix_20261124_v4_REPORT.md"


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
                f"\n- {stamp()}: homog_label_fix_20261124_v4 {purpose} "
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
        ROOT / "scripts/research/overnight_label_fix_v4_6h.py",
        ROOT / "scripts/research/overnight_label_fix_v4_publish.py",
        ROOT / "scripts/research/homog_label_fix_repair.py",
        ROOT / "scripts/research/homog_replay_evaluate.py",
        ROOT / "scripts/research/homog_replay_common.py",
        ROOT / "lerobot_training/dit_experience_replay.py",
        ROOT / "lerobot_training/dataset_contract.py",
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


def protected_hashes():
    paths = [
        SOURCE_DATA / "data/dataset.npz",
        V3_WEIGHTS,
        BASELINE,
        BASE_DATA / "data/dataset.npz",
    ]
    result = {}
    for path in paths:
        if path.is_dir():
            files = sorted(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix in (".pth", ".npz")
            )
            result[str(path)] = {
                str(item.relative_to(path)): file_sha256(item)
                for item in files
            }
        elif path.is_file():
            result[str(path)] = file_sha256(path)
    return result


def _git(*args):
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), *args], text=True
    ).strip()


def ensure_branch():
    if _git("cat-file", "-t", BASE) != "commit":
        raise ValueError("authorized v4 base is missing")
    exists = subprocess.run(
        ["git", "-C", str(SOURCE_REPO), "show-ref", "--verify", "--quiet",
         f"refs/heads/{BRANCH}"],
        check=False,
    ).returncode == 0
    if not exists:
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "branch", BRANCH, BASE],
            check=True,
        )
    if _git("merge-base", BASE, BRANCH) != BASE:
        raise ValueError("v4 experiment branch does not descend from 703003bc")
    return _git("rev-parse", BRANCH)


def assert_assets_unchanged(manifest):
    current = protected_hashes()
    if current != manifest["protected_hashes"]:
        raise ValueError("v3/baseline/source assets changed during the experiment")


def remaining_minutes(started_epoch, now=None):
    now = time.time() if now is None else now
    return NEW_TASK_DEADLINE_MIN - (now - started_epoch) / 60.0


def can_start(started_epoch, eta_minutes, now=None, reserve=CLOSE_OUT_RESERVE_MIN):
    return remaining_minutes(started_epoch, now) >= float(eta_minutes) + reserve


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
    """Run one command in its own process group and hard-stop it at the deadline."""
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
                    f"deadline exceeded; terminated process group {process.pid}"
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
                raise TimeoutError("parallel evaluation deadline exceeded")
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
    dataset,
    destination,
    *,
    batches,
    correction_ratio=.15,
    backbone_lr=1e-6,
    attention_lr=1e-5,
    near_target_um=None,
    near_target_weight=1.0,
    resume=False,
    smoke=False,
):
    command = env_command() + [
        "-m", "lerobot_training.dit_experience_replay", "train",
        "--data-dir", str(BASE_DATA),
        "--correction-data-dir", str(dataset),
        "--original-max-episode", "838",
        "--val-data-dir", str(VAL),
        "--init-from", str(BASELINE),
        "--output-dir", str(destination),
        "--mode", "ratio",
        "--correction-ratio", str(correction_ratio),
        "--backbone-lr", str(backbone_lr),
        "--attention-lr", str(attention_lr),
        "--total-batches", str(2 if smoke else batches),
        "--steps-per-epoch", str(2 if smoke else 1000),
        "--batch-size", "2" if smoke else "16",
        "--val-batch-size", "1" if smoke else "64",
        "--num-workers", "0" if smoke else "4",
        "--checkpoint-every-steps", str(2 if smoke else 1000),
        "--log-every-steps", str(1 if smoke else 100),
        "--lr-schedule", "cosine",
        "--near-target-weight", str(near_target_weight),
    ]
    if smoke:
        command += ["--validation-max-batches", "1"]
    if near_target_um is not None:
        command += ["--near-target-um", str(near_target_um)]
    if resume:
        command += ["--resume"]
    return command


def train_stage(
    destination,
    stage_log,
    deadline_epoch,
    **kwargs,
):
    destination = Path(destination)
    receipt = destination / "training_receipt.json"
    if receipt.is_file() and read(receipt).get("status") == "completed":
        return read(receipt)
    resume = (destination / "resume_state.pth").is_file()
    command = train_command(destination=destination, resume=resume, **kwargs)
    run_managed(command, stage_log, deadline_epoch)
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


def snapshot_stage(checkpoint, destination, deadline_epoch, log_path, name="candidate"):
    destination = Path(destination)
    summary_path = destination / "snapshot_task2_vs_cleanoct.json"
    if summary_path.is_file():
        return read(summary_path)[name]
    run_managed(
        snapshot_command(checkpoint, destination),
        log_path,
        deadline_epoch,
    )
    return read(summary_path)[name]


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


def phase1_candidate(
    arm,
    checkpoint,
    fast,
    snapshot,
    *,
    strict_final_um=STRICT_FINAL_UM,
):
    reverse = snapshot["axis_reverse_snapshot_counts"]
    absolute_drift = fast["absolute_drift_1000_to_1600_um"]
    return {
        "arm": arm,
        "checkpoint": str(checkpoint),
        "final_mean_um": float(fast["final_mean_um"]),
        "min_mean_um": float(fast["min_mean_um"]),
        "absolute_drift_um": float(absolute_drift),
        "toward_fraction": float(snapshot["first_order_toward_fraction"]),
        "axis_reverse_counts": {key: int(value) for key, value in reverse.items()},
        "max_axis_reverse": int(max(reverse.values())),
        "strict": bool(
            float(snapshot["first_order_toward_fraction"]) >= .70
            and max(reverse.values()) <= 2
            and float(fast["final_mean_um"]) < strict_final_um
        ),
    }


def select_phase1(candidates):
    strict = [value for value in candidates if value["strict"]]
    if strict:
        selected = min(
            strict,
            key=lambda value: (
                value["final_mean_um"],
                value["absolute_drift_um"],
                -value["toward_fraction"],
            ),
        )
        return selected, "strict"
    eligible = [value for value in candidates if value["final_mean_um"] < 80]
    if eligible:
        return max(eligible, key=lambda value: value["toward_fraction"]), "fallback_phase1"
    return None, "none"


def phase2_candidate(arm, checkpoint, fast, snapshot):
    value = phase1_candidate(arm, checkpoint, fast, snapshot)
    value["passed"] = bool(
        value["final_mean_um"] < 50
        and value["min_mean_um"] < 30
        and value["absolute_drift_um"] < 15
        and value["toward_fraction"] >= .75
    )
    return value


def select_phase2(candidates):
    passed = [value for value in candidates if value.get("passed")]
    if not passed:
        return None
    return min(
        passed,
        key=lambda value: (
            value["final_mean_um"],
            value["absolute_drift_um"],
            -value["toward_fraction"],
        ),
    )


def smoke(output, rebuild=True):
    output = Path(output).resolve()
    smoke_root = output / "smoke"
    if smoke_root.exists() and (smoke_root / "homog_label_fix_20261124_v4_smoke_receipt.json").is_file():
        receipt = read(smoke_root / "homog_label_fix_20261124_v4_smoke_receipt.json")
        if receipt.get("identity") == source_identity() and receipt.get("passed"):
            return receipt
        raise FileExistsError("existing smoke directory has different provenance")
    smoke_root.mkdir(parents=True, exist_ok=False)
    from scripts.research.homog_label_fix_repair import build_fixed_dataset

    audit_results = {}
    datasets = {}
    for mode in DATASETS:
        dataset = smoke_root / f"homog_label_fix_20261124_v4_smoke_{mode}"
        audit_dir = smoke_root / f"homog_label_fix_20261124_v4_smoke_{mode}_audit"
        audit, receipt = build_fixed_dataset(audit_dir, dataset, mode)
        audit_results[mode] = audit
        datasets[mode] = str(dataset)
    destination = smoke_root / "homog_label_fix_20261124_v4_smoke_train"
    command = train_command(
        Path(datasets["dir_consistent"]),
        destination,
        batches=2,
        smoke=True,
        correction_ratio=.15,
        backbone_lr=0.,
        attention_lr=3e-5,
        near_target_um=80.,
        near_target_weight=2.,
    )
    started = time.monotonic()
    run_managed(
        command,
        smoke_root / "homog_label_fix_20261124_v4_smoke.log",
        time.time() + 30 * 60,
    )
    receipt_path = smoke_root / "homog_label_fix_20261124_v4_smoke_receipt.json"
    receipt = {
        "status": "completed",
        "passed": True,
        "identity": source_identity(),
        "timestamp": stamp(),
        "elapsed_s": time.monotonic() - started,
        "command": command,
        "datasets": datasets,
        "audits": {
            mode: {
                "status": audit["status"],
                "t1_t2_fallback_fraction": audit["t1_t2_fallback_fraction"],
                "t3_stationary_kept_fraction": audit["t3_stationary_kept_fraction"],
                "correction_sample_weight": audit["correction_sample_weight"],
            }
            for mode, audit in audit_results.items()
        },
        "checkpoint": str(destination / "step_000002.pth"),
        "checkpoint_sha256": file_sha256(destination / "step_000002.pth"),
        "training_receipt": read(destination / "training_receipt.json"),
        "six_hour_timer_started": False,
        "simulation_scope": "Software/MuJoCo only",
    }
    write_json(receipt_path, receipt)
    record("2_batch_smoke", "passed", smoke_root, {
        "identity": signature(source_identity()),
        "datasets": datasets,
        "checkpoint": receipt["checkpoint"],
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
            raise ValueError("protected v3/baseline/source assets changed")
        return manifest
    final_paths = [*DATASETS.values(), OUTPUTS]
    if any(path.exists() for path in final_paths):
        raise FileExistsError(
            "v4 data/output destinations are occupied without an execution manifest"
        )
    smoke_receipt = output / "smoke" / "homog_label_fix_20261124_v4_smoke_receipt.json"
    if require_smoke:
        if not smoke_receipt.is_file():
            raise FileNotFoundError(
                f"successful 2-batch smoke receipt is required: {smoke_receipt}"
            )
        smoke_value = read(smoke_receipt)
        if not smoke_value.get("passed") or smoke_value.get("identity") != source_identity():
            raise ValueError("smoke receipt is failed or has different provenance")
    ensure_branch()
    from scripts.research.a_axis_flip_common import gpu_check

    manifest = {
        "status": "preflight_passed",
        "timestamp": stamp(),
        "identity": source_identity(),
        "protected_hashes": protected_hashes(),
        "smoke_receipt": str(smoke_receipt) if smoke_receipt.is_file() else None,
        "branch_parent": ensure_branch(),
        "gpu": gpu_check(),
        "datasets": {mode: str(path) for mode, path in DATASETS.items()},
        "outputs": str(OUTPUTS),
        "gates": {
            "phase1_early_toward_min": .55,
            "phase1_early_final_max_um": 110.,
            "phase1_strict_toward_min": .70,
            "phase1_strict_axis_reverse_max": 2,
            "phase1_strict_final_um": STRICT_FINAL_UM,
            "phase2_final_max_um": 50.,
            "phase2_min_max_um": 30.,
            "phase2_absolute_drift_max_um": 15.,
            "phase2_toward_min": .75,
            "dev20_min_remaining_minutes": 100,
        },
        "budget": {
            "new_task_stop_minute": NEW_TASK_DEADLINE_MIN,
            "final_minute": FINAL_DEADLINE_MIN,
            "close_out_reserve_minutes": CLOSE_OUT_RESERVE_MIN,
        },
        "simulation_scope": "Software/MuJoCo only; no clinical or hardware claim",
    }
    write_json(manifest_path, manifest)
    return manifest


def build_phase0(output, manifest):
    from scripts.research.homog_label_fix_repair import build_fixed_dataset

    results = {}
    for mode, dataset in DATASETS.items():
        audit_dir = output / f"homog_label_fix_20261124_v4_phase0_{mode}"
        if dataset.exists():
            if not (dataset / "dataset_receipt.json").is_file():
                raise FileExistsError(f"unregistered dataset destination: {dataset}")
            receipt = read(dataset / "dataset_receipt.json")
            audit = read(dataset / "dataset_audit.json")
            if receipt.get("mode") != mode or audit.get("mode") != mode:
                raise ValueError(f"resume mode mismatch for {dataset}")
        else:
            audit, receipt = build_fixed_dataset(
                audit_dir, dataset, mode,
            )
        results[mode] = {
            "status": audit["status"],
            "dataset": str(dataset),
            "dataset_sha256": receipt["fixed_dataset_sha256"],
            "audit": str(dataset / "dataset_audit.json"),
            "mode": mode,
            "t1_t2_fallback_fraction": audit["t1_t2_fallback_fraction"],
            "t3_stationary_kept_fraction": audit["t3_stationary_kept_fraction"],
            "correction_sample_weight": audit["correction_sample_weight"],
            "valid_windows": audit["valid_windows"],
        }
        if audit["status"] != "passed" or not all(
            gate["passed"] for gate in audit["gates"].values()
        ):
            raise ValueError(f"Phase 0 audit failed for {mode}")
    write_json(
        output / "homog_label_fix_20261124_v4_phase0_gates.json",
        results,
    )
    record("phase0", "passed", output, {
        mode: value["dataset_sha256"] for mode, value in results.items()
    })
    return results


def phase1(output, started_epoch):
    results = {"arms": {}, "candidates": []}
    for mode in ("executed_only", "dir_consistent", "t3_only"):
        arm = ARM_NAME[mode]
        if not can_start(started_epoch, TRAIN_4000_ETA_MIN):
            results["arms"][arm] = {
                "status": "skipped_time",
                "reason": "conservative 4k ETA plus 25m closeout reserve did not fit",
            }
            continue
        destination = OUTPUTS / f"homog_label_fix_20261124_v4_phase1_{arm}"
        try:
            train_stage(
                destination,
                output / f"homog_label_fix_20261124_v4_phase1_{arm}_train.log",
                time.time() + max(0., remaining_minutes(started_epoch)) * 60,
                dataset=DATASETS[mode],
                batches=PHASE1_BATCHES,
            )
        except Exception as error:
            results["arms"][arm] = {"status": "failed", "reason": repr(error)}
            continue
        arm_result = {"status": "training_completed", "checkpoints": {}}
        for step in (2000, 4000):
            checkpoint = destination / f"step_{step:06d}.pth"
            if not checkpoint.is_file():
                arm_result["checkpoints"][str(step)] = {
                    "status": "missing",
                }
                continue
            eval_dest = (
                output / "reports/homog_label_fix_20261124_v4_phase1"
                / f"{arm}_step{step}_fast5"
            )
            snapshot_dest = (
                output / "reports/homog_label_fix_20261124_v4_phase1"
                / f"{arm}_step{step}_snapshot9"
            )
            deadline = time.time() + max(0., remaining_minutes(started_epoch)) * 60
            try:
                fast = evaluate_stage(
                    checkpoint, eval_dest, FAST5, 5, deadline,
                    output / f"homog_label_fix_20261124_v4_{arm}_{step}_fast5.log",
                )
                snapshot = snapshot_stage(
                    checkpoint, snapshot_dest, deadline,
                    output / f"homog_label_fix_20261124_v4_{arm}_{step}_snapshot.log",
                )
            except Exception as error:
                arm_result["checkpoints"][str(step)] = {
                    "status": "failed", "reason": repr(error),
                }
                break
            candidate = phase1_candidate(
                arm, checkpoint, fast, snapshot,
            )
            candidate["step"] = step
            candidate["mode"] = mode
            candidate["fast5_destination"] = str(eval_dest)
            candidate["snapshot_destination"] = str(snapshot_dest)
            arm_result["checkpoints"][str(step)] = candidate
            results["candidates"].append(candidate)
            if step == 2000 and (
                candidate["toward_fraction"] < .55
                or candidate["final_mean_um"] > 110.
            ):
                arm_result["early_stop"] = True
                arm_result["reason"] = (
                    "step2000 failed toward>=0.55 and final_mean<=110um"
                )
                break
        results["arms"][arm] = arm_result
        write_json(
            output / "homog_label_fix_20261124_v4_phase1_gates.json",
            results,
        )
    selected, selection = select_phase1(results["candidates"])
    results["selected"], results["selection"] = selected, selection
    if selected is None:
        results["D1_used_for_fallback"] = True
    write_json(
        output / "homog_label_fix_20261124_v4_phase1_gates.json",
        results,
    )
    record("phase1", "completed", output, {
        "selection": selection,
        "selected": selected,
        "candidate_count": len(results["candidates"]),
    })
    return results


def _train_and_evaluate_phase2(
    output,
    started_epoch,
    arm,
    mode,
    near_target_um,
):
    destination = OUTPUTS / f"homog_label_fix_20261124_v4_phase2_{arm}"
    try:
        train_stage(
            destination,
            output / f"homog_label_fix_20261124_v4_phase2_{arm}_train.log",
            time.time() + max(0., remaining_minutes(started_epoch)) * 60,
            dataset=DATASETS[mode],
            batches=PHASE2_BATCHES,
            near_target_um=near_target_um,
            near_target_weight=2. if near_target_um is not None else 1.,
        )
    except Exception as error:
        return {"status": "failed", "reason": repr(error), "candidates": []}
    candidates = []
    for step in (4000, 6000, 8000):
        checkpoint = destination / f"step_{step:06d}.pth"
        if not checkpoint.is_file():
            continue
        eval_dest = (
            output / "reports/homog_label_fix_20261124_v4_phase2"
            / f"{arm}_step{step}_fast5"
        )
        snapshot_dest = (
            output / "reports/homog_label_fix_20261124_v4_phase2"
            / f"{arm}_step{step}_snapshot9"
        )
        deadline = time.time() + max(0., remaining_minutes(started_epoch)) * 60
        try:
            fast = evaluate_stage(
                checkpoint, eval_dest, FAST5, 5, deadline,
                output / f"homog_label_fix_20261124_v4_{arm}_{step}_fast5.log",
            )
            snapshot = snapshot_stage(
                checkpoint, snapshot_dest, deadline,
                output / f"homog_label_fix_20261124_v4_{arm}_{step}_snapshot.log",
            )
        except Exception as error:
            candidates.append({
                "arm": arm, "step": step, "status": "failed",
                "reason": repr(error), "passed": False,
            })
            continue
        candidate = phase2_candidate(
            arm, checkpoint, fast, snapshot,
        )
        candidate.update({
            "step": step,
            "mode": mode,
            "near_target_um": near_target_um,
            "fast5_destination": str(eval_dest),
            "snapshot_destination": str(snapshot_dest),
        })
        candidates.append(candidate)
    return {"status": "completed", "candidates": candidates}


def phase2(output, started_epoch, phase1_result):
    selected = phase1_result.get("selected")
    if selected is None or phase1_result.get("selection") == "none":
        result = {
            "status": "skipped_no_phase1_candidate",
            "gate_passed": False,
        }
        write_json(
            output / "homog_label_fix_20261124_v4_phase2_gates.json",
            result,
        )
        return result
    result = {"arms": {}, "candidates": []}
    mode = selected["mode"]
    for arm, near_target_um in (("W1", None), ("W2", 80.)):
        if not can_start(started_epoch, TRAIN_8000_ETA_MIN):
            result["arms"][arm] = {
                "status": "skipped_time",
                "reason": "conservative 8k ETA plus closeout reserve did not fit",
            }
            continue
        value = _train_and_evaluate_phase2(
            output, started_epoch, arm, mode, near_target_um,
        )
        result["arms"][arm] = value
        result["candidates"].extend(value["candidates"])
        write_json(
            output / "homog_label_fix_20261124_v4_phase2_gates.json",
            result,
        )
    best = select_phase2(result["candidates"])
    result["selected"] = best
    result["gate_passed"] = best is not None
    result["status"] = "completed"
    write_json(
        output / "homog_label_fix_20261124_v4_phase2_gates.json",
        result,
    )
    record("phase2", "passed" if best is not None else "failed", output, {
        "selected": best,
        "candidate_count": len(result["candidates"]),
    })
    return result


def fallback(output, started_epoch, phase1_result):
    selected = phase1_result.get("selected")
    mode = selected["mode"] if selected is not None else "executed_only"
    destination = OUTPUTS / "homog_label_fix_20261124_v4_fallback_F"
    if not can_start(started_epoch, TRAIN_4000_ETA_MIN):
        result = {
            "status": "skipped_time",
            "mode": mode,
            "reason": "4k fallback ETA plus 25m closeout reserve did not fit",
        }
        write_json(
            output / "homog_label_fix_20261124_v4_fallback_gates.json",
            result,
        )
        return result
    deadline = time.time() + max(0., remaining_minutes(started_epoch)) * 60
    train_stage(
        destination,
        output / "homog_label_fix_20261124_v4_fallback_F_train.log",
        deadline,
        dataset=DATASETS[mode],
        batches=FALLBACK_BATCHES,
        backbone_lr=0.,
        attention_lr=3e-5,
        correction_ratio=.15,
    )
    checkpoint = destination / "step_004000.pth"
    fast_dest = output / "reports/homog_label_fix_20261124_v4_fallback_F_fast5"
    snapshot_dest = output / "reports/homog_label_fix_20261124_v4_fallback_F_snapshot9"
    fast = evaluate_stage(
        checkpoint, fast_dest, FAST5, 5, deadline,
        output / "homog_label_fix_20261124_v4_fallback_F_fast5.log",
    )
    snapshot = snapshot_stage(
        checkpoint, snapshot_dest, deadline,
        output / "homog_label_fix_20261124_v4_fallback_F_snapshot.log",
    )
    result = {
        "status": "completed",
        "mode": mode,
        "checkpoint": str(checkpoint),
        "fast5": {
            key: fast[key]
            for key in (
                "final_mean_um", "min_mean_um",
                "strict_success_count", "completed_targets",
                "absolute_drift_1000_to_1600_um",
            )
        },
        "snapshot": snapshot,
    }
    write_json(
        output / "homog_label_fix_20261124_v4_fallback_gates.json",
        result,
    )
    record("fallback_F", "completed", output, result["fast5"])
    return result


def phase3_dev20(output, started_epoch, phase2_result):
    selected = phase2_result.get("selected")
    if not phase2_result.get("gate_passed") or selected is None:
        result = {
            "status": "skipped_gate",
            "reason": "Phase 2 checkpoint gate did not pass",
        }
        write_json(
            output / "homog_label_fix_20261124_v4_phase3_dev20.json",
            result,
        )
        return result
    if remaining_minutes(started_epoch) < DEV20_ETA_MIN:
        result = {
            "status": "skipped_time",
            "reason": "fewer than 100 minutes remain before T+335m",
        }
        write_json(
            output / "homog_label_fix_20261124_v4_phase3_dev20.json",
            result,
        )
        return result
    deadline = time.time() + max(
        0., (NEW_TASK_DEADLINE_MIN - (time.time() - started_epoch) / 60.)
    ) * 60
    top_dest = output / "reports/homog_label_fix_20261124_v4_phase3_top1_dev20"
    baseline_dest = output / "reports/homog_label_fix_20261124_v4_phase3_baseline_dev20"
    run_managed_many(
        [
            (
                evaluation_command(selected["checkpoint"], top_dest, DEV20, 3),
                output / "homog_label_fix_20261124_v4_phase3_top1_dev20.log",
            ),
            (
                evaluation_command(BASELINE, baseline_dest, DEV20, 3),
                output / "homog_label_fix_20261124_v4_phase3_baseline_dev20.log",
            ),
        ],
        deadline,
    )
    top = read(top_dest / "summary.json")
    baseline = read(baseline_dest / "summary.json")
    result = {
        "status": "completed",
        "selected": selected,
        "top1": top,
        "baseline": baseline,
        "delta_final_mean_um": top["final_mean_um"] - baseline["final_mean_um"],
        "evaluator_identity_match": (
            top["protocol"]["identity"] == baseline["protocol"]["identity"]
        ),
    }
    write_json(
        output / "homog_label_fix_20261124_v4_phase3_dev20.json",
        result,
    )
    record("phase3_dev20", "completed", output, {
        "top1_final_mean_um": top["final_mean_um"],
        "baseline_final_mean_um": baseline["final_mean_um"],
        "delta_final_mean_um": result["delta_final_mean_um"],
    })
    return result


def baseline_fast5_reference(output):
    summary = read(V3_FAST5 / "summary.json")
    baseline = summary["baseline"]
    if abs(float(baseline["final_mean_um"]) - BASELINE_FAST5_UM) > 1e-3:
        raise ValueError("validated v3 baseline fast5 metric changed")
    value = {
        "source": str(V3_FAST5 / "summary.json"),
        "final_mean_um": float(baseline["final_mean_um"]),
        "min_mean_um": float(baseline["min_mean_um"]),
        "absolute_drift_1000_to_1600_um": float(
            baseline["absolute_drift_1000_to_1600_um"]
        ),
        "strict_success_count": int(baseline["strict_success_count"]),
        "completed_targets": int(baseline["completed_targets"]),
        "reused_not_rerun": True,
    }
    write_json(
        output / "homog_label_fix_20261124_v4_baseline_fast5_reference.json",
        value,
    )
    return value


def best_final_result(stages):
    values = []
    for phase in ("phase1", "phase2", "fallback"):
        value = stages.get(phase)
        if not value:
            continue
        if phase == "phase1":
            values.extend(value.get("candidates", []))
        elif phase == "phase2":
            values.extend(value.get("candidates", []))
        elif value.get("fast5"):
            values.append(value["fast5"])
    if not values:
        return None
    return min(values, key=lambda item: item.get("final_mean_um", float("inf")))


def make_report(output, stages, baseline, status, reason=None):
    best = best_final_result(stages)
    lines = [
        "# Homogeneous label-fix v4 six-hour gated experiment",
        "",
        f"Status: `{status}`. Base: `{BASE}`. Branch: `{BRANCH}`.",
        "",
        "Scope: software and MuJoCo simulation only; this is not clinical or hardware evidence.",
        "",
        "## Baseline",
        "",
        f"Reused validated task2 fast5 final mean: {baseline['final_mean_um']:.6f} um "
        f"({baseline['strict_success_count']}/{baseline['completed_targets']} strict).",
        "",
        "## Gates",
        "",
    ]
    if reason:
        lines += [f"Failure/reason: `{reason}`", ""]
    for phase in ("phase0", "phase1", "phase2", "phase3", "fallback"):
        value = stages.get(phase)
        if value is None:
            lines += [f"- `{phase}`: not started.", ""]
        else:
            brief = (
                value.get("status")
                or value.get("selection")
                or ("completed" if phase == "phase0" else "recorded")
            )
            lines += [f"- `{phase}`: `{brief}`", ""]
    if best is not None:
        lines += [
            "## Best fast5 result",
            "",
            f"- Checkpoint: `{best.get('checkpoint')}`",
            f"- Final mean: {best.get('final_mean_um', math.nan):.6f} um",
            f"- Minimum/global mean: {best.get('min_mean_um', math.nan):.6f} um",
            f"- Strict success: {best.get('strict_success_count', 0)}/"
            f"{best.get('completed_targets', 0)}",
            "",
        ]
    dev20 = stages.get("phase3")
    if dev20:
        lines += [
            "## dev20",
            "",
            f"- Executed: `{dev20.get('status') == 'completed'}`",
            f"- Reason: `{dev20.get('reason', 'Phase 2 gate passed')}`",
        ]
        if dev20.get("status") == "completed":
            lines += [
                f"- Top-1 final mean: {dev20['top1']['final_mean_um']:.6f} um",
                f"- Baseline final mean: {dev20['baseline']['final_mean_um']:.6f} um",
                f"- Delta: {dev20['delta_final_mean_um']:+.6f} um",
                f"- Top-1 strict success: {dev20['top1']['strict_success_count']}/20",
            ]
        lines.append("")
    lines += [
        "## Recommendation",
        "",
        "Continue only with candidates that show direction consistency and reduced drift on the "
        "same evaluator. Retain the frozen v3/baseline assets and treat any non-improvement as a "
        "negative label-repair result rather than retuning the gate.",
        "",
    ]
    path = output / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "homog_label_fix_20261124_v4_workflow.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("v4 workflow is already running") from error
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
                    started_epoch + NEW_TASK_DEADLINE_MIN * 60,
                    timezone.utc,
                ).astimezone().isoformat(timespec="seconds"),
                "final_stop_at": datetime.fromtimestamp(
                    started_epoch + FINAL_DEADLINE_MIN * 60,
                    timezone.utc,
                ).astimezone().isoformat(timespec="seconds"),
            }
            write_json(state_path, state)
            write_json(output / "homog_label_fix_20261124_v4_timer_started.json", state)
        baseline = baseline_fast5_reference(output)
        stage_results = {}
        status = "failed"
        reason = None
        try:
            state.update(status="running", stage="phase0")
            write_json(state_path, state)
            stage_results["phase0"] = build_phase0(output, manifest)
            state.update(stage="phase1")
            write_json(state_path, state)
            stage_results["phase1"] = phase1(output, started_epoch)
            state.update(stage="phase2")
            write_json(state_path, state)
            stage_results["phase2"] = phase2(
                output, started_epoch, stage_results["phase1"],
            )
            state.update(stage="phase3")
            write_json(state_path, state)
            stage_results["phase3"] = phase3_dev20(
                output, started_epoch, stage_results["phase2"],
            )
            if not stage_results["phase2"].get("gate_passed"):
                state.update(stage="fallback")
                write_json(state_path, state)
                stage_results["fallback"] = fallback(
                    output, started_epoch, stage_results["phase1"],
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
            assert_assets_unchanged(manifest)
            report = make_report(
                output, stage_results, baseline, status, reason,
            )
            write_json(
                output / STAGE_NAME,
                {
                    "status": status,
                    "reason": reason,
                    "started_epoch": started_epoch,
                    "elapsed_minutes": (time.time() - started_epoch) / 60.,
                    "stages": stage_results,
                    "report": str(report),
                    "simulation_scope": "Software/MuJoCo only",
                },
            )
            state.update(
                status=status,
                stage="report",
                finished_at=stamp(),
                report=str(report),
            )
            write_json(state_path, state)
            record("workflow", status, output, {
                "reason": reason,
                "elapsed_minutes": (time.time() - started_epoch) / 60.,
                "report": str(report),
            })
            try:
                from scripts.research.overnight_label_fix_v4_publish import publish

                publish(output, push=True)
            except BaseException as error:
                write_json(
                    output / "homog_label_fix_20261124_v4_publication_failure.json",
                    {"status": "failed", "reason": repr(error)},
                )
                record("publication", "failed", output, repr(error))
                if status == "completed":
                    status = "publication_failed"
            if reason is not None:
                raise RuntimeError(reason)
        return stage_results


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
    evaluate_parser.add_argument("--workers", type=int, default=5)
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
