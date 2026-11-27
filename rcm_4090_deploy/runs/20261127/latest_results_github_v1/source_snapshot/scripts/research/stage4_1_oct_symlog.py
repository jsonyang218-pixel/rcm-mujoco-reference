"""Stage 4.1 single-variable OCT-symlog training and verification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/plan2_plus_original_train_224_20261119"
VAL = ROOT / "data/lerobot_rcm_val_acthist_20261117_224"
OVERLAY = ROOT / "data/overnight_c_fix_symlog_081234_overlay/label_override.npz"
INIT = ROOT / "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth"
BASELINE = ROOT / "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth"
BASELINE_FAST5 = (
    ROOT
    / "github_reports/homog_label_fix_20261123_v3_s3_fast5_1600"
    / "baseline_task2_step4000/shards/tasks"
)
BASELINE_DEV20 = (
    ROOT
    / "runs/20261124/homog_label_fix_20261124_v5_3h/reports"
    / "homog_label_fix_20261124_v5_phase0_baseline_dev20"
)
STEP40 = ROOT / "runs/20261125/step4_0_near_target_blindness_probe"
OUTPUT = ROOT / "runs/20261125/stage4_1_oct_symlog"
CHECKPOINT_DIR = OUTPUT / "checkpoints/stage4_1_oct_symlog"
SOURCE_REPO = ROOT / "runs/20261123/homog_clean_git_v1/repository"
BRANCH = "experiment/stage4-oct-symlog-20261125"
BASE_BRANCH_COMMIT = "630c03b6aac27e92a1f3c5dbefc577eb518cf657"
FAST5 = (0, 2, 6, 8, 10)
DEV20 = tuple(range(20))
HARD_BUDGET_MIN = 150.
CLOSE_OUT_MIN = 20.
TRAIN_ETA_MIN = 8.
FAST5_ETA_MIN = 25.
K4_ETA_MIN = 20.
DEV20_ETA_MIN = 55.
DEV20_GATE_REMAINING_MIN = 40.
MODULE = "scripts.research.stage4_1_oct_symlog"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def elapsed_minutes(started):
    return (time.time() - float(started)) / 60.


def remaining_minutes(started):
    return HARD_BUDGET_MIN - elapsed_minutes(started)


def can_start(started, eta_min):
    return remaining_minutes(started) >= float(eta_min) + CLOSE_OUT_MIN


def record(purpose, status, details):
    with (ROOT / "TEST_LOG.md").open("ab") as stream:
        stream.write(
            (
                f"\n- {stamp()}: stage4_1_oct_symlog {purpose} **{status}**; "
                f"command `{' '.join(sys.argv)}`; {details}; output "
                f"`{OUTPUT}`. Software/MuJoCo simulation only.\n"
            ).encode()
        )


def env_command():
    return [
        "env", "-u", "PYTHONHOME", "-u", "PYTHONPATH",
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
        "-E", "-B", "-u",
    ]


def terminate_group(process):
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


def run_command(command, log_path, deadline_epoch):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.time() >= deadline_epoch:
                terminate_group(process)
                raise TimeoutError("Stage 4.1 hard deadline terminated process group")
            time.sleep(.25)
    if process.returncode:
        raise RuntimeError(f"command failed: {command}; log={log_path}")


def train_command(destination, *, smoke=False):
    command = env_command() + [
        "-m", "lerobot_training.dit_experience_replay", "train",
        "--data-dir", str(DATA),
        "--val-data-dir", str(VAL),
        "--init-from", str(INIT),
        "--output-dir", str(destination),
        "--mode", "ratio",
        "--correction-ratio", "0.08",
        "--epochs", "4",
        "--steps-per-epoch", "1000",
        "--batch-size", "2" if smoke else "16",
        "--val-batch-size", "2" if smoke else "64",
        "--num-workers", "0" if smoke else "8",
        "--checkpoint-every-steps", "2" if smoke else "1000",
        "--log-every-steps", "1" if smoke else "100",
        "--backbone-lr", "1e-6",
        "--attention-lr", "1e-5",
        "--seed", "20261118",
        "--validation-seed", "20261119",
        "--label-overlay", str(OVERLAY),
        "--oct-symlog",
    ]
    if smoke:
        command += ["--total-batches", "2", "--validation-max-batches", "1"]
    return command


def train(destination, log_path, deadline_epoch):
    destination = Path(destination)
    receipt = destination / "training_receipt.json"
    if receipt.is_file():
        return read(receipt)
    command = train_command(destination)
    if (destination / "resume_state.pth").is_file():
        command.append("--resume")
    run_command(command, log_path, deadline_epoch)
    if not receipt.is_file():
        raise RuntimeError("training completed without receipt")
    return read(receipt)


def probe(checkpoint, destination, log_path, deadline_epoch):
    destination = Path(destination)
    if (destination / "metrics.json").is_file():
        return read(destination / "metrics.json")
    command = env_command() + [
        "-m", "scripts.research.probe_near_target_blindness",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(destination),
    ]
    run_command(command, log_path, deadline_epoch)
    return read(destination / "metrics.json")


def evaluation_command(checkpoint, destination, indices, workers, aggregation):
    return env_command() + [
        "-m", MODULE, "_evaluate",
        "--checkpoint", str(checkpoint),
        "--destination", str(destination),
        "--indices", ",".join(str(value) for value in indices),
        "--workers", str(workers),
        "--action-aggregation", aggregation,
    ]


def evaluate(checkpoint, destination, indices, workers, aggregation, log_path, deadline):
    destination = Path(destination)
    summary = destination / "summary.json"
    if summary.is_file():
        return read(summary)
    run_command(
        evaluation_command(
            checkpoint, destination, indices, workers, aggregation
        ),
        log_path,
        deadline,
    )
    return read(summary)


def _evaluate_worker(args):
    from scripts.research.homog_replay_evaluate import evaluate

    indices = tuple(int(value) for value in args.indices.split(",") if value)
    evaluate(
        args.checkpoint,
        1,
        "feedback",
        indices,
        args.destination,
        {"identity": __import__(
            "scripts.research.homog_replay_common",
            fromlist=["identity"],
        ).identity()},
        workers=args.workers,
        max_workers=6,
        action_aggregation=args.action_aggregation,
    )


def target_metrics(result_path):
    document = read(result_path)
    episode = document["episodes"][0]
    trajectory = episode["trajectory"]
    errors = np.asarray(trajectory["errors"], dtype=np.float64) * 1000.
    actions = np.asarray(trajectory["actions"], dtype=np.float64)
    at20 = errors <= 20.
    indices = np.flatnonzero(at20)
    max_cycles = 0
    hold_max = None
    hold_start = None
    hold_end = None
    if len(indices):
        cuts = np.flatnonzero(np.diff(indices) > 1)
        starts = np.r_[indices[0], indices[cuts + 1]]
        ends = np.r_[indices[cuts], indices[-1]]
        lengths = ends - starts + 1
        best = int(np.argmax(lengths))
        max_cycles = int(lengths[best])
        hold_start = int(starts[best])
        hold_end = int(ends[best])
        hold_max = float(errors[hold_start:hold_end + 1].max())
    late = actions[1000:1600]
    flips = {}
    for index, axis in enumerate("ABC"):
        left, right = late[:-1, index], late[1:, index]
        active = (np.abs(left) > 1e-12) & (np.abs(right) > 1e-12)
        flips[axis] = float(np.mean(
            np.sign(left[active]) != np.sign(right[active])
        )) if active.any() else 0.
    final = float(episode["final_error_mm"] * 1000.)
    return {
        "episode_index": int(episode["episode_index"]),
        "min_error_um": float(errors.min()),
        "final_error_um": final,
        "entered_20_um": bool(at20.any()),
        "max_consecutive_le_20_cycles": max_cycles,
        "max_consecutive_le_20_seconds": max_cycles * 0.05,
        "hold_segment_max_error_um": hold_max,
        "hold_start_step": hold_start,
        "hold_end_step": hold_end,
        "late_ab_sign_flip_rate": flips,
        "absolute_drift_1000_to_1600_um": float(abs(final - errors[1000])),
        "strict_hold_200_cycles": bool(max_cycles >= 200),
        "raw_result": str(result_path),
    }


def target_metrics_from_tree(root, targets):
    return {
        f"E{index:02d}": target_metrics(
            Path(root) / f"eval_E{index:02d}/raw/evaluation_results.json"
        )
        for index in targets
    }


def aggregate_targets(rows):
    values = list(rows.values())
    return {
        "completed_targets": len(values),
        "final_mean_um": float(np.mean([v["final_error_um"] for v in values])),
        "min_mean_um": float(np.mean([v["min_error_um"] for v in values])),
        "entered_20_um_count": sum(v["entered_20_um"] for v in values),
        "strict_hold_200_count": sum(
            v["strict_hold_200_cycles"] for v in values
        ),
        "late_ab_sign_flip_rate_mean": {
            axis: float(np.mean([
                v["late_ab_sign_flip_rate"][axis] for v in values
            ]))
            for axis in "ABC"
        },
    }


def verify_smoke():
    smoke = OUTPUT / "smoke/train"
    config = read(smoke / "training_config.json")
    receipt = read(smoke / "training_receipt.json")
    metrics_path = smoke / "training_metrics.jsonl"
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    import torch
    from lerobot_training.dit_state_attention import (
        StateConditionedDiTRectifiedFlowPolicy,
    )

    model = StateConditionedDiTRectifiedFlowPolicy(
        n_obs_steps=2,
        n_action_steps=16,
        state_mean=config["state_mean"],
        state_std=config["state_std"],
        oct_mean=config["oct_mean"],
        oct_std=config["oct_std"],
        action_mean=config["action_mean"],
        action_std=config["action_std"],
        velocity_mean=config["velocity_mean"],
        velocity_std=config["velocity_std"],
        previous_action_mean=config["previous_action_mean"],
        previous_action_std=config["previous_action_std"],
        oct_symlog_enabled=True,
    )
    vector = torch.tensor([[.02, .02, .02, 1.]])
    model.train()
    train_norm, _ = model.normalize_oct_symlog(vector)
    model.eval()
    eval_norm, _ = model.normalize_oct_symlog(vector)
    torch.testing.assert_close(train_norm, eval_norm, rtol=0, atol=0)
    if (
        not bool(config["oct_symlog"])
        or not bool(config["oct_symlog_enabled"])
        or not np.all((train_norm.numpy() >= .30))
        or not np.all((train_norm.numpy() <= .45))
        or receipt["status"] != "completed"
        or not np.isfinite(receipt["sampling_counts"]["original"])
        or not np.isfinite(receipt["sampling_counts"]["correction"])
    ):
        raise ValueError("OCT symlog smoke contract failed")
    return {
        "status": "passed",
        "checkpoint_config_oct_symlog": config["oct_symlog"],
        "checkpoint_config_oct_symlog_enabled": config[
            "oct_symlog_enabled"
        ],
        "20um_normalized_delta": train_norm[0].tolist(),
        "train_eval_exact_equal": True,
        "metric_rows": len(metric_rows),
        "metrics_finite": all(
            np.isfinite(row["gradient_norm_before_clip"])
            for row in metric_rows
        ),
        "smoke_checkpoint": str(smoke / "step_000002.pth"),
    }


def gate(new_rows, aggregate):
    return bool(
        (
            aggregate["final_mean_um"] < 58.
            and aggregate["min_mean_um"] < 25.
        )
        or aggregate["entered_20_um_count"] >= 2
    )


def write_report(metrics):
    lines = [
        "# Stage 4.1 controlled OCT-symlog experiment",
        "",
        "Status: `in_progress`",
        "",
        "Single changed training variable: `oct_symlog=False -> True`. "
        "Dataset, initialization, seed, architecture, replay ratio, optimizer, "
        "driver, control period, and 4,000-step budget were preserved.",
        "",
        "## Train/eval parity",
        "",
        f"- Status: `{metrics['smoke']['status']}`.",
        f"- 20 um normalized OCT XYZ delta: "
        f"`{metrics['smoke']['20um_normalized_delta']}`.",
        "- Training and evaluation call the same `normalize_oct_symlog` method.",
        "",
        "## Probe 2/3 before vs after",
        "",
        "| Metric | Baseline | OCT symlog |",
        "|---|---:|---:|",
    ]
    if metrics.get("probe"):
        before = metrics["baseline_probe"]["probe2"]["under_60"]["features"]
        after = metrics["probe"]["probe2"]["under_60"]["features"]
        for feature, label in (
            ("F_cond", "F_cond cosine <60"),
            ("F_cond", "F_cond sign X <60"),
            ("F_cond", "F_cond sign Y <60"),
        ):
            if label.endswith("cosine <60"):
                b = before[feature]["mean_cosine"]
                a = after[feature]["mean_cosine"]
            else:
                axis = label.split()[-2]
                b = before[feature]["sign_accuracy_xyz"][axis]
                a = after[feature]["sign_accuracy_xyz"][axis]
            lines.append(f"| {label} | {b:.4f} | {a:.4f} |")
        for axis in "AB":
            b = metrics["baseline_probe"]["probe3"]["per_axis"][axis][
                "fraction_observations_with_sign_flip_action0"
            ]
            a = metrics["probe"]["probe3"]["per_axis"][axis][
                "fraction_observations_with_sign_flip_action0"
            ]
            lines.append(
                f"| K=8 sign flip {axis} | {b:.4f} | {a:.4f} |"
            )
    lines += [
        "",
        "## fast5 1600",
        "",
        "| Model/Target | min um | final um | entered 20 | max consecutive "
        "<=20 (cycles/s) | hold max um | late A flip | late B flip | drift um | "
        "200-cycle hold |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_name in ("Baseline", "OCT symlog"):
        key = "baseline_fast5" if model_name == "Baseline" else "fast5"
        for target, row in metrics[key].items():
            lines.append(
                f"| {model_name}/{target} | {row['min_error_um']:.3f} | "
                f"{row['final_error_um']:.3f} | {row['entered_20_um']} | "
                f"{row['max_consecutive_le_20_cycles']}/"
                f"{row['max_consecutive_le_20_seconds']:.2f} | "
                f"{row['hold_segment_max_error_um']} | "
                f"{row['late_ab_sign_flip_rate']['A']:.3f} | "
                f"{row['late_ab_sign_flip_rate']['B']:.3f} | "
                f"{row['absolute_drift_1000_to_1600_um']:.3f} | "
                f"{row['strict_hold_200_cycles']} |"
            )
    lines += [
        "",
        f"- Baseline aggregate: `{metrics['baseline_fast5_aggregate']}`",
        f"- OCT-symlog aggregate: `{metrics['fast5_aggregate']}`",
        "",
        "## K=4 mean inference-only check",
        "",
        f"`{metrics.get('k4')}`",
        "",
        "## dev20",
        "",
        f"`{metrics.get('dev20')}`",
        "",
        "## Verdict",
        "",
    ]
    gate_result = metrics["gate"]
    if gate_result["passed"]:
        lines.append(
            "OCT symlog crossed the conditional fast5 gate. The experiment "
            "therefore continued to the budget-permitted validation stages."
        )
    else:
        lines.append(
            "OCT symlog did not cross the conditional fast5 gate; no dev20 "
            "run was started."
        )
    if metrics["fast5_aggregate"]["entered_20_um_count"] > 0:
        lines.append(
            "Targets now enter <=20 um, but sustained 200-cycle holds remain "
            "the decision point for Stage 5 terminal hold/braking data."
        )
    else:
        lines.append(
            "No target entered <=20 um, so Stage 5 terminal hold data is not "
            "yet supported by this controlled run."
        )
    path = OUTPUT / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def publish(metrics):
    files = {
        "rcm_4090_deploy/scripts/research/stage4_1_oct_symlog.py":
            ROOT / "scripts/research/stage4_1_oct_symlog.py",
        "rcm_4090_deploy/scripts/research/probe_near_target_blindness.py":
            ROOT / "scripts/research/probe_near_target_blindness.py",
        "rcm_4090_deploy/lerobot_training/dit_state_attention.py":
            ROOT / "lerobot_training/dit_state_attention.py",
        "rcm_4090_deploy/lerobot_training/eval_policy.py":
            ROOT / "lerobot_training/eval_policy.py",
        "rcm_4090_deploy/scripts/research/a_axis_flip_common.py":
            ROOT / "scripts/research/a_axis_flip_common.py",
        "rcm_4090_deploy/scripts/research/homog_replay_evaluate.py":
            ROOT / "scripts/research/homog_replay_evaluate.py",
        "rcm_4090_deploy/tests/test_stage4_oct_symlog.py":
            ROOT / "tests/test_stage4_oct_symlog.py",
        "rcm_4090_deploy/runs/20261125/stage4_1_oct_symlog/report.md":
            OUTPUT / "report.md",
        "rcm_4090_deploy/runs/20261125/stage4_1_oct_symlog/metrics.json":
            OUTPUT / "metrics.json",
        "rcm_4090_deploy/TEST_LOG.md": ROOT / "TEST_LOG.md",
    }
    for name, path in list(files.items()):
        if not Path(path).is_file():
            del files[name]
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        "ssh -i /home/him1/.ssh/rcm_github_deploy_ed25519 "
        "-o IdentitiesOnly=yes -o BatchMode=yes"
    )
    parent = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", f"refs/heads/{BRANCH}"],
        text=True,
    ).strip()
    if parent != BASE_BRANCH_COMMIT:
        raise ValueError("stage4 branch parent differs from approved base")
    tmpdir = Path(subprocess.check_output(
        ["mktemp", "-d"], text=True
    ).strip())
    index_path = tmpdir / "index"
    index_env = {**env, "GIT_INDEX_FILE": str(index_path)}
    try:
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "read-tree", parent],
            check=True, env=index_env,
        )
        entries = []
        modes = {}
        for entry in subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "ls-tree", "-r", "-z", parent],
            env=index_env,
        ).split(b"\0"):
            if entry:
                metadata, path = entry.split(b"\t", 1)
                modes[path.decode()] = metadata.split()[0].decode()
        for name, path in files.items():
            data = Path(path).read_bytes()
            if Path(name).suffix in (".pth", ".npz", ".npy"):
                raise ValueError(f"forbidden publish asset {name}")
            if any(marker in data for marker in (
                b"BEGIN " + b"OPENSSH PRIVATE KEY",
                b"BEGIN " + b"RSA PRIVATE KEY",
            )):
                raise ValueError("secret material rejected")
            blob = subprocess.run(
                ["git", "-C", str(SOURCE_REPO), "hash-object", "-w", "--stdin"],
                input=data, check=True, capture_output=True, env=index_env,
            ).stdout.strip().decode()
            entries.append(
                f"{modes.get(name, '100644')} {blob}\t{name}\0".encode()
            )
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "update-index", "-z",
             "--index-info"],
            input=b"".join(entries), check=True, env=index_env,
        )
        tree = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "write-tree"],
            text=True, env=index_env,
        ).strip()
        deleted = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "diff-tree", "-r",
             "--no-commit-id", "--diff-filter=D", "--name-only", "-z",
             parent, tree],
            env=index_env,
        )
        if deleted:
            raise ValueError("publication tree contains deletions")
        commit = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "commit-tree", tree, "-p",
             parent, "-m", "Add controlled OCT symlog Stage 4.1 experiment"],
            text=True, env=index_env,
        ).strip()
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "update-ref",
             f"refs/heads/{BRANCH}", commit, parent],
            check=True, env=index_env,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "push", "origin",
             f"refs/heads/{BRANCH}:refs/heads/{BRANCH}"],
            check=True, env=env,
        )
        remote = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "ls-remote", "origin",
             f"refs/heads/{BRANCH}"],
            text=True, env=env,
        ).split()[0]
        if remote != commit:
            raise ValueError("remote commit verification failed")
        metrics["publication"] = {
            "status": "pushed_verified",
            "branch": BRANCH,
            "commit": commit,
            "remote_commit": remote,
            "files": sorted(files),
        }
    finally:
        if index_path.exists():
            index_path.unlink()
        tmpdir.rmdir()


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / "workflow.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Stage 4.1 workflow is already running") from error
        started = time.time()
        deadline = started + HARD_BUDGET_MIN * 60.
        state_path = OUTPUT / "status.json"
        state = {
            "status": "running",
            "started_at": stamp(),
            "started_epoch": started,
            "hard_stop_at": datetime.fromtimestamp(
                deadline, timezone.utc
            ).astimezone().isoformat(timespec="seconds"),
        }
        write_json(state_path, state)
        metrics = {
            "status": "running",
            "baseline_recipe": {
                "data": str(DATA),
                "val": str(VAL),
                "overlay": str(OVERLAY),
                "init": str(INIT),
                "epochs": 4,
                "steps_per_epoch": 1000,
                "batch_size": 16,
                "correction_ratio": .08,
                "seed": 20261118,
                "n_obs_steps": 2,
                "n_action_steps": 16,
                "oct_symlog": True,
            },
            "smoke": verify_smoke(),
            "baseline_probe": read(STEP40 / "metrics.json"),
        }
        try:
            state.update(stage="train")
            write_json(state_path, state)
            receipt = train(
                CHECKPOINT_DIR,
                OUTPUT / "train.log",
                deadline,
            )
            checkpoint = CHECKPOINT_DIR / "step_004000.pth"
            config = read(CHECKPOINT_DIR / "training_config.json")
            if (
                receipt["status"] != "completed"
                or receipt["total_batches"] != 4000
                or not checkpoint.is_file()
            ):
                raise ValueError("Stage 4.1 training contract failed")
            metrics["training"] = {
                "status": "completed",
                "receipt": receipt,
                "checkpoint": str(checkpoint),
                "oct_symlog": config["oct_symlog"],
                "oct_symlog_enabled": config["oct_symlog_enabled"],
                "history": read(CHECKPOINT_DIR / "training_history.json"),
            }
            state.update(stage="probe")
            write_json(state_path, state)
            metrics["probe"] = probe(
                checkpoint,
                OUTPUT / "probe_step4000",
                OUTPUT / "probe_step4000.log",
                deadline,
            )
            state.update(stage="fast5")
            write_json(state_path, state)
            evaluate(
                checkpoint,
                OUTPUT / "eval_fast5/step4000",
                FAST5,
                5,
                "median",
                OUTPUT / "eval_fast5.log",
                deadline,
            )
            metrics["baseline_fast5"] = target_metrics_from_tree(
                BASELINE_FAST5, FAST5
            )
            metrics["fast5"] = target_metrics_from_tree(
                OUTPUT / "eval_fast5/step4000/shards/tasks", FAST5
            )
            metrics["baseline_fast5_aggregate"] = aggregate_targets(
                metrics["baseline_fast5"]
            )
            metrics["fast5_aggregate"] = aggregate_targets(metrics["fast5"])
            metrics["gate"] = {
                "passed": gate(metrics["fast5"], metrics["fast5_aggregate"]),
                "rule": (
                    "(final_mean<58 and min_mean<25) or "
                    "entered_20_um>=2/5"
                ),
            }
            if can_start(started, K4_ETA_MIN):
                state.update(stage="k4_mean")
                write_json(state_path, state)
                evaluate(
                    checkpoint,
                    OUTPUT / "eval_k4_mean",
                    (6, 10),
                    2,
                    "mean",
                    OUTPUT / "eval_k4_mean.log",
                    deadline,
                )
                metrics["k4"] = {
                    "status": "completed",
                    "targets": target_metrics_from_tree(
                        OUTPUT / "eval_k4_mean/shards/tasks", (6, 10)
                    ),
                }
            else:
                metrics["k4"] = {"status": "skipped_time"}
            if (
                metrics["gate"]["passed"]
                and remaining_minutes(started) >= DEV20_GATE_REMAINING_MIN
                and can_start(started, DEV20_ETA_MIN)
            ):
                state.update(stage="dev20")
                write_json(state_path, state)
                evaluate(
                    checkpoint,
                    OUTPUT / "eval_dev20",
                    DEV20,
                    3,
                    "median",
                    OUTPUT / "eval_dev20.log",
                    deadline,
                )
                metrics["dev20"] = {
                    "status": "completed",
                    "targets": target_metrics_from_tree(
                        OUTPUT / "eval_dev20/shards/tasks", DEV20
                    ),
                    "aggregate": aggregate_targets(target_metrics_from_tree(
                        OUTPUT / "eval_dev20/shards/tasks", DEV20
                    )),
                }
            else:
                metrics["dev20"] = {
                    "status": "skipped_gate_or_time",
                    "gate_passed": metrics["gate"]["passed"],
                    "remaining_minutes": remaining_minutes(started),
                }
            metrics["status"] = "completed"
            record("workflow", "completed", metrics["gate"])
        except BaseException as error:
            metrics["status"] = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, TimeoutError))
                else "failed"
            )
            metrics["reason"] = repr(error)
            record("workflow", metrics["status"], metrics["reason"])
        finally:
            metrics["elapsed_minutes"] = elapsed_minutes(started)
            write_json(OUTPUT / "metrics.json", metrics)
            report = write_report(metrics)
            metrics["report"] = str(report)
            try:
                publish(metrics)
                write_json(OUTPUT / "metrics.json", metrics)
            except BaseException as error:
                metrics["publication"] = {
                    "status": "failed",
                    "reason": repr(error),
                }
                write_json(OUTPUT / "metrics.json", metrics)
                record("publication", "failed", repr(error))
            state.update(status=metrics["status"], stage="report")
            write_json(state_path, state)
            if metrics.get("reason"):
                raise RuntimeError(metrics["reason"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    evaluate_parser = subparsers.add_parser("_evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--destination", type=Path, required=True)
    evaluate_parser.add_argument("--indices", required=True)
    evaluate_parser.add_argument("--workers", type=int, default=3)
    evaluate_parser.add_argument(
        "--action-aggregation", choices=("median", "mean"),
        default="median",
    )
    args = parser.parse_args(argv)
    if args.action == "run":
        if args.output_dir.resolve() != OUTPUT:
            raise ValueError(f"authorized output is {OUTPUT}")
        run()
    else:
        _evaluate_worker(args)


if __name__ == "__main__":
    main()
