"""Matched local-scale signed OCT branch experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/20261125/local_oct_branch_v1"
CHECKPOINT_DIR = OUTPUT / "checkpoints/arm_B_local_oct"
DATA = ROOT / "data/plan2_plus_original_train_224_20261119"
VAL = ROOT / "data/lerobot_rcm_val_acthist_20261117_224"
OVERLAY = ROOT / "data/overnight_c_fix_symlog_081234_overlay/label_override.npz"
INIT = ROOT / "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth"
CONTROL = ROOT / "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth"
CONTROL_FAST5 = (
    ROOT / "github_reports/homog_label_fix_20261123_v3_s3_fast5_1600"
    / "baseline_task2_step4000/shards/tasks"
)
CONTROL_DEV20 = (
    ROOT / "runs/20261124/homog_label_fix_20261124_v5_3h/reports"
    / "homog_label_fix_20261124_v5_phase0_baseline_dev20/shards/tasks"
)
STEP40 = ROOT / "runs/20261125/step4_0_near_target_blindness_probe"
SOURCE_REPO = ROOT / "runs/20261123/homog_clean_git_v1/repository"
BRANCH = "experiment/local-oct-branch-20261125"
BASE_COMMIT = "630c03b6aac27e92a1f3c5dbefc577eb518cf657"
FAST5 = (0, 2, 6, 8, 10)
DEV20 = tuple(range(20))
HARD_MINUTES = 150.
CLOSE_OUT_MINUTES = 20.
TRAIN_ETA = 8.
FAST5_ETA = 25.
SAMPLE_ETA = 20.
DEV20_ETA = 55.
MODULE = "scripts.research.local_oct_branch_v1"


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


def elapsed(started):
    return (time.time() - started) / 60.


def remaining(started):
    return HARD_MINUTES - elapsed(started)


def can_start(started, eta):
    return remaining(started) >= eta + CLOSE_OUT_MINUTES


def record(purpose, status, details):
    with (ROOT / "TEST_LOG.md").open("ab") as stream:
        stream.write(
            (
                f"\n- {stamp()}: local_oct_branch_v1 {purpose} **{status}**; "
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


def terminate(process):
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


def run_managed(command, log, deadline):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            if time.time() >= deadline:
                terminate(process)
                raise TimeoutError("local OCT branch hard deadline")
            time.sleep(.25)
    if process.returncode:
        raise RuntimeError(f"command failed; log={log}")


def train_command(destination):
    return env_command() + [
        "-m", "lerobot_training.dit_experience_replay", "train",
        "--data-dir", str(DATA),
        "--val-data-dir", str(VAL),
        "--init-from", str(INIT),
        "--output-dir", str(destination),
        "--mode", "ratio",
        "--correction-ratio", "0.08",
        "--epochs", "4",
        "--steps-per-epoch", "1000",
        "--batch-size", "16",
        "--val-batch-size", "64",
        "--num-workers", "8",
        "--checkpoint-every-steps", "1000",
        "--log-every-steps", "100",
        "--backbone-lr", "1e-6",
        "--attention-lr", "1e-5",
        "--seed", "20261118",
        "--validation-seed", "20261119",
        "--label-overlay", str(OVERLAY),
        "--use-local-oct-branch",
    ]


def train(deadline):
    receipt = CHECKPOINT_DIR / "training_receipt.json"
    if receipt.is_file() and read(receipt).get("status") == "completed":
        return read(receipt)
    command = train_command(CHECKPOINT_DIR)
    if (CHECKPOINT_DIR / "resume_state.pth").is_file():
        command.append("--resume")
    run_managed(command, OUTPUT / "train_arm_B.log", deadline)
    return read(receipt)


def evaluation_command(checkpoint, destination, indices, workers, aggregation):
    return env_command() + [
        "-m", MODULE, "_evaluate",
        "--checkpoint", str(checkpoint),
        "--destination", str(destination),
        "--indices", ",".join(str(value) for value in indices),
        "--workers", str(workers),
        "--action-aggregation", aggregation,
    ]


def evaluate(checkpoint, destination, indices, workers, aggregation, log, deadline):
    destination = Path(destination)
    summary = destination / "summary.json"
    if summary.is_file():
        return read(summary)
    run_managed(
        evaluation_command(
            checkpoint, destination, indices, workers, aggregation
        ),
        log,
        deadline,
    )
    return read(summary)


def probe(checkpoint, deadline):
    destination = OUTPUT / "probe_arm_B"
    metrics_path = destination / "metrics.json"
    if metrics_path.is_file():
        return read(metrics_path)
    run_managed(env_command() + [
        "-m", "scripts.research.probe_near_target_blindness",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(destination),
    ], OUTPUT / "probe_arm_B.log", deadline)
    return read(metrics_path)


def _evaluate_worker(args):
    from scripts.research.homog_replay_evaluate import evaluate
    from scripts.research.homog_replay_common import identity

    indices = tuple(int(value) for value in args.indices.split(",") if value)
    evaluate(
        args.checkpoint,
        1,
        "feedback",
        indices,
        args.destination,
        {"identity": identity()},
        workers=args.workers,
        max_workers=6,
        action_aggregation=args.action_aggregation,
    )


def trajectory_table(root, targets):
    from scripts.research.eval_20um_hold_metrics import trajectory_metrics

    return {
        f"E{index:02d}": trajectory_metrics(
            Path(root) / f"eval_E{index:02d}/raw/evaluation_results.json"
        )
        for index in targets
    }


def toward_and_flips(path):
    from scripts.research.diagnose_20um_bottleneck import load_target

    data = load_target(path)
    errors = data["true_error_um"]
    rows = data["rows"]
    selected = [
        row for row in rows
        if row["true_error_um"] < 60.
        and row["policy_command_norm"] > 1e-12
    ]
    actions = np.asarray(data["accepted_actions"])
    late = actions[1000:1600]
    flips = {}
    for index, axis in enumerate("ABC"):
        left, right = late[:-1, index], late[1:, index]
        active = (np.abs(left) > 1e-12) & (np.abs(right) > 1e-12)
        flips[axis] = float(np.mean(
            np.sign(left[active]) != np.sign(right[active])
        )) if active.any() else 0.
    return {
        "toward_fraction_under_60": (
            float(np.mean([
                row["policy_cosine"] > 0 for row in selected
            ])) if selected else None
        ),
        "policy_cosine_median_under_60": (
            float(np.median([
                row["policy_cosine"] for row in selected
            ])) if selected else None
        ),
        "late_abc_sign_flip_rate": flips,
    }


def aggregate(rows):
    values = [
        value for value in rows.values()
        if value["status"] == "completed"
    ]
    return {
        "completed_targets": len(values),
        "final_mean_um": float(np.mean([
            value["final_error_um"] for value in values
        ])),
        "min_mean_um": float(np.mean([
            value["min_error_um"] for value in values
        ])),
        "entered_20_um_count": sum(
            value["entries_le_20_um"] > 0 for value in values
        ),
        "sustained_200_count": sum(
            value["sustained_200_pass"] for value in values
        ),
    }


def verify_smokes():
    import torch

    result = {}
    for arm, enabled in (("A", False), ("B", True)):
        root = OUTPUT / f"smoke/{arm}/train"
        checkpoint = torch.load(
            root / "step_000002.pth", map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        rows = [
            json.loads(line)
            for line in (root / "training_metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]
        local_keys = [
            name for name in checkpoint["model_state_dict"]
            if name.startswith("local_oct_mlp.")
        ]
        result[arm] = {
            "use_local_oct_branch": bool(config["use_local_oct_branch"]),
            "local_parameter_count": len(local_keys),
            "metrics_finite": all(
                np.isfinite(row["gradient_norm_before_clip"])
                for row in rows
            ),
            "checkpoint": str(root / "step_000002.pth"),
        }
        if bool(config["use_local_oct_branch"]) != enabled:
            raise ValueError(f"smoke flag mismatch for arm {arm}")
    if (
        result["A"]["local_parameter_count"] != 0
        or result["B"]["local_parameter_count"] != 4
        or not result["A"]["metrics_finite"]
        or not result["B"]["metrics_finite"]
    ):
        raise ValueError("local OCT smoke contract failed")
    result["status"] = "passed"
    return result


def matched_control():
    import torch

    baseline = torch.load(CONTROL, map_location="cpu", weights_only=False)
    init = torch.load(INIT, map_location="cpu", weights_only=False)
    cfg = baseline["config"]
    expected = {
        "data_dir": str(DATA.relative_to(ROOT)),
        "val_data_dir": str(VAL.relative_to(ROOT)),
        "init_from": str(INIT.relative_to(ROOT)),
        "epochs": 4,
        "steps_per_epoch": 1000,
        "batch_size": 16,
        "seed": 20261118,
        "correction_ratio": .08,
        "backbone_lr": 1e-6,
        "attention_lr": 1e-5,
    }
    mismatches = {
        key: {"expected": value, "baseline": cfg.get(key)}
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    if mismatches:
        raise ValueError(f"historical baseline recipe mismatch: {mismatches}")
    return {
        "control": "historical baseline_task2_step4000",
        "match_verified": True,
        "same_initialization_checkpoint_epoch": int(init.get("epoch", -1)),
        "expected": expected,
        "baseline_sha256": __import__(
            "lerobot_training.navigation_overlay", fromlist=["file_sha256"]
        ).file_sha256(CONTROL),
    }


def write_report(metrics):
    lines = [
        "# Local-scale signed OCT branch v1",
        "",
        f"Status: `{metrics.get('status', 'running')}`",
        "",
        "## Step 1: unified <=20 um hold metrics",
        "",
        "See `step1_existing_20um_metrics.md` and "
        "`step1_existing_20um_metrics.json`.",
        "",
        "## Step 2: matched control",
        "",
        f"- Control: `{metrics['control']['control']}`",
        f"- Historical recipe verified: `{metrics['control']['match_verified']}`",
        f"- Arm B training: `{metrics.get('training', {}).get('status')}`",
        f"- Arm B config: `{metrics.get('training', {}).get('branch_config')}`",
        "",
        "| Model | final mean um | min mean um | entered <=20 | sustained 200 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("Control", "control_fast5"), ("Local OCT", "arm_B_fast5")):
        value = metrics.get(key, {})
        if not value:
            continue
        lines.append(
            f"| {label} | {value['final_mean_um']:.3f} | "
            f"{value['min_mean_um']:.3f} | {value['entered_20_um_count']}/5 | "
            f"{value['sustained_200_count']}/5 |"
        )
    lines += [
        "",
        "| Model/Target | min um | final um | entries <=20 | max cycles | "
        "sustained 200 | last-200 max um | late A/B/C flip | speed overall | "
        "toward <60 |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for label, key in (("Control", "control_fast5"), ("Local OCT", "arm_B_fast5")):
        for target, row in metrics.get(key + "_targets", {}).items():
            if row["status"] != "completed":
                continue
            flips = metrics.get(key + "_direction", {}).get(target, {}).get(
                "late_abc_sign_flip_rate", {}
            )
            toward = metrics.get(key + "_direction", {}).get(target, {}).get(
                "toward_fraction_under_60"
            )
            lines.append(
                f"| {label}/{target} | {row['min_error_um']:.3f} | "
                f"{row['final_error_um']:.3f} | {row['entries_le_20_um']} | "
                f"{row['max_consecutive_le_20_um_cycles']} | "
                f"{row['sustained_200_pass']} | "
                f"{row['last_200_max_error_um']:.3f} | "
                f"{flips.get('A', 0):.3f}/{flips.get('B', 0):.3f}/"
                f"{flips.get('C', 0):.3f} | "
                f"{row['actual_speed_um_per_s_overall']:.3f} |"
                f"{toward if toward is None else f'{toward:.3f}'} |"
            )
    lines += [
        "",
        "## Probe 2/3",
        "",
        f"`{metrics.get('probe_comparison')}`",
        "",
        "## Step 3: multi-sample check",
        "",
    ]
    step3 = metrics.get("step3", {})
    if step3.get("status") == "completed":
        lines += [
            f"Winner: `{step3['winner']}`.",
            "",
            "| Target | K | min um | final um | entries <=20 | max cycles | "
            "last-200 max um | late A/B/C flip | toward <60 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        reference = step3["single_sample_reference"]
        for target, row in step3["k4_mean_targets"].items():
            if row["status"] != "completed" or target not in reference:
                continue
            old = reference[target]
            direction = step3.get("k4_mean_direction", {}).get(target, {})
            reference_direction = step3.get(
                "single_sample_direction", {}
            ).get(target, {})
            flips = direction.get("late_abc_sign_flip_rate", {})
            toward = direction.get("toward_fraction_under_60")
            old_flips = reference_direction.get(
                "late_abc_sign_flip_rate", {}
            )
            old_toward = reference_direction.get(
                "toward_fraction_under_60"
            )
            lines.append(
                f"| {target} | 1 | {old['min_error_um']:.3f} | "
                f"{old['final_error_um']:.3f} | "
                f"{old['entries_le_20_um']} | "
                f"{old['max_consecutive_le_20_um_cycles']} | "
                f"{old['last_200_max_error_um']:.3f} | "
                f"{old_flips.get('A', 0):.3f}/"
                f"{old_flips.get('B', 0):.3f}/"
                f"{old_flips.get('C', 0):.3f} | "
                f"{old_toward if old_toward is None else f'{old_toward:.3f}'} |"
            )
            lines.append(
                f"| {target} | 4 mean | {row['min_error_um']:.3f} | "
                f"{row['final_error_um']:.3f} | "
                f"{row['entries_le_20_um']} | "
                f"{row['max_consecutive_le_20_um_cycles']} | "
                f"{row['last_200_max_error_um']:.3f} | "
                f"{row['late_ab_sign_flip_rate']['A']:.3f}/"
                f"{row['late_ab_sign_flip_rate']['B']:.3f}/"
                f"{row['late_ab_sign_flip_rate']['C']:.3f} | "
                f"{toward if toward is None else f'{toward:.3f}'} |"
            )
    else:
        lines.append(f"`{step3}`")
    lines += [
        "",
        "## dev20",
        "",
        f"`{metrics.get('dev20')}`",
        "",
        "## Decision",
        "",
        f"`{metrics['decision']}`",
        "",
    ]
    path = OUTPUT / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def publish(metrics):
    files = {
        "rcm_4090_deploy/scripts/research/eval_20um_hold_metrics.py":
            ROOT / "scripts/research/eval_20um_hold_metrics.py",
        "rcm_4090_deploy/scripts/research/local_oct_branch_v1.py":
            ROOT / "scripts/research/local_oct_branch_v1.py",
        "rcm_4090_deploy/scripts/research/probe_near_target_blindness.py":
            ROOT / "scripts/research/probe_near_target_blindness.py",
        "rcm_4090_deploy/scripts/research/diagnose_20um_bottleneck.py":
            ROOT / "scripts/research/diagnose_20um_bottleneck.py",
        "rcm_4090_deploy/lerobot_training/dit_state_attention.py":
            ROOT / "lerobot_training/dit_state_attention.py",
        "rcm_4090_deploy/lerobot_training/dit_experience_replay.py":
            ROOT / "lerobot_training/dit_experience_replay.py",
        "rcm_4090_deploy/lerobot_training/eval_policy.py":
            ROOT / "lerobot_training/eval_policy.py",
        "rcm_4090_deploy/scripts/research/a_axis_flip_common.py":
            ROOT / "scripts/research/a_axis_flip_common.py",
        "rcm_4090_deploy/scripts/research/homog_replay_evaluate.py":
            ROOT / "scripts/research/homog_replay_evaluate.py",
        "rcm_4090_deploy/tests/test_local_oct_branch.py":
            ROOT / "tests/test_local_oct_branch.py",
        "rcm_4090_deploy/runs/20261125/local_oct_branch_v1/step1_existing_20um_metrics.md":
            OUTPUT / "step1_existing_20um_metrics.md",
        "rcm_4090_deploy/runs/20261125/local_oct_branch_v1/step1_existing_20um_metrics.json":
            OUTPUT / "step1_existing_20um_metrics.json",
        "rcm_4090_deploy/runs/20261125/local_oct_branch_v1/report.md":
            OUTPUT / "report.md",
        "rcm_4090_deploy/runs/20261125/local_oct_branch_v1/metrics.json":
            OUTPUT / "metrics.json",
        "rcm_4090_deploy/TEST_LOG.md": ROOT / "TEST_LOG.md",
    }
    files = {name: path for name, path in files.items() if path.is_file()}
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        "ssh -i /home/him1/.ssh/rcm_github_deploy_ed25519 "
        "-o IdentitiesOnly=yes -o BatchMode=yes"
    )
    parent = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", f"refs/heads/{BRANCH}"],
        text=True,
    ).strip()
    if parent != BASE_COMMIT:
        raise ValueError("local OCT branch parent mismatch")
    temp = Path(subprocess.check_output(["mktemp", "-d"], text=True).strip())
    index = temp / "index"
    index_env = {**env, "GIT_INDEX_FILE": str(index)}
    try:
        subprocess.run(
            ["git", "-C", str(SOURCE_REPO), "read-tree", parent],
            check=True, env=index_env,
        )
        modes = {}
        for entry in subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "ls-tree", "-r", "-z", parent],
            env=index_env,
        ).split(b"\0"):
            if entry:
                metadata, name = entry.split(b"\t", 1)
                modes[name.decode()] = metadata.split()[0].decode()
        entries = []
        for name, path in files.items():
            data = Path(path).read_bytes()
            if Path(name).suffix in (".pth", ".npz", ".npy"):
                raise ValueError(f"forbidden asset {name}")
            if any(marker in data for marker in (
                b"BEGIN " + b"OPENSSH PRIVATE KEY",
                b"BEGIN " + b"RSA PRIVATE KEY",
                b"github" + b"_pat_",
                b"gh" + b"p_",
            )):
                raise ValueError("secret material rejected")
            blob = subprocess.run(
                ["git", "-C", str(SOURCE_REPO), "hash-object", "-w", "--stdin"],
                input=data, capture_output=True, check=True, env=index_env,
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
            raise ValueError("publication contains deletions")
        commit = subprocess.check_output(
            ["git", "-C", str(SOURCE_REPO), "commit-tree", tree, "-p",
             parent, "-m", "Add local-scale signed OCT branch experiment"],
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
            raise ValueError("remote verification failed")
        metrics["publication"] = {
            "status": "pushed_verified",
            "branch": BRANCH,
            "commit": commit,
            "remote_commit": remote,
        }
    finally:
        if index.exists():
            index.unlink()
        temp.rmdir()


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("local OCT workflow is already running") from error
        started = time.time()
        deadline = started + HARD_MINUTES * 60.
        state = {
            "status": "running",
            "stage": "preflight",
            "started_at": stamp(),
            "started_epoch": started,
            "hard_stop_at": datetime.fromtimestamp(
                deadline, timezone.utc
            ).astimezone().isoformat(timespec="seconds"),
        }
        metrics = {
            "status": "running",
            "smoke": verify_smokes(),
            "control": matched_control(),
            "decision": {
                "step2_pass": False,
                "winner": None,
                "stage4": "incomplete",
            },
        }
        write_json(OUTPUT / "status.json", state)
        try:
            if not can_start(started, TRAIN_ETA):
                raise TimeoutError("insufficient time for Arm B training")
            state.update(stage="train_arm_B")
            write_json(OUTPUT / "status.json", state)
            receipt = train(deadline)
            checkpoint = CHECKPOINT_DIR / "step_004000.pth"
            config = read(CHECKPOINT_DIR / "training_config.json")
            if (
                receipt["status"] != "completed"
                or receipt["total_batches"] != 4000
                or not config["use_local_oct_branch"]
                or not checkpoint.is_file()
            ):
                raise ValueError("Arm B training contract failed")
            metrics["training"] = {
                "status": "completed",
                "receipt": receipt,
                "checkpoint": str(checkpoint),
                "branch_config": {
                    key: config[key] for key in (
                        "use_local_oct_branch",
                        "local_oct_tanh_scale_mm",
                        "local_oct_log_scale_mm",
                        "local_oct_log_reference_mm",
                    )
                },
            }
            metrics["probe_control"] = read(STEP40 / "metrics.json")
            state.update(stage="probe_arm_B")
            write_json(OUTPUT / "status.json", state)
            metrics["probe_arm_B"] = probe(checkpoint, deadline)
            control_probe = metrics["probe_control"]["probe2"]["under_60"][
                "features"
            ]["F_cond"]
            local_probe = metrics["probe_arm_B"]["probe2"]["under_60"][
                "features"
            ]["F_cond"]
            metrics["probe_comparison"] = {
                "control_cosine": control_probe["mean_cosine"],
                "local_cosine": local_probe["mean_cosine"],
                "control_sign_xyz": control_probe["sign_accuracy_xyz"],
                "local_sign_xyz": local_probe["sign_accuracy_xyz"],
                "control_ab_sign_flip": {
                    axis: metrics["probe_control"]["probe3"]["per_axis"][axis][
                        "fraction_observations_with_sign_flip_action0"
                    ]
                    for axis in "AB"
                },
                "local_ab_sign_flip": {
                    axis: metrics["probe_arm_B"]["probe3"]["per_axis"][axis][
                        "fraction_observations_with_sign_flip_action0"
                    ]
                    for axis in "AB"
                },
                "cosine_improved": (
                    local_probe["mean_cosine"] > control_probe["mean_cosine"]
                ),
            }
            if not can_start(started, FAST5_ETA):
                raise TimeoutError("insufficient time for fast5")
            state.update(stage="fast5_arm_B")
            write_json(OUTPUT / "status.json", state)
            evaluate(
                checkpoint,
                OUTPUT / "eval_fast5_arm_B",
                FAST5,
                5,
                "median",
                OUTPUT / "fast5_arm_B.log",
                deadline,
            )
            metrics["control_fast5_targets"] = trajectory_table(
                CONTROL_FAST5, FAST5
            )
            metrics["arm_B_fast5_targets"] = trajectory_table(
                OUTPUT / "eval_fast5_arm_B/shards/tasks", FAST5
            )
            metrics["control_fast5_direction"] = {
                f"E{index:02d}": toward_and_flips(
                    CONTROL_FAST5
                    / f"eval_E{index:02d}/raw/evaluation_results.json"
                )
                for index in FAST5
            }
            metrics["arm_B_fast5_direction"] = {
                f"E{index:02d}": toward_and_flips(
                    OUTPUT
                    / "eval_fast5_arm_B/shards/tasks"
                    / f"eval_E{index:02d}/raw/evaluation_results.json"
                )
                for index in FAST5
            }
            metrics["control_fast5"] = aggregate(
                metrics["control_fast5_targets"]
            )
            metrics["arm_B_fast5"] = aggregate(
                metrics["arm_B_fast5_targets"]
            )
            control_toward = float(np.nanmean([
                value["toward_fraction_under_60"]
                for value in metrics["control_fast5_direction"].values()
                if value["toward_fraction_under_60"] is not None
            ]))
            local_toward = float(np.nanmean([
                value["toward_fraction_under_60"]
                for value in metrics["arm_B_fast5_direction"].values()
                if value["toward_fraction_under_60"] is not None
            ]))
            metrics["direction_comparison"] = {
                "control_toward_fraction_under_60": control_toward,
                "local_toward_fraction_under_60": local_toward,
                "toward_improved": local_toward > control_toward,
            }
            metrics["step2_pass"] = bool(
                metrics["probe_comparison"]["cosine_improved"]
                and metrics["direction_comparison"]["toward_improved"]
                and metrics["arm_B_fast5"]["final_mean_um"]
                < metrics["control_fast5"]["final_mean_um"]
                and metrics["arm_B_fast5"]["min_mean_um"]
                < metrics["control_fast5"]["min_mean_um"]
            )
            winner = (
                "arm_B"
                if metrics["arm_B_fast5"]["final_mean_um"]
                <= metrics["control_fast5"]["final_mean_um"]
                else "control"
            )
            winner_checkpoint = checkpoint if winner == "arm_B" else CONTROL
            winner_dest = OUTPUT / f"eval_step3_{winner}_k4_mean"
            if can_start(started, SAMPLE_ETA):
                state.update(stage=f"step3_{winner}_k4_mean")
                write_json(OUTPUT / "status.json", state)
                evaluate(
                    winner_checkpoint,
                    winner_dest,
                    (6, 10),
                    2,
                    "mean",
                    OUTPUT / f"step3_{winner}_k4_mean.log",
                    deadline,
                )
                metrics["step3"] = {
                    "status": "completed",
                    "winner": winner,
                    "checkpoint": str(winner_checkpoint),
                    "single_sample_reference": (
                        metrics[
                            "arm_B_fast5_targets"
                            if winner == "arm_B"
                            else "control_fast5_targets"
                        ]
                    ),
                    "single_sample_direction": (
                        metrics["arm_B_fast5_direction"]
                        if winner == "arm_B"
                        else metrics["control_fast5_direction"]
                    ),
                    "k4_mean_targets": trajectory_table(
                        winner_dest / "shards/tasks", (6, 10)
                    ),
                    "k4_mean_direction": {
                        f"E{index:02d}": toward_and_flips(
                            winner_dest
                            / f"shards/tasks/eval_E{index:02d}/raw/evaluation_results.json"
                        )
                        for index in (6, 10)
                    },
                }
            else:
                metrics["step3"] = {"status": "skipped_time"}
            if (
                metrics["step2_pass"]
                and remaining(started) >= 40.
                and can_start(started, DEV20_ETA)
            ):
                state.update(stage="dev20")
                write_json(OUTPUT / "status.json", state)
                evaluate(
                    checkpoint,
                    OUTPUT / "eval_dev20_arm_B",
                    DEV20,
                    3,
                    "median",
                    OUTPUT / "dev20_arm_B.log",
                    deadline,
                )
                dev20_targets = trajectory_table(
                    OUTPUT / "eval_dev20_arm_B/shards/tasks", DEV20
                )
                metrics["dev20"] = {
                    "status": "completed",
                    "targets": dev20_targets,
                    "aggregate": aggregate(dev20_targets),
                }
            else:
                metrics["dev20"] = {
                    "status": "skipped_gate_or_time",
                    "step2_pass": metrics["step2_pass"],
                    "remaining_minutes": remaining(started),
                }
            metrics["status"] = "completed"
            metrics["decision"] = {
                "step2_pass": metrics["step2_pass"],
                "winner": winner,
                "stage4": (
                    "Conditioning/supervision still needs work; local OCT "
                    "branch did not satisfy both direction and performance gates."
                    if not metrics["step2_pass"]
                    else "Continue with the local OCT branch and inspect terminal "
                    "overshoot/braking only if sustained 200-cycle holds remain zero."
                ),
            }
            record("workflow", "completed", metrics["decision"])
        except BaseException as error:
            metrics["status"] = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, TimeoutError))
                else "failed"
            )
            metrics["reason"] = repr(error)
            record("workflow", metrics["status"], metrics["reason"])
        finally:
            metrics["elapsed_minutes"] = elapsed(started)
            write_json(OUTPUT / "metrics.json", metrics)
            write_report(metrics)
            try:
                publish(metrics)
                write_json(OUTPUT / "metrics.json", metrics)
            except BaseException as error:
                metrics["publication"] = {
                    "status": "failed",
                    "reason": repr(error),
                }
                write_json(OUTPUT / "metrics.json", metrics)
            state.update(status=metrics["status"], stage="report")
            write_json(OUTPUT / "status.json", state)
            if metrics.get("reason"):
                raise RuntimeError(metrics["reason"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    eval_parser = subparsers.add_parser("_evaluate")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--destination", type=Path, required=True)
    eval_parser.add_argument("--indices", required=True)
    eval_parser.add_argument("--workers", type=int, default=3)
    eval_parser.add_argument(
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
