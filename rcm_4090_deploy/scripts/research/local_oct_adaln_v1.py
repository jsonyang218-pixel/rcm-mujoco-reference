"""Matched local-scale signed OCT branch experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import shutil
import traceback
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("RCM_RUNTIME_ROOT", CODE_ROOT)).resolve()
OUTPUT = ROOT / "runs/20261125/local_oct_adaln_v1"
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
STAGE4 = ROOT / "runs/20261125/stage4_1_oct_symlog/checkpoints/stage4_1_oct_symlog/step_004000.pth"
SOURCE_REPO = Path(__file__).resolve().parents[3]
BRANCH = "experiment/local-oct-adaln-20261125"
BASE_COMMIT = "a9d6ed37ae1af76658ba4a85928fb545c91196df"
FAST5 = (0, 2, 6, 8, 10)
DEV20 = tuple(range(20))
HARD_MINUTES = 150.
CLOSE_OUT_MINUTES = 20.
TRAIN_ETA = 80.
FAST5_ETA = 25.
SAMPLE_ETA = 20.
DEV20_ETA = 35.
MODULE = "scripts.research.local_oct_adaln_v1"


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
    return remaining(started) >= max(20., eta + 5.)


def record(purpose, status, details):
    with (ROOT / "TEST_LOG.md").open("ab") as stream:
        stream.write(
            (
                f"\n- {stamp()}: local_oct_adaln_v1 {purpose} **{status}**; "
                f"command `{' '.join(sys.argv)}`; {details}; output "
                f"`{OUTPUT}`. Software/MuJoCo simulation only.\n"
            ).encode()
        )


def env_command():
    return [
        "env", "-u", "PYTHONHOME", "-u", "LD_LIBRARY_PATH",
        f"PYTHONPATH={CODE_ROOT}",
        f"RCM_RUNTIME_ROOT={ROOT}",
        "CUDA_VISIBLE_DEVICES=0",
        "RCM_DRIVER_VARIANT=g10",
        "MUJOCO_GL=egl",
        "MUJOCO_EGL_DEVICE_ID=0",
        "MPLBACKEND=Agg",
        "PYTHONDONTWRITEBYTECODE=1",
        "TORCH_DISABLE_NATIVE_JIT=1",
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
    print("COMMAND " + __import__("shlex").join(command), flush=True)
    with log.open("xb", buffering=0) as stream:
        process = subprocess.Popen(command, cwd=CODE_ROOT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            fd = process.stdout.fileno()
            while True:
                if time.time() >= deadline:
                    raise TimeoutError(f"stage deadline; log={log}")
                ready, _, _ = select.select([fd], [], [], .5)
                if ready:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    stream.write(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            code = process.wait(timeout=5)
            if code:
                raise RuntimeError(f"command exit {code}; log={log}")
        finally:
            # Kill the process group even if the leader exited with live workers.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(.2)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
    print(f"RESULT {log}", flush=True)


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


def smoke_command(destination):
    command = train_command(destination)
    replacements = {
        "--epochs": "1",
        "--steps-per-epoch": "2",
        "--checkpoint-every-steps": "2",
        "--log-every-steps": "1",
    }
    for option, value in replacements.items():
        command[command.index(option) + 1] = value
    command += ["--total-batches", "2", "--validation-max-batches", "1"]
    return command


def run_smoke(deadline):
    destination = OUTPUT / "smoke_local_oct/train"
    run_managed(smoke_command(destination), OUTPUT / "smoke_local_oct.log", deadline)
    import torch

    checkpoint = torch.load(destination / "step_000002.pth", map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in (destination / "training_metrics.jsonl").read_text().splitlines() if line]
    local_state = [value for name, value in checkpoint["model_state_dict"].items() if name.startswith("local_oct_mlp.")]
    finite = bool(rows) and all(
        all(value is None or np.isfinite(value) for value in
            (row["original_loss"], row["correction_loss"]))
        and np.isfinite(row["gradient_norm_before_clip"])
        and np.isfinite(row["local_oct_gradient_norm_after_clip"])
        for row in rows
    )
    changed = bool(any(torch.count_nonzero(value).item() for value in local_state[-2:]))
    gradients = [row["local_oct_gradient_norm_after_clip"] for row in rows]
    gradient_pass = bool(gradients and max(gradients) > 0)
    if (checkpoint["config"].get("use_local_oct_branch") is not True
            or len(rows) != 2 or not finite or not changed or not gradient_pass):
        raise ValueError("two-batch local OCT smoke failed gradients/finiteness contract")
    return {"status": "passed", "batches": len(rows), "finite_loss_and_gradients": finite,
            "local_output_projection_updated": changed, "local_gradient_norms": gradients, "checkpoint": str(destination / "step_000002.pth")}


def train(deadline):
    receipt = CHECKPOINT_DIR / "training_receipt.json"
    if CHECKPOINT_DIR.exists():
        raise FileExistsError(CHECKPOINT_DIR)
    command = train_command(CHECKPOINT_DIR)
    run_managed(command, OUTPUT / "train_arm_B.log", deadline)
    return read(receipt)


def evaluation_command(checkpoint, destination, indices, workers, aggregation, samples=1):
    return env_command() + [
        "-m", MODULE, "_evaluate",
        "--checkpoint", str(checkpoint),
        "--destination", str(destination),
        "--indices", ",".join(str(value) for value in indices),
        "--workers", str(workers),
        "--action-aggregation", aggregation,
        "--samples", str(samples),
    ]


def evaluate(checkpoint, destination, indices, workers, aggregation, log, deadline, samples=1):
    destination = Path(destination)
    summary = destination / "summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    run_managed(
        evaluation_command(
            checkpoint, destination, indices, workers, aggregation, samples
        ),
        log,
        deadline,
    )
    return read(summary)


def probe(checkpoint, destination, deadline):
    destination = Path(destination)
    metrics_path = destination / "metrics.json"
    if destination.exists():
        raise FileExistsError(destination)
    run_managed(env_command() + [
        "-m", "scripts.research.probe_near_target_blindness",
        "--checkpoint", str(checkpoint),
        "--output-dir", str(destination),
    ], OUTPUT / (destination.name + ".log"), deadline)
    return read(metrics_path)


def _evaluate_worker(args):
    from scripts.research.homog_replay_evaluate import evaluate
    from scripts.research.homog_replay_common import identity

    indices = tuple(int(value) for value in args.indices.split(",") if value)
    # Read on each new evaluation subprocess so an authorized concurrency
    # change can take effect between stages without restarting training.
    override_path = OUTPUT / "parallel_override.json"
    override = read(override_path) if override_path.is_file() else {}
    requested_workers = int(override.get("workers", args.workers))
    if not 1 <= requested_workers <= 12:
        raise ValueError("evaluation concurrency must be in [1, 12]")
    workers = min(requested_workers, len(indices))
    print(f"Evaluation workers={workers}; requested={requested_workers}; "
          f"targets={len(indices)}; override={override_path if override else None}", flush=True)
    evaluate(
        args.checkpoint,
        args.samples,
        "feedback",
        indices,
        args.destination,
        {"identity": identity()},
        workers=workers,
        max_workers=12,
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
    if not values:
        return {"completed_targets": 0, "final_mean_um": None, "min_mean_um": None,
                "entered_20_um_count": 0, "sustained_200_count": 0}
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


ARCHITECTURE_FINDING = (
    "Original OCT is normalized linearly with joint state, velocity and previous_action. "
    "Geometry enters both per-observation geometry tokens (cross-attention alongside vision) "
    "and a separate global MLP (AdaLN and temporal attention bias). Vision does not enter "
    "the global MLP. The previously quoted 37,632 is flattened visual scalar dimensionality, "
    "not a token count; probe feature_dimensions records the measured dimensions. "
    "A poor full F_cond ridge readout alone cannot establish visual drowning. "
    "The local 7D signed OCT MLP residual enters AdaLN and final modulation only; "
    "geometry tokens and temporal-bias conditioning retain the original global vector. "
    "stage4 symlog adds residuals to the original linear geometry/global channels."
)


def dev20_gate(value):
    return bool(value.get("completed_targets") == 5 and (
        (value["final_mean_um"] < 58. and value["min_mean_um"] < 25.)
        or value["entered_20_um_count"] >= 2))


def selection():
    history = read(CHECKPOINT_DIR / "training_history.json")
    scores = {int(row["batches_seen"]): float(row["val_loss"])
              for row in history if int(row["batches_seen"]) in (2000, 4000)}
    if set(scores) != {2000, 4000} or not all(math.isfinite(x) for x in scores.values()):
        raise ValueError("missing/nonfinite validation scores for step2000/4000")
    step = min(scores, key=lambda k: (scores[k], -k))
    return CHECKPOINT_DIR / f"step_{step:06d}.pth", {
        "rule": "lowest held-out validation loss among step2000/4000; tie -> 4000",
        "validation_losses": scores, "selected_step": step}


def evaluation_metrics(destination, indices, summary=None):
    root = Path(destination) / "shards/tasks"
    targets = trajectory_table(root, indices)
    if any(row["status"] != "completed" for row in targets.values()):
        raise ValueError(f"incomplete target trajectories: {targets}")
    return {"targets": targets, "aggregate": aggregate(targets),
            "direction": {f"E{i:02d}": toward_and_flips(
                root / f"eval_E{i:02d}/raw/evaluation_results.json") for i in indices},
            "strict_success_count": None if summary is None else summary["strict_success_count"],
            "summary_path": str(Path(destination) / "summary.json")}


def preflight():
    from lerobot_training.navigation_overlay import file_sha256
    from scripts.research.a_axis_flip_common import gpu_check
    from scripts.research.homog_replay_common import identity
    assets = [DATA / "metadata.json", VAL / "metadata.json", OVERLAY, INIT, CONTROL,
              STAGE4, ROOT / "configs/v12_reachable_development_targets_0p35um.json",
              ROOT / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2/metadata.json"]
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"new run required: {OUTPUT}")
    for directory in (DATA, VAL, ROOT / "data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2"):
        for name in ("data/dataset.npz", "data/image_paths.json"):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            assets.append(path)
    from lerobot_training.train_diffusion_policy import RCMDataset, image_transform_for_dataset
    for directory in (DATA, VAL):
        dataset = RCMDataset(directory, transform=image_transform_for_dataset(directory),
            n_obs_steps=2, n_action_steps=16, include_velocity=True, include_previous_action=True)
        sample = dataset[0]
        if tuple(sample["observation.image"].shape) != (2, 3, 224, 224):
            raise ValueError(f"dataset image contract mismatch: {directory}")
        del sample, dataset
    control = matched_control()
    historical = trajectory_table(CONTROL_FAST5, FAST5)
    if any(row["status"] != "completed" for row in historical.values()):
        raise ValueError("baseline fast5 raw incomplete")
    # Construct the exact evaluation entry point, including NVIDIA EGL rendering.
    from lerobot_training.eval_policy import PolicyEvaluator
    from scripts.research.a_axis_flip_common import QUALITY
    evaluator = PolicyEvaluator(CONTROL, device="cuda", num_action_samples=1,
        n_action_execute_steps=1, flow_integration_steps=4, measure_inference=True,
        action_aggregation="median", controller_variant="feedback",
        rgbd_render_quality=QUALITY.copy())
    if evaluator.n_obs_steps != 2 or evaluator.n_action_steps != 16:
        raise ValueError("baseline observation/action horizon mismatch")
    if not evaluator.direct_224_preprocessing:
        raise ValueError("baseline training metadata did not enable direct224")
    del evaluator
    from rcm.scene_v9 import RCMSceneV9
    from rcm.strict_rgbd import StrictRGBDSensor
    from rcm.visual_servo import visual_servo_settings
    scene = RCMSceneV9()
    sensor = None
    try:
        settings = visual_servo_settings(scene)
        settings["perception_backend"] = "strict_rgbd"
        settings["rgbd_render_quality"] = QUALITY.copy()
        sensor = StrictRGBDSensor(scene, settings)
        observation = sensor.observe()
        render_shape = list(observation["rgb"].shape)
        if observation["rgb"].dtype != np.uint8:
            raise ValueError("EGL render dtype mismatch")
    finally:
        if sensor is not None:
            sensor.close()
        scene.close()
    return {"egl_rgb_shape": render_shape,"status": "passed", "timestamp": stamp(), "gpu": gpu_check(),
            "code_root": str(CODE_ROOT), "runtime_root": str(ROOT),
            "source_commit": subprocess.check_output(
                ["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True).strip(),
            "identity": identity(), "control": control,
            "asset_sha256": {str(p): file_sha256(p) for p in assets},
            "historical_baseline": "reference only; not reused as current-code evaluation",
            "protocol": {"targets": str(assets[6]), "fast5": FAST5, "seed": "42+index",
                         "steps": 1600, "period_s": .05, "driver": "g10",
                         "strict_threshold_um": .35, "encoder_frames": 3,
                         "stop_on_success": False, "geometric_takeover": False},
            "train_command": train_command(CHECKPOINT_DIR),
            "architecture": ARCHITECTURE_FINDING}


def write_report(metrics):
    lines = ["# Local OCT AdaLN v1", "", f"Workflow: {metrics.get('status')}", "",
             ARCHITECTURE_FINDING, "", "Historical baseline is a reference, not a "
             "provenance-matched reusable evaluation. Same targets/seeds/protocol are recorded "
             "in preflight.json; full evidence is in metrics.json.", "",
             f"Stages: {metrics.get('stages', {})}", "",
             f"Failure/partial reason: {metrics.get('reason', 'none')}", "",
             "## Probe 2 and 3", "",
             "| Model | Band | Feature | cosine | sign XYZ |", "|---|---|---|---:|---|"]
    for label, result in metrics.get("probes", {}).items():
        for band in ("under_60", "under_40"):
            for name in ("F_cond", "F_geom_cond"):
                value = result["probe2"][band]["features"][name]
                lines.append(f"| {label} | {band} | {name} | {value['mean_cosine']} | "
                             f"{value['sign_accuracy_xyz']} |")
        lines += ["", f"{label} K=8: `{result['probe3']['per_axis']}`", ""]
    lines += ["", "## Closed-loop metrics", "",
              "| Model | N | final mean um | min mean um | entered20 | sustained200 | strict |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in metrics.get("evaluations", {}).items():
        value = result["aggregate"]
        lines.append(f"| {name} | {value['completed_targets']} | {value['final_mean_um']} | "
                     f"{value['min_mean_um']} | {value['entered_20_um_count']} | "
                     f"{value['sustained_200_count']} | {result.get('strict_success_count')} |")
    lines += ["", "## Per-target hold and direction metrics", "",
              "| Model/target | status | min um | final um | entries20 | max cycles/s | "
              "hold200 | last200 max um | speed overall/late um/s | late ABC flip | toward<60 |",
              "|---|---|---:|---:|---:|---|---|---:|---|---|---|"]
    for name, result in metrics.get("evaluations", {}).items():
        for target, row in result["targets"].items():
            direction = result.get("direction", {}).get(target, {})
            if row["status"] != "completed":
                lines.append(f"| {name}/{target} | {row['status']} | | | | | | | | | |")
                continue
            lines.append(f"| {name}/{target} | completed | {row['min_error_um']} | "
                f"{row['final_error_um']} | {row['entries_le_20_um']} | "
                f"{row['max_consecutive_le_20_um_cycles']}/{row['max_consecutive_le_20_um_seconds']} | "
                f"{row['sustained_200_pass']} | {row['last_200_max_error_um']} | "
                f"{row['actual_speed_um_per_s_overall']}/{row['actual_speed_um_per_s_last_200']} | "
                f"{direction.get('late_abc_sign_flip_rate')} | {direction.get('toward_fraction_under_60')} |")
    lines += ["", "## Decision and next data", "", str(metrics.get("decision", "incomplete")), "",
              "For entries without 200-cycle hold, next data must cover the last 200–600 "
              "cycles of approach, braking and repeated exit/re-entry: signed OCT, velocity, "
              "previous_action, accepted ABC commands and motor execution speed. Inspect "
              "per-target max hold and last200 maximum before selecting any new training.", "",
              "Step0 full historical hold table follows. Missing/short raw stays explicit.", ""]
    historical = OUTPUT / "step1_existing_20um_metrics.md"
    if historical.is_file():
        lines.append(historical.read_text())
    lines += ["", "Software/MuJoCo simulation only; no hardware accuracy claim."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def publish(metrics, deadline):
    # Explicit allowlist; never stage checkpoints, raw trajectories, or the runtime tree.
    target = CODE_ROOT / "runs/20261125/local_oct_adaln_v1"
    target.mkdir(parents=True, exist_ok=False)
    names = ["report.md", "metrics.json", "preflight.json", "workflow_status.json",
             "step1_existing_20um_metrics.md", "step1_existing_20um_metrics.json"]
    for name in names:
        source = OUTPUT / name
        if source.is_file():
            shutil.copyfile(source, target / name)
    shutil.copyfile(ROOT / "TEST_LOG.md", CODE_ROOT / "TEST_LOG.md")
    paths = [str((target / name).relative_to(SOURCE_REPO)) for name in names
             if (target / name).is_file()] + ["rcm_4090_deploy/TEST_LOG.md"]
    git = ["git", "-C", str(SOURCE_REPO)]
    subprocess.run(git + ["add", "-f", "--", *paths], check=True, timeout=30)
    staged = subprocess.check_output(git + ["diff", "--cached", "--name-only"], text=True).splitlines()
    if set(staged) != set(paths):
        raise ValueError(f"unexpected staged files: {staged}")
    subprocess.run(git + ["-c", "user.name=jsonyang218-pixel", "-c",
        "user.email=jsonyang218-pixel@users.noreply.github.com", "commit", "-m",
        "Record local OCT AdaLN experiment results"], check=True, timeout=30)
    command = ["env", "GIT_SSH_COMMAND=ssh -i /home/him1/.ssh/rcm_github_deploy_ed25519 "
               "-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20"] + git + [
                   "push", "-u", "origin", BRANCH]
    run_managed(command, OUTPUT / "publication.log", deadline)
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    return {"status": "pushed", "branch": BRANCH, "commit": commit}


def run():
    # Do not reuse or overwrite previous runs, including partial failures.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    hard_deadline = started + HARD_MINUTES * 60
    work_deadline = hard_deadline - 5 * 60
    metrics = {"status": "running", "started_at": stamp(), "stages": {},
               "probes": {}, "evaluations": {}, "architecture": ARCHITECTURE_FINDING,
               "step2": {"deterministic_x0": "skipped; sampler exposes no clean x0 input"}}
    current = "preflight"

    def save():
        metrics["elapsed_minutes"] = elapsed(started)
        write_json(OUTPUT / "metrics.json", metrics)
        write_json(OUTPUT / "workflow_status.json", {
            "status": metrics["status"], "stage": current,
            "started_at": metrics["started_at"], "elapsed_minutes": elapsed(started),
            "hard_stop_epoch": hard_deadline, "stages": metrics["stages"]})

    def stage(name, function, minutes=None):
        nonlocal current
        if remaining(started) < 20:
            raise TimeoutError("card: less than 20 minutes remain; no new stage")
        current = name
        metrics["stages"][name] = {"status": "running", "started_at": stamp()}
        save()
        deadline = min(work_deadline, time.time() + minutes * 60) if minutes else work_deadline
        try:
            value = function(deadline)
        except BaseException as error:
            metrics["stages"][name].update(status="interrupted" if isinstance(
                error, (KeyboardInterrupt, TimeoutError)) else "failed", reason=repr(error))
            record(name, metrics["stages"][name]["status"], repr(error))
            save()
            raise
        metrics["stages"][name].update(status="completed", ended_at=stamp())
        # Persist each result before attempting any dependent work.
        write_json(OUTPUT / "stage_results" / (name + ".json"), value)
        record(name, "completed", f"result={OUTPUT / 'stage_results' / (name + '.json')}; "
               f"train_data={DATA}; val_data={VAL}; init={INIT}; baseline={CONTROL}; "
               f"seeds=42+target_index; train_seed=20261118; steps=1600; period=.05s; "
               f"strict=.35um+encoder3; stage_result_summary={str(value)[:1800]}")
        save()
        return value

    try:
        preflight_path = ROOT / "runs/20261125/local_oct_adaln_preflight_v1/launch_preflight.json"
        checked = read(preflight_path)
        from scripts.research.homog_replay_common import identity
        if checked["status"] != "passed" or checked["identity"] != identity():
            raise ValueError("code/scene identity changed since preflight")
        write_json(OUTPUT / "preflight.json", checked)
        metrics["control"] = checked["control"]
        stage("step0_existing_hold", lambda deadline: (
            run_managed(env_command() + ["-m", "scripts.research.eval_20um_hold_metrics",
                "--output-dir", str(OUTPUT)], OUTPUT / "step0_hold.log", deadline),
            read(OUTPUT / "step1_existing_20um_metrics.json"))[1], 5)
        for label, checkpoint in (("baseline", CONTROL), ("stage4_symlog", STAGE4)):
            metrics["probes"][label] = stage("step0_probe_" + label,
                lambda deadline, cp=checkpoint, tag=label: probe(cp, OUTPUT / ("probe_" + tag), deadline), 10)
        metrics["smoke"] = stage("two_batch_smoke", run_smoke, 10)
        receipt = stage("matched_train4000", train, 80)
        checkpoint = CHECKPOINT_DIR / "step_004000.pth"
        config = read(CHECKPOINT_DIR / "training_config.json")
        if (receipt["status"] != "completed" or receipt["total_batches"] != 4000
                or not config["use_local_oct_branch"] or config.get("oct_symlog_enabled", False)
                or not checkpoint.is_file() or not (CHECKPOINT_DIR / "step_002000.pth").is_file()):
            raise ValueError("4000-step matched training contract failed")
        metrics["training"] = {"receipt": receipt, "checkpoint": str(checkpoint), "config": config}
        metrics["probes"]["local_oct_adaln"] = stage("step1_probe_local",
            lambda deadline: probe(checkpoint, OUTPUT / "probe_local", deadline), 10)
        dest = OUTPUT / "eval_fast5_step4000_k1"
        summary = stage("step1_fast5", lambda deadline: evaluate(checkpoint, dest, FAST5, 5,
            "median", OUTPUT / "fast5_k1.log", deadline), 25)
        metrics["evaluations"]["local4000_k1"] = evaluation_metrics(dest, FAST5, summary)
        baseline_targets = trajectory_table(CONTROL_FAST5, FAST5)
        metrics["evaluations"]["historical_baseline_fast5"] = {
            "targets": baseline_targets, "aggregate": aggregate(baseline_targets),
            "direction": {f"E{i:02d}": toward_and_flips(CONTROL_FAST5 /
                f"eval_E{i:02d}/raw/evaluation_results.json") for i in FAST5}}
        symlog_root = ROOT / "runs/20261125/stage4_1_oct_symlog/eval_fast5/step4000/shards/tasks"
        symlog_targets = trajectory_table(symlog_root, FAST5)
        metrics["evaluations"]["historical_stage4_fast5"] = {
            "targets": symlog_targets, "aggregate": aggregate(symlog_targets),
            "direction": {f"E{i:02d}": toward_and_flips(symlog_root /
                f"eval_E{i:02d}/raw/evaluation_results.json") for i in FAST5}}
        save()
        winner, selection_info = selection()
        metrics["step2"]["selection"] = selection_info
        # The card permits E06/E10 for time-constrained variance checks.
        indices = FAST5 if remaining(started) >= 55 else (6, 10)
        k1 = metrics["evaluations"]["local4000_k1"]
        if winner != checkpoint:
            dest = OUTPUT / "eval_step2_selected_k1"
            summary = stage("step2_selected_k1", lambda deadline: evaluate(winner, dest,
                indices, min(5, len(indices)), "median", OUTPUT / "step2_k1.log", deadline), 25)
            k1 = evaluation_metrics(dest, indices, summary)
            metrics["evaluations"]["selected_k1"] = k1
        metrics["step2"]["k1_reference"] = k1
        dest = OUTPUT / "eval_step2_selected_k4_mean"
        summary = stage("step2_k4_mean", lambda deadline: evaluate(winner, dest, indices,
            min(5, len(indices)), "mean", OUTPUT / "step2_k4.log", deadline, samples=4), 25)
        k4 = evaluation_metrics(dest, indices, summary)
        metrics["evaluations"]["selected_k4_mean"] = k4
        candidates = [(checkpoint, 1, "median", metrics["evaluations"]["local4000_k1"]),
                      (winner, 1, "median", k1), (winner, 4, "mean", k4)]
        passed = [entry for entry in candidates if dev20_gate(entry[3]["aggregate"])]
        metrics["dev20_gate"] = {"passed": bool(passed), "remaining_minutes": remaining(started),
            "rule": "complete fast5 AND ((final<58 AND min<25) OR entered20>=2); remaining>=40"}
        if passed and remaining(started) >= 40:
            cp, samples, aggregation, _ = min(passed, key=lambda entry: entry[3]["aggregate"]["final_mean_um"])
            dest = OUTPUT / "eval_dev20"
            summary = stage("step3_dev20", lambda deadline: evaluate(cp, dest, DEV20, 6,
                aggregation, OUTPUT / "dev20.log", deadline, samples=samples))
            metrics["evaluations"]["dev20"] = evaluation_metrics(dest, DEV20, summary)
        else:
            metrics["stages"]["step3_dev20"] = {"status": "skipped_gate_or_time",
                                                     **metrics["dev20_gate"]}
        local = metrics["evaluations"]["local4000_k1"]["aggregate"]
        base = metrics["evaluations"]["historical_baseline_fast5"]["aggregate"]
        metrics["decision"] = {"metric_improvement_vs_historical": {
            "final_mean_um_delta": local["final_mean_um"] - base["final_mean_um"],
            "min_mean_um_delta": local["min_mean_um"] - base["min_mean_um"]},
            "fast5_gate_pass": dev20_gate(local),
            "entered20": local["entered_20_um_count"], "sustained200": local["sustained_200_count"],
            "strict_success": metrics["evaluations"]["local4000_k1"]["strict_success_count"],
            "ab_sample_flip_change": {axis: metrics["probes"]["local_oct_adaln"]["probe3"]["per_axis"][axis][
                "fraction_observations_with_sign_flip_action0"] - metrics["probes"]["baseline"]["probe3"]["per_axis"][axis][
                "fraction_observations_with_sign_flip_action0"] for axis in "AB"}}
        metrics["status"] = "completed"
    except BaseException as error:
        metrics["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed"
        metrics["reason"] = repr(error)
        for name in ("two_batch_smoke", "matched_train4000", "step1_probe_local", "step1_fast5",
                     "step2_k4_mean", "step3_dev20"):
            metrics["stages"].setdefault(name, {"status": "skipped_dependency_or_time"})
        traceback.print_exc()
    finally:
        current = "report"
        save()
        try:
            write_report(metrics)
        except BaseException as error:
            metrics["report_error"] = repr(error)
        record("workflow", metrics["status"], f"stages={metrics['stages']}; metrics={OUTPUT / 'metrics.json'}")
        save()
        try:
            metrics["publication"] = publish(metrics, hard_deadline)
        except BaseException as error:
            metrics["publication"] = {"status": "failed", "reason": repr(error)}
            record("publication", "failed", repr(error))
            traceback.print_exc()
        save()
        print(f"FINAL {metrics['status']}: {OUTPUT / 'report.md'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--output-dir", type=Path, required=True)
    worker = sub.add_parser("_evaluate")
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--destination", type=Path, required=True)
    worker.add_argument("--indices", required=True)
    worker.add_argument("--workers", type=int, default=5)
    worker.add_argument("--samples", type=int, default=1)
    worker.add_argument("--action-aggregation", choices=("median", "mean"), default="median")
    args = parser.parse_args(argv)
    if args.action == "preflight":
        result = preflight()
        write_json(args.output_dir / "launch_preflight.json", result)
        print(json.dumps({"status": "passed", "output": str(args.output_dir)}, indent=2))
    elif args.action == "run":
        if args.output_dir.resolve() != OUTPUT:
            raise ValueError(f"authorized output is {OUTPUT}")
        def stop(signum, frame):
            raise KeyboardInterrupt(signal.Signals(signum).name)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, stop)
        run()
    else:
        _evaluate_worker(args)


if __name__ == "__main__":
    main()
