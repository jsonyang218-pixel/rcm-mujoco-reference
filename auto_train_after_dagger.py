"""Wait for guided DAgger shards, prepare data, then train automatically."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parent
SHARDS = (
    "shard1",
    "shard1_rem_a",
    "shard1_rem_b",
    "shard1_rem_c",
    "shard2",
    "shard3",
)


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_status(status_path, **values):
    write_json_atomic(
        status_path,
        {
            "updated_unix_s": time.time(),
            **values,
        },
    )


def pid_exists(pid):
    if pid is None:
        return True
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        int(pid),
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def validate_manifest(path: Path):
    manifest = load_json(path)
    required = (
        "total_frames",
        "episodes",
        "policy_guided_oct",
        "policy_checkpoint_epoch",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"{path} lacks fields: {missing}")
    if not manifest["policy_guided_oct"]:
        raise ValueError(f"{path} was not collected with a guided policy")
    if int(manifest["policy_checkpoint_epoch"]) != 3:
        raise ValueError(
            f"{path} used checkpoint epoch "
            f"{manifest['policy_checkpoint_epoch']}, expected 3"
        )
    frames = int(manifest["total_frames"])
    episode_frames = sum(int(row["frames"]) for row in manifest["episodes"])
    if frames <= 0 or frames != episode_frames:
        raise ValueError(
            f"{path} frame counts are inconsistent: "
            f"{frames} != {episode_frames}"
        )
    return manifest


def count_valid_teacher_rows(shard_dir: Path):
    count = 0
    for path in sorted(shard_dir.glob("episode_*/dagger_samples.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and bool(json.loads(line).get("teacher_valid")):
                    count += 1
    return count


def run_stage(command, stage, status_path, log_path, **status_values):
    print(f"\n=== {stage} ===", file=sys.stdout, flush=True)
    print(" ".join(str(value) for value in command), flush=True)
    set_status(
        status_path,
        state="running",
        stage=stage,
        command=[str(value) for value in command],
        **status_values,
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(f"\n=== {stage} ===\n")
        stream.write(" ".join(str(value) for value in command) + "\n")
        stream.flush()
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {completed.returncode}"
        )


def wait_for_shards(
    collect_root,
    collector_pids,
    status_path,
    timeout_s,
    poll_s,
):
    deadline = time.monotonic() + timeout_s
    while True:
        ready = [
            name
            for name in SHARDS
            if (collect_root / name / "manifest.json").is_file()
        ]
        set_status(
            status_path,
            state="waiting_for_collection",
            shards_ready=ready,
            shards_missing=[
                name for name in SHARDS if name not in ready
            ],
        )
        if len(ready) == len(SHARDS):
            break
        for name, pid in zip(SHARDS, collector_pids):
            if name not in ready and not pid_exists(pid):
                raise RuntimeError(
                    f"{name} collector process {pid} exited without manifest"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for DAgger shards after {timeout_s}s"
            )
        time.sleep(poll_s)
    manifests = {
        name: validate_manifest(collect_root / name / "manifest.json")
        for name in SHARDS
    }
    return manifests


def validate_converted(shard_dir, converted_dir):
    expected = count_valid_teacher_rows(shard_dir)
    with np.load(
        converted_dir / "data" / "dataset.npz",
        allow_pickle=False,
    ) as data:
        converted = int(len(data["action"]))
        if not np.all(np.isfinite(data["action"])):
            raise ValueError(f"{converted_dir} contains non-finite actions")
    if converted != expected:
        raise ValueError(
            f"{converted_dir} has {converted} frames, but source contains "
            f"{expected} valid teacher labels"
        )
    return {
        "valid_teacher_rows": expected,
        "converted_frames": converted,
    }


def validate_prepared(merged_dir):
    with np.load(
        merged_dir / "data" / "dataset.npz",
        allow_pickle=False,
    ) as data:
        count = int(len(data["action"]))
    base = np.load(
        merged_dir / "data" / "guided_base_action.npy",
        allow_pickle=False,
    )
    limit = np.load(
        merged_dir / "data" / "guided_residual_limit.npy",
        allow_pickle=False,
    )
    if base.shape != (count, 3) or limit.shape != (count, 3):
        raise ValueError("guided sidecar shapes do not match merged actions")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(limit)):
        raise ValueError("guided sidecars contain non-finite values")
    if np.any(limit < 0):
        raise ValueError("guided residual limits contain negative values")
    return {
        "merged_frames": count,
        "base_shape": list(base.shape),
        "limit_shape": list(limit.shape),
    }


def build_plan(args):
    converted = [
        args.data_root / f"dagger_guided_v6_round1_{name}"
        for name in SHARDS
    ]
    dagger_merged_640 = args.data_root / "dagger_guided_v6_round1_all_640"
    dagger_cache_224 = args.data_root / "dagger_guided_v6_round1_all_224"
    merged = args.data_root / "lerobot_rcm_train_guided_dagger1_full_224"
    return {
        "collect_root": args.collect_root,
        "converted": converted,
        "dagger_merged_640": dagger_merged_640,
        "dagger_cache_224": dagger_cache_224,
        "merged": merged,
        "training_output": args.training_output,
        "validation_data": args.validation_data,
        "initial_checkpoint": args.initial_checkpoint,
    }


def serializable_plan(plan):
    return {
        "collect_root": str(plan["collect_root"]),
        "converted": [str(path) for path in plan["converted"]],
        "dagger_merged_640": str(plan["dagger_merged_640"]),
        "dagger_cache_224": str(plan["dagger_cache_224"]),
        "merged": str(plan["merged"]),
        "training_output": str(plan["training_output"]),
        "validation_data": str(plan["validation_data"]),
        "initial_checkpoint": str(plan["initial_checkpoint"]),
    }


def check_inputs(args, plan):
    if not (args.base_train_data / "data" / "dataset.npz").is_file():
        raise FileNotFoundError(args.base_train_data)
    if not (args.validation_data / "data" / "dataset.npz").is_file():
        raise FileNotFoundError(args.validation_data)
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(args.initial_checkpoint)
    if not (ROOT / "collect_dagger_dataset.py").is_file():
        raise FileNotFoundError(ROOT / "collect_dagger_dataset.py")
    if args.reuse_prepared:
        if not (plan["merged"] / "data" / "dataset.npz").is_file():
            raise FileNotFoundError(plan["merged"])
        if not (
            plan["merged"] / "data" / "guided_base_action.npy"
        ).is_file():
            raise FileNotFoundError(
                plan["merged"] / "data" / "guided_base_action.npy"
            )
    else:
        for path in [
            plan["dagger_merged_640"],
            plan["dagger_cache_224"],
            plan["merged"],
        ]:
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing pipeline output: {path}"
                )
    for path in [plan["training_output"]]:
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing pipeline output: {path}"
            )
    if not args.reuse_prepared and not args.reuse_converted:
        for path in plan["converted"]:
            if path.exists():
                raise FileExistsError(
                    f"converted output exists without --reuse-converted: {path}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-root",
        type=Path,
        default=ROOT / "runs" / "dagger_round1_guided_v6_20261105",
    )
    parser.add_argument(
        "--collector-pid",
        type=int,
        nargs="+",
        default=None,
        metavar="PID",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
    )
    parser.add_argument(
        "--base-train-data",
        type=Path,
        default=ROOT / "data" / "lerobot_rcm_train_recovery_full_224",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=ROOT / "data" / "lerobot_rcm_val_recovery_full_224",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "diffusion_policy_v6_guided_oct_finetune"
            / "best_model.pth"
        ),
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "diffusion_policy_v7_guided_dagger1_cached"
        ),
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--reuse-converted",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reuse-prepared",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip collection/conversion/merge and train from prepared data.",
    )
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.collect_root = args.collect_root.resolve()
    args.data_root = args.data_root.resolve()
    args.base_train_data = args.base_train_data.resolve()
    args.validation_data = args.validation_data.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    args.training_output = args.training_output.resolve()
    plan = build_plan(args)
    check_inputs(args, plan)
    if args.check_only:
        print(json.dumps({"result": "PASS", "plan": {
            **serializable_plan(plan),
        }}, ensure_ascii=False, indent=2))
        return 0
    if args.epochs < 1 or args.poll_seconds < 1 or args.timeout_hours <= 0:
        parser.error("epochs, poll-seconds and timeout-hours must be positive")
    if (
        args.collector_pid is not None
        and len(args.collector_pid) != len(SHARDS)
    ):
        parser.error(
            "collector-pid must provide one PID per expected source"
        )

    run_root = args.collect_root
    status_path = run_root / "pipeline_status.json"
    log_path = run_root / "pipeline.log"
    lock_path = run_root / "pipeline.lock"
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        try:
            lock_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            lock_pid = None
        if lock_pid is None or pid_exists(lock_pid):
            raise RuntimeError(
                f"pipeline lock is held by a live or unreadable process: "
                f"{lock_path}"
            ) from exc
        lock_path.unlink()
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()) + "\n")

    try:
        set_status(
            status_path,
            state="starting",
            stage="waiting_for_collection",
            plan=serializable_plan(plan),
        )
        if args.reuse_prepared:
            source_frames = {}
            conversion_records = {}
        else:
            collector_pids = args.collector_pid or [None] * len(SHARDS)
            manifests = wait_for_shards(
                args.collect_root,
                collector_pids,
                status_path,
                int(args.timeout_hours * 3600),
                args.poll_seconds,
            )
            source_frames = {
                name: int(manifest["total_frames"])
                for name, manifest in manifests.items()
            }
            set_status(
                status_path,
                state="running",
                stage="converting",
                source_frames=source_frames,
            )
            conversion_records = {}
            for name, converted_dir in zip(SHARDS, plan["converted"]):
                if not (
                    args.reuse_converted
                    and (converted_dir / "data" / "dataset.npz").is_file()
                    and (converted_dir / "metadata.json").is_file()
                ):
                    run_stage(
                        [
                            sys.executable,
                            ROOT / "lerobot_training" / "convert_to_lerobot.py",
                            "--input-dir",
                            args.collect_root / name,
                            "--output-dir",
                            converted_dir,
                            "--action-source",
                            "teacher",
                            "--image-workers",
                            "8",
                        ],
                        f"convert_{name}",
                        status_path,
                        log_path,
                    )
                conversion_records[name] = validate_converted(
                    args.collect_root / name,
                    converted_dir,
                )
            run_stage(
                [
                    sys.executable,
                    ROOT / "lerobot_training" / "merge_lerobot_datasets.py",
                    "--base-dir",
                    plan["converted"][0],
                    "--additional-dir",
                    *plan["converted"][1:],
                    "--output-dir",
                    plan["dagger_merged_640"],
                ],
                "merge_dagger_sources_640",
                status_path,
                log_path,
            )
            run_stage(
                [
                    sys.executable,
                    ROOT / "lerobot_training" / "prepare_image_cache.py",
                    "--input-dir",
                    plan["dagger_merged_640"],
                    "--output-dir",
                    plan["dagger_cache_224"],
                    "--size",
                    "224",
                    "--workers",
                    "12",
                ],
                "cache_dagger_images_224",
                status_path,
                log_path,
            )
            run_stage(
                [
                    sys.executable,
                    ROOT / "lerobot_training" / "merge_lerobot_datasets.py",
                    "--base-dir",
                    args.base_train_data,
                    "--additional-dir",
                    plan["dagger_cache_224"],
                    "--output-dir",
                    plan["merged"],
                ],
                "merge_recovery_and_dagger_224",
                status_path,
                log_path,
            )
            run_stage(
                [
                    sys.executable,
                    ROOT / "lerobot_training" / "prepare_guided_oct_base.py",
                    "--data-dir",
                    plan["merged"],
                ],
                "prepare_guided_oct_sidecars",
                status_path,
                log_path,
            )
        prepared = validate_prepared(plan["merged"])
        run_stage(
            [
                sys.executable,
                ROOT / "lerobot_training" / "prepare_vision_feature_cache.py",
                "--data-dir",
                plan["merged"],
                "--data-dir",
                args.validation_data,
                "--checkpoint",
                args.initial_checkpoint,
                "--batch-size",
                "256",
                "--workers",
                "12",
            ],
            "prepare_frozen_vision_feature_cache",
            status_path,
            log_path,
        )
        run_stage(
            [
                sys.executable,
                ROOT / "lerobot_training" / "train_oct_policy.py",
                "--data-dir",
                plan["merged"],
                "--val-data-dir",
                args.validation_data,
                "--output-dir",
                plan["training_output"],
                "--init-from",
                args.initial_checkpoint,
                "--guided-oct",
                "--vision-feature-cache",
                "--epochs",
                str(args.epochs),
                "--batch-size",
                "48",
                "--val-batch-size",
                "32",
                "--thread-workers",
                "12",
                "--freeze-vision-encoder",
                "--lr",
                "0.00005",
                "--seed",
                "20261106",
                "--log-file",
                plan["training_output"] / "training.log",
            ],
            "train_guided_dagger1",
            status_path,
            log_path,
        )
        set_status(
            status_path,
            state="complete",
            stage="training_complete",
            source_frames=source_frames,
            conversions=conversion_records,
            prepared=prepared,
            training_output=str(plan["training_output"]),
        )
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "training_output": str(plan["training_output"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        set_status(
            status_path,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        with log_path.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        return 1
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
