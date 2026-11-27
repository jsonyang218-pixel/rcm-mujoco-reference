"""Run disjoint v12 episode ranges with one atomic canonical manifest.

Every worker writes into the single official run directory and never changes
the manifest.  The coordinator advances the manifest only when a contiguous
prefix of complete, strict-zero episode results exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from rcm.reference import HERE
from rcm.waypoint_dataset import write_manifest
from scripts.datasets.collect_zero_error_dataset import load_source_episodes
from scripts.datasets.convert_zero_error_to_lerobot import convert
from scripts.verification.verify_zero_error_dataset import (
    verify_run,
    verify_training_dataset,
)


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed_episode(output_dir, index):
    episode_dir = Path(output_dir) / f"episode_{index:06d}"
    result_path = episode_dir / "episode_result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        int(result.get("episode_index", -1)) != index
        or result.get("result") != "PASS"
        or result.get("strict_zero_error") is not True
        or result.get("final_error_um") != 0.0
    ):
        return None
    for key in ("data", "jsonl", "policy_inputs", "frame_audit", "servo_report"):
        relative = result.get(key)
        if relative is None:
            return None
        candidate = (
            Path(output_dir) / "dataset" / relative
            if key in ("data", "jsonl", "policy_inputs")
            else Path(output_dir) / relative
        )
        if not candidate.is_file():
            return None
    return result


def _contiguous_episodes(output_dir, limit):
    episodes = []
    for index in range(limit):
        result = _completed_episode(output_dir, index)
        if result is None:
            break
        episodes.append(result)
    return episodes


def _write_manifest_atomic(output_dir, episodes, sample_period_s):
    with tempfile.TemporaryDirectory(prefix="rcm_v12_manifest_") as temporary:
        write_manifest(
            temporary,
            episodes,
            [224, 224, 3],
            sample_period_s,
            "visual_current_target",
        )
        manifest = json.loads(
            (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
        )
    manifest.update({
        "dataset_version": "v12",
        "strict_zero_error": True,
        "execution_backend": "MotorBackend",
        "perception_backend": "strict_rgbd",
        "action_semantics": (
            "quantized MotorBackend absolute A/B/C target for the following "
            "control interval; never the next actual state"
        ),
        "target_semantics": (
            "bit-exact stabilized source-terminal needle-tip MuJoCo world position"
        ),
        "audit_path_base": "Parent of this dataset directory",
    })
    _write_json_atomic(Path(output_dir) / "dataset" / "manifest.json", manifest)


def _split_ranges(indices, workers):
    indices = sorted(int(value) for value in indices)
    if not indices:
        return []
    ranges = []
    count = min(workers, len(indices))
    base, remainder = divmod(len(indices), count)
    offset = 0
    for segment in range(count):
        size = base + (1 if segment < remainder else 0)
        chunk = indices[offset:offset + size]
        ranges.append((chunk[0], chunk[-1]))
        offset += size
    return ranges


def _quarantine_incomplete(output_dir, limit):
    quarantined = []
    dataset = Path(output_dir) / "dataset"
    for index in range(limit):
        episode_dir = Path(output_dir) / f"episode_{index:06d}"
        image_dir = dataset / "images" / f"episode_{index:06d}"
        partial_files = []
        for name in (
            "data",
            "policy_inputs",
            "policy_inputs_no_image",
            "action_labels",
        ):
            partial_files.extend(sorted(
                (dataset / name).glob(f"episode_{index:06d}.*")
            ))
        has_incomplete_episode = (
            episode_dir.exists()
            and not (episode_dir / "episode_result.json").is_file()
        )
        result_exists = (episode_dir / "episode_result.json").is_file()
        if not result_exists and (
            has_incomplete_episode or image_dir.exists() or partial_files
        ):
            destination = Path(output_dir) / (
                f"aborted_episode_{index:06d}_{int(time.time())}"
            )
            destination.mkdir(parents=True, exist_ok=False)
            moved = []
            if episode_dir.exists():
                episode_dir.replace(destination / "episode")
                moved.append("episode")
            if image_dir.exists():
                (destination / "dataset" / "images").mkdir(parents=True)
                image_dir.replace(
                    destination / "dataset" / "images" / image_dir.name
                )
                moved.append("dataset/images")
            for source in partial_files:
                relative = source.relative_to(dataset)
                target = destination / "dataset" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                moved.append(f"dataset/{relative.as_posix()}")
            quarantined.append({
                "episode_index": index,
                "source": str(episode_dir),
                "destination": str(destination),
                "moved": moved,
            })
    return quarantined


def _worker_command(args, start, end):
    return [
        sys.executable,
        "-m",
        "scripts.datasets.collect_zero_error_worker",
        "--source-dataset",
        str(args.source_dataset),
        "--output-dir",
        str(args.output_dir),
        "--start-episode",
        str(start),
        "--end-episode",
        str(end),
        "--seed",
        str(args.seed),
        "--max-visual-iterations",
        str(args.max_visual_iterations),
        "--max-episodes",
        "1",
        "--control-resolution",
        "1920",
        "1440",
    ]


def _validate_pilot(output_dir):
    path = Path(output_dir) / "pilot_gate.json"
    if not path.is_file():
        raise RuntimeError("bulk collection requires a passing pilot_gate.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    raw = report.get("raw_pilot", {})
    training = report.get("pilot_training_verification", {})
    if (
        report.get("result") != "PASS"
        or raw.get("result") != "PASS"
        or raw.get("episodes") != 1
        or raw.get("all_terminal_errors_exactly_zero") is not True
        or training.get("result") != "PASS"
    ):
        raise RuntimeError("pilot gate did not pass strict raw and 224 checks")


def _state(output_dir, total, workers, wave, completed, quarantined):
    return {
        "result": "RUNNING" if completed != total else "PASS",
        "completed_episodes": completed,
        "requested_episodes": total,
        "missing_episodes": total - completed,
        "workers": workers,
        "wave": wave,
        "quarantined_incomplete": quarantined,
        "updated_unix_s": time.time(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=HERE / "data" / "lerobot_rcm_train_guided_dagger1_full_224",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=608)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--max-visual-iterations", type=int, default=1600)
    parser.add_argument("--max-waves", type=int, default=32)
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.workers < 1 or args.max_waves < 1:
        parser.error("episodes, workers and max-waves must be positive")
    args.output_dir = args.output_dir.resolve()
    source = load_source_episodes(args.source_dataset)
    if args.episodes != len(source):
        parser.error(f"parallel completion requires all {len(source)} source episodes")
    if not (args.output_dir / "collection_protocol.json").is_file():
        parser.error("output directory is not an initialized v12 run")
    _validate_pilot(args.output_dir)
    sample_period_s = float(json.loads(
        (args.output_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )["navigation"]["visual_servo"]["sample_period_s"])

    log_dir = args.output_dir / "parallel_logs"
    log_dir.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "MUJOCO_GL": "egl",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    quarantined = []
    children = []
    output_streams = []
    started = time.monotonic()
    try:
        for wave in range(1, args.max_waves + 1):
            quarantined.extend(_quarantine_incomplete(
                args.output_dir, args.episodes
            ))
            episodes = _contiguous_episodes(args.output_dir, args.episodes)
            _write_manifest_atomic(args.output_dir, episodes, sample_period_s)
            missing = [
                index for index in range(args.episodes)
                if _completed_episode(args.output_dir, index) is None
            ]
            _write_json_atomic(
                args.output_dir / "parallel_collection.json",
                _state(
                    args.output_dir,
                    args.episodes,
                    args.workers,
                    wave,
                    len(episodes),
                    quarantined,
                ),
            )
            if not missing:
                break
            ranges = _split_ranges(missing, args.workers)
            print(json.dumps({
                "wave": wave,
                "completed_prefix": len(episodes),
                "missing": len(missing),
                "ranges": ranges,
            }, ensure_ascii=False), flush=True)
            children = []
            output_streams = []
            for start, end in ranges:
                log_path = log_dir / (
                    f"wave_{wave:02d}_{start:06d}_{end:06d}.log"
                )
                stream = log_path.open("w", encoding="utf-8")
                output_streams.append(stream)
                child = subprocess.Popen(
                    _worker_command(args, start, end),
                    cwd=HERE,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                children.append((child, start, end, log_path))
                time.sleep(0.5)

            last_count = -1
            while any(child.poll() is None for child, *_ in children):
                episodes = _contiguous_episodes(args.output_dir, args.episodes)
                if len(episodes) != last_count:
                    _write_manifest_atomic(
                        args.output_dir, episodes, sample_period_s
                    )
                    elapsed = time.monotonic() - started
                    print(json.dumps({
                        "completed_prefix": len(episodes),
                        "remaining": args.episodes - len(episodes),
                        "elapsed_s": round(elapsed, 1),
                    }, ensure_ascii=False), flush=True)
                    last_count = len(episodes)
                time.sleep(5.0)
            for child, start, end, log_path in children:
                returncode = child.wait()
                print(json.dumps({
                    "range": [start, end],
                    "returncode": returncode,
                    "log": str(log_path),
                }, ensure_ascii=False), flush=True)
            for stream in output_streams:
                stream.close()
            output_streams = []
        episodes = _contiguous_episodes(args.output_dir, args.episodes)
        if len(episodes) != args.episodes:
            missing = [
                index for index in range(args.episodes)
                if _completed_episode(args.output_dir, index) is None
            ]
            _write_json_atomic(
                args.output_dir / "parallel_collection.json",
                {
                    **_state(
                        args.output_dir,
                        args.episodes,
                        args.workers,
                        args.max_waves,
                        len(episodes),
                        quarantined,
                    ),
                    "result": "FAIL",
                    "failed_missing_episodes": missing,
                },
            )
            return 1

        _write_manifest_atomic(args.output_dir, episodes, sample_period_s)
        verification = verify_run(
            args.output_dir,
            source_dataset=args.source_dataset,
            expected_episodes=args.episodes,
            require_complete=True,
        )
        _write_json_atomic(
            args.output_dir / "dataset_verification.json", verification
        )
        if verification.get("result") != "PASS":
            raise RuntimeError("final raw dataset verification failed")
        if args.training_output_dir.exists():
            raise FileExistsError(
                f"training output already exists: {args.training_output_dir}"
            )
        conversion = convert(
            args.output_dir,
            args.training_output_dir,
            expected_episodes=args.episodes,
            require_complete=True,
        )
        _write_json_atomic(
            args.output_dir / "training_conversion.json", conversion
        )
        training_verification = verify_training_dataset(
            args.output_dir,
            args.training_output_dir,
            expected_episodes=args.episodes,
        )
        _write_json_atomic(
            args.output_dir / "training_verification.json",
            training_verification,
        )
        if training_verification.get("result") != "PASS":
            raise RuntimeError("final 224 dataset verification failed")
        report = {
            **_state(
                args.output_dir,
                args.episodes,
                args.workers,
                args.max_waves,
                len(episodes),
                quarantined,
            ),
            "result": "PASS",
            "elapsed_s": time.monotonic() - started,
            "verification": verification,
            "training_conversion": conversion,
            "training_verification": training_verification,
        }
        _write_json_atomic(
            args.output_dir / "parallel_collection.json", report
        )
        print(json.dumps({
            "result": "PASS",
            "episodes": args.episodes,
            "frames": verification["frames"],
            "fallback_episodes": verification["fallback_episodes"],
            "training_dir": str(args.training_output_dir),
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as error:
        for child, *_ in children:
            if child.poll() is None:
                child.terminate()
        for child, *_ in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        _write_json_atomic(
            args.output_dir / "parallel_collection.json",
            {
                "result": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_episodes": len(_contiguous_episodes(
                    args.output_dir, args.episodes
                )),
                "quarantined_incomplete": quarantined,
                "updated_unix_s": time.time(),
            },
        )
        raise
    finally:
        for stream in output_streams:
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
