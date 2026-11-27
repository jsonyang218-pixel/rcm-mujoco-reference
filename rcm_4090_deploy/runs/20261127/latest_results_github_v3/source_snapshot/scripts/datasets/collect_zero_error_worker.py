"""Collect a disjoint v12 episode range directly into an existing run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from rcm.reference import HERE
from scripts.datasets.collect_zero_error_dataset import (
    collect_episode,
    load_source_episodes,
)


def write_jsonl_append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, required=True)
    parser.add_argument("--end-episode", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--max-visual-iterations", type=int, default=1600)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="exit after this many newly collected episodes",
    )
    parser.add_argument(
        "--control-resolution",
        nargs=2,
        type=int,
        default=[1920, 1440],
    )
    args = parser.parse_args(argv)
    if not 0 <= args.start_episode <= args.end_episode:
        parser.error("invalid episode range")
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")

    source = load_source_episodes(args.source_dataset)
    if args.end_episode >= len(source):
        parser.error("episode range exceeds source dataset")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "collection_protocol.json"
    config_path = output_dir / "config_snapshot.json"
    if not protocol_path.is_file() or not config_path.is_file():
        parser.error("worker requires an initialized v12 output directory")
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config["motor_execution"] = {
        **config.get("motor_execution", {}),
        "enabled": True,
        "control_period_s": float(
            config["navigation"]["visual_servo"]["sample_period_s"]
        ),
    }
    config["navigation"]["visual_servo"]["perception_backend"] = "strict_rgbd"
    config["reference_scene"]["eye_camera"]["resolution_px"] = list(
        args.control_resolution
    )
    if json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("worker effective config differs from the run snapshot")
    log_dir = output_dir / "worker_logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / (
        f"worker_{args.start_episode:06d}_{args.end_episode:06d}.jsonl"
    )
    started = time.monotonic()
    collected = 0
    for index in range(args.start_episode, args.end_episode + 1):
        result_path = output_dir / f"episode_{index:06d}" / "episode_result.json"
        episode_dir = output_dir / f"episode_{index:06d}"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                int(result.get("episode_index", -1)) != index
                or result.get("result") != "PASS"
                or result.get("strict_zero_error") is not True
                or result.get("final_error_um") != 0.0
            ):
                raise ValueError(f"existing episode {index} is not a completed PASS")
            write_jsonl_append(log, {
                "episode_index": index,
                "frames": result["frames"],
                "visual_servo_result": result["visual_servo_result"],
                "fallback_used": result["fallback_used"],
                "result": "SKIP_EXISTING_PASS",
                "wall_s": time.monotonic() - started,
            })
            continue
        if episode_dir.exists():
            raise FileExistsError(
                f"incomplete episode directory requires explicit recovery: {episode_dir}"
            )
        episode = collect_episode(
            config,
            output_dir,
            source[index],
            args.seed + index,
            args.max_visual_iterations,
        )
        collected += 1
        write_jsonl_append(log, {
            "episode_index": index,
            "frames": episode["frames"],
            "visual_servo_result": episode["visual_servo_result"],
            "fallback_used": episode["fallback_used"],
            "result": episode["result"],
            "wall_s": time.monotonic() - started,
        })
        print(json.dumps({
            "worker_start": args.start_episode,
            "worker_end": args.end_episode,
            "episode_complete": index,
            "frames": episode["frames"],
            "visual_servo_result": episode["visual_servo_result"],
            "fallback_used": episode["fallback_used"],
        }, ensure_ascii=False), flush=True)
        if args.max_episodes is not None and collected >= args.max_episodes:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
