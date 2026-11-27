#!/usr/bin/env python3
"""Convert and merge the final accepted plan-2 training trajectories."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "pilot_d1_primary": "data/experience_replay_plan2_20261118_retry2/raw/d1_pilot_primary",
    "pilot_d1_extra": "data/experience_replay_plan2_20261118_retry2/raw/d1_pilot_extra",
    "pilot_d2_overshoot": "data/experience_replay_plan2_20261118_retry2/raw/d2_pilot_overshoot",
    "pilot_d2_oscillation": "data/experience_replay_plan2_20261118_retry2/raw/d2_pilot_oscillation",
    "pilot_d4_full": "data/experience_replay_plan2_20261118_retry2/raw/d4_pilot_full",
    "pilot_d3_hold": "data/experience_replay_plan2_d3_pilot_beta1_20261119/raw/d3_pilot_hold_beta1",
    "pilot_d3_perturb": "data/experience_replay_plan2_d3_pilot_beta1_20261119/raw/d3_pilot_perturb_hold_beta1",
    "pilot_d3_perturb_retry": "data/experience_replay_plan2_d3_pilot_beta1_20261119/raw/d3_pilot_perturb_hold_beta1_retry1",
    "d1_300_1000": "data/experience_replay_plan2_20261119_remaining1/raw/d1_300_1000um_remaining",
    "d1_100_300": "data/experience_replay_plan2_20261119_remaining1/raw/d1_100_300um_remaining",
    "d1_30_100": "data/experience_replay_plan2_20261119_remaining1/raw/d1_30_100um_remaining",
    "d1_under_30": "data/experience_replay_plan2_20261119_remaining1/raw/d1_under_30um_remaining",
    "d2_overshoot": "data/experience_replay_plan2_20261119_remaining1/raw/d2_overshoot_remaining",
    "d2_oscillation": "data/experience_replay_plan2_20261119_remaining1/raw/d2_oscillation_remaining",
    "d3_hold": "data/experience_replay_plan2_20261119_remaining1/raw/d3_hold_remaining",
    "d3_perturb": "data/experience_replay_plan2_20261119_remaining1/raw/d3_perturb_hold_remaining",
    "d3_perturb_retry_a": "data/experience_replay_plan2_20261119_remaining1/raw/d3_perturb_hold_retry_a",
    "d3_perturb_retry_b": "data/experience_replay_plan2_20261119_remaining1/raw/d3_perturb_hold_retry_b",
    "d4_full_original": "data/experience_replay_plan2_20261119_remaining1/raw/d4_full_remaining",
    "d4_full_shard_b": "data/experience_replay_plan2_20261119_remaining1/raw/d4_full_remaining_shard_b",
}


def prepare_d4_shard_a(clean_root):
    """Exclude the first duplicate D4 target without modifying source data."""
    destination = clean_root / "d4_full_shard_a_unique"
    if destination.exists():
        return destination
    source = (
        PROJECT_ROOT
        / "data/experience_replay_plan2_20261119_remaining1/raw"
        / "d4_full_remaining_shard_a"
    )
    images = destination / "dataset" / "images"
    images.mkdir(parents=True)
    for index in range(1, 5):
        name = f"episode_{index:06d}"
        (destination / name).symlink_to(source / name)
        (images / name).symlink_to(source / "dataset" / "images" / name)
    return destination


def convert(item, output_root, image_workers):
    name, source = item
    destination = output_root / name
    if destination.exists():
        raise RuntimeError(f"conversion output already exists: {destination}")
    command = [
        sys.executable,
        "-m",
        "lerobot_training.convert_to_lerobot",
        "--input-dir",
        str(source),
        "--output-dir",
        str(destination),
        "--image-workers",
        str(image_workers),
        "--action-source",
        "teacher",
        "--image-size",
        "224",
    ]
    log = output_root / f"{name}.log"
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode
    return name, destination, code, log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.merged_output.exists():
        parser.error("conversion output already exists")
    if min(args.workers, args.image_workers) < 1:
        parser.error("worker counts must be positive")
    args.output_root.mkdir(parents=True)
    clean_root = args.output_root.parent / (
        args.output_root.name + "_clean_sources"
    )
    clean_root.mkdir(exist_ok=True)
    shard_a = prepare_d4_shard_a(clean_root)
    items = [
        (name, PROJECT_ROOT / path)
        for name, path in SOURCES.items()
    ]
    items.append(("d4_full_shard_a_unique", shard_a))
    converted = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                convert, item, args.output_root, args.image_workers
            ): item[0]
            for item in items
        }
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({
                "run": result[0],
                "returncode": result[2],
                "log": str(result[3]),
            }), flush=True)
            if result[2]:
                failures.append(result[0])
            else:
                converted.append(result[1])
    if failures:
        raise RuntimeError(f"conversion failures: {failures}")
    ordered = [args.output_root / item[0] for item in items]
    command = [
        sys.executable,
        "-m",
        "lerobot_training.merge_lerobot_datasets",
        "--base-dir",
        str(ordered[0]),
    ]
    command.append("--additional-dir")
    command.extend(
        str(item) for item in ordered[1:]
    )
    command.extend(["--output-dir", str(args.merged_output)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
