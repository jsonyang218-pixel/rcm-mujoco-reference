"""Merge completed v12 shard artifacts into the official checkpointed run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rcm.waypoint_dataset import write_manifest
from scripts.datasets.collect_zero_error_dataset import write_json


DATASET_SUBDIRECTORIES = (
    "images",
    "data",
    "policy_inputs",
    "policy_inputs_no_image",
    "action_labels",
)


def _move_tree(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        target = destination / item.name
        if target.exists():
            raise FileExistsError(f"merge target already exists: {target}")
        item.replace(target)
    source.rmdir()


def merge(run_dir, shard_dirs, expected_episodes=608):
    run_dir = Path(run_dir).resolve()
    shards = [Path(path).resolve() for path in shard_dirs]
    if len(set(shards)) != len(shards):
        raise ValueError("shard directories must be unique")
    existing = sorted(run_dir.glob("episode_*"))
    existing_indexes = set()
    for path in existing:
        if not path.is_dir() or not path.name.split("_")[-1].isdigit():
            continue
        if not (path / "episode_result.json").is_file():
            raise ValueError(f"existing episode is incomplete: {path}")
        existing_indexes.add(int(path.name.split("_")[-1]))

    moved = []
    for shard in shards:
        if not shard.is_dir():
            raise FileNotFoundError(shard)
        for source in sorted(shard.glob("episode_*")):
            if not source.is_dir() or not source.name.split("_")[-1].isdigit():
                continue
            index = int(source.name.split("_")[-1])
            if index in existing_indexes:
                raise ValueError(f"duplicate completed episode {index}")
            if not (source / "episode_result.json").is_file():
                raise ValueError(f"shard episode is incomplete: {source}")
            destination = run_dir / source.name
            source.replace(destination)
            existing_indexes.add(index)
            moved.append(index)
        for name in DATASET_SUBDIRECTORIES:
            source = shard / "dataset" / name
            if source.is_dir():
                _move_tree(source, run_dir / "dataset" / name)
        shard_dataset = shard / "dataset"
        if shard_dataset.is_dir() and not any(shard_dataset.iterdir()):
            shard_dataset.rmdir()

    expected_indexes = set(range(expected_episodes))
    if existing_indexes != expected_indexes:
        missing = sorted(expected_indexes - existing_indexes)
        extra = sorted(existing_indexes - expected_indexes)
        raise ValueError(f"episode merge incomplete; missing={missing[:20]} extra={extra[:20]}")

    episodes = [
        json.loads(
            (run_dir / f"episode_{index:06d}" / "episode_result.json").read_text(
                encoding="utf-8"
            )
        )
        for index in range(expected_episodes)
    ]
    config = json.loads(
        (run_dir / "config_snapshot.json").read_text(encoding="utf-8")
    )
    manifest = write_manifest(
        run_dir / "dataset",
        episodes,
        [224, 224, 3],
        float(config["navigation"]["visual_servo"]["sample_period_s"]),
        "visual_current_target",
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
            "bit-exact protocol-reachable source-terminal needle-tip "
            "MuJoCo world position"
        ),
        "audit_path_base": "Parent of this dataset directory",
    })
    write_json(run_dir / "dataset" / "manifest.json", manifest)
    write_json(run_dir / "shard_merge.json", {
        "result": "PASS",
        "episodes": expected_episodes,
        "moved_episodes": len(moved),
        "shards": [str(path) for path in shards],
    })
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-episodes", type=int, default=608)
    args = parser.parse_args(argv)
    manifest = merge(args.run_dir, args.shard_dir, args.expected_episodes)
    print(json.dumps({
        "result": "PASS",
        "episodes": len(manifest["episodes"]),
        "frames": manifest["total_frames"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
