#!/usr/bin/env python3
"""Audit expert labels, source continuity, and near-target replay coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lerobot_training.dataset_contract import valid_window_indices


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def distance_bucket(distance_um):
    if distance_um < 30:
        return "<30"
    if distance_um < 100:
        return "30-100"
    if distance_um < 300:
        return "100-300"
    if distance_um < 1000:
        return "300-1000"
    return ">=1000"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=16)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)

    metadata = json.loads(
        (args.data_dir / "metadata.json").read_text(encoding="utf-8")
    )
    source_by_episode = {
        int(row["episode_index"]): row.get("action_source", "executed")
        for row in metadata["episodes"]
    }
    with np.load(
        args.data_dir / "data" / "dataset.npz", allow_pickle=False
    ) as arrays:
        required = {
            "episode_index",
            "frame_index",
            "state",
            "action",
            "executed_action",
            "source_frame_index",
            "timestamp_s",
            "oct_navigation",
            "oct_navigation_valid",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"dataset lacks required arrays: {missing}")
        episode = arrays["episode_index"].astype(np.int64)
        action = arrays["action"].astype(np.float64)
        executed = arrays["executed_action"].astype(np.float64)
        source_frame = arrays["source_frame_index"].astype(np.int64)
        timestamp = arrays["timestamp_s"].astype(np.float64)
        oct_navigation = arrays["oct_navigation"].astype(np.float64)
        oct_valid = arrays["oct_navigation_valid"].astype(bool)

    if not (
        len(episode)
        == len(action)
        == len(executed)
        == len(source_frame)
        == len(timestamp)
        == len(oct_navigation)
        == len(oct_valid)
    ):
        raise ValueError("dataset arrays have inconsistent lengths")
    if not np.isfinite(action).all() or not np.isfinite(executed).all():
        raise ValueError("action arrays contain non-finite values")

    starts, diagnostics = valid_window_indices(
        episode,
        source_frame,
        timestamp,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        control_period_s=0.05,
        oct_navigation=oct_navigation,
        oct_navigation_valid=oct_valid,
        return_diagnostics=True,
    )
    distances_um = np.linalg.norm(oct_navigation, axis=1) * 1000.0
    sources = {
        source: {
            "frames": int(np.sum(
                [source_by_episode[int(value)] == source for value in episode]
            )),
            "valid_action_windows": 0,
            "label_differs_from_executed": 0,
            "distance_buckets": {
                key: 0
                for key in ("<30", "30-100", "100-300", "300-1000", ">=1000")
            },
            "invalid_oct_frames": 0,
        }
        for source in sorted(set(source_by_episode.values()))
    }
    for row in starts:
        source = source_by_episode[int(episode[row])]
        sources[source]["valid_action_windows"] += 1
        sources[source]["distance_buckets"][
            distance_bucket(float(distances_um[row]))
        ] += 1
    for row in range(len(episode)):
        source = source_by_episode[int(episode[row])]
        if not oct_valid[row]:
            sources[source]["invalid_oct_frames"] += 1
        if np.any(np.abs(action[row] - executed[row]) > 1e-6):
            sources[source]["label_differs_from_executed"] += 1

    source_counts = {
        source: sum(
            source_by_episode[int(episode[row])] == source
            for row in starts
        )
        for source in sources
    }
    total_source_windows = sum(source_counts.values())
    index_hash = hashlib.sha256(
        np.asarray(starts, dtype="<i8").tobytes()
    ).hexdigest()
    report = {
        "status": "PASS",
        "data_dir": str(args.data_dir),
        "frames": len(episode),
        "episodes": int(len(np.unique(episode))),
        "action_block_steps": args.n_action_steps,
        "history_steps": args.n_obs_steps,
        "control_period_s": 0.05,
        "action_label_contract": {
            "supervision_array": "action",
            "execution_array": "executed_action",
            "history_action_array": "executed_action",
            "cross_episode_windows_allowed": False,
            "cross_source_frame_gap_windows_allowed": False,
        },
        "valid_windows": int(len(starts)),
        "valid_index_sha256": index_hash,
        "natural_correction_ratio": (
            source_counts.get("teacher", 0) / total_source_windows
            if total_source_windows
            else 0.0
        ),
        "window_diagnostics": diagnostics,
        "sources": sources,
    }
    write_json(args.output_dir / "experience_replay_contract.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
