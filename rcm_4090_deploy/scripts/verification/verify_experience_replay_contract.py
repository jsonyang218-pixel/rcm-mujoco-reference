#!/usr/bin/env python3
"""Audit expert labels, source continuity, and near-target replay coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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

    continuous_step = np.zeros(len(episode), dtype=bool)
    continuous_step[1:] = (
        episode[1:] == episode[:-1]
    ) & (source_frame[1:] == source_frame[:-1] + 1) & np.isclose(
        np.diff(timestamp), 0.05, rtol=1e-3, atol=1e-6
    )
    valid_window = np.ones(
        len(episode) - args.n_action_steps - args.n_obs_steps + 2,
        dtype=bool,
    )
    for offset in range(-(args.n_obs_steps - 1), args.n_action_steps):
        valid_window &= continuous_step[
            np.arange(
                args.n_obs_steps - 1,
                len(episode) - args.n_action_steps + 1,
            )
            + offset
        ]

    starts = np.arange(
        args.n_obs_steps - 1,
        len(episode) - args.n_action_steps + 1,
    )
    starts = starts[valid_window]
    starts = starts[
        oct_valid[starts]
        & oct_valid[starts + 1]
        & np.isfinite(oct_navigation[starts]).all(axis=1)
        & np.isfinite(oct_navigation[starts + 1]).all(axis=1)
    ]
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
        "sources": sources,
    }
    write_json(args.output_dir / "experience_replay_contract.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
