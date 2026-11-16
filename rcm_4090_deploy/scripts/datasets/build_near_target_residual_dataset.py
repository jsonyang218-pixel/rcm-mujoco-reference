"""Build an OCT-valid near-target residual dataset from covariate DAgger."""
from __future__ import annotations

import argparse
from itertools import groupby
import json
import os
from pathlib import Path

import numpy as np


ALLOWED_EXECUTORS = {"policy", "teacher", "beta_teacher"}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _contiguous_runs(indices):
    runs = []
    for _, group in groupby(
        enumerate(indices),
        key=lambda pair: pair[1] - pair[0],
    ):
        values = [value for _, value in group]
        runs.append((values[0], values[-1] + 1))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", default=None)
    parser.add_argument("--min-error-um", type=float, default=0.1)
    parser.add_argument("--max-error-um", type=float, default=2500.0)
    parser.add_argument("--min-segment-frames", type=int, default=2)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (
        args.min_error_um < 0
        or args.max_error_um <= args.min_error_um
        or args.min_segment_frames < 1
    ):
        raise ValueError("invalid near-target selection thresholds")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    (output_dir / "data").mkdir()
    states = []
    actions = []
    oct_navigation = []
    oct_navigation_valid = []
    episode_indices = []
    frame_indices = []
    image_paths = []
    segments = []
    source_records = []
    selected_errors_um = []
    skipped = []

    selected_shards = (
        None if args.shards is None else set(args.shards)
    )
    for raw_shard in sorted(args.raw_root.glob("g*")):
        if (
            selected_shards is not None
            and raw_shard.name not in selected_shards
        ):
            continue
        converted = (
            args.converted_root / f"online_dagger_covshift2_{raw_shard.name}_224"
        )
        metadata_path = converted / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = _load_json(metadata_path)
        raw_episode_dirs = sorted(raw_shard.glob("episode_*"))
        shard_segments = 0
        shard_frames = 0
        for source_episode in metadata.get("episodes", []):
            if source_episode.get("strict_zero_error") is not True:
                continue
            source_index = int(source_episode["source_episode_index"])
            output_source_episode = int(source_episode["episode_index"])
            if source_index >= len(raw_episode_dirs):
                raise ValueError(
                    f"{raw_shard} source episode index {source_index} "
                    "does not map to a raw episode"
                )
            raw_episode = raw_episode_dirs[source_index]
            samples_path = raw_episode / "dagger_samples.jsonl"
            if not samples_path.is_file():
                raise FileNotFoundError(samples_path)
            rows = _load_jsonl(samples_path)
            selected = []
            for row_index, row in enumerate(rows):
                teacher_delta = row.get("teacher_action_deg_deg_mm")
                error_um = float(row["automatic_truth_error_mm"]) * 1000.0
                if (
                    row.get("teacher_valid") is True
                    and row.get("executor") in ALLOWED_EXECUTORS
                    and row.get("oct_navigation_valid") is True
                    and teacher_delta is not None
                    and np.linalg.norm(
                        np.asarray(teacher_delta, dtype=np.float64)
                    )
                    > 1e-12
                    and args.min_error_um < error_um <= args.max_error_um
                ):
                    selected.append(row_index)
            if not selected:
                skipped.append({
                    "shard": raw_shard.name,
                    "source_episode_index": source_index,
                    "reason": "NO_ELIGIBLE_NEAR_TARGET_FRAMES",
                })
                continue
            for start, end in _contiguous_runs(selected):
                if end - start < args.min_segment_frames:
                    continue
                new_episode = len(segments)
                new_frame_index = 0
                for source_frame in range(start, end):
                    row = rows[source_frame]
                    state = np.asarray(
                        row["state_A_deg_B_deg_C_mm"],
                        dtype=np.float32,
                    )
                    teacher_delta = np.asarray(
                        row["teacher_action_deg_deg_mm"],
                        dtype=np.float32,
                    )
                    oct_xyz = np.asarray(
                        row["oct_navigation_mm"],
                        dtype=np.float32,
                    )
                    image = (
                        converted
                        / "images"
                        / (
                            f"ep{output_source_episode:06d}_"
                            f"frame{source_frame:06d}.png"
                        )
                    )
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    states.append(state)
                    actions.append(teacher_delta)
                    oct_navigation.append(oct_xyz)
                    oct_navigation_valid.append(True)
                    episode_indices.append(new_episode)
                    frame_indices.append(new_frame_index)
                    new_frame_index += 1
                    image_paths.append(
                        os.path.relpath(image.resolve(), output_dir)
                    )
                    selected_errors_um.append(
                        float(row["automatic_truth_error_mm"]) * 1000.0
                    )
                segment = {
                    "episode_index": new_episode,
                    "shard": raw_shard.name,
                    "source_episode_index": source_index,
                    "converted_episode_index": output_source_episode,
                    "source_frame_start": start,
                    "source_frame_end_exclusive": end,
                    "num_frames": end - start,
                }
                segments.append(segment)
                shard_segments += 1
                shard_frames += end - start
        source_records.append({
            "shard": raw_shard.name,
            "raw_directory": str(raw_shard.resolve()),
            "converted_directory": str(converted.resolve()),
            "segments": shard_segments,
            "frames": shard_frames,
        })

    if not selected_errors_um:
        raise ValueError("no eligible near-target frames found")
    local_frame = np.asarray(frame_indices, dtype=np.int64)
    local_episode = np.asarray(episode_indices, dtype=np.int64)
    for episode in np.unique(local_episode):
        frames = local_frame[local_episode == episode]
        if not np.array_equal(
            frames,
            np.arange(len(frames), dtype=np.int64),
        ):
            raise ValueError("segment frame indexing is not contiguous")

    np.savez(
        output_dir / "data" / "dataset.npz",
        episode_index=local_episode,
        frame_index=local_frame,
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        oct_navigation=np.asarray(oct_navigation, dtype=np.float32),
        oct_navigation_valid=np.asarray(
            oct_navigation_valid,
            dtype=np.bool_,
        ),
    )
    (output_dir / "data" / "image_paths.json").write_text(
        json.dumps(image_paths, indent=2) + "\n",
        encoding="utf-8",
    )
    errors = np.asarray(selected_errors_um, dtype=np.float64)
    metadata = {
        "dataset_version": "v12",
        "dataset_variant": "covshift2_near_target_residual",
        "num_episodes": len(segments),
        "num_frames": len(selected_errors_um),
        "observation_space": {
            "image": {"shape": [224, 224, 3], "dtype": "uint8"},
            "state": {
                "shape": [3],
                "dtype": "float32",
                "names": ["A_deg", "B_deg", "C_mm"],
            },
            "oct_navigation": {
                "shape": [3],
                "dtype": "float32",
                "units": "mm",
                "frame": "camera_optical",
                "definition": "target_minus_tip",
            },
        },
        "action_space": {
            "shape": [3],
            "dtype": "float32",
            "names": ["A_deg", "B_deg", "C_mm"],
            "type": "delta",
        },
        "selection": {
            "source": "strict-zero covariate-shift DAgger round 2",
            "minimum_error_um_exclusive": args.min_error_um,
            "maximum_error_um_inclusive": args.max_error_um,
            "allowed_executors": sorted(ALLOWED_EXECUTORS),
            "requires_teacher_valid": True,
            "requires_oct_valid": True,
            "requires_nonzero_teacher_action": True,
            "excludes_exact_terminal": True,
            "excludes_hold": True,
            "error_um": {
                "minimum": float(errors.min()),
                "median": float(np.median(errors)),
                "maximum": float(errors.max()),
            },
        },
        "source_records": source_records,
        "segments": segments,
        "skipped_episodes": skipped,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "result": "PASS",
        "output_dir": str(output_dir),
        "episodes": len(segments),
        "frames": len(selected_errors_um),
        "error_um": metadata["selection"]["error_um"],
        "skipped_episodes": len(skipped),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
