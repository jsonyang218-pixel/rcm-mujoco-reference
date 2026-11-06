"""Merge converted LeRobot-style datasets without copying image files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_ARRAYS = (
    "episode_index",
    "frame_index",
    "action",
    "state",
    "oct_navigation",
    "oct_navigation_valid",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_image_path(source_dir: Path, output_dir: Path, value: str) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else source_dir / path
    absolute = absolute.resolve()
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return os.path.relpath(absolute, output_dir.resolve())


def merge_datasets(
    base_dir: Path,
    additional_dirs: Iterable[Path],
    output_dir: Path,
) -> dict:
    """Merge numerical arrays and preserve image references by relative path."""
    sources = [Path(base_dir), *map(Path, additional_dirs)]
    sources = [path.resolve() for path in sources]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if len(set(sources)) != len(sources):
        raise ValueError("Source datasets must be unique")
    for source in sources:
        if not (source / "data" / "dataset.npz").is_file():
            raise FileNotFoundError(source / "data" / "dataset.npz")
        if not (source / "data" / "image_paths.json").is_file():
            raise FileNotFoundError(source / "data" / "image_paths.json")
        if not (source / "metadata.json").is_file():
            raise FileNotFoundError(source / "metadata.json")

    output_dir.mkdir(parents=True)
    (output_dir / "data").mkdir()
    arrays = {name: [] for name in REQUIRED_ARRAYS}
    image_paths = []
    episodes = []
    source_records = []
    next_episode = 0
    source_spaces = None

    for source_order, source in enumerate(sources):
        metadata = _load_json(source / "metadata.json")
        spaces = (metadata.get("observation_space"), metadata.get("action_space"))
        if source_spaces is None:
            source_spaces = spaces
        elif spaces != source_spaces:
            raise ValueError(f"Dataset schemas differ: {source}")

        with np.load(source / "data" / "dataset.npz", allow_pickle=False) as data:
            missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
            if missing:
                raise ValueError(f"{source} lacks arrays: {missing}")
            local_episode = data["episode_index"].astype(np.int64)
            local_frame = data["frame_index"].astype(np.int64)
            local_count = len(local_episode)
            expected_episode = 0
            shifted_episode = np.zeros(local_count, dtype=np.int64)
            for episode_index in np.unique(local_episode):
                if int(episode_index) != expected_episode:
                    raise ValueError(
                        f"{source} episode indexes are not contiguous at "
                        f"{episode_index}; expected {expected_episode}"
                    )
                rows = local_episode == episode_index
                expected_frames = np.arange(int(rows.sum()), dtype=np.int64)
                if not np.array_equal(local_frame[rows], expected_frames):
                    raise ValueError(
                        f"{source} episode {episode_index} frame indexes are invalid"
                    )
                shifted_episode[rows] = next_episode + int(episode_index)
                expected_episode += 1
            for name in REQUIRED_ARRAYS:
                value = data[name].copy()
                if name == "episode_index":
                    value = shifted_episode
                arrays[name].append(value)

        paths = _load_json(source / "data" / "image_paths.json")
        if len(paths) != local_count:
            raise ValueError(
                f"{source} image path count {len(paths)} != samples {local_count}"
            )
        image_paths.extend(
            _relative_image_path(source, output_dir, value)
            for value in paths
        )
        for episode in metadata.get("episodes", []):
            shifted = dict(episode)
            shifted["episode_index"] = int(episode["episode_index"]) + next_episode
            shifted["source_dataset"] = str(source)
            episodes.append(shifted)
        local_episodes = int(np.unique(local_episode).size)
        source_records.append({
            "directory": str(source),
            "episode_offset": next_episode,
            "episodes": local_episodes,
            "frames": local_count,
        })
        next_episode += local_episodes

    merged = {}
    for name, values in arrays.items():
        array = np.concatenate(values)
        if name == "oct_navigation_valid":
            array = array.astype(np.bool_)
        merged[name] = array
    np.savez(output_dir / "data" / "dataset.npz", **merged)
    _write_json(output_dir / "data" / "image_paths.json", image_paths)

    metadata = _load_json(sources[0] / "metadata.json")
    metadata.update({
        "num_episodes": next_episode,
        "num_frames": int(len(merged["episode_index"])),
        "episodes": episodes,
        "source_datasets": source_records,
    })
    if len(episodes) != next_episode:
        raise ValueError("Episode metadata count does not match episode arrays")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Merge LeRobot-style datasets without duplicating images"
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--additional-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = merge_datasets(
        args.base_dir,
        args.additional_dir,
        args.output_dir,
    )
    print(json.dumps({
        "result": "PASS",
        "output_dir": str(args.output_dir.resolve()),
        "episodes": metadata["num_episodes"],
        "frames": metadata["num_frames"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
