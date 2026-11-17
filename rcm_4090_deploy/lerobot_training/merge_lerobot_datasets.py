"""Merge converted LeRobot-style datasets without copying image files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .dataset_contract import (
        CORE_ARRAYS,
        DATA_CONTRACT_VERSION,
        HISTORY_ARRAYS,
        validate_dataset_contract,
    )
except ImportError:
    from dataset_contract import (
        CORE_ARRAYS,
        DATA_CONTRACT_VERSION,
        HISTORY_ARRAYS,
        validate_dataset_contract,
    )

REQUIRED_ARRAYS = CORE_ARRAYS
OUTPUT_ARRAYS = (*CORE_ARRAYS, *HISTORY_ARRAYS)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_image_path(source_dir: Path, output_dir: Path, value: str) -> str:
    path = Path(str(value).replace("\\", "/"))
    absolute = path if path.is_absolute() else source_dir / path
    absolute = absolute.resolve()
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return os.path.relpath(absolute, output_dir.resolve())


def _normalized_spaces(metadata: dict) -> tuple[dict, dict]:
    """Normalize legacy scalar shapes before comparing dataset schemas."""
    comparable_fields = {
        "shape",
        "dtype",
        "names",
        "units",
        "frame",
        "definition",
        "type",
    }
    observation_space = {}
    for name, spec in metadata.get("observation_space", {}).items():
        normalized = {
            key: value
            for key, value in spec.items()
            if key in comparable_fields
        }
        if isinstance(normalized.get("shape"), int):
            normalized["shape"] = [normalized["shape"]]
        observation_space[name] = normalized
    action_space = {
        key: value
        for key, value in metadata.get("action_space", {}).items()
        if key in comparable_fields
    }
    if isinstance(action_space.get("shape"), int):
        action_space["shape"] = [action_space["shape"]]
    return observation_space, action_space


def merge_datasets(
    base_dir: Path,
    additional_dirs: Iterable[Path],
    output_dir: Path,
    legacy_action_source: str | None = None,
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
    arrays = {name: [] for name in OUTPUT_ARRAYS}
    image_paths = []
    episodes = []
    source_records = []
    action_sources = set()
    next_episode = 0
    source_spaces = None

    for source_order, source in enumerate(sources):
        metadata = _load_json(source / "metadata.json")
        effective_action_source = (
            metadata.get("action_source")
            or legacy_action_source
            or "unknown"
        )
        spaces = _normalized_spaces(metadata)
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
            source_arrays = {
                name: data[name].copy()
                for name in data.files
            }
            present_history = [
                name for name in HISTORY_ARRAYS if name in source_arrays
            ]
            if present_history and len(present_history) != len(HISTORY_ARRAYS):
                missing_history = sorted(
                    set(HISTORY_ARRAYS) - set(present_history)
                )
                raise ValueError(
                    f"{source} has a partial history contract: "
                    f"missing {missing_history}"
                )
            if not present_history:
                if effective_action_source != "executed":
                    raise ValueError(
                        f"{source} lacks executed-action history arrays and "
                        "is not unambiguously executed-source data"
                    )
                source_arrays["executed_action"] = source_arrays[
                    "action"
                ].copy()
                source_arrays["source_frame_index"] = local_frame.copy()
                fps = float(metadata.get("fps", 20.0))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{source} has invalid fps metadata")
                period_s = 1.0 / fps
                source_arrays["timestamp_s"] = (
                    local_frame.astype(np.float64) * period_s
                )
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
            validate_dataset_contract(source_arrays, require_history=True)
            for name in OUTPUT_ARRAYS:
                value = source_arrays[name].copy()
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
            shifted["action_source"] = metadata.get("action_source")
            episodes.append(shifted)
        local_episodes = int(np.unique(local_episode).size)
        source_records.append({
            "directory": str(source),
            "episode_offset": next_episode,
            "episodes": local_episodes,
            "frames": local_count,
            "action_source": effective_action_source,
            "action_sources": list(metadata.get(
                "action_sources",
                [effective_action_source],
            )),
        })
        action_sources.update(
            str(value)
            for value in metadata.get(
                "action_sources",
                [effective_action_source],
            )
        )
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
        "data_contract_version": DATA_CONTRACT_VERSION,
        "history_action_source": "executed_action",
        "continuity_source": "source_frame_index",
        "action_sources": sorted(action_sources),
        "action_source": (
            next(iter(action_sources))
            if len(action_sources) == 1
            else "per_episode"
        ),
        "action_supervision_source": (
            "per_episode; see source_datasets[].action_source"
        ),
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
    parser.add_argument(
        "--additional-dir",
        type=Path,
        nargs="*",
        default=[],
        help="Additional datasets to append after the base dataset",
    )
    parser.add_argument(
        "--legacy-action-source",
        choices=("executed", "teacher"),
        default=None,
        help=(
            "Explicit action semantics for legacy metadata that lacks "
            "action_source"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = merge_datasets(
        args.base_dir,
        args.additional_dir,
        args.output_dir,
        legacy_action_source=args.legacy_action_source,
    )
    print(json.dumps({
        "result": "PASS",
        "output_dir": str(args.output_dir.resolve()),
        "episodes": metadata["num_episodes"],
        "frames": metadata["num_frames"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
