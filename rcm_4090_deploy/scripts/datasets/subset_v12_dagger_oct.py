"""Extract the v12 DAgger-derived episodes into a standalone OCT dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build(source_dir, output_dir):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output dataset already exists: {output_dir}")

    metadata = json.loads(
        (source_dir / "metadata.json").read_text(encoding="utf-8")
    )
    source_episode_count = len(metadata["episodes"])
    source_image_paths = json.loads(
        (source_dir / "data" / "image_paths.json").read_text(encoding="utf-8")
    )
    source_audits = json.loads(
        (source_dir / "data" / "terminal_audits.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(source_dir / "data" / "dataset.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    required = {
        "episode_index",
        "frame_index",
        "action",
        "state",
        "oct_navigation",
        "oct_navigation_valid",
    }
    if set(arrays) != required:
        raise ValueError(f"unexpected source arrays: {sorted(arrays)}")

    selected = [
        episode
        for episode in metadata["episodes"]
        if "dagger" in str(episode.get("source_dataset", "")).lower()
    ]
    if not selected:
        raise ValueError("source metadata contains no DAgger episodes")

    output_arrays = {
        name: [] for name in (
            "episode_index",
            "frame_index",
            "action",
            "state",
            "oct_navigation",
            "oct_navigation_valid",
        )
    }
    output_episodes = []
    output_audits = []
    output_image_paths = []
    mapping = []
    (output_dir / "data").mkdir(parents=True)

    for new_episode, source_episode in enumerate(selected):
        old_episode = int(source_episode["episode_index"])
        rows = np.flatnonzero(arrays["episode_index"] == old_episode)
        frame_count = int(source_episode["num_frames"])
        if len(rows) != frame_count:
            raise ValueError(f"source episode {old_episode} row count changed")
        output_arrays["episode_index"].append(
            np.full(len(rows), new_episode, dtype=arrays["episode_index"].dtype)
        )
        for name in required - {"episode_index"}:
            output_arrays[name].append(arrays[name][rows])

        for local_frame, global_row in enumerate(rows):
            source_image = source_dir / source_image_paths[global_row]
            relative = (
                Path("images")
                / f"ep{new_episode:06d}_frame{local_frame:06d}.png"
            )
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source_image, destination)
            output_image_paths.append(relative.as_posix())

        episode = dict(source_episode)
        episode["episode_index"] = new_episode
        episode["v12_full_train_episode_index"] = old_episode
        episode["terminal_audit_index"] = new_episode
        output_episodes.append(episode)

        audit = dict(source_audits[old_episode])
        audit["v12_full_train_episode_index"] = old_episode
        audit["episode_index"] = new_episode
        output_audits.append(audit)
        mapping.append({
            "dagger_episode_index": new_episode,
            "v12_full_train_episode_index": old_episode,
            "source_episode_index": int(source_episode["source_episode_index"]),
            "source_dataset": source_episode["source_dataset"],
            "frames": frame_count,
        })

    final_arrays = {
        name: np.concatenate(parts, axis=0)
        for name, parts in output_arrays.items()
    }
    expected_frames = sum(int(episode["num_frames"]) for episode in selected)
    if len(final_arrays["state"]) != expected_frames:
        raise ValueError("selected DAgger frame count changed")
    if not np.any(final_arrays["oct_navigation_valid"]):
        raise ValueError("selected DAgger episodes contain no valid OCT rows")

    np.savez(output_dir / "data" / "dataset.npz", **final_arrays)
    _write_json(output_dir / "data" / "image_paths.json", output_image_paths)
    _write_json(output_dir / "data" / "terminal_audits.json", output_audits)

    metadata["num_episodes"] = len(output_episodes)
    metadata["num_frames"] = expected_frames
    metadata["episodes"] = output_episodes
    metadata["dataset_variant"] = "v12_zero_error_dagger_oct_navigation"
    metadata["subset_definition"] = {
        "selection": "source_dataset contains 'dagger'",
        "source_dataset": str(source_dir),
        "source_episodes": source_episode_count,
        "selected_episodes": len(output_episodes),
        "episode_mapping": "data/dagger_episode_mapping.json",
        "scope": (
            "DAgger-derived rollout states relabeled/executed by the v12 "
            "strict RGB-D teacher; not a newly collected DAgger round"
        ),
    }
    _write_json(output_dir / "metadata.json", metadata)

    valid = final_arrays["oct_navigation_valid"]
    valid_values = final_arrays["oct_navigation"][valid]
    report = {
        "result": "PASS",
        "source_dataset": str(source_dir),
        "output_dir": str(output_dir),
        "episodes": len(output_episodes),
        "frames": expected_frames,
        "valid_oct_frames": int(valid.sum()),
        "invalid_oct_frames": int((~valid).sum()),
        "valid_oct_fraction": float(valid.mean()),
        "oct_navigation_min_mm": valid_values.min(axis=0).tolist(),
        "oct_navigation_max_mm": valid_values.max(axis=0).tolist(),
        "mapping": mapping,
    }
    _write_json(output_dir / "data" / "dagger_episode_mapping.json", mapping)
    _write_json(output_dir / "data" / "dagger_subset_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args.source_dataset, args.output_dir)
    print(json.dumps({
        key: report[key]
        for key in (
            "result",
            "episodes",
            "frames",
            "valid_oct_frames",
            "invalid_oct_frames",
            "valid_oct_fraction",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
