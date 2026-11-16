"""Build an OCT-guided v12 training dataset from per-frame raw OCT audits."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link_image_tree(source, destination):
    linked = 0
    for source_path in sorted((Path(source) / "images").rglob("*")):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(Path(source) / "images")
        target = Path(destination) / "images" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, target)
        linked += 1
    return linked


def build(source_dir, raw_run_dir, output_dir):
    source_dir = Path(source_dir).resolve()
    raw_run_dir = Path(raw_run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output dataset already exists: {output_dir}")

    metadata = json.loads(
        (source_dir / "metadata.json").read_text(encoding="utf-8")
    )
    image_paths = json.loads(
        (source_dir / "data" / "image_paths.json").read_text(encoding="utf-8")
    )
    with np.load(source_dir / "data" / "dataset.npz", allow_pickle=False) as data:
        episode_indices = data["episode_index"].copy()
        frame_indices = data["frame_index"].copy()
        actions = data["action"].copy()
        states = data["state"].copy()

    if (
        len(episode_indices) != len(frame_indices)
        or len(episode_indices) != len(actions)
        or len(episode_indices) != len(states)
        or len(episode_indices) != len(image_paths)
    ):
        raise ValueError("source dataset arrays are not row aligned")

    oct_navigation = np.zeros((len(states), 3), dtype=np.float32)
    oct_valid = np.zeros(len(states), dtype=np.bool_)
    matched_visual_frames = 0
    unmatched_visual_frames = 0
    episode_reports = []

    for episode in metadata["episodes"]:
        episode_index = int(episode["episode_index"])
        frame_count = int(episode["num_frames"])
        rows = np.flatnonzero(episode_indices == episode_index)
        if len(rows) != frame_count:
            raise ValueError(f"episode {episode_index} row count changed")
        if not np.array_equal(
                frame_indices[rows],
                np.arange(frame_count, dtype=frame_indices.dtype)):
            raise ValueError(f"episode {episode_index} frames are not contiguous")

        episode_dir = raw_run_dir / f"episode_{episode_index:06d}"
        audits = _read_jsonl(episode_dir / "frame_audit.jsonl")
        oct_rows = _read_jsonl(episode_dir / "oct_observations.jsonl")
        if len(audits) != frame_count:
            raise ValueError(
                f"episode {episode_index} audit/raw frame count mismatch"
            )
        oct_by_iteration = {}
        for row in oct_rows:
            iteration = int(row["iteration"])
            if iteration in oct_by_iteration:
                raise ValueError(
                    f"episode {episode_index} repeats OCT iteration {iteration}"
                )
            delta_m = np.asarray(
                row["navigation_depth"]["delta_xyz_m"],
                dtype=np.float64,
            )
            if delta_m.shape != (3,) or not np.all(np.isfinite(delta_m)):
                raise ValueError(
                    f"episode {episode_index} has invalid OCT navigation"
                )
            oct_by_iteration[iteration] = (delta_m * 1000.0).astype(np.float32)

        episode_matched = 0
        episode_unmatched = 0
        for local_frame, global_row in enumerate(rows):
            control_index = audits[local_frame].get("control_frame_index")
            if control_index is None:
                continue
            delta_mm = oct_by_iteration.get(int(control_index))
            if delta_mm is None:
                episode_unmatched += 1
                unmatched_visual_frames += 1
                continue
            oct_navigation[global_row] = delta_mm
            oct_valid[global_row] = True
            episode_matched += 1
            matched_visual_frames += 1
        episode_reports.append({
            "episode_index": episode_index,
            "frames": frame_count,
            "raw_oct_observations": len(oct_rows),
            "matched_frames": episode_matched,
            "unmatched_visual_frames": episode_unmatched,
            "invalid_frames": frame_count - episode_matched,
        })

    if not np.any(oct_valid):
        raise ValueError("no OCT navigation rows were matched")
    valid_values = oct_navigation[oct_valid]
    if not np.all(np.isfinite(valid_values)):
        raise ValueError("matched OCT navigation contains non-finite values")

    (output_dir / "data").mkdir(parents=True)
    np.savez(
        output_dir / "data" / "dataset.npz",
        episode_index=episode_indices,
        frame_index=frame_indices,
        action=actions,
        state=states,
        oct_navigation=oct_navigation,
        oct_navigation_valid=oct_valid,
    )
    _write_json(output_dir / "data" / "image_paths.json", image_paths)
    _write_json(
        output_dir / "data" / "terminal_audits.json",
        json.loads(
            (source_dir / "data" / "terminal_audits.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    linked_images = _link_image_tree(source_dir, output_dir)

    metadata["dataset_variant"] = "oct_navigation_from_registered_rgbd"
    metadata["oct_navigation"] = {
        "available": True,
        "source": "registered RGB-D visible-depth simulation",
        "source_files": "episode_*/oct_observations.jsonl",
        "definition": "target_minus_tip",
        "frame": "camera_optical",
        "units": "mm",
        "is_simulation": True,
        "raw_oct_available": False,
        "alignment": "frame_audit.control_frame_index -> OCT iteration",
        "invalid_policy": (
            "fallback, exact-terminal and hold frames are zero with valid=false; "
            "a visual callback without a matching OCT row is also invalid"
        ),
        "scope": (
            "Target-relative navigation input for an OCT-guided policy. "
            "This is simulation RGB-D used as the OCT navigation value, not a "
            "physical OCT acquisition."
        ),
    }
    metadata["observation_space"]["oct_navigation"] = {
        "shape": [3],
        "dtype": "float32",
        "units": "mm",
        "frame": "camera_optical",
        "definition": "target_minus_tip",
        "available": True,
        "source": "registered RGB-D visible-depth simulation",
    }
    metadata["policy_input_fields"] = [
        "observation.images.microscope_rgb",
        "observation.state",
        "observation.oct",
        "task",
    ]
    metadata["excluded_from_policy_inputs"] = [
        "target world coordinates",
        "terminal truth error",
        "execution source",
        "fallback metadata",
        "source episode identifiers",
    ]
    report = {
        "result": "PASS",
        "source_dataset": str(source_dir),
        "raw_run_dir": str(raw_run_dir),
        "output_dir": str(output_dir),
        "episodes": len(metadata["episodes"]),
        "frames": len(states),
        "matched_visual_frames": matched_visual_frames,
        "unmatched_visual_frames": unmatched_visual_frames,
        "valid_frames": int(oct_valid.sum()),
        "invalid_frames": int((~oct_valid).sum()),
        "valid_fraction": float(oct_valid.mean()),
        "oct_navigation_min_mm": valid_values.min(axis=0).tolist(),
        "oct_navigation_max_mm": valid_values.max(axis=0).tolist(),
        "linked_images": linked_images,
        "episode_reports": episode_reports,
    }
    metadata["oct_navigation"]["build_report"] = (
        "data/oct_navigation_report.json"
    )
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "data" / "oct_navigation_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--raw-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build(args.source_dataset, args.raw_run_dir, args.output_dir)
    print(json.dumps({
        key: report[key]
        for key in (
            "result",
            "episodes",
            "frames",
            "valid_frames",
            "invalid_frames",
            "valid_fraction",
            "unmatched_visual_frames",
            "linked_images",
        )
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
