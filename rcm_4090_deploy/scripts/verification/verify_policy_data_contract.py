"""Verify migrated DAgger action-history and frame-continuity semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from lerobot_training.dataset_contract import (
    HISTORY_ARRAYS,
    validate_dataset_contract,
)
from lerobot_training.train_diffusion_policy import RCMDataset


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _shard_name(data_dir):
    match = re.search(r"(g\d+)", Path(data_dir).name)
    if match is None:
        raise ValueError(f"cannot infer shard from {data_dir}")
    return match.group(1)


def verify(data_dir, raw_root, round2_data_dirs):
    data_dir = Path(data_dir).resolve()
    raw_root = Path(raw_root).resolve()
    round2_data_dirs = [Path(path).resolve() for path in round2_data_dirs]
    metadata = _read_json(data_dir / "metadata.json")
    if int(metadata.get("data_contract_version", -1)) != 2:
        raise ValueError("merged dataset is not data contract version 2")
    if (
        metadata.get("action_source") != "per_episode"
        or metadata.get("action_sources") != ["executed", "teacher"]
        or metadata.get("history_action_source") != "executed_action"
        or metadata.get("continuity_source") != "source_frame_index"
    ):
        raise ValueError("merged action/history metadata is inconsistent")
    with np.load(data_dir / "data" / "dataset.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    validate_dataset_contract(arrays, require_history=True)

    source_offsets = {
        str(Path(record["directory"]).resolve()): int(
            record["episode_offset"]
        )
        for record in metadata["source_datasets"]
    }
    accepted_episodes = []
    raw_source_frames = 0
    raw_retained_frames = 0
    policy_frames = 0
    valid_policy_frames = 0
    invalid_frames = []
    gaps = []
    compared_rows = 0
    policy_action_differences = []

    for converted_dir in round2_data_dirs:
        converted_metadata = _read_json(converted_dir / "metadata.json")
        shard = _shard_name(converted_dir)
        raw_episode_dirs = sorted(
            (raw_root / shard).glob("episode_*")
        )
        global_offset = source_offsets.get(str(converted_dir))
        if global_offset is None:
            raise ValueError(
                f"merged dataset has no source record for {converted_dir}"
            )
        for converted_episode in converted_metadata["episodes"]:
            local_episode = int(converted_episode["episode_index"])
            source_index = int(converted_episode["source_episode_index"])
            if source_index >= len(raw_episode_dirs):
                raise ValueError(
                    f"{converted_dir} source episode {source_index} is invalid"
                )
            raw_episode = raw_episode_dirs[source_index]
            report = _read_json(raw_episode / "control_report.json")
            if not (
                report.get("result") == "PASS"
                and report.get("strict_zero_error") is True
            ):
                raise ValueError(f"{raw_episode} is not strict-zero PASS")
            raw_rows = _read_jsonl(raw_episode / "dagger_samples.jsonl")
            retained_rows = [
                row
                for row in raw_rows
                if row.get("teacher_valid") is True
                and row.get("teacher_action_deg_deg_mm") is not None
            ]
            raw_source_frames += len(raw_rows)
            raw_retained_frames += len(retained_rows)
            for source_frame, row in enumerate(raw_rows):
                if row.get("executor") == "policy":
                    policy_frames += 1
                    if row.get("teacher_valid") is True:
                        valid_policy_frames += 1
                        if row.get("teacher_action_deg_deg_mm") is not None:
                            policy_action_differences.append(
                                np.asarray(
                                    row["teacher_action_deg_deg_mm"],
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    row["executed_action_deg_deg_mm"],
                                    dtype=np.float64,
                                )
                            )
                if row.get("teacher_valid") is not True:
                    invalid_frames.append({
                        "episode": str(raw_episode),
                        "source_frame_index": source_frame,
                        "reason": row.get("teacher_reason"),
                    })

            global_episode = global_offset + local_episode
            rows = np.flatnonzero(
                arrays["episode_index"] == global_episode
            )
            if len(rows) != len(retained_rows):
                raise ValueError(
                    f"episode {global_episode} retained frame count differs: "
                    f"{len(rows)} != {len(retained_rows)}"
                )
            expected_source = np.arange(
                len(raw_rows),
                dtype=np.int64,
            )[
                np.asarray([
                    row.get("teacher_valid") is True
                    and row.get("teacher_action_deg_deg_mm") is not None
                    for row in raw_rows
                ])
            ]
            if not np.array_equal(
                arrays["source_frame_index"][rows],
                expected_source,
            ):
                raise ValueError(
                    f"episode {global_episode} source frame indexes differ"
                )
            expected_timestamps = np.asarray(
                [float(row["timestamp_s"]) for row in retained_rows],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                arrays["timestamp_s"][rows],
                expected_timestamps,
                rtol=0,
                atol=1e-12,
                err_msg=f"episode {global_episode} timestamps differ",
            )
            expected_action = np.asarray(
                [
                    row["teacher_action_deg_deg_mm"]
                    for row in retained_rows
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                arrays["action"][rows],
                expected_action,
                rtol=0,
                atol=1e-6,
                err_msg=f"episode {global_episode} teacher labels differ",
            )
            expected_executed = np.asarray(
                [
                    row["executed_action_deg_deg_mm"]
                    for row in retained_rows
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                arrays["executed_action"][rows],
                expected_executed,
                rtol=0,
                atol=1e-6,
                err_msg=f"episode {global_episode} executed actions differ",
            )
            source = arrays["source_frame_index"][rows]
            for left, right in zip(source, source[1:]):
                if right != left + 1:
                    gaps.append({
                        "episode": global_episode,
                        "before_source_frame_index": int(left),
                        "after_source_frame_index": int(right),
                    })
            compared_rows += len(rows)
            accepted_episodes.append(str(raw_episode))

    expected = {
        "episodes": 79,
        "raw_source_frames": 15886,
        "retained_frames": 15883,
        "policy_frames": 5617,
        "valid_policy_frames": 5614,
        "invalid_frames": 3,
        "source_gaps": 3,
    }
    actual = {
        "episodes": len(accepted_episodes),
        "raw_source_frames": raw_source_frames,
        "retained_frames": raw_retained_frames,
        "policy_frames": policy_frames,
        "valid_policy_frames": valid_policy_frames,
        "invalid_frames": len(invalid_frames),
        "source_gaps": len(gaps),
    }
    if actual != expected:
        raise ValueError(f"unexpected migration counts: {actual} != {expected}")
    if compared_rows != raw_retained_frames:
        raise ValueError("not all round2 rows were compared")
    if len(metadata["episodes"]) != 839:
        raise ValueError("merged episode count is not 839")
    if len(arrays["state"]) != 234316:
        raise ValueError("merged frame count is not 234316")

    dataset = RCMDataset(
        data_dir,
        n_obs_steps=2,
        n_action_steps=8,
        include_velocity=True,
        include_previous_action=True,
    )
    for index in dataset.valid_indices:
        history_rows = index - np.arange(1, dataset.n_obs_steps)
        future_rows = index + np.arange(1, dataset.n_action_steps)
        history_source = dataset.continuity_frame_indices[history_rows]
        future_source = dataset.continuity_frame_indices[future_rows]
        if not np.all(
            history_source
            == dataset.continuity_frame_indices[index]
            - np.arange(1, dataset.n_obs_steps)
        ):
            raise ValueError(f"observation window crosses a gap at {index}")
        if not np.all(
            future_source
            == dataset.continuity_frame_indices[index]
            + np.arange(1, dataset.n_action_steps)
        ):
            raise ValueError(f"action window crosses a gap at {index}")
    velocities, previous_actions, history_valid = dataset.history_features(
        dataset.valid_indices
    )
    if not np.all(np.isfinite(velocities)):
        raise ValueError("velocity history contains non-finite values")
    if not np.all(np.isfinite(previous_actions)):
        raise ValueError("previous-action history contains non-finite values")
    if not np.all(history_valid):
        raise ValueError("valid dataset windows include a broken history")

    action_differences = np.asarray(
        policy_action_differences,
        dtype=np.float64,
    )
    if action_differences.shape != (valid_policy_frames, 3):
        raise ValueError("not every valid policy frame has an action difference")
    identical_policy_actions = int(np.sum(
        np.all(np.abs(action_differences) <= 1e-12, axis=1)
    ))
    if identical_policy_actions != 0:
        raise ValueError(
            "expected DAgger teacher and accepted policy actions to differ"
        )

    return {
        "result": "PASS",
        "data_dir": str(data_dir),
        "raw_root": str(raw_root),
        "round2_data_dirs": [str(path) for path in round2_data_dirs],
        "contract_fields": list(HISTORY_ARRAYS),
        "counts": actual,
        "invalid_frames": invalid_frames,
        "gaps": gaps,
        "merged_episodes": len(metadata["episodes"]),
        "merged_frames": len(arrays["state"]),
        "valid_training_windows": len(dataset.valid_indices),
        "checked_rows": compared_rows,
        "policy_teacher_action_comparison": {
            "different_frames": len(action_differences),
            "identical_frames": identical_policy_actions,
            "mean_absolute_difference_deg_deg_mm": np.mean(
                np.abs(action_differences),
                axis=0,
            ).tolist(),
            "rms_difference_deg_deg_mm": np.sqrt(
                np.mean(action_differences ** 2, axis=0)
            ).tolist(),
            "absolute_difference_p99_deg_deg_mm": np.quantile(
                np.abs(action_differences),
                0.99,
                axis=0,
            ).tolist(),
            "absolute_difference_max_deg_deg_mm": np.max(
                np.abs(action_differences),
                axis=0,
            ).tolist(),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--round2-data-dir",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "verification_report.json"
    try:
        report = verify(
            args.data_dir,
            args.raw_root,
            args.round2_data_dir,
        )
    except Exception as error:
        report_path.write_text(
            json.dumps({
                "result": "FAIL",
                "error": str(error),
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
