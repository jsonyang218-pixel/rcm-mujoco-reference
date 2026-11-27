"""Verify v12 source provenance, exact terminal truth, and training artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.datasets.collect_zero_error_dataset import (
    EXECUTION_SOURCES,
    load_source_episodes,
    sha256_file,
)
from scripts.verification.verify_v9_dataset import read_rows, verify


def _inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path escapes root: {relative}")
    return path


def _absolute_error_um(tip_m, target_m):
    tip = np.asarray(tip_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    if tip.shape != (3,) or target.shape != (3,):
        raise ValueError("terminal coordinates must be length 3")
    return float(np.linalg.norm((tip - target) * 1000.0))


def verify_run(
        run_dir, source_dataset, expected_episodes=608, require_complete=True,
        allow_manifest_extra=False):
    """Verify raw v12 data and exact source/terminal traceability."""
    run_dir = Path(run_dir).resolve()
    source_dataset = Path(source_dataset).resolve()
    expected = load_source_episodes(source_dataset)
    if expected_episodes < 1 or expected_episodes > len(expected):
        raise ValueError("expected_episodes is outside the source dataset")

    protocol = json.loads(
        (run_dir / "collection_protocol.json").read_text(encoding="utf-8")
    )
    if protocol.get("dataset_version") != "v12":
        raise ValueError("run is not a v12 collection")
    current_source_hashes = {
        "source_dataset.npz": sha256_file(source_dataset / "data" / "dataset.npz"),
        "source_metadata.json": sha256_file(source_dataset / "metadata.json"),
    }
    for name, digest in current_source_hashes.items():
        assert protocol["source_sha256"][name] == digest

    base = verify(run_dir)
    assert base["result"] == "PASS"
    manifest = json.loads(
        (run_dir / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.get("dataset_version") == "v12"
    assert manifest.get("strict_zero_error") is True
    assert manifest.get("execution_backend") == "MotorBackend"
    assert manifest.get("perception_backend") == "strict_rgbd"
    manifest_episodes = list(manifest["episodes"])
    if allow_manifest_extra:
        assert len(manifest_episodes) >= expected_episodes
        episodes = manifest_episodes[:expected_episodes]
    else:
        assert len(manifest_episodes) == expected_episodes
        episodes = manifest_episodes
    if require_complete:
        assert expected_episodes == len(expected)
    assert [int(row["episode_index"]) for row in episodes] == list(
        range(expected_episodes)
    )

    fallback_episodes = 0
    visual_pass_episodes = 0
    source_shifts_m = []
    execution_totals = {name: 0 for name in EXECUTION_SOURCES}
    episode_results = []
    for source, episode in zip(expected[:expected_episodes], episodes):
        index = int(episode["episode_index"])
        assert episode["result"] == "PASS"
        assert episode["strict_zero_error"] is True
        assert episode["final_error_um"] == 0.0
        assert episode["source_episode_index"] == source.source_episode_index
        assert episode["source_dataset"] == source.source_dataset
        np.testing.assert_array_equal(
            episode["source_start_state_A_deg_B_deg_C_mm"],
            source.start_state_A_deg_B_deg_C_mm,
        )
        np.testing.assert_array_equal(
            episode["source_terminal_state_A_deg_B_deg_C_mm"],
            source.terminal_state_A_deg_B_deg_C_mm,
        )
        result_path = run_dir / f"episode_{index:06d}" / "episode_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["episode_index"] == index
        assert result["strict_zero_error"] is True
        assert result["final_error_um"] == 0.0
        assert result["visual_servo_result"] in ("PASS", "FAIL")
        if result["visual_servo_result"] == "PASS":
            assert result["fallback_used"] is False
            visual_pass_episodes += 1
        else:
            assert result["fallback_used"] is True
            assert result["fallback_mode"] == (
                "privileged_joint_space_velocity_acceleration_limited"
            )
            fallback_episodes += 1

        tip = result["terminal_tip_world_m"]
        target = result["terminal_target_world_m"]
        if list(tip) != list(target):
            raise AssertionError(f"episode {index} terminal coordinates are not bit-equal")
        if _absolute_error_um(tip, target) != 0.0:
            raise AssertionError(f"episode {index} terminal error is not exactly 0.0")
        assert result["motor_execution"]["encoder_settled"] is True
        assert not result["motor_execution"]["fault"]
        assert result["terminal_robot_ABCE"] == result["endpoint_robot_ABCE"]
        assert result["terminal_target_robot_ABCE"] == result["endpoint_robot_ABCE"]
        assert result["target_minus_source_terminal_tip_m"] == [0.0, 0.0, 0.0]
        source_shift = np.asarray(
            result["target_minus_source_nominal_target_m"], dtype=np.float64
        )
        source_shifts_m.append(source_shift)

        servo_report = json.loads(
            _inside(run_dir, episode["servo_report"]).read_text(encoding="utf-8")
        )
        assert servo_report["result"] == "PASS"
        assert servo_report["visual_servo_result"] == result["visual_servo_result"]
        assert servo_report["final_tip_target_error_um"] == 0.0
        assert servo_report["strict_zero_error"] is True
        assert servo_report["target_coordinates_are_privileged_audit_only"] is True
        assert servo_report["policy_inputs_exclude_target_truth"] is True

        audits = read_rows(_inside(run_dir, episode["frame_audit"]))
        assert audits[-1]["terminal"] is True
        assert audits[-1]["execution_source"] == "hold"
        assert not any(row["terminal"] for row in audits[:-1])
        assert all(
            row["execution_source"] in EXECUTION_SOURCES
            for row in audits
        )
        audit_source_counts = {
            name: sum(
                row["execution_source"] == name
                for row in audits
            )
            for name in EXECUTION_SOURCES
        }
        assert episode["execution_source_counts"] == audit_source_counts
        if result["fallback_used"]:
            assert audit_source_counts["privileged_joint_fallback"] > 0
        else:
            assert audit_source_counts["privileged_joint_fallback"] == 0
        terminal_target = np.asarray(target, dtype=np.float64)
        for row in audits:
            assert np.array_equal(
                np.asarray(row["target_world_m"], dtype=np.float64),
                terminal_target,
            )
            execution_totals[row["execution_source"]] += 1
        assert np.array_equal(
            np.asarray(audits[-1]["tip_world_m"], dtype=np.float64),
            terminal_target,
        )
        assert audits[-1]["tip_target_error_um"] == 0.0
        with np.load(
                _inside(run_dir / "dataset", episode["policy_inputs"]),
                allow_pickle=False) as policy:
            assert set(policy.files) == {
                "observation.images.microscope_rgb",
                "observation.state",
                "task",
            }
            assert len(set(policy["task"].tolist())) == 1
            task = policy["task"][0]
            assert "不提供目标三维坐标" in task
        episode_results.append({
            "episode_index": index,
            "frames": int(episode["frames"]),
            "visual_servo_result": result["visual_servo_result"],
            "fallback_used": bool(result["fallback_used"]),
            "source_dataset": source.source_dataset,
            "source_episode_index": source.source_episode_index,
            "source_target_shift_m": source_shift.tolist(),
        })

    shifts = np.asarray(source_shifts_m, dtype=np.float64).reshape(-1, 3)
    return {
        "result": "PASS",
        "dataset_version": "v12",
        "scope": (
            "MuJoCo simulation target truth, exact motor/encoder replay, and "
            "policy-input isolation; not physical or clinical accuracy"
        ),
        "episodes": len(episodes),
        "frames": int(manifest["total_frames"]),
        "expected_episodes": expected_episodes,
        "complete_dataset": bool(
            expected_episodes == len(expected) == len(manifest_episodes)
        ),
        "manifest_episodes": len(manifest_episodes),
        "all_terminal_errors_exactly_zero": True,
        "visual_pass_episodes": visual_pass_episodes,
        "fallback_episodes": fallback_episodes,
        "fallback_fraction": fallback_episodes / len(episodes),
        "execution_source_frames": execution_totals,
        "source_target_shift_m": {
            "mean": shifts.mean(axis=0).tolist(),
            "mean_absolute": np.mean(np.abs(shifts), axis=0).tolist(),
            "maximum_absolute": np.max(np.abs(shifts), axis=0).tolist(),
            "norm_mean_m": float(np.mean(np.linalg.norm(shifts, axis=1))),
            "norm_max_m": float(np.max(np.linalg.norm(shifts, axis=1))),
        },
        "base_verification": base,
        "episode_results": episode_results,
    }


def verify_training_dataset(run_dir, training_dir, expected_episodes=608):
    """Verify the final 224 container and its propagated terminal audits."""
    run_dir = Path(run_dir).resolve()
    training_dir = Path(training_dir).resolve()
    metadata = json.loads(
        (training_dir / "metadata.json").read_text(encoding="utf-8")
    )
    audits = json.loads(
        (training_dir / "data" / "terminal_audits.json").read_text(encoding="utf-8")
    )
    image_paths = json.loads(
        (training_dir / "data" / "image_paths.json").read_text(encoding="utf-8")
    )
    with np.load(training_dir / "data" / "dataset.npz", allow_pickle=False) as data:
        expected_keys = {
            "episode_index",
            "frame_index",
            "action",
            "state",
            "oct_navigation",
            "oct_navigation_valid",
        }
        assert set(data.files) == expected_keys
        episodes = data["episode_index"].copy()
        frames = data["frame_index"].copy()
        actions = data["action"].copy()
        states = data["state"].copy()

    assert metadata["dataset_version"] == "v12"
    assert metadata["strict_zero_error"] is True
    assert metadata["num_episodes"] == expected_episodes == len(audits)
    assert len(image_paths) == len(episodes) == len(states) == len(actions)
    assert metadata["num_frames"] == len(states)
    assert states.dtype == np.float32 and actions.dtype == np.float32
    assert states.shape == actions.shape == (len(states), 3)
    assert not np.any(np.isnan(states)) and not np.any(np.isnan(actions))
    assert metadata["observation_space"]["image"]["shape"] == [224, 224, 3]
    assert metadata["observation_space"]["state"]["dtype"] == "float32"
    assert metadata["action_space"]["dtype"] == "float32"
    assert metadata["policy_input_fields"] == [
        "observation.images.microscope_rgb",
        "observation.state",
        "task",
    ]
    forbidden_columns = {
        "target_world_m",
        "terminal_tip_world_m",
        "final_error_um",
        "execution_source",
        "fallback_used",
        "source_episode_index",
    }
    assert forbidden_columns.isdisjoint(metadata["observation_space"])

    for episode in range(expected_episodes):
        rows = episodes == episode
        assert rows.any()
        assert np.array_equal(
            frames[rows],
            np.arange(int(rows.sum()), dtype=frames.dtype),
        )
        audit = audits[episode]
        assert audit["episode_index"] == episode
        assert audit["strict_zero_error"] is True
        assert audit["final_error_um"] == 0.0
        assert audit["terminal_tip_world_m"] == audit["terminal_target_world_m"]
        assert _absolute_error_um(
            audit["terminal_tip_world_m"],
            audit["terminal_target_world_m"],
        ) == 0.0
        assert metadata["episodes"][episode]["num_frames"] == int(rows.sum())
        assert metadata["episodes"][episode]["terminal_audit_index"] == episode

    for relative in image_paths:
        path = _inside(training_dir, relative)
        assert path.is_file()
    for relative in image_paths[::max(1, len(image_paths) // 128)]:
        with Image.open(_inside(training_dir, relative)) as image:
            assert image.mode == "RGB" and image.size == (224, 224)

    raw_manifest = json.loads(
        (run_dir / "dataset" / "manifest.json").read_text(encoding="utf-8")
    )
    raw_episodes = list(raw_manifest["episodes"])
    if len(raw_episodes) < expected_episodes:
        raise AssertionError("raw manifest has fewer episodes than the training prefix")
    expected_raw_frames = sum(
        int(episode["frames"])
        for episode in raw_episodes[:expected_episodes]
    )
    assert metadata["num_frames"] == expected_raw_frames
    return {
        "result": "PASS",
        "training_dir": str(training_dir),
        "episodes": expected_episodes,
        "frames": len(states),
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "state_dtype": str(states.dtype),
        "action_dtype": str(actions.dtype),
        "image_shape": [224, 224, 3],
        "all_terminal_errors_exactly_zero": True,
        "policy_input_truth_leak_check": "PASS",
        "training_executed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("data/lerobot_rcm_train_guided_dagger1_full_224"),
    )
    parser.add_argument("--expected-episodes", type=int, default=608)
    parser.add_argument("--training-dir", type=Path)
    args = parser.parse_args(argv)
    report = verify_run(
        args.run_dir,
        args.source_dataset,
        expected_episodes=args.expected_episodes,
        require_complete=(args.expected_episodes == 608),
        allow_manifest_extra=(args.expected_episodes == 1),
    )
    if args.training_dir is not None:
        report["training_verification"] = verify_training_dataset(
            args.run_dir,
            args.training_dir,
            expected_episodes=args.expected_episodes,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
