"""Convert a completed v12 raw run to the existing 224 training layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_json(path, value):
    path = Path(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _terminal_audit(episode):
    return {
        key: episode[key]
        for key in (
            "episode_index",
            "frames",
            "result",
            "strict_zero_error",
            "final_error_um",
            "visual_servo_result",
            "fallback_used",
            "fallback_reason",
            "source_episode_index",
            "source_dataset",
            "source_start_state_A_deg_B_deg_C_mm",
            "source_terminal_state_A_deg_B_deg_C_mm",
            "collection_start_state_A_deg_B_deg_C_mm",
            "endpoint_robot_ABCE",
            "terminal_robot_ABCE",
            "terminal_target_robot_ABCE",
            "terminal_tip_world_m",
            "terminal_target_world_m",
            "source_terminal_tip_world_m",
            "source_nominal_target_world_m",
            "target_minus_source_nominal_target_m",
            "target_minus_source_terminal_tip_m",
            "execution_source_counts",
            "motor_execution",
        )
    }


def convert(
        run_dir, output_dir, expected_episodes=608, require_complete=True,
        allow_manifest_extra=False):
    """Build a new dataset; never overwrite an existing result directory."""
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"training output already exists: {output_dir}")
    raw_dataset = run_dir / "dataset"
    manifest = json.loads(
        (raw_dataset / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("dataset_version") != "v12":
        raise ValueError("input manifest is not v12")
    if manifest.get("strict_zero_error") is not True:
        raise ValueError("input manifest does not declare strict zero error")
    manifest_episodes = list(manifest["episodes"])
    if allow_manifest_extra:
        if len(manifest_episodes) < int(expected_episodes):
            raise ValueError(
                f"v12 conversion expected at least {expected_episodes} episodes, "
                f"got {len(manifest_episodes)}"
            )
        episodes = manifest_episodes[:int(expected_episodes)]
    else:
        episodes = manifest_episodes
        if len(episodes) != int(expected_episodes):
            raise ValueError(
                f"v12 conversion expected {expected_episodes} episodes, "
                f"got {len(episodes)}"
            )
    if require_complete and int(expected_episodes) != 608:
        raise ValueError("complete v12 conversion requires 608 episodes")

    output_dir.mkdir(parents=True)
    data_dir = output_dir / "data"
    image_dir = output_dir / "images"
    data_dir.mkdir()
    image_dir.mkdir()

    state_rows = []
    action_rows = []
    executed_action_rows = []
    episode_rows = []
    frame_rows = []
    source_frame_rows = []
    timestamp_rows = []
    image_paths = []
    terminal_audits = []
    metadata_episodes = []

    for global_episode, episode in enumerate(episodes):
        if int(episode["episode_index"]) != global_episode:
            raise ValueError("v12 episode indexes are not contiguous")
        rows = _read_jsonl(raw_dataset / episode["jsonl"])
        audits = _read_jsonl(run_dir / episode["frame_audit"])
        if len(rows) != len(audits) or len(rows) != int(episode["frames"]):
            raise ValueError(f"episode {global_episode} row counts differ")
        if not rows[-1]["is_terminal"]:
            raise ValueError(f"episode {global_episode} lacks a terminal row")
        result_path = run_dir / f"episode_{global_episode:06d}" / "episode_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["strict_zero_error"] is not True or result["final_error_um"] != 0.0:
            raise ValueError(f"episode {global_episode} is not strict zero error")
        terminal_audits.append(_terminal_audit(result))

        for local_frame, (row, audit) in enumerate(zip(rows, audits)):
            if int(row["frame_index"]) != local_frame:
                raise ValueError(f"episode {global_episode} frames are not contiguous")
            source = (raw_dataset / row["observation.images.microscope_rgb"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = image_dir / (
                f"ep{global_episode:06d}_frame{local_frame:06d}.png"
            )
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                if rgb.size != (224, 224):
                    rgb = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                rgb.save(destination)
            relative = destination.relative_to(output_dir).as_posix()
            state = np.asarray(
                row["observation.state"], dtype=np.float32
            )
            absolute_action = np.asarray(row["action"], dtype=np.float32)
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(absolute_action)):
                raise ValueError(f"episode {global_episode} contains non-finite data")
            state_rows.append(state)
            delta_action = (absolute_action - state).astype(np.float32)
            action_rows.append(delta_action)
            executed_action_rows.append(delta_action.copy())
            episode_rows.append(global_episode)
            frame_rows.append(local_frame)
            source_frame_rows.append(local_frame)
            timestamp_rows.append(local_frame * 0.05)
            image_paths.append(relative)

        metadata_episodes.append({
            "episode_index": global_episode,
            "num_frames": len(rows),
            "success": True,
            "final_error_um": 0.0,
            "strict_zero_error": True,
            "visual_servo_result": episode["visual_servo_result"],
            "fallback_used": bool(episode["fallback_used"]),
            "source_dataset": episode["source_dataset"],
            "source_episode_index": int(episode["source_episode_index"]),
            "terminal_audit_index": global_episode,
        })

    states = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    executed_actions = np.asarray(
        executed_action_rows,
        dtype=np.float32,
    )
    episode_indices = np.asarray(episode_rows, dtype=np.int64)
    frame_indices = np.asarray(frame_rows, dtype=np.int32)
    source_frame_indices = np.asarray(source_frame_rows, dtype=np.int64)
    timestamps = np.asarray(timestamp_rows, dtype=np.float64)
    zero_navigation = np.zeros((len(states), 3), dtype=np.float32)
    invalid_navigation = np.zeros(len(states), dtype=np.bool_)
    _validate_training_arrays(
        states,
        actions,
        episode_indices,
        frame_indices,
    )
    np.savez(
        data_dir / "dataset.npz",
        episode_index=episode_indices,
        frame_index=frame_indices,
        action=actions,
        state=states,
        executed_action=executed_actions,
        source_frame_index=source_frame_indices,
        timestamp_s=timestamps,
        oct_navigation=zero_navigation,
        oct_navigation_valid=invalid_navigation,
    )
    _write_json(data_dir / "image_paths.json", image_paths)
    _write_json(data_dir / "terminal_audits.json", terminal_audits)

    metadata = {
        "num_episodes": len(metadata_episodes),
        "num_frames": len(states),
        "fps": 20,
        "data_contract_version": 2,
        "action_supervision_source": "accepted_executed_command",
        "history_action_source": "executed_action",
        "continuity_source": "source_frame_index",
        "dataset_version": "v12",
        "strict_zero_error": True,
        "target_semantics": (
            "target copied bit-exactly from stabilized source-terminal needle tip"
        ),
        "success_definition": (
            "terminal_tip_world_m component-wise bit-equal to "
            "terminal_target_world_m; float64 Euclidean error exactly 0.0 um"
        ),
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
                "available": False,
            },
        },
        "action_space": {
            "shape": [3],
            "dtype": "float32",
            "names": ["dA_deg", "dB_deg", "dC_mm"],
            "type": "delta",
            "raw_action_semantics": (
                "quantized MotorBackend absolute A/B/C target for the "
                "following interval; converter stores target minus current state"
            ),
        },
        "action_source": "executed",
        "policy_input_fields": [
            "observation.images.microscope_rgb",
            "observation.state",
            "task",
        ],
        "excluded_from_policy_inputs": [
            "target coordinates",
            "terminal truth error",
            "execution source",
            "fallback metadata",
            "source episode identifiers",
        ],
        "episodes": metadata_episodes,
    }
    _write_json(output_dir / "metadata.json", metadata)
    return {
        "result": "PASS",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "episodes": len(metadata_episodes),
        "frames": len(states),
        "state_dtype": str(states.dtype),
        "action_dtype": str(actions.dtype),
        "image_shape": [224, 224, 3],
        "terminal_audits": str(data_dir / "terminal_audits.json"),
    }


def _validate_training_arrays(states, actions, episode_indices, frame_indices):
    if len(states) == 0:
        raise ValueError("training dataset is empty")
    if states.shape != actions.shape or states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("state and action arrays must both be (N, 3)")
    if states.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError("state and action arrays must be float32")
    if len(episode_indices) != len(states) or len(frame_indices) != len(states):
        raise ValueError("episode/frame arrays are not row aligned")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("state or action contains non-finite values")
    for episode in np.unique(episode_indices):
        rows = episode_indices == episode
        expected = np.arange(int(rows.sum()), dtype=np.int32)
        if not np.array_equal(frame_indices[rows], expected):
            raise ValueError(f"episode {episode} frame indexes are not contiguous")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=608)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-prefix", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(convert(
        args.run_dir,
        args.output_dir,
        expected_episodes=args.expected_episodes,
        require_complete=not args.allow_partial,
        allow_manifest_extra=args.allow_prefix,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
