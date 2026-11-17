"""Regressions for action history, time continuity, and terminal success."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from lerobot_training.convert_to_lerobot import V9ToLeRobotConverter
from lerobot_training.eval_policy import (
    advance_terminal_success_streak,
    terminal_success_frame,
)
from lerobot_training.train_diffusion_policy import RCMDataset
from lerobot_training.summarize_flow_zero_error_evaluation import (
    _is_canonical_success,
)


def _write_dataset(root: Path, *, include_history: bool):
    data_dir = root / "data"
    data_dir.mkdir()
    frame_index = np.arange(5, dtype=np.int32)
    arrays = {
        "episode_index": np.zeros(5, dtype=np.int64),
        "frame_index": frame_index,
        "state": np.arange(15, dtype=np.float32).reshape(5, 3) * 0.01,
        "action": np.arange(15, dtype=np.float32).reshape(5, 3) + 10.0,
        "oct_navigation": np.zeros((5, 3), dtype=np.float32),
        "oct_navigation_valid": np.ones(5, dtype=np.bool_),
    }
    if include_history:
        arrays.update({
            "executed_action": (
                np.arange(15, dtype=np.float32).reshape(5, 3) + 1.0
            ),
            "source_frame_index": np.asarray(
                [0, 2, 3, 4, 5],
                dtype=np.int64,
            ),
            "timestamp_s": np.asarray(
                [0.0, 0.1, 0.2, 0.3, 0.4],
                dtype=np.float64,
            ),
        })
    np.savez(data_dir / "dataset.npz", **arrays)
    (data_dir / "image_paths.json").write_text(
        json.dumps(["missing.png"] * 5),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps({
            "num_episodes": 1,
            "num_frames": 5,
            "fps": 20,
            "data_contract_version": 2 if include_history else 1,
        }),
        encoding="utf-8",
    )


class ConverterContractTests(unittest.TestCase):
    def test_teacher_label_and_executed_command_are_distinct(self):
        with tempfile.TemporaryDirectory(
            prefix="rcm_teacher_contract_test_"
        ) as temporary:
            root = Path(temporary)
            images = root / "raw" / "episode_000000" / "images"
            images.mkdir(parents=True)
            for frame in (0, 2):
                path = images / f"frame_{frame:06d}.png"
                Image.fromarray(
                    np.full((4, 4, 3), frame, dtype=np.uint8)
                ).save(path)
            rows = [
                {
                    "teacher_valid": True,
                    "state_A_deg_B_deg_C_mm": [0.0, 0.0, 0.0],
                    "teacher_action_deg_deg_mm": [0.1, -0.2, 0.3],
                    "executed_action_deg_deg_mm": [0.05, -0.1, 0.15],
                    "timestamp_s": 0.0,
                    "image_path": "frame_000000.png",
                    "oct_navigation_mm": [1.0, 2.0, 3.0],
                    "oct_navigation_valid": True,
                },
                {
                    "teacher_valid": False,
                    "timestamp_s": 0.05,
                },
                {
                    "teacher_valid": True,
                    "state_A_deg_B_deg_C_mm": [0.02, -0.01, 0.001],
                    "teacher_action_deg_deg_mm": [0.2, 0.1, -0.2],
                    "executed_action_deg_deg_mm": [0.1, 0.05, -0.1],
                    "timestamp_s": 0.1,
                    "image_path": "frame_000002.png",
                    "oct_navigation_mm": [2.0, 3.0, 4.0],
                    "oct_navigation_valid": True,
                },
            ]
            converter = V9ToLeRobotConverter(
                str(root / "raw"),
                str(root / "converted"),
                action_source="teacher",
                image_size=2,
            )
            frames = converter.extract_frames(
                {
                    "dagger_samples": rows,
                    "images_dir": images,
                },
                0,
            )
            self.assertEqual(
                [frame["frame_index"] for frame in frames],
                [0, 1],
            )
            self.assertEqual(
                [frame["source_frame_index"] for frame in frames],
                [0, 2],
            )
            np.testing.assert_allclose(
                frames[0]["delta_action"],
                [0.1, -0.2, 0.3],
            )
            np.testing.assert_allclose(
                frames[0]["executed_action"],
                [0.05, -0.1, 0.15],
            )
            np.testing.assert_allclose(
                [frame["timestamp"] for frame in frames],
                [0.0, 0.1],
            )

            converter.save_lerobot_format(frames)
            with np.load(
                root / "converted" / "data" / "dataset.npz"
            ) as saved:
                np.testing.assert_array_equal(
                    saved["source_frame_index"],
                    [0, 2],
                )
                np.testing.assert_allclose(
                    saved["executed_action"][0],
                    [0.05, -0.1, 0.15],
                )


class HistoryContractTests(unittest.TestCase):
    def test_history_uses_executed_action_and_timestamp_duration(self):
        with tempfile.TemporaryDirectory(
            prefix="rcm_history_contract_test_"
        ) as temporary:
            root = Path(temporary)
            _write_dataset(root, include_history=True)
            dataset = RCMDataset(
                root,
                n_obs_steps=2,
                n_action_steps=2,
                include_velocity=True,
                include_previous_action=True,
            )
            self.assertEqual(dataset.valid_indices, [2, 3])

            velocity, previous_action, valid = dataset.history_features([3])
            self.assertTrue(valid[0])
            np.testing.assert_allclose(
                velocity[0],
                (dataset.states[3] - dataset.states[2]) / 0.1,
            )
            np.testing.assert_allclose(
                previous_action[0],
                dataset.executed_actions[2],
            )
            self.assertFalse(np.allclose(
                previous_action[0],
                dataset.actions[2],
            ))

            motion_velocity, motion_previous = dataset.motion_samples()
            self.assertEqual(len(motion_velocity), 3)
            np.testing.assert_allclose(
                motion_previous,
                dataset.executed_actions[[1, 2, 3]],
            )

    def test_legacy_dataset_fails_history_training(self):
        with tempfile.TemporaryDirectory(
            prefix="rcm_legacy_contract_test_"
        ) as temporary:
            root = Path(temporary)
            _write_dataset(root, include_history=False)
            with self.assertRaisesRegex(
                ValueError,
                "executed_action",
            ):
                RCMDataset(
                    root,
                    n_obs_steps=2,
                    n_action_steps=2,
                    include_previous_action=True,
                )


class TerminalSuccessTests(unittest.TestCase):
    def test_low_speed_alone_does_not_pass(self):
        self.assertFalse(
            terminal_success_frame(0.001, 0.02, encoder_settled=False)
        )

    def test_encoder_settle_requires_three_consecutive_frames(self):
        streak = 0
        for frame in range(3):
            streak, success = advance_terminal_success_streak(
                streak,
                terminal_success_frame(0.001, 0.02, True),
            )
            self.assertEqual(success, frame == 2)
        _, reset_success = advance_terminal_success_streak(
            streak,
            terminal_success_frame(0.001, 0.02, False),
        )
        self.assertFalse(reset_success)

    def test_legacy_success_is_not_reported_as_canonical(self):
        self.assertFalse(_is_canonical_success({
            "success": True,
            "_evaluation_contract": {},
        }))
        self.assertTrue(_is_canonical_success({
            "success": True,
            "_evaluation_contract": {
                "version": 2,
                "terminal_condition": "MotorExecution.settled",
            },
        }))


if __name__ == "__main__":
    unittest.main(verbosity=2)
