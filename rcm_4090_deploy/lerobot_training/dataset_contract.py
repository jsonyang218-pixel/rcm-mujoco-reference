"""Shared data-contract helpers for policy training datasets."""
from __future__ import annotations

import numpy as np


DATA_CONTRACT_VERSION = 2
CORE_ARRAYS = (
    "episode_index",
    "frame_index",
    "state",
    "action",
    "oct_navigation",
    "oct_navigation_valid",
)
HISTORY_ARRAYS = (
    "executed_action",
    "source_frame_index",
    "timestamp_s",
)


def validate_dataset_contract(arrays, *, require_history):
    """Validate row alignment and history/continuity fields.

    Legacy datasets remain usable for models that do not consume velocity or a
    previous action. History-enabled training is strict: an expert action is not
    an acceptable substitute for the command that was actually accepted.
    """
    missing_core = [name for name in CORE_ARRAYS if name not in arrays]
    if missing_core:
        raise ValueError(f"dataset lacks core arrays: {missing_core}")

    sample_count = len(np.asarray(arrays["state"]))
    if sample_count == 0:
        raise ValueError("dataset has no samples")
    for name in CORE_ARRAYS:
        if len(np.asarray(arrays[name])) != sample_count:
            raise ValueError(
                f"{name} length {len(np.asarray(arrays[name]))} does not "
                f"match state length {sample_count}"
            )
    if np.asarray(arrays["state"]).shape != (sample_count, 3):
        raise ValueError("state shape must be (N, 3)")
    if np.asarray(arrays["action"]).shape != (sample_count, 3):
        raise ValueError("action shape must be (N, 3)")
    if np.asarray(arrays["oct_navigation"]).shape != (sample_count, 3):
        raise ValueError("oct_navigation shape must be (N, 3)")
    if np.asarray(arrays["oct_navigation_valid"]).shape != (sample_count,):
        raise ValueError("oct_navigation_valid shape must be (N,)")
    if not np.all(np.isfinite(np.asarray(arrays["state"], dtype=np.float64))):
        raise ValueError("state contains non-finite values")
    if not np.all(np.isfinite(np.asarray(arrays["action"], dtype=np.float64))):
        raise ValueError("action contains non-finite values")

    present_history = [name for name in HISTORY_ARRAYS if name in arrays]
    if present_history and len(present_history) != len(HISTORY_ARRAYS):
        missing = sorted(set(HISTORY_ARRAYS) - set(present_history))
        raise ValueError(
            "dataset has a partial v2 data contract; missing "
            f"{missing}"
        )
    if require_history and not present_history:
        raise ValueError(
            "velocity/previous-action training requires executed_action, "
            "source_frame_index and timestamp_s; reconvert teacher data or "
            "use a no-history model"
        )
    if not present_history:
        return

    executed_action = np.asarray(arrays["executed_action"])
    source_frame_index = np.asarray(arrays["source_frame_index"])
    timestamp_s = np.asarray(arrays["timestamp_s"])
    if executed_action.shape != (sample_count, 3):
        raise ValueError("executed_action shape must be (N, 3)")
    if source_frame_index.shape != (sample_count,):
        raise ValueError("source_frame_index shape must be (N,)")
    if timestamp_s.shape != (sample_count,):
        raise ValueError("timestamp_s shape must be (N,)")
    if not np.all(
        np.isfinite(np.asarray(executed_action, dtype=np.float64))
    ):
        raise ValueError("executed_action contains non-finite values")
    if not np.all(np.isfinite(np.asarray(timestamp_s, dtype=np.float64))):
        raise ValueError("timestamp_s contains non-finite values")

    episode_index = np.asarray(arrays["episode_index"], dtype=np.int64)
    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        source = np.asarray(source_frame_index[rows], dtype=np.int64)
        timestamps = np.asarray(timestamp_s[rows], dtype=np.float64)
        if np.any(np.diff(source) <= 0):
            raise ValueError(
                f"episode {int(episode)} source_frame_index is not increasing"
            )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f"episode {int(episode)} timestamp_s is not strictly increasing"
            )


def adjacent_sample_mask(episode_index, source_frame_index):
    """Return a mask selecting rows immediately following a source frame."""
    if len(episode_index) < 2:
        return np.zeros(0, dtype=np.bool_)
    episode = np.asarray(episode_index, dtype=np.int64)
    source = np.asarray(source_frame_index, dtype=np.int64)
    return (
        (episode[1:] == episode[:-1])
        & (source[1:] == source[:-1] + 1)
    )


def motion_samples(
    episode_index,
    source_frame_index,
    timestamp_s,
    state,
    executed_action,
):
    """Return only motion samples with a one-frame, positive-duration history."""
    mask = adjacent_sample_mask(episode_index, source_frame_index)
    if not np.any(mask):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    timestamps = np.asarray(timestamp_s, dtype=np.float64)
    duration = timestamps[1:] - timestamps[:-1]
    if np.any(duration[mask] <= 0):
        raise ValueError("adjacent source frames have non-positive duration")
    velocity = (
        np.asarray(state[1:], dtype=np.float64)
        - np.asarray(state[:-1], dtype=np.float64)
    )[mask] / duration[mask, None]
    previous_action = np.asarray(executed_action)[:-1][mask]
    return (
        velocity.astype(np.float32),
        np.asarray(previous_action, dtype=np.float32),
    )


def history_features(
    rows,
    episode_index,
    source_frame_index,
    timestamp_s,
    state,
    executed_action,
):
    """Build velocity/previous-action features for selected dataset rows."""
    selected = np.asarray(rows, dtype=np.int64)
    velocity = np.zeros((len(selected), 3), dtype=np.float32)
    previous_action = np.zeros((len(selected), 3), dtype=np.float32)
    valid = selected > 0
    previous = np.maximum(selected - 1, 0)
    if np.any(valid):
        episode = np.asarray(episode_index, dtype=np.int64)
        source = np.asarray(source_frame_index, dtype=np.int64)
        valid &= episode[previous] == episode[selected]
        valid &= source[selected] == source[previous] + 1
    if np.any(valid):
        timestamps = np.asarray(timestamp_s, dtype=np.float64)
        duration = timestamps[selected[valid]] - timestamps[previous[valid]]
        if np.any(duration <= 0):
            raise ValueError("history rows have non-positive duration")
        velocity[valid] = (
            (
                np.asarray(state[selected[valid]], dtype=np.float64)
                - np.asarray(state[previous[valid]], dtype=np.float64)
            )
            / duration[:, None]
        ).astype(np.float32)
        previous_action[valid] = np.asarray(
            executed_action[previous[valid]],
            dtype=np.float32,
        )
    return velocity, previous_action, valid
