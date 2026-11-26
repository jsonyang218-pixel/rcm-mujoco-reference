"""Shared data-contract helpers for policy training datasets."""
from __future__ import annotations

import numpy as np


DATA_CONTRACT_VERSION = 2
DEFAULT_CONTROL_PERIOD_S = 0.05
CONTROL_PERIOD_ATOL_S = 1e-6
CONTROL_PERIOD_RTOL = 1e-3
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


def valid_window_indices(
    episode_index,
    source_frame_index,
    timestamp_s,
    *,
    n_obs_steps,
    n_action_steps,
    control_period_s=DEFAULT_CONTROL_PERIOD_S,
    oct_navigation=None,
    oct_navigation_valid=None,
    return_diagnostics=False,
):
    """Return contiguous action-block start rows shared by data and replay.

    A valid start covers ``n_obs_steps`` observations ending at the start row
    and ``n_action_steps`` supervision actions beginning at the start row.
    Every transition in that span must remain in one episode, use adjacent
    source frames, and match the control period. Observation OCT inputs must be
    finite and valid when supplied.
    """
    episode = np.asarray(episode_index, dtype=np.int64)
    source = np.asarray(source_frame_index, dtype=np.int64)
    timestamps = np.asarray(timestamp_s, dtype=np.float64)
    sample_count = len(episode)
    if not (
        len(source) == len(timestamps) == sample_count
    ):
        raise ValueError("window contract arrays must have equal length")
    n_obs_steps = int(n_obs_steps)
    n_action_steps = int(n_action_steps)
    if n_obs_steps < 1 or n_action_steps < 1:
        raise ValueError("history and action window lengths must be positive")
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamp_s contains non-finite values")
    if control_period_s is not None and float(control_period_s) <= 0:
        raise ValueError("control period must be positive")

    span = n_obs_steps + n_action_steps - 1
    if sample_count < span:
        empty = np.empty(0, dtype=np.int64)
        diagnostics = {
            "candidate_windows": 0,
            "trajectory_boundary_excluded": 0,
            "source_frame_gap_excluded": 0,
            "time_interval_excluded": 0,
            "invalid_input_excluded": 0,
            "valid_windows": 0,
        }
        return (empty, diagnostics) if return_diagnostics else empty

    starts = np.arange(n_obs_steps - 1, sample_count - n_action_steps + 1)
    trajectory_ok = np.ones(len(starts), dtype=np.bool_)
    frame_ok = np.ones(len(starts), dtype=np.bool_)
    time_ok = np.ones(len(starts), dtype=np.bool_)
    input_ok = np.ones(len(starts), dtype=np.bool_)

    # There are history-1 plus action-1 transitions in a candidate window.
    for offset in range(-(n_obs_steps - 1), n_action_steps - 1):
        left = starts + offset
        right = left + 1
        trajectory_ok &= episode[left] == episode[right]
        frame_ok &= source[right] == source[left] + 1
        time_ok &= np.isclose(
            timestamps[right] - timestamps[left],
            float(control_period_s),
            rtol=CONTROL_PERIOD_RTOL,
            atol=CONTROL_PERIOD_ATOL_S,
        )

    if oct_navigation is not None or oct_navigation_valid is not None:
        if oct_navigation is None or oct_navigation_valid is None:
            raise ValueError(
                "OCT values and validity flags must be provided together"
            )
        oct_values = np.asarray(oct_navigation, dtype=np.float64)
        oct_valid = np.asarray(oct_navigation_valid, dtype=np.bool_)
        if oct_values.shape != (sample_count, 3):
            raise ValueError("oct_navigation shape must be (N, 3)")
        if oct_valid.shape != (sample_count,):
            raise ValueError("oct_navigation_valid shape must be (N,)")
        finite_oct = np.isfinite(oct_values).all(axis=1)
        for offset in range(-(n_obs_steps - 1), 1):
            row = starts + offset
            input_ok &= oct_valid[row] & finite_oct[row]

    trajectory_count = int(trajectory_ok.sum())
    frame_count = int((trajectory_ok & frame_ok).sum())
    time_count = int((trajectory_ok & frame_ok & time_ok).sum())
    valid = trajectory_ok & frame_ok & time_ok & input_ok
    indices = starts[valid].astype(np.int64)
    diagnostics = {
        "candidate_windows": int(len(starts)),
        "trajectory_boundary_excluded": int(len(starts) - trajectory_count),
        "source_frame_gap_excluded": int(trajectory_count - frame_count),
        "time_interval_excluded": int(frame_count - time_count),
        "invalid_input_excluded": int(time_count - len(indices)),
        "valid_windows": int(len(indices)),
    }
    return (indices, diagnostics) if return_diagnostics else indices


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
