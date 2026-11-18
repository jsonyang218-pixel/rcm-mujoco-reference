"""Index-backed transition replay for the RCM v2 dataset (no image copies)."""
from __future__ import annotations

import numpy as np

try:
    from .dataset_contract import adjacent_sample_mask
except ImportError:
    from dataset_contract import adjacent_sample_mask


# Match the existing evaluator's action limits; these do not change execution.
ACTION_LIMIT = np.array(
    [0.2897999882698059, 0.14434358477592468, 0.04271206632256508],
    dtype=np.float32,
)
RECORD_DTYPE = np.dtype([
    ("source", np.int32), ("row", np.int64), ("next_row", np.int64),
    ("reward", np.float32), ("terminated", np.bool_),
])


def transition_records(
    dataset, source, *, n_obs_steps=2, control_period_s=0.05,
    distance_scale_mm=1.0, progress_weight=1.0, distance_weight=0.1,
    effort_weight=0.001, terminated=None,
):
    """Pair consecutive observations with the action actually executed at s.

    ``action`` contains expert labels and MUST NOT be used as the transition's
    executed action. Source-frame gaps, episode boundaries and period changes
    cannot be treated as one-step transitions. Missing terminal labels describe
    a continuing positioning objective; dataset ends are not success labels.
    """
    count = len(dataset.states)
    if n_obs_steps < 2 or control_period_s <= 0 or distance_scale_mm <= 0:
        raise ValueError("observation count, control period and scale are invalid")
    if not dataset.has_history_contract:
        raise ValueError("replay requires the executed-action v2 data contract")
    if terminated is not None and np.asarray(terminated).shape != (count,):
        raise ValueError("terminated must have one boolean per observation")
    if count <= n_obs_steps:
        return np.empty(0, dtype=RECORD_DTYPE)

    incoming = np.zeros(count, dtype=np.bool_)
    incoming[1:] = adjacent_sample_mask(
        dataset.episode_indices, dataset.source_frame_indices,
    ) & np.isclose(
        np.diff(dataset.timestamps_s), control_period_s, rtol=1e-3, atol=1e-6,
    )
    rows = np.arange(n_obs_steps - 1, count - 1)
    keep = incoming[rows + 1].copy()
    for offset in range(n_obs_steps - 1):
        keep &= incoming[rows - offset]
    oct_values = np.asarray(dataset.oct_navigation, dtype=np.float64)
    finite_oct = np.isfinite(oct_values).all(axis=1)
    valid = np.asarray(dataset.oct_navigation_valid, dtype=np.bool_).copy()
    valid &= finite_oct
    keep &= valid[rows] & valid[rows + 1]
    for offset in range(1, n_obs_steps):
        keep &= finite_oct[rows - offset]
    rows = rows[keep]
    current_distance = np.linalg.norm(oct_values[rows], axis=1)
    next_distance = np.linalg.norm(oct_values[rows + 1], axis=1)
    executed = np.asarray(dataset.executed_actions[rows], dtype=np.float64)
    rewards = (
        progress_weight * (current_distance - next_distance) / distance_scale_mm
        - distance_weight * next_distance / distance_scale_mm
        - effort_weight * np.mean((executed / ACTION_LIMIT) ** 2, axis=1)
    )
    if not np.isfinite(rewards).all():
        raise ValueError("non-finite transition rewards")
    records = np.empty(len(rows), dtype=RECORD_DTYPE)
    records["source"] = source
    records["row"] = rows
    records["next_row"] = rows + 1
    records["reward"] = rewards
    records["terminated"] = (
        False if terminated is None else np.asarray(terminated, dtype=bool)[rows]
    )
    return records


class TransitionReplayBuffer:
    """Uniform replay with reservoir retention when capacity is exceeded."""

    def __init__(self, capacity=300_000, seed=20261117):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.records = np.empty(0, dtype=RECORD_DTYPE)
        self.seen = 0

    def __len__(self):
        return len(self.records)

    def extend(self, records):
        records = np.asarray(records, dtype=RECORD_DTYPE)
        # Fill before reservoir replacement; older sources retain equal chance.
        available = min(self.capacity - len(self.records), len(records))
        if available:
            self.records = np.concatenate([self.records, records[:available]])
            self.seen += available
        for record in records[available:]:
            self.seen += 1
            index = int(self.rng.integers(self.seen))
            if index < self.capacity:
                self.records[index] = record

    def sample(self, batch_size):
        if not len(self) or batch_size < 1:
            raise ValueError("sampling requires a nonempty buffer and batch")
        return self.records[self.rng.integers(len(self), size=batch_size)].copy()

    def save(self, path):
        np.savez_compressed(path, records=self.records, seen=np.int64(self.seen))
