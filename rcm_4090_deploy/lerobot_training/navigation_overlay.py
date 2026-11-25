"""Checked, immutable-source OCT and C-label overlays for replay training."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path, dataset_sha256, count):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if str(values["dataset_sha256"]) != dataset_sha256:
        raise ValueError("overlay source dataset SHA256 mismatch")
    rows = values["rows"]
    if (rows.dtype != np.int64 or rows.ndim != 1
            or (len(rows) and (rows[0] < 0 or rows[-1] >= count
                              or np.any(np.diff(rows) <= 0)))):
        raise ValueError("overlay rows must be unique, sorted int64 source rows")
    return values, rows


def apply_oct_overlay(oct_values, valid, path, dataset_sha256):
    values, rows = _load(path, dataset_sha256, len(oct_values))
    before, after = values["old_oct_mm"], values["new_oct_mm"]
    before_valid, after_valid = values["old_valid"], values["new_valid"]
    if (before.shape != (len(rows), 3) or after.shape != before.shape
            or before_valid.shape != rows.shape or after_valid.shape != rows.shape
            or before_valid.dtype != np.bool_ or after_valid.dtype != np.bool_
            or not np.array_equal(oct_values[rows], before, equal_nan=True)
            or not np.array_equal(valid[rows], before_valid)
            or not np.isfinite(after).all()):
        raise ValueError("invalid OCT overlay values or source alignment")
    result, result_valid = oct_values.copy(), valid.copy()
    result[rows], result_valid[rows] = after, after_valid
    return result, result_valid


def apply_c_overlay(actions, path, dataset_sha256):
    """Accept composed labels against original actions, including positive C."""
    values, rows = _load(path, dataset_sha256, len(actions))
    before, after = values["old_dC_mm"], values["new_dC_mm"]
    if (before.shape != rows.shape or after.shape != rows.shape
            or not np.array_equal(actions[rows, 2], before)
            or not np.isfinite(after).all()
            or np.any(after < 0) or np.any(after > .030 + 1e-9)):
        raise ValueError("invalid C overlay or 0.030 mm/step cap exceeded")
    result = actions.copy()
    result[rows, 2] = after
    return result


def apply_dataset_overlays(dataset, *, oct_overlay=None, c_overlay=None):
    """Apply before splitting source pools; recompute OCT-dependent windows."""
    if oct_overlay is None and c_overlay is None:
        return {}
    source_sha = file_sha256(dataset.data_dir / "data/dataset.npz")
    before = np.asarray(dataset.valid_indices, dtype=np.int64)
    receipt = {"dataset_sha256": source_sha, "windows_before": len(before)}
    if oct_overlay is not None:
        dataset.oct_navigation, dataset.oct_navigation_valid = apply_oct_overlay(
            dataset.oct_navigation, dataset.oct_navigation_valid,
            oct_overlay, source_sha,
        )
        dataset.valid_indices = dataset._build_valid_indices()
        receipt["oct_overlay_sha256"] = file_sha256(oct_overlay)
    if c_overlay is not None:
        dataset.actions = apply_c_overlay(dataset.actions, c_overlay, source_sha)
        receipt["c_overlay_sha256"] = file_sha256(c_overlay)
    after = np.asarray(dataset.valid_indices, dtype=np.int64)
    receipt.update(windows_after=len(after),
                   excluded_windows=len(np.setdiff1d(before, after)),
                   added_windows=len(np.setdiff1d(after, before)))
    return receipt
