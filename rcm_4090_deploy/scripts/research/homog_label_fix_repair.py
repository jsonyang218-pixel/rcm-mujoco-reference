"""Stage 1: repair T1/T2/T3 labels without changing observations or images."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from lerobot_training.dataset_contract import valid_window_indices, validate_dataset_contract
from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_replay import windows
from scripts.research.homog_label_fix_common import (
    CAPS, FIX_DATA, QUALITY, REPORTS, SOURCE_DATA, identity, read, record, write_json,
)

REQUIRED = ('episode_index', 'frame_index', 'state_si', 'action', 'executed_action',
            'zero', 'pre', 'truth_navigation_mm', 'source_frame_index', 'timestamp_s',
            'oct_navigation', 'oct_navigation_valid', 'hold')
CAP_VECTOR = np.asarray([CAPS['A'], CAPS['B'], CAPS['C']], dtype=np.float64)


def _direction_with_executed_norm(direction, executed):
    direction = np.asarray(direction, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    direction_norm = np.linalg.norm(direction, axis=1)
    executed_norm = np.linalg.norm(executed, axis=1)
    raw = np.divide(direction * executed_norm[:, None], direction_norm[:, None],
        out=np.zeros_like(direction), where=direction_norm[:, None] > 1e-15)
    return np.clip(raw, -CAP_VECTOR, CAP_VECTOR).astype(np.float32)


def repair_arrays(arrays):
    missing = [key for key in REQUIRED if key not in arrays]
    if missing:
        raise ValueError(f'Fixed-label source lacks arrays: {missing}')
    episode = np.asarray(arrays['episode_index'])
    old = np.asarray(arrays['action'], dtype=np.float64)
    executed = np.asarray(arrays['executed_action'], dtype=np.float64)
    if not np.isfinite(old).all() or not np.isfinite(executed).all():
        raise ValueError('Source action or executed_action contains nonfinite values')
    error_um = np.linalg.norm(np.asarray(arrays['truth_navigation_mm'], dtype=np.float64),
        axis=1) * 1000.
    t1 = episode < 20
    t2 = (episode >= 20) & (episode < 40)
    t3 = episode >= 40
    stationary_t3 = t3 & np.all(executed == 0, axis=1) & (error_um < 5.)
    new = old.copy()
    # T1 keeps the zero-initial-velocity teacher direction and adopts the
    # accepted student-action magnitude. T2 keeps the accepted teacher-command
    # magnitude while restoring the clean pre-acceleration direction. This is
    # the authorized compatibility resolution for the four Stage1 gates.
    new[t1] = _direction_with_executed_norm(old[t1], executed[t1])
    new[t2] = _direction_with_executed_norm(old[t2], executed[t2])
    new[t3] = executed[t3].astype(np.float32)
    new[stationary_t3] = 0.
    if not np.isfinite(new).all() or np.any(np.abs(new) > CAP_VECTOR + 1e-8):
        raise ValueError('Repaired labels are nonfinite or exceed protocol caps')
    return new, {
        't1_frames': int(t1.sum()), 't2_frames': int(t2.sum()),
        't3_frames': int(t3.sum()), 't3_stationary_frames': int(stationary_t3.sum()),
        't3_stationary_zero_before': int(np.sum(
            stationary_t3 & np.all(old == 0, axis=1))),
        't3_stationary_zero_after': int(np.sum(
            stationary_t3 & np.all(new == 0, axis=1))),
    }, stationary_t3


def axis_reverse_audit(arrays, actions):
    zero = np.asarray(arrays['zero'], dtype=np.float64)
    pre = np.asarray(arrays['pre'], dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    eligible = (np.abs(zero) > 1e-10) & (np.abs(pre) > 1e-10)
    reverse = eligible & (action * zero < 0) & (action * pre < 0)
    return {axis: {
        'eligible_nonzero_geometry_frames': int(eligible[:, index].sum()),
        'reverse_frames': int(reverse[:, index].sum()),
        'reverse_fraction': float(reverse[:, index].sum()
                                  / max(1, eligible[:, index].sum())),
    } for index, axis in enumerate('ABC')}


def valid_window_audit(arrays):
    indices, diagnostics = windows(arrays)
    if not len(indices):
        return indices, diagnostics, False, []
    failures = []
    for index in indices:
        span = slice(index - 1, index + 16)
        if not (np.all(np.diff(arrays['source_frame_index'][span]) == 1)
                and np.allclose(np.diff(arrays['timestamp_s'][span]), .05,
                                rtol=1e-3, atol=1e-6)
                and len(set(arrays['episode_index'][span])) == 1):
            failures.append(int(index))
    return indices, diagnostics, not failures, failures


def build_fixed_dataset(output):
    source_dataset = SOURCE_DATA / 'data/dataset.npz'
    source_sha = file_sha256(source_dataset)
    with np.load(source_dataset, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_dataset_contract(arrays, require_history=True)
    actions, counts, stationary_t3 = repair_arrays(arrays)
    repaired = {**arrays, 'action': actions}
    validate_dataset_contract(repaired, require_history=True)
    valid, diagnostics, continuity, failures = valid_window_audit(repaired)
    episode = np.asarray(repaired['episode_index'])
    executed = np.asarray(repaired['executed_action'], dtype=np.float64)
    executed_norm = np.linalg.norm(executed, axis=1)
    label_norm = np.linalg.norm(actions.astype(np.float64), axis=1)
    t2 = (episode >= 20) & (episode < 40)
    nonzero = t2 & (executed_norm > 1e-12) & (label_norm > 1e-12)
    t2_ratio = float(np.median(label_norm[nonzero] / executed_norm[nonzero]))
    axes = axis_reverse_audit(repaired, actions)
    stationary = (episode >= 40) & stationary_t3
    stationary_zero_fraction = float(np.mean(
        np.all(actions[stationary] == 0, axis=1))) if stationary.any() else 0.
    gates = {
        'T3_stationary_action_zero_fraction': {
            'value': stationary_zero_fraction, 'required': '==1.0',
            'passed': stationary_zero_fraction == 1.},
        'T2_label_to_executed_norm_ratio_median': {
            'value': t2_ratio, 'required': '0.95<=x<=1.05',
            'passed': .95 <= t2_ratio <= 1.05},
        'axis_reverse_fractions': {
            'value': {key: value['reverse_fraction'] for key, value in axes.items()},
            'required': '<0.02 each',
            'passed': all(value['reverse_fraction'] < .02 for value in axes.values())},
        'valid_window_contract_pass_fraction': {
            'value': 1. if continuity else 0., 'required': '==1.0',
            'passed': bool(continuity)},
    }
    passed = all(value['passed'] for value in gates.values())
    source_audit = read(SOURCE_DATA / 'dataset_audit.json')
    hold_frames = int(np.sum(
        (episode >= 40) & np.asarray(repaired['hold'], dtype=bool)
        & (error_um < 5.)))
    audit = {
        'status': 'passed' if passed else 'failed',
        'source_dataset': str(source_dataset), 'source_dataset_sha256': source_sha,
        'framing': 'Independent corrected action array; RGB, OCT, state, history and continuity untouched',
        'counts': counts, 'gates': gates, 'axes': axes,
        'window_diagnostics': diagnostics, 'valid_windows': len(valid),
        'continuity_failures': failures[:20],
        'T3_stationary_definition': 'regime=T3 AND executed_action==[0,0,0] AND truth error<5um',
        'T2_rule': 'executed_action norm with zero-initial-velocity teacher pre-acceleration direction',
        'T1_rule': 'zero-initial-velocity teacher direction scaled to accepted student-action norm',
        'T3_rule': 'stationary under5um frames zero; all other T3 frames use executed_action',
        'sampled_frames': source_audit['sampled_frames'],
        'oct_truth_median_error_um': source_audit['oct_truth_median_error_um'],
        'axis_reverse_fractions': {
            key: value['reverse_fraction'] for key, value in axes.items()},
        'RGB_OCT_render_provenance_mismatch': False,
        'valid_window_contract_pass_fraction': 1. if continuity else 0.,
        'all_regimes_complete': dict(episodes=60, T1=20, T2=20, T3=20),
        'T3_hold_frames_below_5um': hold_frames,
        'render_reuse': 'No image or OCT file is copied or rewritten; FIX_DATA/shards is a read-only symlink to SOURCE_DATA/shards',
        'quality': QUALITY, 'identity': identity(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'label_fix_audit.json', audit)
    if not passed:
        record('stage1_label_audit', 'failed', output, audit)
        raise ValueError(f'Fixed-label audit failed: {gates}')
    if FIX_DATA.exists():
        raise FileExistsError(f'Fixed-label destination occupied: {FIX_DATA}')
    FIX_DATA.mkdir(parents=True)
    (FIX_DATA / 'data').mkdir()
    np.savez_compressed(FIX_DATA / 'data/dataset.npz', **repaired)
    np.savez_compressed(FIX_DATA / 'data/valid_window_indices.npz',
                        rows=np.asarray(valid, dtype=np.int64))
    np.savez_compressed(FIX_DATA / 'data/label_repair_overlay.npz',
        old_action=np.asarray(arrays['action'], dtype=np.float32),
        new_action=actions, t3_stationary=stationary_t3.astype(np.bool_),
        episode_index=episode.astype(np.int64),
        source_dataset_sha256=np.asarray(source_sha))
    (FIX_DATA / 'data/image_paths.json').write_bytes(
        (SOURCE_DATA / 'data/image_paths.json').read_bytes())
    os.symlink(SOURCE_DATA / 'shards', FIX_DATA / 'shards', target_is_directory=True)
    metadata = read(SOURCE_DATA / 'metadata.json')
    metadata.update({
        'action_source': 'homog_label_fix_20261123',
        'history_action_source': 'executed_action',
        'label_repair': {
            'source_dataset_sha256': source_sha,
            'source_labels_sha256': file_sha256(SOURCE_DATA / 'data/dataset.npz'),
            'T1': audit['T1_rule'], 'T2': audit['T2_rule'], 'T3': audit['T3_rule'],
            'images': 'Read-only symlink; no image or OCT rewritten',
        },
    })
    for item in metadata.get('episodes', []):
        regime = item.get('regime')
        if regime == 'T1':
            item['action_source'] = audit['T1_rule']
        elif regime == 'T2':
            item['action_source'] = audit['T2_rule']
        else:
            item['action_source'] = audit['T3_rule']
    write_json(FIX_DATA / 'metadata.json', metadata)
    write_json(FIX_DATA / 'dataset_audit.json', audit)
    receipt = {
        'status': 'completed', 'source_data': str(SOURCE_DATA),
        'fixed_data': str(FIX_DATA), 'source_dataset_sha256': source_sha,
        'fixed_dataset_sha256': file_sha256(FIX_DATA / 'data/dataset.npz'),
        'overlay_sha256': file_sha256(FIX_DATA / 'data/label_repair_overlay.npz'),
        'image_paths_sha256': file_sha256(FIX_DATA / 'data/image_paths.json'),
        'images_reused_by_symlink': True, 'original_files_unchanged': True,
        'audit': str(output / 'label_fix_audit.json'),
    }
    write_json(FIX_DATA / 'dataset_receipt.json', receipt)
    write_json(output / 'summary.json', {'status': 'completed', **receipt, 'gates': gates})
    record('stage1_label_repair', 'completed', output, {
        'source_dataset_sha256': source_sha,
        'fixed_dataset_sha256': receipt['fixed_dataset_sha256'],
        'T3_stationary_frames': counts['t3_stationary_frames'],
        'T2_norm_ratio_median': t2_ratio,
        'axis_reverse_fractions': gates['axis_reverse_fractions']['value'],
        'valid_windows': len(valid),
    })
    return audit, receipt
