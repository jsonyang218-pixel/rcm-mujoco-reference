"""Build label-repair datasets without changing observations or images."""
from __future__ import annotations

import argparse
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
MODES = ('compat_v3', 'executed_only', 'dir_consistent', 't3_only')
OUTER_WEIGHT = {
    'executed_only': {'T3_stationary': .25, 'default': 1.},
    'dir_consistent': {'T3_stationary': .25, 'default': 1.},
    't3_only': {'T1_T2': 0., 'T3_stationary': .30, 'default': 1.},
}


def _direction_with_executed_norm(direction, executed):
    direction = np.asarray(direction, dtype=np.float64)
    executed = np.asarray(executed, dtype=np.float64)
    direction_norm = np.linalg.norm(direction, axis=1)
    executed_norm = np.linalg.norm(executed, axis=1)
    raw = np.divide(direction * executed_norm[:, None], direction_norm[:, None],
        out=np.zeros_like(direction), where=direction_norm[:, None] > 1e-15)
    return np.clip(raw, -CAP_VECTOR, CAP_VECTOR).astype(np.float32)


def _regime_masks(arrays):
    episode = np.asarray(arrays['episode_index'])
    t1 = episode < 20
    t2 = (episode >= 20) & (episode < 40)
    t3 = episode >= 40
    return t1, t2, t3


def repair_arrays(arrays, mode='compat_v3'):
    missing = [key for key in REQUIRED if key not in arrays]
    if missing:
        raise ValueError(f'Fixed-label source lacks arrays: {missing}')
    if mode not in MODES:
        raise ValueError(f'Unsupported repair mode: {mode}')
    old = np.asarray(arrays['action'], dtype=np.float64)
    executed = np.asarray(arrays['executed_action'], dtype=np.float64)
    if not np.isfinite(old).all() or not np.isfinite(executed).all():
        raise ValueError('Source action or executed_action contains nonfinite values')
    error_um = np.linalg.norm(np.asarray(arrays['truth_navigation_mm'], dtype=np.float64),
        axis=1) * 1000.
    t1, t2, t3 = _regime_masks(arrays)
    stationary_t3 = t3 & np.all(executed == 0, axis=1) & (error_um < 5.)
    new = old.copy()
    t1_fallback = np.zeros(len(old), dtype=np.bool_)
    t2_fallback = np.zeros(len(old), dtype=np.bool_)
    if mode == 'compat_v3':
        # Preserve the exact v3 behavior used by the prior experiment.
        new[t1] = _direction_with_executed_norm(old[t1], executed[t1])
        new[t2] = _direction_with_executed_norm(old[t2], executed[t2])
    else:
        # Every v4 arm starts T1 from the command that was actually executed.
        new[t1] = executed[t1]
        t1_fallback[t1] = True
        if mode == 'dir_consistent':
            directed = _direction_with_executed_norm(old[t2], executed[t2])
            dot = np.sum(directed * executed[t2], axis=1)
            use_direction = dot > 0
            rows = np.flatnonzero(t2)
            new[rows[use_direction]] = directed[use_direction]
            new[rows[~use_direction]] = executed[rows[~use_direction]]
            t2_fallback[rows[~use_direction]] = True
        else:
            new[t2] = executed[t2]
            t2_fallback[t2] = True
    new[t3] = executed[t3].astype(np.float32)
    new[stationary_t3] = 0.
    if not np.isfinite(new).all():
        raise ValueError('Repaired labels are nonfinite')
    # The v4 executed-only and t3-only arms intentionally preserve accepted
    # commands exactly, including rare mechanical C values above the legacy
    # teacher-command cap. Keep the old cap for compatibility mode only.
    if mode == 'compat_v3' and np.any(np.abs(new) > CAP_VECTOR + 1e-8):
        raise ValueError('Compatibility labels exceed protocol caps')
    return new, {
        't1_frames': int(t1.sum()), 't2_frames': int(t2.sum()),
        't3_frames': int(t3.sum()), 't3_stationary_frames': int(stationary_t3.sum()),
        't1_fallback_frames': int(t1_fallback.sum()),
        't2_fallback_frames': int(t2_fallback.sum()),
        't3_stationary_zero_before': int(np.sum(
            stationary_t3 & np.all(old == 0, axis=1))),
        't3_stationary_zero_after': int(np.sum(
            stationary_t3 & np.all(new == 0, axis=1))),
        'mode': mode,
    }, stationary_t3


def repair_sample_weights(
    arrays,
    mode,
    stationary_t3,
    t3_stationary_weight=None,
):
    """Return per-frame sample weights for an independent correction arm."""
    sample_count = len(np.asarray(
        arrays['state'] if 'state' in arrays else arrays['state_si']
    ))
    if mode == 'compat_v3':
        if t3_stationary_weight is not None:
            raise ValueError(
                'compatibility mode does not accept a T3 stationary weight'
            )
        return np.ones(sample_count, dtype=np.float32)
    if t3_stationary_weight is None:
        t3_stationary_weight = OUTER_WEIGHT[mode]['T3_stationary']
    t3_stationary_weight = float(t3_stationary_weight)
    if (
        not np.isfinite(t3_stationary_weight)
        or t3_stationary_weight < 0
    ):
        raise ValueError(
            'T3 stationary sample weight must be finite and nonnegative'
        )
    t1, t2, _ = _regime_masks(arrays)
    policy = OUTER_WEIGHT[mode]
    weights = np.full(sample_count, policy.get('default', 1.),
                      dtype=np.float32)
    if mode == 't3_only':
        weights[t1 | t2] = policy['T1_T2']
    weights[np.asarray(stationary_t3, dtype=np.bool_)] = t3_stationary_weight
    return weights


def window_weights(indices, sample_weights, action_steps=16):
    indices = np.asarray(indices, dtype=np.int64)
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if len(indices) and indices[-1] + action_steps > len(sample_weights):
        raise ValueError('valid window extends beyond correction sample weights')
    return np.asarray([
        np.mean(sample_weights[index:index + action_steps])
        for index in indices
    ], dtype=np.float32)


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


def _arrays_equal(left, right):
    return (np.asarray(left).shape == np.asarray(right).shape
            and np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


def build_fixed_dataset(
    output,
    fixed_data=None,
    mode='compat_v3',
    t3_stationary_weight=None,
):
    fixed_data = FIX_DATA if fixed_data is None else Path(fixed_data)
    source_dataset = SOURCE_DATA / 'data/dataset.npz'
    source_sha = file_sha256(source_dataset)
    with np.load(source_dataset, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_dataset_contract(arrays, require_history=True)
    actions, counts, stationary_t3 = repair_arrays(arrays, mode)
    configured_t3_weight = (
        float(t3_stationary_weight)
        if t3_stationary_weight is not None
        else (
            float(OUTER_WEIGHT[mode]['T3_stationary'])
            if mode != 'compat_v3'
            else 1.
        )
    )
    sample_weights = repair_sample_weights(
        arrays,
        mode,
        stationary_t3,
        t3_stationary_weight=(
            t3_stationary_weight if mode != 'compat_v3' else None
        ),
    )
    repaired = {**arrays, 'action': actions, 'correction_sample_weight': sample_weights}
    validate_dataset_contract(repaired, require_history=True)
    valid, diagnostics, continuity, failures = valid_window_audit(repaired)
    valid_weights = window_weights(valid, sample_weights)
    episode = np.asarray(repaired['episode_index'])
    error_um = np.linalg.norm(
        np.asarray(repaired['truth_navigation_mm'], dtype=np.float64), axis=1) * 1000.
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
    t1, t2, t3 = _regime_masks(repaired)
    t1_t2 = t1 | t2
    fallback_fraction = float(
        (counts['t1_fallback_frames'] + counts['t2_fallback_frames'])
        / max(1, counts['t1_frames'] + counts['t2_frames'])
    )
    nonzero_weight_fraction = float(np.mean(sample_weights[stationary] > 0)
                                    if stationary.any() else 0.)
    empirical_stationary_fraction = float(
        stationary_t3.sum() / max(1, int(t3.sum()))
    )
    window_stationary = np.asarray([
        np.count_nonzero(stationary_t3[index:index + 16])
        for index in valid
    ], dtype=np.int64)
    weighted_window_mass = float(np.sum(
        valid_weights.astype(np.float64)
        * window_stationary.astype(np.float64)
    ))
    weighted_action_frame_mass = float(np.sum(
        valid_weights.astype(np.float64) * 16.
    ))
    theoretical_weighted_stationary_fraction = (
        weighted_window_mass / weighted_action_frame_mass
        if weighted_action_frame_mass > 0. else 0.
    )
    directed_t2 = _direction_with_executed_norm(
        np.asarray(arrays['action'][t2], dtype=np.float64),
        np.asarray(arrays['executed_action'][t2], dtype=np.float64),
    )
    t2_dot = np.sum(directed_t2 * np.asarray(arrays['executed_action'][t2],
                                             dtype=np.float64), axis=1)
    if mode == 'compat_v3':
        mode_rule_ok = True
    elif mode == 'executed_only':
        mode_rule_ok = bool(
            np.array_equal(actions[t1_t2], executed[t1_t2])
            and np.array_equal(actions[t3], executed[t3])
        )
    elif mode == 'dir_consistent':
        expected_t2 = np.where(
            t2_dot[:, None] > 0,
            directed_t2,
            np.asarray(arrays['executed_action'][t2], dtype=np.float64),
        )
        mode_rule_ok = bool(
            np.array_equal(actions[t1], executed[t1])
            and np.allclose(actions[t2], expected_t2, rtol=0, atol=1e-8)
            and np.array_equal(actions[t3], executed[t3])
        )
    else:
        mode_rule_ok = bool(
            np.array_equal(actions[t1_t2], executed[t1_t2])
            and np.array_equal(actions[t3], executed[t3])
        )
    if mode == 't3_only':
        expected_weights = np.ones(len(actions), dtype=np.float32)
        expected_weights[t1_t2] = 0.
        expected_weights[stationary] = configured_t3_weight
    elif mode in ('executed_only', 'dir_consistent'):
        expected_weights = np.ones(len(actions), dtype=np.float32)
        expected_weights[stationary] = configured_t3_weight
    else:
        expected_weights = np.ones(len(actions), dtype=np.float32)
    preserved_keys = sorted(set(arrays) - {'action'})
    preserved = all(_arrays_equal(arrays[key], repaired[key])
                    for key in preserved_keys)
    source_windows_path = SOURCE_DATA / 'data/valid_window_indices.npz'
    source_windows_equal = True
    if source_windows_path.is_file():
        with np.load(source_windows_path, allow_pickle=False) as source_windows:
            source_rows = source_windows['rows']
        source_windows_equal = bool(np.array_equal(source_rows, valid))

    gates = {
        'T3_stationary_action_zero_fraction': {
            'value': stationary_zero_fraction, 'required': '==1.0',
            'passed': stationary_zero_fraction == 1.},
        'executed_action_unchanged': {
            'value': True, 'required': '==True',
            'passed': _arrays_equal(arrays['executed_action'],
                                    repaired['executed_action'])},
        'observation_OCT_state_history_unchanged': {
            'value': preserved, 'required': '==True', 'passed': preserved},
        'mode_label_rule': {
            'value': mode_rule_ok, 'required': '==True', 'passed': mode_rule_ok},
        'correction_sample_weight_contract': {
            'value': bool(np.array_equal(sample_weights, expected_weights)),
            'required': 'exact frame policy',
            'passed': bool(np.array_equal(sample_weights, expected_weights))},
        'valid_window_indices_unchanged': {
            'value': source_windows_equal, 'required': '==True',
            'passed': source_windows_equal},
        'valid_window_contract_pass_fraction': {
            'value': 1. if continuity else 0., 'required': '==1.0',
            'passed': bool(continuity)},
    }
    if mode == 'compat_v3':
        gates.update({
            'T2_label_to_executed_norm_ratio_median': {
                'value': t2_ratio, 'required': '0.95<=x<=1.05',
                'passed': .95 <= t2_ratio <= 1.05},
            'axis_reverse_fractions': {
                'value': {key: value['reverse_fraction']
                          for key, value in axes.items()},
                'required': '<0.02 each',
                'passed': all(value['reverse_fraction'] < .02
                              for value in axes.values())},
        })
    passed = all(value['passed'] for value in gates.values())
    source_audit = read(SOURCE_DATA / 'dataset_audit.json')
    hold_frames = int(np.sum(
        (episode >= 40) & np.asarray(repaired['hold'], dtype=bool)
        & (error_um < 5.)))
    audit = {
        'status': 'passed' if passed else 'failed',
        'source_dataset': str(source_dataset), 'source_dataset_sha256': source_sha,
        'mode': mode,
        't3_stationary_weight': configured_t3_weight,
        't1_t2_fallback_fraction': fallback_fraction,
        # kept_fraction is retained for v4 readers. It only proves a nonzero
        # weight and must not be presented as evidence that downsampling worked.
        't3_stationary_kept_fraction': nonzero_weight_fraction,
        'nonzero_weight_fraction': nonzero_weight_fraction,
        'empirical_t3_stationary_action_fraction': (
            empirical_stationary_fraction
        ),
        't3_stationary_action_frames': int(stationary_t3.sum()),
        't3_action_frames': int(t3.sum()),
        'theoretical_weighted_t3_stationary_action_fraction': (
            theoretical_weighted_stationary_fraction
        ),
        'correction_sample_weight': {
            'kind': 'per_frame_then_mean_over_16_action_steps',
            't3_stationary_weight': configured_t3_weight,
            'values': (sorted(float(value) for value in np.unique(sample_weights))),
            'window_min': float(valid_weights.min()) if len(valid_weights) else None,
            'window_max': float(valid_weights.max()) if len(valid_weights) else None,
            'window_mean': float(valid_weights.mean()) if len(valid_weights) else None,
            'nonzero_windows': int(np.sum(valid_weights > 0)),
            'zero_windows': int(np.sum(valid_weights == 0)),
            'stationary_windows': int(np.sum(window_stationary > 0)),
            'stationary_action_frames_in_valid_windows': int(
                window_stationary.sum()
            ),
        },
        'framing': 'Independent corrected action array; RGB, OCT, state, history and continuity untouched',
        'counts': counts, 'gates': gates, 'axes': axes,
        'window_diagnostics': diagnostics, 'valid_windows': len(valid),
        'continuity_failures': failures[:20],
        'T3_stationary_definition': 'regime=T3 AND executed_action==[0,0,0] AND truth error<5um',
        'T2_rule': (
            'executed_action norm with zero-initial-velocity teacher pre-acceleration '
            'direction only when dot(direction,executed)>0; otherwise executed_action'
            if mode == 'dir_consistent' else
            'executed_action norm with zero-initial-velocity teacher pre-acceleration direction'
            if mode == 'compat_v3' else 'exact executed_action'
        ),
        'T1_rule': (
            'zero-initial-velocity teacher direction scaled to accepted student-action norm'
            if mode == 'compat_v3' else 'exact executed_action'
        ),
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
    if fixed_data.exists():
        raise FileExistsError(f'Fixed-label destination occupied: {fixed_data}')
    fixed_data.mkdir(parents=True)
    (fixed_data / 'data').mkdir()
    np.savez_compressed(fixed_data / 'data/dataset.npz', **repaired)
    np.savez_compressed(fixed_data / 'data/valid_window_indices.npz',
                        rows=np.asarray(valid, dtype=np.int64))
    fallback_mask = np.zeros(len(actions), dtype=np.bool_)
    if mode != 'compat_v3':
        fallback_mask[t1] = True
        if mode in ('executed_only', 't3_only'):
            fallback_mask[t2] = True
        elif mode == 'dir_consistent':
            fallback_mask[np.flatnonzero(t2)[t2_dot <= 0]] = True
    np.savez_compressed(fixed_data / 'data/label_repair_overlay.npz',
        old_action=np.asarray(arrays['action'], dtype=np.float32),
        new_action=actions, t3_stationary=stationary_t3.astype(np.bool_),
        episode_index=episode.astype(np.int64),
        correction_sample_weight=sample_weights,
        correction_window_indices=valid.astype(np.int64),
        correction_window_weight=valid_weights,
        t3_stationary_weight=np.asarray(configured_t3_weight),
        t1_t2_fallback=fallback_mask,
        mode=np.asarray(mode),
        source_dataset_sha256=np.asarray(source_sha))
    (fixed_data / 'data/image_paths.json').write_bytes(
        (SOURCE_DATA / 'data/image_paths.json').read_bytes())
    os.symlink(SOURCE_DATA / 'shards', fixed_data / 'shards', target_is_directory=True)
    metadata = read(SOURCE_DATA / 'metadata.json')
    metadata.update({
        'action_source': 'homog_label_fix_20261124_v5',
        'history_action_source': 'executed_action',
        'mode': mode,
        't3_stationary_weight': configured_t3_weight,
        't1_t2_fallback_fraction': fallback_fraction,
        't3_stationary_kept_fraction': nonzero_weight_fraction,
        'nonzero_weight_fraction': nonzero_weight_fraction,
        'empirical_t3_stationary_action_fraction': (
            empirical_stationary_fraction
        ),
        'theoretical_weighted_t3_stationary_action_fraction': (
            theoretical_weighted_stationary_fraction
        ),
        'correction_sample_weight': audit['correction_sample_weight'],
        'label_repair': {
            'source_dataset_sha256': source_sha,
            'source_labels_sha256': file_sha256(SOURCE_DATA / 'data/dataset.npz'),
            'T1': audit['T1_rule'], 'T2': audit['T2_rule'], 'T3': audit['T3_rule'],
            'mode': mode,
            't3_stationary_weight': configured_t3_weight,
            't1_t2_fallback_fraction': fallback_fraction,
            't3_stationary_kept_fraction': nonzero_weight_fraction,
            'nonzero_weight_fraction': nonzero_weight_fraction,
            'empirical_t3_stationary_action_fraction': (
                empirical_stationary_fraction
            ),
            'theoretical_weighted_t3_stationary_action_fraction': (
                theoretical_weighted_stationary_fraction
            ),
            'correction_sample_weight': audit['correction_sample_weight'],
            'images': 'Read-only symlink; no image or OCT rewritten',
        },
    })
    for item in metadata.get('episodes', []):
        regime = item.get('regime')
        if mode == 't3_only' and regime in ('T1', 'T2'):
            item['action_source'] = 'executed'
        elif regime == 'T1':
            item['action_source'] = audit['T1_rule']
        elif regime == 'T2':
            item['action_source'] = audit['T2_rule']
        else:
            item['action_source'] = 'teacher'
    write_json(fixed_data / 'metadata.json', metadata)
    write_json(fixed_data / 'dataset_audit.json', audit)
    receipt = {
        'status': 'completed', 'mode': mode, 'source_data': str(SOURCE_DATA),
        't3_stationary_weight': configured_t3_weight,
        'fixed_data': str(fixed_data), 'source_dataset_sha256': source_sha,
        'fixed_dataset_sha256': file_sha256(fixed_data / 'data/dataset.npz'),
        'overlay_sha256': file_sha256(fixed_data / 'data/label_repair_overlay.npz'),
        'image_paths_sha256': file_sha256(fixed_data / 'data/image_paths.json'),
        'images_reused_by_symlink': True, 'original_files_unchanged': True,
        'valid_window_index_sha256': file_sha256(
            fixed_data / 'data/valid_window_indices.npz'),
        'correction_sample_weight': audit['correction_sample_weight'],
        'nonzero_weight_fraction': nonzero_weight_fraction,
        'empirical_t3_stationary_action_fraction': (
            empirical_stationary_fraction
        ),
        'theoretical_weighted_t3_stationary_action_fraction': (
            theoretical_weighted_stationary_fraction
        ),
        'audit': str(output / 'label_fix_audit.json'),
    }
    write_json(fixed_data / 'dataset_receipt.json', receipt)
    write_json(output / 'summary.json', {'status': 'completed', **receipt, 'gates': gates})
    record('stage1_label_repair', 'completed', output, {
        'source_dataset_sha256': source_sha,
        'fixed_dataset_sha256': receipt['fixed_dataset_sha256'],
        'mode': mode,
        'T3_stationary_frames': counts['t3_stationary_frames'],
        'T2_norm_ratio_median': t2_ratio,
        't1_t2_fallback_fraction': fallback_fraction,
        't3_stationary_kept_fraction': nonzero_weight_fraction,
        'nonzero_weight_fraction': nonzero_weight_fraction,
        'empirical_t3_stationary_action_fraction': (
            empirical_stationary_fraction
        ),
        'theoretical_weighted_t3_stationary_action_fraction': (
            theoretical_weighted_stationary_fraction
        ),
        'axis_reverse_fractions': audit['axis_reverse_fractions'],
        'valid_windows': len(valid),
    })
    return audit, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='compat_v3')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fixed-data', type=Path, required=True)
    parser.add_argument('--t3-stationary-weight', type=float)
    args = parser.parse_args(argv)
    audit, receipt = build_fixed_dataset(
        args.output_dir.resolve(), args.fixed_data.resolve(), args.mode,
        t3_stationary_weight=args.t3_stationary_weight,
    )
    print(json.dumps({'audit': audit, 'receipt': receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
