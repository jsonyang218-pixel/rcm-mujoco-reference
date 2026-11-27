"""GPU clean-OCT preparation, replay training and conditional evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from lerobot_training.navigation_overlay import (
    apply_c_overlay, apply_oct_overlay, file_sha256,
)
from scripts.research.clean_oct_workers import resolve_target
from scripts.research.gpu_episode_pool import record, run_pool, signature, write_json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/plan2_plus_original_train_224_20261119'
VAL = ROOT / 'data/lerobot_rcm_val_acthist_20261117_224'
INITIAL = ROOT / 'outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth'
BASE_C = ROOT / 'data/overnight_c_fix_symlog_081234_overlay/label_override.npz'
TARGETS = ROOT / 'configs/v12_reachable_development_targets_0p35um.json'
BASELINE = ROOT / 'runs/20261121/control_p2_parallel_v1/parallel/s3_v0/summary.json'
FAST5 = (0, 2, 6, 8, 10)
FACTORY = 'scripts.research.clean_oct_workers:'


def identity():
    from scripts.tools.experiment_workflow import scene_and_code
    return {**scene_and_code(), **{str(path.relative_to(ROOT)): file_sha256(path)
             for path in (Path(__file__), ROOT / 'scripts/research/clean_oct_workers.py',
                          ROOT / 'scripts/research/gpu_episode_pool.py',
                          ROOT / 'scripts/research/control_g15_diagnostics.py', TARGETS)}}


def load_arrays():
    with np.load(DATA / 'data/dataset.npz', allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def render_groups(*, smoke=False):
    arrays = load_arrays()
    valid = arrays['oct_navigation_valid'].astype(bool)
    mask = valid & np.isfinite(arrays['oct_navigation']).all(1)
    mask &= np.linalg.norm(arrays['oct_navigation'].astype(float), axis=1) < .2
    episodes = np.unique(arrays['episode_index'][mask])
    if smoke:
        wanted = [index for index in (0, 760, 839) if index in episodes]
        if len(wanted) != 3:
            raise ValueError('explicit three-source smoke targets unavailable')
        episodes = wanted
    tasks = []
    for episode in episodes:
        rows = np.flatnonzero(mask & (arrays['episode_index'] == episode))
        if smoke:
            rows = rows[np.unique(np.linspace(0, len(rows) - 1, min(8, len(rows)), dtype=int))]
        tasks.append({'id': f'E{int(episode):04d}', 'episode_index': int(episode),
                      'rows': rows.tolist(), 'target': resolve_target(DATA, int(episode))})
    return {'render': {'factory': FACTORY + 'OverlayWorker',
                        'spec': {'data_dir': str(DATA), 'dataset_sha256': file_sha256(DATA / 'data/dataset.npz'),
                                 'base_c_overlay': str(BASE_C), 'base_c_sha256': file_sha256(BASE_C),
                                 'identity': identity()}, 'tasks': tasks}}


def atomic_npz(path, **values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def assemble_overlay(results, destination, expected_rows, *, smoke=False):
    destination.mkdir(parents=True, exist_ok=False)
    blocks, audits, c_audit, proposals = [], {}, [], []
    for result in results.values():
        with np.load(result['oct_path'], allow_pickle=False) as archive:
            blocks.append({key: archive[key].copy() for key in archive.files})
        audit = json.loads(Path(result['audit_path']).read_text())
        audits.update(zip(audit['rows'], audit['error_um']))
        c_audit.extend(audit['c_audit'])
        proposals.extend(audit['c_proposals'])
    rows = np.concatenate([item['rows'] for item in blocks])
    order = np.argsort(rows)
    rows = rows[order]
    if not np.array_equal(rows, expected_rows):
        raise ValueError('overlay coverage is incomplete, duplicated or unexpected')
    source_sha = file_sha256(DATA / 'data/dataset.npz')
    if any(str(block['dataset_sha256']) != source_sha for block in blocks):
        raise ValueError('source dataset changed during rendering; rejecting mixed overlay')
    values = {key: np.concatenate([item[key] for item in blocks])[order]
              for key in ('old_oct_mm', 'new_oct_mm', 'old_valid', 'new_valid')}
    oct_path = destination / 'oct_override.npz'
    atomic_npz(oct_path, rows=rows, dataset_sha256=source_sha, **values)
    sample = np.random.default_rng(42).choice(rows, min(200, len(rows)), replace=False)
    errors = [audits[int(row)] for row in sample]
    sample_valid = all(value is not None and np.isfinite(value) for value in errors)
    median = float(np.median(errors)) if sample_valid else None
    audit_pass = sample_valid and median < .2 and (smoke or len(sample) == 200)
    # Directional paired evidence, not a statistical-significance assertion.
    old = np.asarray([item['old_overlay_dC_mm'] for item in c_audit])
    wanted = np.asarray([item['wanted_dC_mm'] for item in c_audit])
    c_confirmed = bool(len(old) and np.median(wanted) > np.median(old) + 1e-9
                       and np.count_nonzero(wanted > old + 1e-9) > len(old) / 2)
    arrays = load_arrays()
    with np.load(BASE_C, allow_pickle=False) as base:
        replacement = dict(zip(base['rows'].tolist(), base['new_dC_mm'].tolist()))
    if c_confirmed:
        replacement.update({int(row): float(value) for row, value in proposals})
    c_rows = np.asarray(sorted(replacement), dtype=np.int64)
    c_path = destination / 'c_label_override.npz'
    atomic_npz(c_path, rows=c_rows, old_dC_mm=arrays['action'][c_rows, 2],
               new_dC_mm=np.asarray([replacement[int(row)] for row in c_rows], dtype=np.float32),
               dataset_sha256=source_sha)
    actions = apply_c_overlay(arrays['action'], c_path, source_sha)
    np.testing.assert_array_equal(actions[:, :2], arrays['action'][:, :2])
    oct_values, valid = apply_oct_overlay(arrays['oct_navigation'], arrays['oct_navigation_valid'], oct_path, source_sha)
    from lerobot_training.dataset_contract import valid_window_indices
    indices, diagnostics = valid_window_indices(
        arrays['episode_index'], arrays['source_frame_index'], arrays['timestamp_s'],
        n_obs_steps=2, n_action_steps=16, oct_navigation=oct_values, oct_navigation_valid=valid,
        return_diagnostics=True)
    np.save(destination / 'valid_window_indices.npy', indices)
    report = {'status': 'completed' if audit_pass else 'gate_failed', 'smoke': smoke,
              'covered_frames': len(rows), 'valid_frames': int(values['new_valid'].sum()),
              'audit_rows': sample.tolist(), 'audit_error_um': errors, 'audit_median_error_um': median,
              'audit_passed': audit_pass, 'c_audit_rows': len(old),
              'c_old_median_mm': float(np.median(old)) if len(old) else None,
              'c_pre_accel_median_mm': float(np.median(wanted)) if len(wanted) else None,
              'c_repair_confirmed': c_confirmed, 'c_proposals_applied': len(proposals) if c_confirmed else 0,
              'c_condition': 'paired wanted median > old median +1e-9 mm and majority individually larger; directional audit only',
              'window_diagnostics': diagnostics, 'oct_overlay': str(oct_path), 'c_overlay': str(c_path),
              'dataset_sha256': source_sha, 'oct_sha256': file_sha256(oct_path), 'c_sha256': file_sha256(c_path)}
    write_json(destination / 'c_audit.json', {'rows': c_audit, 'proposals': proposals})
    write_json(destination / 'summary.json', report)
    return report


def render(args):
    started = time.monotonic()
    groups = render_groups(smoke=args.mode == 'smoke')
    expected = np.sort(np.concatenate([np.asarray(task['rows'], dtype=np.int64) for task in groups['render']['tasks']]))
    results = run_pool(groups, args.output_dir / 'shards', workers=min(args.workers, len(groups['render']['tasks'])),
                       max_workers=min(args.max_workers, len(groups['render']['tasks'])),
                       timeout_s=180, stage_timeout_s=args.stage_timeout_s, resume=args.resume)
    spec = groups['render']['spec']
    if (file_sha256(DATA / 'data/dataset.npz') != spec['dataset_sha256']
            or file_sha256(BASE_C) != spec['base_c_sha256']):
        raise ValueError('source dataset or base C overlay changed during rendering')
    destination = args.output_dir / 'overlay' if args.mode == 'smoke' else args.overlay_dir
    if destination is None:
        raise ValueError('full rendering requires a separate --overlay-dir under data/')
    if args.resume and destination.exists():
        summary_path = destination / 'summary.json'
        if summary_path.is_file():
            report = json.loads(summary_path.read_text())
            if (report['dataset_sha256'] != file_sha256(DATA / 'data/dataset.npz')
                    or report['oct_sha256'] != file_sha256(destination / 'oct_override.npz')
                    or report['c_sha256'] != file_sha256(destination / 'c_label_override.npz')):
                raise ValueError('completed overlay changed; refusing resume')
            with np.load(destination / 'oct_override.npz', allow_pickle=False) as archive:
                if not np.array_equal(archive['rows'], expected):
                    raise ValueError('completed overlay target coverage differs')
        else:
            destination.rename(destination.with_name(destination.name + f'_partial_{time.time_ns()}'))
            report = assemble_overlay(results, destination, expected, smoke=args.mode == 'smoke')
    else:
        report = assemble_overlay(results, destination, expected, smoke=args.mode == 'smoke')
    report['elapsed_s'] = time.monotonic() - started
    report['frames_per_second'] = len(expected) / report['elapsed_s']
    write_json(args.output_dir / 'summary.json', report)
    record(args.mode, report['status'], args.output_dir,
           f"dataset={DATA}; frames={len(expected)}; median_um={report['audit_median_error_um']}; no training")
    if not report['audit_passed']:
        raise ValueError('OCT audit gate failed; no training allowed')


def train(args):
    source = json.loads((args.overlay_dir / 'summary.json').read_text())
    if not source['audit_passed'] or source['smoke']:
        raise ValueError('training requires the full audited overlay')
    if (source['dataset_sha256'] != file_sha256(DATA / 'data/dataset.npz')
            or source['oct_sha256'] != file_sha256(source['oct_overlay'])
            or source['c_sha256'] != file_sha256(source['c_overlay'])):
        raise ValueError('audited overlay or source changed before training')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-E', '-B', '-u', '-m', 'lerobot_training.dit_experience_replay', 'train',
               '--data-dir', str(DATA), '--val-data-dir', str(VAL), '--init-from', str(INITIAL),
               '--output-dir', str(args.checkpoint_dir), '--mode', 'ratio', '--correction-ratio', '.08',
               '--epochs', '4', '--steps-per-epoch', '1000', '--batch-size', '16', '--val-batch-size', '64',
               '--backbone-lr', '1e-6', '--attention-lr', '1e-5', '--weight-decay', '1e-4',
               '--seed', '20261118', '--validation-seed', '20261119', '--device', 'cuda', '--amp',
               '--checkpoint-every-steps', '1000', '--oct-overlay', source['oct_overlay'],
               '--c-label-overlay', source['c_overlay']]
    write_json(args.output_dir / 'command.json', command)
    subprocess.run(command, check=True, cwd=ROOT, timeout=args.stage_timeout_s)
    history = json.loads((args.checkpoint_dir / 'training_history.json').read_text())
    if (len(history) != 4 or history[-1]['batches_seen'] != 4000
            or history[-1]['sampling_counts'] != {'original': 58880, 'correction': 5120}
            or not all(np.isfinite(row['train_loss']) and np.isfinite(row['val_loss']) for row in history)):
        raise ValueError('training completeness/finite-loss/quota contract failed')
    for step in (1000, 2000, 3000, 4000):
        if not (args.checkpoint_dir / f'step_{step:06d}.pth').is_file():
            raise ValueError('missing checkpoint')
    write_json(args.output_dir / 'summary.json', {'status': 'completed', 'checkpoint_dir': str(args.checkpoint_dir),
                                                'batches': 4000, 'history': history})
    record('training', 'completed', args.checkpoint_dir, f'init={INITIAL}; dataset={DATA}; overlay={args.overlay_dir}; {history[-1]}')


def eval_groups(checkpoints, variants, indices, steps):
    result = {}
    fingerprint = identity()
    for name, checkpoint in checkpoints.items():
        for variant in variants:
            result[f'{name}_{variant}'] = {
                'factory': FACTORY + 'EvaluationWorker',
                'spec': {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
                         'target_manifest': str(TARGETS), 'max_steps': steps,
                         'variant': variant, 'identity': fingerprint},
                'tasks': [{'id': f'E{index:02d}', 'episode_index': index, 'seed': 42 + index} for index in indices]}
    return result


def summarize_group(rows):
    from scripts.research.control_g15_diagnostics import aggregate
    return {**aggregate(rows), 'episodes': rows, 'status': 'completed'}


def plots(output, rows):
    from matplotlib import pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 4))
    endpoints = []
    for row in rows:
        episode = json.loads(Path(row['result_path']).read_text())['episodes'][0]
        error = np.asarray(episode['trajectory']['errors']) * 1000
        axis.plot(error, label=f"E{row['episode_index']:02d}")
        endpoints.append([float(error.min()), row['final_error_um']])
    axis.set(xlabel='Control step', ylabel='Tip error (um)', yscale='log')
    axis.legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / 'error_curves_all.png', dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 4))
    x = np.arange(len(rows))
    values = np.asarray(endpoints)
    axis.bar(x-.2, values[:, 0], .4, label='Minimum')
    axis.bar(x+.2, values[:, 1], .4, label='Final')
    axis.set(xticks=x, xticklabels=[f"E{row['episode_index']:02d}" for row in rows], ylabel='Tip error (um)')
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'best_and_final_error.png', dpi=150)
    plt.close(fig)
    with (output / 'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['episode_index', 'seed', 'steps', 'final_error_um', 'strict_success'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)


def axis_diagnostics(rows):
    """Signed task residuals and first-order tip contributions, last 200 frames."""
    from types import SimpleNamespace
    from rcm.scene_v9 import RCMSceneV9
    from rcm.visual_servo import VisualServoController
    from scripts.research.clean_oct_workers import UNIT
    scene = RCMSceneV9()
    result = {'scope':'offline last min(200,N) frames; Jacobian column norms are not additive Euclidean components',
              'targets':{}}
    try:
        for row in rows:
            if row['episode_index'] not in (0,2,6,8):
                continue
            ep=json.loads(Path(row['result_path']).read_text())['episodes'][0]
            states=np.asarray(ep['trajectory']['states'],dtype=float)
            target=scene.task_from_encoders(ep['reachable_encoder_counts_ABC'], reference_task=states[-1]*UNIT)/UNIT
            gains=[]
            for state in states[-200:]:
                jacobian=VisualServoController._model_jacobian(SimpleNamespace(reference=scene),state*UNIT)
                gains.append(np.linalg.norm(jacobian*UNIT,axis=0)*1000)
            residual=states[-200:]-target
            result['targets'][f"E{row['episode_index']:02d}"]={
                'mean_residual_A_deg_B_deg_C_mm':np.mean(residual,axis=0).tolist(),
                'mean_first_order_axis_contribution_um':np.mean(np.abs(residual)*gains,axis=0).tolist(),
                'A_active_command_no_motion_fraction':row['a_nonzero_command_no_motion_fraction']}
    finally:
        scene.close()
    return result


def dev20_admission(selection, selection_path, authorization_path=None):
    observed = float(selection['best_final_mean_um'])
    if not np.isfinite(observed):
        raise ValueError('invalid fast5 selection metric')
    admission = {'fast5_summary': str(selection_path), 'fast5_final_mean_um': observed,
                 'original_gate': 'best_final_mean_um <45', 'original_gate_passed': observed < 45,
                 'user_authorized_exception': False}
    if authorization_path is not None:
        authorization = json.loads(Path(authorization_path).read_text())
        if (authorization.get('scope') != 'dev20_only'
                or not authorization.get('user_instruction')
                or not authorization.get('reason')
                or authorization.get('fast5_summary_sha256') != file_sha256(selection_path)
                or authorization.get('checkpoint_sha256') != file_sha256(selection['best_checkpoint'])
                or authorization.get('variant') != selection['best_variant']
                or authorization.get('observed_fast5_final_mean_um') != observed):
            raise ValueError('dev20 authorization does not match frozen selection/checkpoint')
        admission.update(user_authorized_exception=not admission['original_gate_passed'],
                         authorization=authorization,
                         authorization_sha256=file_sha256(authorization_path))
    if not admission['original_gate_passed'] and not admission['user_authorized_exception']:
        raise ValueError('dev20 branch not selected')
    return admission


def evaluate(args):
    from scripts.research.control_g15_diagnostics import paired
    started = time.monotonic()
    if args.mode == 'eval-smoke':
        groups=eval_groups({'initial':INITIAL}, ['V0'], (0,2,6), 8)
        result=run_pool(groups,args.output_dir/'episodes',workers=3,max_workers=3,
                        timeout_s=180,stage_timeout_s=180,resume=args.resume)
        rows=sorted(result.values(),key=lambda row:row['episode_index'])
        summary={**summarize_group(rows),'tier':'smoke','max_steps':8,
                 'purpose':'CUDA policy/driver/recording chain only; not improvement evidence'}
    elif args.mode == 'dev20':
        selection_path = args.fast5_dir / 'summary.json'
        selection = json.loads(selection_path.read_text())
        admission = dev20_admission(selection, selection_path,
                                   getattr(args, 'dev20_gate_authorization', None))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / 'admission.json', admission)
        record('dev20_admission', 'authorized', args.output_dir, json.dumps(admission))
        groups = eval_groups({'best': Path(selection['best_checkpoint'])}, [selection['best_variant']], list(range(20)), 1600)
        result = run_pool(groups, args.output_dir / 'episodes', workers=args.workers, max_workers=args.max_workers,
                          timeout_s=args.episode_timeout_s, stage_timeout_s=args.stage_timeout_s, resume=args.resume)
        rows = sorted(result.values(), key=lambda row: row['episode_index'])
        summary = summarize_group(rows)
        summary.update(tier='dev20', indices=list(range(20)), seeds=list(range(42,62)),
                       checkpoint=selection['best_checkpoint'], variant=selection['best_variant'],
                       admission=admission,
                       paired_teacher_excess_um=None, teacher_comparison_reason='no matched dev20 teacher evaluation')
    else:
        checkpoints = {f'step{step}': args.checkpoint_dir / f'step_{step:06d}.pth' for step in (2000,4000)}
        first_groups = eval_groups(checkpoints, ['V0'], FAST5, 1000)
        first = run_pool(first_groups, args.output_dir / 'v0_selection', workers=args.workers,
                         max_workers=args.max_workers, timeout_s=args.episode_timeout_s,
                         stage_timeout_s=args.stage_timeout_s, resume=args.resume)
        runs = {name: summarize_group([first[f'{name}_V0_E{i:02d}'] for i in FAST5]) for name in checkpoints}
        best_name = min(runs, key=lambda name: (runs[name]['final_mean_um'], runs[name]['final_max_um'], name))
        checkpoint = checkpoints[best_name]
        remaining = args.stage_timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('Stage 3 combined budget exhausted after checkpoint selection')
        second_groups = eval_groups({'best': checkpoint}, ['V3','V2'], FAST5, 1000)
        second = run_pool(second_groups, args.output_dir / 'p2_comparison', workers=args.workers,
                          max_workers=args.max_workers, timeout_s=args.episode_timeout_s,
                          stage_timeout_s=args.stage_timeout_s, execution_timeout_s=remaining, resume=args.resume)
        variants = {'V0': runs[best_name], **{variant: summarize_group([second[f'best_{variant}_E{i:02d}'] for i in FAST5])
                                             for variant in ('V3','V2')}}
        best_variant = min(variants, key=lambda name: (variants[name]['final_mean_um'], variants[name]['final_max_um'], name))
        baseline = json.loads(BASELINE.read_text())
        summary = {'status': 'completed', 'tier': 'diagnostic_fast5', 'indices': list(FAST5),
                   'seeds': [42+i for i in FAST5], 'max_steps': 1000,
                   'checkpoints_v0': runs, 'variants': variants, 'best_checkpoint': str(checkpoint),
                   'best_variant': best_variant, 'best_final_mean_um': variants[best_variant]['final_mean_um'],
                   'baseline': str(BASELINE), 'baseline_checkpoint': str(INITIAL),
                   'baseline_reference_only': True,
                   'v0_minus_historical_baseline': paired(variants['V0']['episodes'], baseline['episodes']),
                   'p2_minus_new_v0': {variant: paired(variants[variant]['episodes'], variants['V0']['episodes'])
                                       for variant in ('V3','V2')},
                   'dev20_selected': variants[best_variant]['final_mean_um'] < 45}
        rows = variants[best_variant]['episodes']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots(args.output_dir, rows)
    write_json(args.output_dir / 'axis_diagnostics.json',axis_diagnostics(rows))
    summary['elapsed_s'] = time.monotonic() - started
    write_json(args.output_dir / 'summary.json', summary)
    record(args.mode, 'completed', args.output_dir, f'dataset=N/A; seeds=42+index; {summary.get("best_final_mean_um", summary.get("final_mean_um"))} um; strict={sum(row["strict_success"] for row in rows)}/{len(rows)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight','smoke','eval-smoke','render','train','fast5','dev20'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--overlay-dir', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--fast5-dir', type=Path)
    parser.add_argument('--dev20-gate-authorization', type=Path,
                        help='Explicit user exception bound to the frozen fast5/checkpoint SHA256')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--episode-timeout-s', type=float, default=180)
    parser.add_argument('--stage-timeout-s', type=float, default=2400)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.dev20_gate_authorization is not None and args.mode != 'dev20':
        parser.error('--dev20-gate-authorization requires --mode dev20')
    args.output_dir = args.output_dir.resolve()
    for name in ('overlay_dir', 'checkpoint_dir', 'fast5_dir'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES='0', MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                      MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    before = identity()
    try:
        if args.mode == 'preflight':
            if args.output_dir.exists():
                raise FileExistsError(args.output_dir)
            groups = render_groups()
            args.output_dir.mkdir(parents=True)
            write_json(args.output_dir / 'render_tasks.json', groups)
            write_json(args.output_dir / 'summary.json', {'status': 'completed', 'identity': before,
                        'episodes': len(groups['render']['tasks']),
                        'frames': sum(len(task['rows']) for task in groups['render']['tasks']),
                        'task_signature': signature(groups)})
            record('preflight', 'completed', args.output_dir, 'all frozen source targets resolved; no render/training')
        elif args.mode in ('render','smoke'):
            render(args)
        elif args.mode == 'train':
            train(args)
        else:
            evaluate(args)
        if identity() != before:
            raise ValueError('code or scene changed during stage')
        write_json(args.output_dir / 'status.json', {'status':'completed','mode':args.mode})
    except BaseException as error:
        if args.output_dir.is_dir():
            write_json(args.output_dir / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                                        'failure':repr(error),'mode':args.mode})
        record(args.mode, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', args.output_dir, repr(error))
        raise
    print(f'{args.mode} result: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
