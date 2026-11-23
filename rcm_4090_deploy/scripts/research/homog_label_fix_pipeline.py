"""Execute planning file 6 with immutable audits and conditional evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import gc
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_label_fix_common import (
    BASE, BRANCH, FAST5, FIX_DATA, LATE_DRIFT_START, OLD, QUALITY, REPORTS,
    ROOT, SOURCE_DATA, SOURCE_REPO, STEPS, TARGETS, TASK, VAL, WEIGHTS, WORKFLOW,
    asset_fingerprints, env_command, identity, read, record, write_json,
)
from scripts.research.homog_label_fix_repair import build_fixed_dataset

MODULE = 'scripts.research.homog_label_fix_pipeline'


def card():
    return {
        'schema': 1, 'task_file': str(TASK), 'task_sha256': file_sha256(TASK),
        'base': BASE, 'branch': BRANCH,
        'hypothesis': 'Correct terminal and magnitude label contradictions in the clean same-render replay package',
        'main_change': 'Reuse all RGB/OCT/state images; repair T1/T2/T3 action supervision only',
        'authorization': '2026-11-23 user supplied planning6.txt and confirmed the T2 compatibility resolution',
        'T1_rule': 'Zero-initial-velocity teacher direction scaled to accepted student-action norm',
        'T2_rule': 'Zero-initial-velocity teacher direction scaled to accepted teacher-action norm',
        'T3_rule': 'Exact zero label on encoder-stationary below5um frames; executed_action elsewhere',
        'data': {'source': str(SOURCE_DATA), 'fixed_labels': str(FIX_DATA)},
        'initial_checkpoint': str(OLD),
        'training': {'batches': 4000, 'batches_per_epoch': 1000, 'batch_size': 16,
            'correction_ratio': .08, 'dagger_repeat': 1, 'amp': True,
            'backbone_lr': 1e-6, 'attention_lr': 1e-5,
            'architecture': 'state_conditioned_temporal_attention_v1'},
        'fast5': {'indices': list(FAST5), 'seeds': [42+i for i in FAST5], 'steps': STEPS,
            'samples': 1, 'controller': 'feedback', 'stop_on_success': False},
        'stage1_gates': {
            'T3_stationary_zero_fraction': 1.0,
            'T2_label_executed_norm_median': [.95, 1.05],
            'axis_reverse_fraction_max': .02,
            'valid_window_continuity': 1.0,
        },
        'stage3_gates': {
            'late_C_rms_max_mm_per_step': .0019,
            'A_snapshot_reverse_max': '2/9',
            'first_order_toward_fraction_min': .65,
            'paired_final_mean_must_improve': True,
        },
        'optional_arms_trigger': 'best V0 fast5 final mean < paired task2_step4000',
        'optional_arms': ['I_med5', 'V3'],
        'dev20_trigger': 'best final mean<40um, paired improvement, Stage3 checks pass, absolute late drift<15um',
        'budgets': dict.fromkeys(('stage_timeout_s', 'training_timeout_s',
                                  'episode_timeout_s', 'initializer_timeout_s'), None),
        'failure_policy': 'Stop dependent stages on audit/NaN/illegal state; no wall-clock timeout; no automatic retry',
        'simulation_scope': 'Software/MuJoCo only; no clinical or hardware claim',
    }


def ensure_branch():
    git = ['git', '-C', str(SOURCE_REPO)]
    subprocess.run(git+['cat-file', '-e', BASE+'^{commit}'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    exists = subprocess.run(git+['show-ref', '--verify', '--quiet',
        f'refs/heads/{BRANCH}']).returncode == 0
    if not exists:
        subprocess.run(git+['update-ref', f'refs/heads/{BRANCH}', BASE], check=True)
    branch_parent = subprocess.check_output(
        git+['rev-parse', BRANCH], text=True).strip()
    if subprocess.check_output(
            git+['merge-base', BASE, branch_parent], text=True).strip() != BASE:
        raise ValueError('New publication branch does not descend from authorized base')
    return branch_parent


def preflight(output):
    if output != WORKFLOW:
        raise ValueError(f'Use the authorized workflow path: {WORKFLOW}')
    if output.exists():
        manifest_path = output/'execution_manifest.json'
        if not manifest_path.is_file():
            raise FileExistsError(f'Unregistered workflow destination: {output}')
        frozen = read(manifest_path)
        expected_card = card()
        if (frozen.get('card') != expected_card
                or frozen.get('identity') != identity()
                or frozen.get('assets_sha256') != asset_fingerprints()):
            raise ValueError('Existing preflight source/card/assets changed')
        return frozen
    if FIX_DATA.exists() or WEIGHTS.exists() or any(path.exists() for path in REPORTS):
        raise FileExistsError('One or more fixed-label/report/weight destinations are occupied')
    if (FIX_DATA / 'shards').is_symlink():
        raise ValueError('Unexpected fixed-label symlink before preflight')
    for path in (TASK, SOURCE_DATA/'data/dataset.npz', SOURCE_DATA/'data/image_paths.json',
                 SOURCE_DATA/'dataset_audit.json', OLD, VAL/'data/dataset.npz', TARGETS):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hash = file_sha256(SOURCE_DATA/'data/dataset.npz')
    blob = subprocess.check_output(
        ['git', '-C', str(SOURCE_REPO), 'hash-object', str(SOURCE_DATA/'data/dataset.npz')],
        text=True).strip()
    published_blob = subprocess.check_output(
        ['git', '-C', str(SOURCE_REPO), 'rev-parse',
         BASE+':rcm_4090_deploy/data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2/data/dataset.npz'],
        text=True).strip()
    if blob != published_blob:
        raise ValueError('Source dataset differs from authorized cd22e10 tree')
    from scripts.research.a_axis_flip_common import gpu_check
    gpu = gpu_check()
    branch_parent = ensure_branch()
    with np.load(SOURCE_DATA/'data/dataset.npz', allow_pickle=False) as archive:
        source_shape = {key: list(archive[key].shape) for key in archive.files}
    output.mkdir(parents=True)
    manifest = {'card': card(), 'identity': identity(), 'assets_sha256': asset_fingerprints(),
        'source_dataset_sha256': source_hash, 'source_dataset_git_blob': blob,
        'branch_parent': branch_parent, 'source_shape': source_shape,
        'gpu': gpu, 'fixed_data': str(FIX_DATA), 'weights': str(WEIGHTS),
        'reports': [str(path) for path in REPORTS],
        'source_images_policy': 'Read-only symlink; no pixel or OCT rewrite',
        'all_timeouts': None, 'automatic_retry': False}
    write_json(output/'execution_manifest.json', manifest)
    write_json(output/'preflight.json', {'status': 'passed', **manifest})
    record('preflight', 'passed', output, {
        'base': BASE, 'source_dataset_sha256': source_hash,
        'source_frames': source_shape['state'], 'fast5': FAST5,
        'fixed_data': str(FIX_DATA), 'weights': str(WEIGHTS)})
    return manifest


def stage1(manifest):
    destination = REPORTS[0]
    summary_path = destination/'summary.json'
    if summary_path.exists():
        audit = read(destination/'label_fix_audit.json')
        receipt = read(summary_path)
        if (audit['identity'] != manifest['identity']
                or file_sha256(FIX_DATA/'data/dataset.npz') != receipt['fixed_dataset_sha256']):
            raise ValueError('Fixed-label resume provenance changed')
        return audit
    return build_fixed_dataset(destination)


def stage2(manifest, audit):
    from scripts.research.homog_replay_train import train
    destination = REPORTS[1]
    destination.mkdir(parents=True, exist_ok=True)
    replay = {'branch': 'H', 'audit': audit, 'data_dir': str(FIX_DATA),
        'dataset_sha256': file_sha256(FIX_DATA/'data/dataset.npz'),
        'label_repair_sha256': file_sha256(REPORTS[0]/'label_fix_audit.json'),
        'history_action_source': 'executed_action',
        'supervision_action_source': 'homog_label_fix_20261123'}
    receipt = train(replay, WEIGHTS, manifest['identity'])
    result = {'status': 'completed', 'training': receipt,
        'checkpoint_dir': str(WEIGHTS),
        'fixed_dataset_sha256': replay['dataset_sha256'],
        'checkpoints': {path.name: file_sha256(path) for path in sorted(WEIGHTS.glob('step_*.pth'))}}
    write_json(destination/'summary.json', result)
    return result


def late_c_rms(result):
    values = []
    for row in result['episodes']:
        episode = read(row['result_path'])['episodes'][0]
        action = np.asarray(episode['trajectory']['raw_actions'], dtype=np.float64)
        values.extend(action[1200:1600, 2].tolist())
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate_arm(checkpoint, samples, controller, destination, manifest):
    from scripts.research.homog_replay_evaluate import evaluate
    from scripts.research.homog_replay_pipeline import plots
    result = evaluate(checkpoint, samples, controller, FAST5, destination, manifest)
    plots(result, destination)
    result['hit_rates'] = {
        f'le_{threshold}_um': result[f'final_hits_le_{threshold}_um']/len(FAST5)
        for threshold in (5, 10, 30)}
    result['late_C_rms_1200_to_1600_mm_per_step'] = late_c_rms(result)
    write_json(destination/'summary.json', result)
    return result


def stage3(trained, manifest):
    from scripts.research.a_axis_flip_audit import snapshots
    destination = REPORTS[2]
    destination.mkdir(parents=True, exist_ok=True)
    baseline = evaluate_arm(OLD, 1, 'feedback',
        destination/'baseline_task2_step4000', manifest)
    comparisons = {
        f'V0_step{step}': evaluate_arm(WEIGHTS/f'step_{step:06d}.pth', 1, 'feedback',
            destination/f'V0_step{step}', manifest)
        for step in (2000, 4000)}
    selected_name = min(comparisons, key=lambda name: comparisons[name]['final_mean_um'])
    selected = comparisons[selected_name]
    selected_checkpoint = selected['protocol']['checkpoint']
    snapshot_dir = destination/'selected_snapshot_recheck'
    snapshot_dir.mkdir(exist_ok=True)
    snapshot = snapshots({'BEST_CKPT': selected_checkpoint}, snapshot_dir,
                         manifest['identity'])['BEST_CKPT']
    reverse_a = int(snapshot['axis_reverse_snapshot_counts']['A'])
    toward = float(snapshot['first_order_toward_fraction'])
    late_c = float(selected['late_C_rms_1200_to_1600_mm_per_step'])
    improved = selected['final_mean_um'] < baseline['final_mean_um']
    diagnostics = {
        'late_C_rms_le_0p0019': {'value': late_c, 'passed': late_c <= .0019},
        'A_snapshot_reverse_le_2_of_9': {'value': reverse_a, 'passed': reverse_a <= 2},
        'first_order_toward_gt_0p65': {'value': toward, 'passed': toward > .65},
        'paired_final_mean_improved': {
            'value': selected['final_mean_um'] - baseline['final_mean_um'],
            'passed': improved},
    }
    checks_passed = all(value['passed'] for value in diagnostics.values())
    optional_status = 'skipped_trigger_failed'
    if selected['final_mean_um'] < baseline['final_mean_um']:
        comparisons['I_med5'] = evaluate_arm(selected_checkpoint, 5, 'feedback',
            destination/'I_med5', manifest)
        comparisons['V3'] = evaluate_arm(selected_checkpoint, 1, 'continuous',
            destination/'V3', manifest)
        optional_status = 'completed'
    best_name = min(comparisons, key=lambda name: comparisons[name]['final_mean_um'])
    best = comparisons[best_name]
    best_late_drift = float(best[f'absolute_drift_{LATE_DRIFT_START}_to_{STEPS}_um'])
    final_under_40 = best['final_mean_um'] < 40.
    late_stable = best_late_drift < 15.
    dev20_gate = bool(checks_passed and final_under_40 and late_stable)
    result = {
        'status': 'completed', 'baseline': baseline, 'comparisons': comparisons,
        'selected_V0': selected_name, 'selected_checkpoint': selected_checkpoint,
        'snapshot_recheck': snapshot, 'diagnostics': diagnostics,
        'stage3_checks_passed': checks_passed,
        'paired_final_error_improved': improved,
        'optional_arms_status': optional_status,
        'best_combination': best_name, 'best': best,
        'final_under_40um': final_under_40,
        'late_rebound_gate_abs_drift_lt_15um': late_stable,
        'dev20_gate_passed': dev20_gate,
    }
    write_json(destination/'summary.json', result)
    import shutil
    for name in ('error_curves_all.png', 'best_and_final_error.png',
                 'axis_floor_summary.json'):
        shutil.copy2(destination/best_name/name, destination/name)
    record('stage3_fast5', 'completed', destination, {
        'baseline_final_mean_um': baseline['final_mean_um'],
        'selected_V0': selected_name,
        'selected_final_mean_um': selected['final_mean_um'],
        'late_C_rms': late_c, 'A_reverse': reverse_a,
        'toward_fraction': toward, 'checks_passed': checks_passed,
        'optional_arms': optional_status, 'best_combination': best_name,
        'best_final_mean_um': best['final_mean_um'],
        'best_absolute_late_drift_um': best_late_drift,
        'dev20_gate_passed': dev20_gate})
    return result


def stage4(selected, manifest):
    destination = REPORTS[3]
    destination.mkdir(parents=True, exist_ok=True)
    protocol = selected['best']['protocol']
    result = evaluate_arm(protocol['checkpoint'], protocol['samples'],
        protocol['controller'], destination, manifest)
    result['best_combination'] = selected['best_combination']
    result['trigger_evidence'] = {
        'fast5_final_mean_um': selected['best']['final_mean_um'],
        'fast5_absolute_late_drift_um':
            selected['best'][f'absolute_drift_{LATE_DRIFT_START}_to_{STEPS}_um'],
        'stage3_checks_passed': selected['stage3_checks_passed'],
    }
    write_json(destination/'summary.json', result)
    record('stage4_dev20', 'completed', destination, {
        'checkpoint': protocol['checkpoint'], 'final_mean_um': result['final_mean_um'],
        'absolute_late_drift_um':
            result[f'absolute_drift_{LATE_DRIFT_START}_to_{STEPS}_um'],
        'strict_success_count': result['strict_success_count']})
    return result


def summary(output, stages, status):
    lines = ['# Homogeneous replay label repair and retraining — 2026-11-23', '',
        f'Workflow status: `{status}`. Base: `{BASE}`. Branch: `{BRANCH}`.', '',
        'All images and OCT observations were reused. Only the action supervision array changed.', '',
        'No wall-clock timeout was configured. Training/inference resume state is written atomically.', '']
    for name, result in stages.items():
        brief = {key: result.get(key) for key in (
            'status', 'reason', 'stage3_checks_passed', 'dev20_gate_passed',
            'best_combination', 'optional_arms_status') if key in result}
        lines += [f'## {name}', '', '```json',
                  json.dumps(brief, ensure_ascii=False, indent=2), '```', '']
    if 'stage3' in stages and stages['stage3'].get('status') == 'completed':
        selected = stages['stage3']
        lines += [
            f"Baseline fast5 final mean: {selected['baseline']['final_mean_um']:.6f} um.",
            f"Best fast5 final mean: {selected['best']['final_mean_um']:.6f} um.",
            f"Best absolute late drift: {selected['best'][f'absolute_drift_{LATE_DRIFT_START}_to_{STEPS}_um']:.6f} um.",
            f"Strict success: {selected['best']['strict_success_count']}/{selected['best']['completed_targets']}.", '']
    lines += ['Weights and raw correction RGB remain local with path/size/SHA256 inventories in the publication record.',
              '', 'This result validates software and MuJoCo behavior only; it is not clinical or hardware accuracy evidence.', '']
    (output/'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def run(output):
    with (output/'workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Workflow is already running') from error
        stages = {}
        status = 'failed'
        try:
            manifest = read(output/'execution_manifest.json')
            if manifest['identity'] != identity():
                raise ValueError('Source identity changed before execution')
            def interrupted(sig, _):
                raise KeyboardInterrupt(signal.Signals(sig).name)
            for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, interrupted)
            write_json(output/'workflow_status.json',
                       {'status': 'running', 'stage': 'stage1'})
            audit = stage1(manifest)
            stages['stage1'] = {**audit, 'status': 'completed'}
            write_json(output/'stage_results.json', stages)
            write_json(output/'workflow_status.json',
                       {'status': 'running', 'stage': 'stage2'})
            stages['stage2'] = stage2(manifest, audit)
            write_json(output/'stage_results.json', stages)
            write_json(output/'workflow_status.json',
                       {'status': 'running', 'stage': 'stage3'})
            stages['stage3'] = stage3(stages['stage2'], manifest)
            write_json(output/'stage_results.json', stages)
            if stages['stage3']['dev20_gate_passed']:
                write_json(output/'workflow_status.json',
                           {'status': 'running', 'stage': 'stage4'})
                stages['stage4'] = stage4(stages['stage3'], manifest)
            else:
                stages['stage4'] = {'status': 'skipped',
                    'reason': 'Stage3 late-C/direction/improvement or final<40um/late-drift gate failed'}
            write_json(output/'stage_results.json', stages)
            status = 'completed'
            write_json(output/'workflow_status.json',
                       {'status': status, 'stages': {
                           key: value['status'] for key, value in stages.items()}})
            record('workflow', status, output,
                   {key: value['status'] for key, value in stages.items()})
        except BaseException as error:
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            stages['failure'] = {'status': status, 'reason': repr(error)}
            write_json(output/'failure.json', stages['failure'])
            write_json(output/'workflow_status.json', stages['failure'])
            write_json(output/'stage_results.json', stages)
            record('workflow', status, output, f'{error!r}; no automatic retry')
            raise
        finally:
            summary(output, stages, status)
            from scripts.research.homog_label_fix_publish import publish
            try:
                publish(output, push=True)
            except BaseException as error:
                write_json(output/'publication_failure.json', {'reason': repr(error)})
                record('publication', 'failed', output, repr(error))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'launch', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action in ('preflight', 'launch'):
        preflight(output)
        if args.action == 'preflight':
            print(output/'preflight.json')
            return
        command = env_command()+['-m', MODULE, 'run', '--output-dir', str(output)]
        log = output/'terminal.log'
        shell = ('set -o pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(command)
            +' 2>&1 | tee -a '+shlex.quote(str(log))
            +'\ncode=${PIPESTATUS[0]}\nprintf "\\nExperiment exited: %s\\n" "$code"\n'
            +'read -r -p "Press Enter to close this terminal..."\nexit "$code"\n')
        result = subprocess.run(['gnome-terminal',
            '--title=Homogeneous label repair and retraining 20261123',
            '--working-directory='+str(ROOT), '--', 'bash', '-c', shell],
            capture_output=True, text=True)
        receipt = {'accepted': result.returncode == 0, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr, 'command': command,
            'log': str(log), 'timestamp': datetime.now().astimezone().isoformat(),
            'no_wall_clock_limits': True, 'automatic_retry': False}
        write_json(output/'launch_receipt.json', receipt)
        record('launch', 'accepted' if receipt['accepted'] else 'failed', output, receipt)
        if not receipt['accepted']:
            raise RuntimeError(f'GNOME Terminal failed: {result.stderr}; no fallback')
        print(receipt)
        return
    return run(output)


if __name__ == '__main__':
    main()
