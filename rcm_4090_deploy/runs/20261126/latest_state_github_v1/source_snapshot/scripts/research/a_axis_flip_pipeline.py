"""Execute 规划4.txt once in GNOME Terminal, with explicit conditional gates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.a_axis_flip_common import (
    AUDIT, CLEAN, DATA, FAST5, FROZEN, NEW, OLD, PHASE2, PHASE3, PHASE4,
    QUALITY, ROOT, SOURCE, TARGETS, VAL, aggregate, arrays, code_identity, gpu_check,
    near_mask, pool, read, record, signature, write_json,
)

MODULE = 'scripts.research.a_axis_flip_pipeline'
BASE = 'dad76c522f7e74f3a86b56d41f6c89488ae878ff'
BRANCH = 'fix/a-axis-flip-20261123'
TASK = ROOT.parent / '规划4.txt'
PUB = ROOT / 'runs/20261123/a_axis_flip_git_v1'


def experiment_card():
    return {'schema': 1, 'task_file': str(TASK), 'task_sha256': file_sha256(TASK),
        'base_commit': BASE, 'branch': BRANCH,
        'hypothesis': 'Near-target inertia-reversed ABC labels or mixed RGB/OCT rendering provenance contribute to A reversal and late drift.',
        'main_change': 'One audit-selected repair branch only; B has priority over A.',
        'data_dir': str(DATA), 'validation_data_dir': str(VAL), 'initial_checkpoint': str(OLD),
        'original_pool': 'All valid unmodified merged training windows, 92%',
        'correction_pool': 'Independent repaired windows (A) or 50 same-observation training-target trajectories (B), 8%',
        'training': {'epochs': 4, 'batches_per_epoch': 1000, 'batch_size': 16,
                     'dagger_repeat': 1, 'amp': True, 'save_steps': [2000, 4000]},
        'budgets': {'stage_timeout_s': None, 'training_timeout_s': None,
                    'episode_timeout_s': None, 'initializer_timeout_s': None},
        'snapshot_protocol': {'episodes': [0, 8, 10], 'steps': [800, 1000, 1200],
                              'seeds': list(range(2026112200, 2026112232)), 'samples_per_snapshot': 32},
        'evaluation_protocol': {'tier': 'user_authorized_fast5_then_conditional_dev20',
            'fast5': list(FAST5), 'seeds': [42+i for i in FAST5], 'steps': 1600,
            'control_period_s': .05, 'quality': QUALITY, 'flow_steps': 4,
            'stop_on_success': False, 'strict_success': 'error<0.35um and 3 settled frames, diagnostic only',
            'drift': 'final post1600 error minus pre-action trace[1000] error; signed mean gates, absolute mean also reported'},
        'branch_rule': 'A if A or B direction-conflict >=20% of all-bucket eligible nonzero zero/pre-acceleration teacher frames; B if old A reverse<=3/9 and new>=7/9 plus provenance mismatch or increased boundary median. B priority.',
        'direction_gate': 'A reverse<=1/9 and toward>75% and fast5 signed late drift<15um',
        'dev20_gate': 'Best complete fast5 finalmean<35um and signed late drift<15um',
        'collection': {'trajectories': 50, 'steps_per_trajectory': 1600,
            'targets': 'First50 eligible training plan2 episodes, frozen before launch, excluding development targets',
            'start': 'First source valid150-200um frame; if absent first valid<200um frame',
            'seed': '2026112300+trajectory index', 'near_labels_only': True},
        'output_paths': {'audit': str(AUDIT), 'phase2': str(PHASE2),
            'phase3': str(PHASE3), 'phase4': str(PHASE4),
            'dataset': str(ROOT / 'data/a_axis_fix_20261123_<A_or_B>_v1'),
            'checkpoints': str(ROOT / 'outputs/a_axis_fix_20261123_<A_or_B>_v1')},
        'reuse': 'Only complete per-task receipts with matching frozen inputs/code and evidence hashes; no failed-stage automatic retries.',
        'simulation_scope': 'Software/MuJoCo only; no clinical or hardware readiness claims.'}


def preflight(output, *, runtime=False):
    from scripts.research.a_axis_flip_audit import (
        snapshot_tasks, static_scene, make_teacher, pre_acceleration, seeded_parallel_actions,
    )
    from scripts.research.a_axis_flip_replay import collection_tasks
    from scripts.research.clean_oct_workers import egl_identity, resolve_target, target_world
    gpu = gpu_check()
    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise RuntimeError('No desktop display for required GNOME Terminal launch')
    card = experiment_card()
    checks = {}
    for path in (DATA / 'data/dataset.npz', DATA / 'metadata.json', DATA / 'data/image_paths.json',
                 VAL / 'data/dataset.npz', VAL / 'metadata.json', VAL / 'data/image_paths.json',
                 OLD, NEW, CLEAN / 'oct_override.npz', CLEAN / 'c_label_override.npz',
                 NEW.parent / 'training_config.json', TARGETS, SOURCE, FROZEN / 'summary.json'):
        checks[str(path)] = file_sha256(path)
    if checks[str(DATA / 'data/dataset.npz')] != '9d80523f519faf4e00bbaef3d30e29dec1401181f3dcddd438c5087c05eb54db':
        raise ValueError('user base training dataset changed')
    if checks[str(NEW)] != '705dc36de56d3efd659fd9d0c47f33bd91d8a9c3c6e5ea32e5dbcbcec0266523':
        raise ValueError('clean OCT checkpoint changed')
    git_info = read(PUB / 'branch.json')
    if git_info['base'] != BASE or git_info['branch'] != BRANCH:
        raise ValueError('independent branch base/name mismatch')
    current_base = subprocess.check_output(['git', '-C', git_info['repository'], 'merge-base', BRANCH, BASE], text=True).strip()
    if current_base != BASE:
        raise ValueError('branch no longer descends from authorized base')
    for folder in (DATA, VAL):
        paths = read(folder / 'data/image_paths.json')
        missing = [p for p in paths if not (folder / p.replace('\\', '/')).is_file()]
        if missing:
            raise FileNotFoundError(f'{folder}: {len(missing)} images missing; first={missing[:2]}')
    frozen = read(FROZEN / 'summary.json')
    if frozen['status'] != 'completed' or frozen['snapshots'] != 9 or frozen['samples'] != 288:
        raise ValueError('frozen diagnostic evidence incomplete')
    source_hashes = read(ROOT / 'runs/20261122/analysis/clean_oct_command_execution_v1/source_hashes.json')
    snaps = snapshot_tasks()
    for t in snaps:
        if t['source_sha256'] != source_hashes[t['source_path']]:
            raise ValueError('snapshot source trace differs from frozen audit')
        receipt_dir = FROZEN / f"shards/tasks/fixed_E{t['episode_index']:02d}"
        receipt = read(receipt_dir / 'receipt.json')
        for name, sha in receipt['files'].items():
            if file_sha256(receipt_dir / name) != sha:
                raise ValueError(f'frozen snapshot receipt mismatch: {name}')
    tasks = collection_tasks()
    a = arrays()
    near = near_mask(a)
    # Resolve all audit targets before starting a long render pass.
    targets = {int(i): resolve_target(DATA, int(i)) for i in np.unique(a['episode_index'][near])}
    scene = static_scene()
    teacher = None
    try:
        for i, target in targets.items():
            row = int(np.flatnonzero(near & (a['episode_index'] == i))[0])
            world = target_world(scene, target, a['state'][row] * np.array([np.pi/180, np.pi/180, .001]))
            scene.set_target_world(world)
            scene.set_task(a['state'][row] * np.array([np.pi/180, np.pi/180, .001]))
        # A small explicit NVIDIA render/teacher preflight, not an accuracy experiment.
        first = tasks[0]
        from scripts.research.a_axis_flip_common import UNIT
        q = np.asarray(first['initial_state_deg_deg_mm']) * UNIT
        world = target_world(scene, first['target'], q)
        scene.set_target_world(world)
        scene.set_task(q)
        teacher = make_teacher(scene)
        teacher.sensor.observation_time_s = 0.
        obs = teacher.sensor.observe()
        zero = teacher.geometry_action_from_observation(q, obs, previous_velocity=np.zeros(3), command_q=q)
        if not zero['valid']:
            raise ValueError(f'preflight actual RGB-D teacher invalid: {zero}')
        pre_acceleration(teacher, scene, q, obs, zero)
        egl = egl_identity()
    finally:
        if teacher is not None:
            teacher.sensor.close()
        scene.close()
    import torch
    from lerobot_training.dit_experience_replay import _load_policy
    architectures = {}
    for path in (OLD, NEW):
        model, config, epoch = _load_policy(path, torch.device('cuda'))
        model.eval()
        with np.load(FROZEN/'shards/tasks/fixed_E00/state_800_reconstruction.npz', allow_pickle=False) as f:
            tensors = [torch.from_numpy(f[key].copy()).to('cuda')
                       for key in ('images', 'states', 'oct', 'velocity', 'previous_action')]
        batched = seeded_parallel_actions(model, tensors)
        with torch.no_grad():
            single = torch.cat([model.sample(*tensors, steps=4,
                generator=torch.Generator(device='cuda').manual_seed(seed))
                for seed in card['snapshot_protocol']['seeds']])
        torch.testing.assert_close(batched, single, rtol=2e-4, atol=2e-5)
        architectures[str(path)] = {'epoch': epoch, 'n_obs_steps': config['n_obs_steps'],
                                    'n_action_steps': config['n_action_steps'],
                                    'parallel32_sequential32_max_abs': float((batched-single).abs().max())}
        if config['n_obs_steps'] != 2 or config['n_action_steps'] != 16:
            raise ValueError('checkpoint architecture/history differs')
        del model, batched, single, tensors
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {'card': card, 'inputs_sha256': checks, 'identity': code_identity(),
                'snapshot_tasks': snaps, 'collection_tasks': tasks,
                'audit_target_provenance': targets, 'all_near_source_frames': int(near.sum())}
    # JSON normalizes integer dictionary keys; use the normalized form for resume comparison.
    manifest = json.loads(json.dumps(manifest))
    destination = output / 'execution_manifest.json'
    if destination.exists():
        if read(destination) != manifest:
            raise ValueError('preflight inputs/code/task changed; refusing mixed-provenance resume')
    else:
        for path in (AUDIT, PHASE2, PHASE3, PHASE4):
            if path.exists():
                raise FileExistsError(f'unregistered occupied plan destination: {path}')
        write_json(destination, manifest)
    result = {'status': 'passed', 'gpu': gpu, 'egl': egl, 'architectures': architectures,
              'all_near_source_frames': int(near.sum()), 'collection_trajectories': len(tasks),
              'no_wall_clock_limits': True, 'manifest_sha256': signature(manifest)}
    write_json(output / ('runtime_preflight.json' if runtime else 'preflight.json'), result)
    record('preflight', 'completed', output, result)
    return manifest


def completed_stage(output, name, function):
    """Reuse a completed stage only after verifying every retained evidence hash."""
    receipt_path = output / f'{name}_receipt.json'
    if receipt_path.exists():
        r = read(receipt_path)
        if any(not Path(p).is_file() or file_sha256(p) != sha for p, sha in r['files'].items()):
            raise ValueError(f'completed {name} evidence changed')
        return r['result']
    result = function()
    # Per-episode receipts protect their own trees; stage JSON/NPZ protects summary assembly.
    directory = {'audit': AUDIT, 'phase2': PHASE2, 'phase3': PHASE3, 'phase4': PHASE4}[name]
    files = {str(p): file_sha256(p) for p in directory.rglob('*')
             if p.is_file() and p.suffix in ('.json', '.npz', '.md', '.csv', '.png')}
    write_json(receipt_path, {'result': result, 'files': files})
    record(name, 'completed', directory, result)
    return result


def phase1(manifest):
    from scripts.research.a_axis_flip_audit import snapshots, label_audit, consistency_audit, select_branch
    AUDIT.mkdir(parents=True, exist_ok=True)
    snap = snapshots({'A_old': OLD, 'A_new': NEW}, AUDIT, manifest['identity'])
    labels = label_audit(AUDIT, manifest['identity'])
    consistency = consistency_audit(AUDIT)
    branch = select_branch(labels, snap, consistency)
    write_json(AUDIT / 'branch_decision.json', branch)
    (AUDIT / 'AUDIT_SUMMARY.md').write_text(
        '# A轴反向偏置审计\n\n' +
        f"分支 A 命中：{branch['branch_A_hit']}；分支 B 命中：{branch['branch_B_hit']}。\n\n"
        f"选定分支：{branch['selected_branch'] or '无；停止依赖阶段'}。B 优先。\n\n"
        f"全量近目标有效源帧覆盖：{labels['covered_frames']}；original/plan2 以 episode 839 分界。\n\n"
        f"原模型 A 反向快照 {branch['old_A_reverse_snapshots']}/9，新模型 {branch['new_A_reverse_snapshots']}/9。\n\n"
        '反向污染分母为零初速与限加速度前期望均非零的全量合格轴帧；同时保留全部近目标帧分母与历史速度重放兼容数。\n\n'
        '方向冲突是关联证据，不能单独证明惯性因果；RGB/OCT 渲染来源不同不等于逐像素不一致。\n\n'
        '本报告只验证软件与 MuJoCo 仿真。\n')
    return {'status': 'completed', **branch, 'snapshot': snap, 'labels': labels,
            'consistency': consistency}


def phase2(audit, manifest):
    from scripts.research.a_axis_flip_replay import build_abc_overlay, collect_dataset
    from scripts.research.a_axis_flip_train import train
    PHASE2.mkdir(parents=True, exist_ok=True)
    branch = audit['selected_branch']
    destination = ROOT / f'data/a_axis_fix_20261123_{branch}_v1'
    receipt_path = destination / 'dataset_receipt.json'
    if receipt_path.exists():
        replay = read(receipt_path)
        # Re-enter collection pool to verify every image/shard receipt before training reuse.
        if branch == 'B':
            replay = collect_dataset(destination, PHASE2, manifest['collection_tasks'], manifest['identity'])
        elif file_sha256(replay['overlay']) != replay['overlay_sha256']:
            raise ValueError('ABC overlay changed')
    elif branch == 'A':
        replay = build_abc_overlay(audit['labels'], destination)
    else:
        replay = collect_dataset(destination, PHASE2, manifest['collection_tasks'], manifest['identity'])
    checkpoint_dir = ROOT / f'outputs/a_axis_fix_20261123_{branch}_v1'
    trained = train(replay, checkpoint_dir, manifest['identity'])
    result = {'status': 'completed', 'branch': branch, 'replay': replay,
              'training': trained, 'checkpoint_dir': str(checkpoint_dir)}
    write_json(PHASE2 / 'summary.json', result)
    return result


def evaluate(checkpoint, samples, controller, indices, destination, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    spec = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
            'samples': samples, 'controller': controller, 'quality': QUALITY,
            'identity': manifest['identity'], 'steps': 1600, 'stop_on_success': False}
    groups = {'eval': {'factory': 'scripts.research.a_axis_flip_common:EvaluationWorker', 'spec': spec,
                      'tasks': [{'id': f'E{i:02d}', 'episode_index': i, 'seed': 42+i} for i in indices]}}
    results = pool(groups, destination / 'shards', snapshot_every=5)
    rows = sorted(results.values(), key=lambda r: r['episode_index'])
    if [r['episode_index'] for r in rows] != list(indices):
        raise ValueError('evaluation target coverage mismatch')
    result = {'status': 'completed', 'protocol': spec, **aggregate(rows),
              'metric_scope': 'User-authorized development fast5 or dev20; strict success is diagnostic only.'}
    for row in rows:
        write_json(destination / f"E{row['episode_index']:02d}.json", row)
    write_json(destination / 'summary.json', result)
    record('fixed1600_evaluation', 'completed', destination,
           {k: v for k, v in result.items() if k not in ('protocol', 'episodes')})
    return result


def direction_gate(snap, fast):
    return (snap['axis_reverse_snapshot_counts']['A'] <= 1
            and snap['first_order_toward_fraction'] > .75
            and fast['drift_1000_to_1600_um'] < 15)


def dev20_gate(best):
    return best['final_mean_um'] < 35 and best['drift_1000_to_1600_um'] < 15


def phase3(trained, manifest):
    from scripts.research.a_axis_flip_audit import snapshots
    PHASE3.mkdir(parents=True, exist_ok=True)
    ckpts = Path(trained['checkpoint_dir'])
    comparisons = {f'V0_step{step}': evaluate(ckpts/f'step_{step:06d}.pth', 1, 'feedback', FAST5,
                                            PHASE3/f'V0_step{step}', manifest) for step in (2000, 4000)}
    selected = min(comparisons, key=lambda k: comparisons[k]['final_mean_um'])
    best_checkpoint = comparisons[selected]['protocol']['checkpoint']
    snap_dir = PHASE3 / 'selected_snapshot_recheck'
    snap_dir.mkdir(exist_ok=True)
    snap = snapshots({'BEST_CKPT': best_checkpoint}, snap_dir, manifest['identity'])['BEST_CKPT']
    improved = direction_gate(snap, comparisons[selected])
    if improved:
        comparisons['I_med5'] = evaluate(best_checkpoint, 5, 'feedback', FAST5, PHASE3/'I_med5', manifest)
        comparisons['V3'] = evaluate(best_checkpoint, 1, 'continuous', FAST5, PHASE3/'V3', manifest)
    best_name = min(comparisons, key=lambda k: comparisons[k]['final_mean_um'])
    result = {'status': 'completed', 'selected_V0': selected, 'best_checkpoint': best_checkpoint,
              'direction_gate_passed': improved, 'snapshot_recheck': snap,
              'optional_arms_status': 'completed' if improved else 'skipped_direction_gate_failed',
              'best_combination': best_name, 'best': comparisons[best_name],
              'dev20_gate_passed': dev20_gate(comparisons[best_name]), 'comparisons': comparisons,
              'improvement_reference': 'New checkpoints paired under identical full1600 protocol. Historical old runs are contextual, not silently reused as matched new-protocol baseline.'}
    write_json(PHASE3 / 'summary.json', result)
    return result


def phase4(selected, manifest):
    if not selected['dev20_gate_passed']:
        raise ValueError('dev20 gate failed; no launch authorized')
    PHASE4.mkdir(parents=True, exist_ok=True)
    p = selected['best']['protocol']
    result = evaluate(p['checkpoint'], p['samples'], p['controller'], tuple(range(20)), PHASE4, manifest)
    from matplotlib import pyplot as plt
    rows = result['episodes']
    fig, ax = plt.subplots(figsize=(12, 5))
    for row in rows:
        ep = read(row['result_path'])['episodes'][0]
        values = np.r_[np.asarray(ep['trajectory']['errors'])*1000, row['final_error_um']]
        ax.plot(np.arange(1601), values, label=f"E{row['episode_index']:02d}")
    ax.set(xlabel='Control step (50 ms)', ylabel='Tip error (um)', yscale='log')
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(PHASE4/'error_curves_all.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(20)
    ax.bar(x-.2, [r['min_error_um'] for r in rows], .4, label='Minimum')
    ax.bar(x+.2, [r['final_error_um'] for r in rows], .4, label='Final')
    ax.set(xticks=x, xticklabels=[f'E{i:02d}' for i in x], ylabel='Tip error (um)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(PHASE4/'best_and_final_error.png', dpi=150)
    plt.close(fig)
    with (PHASE4/'episodes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result['hit_rates'] = {f'le_{t}_um': result[f'final_hits_le_{t}_um']/20 for t in (5, 10, 30)}
    write_json(PHASE4/'summary.json', result)
    return result


def summary(output, stages, status):
    lines = ['# A-axis flip repair — 2026-11-23', '', f'流程状态：{status}。', '',
             f'基点 `{BASE}`；独立分支 `{BRANCH}`。', '',
             '所有阶段、训练、单集和初始化均无 wall-clock 超时。评估不以 0.35 µm 停机。', '']
    for name, result in stages.items():
        brief = {k: v for k, v in result.items() if k in ('status', 'reason', 'selected_branch',
                  'direction_gate_passed', 'dev20_gate_passed', 'best_combination',
                  'final_mean_um', 'min_mean_um', 'drift_1000_to_1600_um', 'strict_success_count')}
        lines += [f'## {name}', '', '```json', json.dumps(brief, ensure_ascii=False, indent=2), '```', '']
    if 'phase3' in stages and stages['phase3']['status'] == 'completed':
        best = stages['phase3']['best']
        lines += [f"Phase3 最优终值均值 {best['final_mean_um']:.6f} µm；最小误差均值 {best['min_mean_um']:.6f} µm；"
                  f"末段带符号漂移 {best['drift_1000_to_1600_um']:.6f} µm，绝对漂移均值 {best['absolute_drift_1000_to_1600_um']:.6f} µm。", '',
                  f"严格成功 {best['strict_success_count']}/{best['completed_targets']}。流程完成、指标改善、严格成功分别报告。", '']
    lines += ['报告见 `rcm_4090_deploy/github_reports/` 对应 audit/phase2/phase3/phase4 目录。', '',
              '训练权重与原始纠偏 RGB 图像保留本机，发布清单登记路径、大小和 SHA256；GitHub 保留代码、配置、日志、结构化审计和逐集评估证据。', '',
              '软件/MuJoCo 仿真结果，不代表临床精度或硬件就绪。']
    text = '\n'.join(lines)+'\n'
    (output/'SUMMARY.md').write_text(text)
    # Explicit task-requested repository-root summary, mirrored on publication.
    (PUB/'SUMMARY.md').write_text(text)


def _run(output):
    output.mkdir(parents=True, exist_ok=True)
    stages = {}
    try:
        manifest = preflight(output, runtime=True)
        def interrupted(sig, _):
            raise KeyboardInterrupt(signal.Signals(sig).name)
        for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        for name, fn in [('audit', lambda: phase1(manifest)),
                         ('phase2', lambda: phase2(stages['audit'], manifest)),
                         ('phase3', lambda: phase3(stages['phase2'], manifest)),
                         ('phase4', lambda: phase4(stages['phase3'], manifest))]:
            if name in ('phase2', 'phase3', 'phase4') and not stages['audit']['selected_branch']:
                stages[name] = {'status': 'skipped', 'reason': 'No audit branch triggered'}
            elif name == 'phase4' and not stages['phase3']['dev20_gate_passed']:
                stages[name] = {'status': 'skipped', 'reason': 'Phase3 finalmean<35um and late drift<15um gate failed'}
            else:
                if code_identity() != manifest['identity']:
                    raise ValueError('execution code/scene changed after frozen preflight')
                write_json(output/'workflow_status.json', {'status': 'running', 'stage': name,
                           'completed_stages': list(stages)})
                stages[name] = completed_stage(output, name, fn)
                if code_identity() != manifest['identity']:
                    raise ValueError('execution code/scene changed during stage')
            write_json(output/'stage_results.json', stages)
            print(f'STAGE {name}: {stages[name]["status"]}; results: {output/"stage_results.json"}', flush=True)
        status = 'completed_with_conditional_skips' if any(s['status']=='skipped' for s in stages.values()) else 'completed'
        summary(output, stages, status)
        write_json(output/'workflow_status.json', {'status': status, 'stages': {k: v['status'] for k,v in stages.items()}})
        record('workflow', status, output, {k: v['status'] for k,v in stages.items()})
        from scripts.research.a_axis_flip_publish import publish
        publish(output, push=True)
        print(f'FINAL SUMMARY: {output/"SUMMARY.md"}', flush=True)
    except BaseException as error:
        status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        write_json(output/'failure.json', {'status': status, 'reason': repr(error), 'completed_stages': list(stages)})
        write_json(output/'workflow_status.json', {'status': status, 'reason': repr(error)})
        summary(output, stages, status)
        record('workflow', status, output, f'{error!r}; completed={list(stages)}; partial evidence retained; no automatic retry')
        raise


def run(output):
    # Prevent two manual resume invocations from mutating the same evidence.
    import fcntl
    with (output/'workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This plan is already running; duplicate execution refused') from error
        return _run(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'launch', 'run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not str(output).startswith(str(ROOT/'runs/20261123')+'/'):
        raise ValueError('Use a purpose-specific runs/20261123 directory')
    output.mkdir(parents=True, exist_ok=True)
    if args.action == 'run':
        return run(output)
    try:
        preflight(output)
    except BaseException as error:
        write_json(output/'preflight_failure.json', {'reason': repr(error)})
        record('preflight', 'failed', output, repr(error))
        raise
    if args.action == 'preflight':
        print(f'PREFLIGHT PASSED: {output/"preflight.json"}', flush=True)
        return
    launch_receipt = output/'launch_receipt.json'
    if launch_receipt.exists() and read(launch_receipt).get('accepted'):
        raise FileExistsError('This plan has already been launched; no duplicate terminal/run')
    log = output/'terminal.log'
    env_args = ['env', '-u', 'PYTHONHOME', '-u', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES=0',
                'MUJOCO_GL=egl', 'MUJOCO_EGL_DEVICE_ID=0', 'MPLBACKEND=Agg',
                'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    command = env_args + [str(ROOT/'.venv/bin/python'), '-E', '-B', '-u', '-m', MODULE,
                          'run', '--output-dir', str(output)]
    shell = ('set -o pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
             +shlex.join(command)+' 2>&1 | tee -a '+shlex.quote(str(log))
             +'\ncode=${PIPESTATUS[0]}\nprintf "\\nExperiment exited: %s\\n" "$code"\n'
             +'read -r -p "Press Enter to close this terminal..."\nexit "$code"\n')
    # No --wait and no subprocess timeout. Do not fall back to nohup/background.
    result = subprocess.run(['gnome-terminal', '--title=A-axis audit and repair 20261123',
                             '--working-directory='+str(ROOT), '--', 'bash', '-c', shell],
                            capture_output=True, text=True)
    receipt = {'accepted': result.returncode == 0, 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr, 'command': command,
               'log': str(log), 'timestamp': datetime.now().astimezone().isoformat(),
               'no_wall_clock_limits': True}
    write_json(launch_receipt, receipt)
    record('launch', 'accepted' if receipt['accepted'] else 'failed', output, receipt)
    if not receipt['accepted']:
        raise RuntimeError(f'GNOME Terminal failed, no background fallback: {result.stderr}')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
