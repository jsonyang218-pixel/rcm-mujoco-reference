"""One frozen smoke comparison: existing single sample versus median of nine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.clean_oct_drift_diagnosis import INDICES, ROOT, SOURCE, source_inputs
from scripts.research.clean_oct_pipeline import identity
from scripts.research.gpu_episode_pool import record, run_pool, write_json


DIAGNOSIS = ROOT / 'runs/20261122/diagnosis/clean_oct_fixed_input_v1/summary.json'
COMMAND = ROOT / 'runs/20261122/analysis/clean_oct_command_execution_v1'
QUALITY = {'offsamples': 0, 'numslices': 112, 'numstacks': 64}
SAMPLES = 9


def compare_contract(candidate, baseline):
    """Reject every protocol change except sampling count and its aggregation."""
    if (candidate.get('num_action_samples') != SAMPLES
            or candidate.get('action_aggregation') != 'median'
            or baseline.get('num_action_samples') != 1
            or baseline.get('action_aggregation') != 'single_sample'):
        raise ValueError('sampling arm differs from frozen 1 versus 9 median comparison')
    excluded = {'num_action_samples', 'action_aggregation'}
    if ({k: v for k, v in candidate.items() if k not in excluded}
            != {k: v for k, v in baseline.items() if k not in excluded}):
        raise ValueError('non-sampling evaluation protocol changed')


def episode_metrics(episode):
    errors = np.asarray(episode['trajectory']['errors'], dtype=float) * 1000
    final = float(episode['final_error_mm'] * 1000)
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(final):
        raise ValueError('empty or nonfinite error trace')
    # Preserve the evaluator's early strict-success stop. No synthetic frames.
    late_drift = final - float(errors[1000]) if len(errors) > 1000 else 0.
    if len(errors) <= 1000 and not episode['success']:
        raise ValueError('incomplete rollout before drift interval')
    return {'episode_index': episode['episode_index'], 'seed': episode['seed'],
            'target_id': episode['target_id'], 'steps': episode['steps'],
            'final_error_um': final, 'last200_mean_error_um': float(errors[-200:].mean()),
            'drift_1000_to_exit_um': late_drift,
            'absolute_drift_1000_to_exit_um': abs(late_drift),
            'drift_interval_reached': len(errors) > 1000,
            'early_strict_success': bool(episode['success'] and len(errors) <= 1000),
            'strict_success': bool(episode['success']),
            'minimum_error_um': float(errors.min()),
            'safety_limited_steps': episode['safety_limited_steps'],
            'metric_definition': 'last200 recorded pre-action errors; drift from trace[1000] to final post-action error, zero if strict success stops before index1000; no padded frames'}


def paired_metrics(candidate, baseline):
    if (len(candidate) != 3 or len(baseline) != 3
            or {r['episode_index'] for r in candidate} != set(INDICES)
            or {r['episode_index'] for r in baseline} != set(INDICES)):
        raise ValueError('comparison requires exactly E00/E08/E10')
    base = {r['episode_index']: r for r in baseline}
    for row in candidate:
        old = base[row['episode_index']]
        if any(row[k] != old[k] for k in ('target_id', 'seed')):
            raise ValueError('paired target or seed mismatch')
    keys = ('final_error_um', 'last200_mean_error_um', 'absolute_drift_1000_to_exit_um')
    means = {arm: {k: float(np.mean([r[k] for r in rows])) for k in keys}
             for arm, rows in (('baseline', baseline), ('candidate', candidate))}
    gates = {k: means['candidate'][k] < means['baseline'][k] for k in keys}
    return {'means': means, 'gates': gates, 'candidate_passed': int(all(gates.values())),
            'pairs': [{'episode_index': r['episode_index'], 'seed': r['seed'],
                       'delta_candidate_minus_baseline': {k: r[k] - base[r['episode_index']][k] for k in keys}}
                      for r in sorted(candidate, key=lambda x: x['episode_index'])]}


def baseline_inputs():
    os.environ['RCM_DRIVER_VARIANT'] = 'g10'
    from scripts.research.overnight_cfix_eval import validate_episode
    summary, group, rows = source_inputs()
    diagnostic = json.loads(DIAGNOSIS.read_text())
    command = json.loads((COMMAND / 'summary.json').read_text())
    if (diagnostic['status'] != 'completed' or diagnostic['snapshots'] != 9
            or diagnostic['samples'] != 288 or diagnostic['reconstruction_passed'] != 1
            or command['status'] != 'completed' or command['analyzed_targets'] != 20):
        raise ValueError('A/B diagnostic completion gate failed')
    hashes = json.loads((COMMAND / 'source_hashes.json').read_text())
    targets = json.loads(Path(group['spec']['target_manifest']).read_text())['targets']
    episodes, contracts = {}, {}
    for index in INDICES:
        row = rows[index]
        path = Path(row['result_path'])
        if file_sha256(path) != hashes[str(path)]:
            raise ValueError('baseline trace differs from completed diagnosis')
        payload = json.loads(path.read_text())
        episode = payload['episodes'][0]
        validate_episode(episode, targets[index], index, require_latency=True, max_steps=1600)
        if episode['motor_summary']['driver']['variant'] != 'g10':
            raise ValueError('baseline driver differs from G10')
        if not np.isclose(episode['final_error_mm'] * 1000, row['final_error_um'], rtol=0, atol=1e-9):
            raise ValueError('baseline aggregate differs from episode')
        episodes[index], contracts[index] = episode, payload['evaluation_contract']
    if any(v != contracts[INDICES[0]] for v in contracts.values()):
        raise ValueError('baseline episode contracts differ')
    return summary, group, rows, targets, episodes, contracts


class SamplingWorker:
    def __init__(self, spec):
        os.environ['RCM_DRIVER_VARIANT'] = 'g10'
        from scripts.research.control_g15_diagnostics import gpu_check
        gpu_check()
        import torch
        from lerobot_training.eval_policy import PolicyEvaluator
        torch.set_num_threads(1)
        self.spec = spec
        if (identity() != spec['identity'] or file_sha256(spec['checkpoint']) != spec['checkpoint_sha256']
                or file_sha256(__file__) != spec['ablation_module_sha256']):
            raise ValueError('checkpoint/code/scene differs before worker initialization')
        self.evaluator = PolicyEvaluator(spec['checkpoint'], device='cuda', num_action_samples=SAMPLES,
            n_action_execute_steps=1, flow_integration_steps=4, controller_variant='feedback',
            parallel_action_samples=True, measure_inference=True, rgbd_render_quality=QUALITY.copy())
        if not self.evaluator.direct_224_preprocessing or self.evaluator.n_obs_steps != 2:
            raise ValueError('original preprocessing/history contract differs')
        self.targets = json.loads(Path(spec['target_manifest']).read_text())['targets']

    def run(self, task, output):
        from scripts.research.overnight_cfix_eval import validate_episode
        index, evaluator = task['episode_index'], self.evaluator
        baseline_path = Path(task['baseline_path'])
        if file_sha256(baseline_path) != task['baseline_sha256']:
            raise ValueError('baseline changed before candidate rollout')
        baseline = json.loads(baseline_path.read_text())
        folder = output / f'E{index:02d}'
        folder.mkdir(exist_ok=False)
        with (folder / 'motor_steps.jsonl').open('x') as stream:
            def observe(row):
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
            evaluator.step_observer = observe
            try:
                payload = evaluator.evaluate(num_episodes=1, output_dir=folder, seed=42,
                    max_steps=1600, error_threshold_mm=.00035, control_period_s=.05,
                    target_episodes=self.targets, target_start=index, target_end=index + 1,
                    progress_every=100)
            finally:
                evaluator.step_observer = None
        compare_contract(payload['evaluation_contract'], baseline['evaluation_contract'])
        episode = payload['episodes'][0]
        validate_episode(episode, self.targets[index], index, require_latency=True, max_steps=1600)
        if episode['motor_summary']['driver']['variant'] != 'g10':
            raise ValueError('candidate driver differs from G10')
        if (identity() != self.spec['identity']
                or file_sha256(self.spec['checkpoint']) != self.spec['checkpoint_sha256']
                or file_sha256(__file__) != self.spec['ablation_module_sha256']
                or file_sha256(baseline_path) != task['baseline_sha256']):
            raise ValueError('checkpoint/code/scene/baseline changed during candidate rollout')
        row = episode_metrics(episode)
        row.update(result_path=str(folder / 'evaluation_results.json'),
                   result_sha256=file_sha256(folder / 'evaluation_results.json'))
        write_json(output / 'episode_metrics.json', row)
        return row

    def close(self):
        pass  # evaluate() closes each episode's sensor and scene in finally.


def run(args):
    summary, group, rows, _, episodes, _ = baseline_inputs()
    baseline = [episode_metrics(episodes[i]) for i in INDICES]
    write_json(args.artifact_dir / 'baseline_metrics.json', baseline)
    spec = {**group['spec'], 'num_action_samples': SAMPLES, 'action_aggregation': 'torch.median',
            'ablation_module_sha256': file_sha256(__file__)}
    tasks = [{'id': f'E{i:02d}', 'episode_index': i, 'baseline_path': rows[i]['result_path'],
              'baseline_sha256': file_sha256(rows[i]['result_path'])} for i in INDICES]
    results = run_pool({'median9': {'factory': 'scripts.research.clean_oct_sampling_ablation:SamplingWorker',
                                   'spec': spec, 'tasks': tasks}},
                       args.artifact_dir / 'shards', workers=4, max_workers=4,
                       timeout_s=900, stage_timeout_s=2100)
    candidate = list(results.values())
    comparison = paired_metrics(candidate, baseline)
    report = {'status': 'completed', 'completed_targets': len(candidate), 'new_episodes': len(candidate),
              'baseline_reused_targets': len(baseline), 'checkpoint': summary['checkpoint'],
              'checkpoint_sha256': spec['checkpoint_sha256'], 'baseline': baseline, 'candidate': candidate,
              'comparison': comparison, 'candidate_passed': comparison['candidate_passed'],
              'strict_success_count': sum(r['strict_success'] for r in candidate),
              'baseline_strict_success_count': sum(r['strict_success'] for r in baseline),
              'sampling_protocol_change': 'single sample -> componentwise median of 9 parallel samples; same episode seeds, different RNG draw count/shape, no identical noise stream claim',
              'scope': '3-target smoke ablation only. No training, teacher injection, screen10/dev20 or deployment.'}
    write_json(args.artifact_dir / 'comparison.json', report)
    lines = ['# 单因素采样闭环对照', '',
             '仅改变每步采样：1 次 → 9 次，沿用现有逐分量中位数聚合。', '',
             '| 目标 | 基线终点 µm | 候选终点 µm | 末200步均值差 µm | 末段漂移幅度差 µm |',
             '|---|---:|---:|---:|---:|']
    for row in comparison['pairs']:
        idx = row['episode_index']
        b = next(r for r in baseline if r['episode_index'] == idx)
        c = next(r for r in candidate if r['episode_index'] == idx)
        d = row['delta_candidate_minus_baseline']
        lines.append(f"| E{idx:02d} | {b['final_error_um']:.6f} | {c['final_error_um']:.6f} | {d['last200_mean_error_um']:.6f} | {d['absolute_drift_1000_to_exit_um']:.6f} |")
    lines += ['', f"候选门禁通过：{bool(report['candidate_passed'])}；严格成功 {report['strict_success_count']}/3。",
              '差值为候选减基线；流程完成与指标门禁分开记录。3目标仅用于筛选。',
              '同目标同种子；采样数改变随机数消耗，不声称噪声流相同。',
              '若提前严格成功，末200步取实际记录，1000步前成功的末段漂移记0，不填充仿真帧。',
              '结果限软件/MuJoCo仿真，不代表临床精度或硬件就绪。']
    (args.artifact_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir, args.artifact_dir = args.output_dir.resolve(), args.artifact_dir.resolve()
    created, started = [], time.monotonic()
    try:
        for path in (args.output_dir, args.artifact_dir):
            path.mkdir(parents=True, exist_ok=False)
            created.append(path)
        report = run(args)
        report['elapsed_s'] = time.monotonic() - started
        for path in created:
            write_json(path / 'summary.json', report)
            write_json(path / 'status.json', {'status': 'completed'})
        record('median9_smoke', 'completed', args.artifact_dir,
               f"checkpoint={report['checkpoint']}; baseline=step2000 V0/G10 dev20; E00/E08/E10,seeds42/50/52,max1600x50ms; "
               f"means={report['comparison']['means']}; candidate_passed={report['candidate_passed']}; "
               f"strict={report['strict_success_count']}/3; dataset=plan2_plus_original_train_224_20261119 unchanged,no training")
        print(f"Sampling comparison result: {args.artifact_dir / 'summary.json'}", flush=True)
    except BaseException as error:
        report = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                  'reason': repr(error), 'elapsed_s': time.monotonic() - started,
                  'partial_scope_path': str(args.artifact_dir / 'shards/partial_summary.json')}
        for path in created:
            write_json(path / 'status.json', report)
        record('median9_smoke', report['status'], args.artifact_dir,
               f"checkpoint=step2000 clean_oct_cbias_resume_v2; E00/E08/E10 seeds42/50/52; no training; {report}")
        raise


if __name__ == '__main__':
    main()
