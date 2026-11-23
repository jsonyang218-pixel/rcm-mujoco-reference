"""Verify and freeze the authorized clean-OCT code and result publication."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.research.clean_oct_pipeline import identity
from scripts.research.clean_oct_sampling_ablation import baseline_inputs, compare_contract, episode_metrics, paired_metrics
from scripts.research.gpu_episode_pool import artifact_hashes

ROOT = Path.cwd()
PUB = ROOT / 'runs/20261123/clean_oct_publication_v1'
REPO = PUB / 'staging_repo'
SNAP = PUB / 'snapshot'
REPORT = 'rcm_4090_deploy/runs/20261123/clean_oct_publication_v1/report'
BASE = json.loads((PUB / 'base.json').read_text())['base']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def save(relative, data):
    path = SNAP / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_json(relative, value):
    save(relative, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


SNAP.mkdir(exist_ok=False)
summary, group, rows, targets, baselines, contracts = baseline_inputs()
result = load('runs/20261122/diagnosis/clean_oct_terminal_ablation_v1/summary.json')
assert result['status'] == 'completed' and result['completed_targets'] == 3
assert result['candidate_passed'] == 1 and result['strict_success_count'] == 0
actual = []
for row in result['candidate']:
    path = Path(row['result_path'])
    data = path.read_bytes()
    assert sha(data) == row['result_sha256'], path
    payload = json.loads(data)
    compare_contract(payload['evaluation_contract'], contracts[row['episode_index']])
    metrics = episode_metrics(payload['episodes'][0])
    assert all(metrics[k] == row[k] for k in metrics), path
    actual.append(metrics)
comparison = paired_metrics(actual, [episode_metrics(baselines[i]) for i in (0, 8, 10)])
assert comparison == result['comparison']
for relative in ('runs/20261122/diagnosis/clean_oct_drift_workflow_v1',
                 'runs/20261122/diagnosis/clean_oct_median9_workflow_v1',
                 'runs/20261121/clean_oct_dev20_authorized_v1'):
    assert load(relative + '/workflow_status.json')['status'] == 'completed'

# Validate all last-round execution assets against the launch fingerprints.
pinned = {}
for relative in ('runs/20261122/diagnosis/clean_oct_drift_workflow_v1/preflight.json',
                 'runs/20261122/diagnosis/clean_oct_median9_workflow_v1/preflight.json'):
    for stage in load(relative)['stages']:
        for path, expected in stage['assets'].items():
            assert sha(Path(path).read_bytes()) == expected, path
            pinned[path] = expected

evidence_roots = [
    'runs/20261121/clean_oct_cbias_gpu_v1',
    'runs/20261121/clean_oct_cbias_gpu_resume_v2',
    'runs/20261121/clean_oct_dev20_authorized_v1',
    'runs/20261122/diagnosis/clean_oct_drift_workflow_v1',
    'runs/20261122/diagnosis/clean_oct_fixed_input_v1',
    'runs/20261122/diagnosis/clean_oct_median9_workflow_v1',
    'runs/20261122/diagnosis/clean_oct_terminal_ablation_v1',
    'runs/20261122/analysis/clean_oct_command_execution_v1',
    'runs/20261122/analysis/clean_oct_dev20_late_drift_v1',
    'runs/20261122/analysis/clean_oct_drift_interpretation_v1',
]
evidence_roots += [str(p.relative_to(ROOT)) for p in (ROOT / 'runs/20261122/verification').glob('clean_oct_*') if p.is_dir()]
evidence_roots += [str(p.relative_to(ROOT)) for p in (ROOT / 'runs/20261121/verification').glob('clean_oct_*') if p.is_dir()]
receipts, receipt_files = 0, 0
for relative in evidence_roots:
    for path in (ROOT / relative).rglob('receipt.json'):
        receipt = json.loads(path.read_text())
        if 'files' in receipt and receipt.get('status') == 'completed':
            assert artifact_hashes(path.parent) == receipt['files'], path
            receipts += 1
            receipt_files += len(receipt['files'])

selected = set()
for relative in evidence_roots:
    selected.update(p for p in (ROOT / relative).rglob('*') if p.is_file())
for folder in ('scripts/research', 'scripts/verification', 'configs', 'docs'):
    selected.update(p for p in (ROOT / folder).iterdir()
                    if p.is_file() and 'clean_oct' in p.name.lower())
selected.update(Path(p) for p in pinned if Path(p).suffix not in ('.pth', '.pt', '.ckpt'))
selected.update(p for p in (ROOT / 'outputs/clean_oct_cbias_20261121_resume_v2').iterdir()
                if p.is_file() and p.suffix not in ('.pth', '.pt', '.ckpt'))
selected.update(p for p in (ROOT / 'data/clean_oct_cbias_20261121_resume_v2').rglob('*') if p.is_file())
for relative in ('scripts/research/resume_clean_oct_overlay.py', 'docs/EXPERIMENT_SOP.md',
                 'docs/OUTPUT_LAYOUT.md', 'scripts/tools/experiment_workflow.py', 'config.json'):
    selected.add(ROOT / relative)
selected.add(PUB / 'prepare.py')
skipped = []
for path in sorted(selected):
    if path.is_symlink() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.pth', '.pt', '.ckpt') or path.name == 'worker.claim':
        continue
    data = path.read_bytes()
    assert len(data) < 95_000_000, path
    if path.suffix == '.json':
        json.loads(data)
    elif path.suffix == '.jsonl':
        for line in data.splitlines():
            if line.strip():
                json.loads(line)
    save('rcm_4090_deploy/' + path.relative_to(ROOT).as_posix(), data)

checkpoint = Path(summary['checkpoint'])
acceptance = {'status': 'verified', 'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
              'command': 'python -B -m runs.20261123.clean_oct_publication_v1.prepare',
              'base_commit': BASE, 'pinned_assets_verified': len(pinned),
              'completed_task_receipts_verified': receipts, 'receipt_files_verified': receipt_files,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint.read_bytes()),
              'model_binary_uploaded': False, 'model_binary_reason': '414734287-byte model exceeds regular GitHub single-file limit; identity and training records included, following previous result publications.',
              'comparison': comparison, 'candidate_strict_success': '0/3', 'baseline_strict_success': '0/3',
              'full_dev20_final_mean_um': summary['final_mean_um'],
              'source_identity': group['spec']['identity'],
              'scope': 'Publication only, no new experiments; preserve failed/interrupted earlier workflow states.'}
save_json(REPORT + '/results_acceptance.json', acceptance)
save_json(REPORT + '/pinned_assets.json', pinned)
means = comparison['means']
improvement = (1 - means['candidate']['final_error_um'] / means['baseline']['final_error_um']) * 100
readme = f'''# Clean OCT：诊断与9采样闭环结果

本次上传包含最近的代码、实验卡、训练记录、完整dev20、A/B诊断、单因素3目标闭环、原始轨迹及TEST_LOG。
保留原始文件内容与SHA256；绝对路径记录对应原机器，仓库路径去掉 `/home/him1/mujoco_bc/` 前缀。

## 最近的单因素闭环

流程3/3完成，约11分25秒；Flow4、execute1、G10 feedback、1600步、50ms保持原协议。
唯一候选改动：每步1次采样 → 9次并行采样，沿用逐分量torch.median。
E00/E08/E10，种子42/50/52，step_002000同一权重；完整同身份基线复用。

| 指标 | 基线 | 9采样候选 |
|---|---:|---:|
| 平均终点误差 µm | {means['baseline']['final_error_um']:.6f} | {means['candidate']['final_error_um']:.6f} |
| 末200步平均误差 µm | {means['baseline']['last200_mean_error_um']:.6f} | {means['candidate']['last200_mean_error_um']:.6f} |
| 末段漂移绝对值均值 µm | {means['baseline']['absolute_drift_1000_to_exit_um']:.6f} | {means['candidate']['absolute_drift_1000_to_exit_um']:.6f} |
| 严格成功 | 0/3 | 0/3 |

终点均值下降{improvement:.2f}%，三个平均指标门禁通过。E08终点增加17.010574µm，不能忽略单目标恶化。
这只是3目标smoke筛选；未执行该候选screen10/dev20，不外推总体成功率。真值不用于动作选择。
严格成功维持≤0.35µm及连续3帧编码器停稳。采样数改变随机数消耗，不声称相同噪声流。

## 前置记录与限制

完整单采样dev20均值104.285155µm、严格成功0/20；与本次3目标均值分母不同，不能直接混比。
A阶段20目标40段完成；B阶段9快照288采样完成，原种子动作9/9精确复现。
7/9快照平均A方向与教师参考相反；一阶朝向测量目标109/288。原RGB未保存，不声称像素完全一致。
这些是诊断线索，未证明驱动bug，也未证明增加采样足以消除方向偏差。
早期fast5分支失败和后续dev20用户例外授权均保留；早期中断工作流不伪装为成功。

## 文件入口

- `rcm_4090_deploy/docs/CLEAN_OCT_MEDIAN9_SMOKE_20261122.md`：冻结协议和门禁。
- `rcm_4090_deploy/runs/20261122/diagnosis/clean_oct_terminal_ablation_v1/REPORT.md`：逐目标对照。
- `rcm_4090_deploy/runs/20261122/analysis/clean_oct_drift_interpretation_v1/REPORT.md`：诊断解读。
- `rcm_4090_deploy/runs/20261121/clean_oct_dev20_authorized_v1/dev20/summary.json`：完整20目标基线。
- `rcm_4090_deploy/outputs/clean_oct_cbias_20261121_resume_v2/`：训练配置、日志、历史和清单。
- `rcm_4090_deploy/TEST_LOG.md`：保留远端历史并追加本地新记录；`local_TEST_LOG.md`保存本地原件。
- 本目录 `publication_manifest.json`：全部发布文件（自身除外）的字节数及SHA256。

权重 `step_002000.pth` SHA256：`{acceptance['checkpoint_sha256']}`。
模型文件约415MB，超过GitHub普通单文件限制，沿用之前发布方式记录身份，不上传模型二进制。
原训练/验证数据集不重复上传；本轮clean OCT/C标签覆盖层和审计结果包含在内。
发布准备属于CPU文件I/O；原实验使用物理GPU0 RTX4090 CUDA/EGL。所有结论仅限软件/MuJoCo仿真。
'''
save(REPORT + '/README.md', readme.encode())
entry = f"\n- {acceptance['timestamp']}: clean OCT GitHub publication acceptance **PASS**; command `runs/20261123/clean_oct_publication_v1/prepare.py`; checkpoint step2000 SHA256={acceptance['checkpoint_sha256']}; original train/val frozen, no new experiment; E00/E08/E10 seeds42/50/52 Flow4/G10/1600x50ms; median9 mean final127.641903->113.314311um (-{improvement:.2f}%), last200 111.339233->107.350691um, absolute drift74.964477->27.775900um; E08 worsened17.010574um; candidate gatePASS, strict0/3; assets{len(pinned)}, receipts{receipts}, files{receipt_files} verified; output runs/20261123/clean_oct_publication_v1/report/results_acceptance.json; CPU I/O archive, CUDA/EGL original experiments. Software/MuJoCo only.\n"
with (ROOT / 'TEST_LOG.md').open('ab') as stream:
    stream.write(entry.encode())
local = (ROOT / 'TEST_LOG.md').read_bytes()
remote = subprocess.check_output(['git', '-C', str(REPO), 'show', BASE + ':rcm_4090_deploy/TEST_LOG.md'])
if local.startswith(remote):
    merged = local
else:
    common = 0
    for a, b in zip(local, remote):
        if a != b:
            break
        common += 1
    boundary = local.rfind(b'\n', 0, common) + 1
    merged = remote + b'\n\n## Local continuation records (append-only publication)\n\n' + local[boundary:]
assert merged.startswith(remote)
save('rcm_4090_deploy/TEST_LOG.md', merged)
save(REPORT + '/local_TEST_LOG.md', local)
files = [{'path': p.relative_to(SNAP).as_posix(), 'bytes': p.stat().st_size,
          'sha256': sha(p.read_bytes())} for p in sorted(SNAP.rglob('*')) if p.is_file()]
manifest = {'base_commit': BASE, 'files': files, 'total_bytes': sum(r['bytes'] for r in files),
            'remote_test_log_prefix_preserved': True, 'status': 'prepared_verified'}
save_json(REPORT + '/publication_manifest.json', manifest)
(PUB / 'preparation.json').write_text(json.dumps({'files': len(files) + 1, 'bytes': manifest['total_bytes'],
    'acceptance': acceptance, 'report': REPORT + '/README.md'}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'prepared_verified', 'files': len(files) + 1, 'bytes': manifest['total_bytes'],
                  'pinned_assets': len(pinned), 'receipts': receipts}, indent=2))
