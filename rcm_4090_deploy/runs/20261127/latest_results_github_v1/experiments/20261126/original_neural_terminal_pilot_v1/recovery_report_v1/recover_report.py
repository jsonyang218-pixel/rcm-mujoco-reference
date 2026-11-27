"""Read-only scientific re-audit; write reports only in this recovery directory."""
from __future__ import annotations

import ast
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
STATUS_CN = '训练与离线验证完成、动作晋级未通过，闭环跳过'


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    path = OUT/name
    assert path.parent == OUT, 'Recovery output must remain local'
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def log(stage, value):
    line = '\n- '+json.dumps(dict(time=now(), experiment=BASE.name, recovery=OUT.name,
        stage=stage, result=value, command=sys.argv, output=str(OUT)), ensure_ascii=False, allow_nan=False)+'\n'
    for p in (OUT/'TEST_LOG.md', ROOT/'TEST_LOG.md'):
        with p.open('ab') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line.encode())
    print(line.strip(), flush=True)


def same(a, b, label):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a)==set(b), label+' keys'
        for k in a:
            same(a[k], b[k], label+'.'+k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a)==len(b), label+' length'
        for i,(x,y) in enumerate(zip(a,b)):
            same(x,y,f'{label}[{i}]')
    else:
        assert a==b, f'{label}: {a!r} != {b!r}'


def verify_originals(name):
    baseline = read(OUT/'original_hashes_before.json')
    current = {str(p.relative_to(BASE)):dict(sha256=sha(p),size=p.stat().st_size)
        for p in sorted(BASE.rglob('*')) if p.is_file() and OUT not in p.parents}
    same(baseline['files'],current,'original_files')
    result = dict(time=now(),status='passed',files_verified=len(current),all_original_bytes_unchanged=True,
        files=current,nonregular_entries=baseline['nonregular_entries'])
    if name:
        save(name,result)
    return result


def audit_groups(split, index, windows, card):
    entries = windows[split]
    b = torch.load(BASE/'real_input_windows.pt',map_location='cpu',weights_only=False)[split]
    assert len(entries)==len(b['action'])==50
    expected=[]
    rebuilt=[]
    source_receipts=[]
    for kind in ('below_recede','above_recede'):
        path=BASE/'data'/f'E{index}_{kind}'/'observations.npz'
        with np.load(path,allow_pickle=False) as a:
            data={k:a[k].copy() for k in a.files if k!='rgb'}
            rgb_shape=list(a['rgb'].shape)
        tr=read(path.parent/'teacher_trajectory.json')['trajectory']
        np.testing.assert_array_equal(data['accepted_teacher_action'],np.asarray(tr['actions']))
        assert data['accepted_teacher_action'].shape==(40,3) and rgb_shape[0]==42
        np.testing.assert_allclose(np.diff(data['timestamp_s']),.05,rtol=0,atol=1e-8)
        source_receipts.append(dict(path=str(path),sha256=sha(path),rgb_shape=rgb_shape,accepted_teacher_actions=40))
        selected=[r for r in entries if r['kind']==kind]
        assert [r['start'] for r in selected]==list(range(25))
        for original in selected:
            start=original['start']
            j=start+2
            pre=float(tr['errors'][start]*1000)
            post=float(data['post_true_error_um'][start])
            c=float(data['accepted_teacher_action'][start,2])
            velocity=float(data['actual_encoder_velocity_si'][start,2]*1000)
            crossed=bool(data['oct'][j,2]*data['oct'][j+1,2]<0)
            group=('hold' if abs(c)<=1e-9 and abs(velocity)<=1e-6 else 'crossing' if crossed
                else 'approaching' if post<pre else 'receding')
            row=dict(original,group=group,moving_encoder=abs(velocity)>1e-6,nonzero_C=abs(c)>1e-9,
                teacher_C_mm=c,actual_C_velocity_mm_s=velocity,pre_true_error_um=pre,post_true_error_um=post)
            same(original,row,'reconstructed_window')
            assert original['source_sha256']==sha(path) and Path(original['source'])==path
            assert original['history_frames']==[j-1,j] and original['action_frames_inclusive']==[start,start+15]
            assert original['target']==f'E{index}'
            full_label=data['accepted_teacher_action'][start:start+16].astype(np.float32)
            expected.append(full_label)
            rebuilt.append(row)
    expected=np.asarray(expected)
    np.testing.assert_array_equal(b['action'].numpy(),expected)
    assert b['observation.image'].shape[:3]==(50,2,3)
    teacher=expected[:,0]
    result={}
    for model in ('before','after'):
        path=BASE/'offline'/model/f'{"train_E40" if split=="train" else "heldout_E41"}_predictions.npz'
        with np.load(path,allow_pickle=False) as a:
            arrays={k:a[k].copy() for k in a.files}
        raw=arrays['raw']
        assert raw.shape==(50,4,3) and np.isfinite(raw).all()
        np.testing.assert_array_equal(arrays['teacher'],teacher)
        np.testing.assert_array_equal(arrays['seeds'],card['sampling']['sample_seeds'])
        np.testing.assert_array_equal(arrays['flow_seeds'],card['sampling']['flow_loss_probe_seeds'])
        clipped=np.clip(raw,-np.asarray(card['model_caps_deg_deg_mm']),np.asarray(card['model_caps_deg_deg_mm']))
        np.testing.assert_array_equal(clipped,arrays['production_clipped'])
        masks=dict(all=np.ones(len(rebuilt),bool),moving_encoder=np.asarray([r['moving_encoder'] for r in rebuilt]),
            nonzero_C=np.asarray([r['nonzero_C'] for r in rebuilt]))
        masks.update({g:np.asarray([r['group']==g for r in rebuilt]) for g in ('approaching','receding','crossing','hold')})
        masks.update({kind:np.asarray([r['kind']==kind for r in rebuilt]) for kind in ('below_recede','above_recede')})
        masks['teacher_C_positive']=teacher[:,2]>1e-9
        masks['teacher_C_negative']=teacher[:,2]<-1e-9
        groups={}
        for name,mask in masks.items():
            signs=mask & masks['nonzero_C']
            groups[name]=dict(windows=int(mask.sum()),nonzero_C_windows=int(signs.sum()),
                raw_first_action_mae_deg_deg_um=(np.abs(raw[mask]-teacher[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
                production_clipped_first_action_mae_deg_deg_um=(np.abs(clipped[mask]-teacher[mask,None]).mean((0,1))*[1,1,1000]).tolist() if mask.any() else None,
                C_direction_accuracy=float((np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).mean()) if signs.any() else None,
                C_direction_correct=int((np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).sum()),
                C_direction_draws=int(signs.sum()*4),
                per_seed_production_mae_deg_deg_um=(np.abs(clipped[mask]-teacher[mask,None]).mean(0)*[1,1,1000]).tolist() if mask.any() else None,
                per_seed_C_direction_accuracy=(np.sign(clipped[signs,:,2])==np.sign(teacher[signs,None,2])).mean(0).tolist() if signs.any() else None)
        old=read(path.with_name(path.name.replace('_predictions.npz','_metrics.json')))
        same(old['groups'],groups,'all_original_group_metrics')
        flow=float(np.mean(arrays['flow_loss']))
        same(old['flow_loss_4_fixed_probes'],flow,'flow_loss')
        result[model]=dict(status='recomputed',model=model,split=old['split'],windows=50,groups=groups,
            flow_loss_4_fixed_probes=flow,flow_loss_excluded_from_gate=True,all_original_metric_values_exact_match=True,
            raw_predictions_path=str(path),raw_predictions_sha256=sha(path),physical_labels_bit_exact_to_saved_horizon16_windows=True,
            physical_labels_bit_exact_to_accepted_teacher_trajectories=True)
        save(f'{split}_{model}_recomputed_metrics.json',result[model])
    save(f'{split}_group_alignment_audit.json',dict(status='passed',windows=50,target=f'E{index}',
        group_rules_unchanged=True,all_groups_rebuilt_from_original_measurements=True,sources=source_receipts,windows_rebuilt=rebuilt))
    del b
    return result


def recompute_gate(heldout):
    b=heldout['before']['groups']
    a=heldout['after']['groups']
    assert b['moving_encoder']['windows']>0 and b['moving_encoder']['nonzero_C_windows']>0
    checks=[]
    def add(name,x,y,strict,required):
        checks.append(dict(name=name,passed=bool(y>x if strict=='greater' else y<x if strict=='less' else y<=x),before=x,after=y,required=required))
    add('heldout_moving_C_MAE_um',b['moving_encoder']['production_clipped_first_action_mae_deg_deg_um'][2],
        a['moving_encoder']['production_clipped_first_action_mae_deg_deg_um'][2],'less','after < before')
    add('heldout_moving_nonzero_C_direction_accuracy',b['moving_encoder']['C_direction_accuracy'],a['moving_encoder']['C_direction_accuracy'],
        'greater','after > before')
    for group in ('all','moving_encoder'):
        for i,axis in enumerate(('A','B')):
            add(f'heldout_{group}_{axis}_MAE_deg',b[group]['production_clipped_first_action_mae_deg_deg_um'][i],
                a[group]['production_clipped_first_action_mae_deg_deg_um'][i],'le','after <= before; zero tolerance')
    old=read(BASE/'offline_gate.json')
    same(checks,old['checks'],'original_gate_checks')
    status='passed' if all(c['passed'] for c in checks) else 'failed'
    assert status==old['status']=='failed'
    gate=dict(status=status,time=now(),checks=checks,unchanged_criteria=True,original_gate_exactly_reproduced=True,
        original_gate_sha256=sha(BASE/'offline_gate.json'),flow_loss_excluded=True)
    save('recomputed_offline_gate.json',gate)
    return gate


def checkpoint_audit(card):
    before=torch.load(BASE/'before.pth',map_location='cpu',weights_only=False)
    after=torch.load(BASE/'after.pth',map_location='cpu',weights_only=False)
    training=read(BASE/'training_result.json')
    p=read(BASE/'parameters.json')
    rows=[json.loads(line) for line in (BASE/'training.jsonl').read_text().splitlines()]
    updates=0
    for j,r in enumerate(rows):
        updates+=int(r['optimizer_updated'])
        assert r['attempt']==j+1 and r['step']==updates
        assert len(r['sample_indices'])==4 and all(0<=i<50 for i in r['sample_indices'])
    assert updates==128 and len(rows)==129 and sum(not r['optimizer_updated'] for r in rows)==1
    assert after['progress']==dict(global_step=128,optimizer_updates=128)
    assert training['optimizer_updates']==128 and training['attempted_batches']==129 and training['amp_skipped_attempts']==1
    assert sha(BASE/'before.pth')==card['init_sha256']==training['init_sha256']==sha(card['init_checkpoint'])
    assert sha(BASE/'after.pth')==training['after_sha256']
    states_before=before['model_state_dict']
    states_after=after['model_state_dict']
    assert states_before.keys()==states_after.keys()
    vision=[k for k in states_before if k.startswith('vision.')]
    assert vision
    assert all(torch.equal(states_before[k],states_after[k]) for k in vision)
    changed=[k for k in states_before if not torch.equal(states_before[k],states_after[k])]
    assert changed and not any(k.startswith('vision.') for k in changed)
    config_keys=[k for k in before['config'] if k.endswith('_mean') or k.endswith('_std')]
    for k in config_keys:
        same(before['config'][k],after['config'][k],'normalization.'+k)
    for k in ('schema_version','policy_type','architecture'):
        assert before[k]==after[k]
    assert before['config'].get('oct_symlog_enabled',False)==after['config'].get('oct_symlog_enabled',False)==False
    assert before['config'].get('use_local_oct_branch',False)==after['config'].get('use_local_oct_branch',False)==False
    # Instantiate the untouched production entry. No get_action, forward, sample, scene or training calls.
    from lerobot_training.eval_policy import PolicyEvaluator,DEFAULT_ACTION_LIMIT_DEG_DEG_MM
    assert list(DEFAULT_ACTION_LIMIT_DEG_DEG_MM)==card['model_caps_deg_deg_mm']
    e=PolicyEvaluator(str(BASE/'after.pth'),device='cuda',num_action_samples=1,n_action_execute_steps=1,
        flow_integration_steps=4,controller_variant='feedback')
    assert e.device.type=='cuda' and '4090' in torch.cuda.get_device_name(0)
    assert e.n_obs_steps==2 and e.n_action_steps==16 and e.num_action_samples==1 and e.flow_integration_steps==4 and e.n_action_execute_steps==1
    assert e.fine_evaluator is None and e.model.training is False and all(not x.training for x in e.model.vision.modules())
    actual={n:(list(v.shape),v.numel()) for n,v in e.model.named_parameters()}
    for item in p['parameters']:
        assert actual[item['name']]==(item['shape'],item['count'])
        assert item['trainable']==(not item['name'].startswith('vision.'))
    assert set(actual)=={r['name'] for r in p['parameters']}
    names=[n for n,_ in e.model.named_parameters() if not n.startswith('vision.')]
    frozen_names=[n for n,_ in e.model.named_parameters() if n.startswith('vision.')]
    assert after['freeze_config']['trainable_parameters']==names
    assert after['freeze_config']['frozen_parameters']==frozen_names
    assert sum(actual[n][1] for n in names)==p['trainable_count']==23164611
    assert sum(actual[n][1] for n in frozen_names)==p['frozen_count']==11373504
    optimizer=after['optimizer_state_dict']
    ids=[i for group in optimizer['param_groups'] for i in group['params']]
    assert len(ids)==len(names) and set(ids)==set(optimizer['state'])
    assert all(int(value['step'].item())==128 for value in optimizer['state'].values())
    assert all(group['lr']==3e-5 and group['weight_decay']==1e-4 for group in optimizer['param_groups'])
    bn=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in e.model.vision.modules())
    gn=sum(isinstance(m,torch.nn.GroupNorm) for m in e.model.vision.modules())
    assert bn==0 and gn==20
    for n,buffer in e.model.named_buffers():
        if n in config_keys:
            torch.testing.assert_close(buffer.cpu(),torch.tensor(before['config'][n],dtype=buffer.dtype),rtol=0,atol=0)
    source=ast.parse((BASE/'pilot.py').read_text())
    configure=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='configure_training')
    configure_source=ast.get_source_segment((BASE/'pilot.py').read_text(),configure)
    assert 'm.vision.eval()' in configure_source and "p.requires_grad_(not name.startswith('vision.'))" in configure_source
    result=dict(status='passed',time=now(),before_sha256=sha(BASE/'before.pth'),after_sha256=sha(BASE/'after.pth'),
        optimizer_updates=128,optimizer_state_steps_min=128,optimizer_state_steps_max=128,optimizer_parameter_tensors=len(ids),
        attempted_batches=129,amp_skipped_attempts=1,training_seed=card['training']['seed'],batch_size=4,learning_rate=3e-5,
        trainable_parameter_count=p['trainable_count'],frozen_parameter_count=p['frozen_count'],
        trainable_parameter_tensors=len(names),frozen_parameter_tensors=len(frozen_names),
        vision_state_tensors_bit_exact=len(vision),vision_all_weights_buffers_bit_exact=True,
        vision_batchnorm_modules=bn,vision_groupnorm_modules=gn,vision_eval_in_original_training_source=True,
        normalized_stat_keys_bit_exact=config_keys,changed_parameter_tensors=len(changed),changed_parameter_names=changed,
        production_PolicyEvaluator_load='passed, original class and checkpoint schema; no forward/sample/action calls',
        production_runtime=dict(device=str(e.device),policy_type=e.policy_type,architecture=after['architecture'],
            n_obs_steps=e.n_obs_steps,n_action_steps=e.n_action_steps,K=e.num_action_samples,flow_steps=e.flow_integration_steps,
            exec_steps=e.n_action_execute_steps,controller=e.controller_variant,vision_eval=True),
        additional_training_updates=0,additional_policy_samples=0,additional_simulation_steps=0)
    save('checkpoint_training_audit.json',result)
    return result


def make_report(train,heldout,audit,gate,integrity,status):
    lines=['# 原神经网络末端最小实验：报告恢复验收','','**'+STATUS_CN+'。**','',
        '本恢复只读取既有产物复算与验收，不重训、不更改 gate、不生成新候选、不执行动作或闭环。',
        '原实验的两周期启动短门禁已经存在；本次新增仿真周期为 0，原计划四条 240 周期闭环均未启动。','',
        '原报告故障：`state(\'finished\', **result)` 把 `summary.json` 已含的 `time` 与状态函数新增的 `time` 重复传给 `dict()`。',
        '`TypeError("dict() got multiple values for keyword argument \'time\'")` 发生在负结果 summary 保存之后。',
        '原 `workflow_status.json`、`failure.json`（含完整 TypeError 与 traceback）、`summary.json`、`pilot.py`、checkpoint、数据和旧 metrics 均原样保留。',
        '恢复状态单独写入 `recovered_workflow_status.json`。报告收尾技术错误不改变科学门禁失败结论。','',
        '## 原判据与复算方式','',
        '从四份原始 before/after 预测 NPZ 读取物理首动作，与保存的 horizon16 动作窗口及实际受理教师轨迹逐位核对标签。',
        '以原生产限幅后的首动作计算 ABC MAE（A/B 为度，C 为 µm），四个原采样种子 92001–92004，K1/flow4/exec1。',
        'moving 定义保持 `abs(实际编码器C速度 mm/s)>1e-6`；方向准确率进一步要求 `abs(受理教师C mm)>1e-9`。',
        '靠近/远离/越过/hold 采用原代码优先级和定义，从原测量与真值评分重建。空组保留 0 与空值，不补样、不改门。',
        '六项 gate 要求：E41 moving C MAE 严格下降、非零 C 方向准确率严格上升，以及 E41 全部/moving 的 A/B MAE 各自不增加（零容差）。',
        '所有训练/留出、全部分组和逐种子指标均与旧 metrics 精确相等；六项 gate 数值与布尔结果也精确相等。flow loss 单列，不参与晋级。','',
        '## 小型数值表','',
        '|目标/分组|窗数|C MAE µm（前→后）|C方向准确率（前→后）|A MAE °（前→后）|B MAE °（前→后）|',
        '|---|---:|---:|---:|---:|---:|']
    for label,data in (('E40 训练',train),('E41 留出',heldout)):
        for group in ('all','moving_encoder','approaching','receding','crossing','hold','below_recede','above_recede'):
            b=data['before']['groups'][group]
            a=data['after']['groups'][group]
            if not b['windows']:
                lines.append(f'|{label}/{group}|0|—|—|—|—|')
                continue
            x=b['production_clipped_first_action_mae_deg_deg_um']
            y=a['production_clipped_first_action_mae_deg_deg_um']
            acc=f"{b['C_direction_accuracy']:.2%} → {a['C_direction_accuracy']:.2%}" if b['C_direction_accuracy'] is not None else '—'
            lines.append(f"|{label}/{group}|{b['windows']}|{x[2]:.7f} → {y[2]:.7f}|{acc}|{x[0]:.9f} → {y[0]:.9f}|{x[1]:.9f} → {y[1]:.9f}|")
    lines+=['','|Flow loss（单列）|before|after|','|---|---:|---:|']
    for label,data in (('E40 训练',train),('E41 留出',heldout)):
        lines.append(f"|{label}|{data['before']['flow_loss_4_fixed_probes']:.9f}|{data['after']['flow_loss_4_fixed_probes']:.9f}|")
    lines+=['','## 参数、checkpoint 与文件验收','',
        f"- 实际更新 {audit['optimizer_updates']} 次；日志共 {audit['attempted_batches']} 次 AMP 尝试，其中 {audit['amp_skipped_attempts']} 次溢出跳过。优化器全部 {audit['optimizer_parameter_tensors']} 个参数状态的 step 均为 128。",
        f"- 原动作网络训练参数 {audit['trainable_parameter_count']:,}，视觉冻结参数 {audit['frozen_parameter_count']:,}；视觉 {audit['vision_state_tensors_bit_exact']} 个状态张量 before/after 逐位相同。",
        '- 原视觉编码器为 20 个 GroupNorm、0 个 BatchNorm；原训练代码显式冻结 vision 并调用 eval，全部归一化统计配置保持逐位相同。',
        '- after checkpoint 已经通过未经修改的原生产 `PolicyEvaluator` 正常加载，维持 RGB2/horizon16/K1/flow4/exec1；仅加载，无新策略采样。',
        f"- before SHA-256：`{audit['before_sha256']}`。",
        f"- after SHA-256：`{audit['after_sha256']}`。",
        f"- {integrity['files_verified']} 个原始文件全部通过恢复前后 SHA-256/字节数一致性核验；另核验原冻结资产清单。详见 `original_hashes_before.json` 与 `original_hashes_verified.json`。",'',
        '实际改动仅为新增本 `recovery_report_v1` 下恢复脚本、命令、复算指标、gate、验收哈希、中文报告、恢复状态和本地 TEST_LOG；另向部署根 TEST_LOG 追加本次恢复验收记录。原实验内既有文件零覆盖。','',
        '本单一 128 更新候选未通过动作晋级门；这不证明原 DiT 架构不具备学习能力，也不构成独立 20µm 闭环成功或失败的验收。','',
        '恢复完成时间：'+status['time']+'。']
    (OUT/'报告.md').write_text('\n'.join(lines)+'\n')


def main():
    assert not Path('/proc/328489').exists(), 'Original process still active'
    assert not (BASE/'closed_loop').exists(), 'Unexpected original closed-loop output'
    assert not (OUT/'recovered_workflow_status.json').exists(), 'Refuse duplicate recovery'
    verify_originals(None)
    card=read(BASE/'experiment_card.json')
    original_status=read(BASE/'workflow_status.json')
    error=read(BASE/'failure.json')
    summary=read(BASE/'summary.json')
    assert original_status==error and error['status']=='technical_failure' and 'multiple values' in error['error']
    assert summary['status']=='completed_offline_gate_failed' and not summary['closed_loop_started']
    fingerprints=read(BASE/'input_fingerprints.json')
    for path,expected in fingerprints['files'].items():
        assert sha(path)==expected, 'Original frozen asset differs: '+path
    from scripts.tools.experiment_workflow import scene_and_code
    assert scene_and_code()==fingerprints['runtime']
    save('original_report_error_preserved.json',dict(original_status=original_status,original_failure=error,
        workflow_status_sha256=sha(BASE/'workflow_status.json'),failure_sha256=sha(BASE/'failure.json'),
        original_summary_sha256=sha(BASE/'summary.json'),failure_stage_reclassified='report_finalization_after_scientific_gate',
        recovery_does_not_edit_original=True))
    windows=read(BASE/'window_manifest.json')
    assert windows['whole_target_isolation'] and not windows['overlap_between_splits']
    train=audit_groups('train',40,windows,card)
    heldout=audit_groups('heldout',41,windows,card)
    gate=recompute_gate(heldout)
    same(summary['offline_gate'],read(BASE/'offline_gate.json'),'summary_gate')
    audit=checkpoint_audit(card)
    integrity=verify_originals('original_hashes_verified.json')
    status=dict(time=now(),status='completed_offline_gate_failed',status_cn=STATUS_CN,recovery_status='completed',
        training='completed',offline_validation='completed',action_promotion='failed',closed_loop='skipped_gate_failed',
        original_reporting_error='preserved; isolated report-finalization failure',reporting_error_recovered=True,
        original_workflow_status_preserved=True,original_failure_preserved=True,original_scientific_summary_preserved=True,
        original_file_hashes_verified=integrity['files_verified'],original_frozen_input_files_verified=len(fingerprints['files']),
        optimizer_updates=128,attempted_batches=129,amp_skipped_attempts=1,
        additional_training_updates=0,additional_policy_samples=0,additional_simulation_cycles=0,new_candidates=0,
        offline_gate_unchanged=True,all_original_metrics_exactly_reproduced=True,production_checkpoint_load='passed',
        original_checkpoint_sha256=audit['before_sha256'],output_checkpoint_sha256=audit['after_sha256'],
        gate_path=str(OUT/'recomputed_offline_gate.json'),report_path=str(OUT/'报告.md'),
        scientific_conclusion='单一128更新候选未通过动作晋级门；原报告TypeError与该科学负结果分开；不外推架构不可能或20µm闭环结果。')
    save('recovered_workflow_status.json',status)
    make_report(train,heldout,audit,gate,integrity,status)
    log('报告恢复验收完成',dict(status_cn=STATUS_CN,optimizer_updates=128,attempts=129,
        heldout_moving_C_MAE_um_before=gate['checks'][0]['before'],heldout_moving_C_MAE_um_after=gate['checks'][0]['after'],
        original_files_hash_verified=integrity['files_verified'],original_error_preserved=True,
        new_training_updates=0,new_samples=0,new_simulation_cycles=0,recovered_status=str(OUT/'recovered_workflow_status.json')))
    verify_originals(None)
    print('RECOVERY_REPORT',str(OUT/'报告.md'),flush=True)
    print('RECOVERED_STATUS',json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    try:
        main()
    except BaseException as exc:
        problem=dict(time=now(),status='recovery_technical_failure',error=repr(exc),traceback=traceback.format_exc(),
            scientific_result_unchanged=True)
        if not (OUT/'recovery_failure.json').exists():
            save('recovery_failure.json',problem)
        log('恢复验收技术异常',problem)
        raise
