"""Read back saved audit evidence; append interpretation, never rerun measurements."""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

START = time.monotonic()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read(name):
    return json.loads((OUT/name).read_text())


def write_new(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    state = read('task_status.json')
    state.update(status='running', stage='final_saved_report_readback')
    (OUT/'task_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'status_history.jsonl').open('a') as f:
        f.write(json.dumps(state,ensure_ascii=False)+'\n')
    rows = read('raw_state_evidence.json')
    summary = read('summary.json')
    original_summary_hash = hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()
    ambiguous = [r for r in rows if r['incoming_physical_phase']=='ambiguous']
    maximum_raw_accepted = [max(abs(r['raw_teacher_deg_deg_mm'][i]-r['label_accepted_deg_deg_mm'][i]) for r in rows) for i in range(3)]
    maximum_request_actual = [max(abs(r['label_accepted_deg_deg_mm'][i]-r['actual_execution_delta_deg_deg_mm'][i]) for r in rows) for i in range(3)]
    witness = max(rows, key=lambda r: abs(r['label_accepted_deg_deg_mm'][2]-r['actual_execution_delta_deg_deg_mm'][2]))
    episodes = {}
    for name in sorted({r['episode'] for r in rows}):
        ep = [r for r in rows if r['episode']==name]
        episodes[name] = dict(final_true_error_um=ep[-1]['post_true_distance_um'],
            last10_max_um=max(r['post_true_distance_um'] for r in ep[-10:]),
            raw_original_counts={g:sum(r['original_group']==g for r in ep) for g in ('approaching','receding','crossing','hold')},
            incoming_truth_classes_except_prefix= dict(collections.Counter('approaching' if r['incoming_truth_radial_um_s']<0 else 'receding' if r['incoming_truth_radial_um_s']>0 else 'stationary' for r in ep[1:])) )
    for log, result in read('test_log_write_verification.json')['logs'].items():
        assert result['all_stage_records_present'] and Path(log).exists()
    assert summary['recomputed_original_gate']['status']=='failed'
    assert len(rows)==160 and summary['input_tensor_mismatches']==0
    assert summary['all_input_hashes_unchanged']
    report = OUT/'report.md'
    initial_report = report.read_bytes()
    with (OUT/'report_initial.md').open('xb') as f:
        f.write(initial_report)
    result = dict(status='passed', time=datetime.datetime.now().astimezone().isoformat(),
        source_measurements_not_overwritten=True, summary_sha256=original_summary_hash,
        raw_accepted_max_abs_difference_deg_deg_mm=maximum_raw_accepted,
        accepted_minus_executed_max_abs_difference_deg_deg_mm=maximum_request_actual,
        max_C_accepted_vs_actual_example=witness, OCT_incoming_ambiguities=ambiguous,
        episode_readback=episodes,
        diagnostic_technical_error='First diagnostic attempt used unavailable private NumPy header API; corrected to public API. Original diagnostic_failure.json and audit_attempt1.py retained.',
        diagnostic_error_is_not_scientific_failure=True,
        unresolved_limits=['缺各中间时刻独立needle-site XYZ，不能独立重算所有3D范数；仅重算已存距离差及终点XYZ范数。',
            '缺最后执行后的RGB/OCT，step39的crossing不能由末帧直接确认。',
            '真实相机延迟/硬件时钟、latent采样归一化轨迹未存，不能以零差代替缺失证据。',
            'E40 above step26的OCT模长前后相同而真值有微小下降，传感观测原因未确认；保留ambiguous，不强改类。'],
        model_inference_calls=0, policy_samples=0, gradient_updates=0, new_simulation_cycles=0,
        command='.venv/bin/python -E -B runs/20261126/original_neural_terminal_data_audit_v1/finalize_readback.py',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    text = ['\n## 保存结果读回与补充说明（不重跑测量）\n',
        '完整160步原分类合计：approaching 104、receding 56、crossing 0、strict hold 0；其中56个receding全部是距离不变，真正执行后远离区间为0。输入运动另计：远离4、靠近103、停稳52、OCT方向未决1。\n',
        '未决行为 E40_above_recede step26，未入首动作窗：实际C速度 −0.000313368054561 mm/s；输入前一执行区间已存真值径向速度 −0.313170605126 µm/s，但同源OCT模长差为0。保留OCT方向ambiguous；不能把这一条硬改为远离，也不能据此认定时序错位。\n',
        f'raw与accepted三轴最大差为 {maximum_raw_accepted} deg/deg/mm。accepted请求与实际执行位移三轴最大差为 {maximum_request_actual} deg/deg/mm，系不同接口量，不作为标签错位；逐步数值已存raw_state_evidence。最大C差代表为 {witness["episode"]} step{witness["step"]}：受理C={witness["label_accepted_deg_deg_mm"][2]:.15g}mm，实际ΔC={witness["actual_execution_delta_deg_deg_mm"][2]:.15g}mm。\n',
        '|原离线验收复算|A MAE ° before→after|B MAE ° before→after|C符号准确率 before→after|\n',
        '|---|---:|---:|---:|\n']
    for split, label in [('train','E40训练'),('heldout','E41留出')]:
        b=summary['original_metrics'][split+'_before']['groups']['all']
        a=summary['original_metrics'][split+'_after']['groups']['all']
        text.append(f'|{label}|{b["physical_first_action_MAE_deg_deg_um"][0]:.10f} → {a["physical_first_action_MAE_deg_deg_um"][0]:.10f}|{b["physical_first_action_MAE_deg_deg_um"][1]:.10f} → {a["physical_first_action_MAE_deg_deg_um"][1]:.10f}|{b["C_direction_accuracy"]:.3%} → {a["C_direction_accuracy"]:.3%}|\n')
    text += ['\n原6项gate数值与真假结果逐项一致：仅C符号准确率改善，其余5项未通过。详见recomputed_gate.json；原文件未改。\n',
        '所有原片段终点及末10步最大真值误差见final_readback.json。原数据中的停稳尾部与真正零受理C的hold明确分开。\n',
        '第一次诊断因本机NumPy不提供私有_read_array_header接口而停止；只修改新诊断脚本，使用公共API继续。原失败栈diagnostic_failure.json、首次脚本audit_attempt1.py、首次日志verification.log完整保留；修正与续跑命令见diagnostic_correction_v1.json和verification_resume1.log。该技术错误与原128更新科学负结果分开。\n',
        '中文局限：中间逐拍site XYZ、最后post-action RGB/OCT、相机真实延迟与latent采样轨迹缺失；未重建仿真或模型来补证。不能宣称这些缺失项已验证零差。\n',
        '本次唯一建议仍是先补齐并验收同误差异速且history有效后仍有远离/制动/保持的连续视觉教师数据。此建议没有执行，负结果不推导为原架构不可能。\n']
    with report.open('a') as f:
        f.write(''.join(text)); f.flush(); os.fsync(f.fileno())
    result['report_sha256']=hashlib.sha256(report.read_bytes()).hexdigest()
    result['total_accounted_compute_seconds']=summary['compute_seconds']+time.monotonic()-START
    assert result['total_accounted_compute_seconds']<60
    write_new('final_readback.json',result)
    record=dict(time=result['time'],stage='check6_saved_report_readback',status='passed',
        command=result['command'],script_sha256=result['script_sha256'],
        data_source=summary['source'],
        before_sha256=read('input_hashes.json')['files'][summary['source']+'/before.pth']['sha256'],
        after_sha256=read('input_hashes.json')['files'][summary['source']+'/after.pth']['sha256'],
        model_hash_reference=str(OUT/'input_hashes.json'),result_path=str(OUT/'final_readback.json'),
        summary_sha256=original_summary_hash,raw_accepted_max_abs_difference=maximum_raw_accepted,
        limitations=result['unresolved_limits'],technical_failure_retained_and_corrected=True,
        total_accounted_compute_seconds=result['total_accounted_compute_seconds'],
        model_inference_calls=0,gradient_updates=0,new_simulation_cycles=0)
    encoded='\n- '+json.dumps(record,ensure_ascii=False)+'\n'
    logs={}
    for path in (OUT/'TEST_LOG.md',ROOT/'TEST_LOG.md'):
        with path.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX); f.write(encoded); f.flush(); os.fsync(f.fileno())
        assert encoded in path.read_text()
        logs[str(path)]=dict(new_record_read_back=True,bytes=path.stat().st_size)
    assert hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()==original_summary_hash
    write_new('final_log_readback.json',logs)
    state.update(status='completed',stage='finished_after_saved_report_readback',
        time=datetime.datetime.now().astimezone().isoformat(),
        final_readback_path=str(OUT/'final_readback.json'),TEST_LOG_append_verified=True,
        compute_elapsed_seconds=result['total_accounted_compute_seconds'])
    (OUT/'task_status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'status_history.jsonl').open('a') as f:
        f.write(json.dumps(state,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='completed',compute_seconds=result['total_accounted_compute_seconds'],
        accepted_vs_actual_max=maximum_request_actual,episode_readback=episodes,both_TEST_LOG_appends_read_back=True),ensure_ascii=False))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        state=read('task_status.json')
        state.update(status='failed',stage='final_readback_failed',failure_reason=repr(error))
        (OUT/'task_status.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        raise
