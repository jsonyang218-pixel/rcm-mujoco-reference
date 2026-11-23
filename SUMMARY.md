# 同源新渲染近目标回放 — 2026-11-23

流程状态：completed_with_conditional_skips。

基点 `71040dd33a48644da662dc92ad28053a2a326871`；独立分支 `feature/clean-render-homog-replay-20261123`。

全部阶段、训练、初始化及单episode均无wall-clock时长限制；0.35µm只用于诊断，不提前停止。

92%池仅使用原始episode0–838完整有效轨迹窗口，排除全部Plan-2窗口；8%为本轮同源新包。

## stage1

```json
{
  "status": "completed"
}
```

## stage2

```json
{
  "status": "completed"
}
```

## stage3

```json
{
  "status": "completed",
  "direction_gate_passed": false,
  "paired_final_error_improved": false,
  "best_combination": "V0_step2000",
  "dev20_gate_passed": false
}
```

## stage4

```json
{
  "status": "skipped",
  "reason": "Fast5 finalmean<35um and absolute_late_drift<15um gate failed"
}
```

配对原模型终值均值：65.569288µm；本轮最优：187.944304µm。

最小误差均值：79.779408µm；1000→1600步绝对漂移均值：32.444062µm。

严格成功：0/5。流程完成、指标改善、严格成功分别报告。

结构化报告目录：`github_reports/homog_mixed_1600_600_12_hold_v2_20261123_s1_collection`, `github_reports/homog_mixed_1600_600_12_hold_v2_20261123_s2_train`, `github_reports/homog_mixed_1600_600_12_hold_v2_20261123_s3_fast5_1600`, `github_reports/homog_mixed_1600_600_12_hold_v2_20261123_s4_dev20`。

权重、优化器断点及原始纠偏RGB保留本机；发布路径/大小/SHA256清单。代码、配置、日志及结构化证据上传GitHub。

仅验证软件与MuJoCo仿真，不代表临床精度或硬件就绪。
