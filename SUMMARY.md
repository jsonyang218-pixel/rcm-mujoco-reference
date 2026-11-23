# A-axis flip repair — 2026-11-23

流程状态：completed_with_conditional_skips。

基点 `dad76c522f7e74f3a86b56d41f6c89488ae878ff`；独立分支 `fix/a-axis-flip-20261123`。

所有阶段、训练、单集和初始化均无 wall-clock 超时。评估不以 0.35 µm 停机。

## audit

```json
{
  "status": "completed",
  "selected_branch": "A"
}
```

## phase2

```json
{
  "status": "completed"
}
```

## phase3

```json
{
  "status": "completed",
  "direction_gate_passed": false,
  "best_combination": "V0_step4000",
  "dev20_gate_passed": false
}
```

## phase4

```json
{
  "status": "skipped",
  "reason": "Phase3 finalmean<35um and late drift<15um gate failed"
}
```

Phase3 最优终值均值 71.861782 µm；最小误差均值 52.931130 µm；末段带符号漂移 -23.270349 µm，绝对漂移均值 34.106923 µm。

严格成功 0/5。流程完成、指标改善、严格成功分别报告。

报告见 `rcm_4090_deploy/github_reports/` 对应 audit/phase2/phase3/phase4 目录。

训练权重与原始纠偏 RGB 图像保留本机，发布清单登记路径、大小和 SHA256；GitHub 保留代码、配置、日志、结构化审计和逐集评估证据。

软件/MuJoCo 仿真结果，不代表临床精度或硬件就绪。
