# 针尖耦合遮挡策略实验

## 边界

本实验不重新运行 `collect_v9_dataset.py`。现有数据集保留为正常视觉伺服与动作分布参照；遮挡策略对照由 `run_occlusion_experiment.py` 生成新的评估 episode。

策略只接收正常严格 RGB-D 观测。遮挡体位置、目标投影覆盖率、遮挡开始时间和实际持续时间均由实验调度器使用仿真真值生成，仅用于场景干预和事后审计，不进入控制器特征。

## 策略

| 名称 | 行为 |
|---|---|
| `S0a_stop_instant` | 首次观测丢失立即仿真保持；报告由此产生的加速度跳变 |
| `S0b_stop_brake` | 按配置最大加速度制动到零，然后安全失败 |
| `S1_open_loop` | 保持最后一条非零可行指令继续运动，感知恢复并稳定后重新闭环 |
| `S2_target_memory` | 仅在针尖仍可见时使用有界静态目标记忆，超时、位移或质量超限则失败 |
| `S3_retract_reobserve` | 制动、C 轴回撤、等待新鲜观测；超时或超限则失败 |

旧的 `stop` 兼容为 S0a。旧的 `static_memory_reobserve` 保留为历史运动学路径；当前 strict RGB-D 与 MotorBackend 实验使用显式 S2/S3。

## 针尖耦合遮挡体

`rcm_tip_occluder.py` 在 v9 运行时 MJCF 中加入一个仅对光学通道可见的暗色球体。遮挡触发时，调度器根据当前真实相机、目标、针尖和针轴构造一个针尖局部偏移；此后每一帧都在渲染前按以下关系更新：

```text
occluder_world = tip_world + R_tip_local @ occluder_offset_tip_local
```

遮挡是否结束由真实投影覆盖率决定，而不是按固定时间关闭。`--max-active-s` 仅作为安全和实验对比的曝光上限；实际持续时间可能更短。

当前校准值：

```text
occluder radius       0.18 mm
front offset          0.05 mm
release coverage      0.10
S3 retract distance   1.00 mm
S3 retract speed      0.30 mm/s
S3 recovery timeout   5.00 s
S3 max attempts       3
```

这些参数需要在 pilot 中按实际触发率、持续时间和覆盖率继续校准。640×480 仅适合检查遮挡和策略状态；目标在该分辨率下约 2 px，不能可靠完成终点闭环。最终精度实验应使用默认 1920×1440。

## Pilot

先只生成 case 清单：

```powershell
$py = 'E:\30_RESOURCES\32_课程教程\医学\中科院\rcm_mujoco_control_limit\rcm_mujoco_reference\.venv\Scripts\python.exe'
& $py .\run_occlusion_experiment.py --dry-run
```

由于 S0 已拆成 S0a/S0b，原计划的 4 策略 72 次扩展为 **5 × 3 × 3 × 2 × 1 = 90 次**：

```powershell
& $py .\run_occlusion_experiment.py `
  --output-dir runs\occlusion_strategy_pilot `
  --strategies S0a_stop_instant S0b_stop_brake S1_open_loop S2_target_memory S3_retract_reobserve `
  --progress 0.5 0.8 0.9 `
  --coverage 0.5 1.0 `
  --max-active-s 0.5 1.0 2.0 `
  --seeds 20261102 `
  --motion-backend motor `
  --max-iterations 1600
```

Pilot 输出：

- `experiment_protocol.json`：冻结全部 case 与配对种子。
- 每个 case 的 `scenario.json`、`config_snapshot.json`、`controller.csv`、`result.json`。
- `summary.json`：按策略汇总实际触发率、遮挡持续时间、终点误差和成功率。
- `metrics.csv`：逐 case 的紧凑结果表。

检查重点：

1. 每个目标场景组合是否都实际触发目标遮挡。
2. 实际覆盖率和持续时间分布是否覆盖预期的 50%/100% 与短/长遮挡。
3. S3 是否在回撤过程中因离开遮挡构型而提前释放。
4. 在相同 seed、目标、起始位姿和遮挡设置下，各策略是否严格配对。

如果覆盖不足或 S3 无法在允许回撤内离开遮挡，只调整遮挡体半径、局部偏移方向和 S3 回撤量，不先展开全量种子矩阵。

## 验证

```powershell
& $py .\verify_occlusion_strategy.py
& $py .\verify_tip_occluder.py
& $py .\run_reference.py --no-viewer
```

`verify_occlusion_strategy.py` 使用合成传感器验证五种策略的状态转换；`verify_tip_occluder.py` 在真实 v9 RGB-D 渲染器中验证目标从可见变为不可见、遮挡体随针尖位姿运动以及回撤后恢复。

## 结果解释边界

- 这是仿真场景干预，不是实际针具遮挡、真实 OCT 或硬件制动验证。
- S0a 的加速度不连续必须单独报告，不能包装成安全制动。
- 失败 episode 必须保留；成功率分母是全部计划 case，不能只统计成功运行。
- 最终误差、任务成功、恢复时间和总任务时间应同时报告，不能只报告成功 episode 的误差均值。
