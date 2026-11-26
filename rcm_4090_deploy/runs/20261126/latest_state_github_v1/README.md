# 最新状态快照（2026-11-26）

截点：2026-11-26T10:15:08+08:00。基于提交 `b362ea79e00a24bbc1ec68199e78ac4656e59463`。

归档当前部署源码、规划 1–8、六组新增/更新实验的卡片、脚本、指标、失败记录、轻量日志及完整 `TEST_LOG.md` 截点副本。历史提交中的代码和实验结果保留原位；`source_snapshot/` 保存当前实际部署源码。机器路径原样保留，换机器复现需映射数据、权重和场景路径。二进制场景资产及 pinned vendor 驱动沿用仓库历史版本；本次不修改。

## 最新原神经网络试验

`original_neural_terminal_pilot_v1` 已完成 128 次优化更新及训练后离线评估，但门槛未通过，未进入闭环：

- 初始化：v5 phase1_A `step_004000.pth`；SHA-256 `6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11`。
- E40 两段训练、E41 整目标留出；每目标 50 个 horizon16 窗口。保持原生产 PolicyEvaluator、归一化、AMP 和冻结视觉编码器；详见卡片与 step0 parity。
- 留出集 C 首动作 MAE：1.4682645 → 1.9467759 µm，变差；C 方向准确率：0.750 → 0.755；A/B MAE 均变差。
- 写结束状态时发生 `TypeError: dict() got multiple values for keyword argument 'time'`。`summary.json`、`offline_gate.json` 已保存离线门槛失败；`workflow_status.json`/`failure.json` 保留技术异常。两类事实分别保留，不能标为流程成功。
- 本次上传未修复、重跑或增加候选。

## 最近完整标准起点闭环

固定 1600 动作、50 ms 周期、seed=42+episode_index；v5 结合 learned-local 模型，teacher/geometry 不参与在线动作。完整输入、动作限幅、切换逻辑与基线一致性详见各卡及审计。

| 运行 | 目标 | 最终真值误差 µm | 最后200周期最大误差 µm | 最后200周期均≤20 µm | 原0.35 µm严格成功 |
|---|---|---:|---:|---|---|
| v5_learned_local_standard2_protocol_v2 | E06 | 2.903941 | 3.560359 | 是 | 否 |
| v5_learned_local_standard2_protocol_v2 | E08 | 3.263997 | 5.579105 | 是 | 否 |
| v5_learned_local_standard3_v1 | E00 | 3.032142 | 4.795080 | 是 | 否 |
| v5_learned_local_standard3_v1 | E02 | 59.135876 | 74.929990 | 否 | 否 |
| v5_learned_local_standard3_v1 | E10 | 2.710038 | 6.559135 | 是 | 否 |

两次运行共5个目标：20 µm保持条件4/5；原0.35 µm严格成功0/5。这是已完成的两次指定目标运行的汇总，不等同于 SOP screen10/dev20 验收或独立 final_test。E02 的最终误差仍为59.135876 µm。旧 `standard2_v1` 的协议一致性失败记录也完整保留，不能替代修正后的 protocol_v2 结果。

## 规划8及局部恢复

- 规划8 output-head 原始尝试因缓存精度一致性失败停止；修复后 recovery_v2 流程完成，最终报告此前已上传于 `b362ea7`。
- R2 在配对训练目标的20周期短重放中改善，但 fast5 每目标1600周期未通过门槛，严格成功0/5；不能把训练目标短重放泛化为完整闭环精度。
- learned-terminal recovery 的 E43 局部试验最后200周期≤20 µm为3/3，仅证明指定局部条件可行；其初始终端启动失败和后续恢复分别保留。

## 文件与验证

- `current_state.json`：分开记录流程状态、门槛、严格成功与错误。
- `experiments/<日期>/<实验>/`：原样卡片、脚本、报告、指纹及轻量日志。
- `source_snapshot/`：当前部署源码/配置/文档；不覆盖历史分支根目录。
- `capture_inventory.json`：来源、捕获时间、文件mtime、SHA-256、日志截点和未上传文件清单。
- `manifest.json`：归档文件大小及 SHA-256（清单自身除外）。
- `verify_snapshot.py`：离线复核文件集、哈希、JSON/JSONL解析、Python AST及禁止文件类型；不导入模型、不运行训练。
- `TEST_LOG.md`：主部署日志的固定字节截点；`publication_record.md` 单独保存本次上传前验证的追加记录。

验证命令（仓库根目录）：

```bash
python3 -E -B rcm_4090_deploy/runs/20261126/latest_state_github_v1/verify_snapshot.py --snapshot-dir rcm_4090_deploy/runs/20261126/latest_state_github_v1
```

权重、原始轨迹、数据图像、缓存和视频留在本地，现有模型及数据标识/指纹随卡片归档。全部结果仅验证软件与 MuJoCo 仿真行为，不代表临床精度或硬件就绪。
