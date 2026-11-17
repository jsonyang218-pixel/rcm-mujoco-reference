# Linux 模型训练、验证与闭环评测流水线

基于 `experiment/state-attention-v1-20261117` 的接口编写。
使用现有训练环境；流水线本身只依赖 Python 标准库。

## 完整运行

在 `rcm_4090_deploy` 目录执行：

```bash
python lerobot_training/run_model_pipeline.py
```

默认读取 `configs/state_attention_pipeline.json`：

1. 将当前 `lerobot_training` 源码复制到本次运行的 `code/`。
2. 从该代码快照执行模型训练；每轮验证由原训练器完成。
3. 训练结束后，使用本次产生的 `training/best_model.pth`。
4. 从同一代码快照启动 10 个闭环评测分片，共覆盖第 0–19 个 development 目标。
5. 合并全部评测结果，保存 JSON、CSV 和 Markdown 报告。

沿用当前协议：4 轮训练，0.35 μm 成功阈值，50 ms 控制周期，
1600 步，1 次动作采样，16 步预测中执行首步，连续 3 帧编码器停稳。
40 目标清单默认只取当前实验使用的前 20 个目标。
不同分片使用相同的基础 seed，目标的实际 seed 仍为 `seed + episode_index`。

## 已有模型只做评测

```bash
python lerobot_training/run_model_pipeline.py \
  --eval-only outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth
```

这会创建新的记录目录，不重复训练或验证。配置中的模型入口必须与该权重架构匹配。

## 运行目录与日志

默认目录为 `outputs/<model_name>_<北京时间时间戳>/`，每次独立保存。
也可以指定一个尚不存在的目录：

```bash
python lerobot_training/run_model_pipeline.py \
  --output-dir outputs/state_attention_v2_pilot
```

| 文件 | 内容 |
|---|---|
| `config.json` | 本次配置 |
| `code/`、`code_version.json` | 实际执行的代码快照、源码 SHA-256、仓库提交 |
| `commands.jsonl` | 每个训练/评测进程的参数、启动时间、工作目录和日志路径 |
| `pipeline.log`、`status.json` | 流程进度、失败阶段及错误 |
| `training.log` | 训练与验证进程输出 |
| `training/training_history.json` | 每轮训练 loss、验证 loss 和耗时 |
| `training/best_model.pth`、`last_model.pth` | 原训练器保存的权重 |
| `evaluation/g*/` | 分片日志和原始逐步轨迹 |
| `evaluation/evaluation_results.json` | 合并后的完整评测结果和轨迹 |
| `episodes.csv` | 每目标成功情况、最终误差、轨迹最小误差与停稳状态 |
| `summary.json`、`REPORT.md` | 总体结果、实际权重路径与 SHA-256、模型元数据 |

### 分段结果输出

流水线在每个主要阶段完成时立即写出独立结果，不必等待最终闭环评测：

- 训练与验证完成后，生成 `training/result.json` 和
  `training/REPORT.md`，记录最佳 epoch、best/last train/val loss、训练耗时、
  checkpoint 路径和 SHA-256；同时更新 `status.json` 为
  `training_completed`。
- 每个闭环评测分片完成后，生成对应的
  `evaluation/gXX/result.json`，记录该分片覆盖的目标、成功数和耗时完成时间。
- 每个分片完成后，刷新 `evaluation/partial_summary.json` 和
  `evaluation/partial_episodes.csv`，包含当前所有已完成目标的累计指标。
- 每到一个新的完成数量，还会保存不可变快照
  `evaluation/progress/after_NN_shards/`，内含当时的 `summary.json`、
  `episodes.csv` 和 `REPORT.md`。
- 部分结果明确标记为 `status=partial`。只有全部目标完成、缺失和重复检查
  通过后，才生成根目录的最终 `summary.json`、`episodes.csv` 和
  `REPORT.md`。

### 固定基线对照

`state_attention_v1` 已冻结为后续实验的第一版本基线，配置位于
`configs/baselines/state_attention_v1/`。工作流配置通过
`baseline_manifest` 指定基线清单；当前默认指向
`configs/baselines/state_attention_v1/BASELINE.json`。

最终评测完成时，流水线会校验候选模型与基线使用完全相同的目标清单
SHA-256、episode 索引和 target ID。校验通过后额外生成：

- `comparison.json`：机器可读的基线、候选、差值和逐目标配对比较；
- `COMPARISON.md`：成功率、最终误差、每目标最小误差及配对改善报告。

如果目标清单或目标 ID 不一致，流水线会把对照阶段标记为失败，而不会生成
不可比的指标。后续版本应创建新配置，保留 `baseline_manifest` 指向 v1；
不要修改 `BASELINE.json` 中冻结的权重、数据或评测指标。需要更换基准时，
应新增版本化 baseline 目录。

终端显示运行目录和各阶段日志位置。查看训练进度可运行：

```bash
tail -f outputs/你的运行目录/training.log
```

失败时停止后续阶段并保留日志；评测缺集或重复目标不会被当成完整结果。
`status=completed` 表示流程完成，不代表任务全部成功；任务结果看成功数和误差。
`trajectory_minimum_error_um` 沿用轨迹中动作前误差记录的最小值，最终误差来自
最后动作后的 `final_error_mm`，两者时间口径与原记录一致。

## 新增模块或更换模型

如果修改当前模型内部模块，直接运行即可：每次都会复制并执行运行时的当前源码。
训练与评测都从同一快照启动新进程，后续编辑源文件不会改变已开始运行的模型。

如果新建模型入口，修改配置中的这两项：

```json
{
  "model_name": "state_attention_v2",
  "model_entry": "lerobot_training/dit_state_attention_v2.py"
}
```

模型入口沿用当前约定：支持 `train` 和 `eval` 子命令；训练负责构建新模型，
评测也必须构建同一架构。仅创建一个模块文件而未接入模型 forward，不会自动启用它。
入口放在 `lerobot_training` 下，与所需的 Python 模块一起被快照保存。
源码外的仿真模型与资源仍由原项目提供，因此不要把本工具当作独立运行环境。

配置中的训练/评测字段直接转成原脚本参数。数据路径、初始权重和目标清单路径
均相对 `rcm_4090_deploy`，也可使用绝对路径；`model_entry` 需位于本项目内。
更换目录可用 `--project-root /绝对路径/rcm_4090_deploy`。
使用其他配置文件可用 `--config configs/你的配置.json`。
完整流程不会回退到旧权重：训练失败便停止；评测使用本次所选 checkpoint。

不增加依赖安装、数据重采集、额外验证轮次、消融或预运行检查。
