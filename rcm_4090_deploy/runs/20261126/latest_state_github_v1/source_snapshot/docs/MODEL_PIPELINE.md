# Linux 模型训练、验证与闭环评测流水线

使用部署虚拟环境运行训练和评测。一次命令会完成训练、逐轮验证、闭环评测和结果登记。
新的实验任务默认使用[实验 SOP](EXPERIMENT_SOP.md) 的实验卡入口；下方是底层流水线接口。

## 完整运行

在 `rcm_4090_deploy` 目录执行。下面用 State Attention v1 训练合并数据，
随后评估前 10 个 development 目标：

```bash
source .venv/bin/activate
python -u lerobot_training/run_model_pipeline.py \
  --config configs/state_attention_v1_plan2_plus_original200_pipeline_20261119.json \
  --output-dir outputs/state_attention_v1_plan2_plus_original200_pipeline_20261119_1
```

原版 DiT-Flow 可使用 `configs/dit_rectified_flow_plan2_plus_original200_pipeline_20261119.json`，
并指定另一个新的 `--output-dir`。该配置评估前 20 个 development 目标。
两份配置均从各自架构的初始权重训练 4 轮，使用原有 839 条加规划 2 新增 200 条训练轨迹、独立验证集，以及 CUDA 和 EGL。
不传 `--config` 时仍使用 `configs/state_attention_pipeline.json`。

1. 预先检查目标清单、目标范围和配对基线协议，并把当前 `lerobot_training` 源码复制到本次运行的 `code/`。
2. 从该代码快照执行模型训练；每轮验证由原训练器完成。
3. 训练结束后，使用本次产生的 `training/best_model.pth`。
4. 从同一代码快照启动闭环评测分片，覆盖配置指定的目标范围。
5. 合并全部评测结果，保存 JSON、CSV 和 Markdown 报告。

沿用当前协议：4 轮训练，0.35 μm 成功阈值，50 ms 控制周期，
1600 步，1 次动作采样，16 步预测中执行首步，连续 3 帧编码器停稳。
不同分片使用相同的基础 seed，目标的实际 seed 仍为 `seed + episode_index`。

## 已有模型只做评测

```bash
python -u lerobot_training/run_model_pipeline.py \
  --config configs/state_attention_v1_plan2_plus_original200_pipeline_20261119.json \
  --eval-only outputs/state_attention_v1_plan2_plus_original200_20261119_1/best_model.pth \
  --output-dir outputs/state_attention_v1_plan2_plus_original200_eval_dev10_20261119_2
```

这会创建新的记录目录，不重复训练或验证。配置中的模型入口必须与该权重架构匹配。

## 运行目录与日志

请用 `--output-dir` 指定一个尚不存在的目录。保留实验结果用 `outputs/<experiment>/`，
验证用 `runs/YYYYMMDD/verification/<purpose>/`。运行结束后，流水线会向
`TEST_LOG.md` 追加完成、失败或中断记录。


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

训练前，流水线会核对目标清单 SHA-256、评测协议，以及基线是否包含每个选中目标的
episode 索引和 target ID。可以用 20 条基线中的前 10 条做配对，基线指标只按这 10 条重算。
最终评测结果也必须覆盖全部选中目标，且索引、target ID 和实际 seed 必须匹配。校验通过后额外生成：

- `comparison.json`：机器可读的基线、候选、差值和逐目标配对比较；
- `COMPARISON.md`：成功率、最终误差、每目标最小误差及配对改善报告。

如果协议或目标不匹配，流水线会报告失败。使用不同协议做冒烟验证时，可省略 `baseline_manifest`。
后续要做同协议对照时，应创建新配置，保留 `baseline_manifest` 指向 v1；
不要修改 `BASELINE.json` 中冻结的权重、数据或评测指标。需要更换基准时，
应新增版本化 baseline 目录。

终端同时显示训练、验证和各分片的真实日志，文件仍在原路径落盘。
闭环在首步和每 100 步报告进度（可用 `RCM_EVAL_PROGRESS_EVERY` 调整）。
查看已保存的训练日志可运行：

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

默认使用 `model_entry train …` 和 `model_entry eval …`。训练脚本独立时增加
`training_entry`，例如 `lerobot_training/train_dit_rectified_flow.py`，
流水线会直接把训练参数传给该脚本。`model_entry` 继续接收 `eval` 子命令，且必须支持训练产生的权重架构。
两个入口放在 `lerobot_training` 下，与所需的 Python 模块一起被快照保存。
源码外的仿真模型与资源仍由原项目提供，因此不要把本工具当作独立运行环境。

配置中的训练/评测字段直接转成原脚本参数。数据路径、初始权重和目标清单路径
均相对 `rcm_4090_deploy`，也可使用绝对路径；两个入口均需位于本项目内。
更换目录可用 `--project-root /绝对路径/rcm_4090_deploy`。
使用其他配置文件可用 `--config configs/你的配置.json`。
完整流程不会回退到旧权重：训练失败便停止；评测使用本次所选 checkpoint。

