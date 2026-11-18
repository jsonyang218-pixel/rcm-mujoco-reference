# Frontiers 2022：公式复现与 RCM 适配

本文对应 Chen、Jiang、Cheng、Knoll、Zhou 的 **Deep Reinforcement Learning Based Trajectory Planning Under Uncertain Constraints**，Frontiers in Neurorobotics，2022，16:883562。

- [论文全文与 Algorithm 1/2](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.883562/full)
- [原文 PDF](https://public-pages-files-2025.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.883562/pdf)
- DOI：10.3389/fnbot.2022.883562

这里提供的是依据公式与伪代码独立编写的 **Algorithm 1 DDPG 更新规则 + DiT-Flow/RCM 离线适配**。尚未确认该论文的作者官方源码；本代码不能称为作者源码，也不是原论文 PyBullet/Panda 实验的完整复现。SAC（Algorithm 2）本次未实现。

## 公式和代码对应

原文 PDF 第 6–7 页公式 (5)–(7)：

$$R=-c_1R_T-c_2R_O,\quad R_T=\frac12d_T^2,\quad R_O=\left(\frac{a}{a+d_O}\right)^n.$$

距离使用米，原文设置 $c_1=500,c_2=15,a=1,n=35$。`frontiers2022_ddpg.paper_reward()` 同时提供目标项与可选障碍项。RCM 数据缺少最近障碍距离，本次适配明确关闭障碍项；每个真实转移的奖励为：

$$r_t=-250\left(\frac{\|\operatorname{OCT}(s_{t+1})\|_{\mathrm{mm}}}{1000}\right)^2.$$

采用动作执行后的下一状态计算距离，是本项目的时序约定。20 mm、1 mm、100 μm、0.35 μm 的奖励分别为 −0.1、−0.00025、−0.0000025、约 −3.0625×10⁻¹¹；没有暗中扩大微米区间奖励或增加进度/动作惩罚。

原文第 8 页 Algorithm 1 与公式 (9)：

$$y_i=r_i+\gamma Q_{\bar\phi}(s'_i,\mu_{\bar\theta}(s'_i)),$$
$$L_Q=\frac1B\sum_i\big[Q_\phi(s_i,a_i)-y_i\big]^2,$$
$$L_\mu=-\frac1B\sum_i Q_\phi(s_i,\mu_\theta(s_i)),$$
$$\bar\phi\leftarrow\tau\phi+(1-\tau)\bar\phi,\quad
\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta.$$

| 原文部分 | 代码位置 | 实际实现 |
|---|---|---|
| 公式 (5)–(7) | `frontiers2022_ddpg.py: paper_reward` | 原系数与米单位；RCM 接口关闭缺失的障碍项 |
| 转移回放 | `experience_replay.py` 与新 `transition_records` | 均匀抽样；容量超限时均匀 reservoir 保留；使用实际执行动作 |
| 一个 Q 与目标 Q | `SingleQ`, `update_critic` | 单 Q，平方 TD 误差，无双 Q 的最小值 |
| 确定性策略梯度 | `update_actor` | 沿 Q 对动作的导数更新 DiT；不按 Q 绝对值归一化 |
| 目标网络平滑更新 | 共用 `soft_update` | actor/Q 均使用 τ 平滑更新 |
| DiT/Flow 接入 | `DeterministicFlowPolicy` | 固定同一个初始高斯噪声块，四步 Flow 积分；只执行动作块第 0 步 |

DDPG 的动作梯度通过 Flow 积分回传到原 DiT 权重，包含可学习的 Q/K/V；没有另换 QKV 结构，也没有使用 KV8 压缩。新增的单 Q 是训练辅助网络，部署推理只用 actor。

## 论文与项目的边界

| 项目 | 原论文 | 本版本 |
|---|---|---|
| 环境 | PyBullet、7 自由度 Panda | MuJoCo RCM |
| 状态 | 末端位置/速度、相对目标和障碍位置/速度 | 保留 RGB、编码器、OCT、速度、前一步实际动作；Q 只看归一化几何历史 |
| 动作 | 笛卡尔 Δx/Δy/Δz | ΔA_deg/ΔB_deg/ΔC_mm，保留原评测限幅 |
| Actor | 确定性策略网络 | 已训练 v1 DiT-Flow，固定噪声以得到确定性 μ(s) |
| 数据 | 在线交互、OU 动作探索后存池 | 当前已有 v2 真实转移，离线采样；本次不新增在线 OU 采集器 |
| 初始化 | 随机初始化 actor/Q | v1 actor 热启动，Q 随机初始化 |
| 奖励 | 目标距离项和障碍项 | 只接入有数据支持的目标项 |
| 专家约束 | Algorithm 1 无此项 | 默认 `fm_weight=0`；非零是额外的项目 Flow 约束 |

终止标签掩码、梯度范数裁剪 1、观测/动作归一化、256 宽 SiLU Q 网络、初始 critic 预热 250 步、优化器和具体学习率均是工程适配。不能说这些全部是作者设置。缺少终止标签时按持续定位任务 bootstrap，不把文件末尾或超时伪造成成功。数据连续性、OCT 有效性规则沿用现有 Replay，防止跨轨迹拼接。

SAC 要求随机策略及其 log π；现有隐式 Flow 不能直接作为带可计算概率密度的 Gaussian SAC actor。原文 Algorithm 2 的印刷伪代码还需要核对若干项，因此本次没有把旧的“双 Q + Flow 约束”改名为 SAC。

普通经验回放与 DDPG 都已有前人方法。接入本论文的规则可以作为方法依据或对照；仅接入模块不构成创新证明。本次实验只回答“这套有出处的 DDPG 更新，在当前 DiT/RCM 数据上是否有用”。

## Linux 运行

本补丁基于已上传的 `28c7182` 版本。在已有训练环境、数据和原 v1 checkpoint 的服务器上，将 Linux ZIP 解压覆盖到 `rcm_4090_deploy` 目录（包内从 `lerobot_training/`、`configs/` 等目录开始）；随后从该目录运行：

```bash
bash scripts/run_frontiers2022_ddpg.sh
```

等价命令：

```bash
python lerobot_training/run_model_pipeline.py \
  --config configs/frontiers2022_ddpg_pipeline.json
```

配置沿用现有训练/验证路径。训练从原 v1 checkpoint 开始，单独输出到 `outputs/state_attention_frontiers2022_ddpg_时间戳/`，不覆盖旧 Replay 最优模型。默认 4×1000 次 Q 更新、3750 次 actor 更新，replay batch 16、γ=0.99、τ=0.005、actor lr=1e-6、critic lr=1e-4。固定噪声种子 20261118 写入配置，训练采样种子沿用 Replay 的 20261117。新模型默认无 Flow anchor、无 actor 权重衰减、bias 学习率倍率 1；这些与旧 Replay 存在差异，所以整体对比不能解释为单独某一项的因果效果。

流程自动完成训练、验证、固定 20 目标评测、结果汇总和与已上传 Replay 的逐目标比较。新增 `configs/baselines/state_attention_replay_v1/BASELINE.json` 只整理已上传结果，基线均值约 171.028 μm，未重新训练或修改旧结果。20 个目标、seed 42、1600 步、0.05 s 控制周期、每次执行一个动作均保持现有口径；策略采样改为固定噪声是本方法的明确变化。

结果目录记录：

- `code_version.json`、`commands.jsonl`：实际代码与执行命令。
- `training/training_config.json`、`replay_manifest.json`、`replay_index.npz`：固定噪声、论文 DOI、单位、数据来源及回放索引。
- `training/training_history.json`：actor/Q 更新数、Q loss、奖励、TD 目标、策略 Q、验证 Flow loss。
- `training/best_model.pth` 与 `last_model.pth`：actor、单 Q、目标网络及优化器状态；新 training_module 标识。
- `episodes.csv`、`summary.json`、`REPORT.md`、`comparison.json`、`COMPARISON.md`：闭环结果与 Replay 对照；分片完成时也记录阶段结果。

默认 `train_loss` 表示负加权策略 Q，**不是旧 Replay 的 Flow loss，不能横向比较这列**；`fm_weight>0` 时它表示 Flow loss。`val_loss` 仍为专家动作 Flow loss，`best_model.pth` 仍按最小验证 Flow loss选取，这只是兼容现有流水线的选择规则，不能代替闭环性能评测。也保留 `last_model.pth`。

单独评测某个新 checkpoint：

```bash
bash scripts/run_frontiers2022_ddpg.sh \
  --eval-only outputs/你的新实验目录/training/best_model.pth
```

评测入口要求新 `training_module` 标识并从 checkpoint 恢复固定噪声种子，防止误用旧模型。原有模型继续使用原有入口。

## 已验证结果

开发侧使用 CPU 验证奖励单位和障碍公式、连续转移索引、真实 ResNet/DiT/Flow 的确定性与权重重载、Q 梯度更新 actor、critic 保持冻结、终止掩码和目标网络更新；5 项新检查和 6 项旧 Replay 回归检查通过。另用合成 v2 数据跑通 3 次 Q 更新、2 次 actor 更新、验证、checkpoint 保存及评测入口的 actor 构造/加载，确认固定噪声种子恢复、旧模型标识被拒绝。此加载检查使用了评测入口替身，没有冒充真实仿真。测试不下载 ImageNet 权重，不替换生产模型的预训练逻辑。

随后在 RTX 4090 上使用真实训练集、原 v1 权重和 MuJoCo 评测环境完成完整流水线。4×1000 步训练的 best epoch 为 0，val loss `1.103461`；epoch 1–3 的 val loss 分别为 `13.099834/13.174330/16.239620`。固定 20 目标闭环 canonical success 为 `0/20`，最终误差
mean/median/min/max=`15909.594/13546.894/11030.918/23944.568 μm`，
每目标最小误差均值 `12334.938 μm`，safety-limited steps `108`。与
Replay v1 逐目标配对时 0 改善、20 恶化，最终误差均值恶化
`15738.567 μm`。因此这次离线 DDPG 适配不能替代现有 Replay 路径。

训练后期策略 Q 均值由 `0.021241` 升至 `0.673852`，而平均奖励约为
`-0.021`；闭环误差仍为毫米级。结合 best checkpoint 停留在 epoch 0，这与
critic/value 不匹配和 actor 外推风险一致，但当前实验不能单独确定因果。
结果目录为 `outputs/state_attention_frontiers2022_ddpg_20261118_1/`，紧凑
证据见 `github_reports/model_status_20261118_frontiers2022_ddpg/`。
结果仅为 MuJoCo 仿真，不代表硬件或临床精度。
