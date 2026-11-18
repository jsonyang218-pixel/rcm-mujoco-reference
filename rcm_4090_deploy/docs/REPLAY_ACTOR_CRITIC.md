# v1 经验回放训练模块

在 state-attention v1 上直接加入经验池、双 Q 网络、目标网络和专家示范约束。
这是适配现有 DiT/Rectified Flow 的 actor-critic 扩展，尚无训练结果。
代码基于远端 `040fe0b`（Publish state attention E1 error trajectories）；
安装包仅包含本次新增的五个文件，解压到现有仓库根目录即可。

## Linux 一条命令

在 `rcm_4090_deploy` 下，用现有训练环境执行：

```bash
python -u lerobot_training/run_model_pipeline.py \
  --config configs/state_attention_replay_pipeline.json
```

配置从已经冻结的 **v1 best checkpoint** 初始化，自动训练、验证、评测新权重，
并与冻结 v1 的同一组 20 个 development 目标比较。正式测试目标不进入训练。
原来的 v1 权重路径来自已有配置；若服务器上的文件位置不同，修改 `training.init_from`。

## 加在哪里

- 策略网络：沿用 v1 的视觉编码、状态偏置、DiT 和四步 Flow 采样，更新其参数。
- 经验池：从 v2 交互轨迹构造 `(observation, executed_action, reward, next_observation)`。
  缓存的是帧索引和奖励，图像继续从原数据集读取。达到容量后均匀蓄水池保留，
  更新时均匀抽样；不额外引入优先回放或新的注意力机制。
- 双 Q 网络：各有两个 256 维隐藏层，只使用已有的编码器状态、OCT、速度、
  上一实际动作及当前动作，不读取仿真真值。目标 actor 和目标 Q 用软更新。
- 专家约束：继续使用原来的专家动作序列做 Flow matching。
  交互动作不是专家标签，不会用失败动作作为模仿目标。
- Q 引导：对预测 chunk 的**第一步动作**求 Q 梯度，反传穿过四步 Flow 积分，
  与部署时只执行第一步保持一致。原来的 `sample()` 带 `no_grad`，因此训练模块
  有单独的可微采样函数。
- 推理：只加载新训练得到的策略参数；Q 网络不参与实时执行。
  新 eval 入口要求 checkpoint 的 `training_module` 匹配，避免误评旧 v1。

这不是原论文 ILOSAC 的复现：原方法的高斯 actor、SAC 熵项、KL 分布约束和
THMM 约束均未照搬。当前保留 Flow 策略，用专家 Flow 损失约束更新，
没有把 Flow 输出假设成可直接计算 SAC 对数概率的高斯策略。

## 更新目标

双 Q 的目标使用实际执行动作对应的状态转移：

\[
y=r+\gamma(1-\mathrm{terminated})\min_j\bar Q_j(s',\bar\pi(s')).
\]

策略的目标为：

\[
L_\pi=\lambda_{FM}L_{FM}
-\lambda_Q\frac{\mathbb E[\min_j Q_j(s,\pi(s))]}
{\max(1,\operatorname{stopgrad}(\mathbb E[|\min_j Q_j(s,\pi(s))|]))}.
\]

策略 Q 项取回放状态，Flow 项取专家数据中的完整动作窗口。
双 Q 使用几何/OCT 观测而非图像，因此它的观测表征能力也是本版的一个限制。

奖励使用观测到的 OCT 偏差范数，距离单位 mm：

\[
r=w_p(d_t-d_{t+1})/s-w_d d_{t+1}/s
-w_a\operatorname{mean}[(a_t^{exec}/a_{limit})^2].
\]

默认 `s=1 mm, w_p=1, w_d=0.1, w_a=0.001`。这些只是第一轮试验的起始值，
不表示已验证最优，也不承诺回放能解决当前漂移。
奖励中的动作尺度和 Q 查询中的截断均沿用现有 evaluator 的动作范围，
本改动没有改变执行器或评测器的控制规则。

v2 转换数据目前没有可靠的逐帧终止标签，因此默认按持续定位任务学习：
数据结束不当作成功，轨迹末尾也不凭距离推断成功。若额外数据明确提供
与状态行对齐的布尔 `terminated` 数组，则使用该标签停止 bootstrap。
评测仍沿用原来的误差阈值和连续三帧编码器停稳判据。

## 数据与后续交互

第一轮直接使用配置中原有专家/DAgger 轨迹中的已记录交互。
这是**离线回放微调**，本命令不会自动采集当前策略的新轨迹。
回放旧轨迹本身不能补齐当前策略未覆盖的状态。

若有新交互轨迹，转换为现有 v2 格式后，在配置的 `training` 中加入：

```json
"replay_data_dir": ["data/your_new_v2_rollouts"]
```

这些额外目录只进入 Q 引导的经验池，专家 Flow 监督仍来自 `data_dir`。
额外轨迹必须保留实际执行动作、原始帧号、时间戳、OCT 和图像。
有断帧、跨 episode、控制周期不符或奖励端点 OCT 无效的转移不进入经验池；
最后不足 16 步的片段仍可用于一步回放，不会因 BC 窗口不足全部丢弃。

## 结果记录

现有 pipeline 继续保存代码快照、命令、训练/验证历史、新权重、20 目标评测、
逐目标 CSV、summary 和相对 v1 的 comparison。训练目录还保存：

| 文件 | 内容 |
|---|---|
| `training_config.json` | 本轮实际训练设置与继承的模型配置 |
| `replay_manifest.json` | 数据来源、有效转移数、奖励公式和终止语义 |
| `replay_index.npz` | 本轮保留的经验索引、奖励和终止标记 |
| `training_history.json` | 专家 Flow loss、Q loss、Q 引导项、奖励与验证 loss |
| `best_model.pth` / `last_model.pth` | 策略、双 Q、目标网络和优化器参数 |

`train_loss` 和 `val_loss` 均为专家 Flow loss；组合策略目标另记为
`q_objective`，不与它们混成同一指标。best checkpoint 仍按验证 Flow loss 选择，
它未必有最好的闭环控制效果，最终结论以同协议的轨迹评测为准。
本版不提供训练断点续跑，保存优化器状态不代表已有 resume 命令。

必要检查可在 `rcm_4090_deploy` 下执行：

```bash
python -m unittest discover -s tests -p 'test_experience_replay.py' -v
```

没有 PyTorch/torchvision 时仅执行数据回放检查，梯度检查会明确跳过。

本次交付：四项数据检查通过，两个 PyTorch 梯度检查因依赖缺失跳过；
Python 编译和 pipeline 配置参数匹配检查通过。当前环境没有 PyTorch、
torchvision 和 MuJoCo，尚未验证完整训练、GPU 显存占用或闭环性能。
