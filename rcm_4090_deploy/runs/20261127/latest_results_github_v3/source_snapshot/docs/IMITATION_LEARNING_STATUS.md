# 模仿学习实现状态与 Baseline

本文总结当前 RCM 项目模仿学习的实现成熟度、主要模型版本、闭环结果、
baseline 定义和尚未解决的问题。所有数字均为 MuJoCo 仿真结果，不代表
真实硬件、真实 OCT 或临床精度。

## 1. 当前结论

项目已经从“能够训练策略”发展到“能够完成严格 RGB-D + MotorBackend
单目标闭环”，但尚未达到跨目标泛化或实机部署阶段。

当前状态可概括为：

```text
工程链路：较完整
主分布仿真性能：较好
跨目标泛化：不足
终端稳定保持：不足
实机部署：尚未验证
```

当前主模型仍是 v7。v8 和 v9 都改善了失败尾部，但没有提升固定 20 目标的
成功数量或整体中位误差，因此尚未替换 v7。

## 2. 当前实现链路

```text
v9 眼球场景与 eye_internal 相机
        │
        ▼
严格 RGB-D 感知 + 仿真 OCT 导航接口
        │
        ▼
几何视觉伺服专家 / DAgger 教师
        │
        ▼
MotorBackend 执行与同步数据采集
        │
        ▼
LeRobot-style 数据转换与图像缓存
        │
        ▼
guided OCT Diffusion Policy
        │
        ▼
7-sample median + 动作限幅
        │
        ▼
严格 RGB-D + MotorBackend 闭环评估
```

当前实现包含：

- MuJoCo v9 场景和 `eye_internal` 相机。
- 严格 RGB-D 感知，不使用仿真 geom ID 控制。
- 可见几何目标/针尖测量。
- 仿真 OCT 目标相对向量接口。
- MotorBackend 编码器和一阶电机响应。
- 基础 recovery 数据集和 DAgger 数据集。
- ResNet18 视觉编码器。
- A/B/C 状态和 OCT 导航输入。
- guided base action + bounded diffusion residual。
- 7-sample median 动作聚合。
- 逐轴动作限幅和独立闭环验收。
- 冻结训练/验证/测试目标划分。

该实现是自定义 PyTorch 训练栈，不是官方 LeRobot 框架，也不是 π0。

## 3. 数据集

最终主训练数据规模：

```text
train: 608 episodes / 195,578 frames
val:   100 episodes /  29,000 frames
test:  100 episodes /  28,987 frames
```

数据包括：

- 原始 RGB
- 实际 A/B/C 编码器状态
- 执行动作
- 仿真 OCT 导航向量
- DAgger 教师动作和有效性
- 图像哈希与电机重放审计

训练输入图像缓存为 `224×224` RGB。

## 4. 当前主模型 v7

v7 定义为：

```text
guided OCT Diffusion Policy
+ DAgger 第 1 轮数据
+ 冻结视觉编码器
+ 预计算视觉特征缓存
```

### 模型结构

```text
RGB [2,3,224,224]
→ ResNet18
→ 每帧 512 维视觉特征

A/B/C、OCT 方向/log-distance/valid、guided base
→ 每帧 64 维状态编码

两帧融合
→ 1152
→ 512
→ 256
→ Diffusion
→ 未来 8 步 [dA,dB,dC]
```

训练时动作目标为：

```text
expert action - guided base action
```

网络学习有界残差，而不是从零预测完整动作。

### 闭环结果

第一组 20 个独立目标：

```text
20 µm 成功：19/20
成功率：95%
超过 30 mm 失控：0/20
safety-limited steps：0
最终误差 mean/median/min/max：
11.249 / 9.920 / 4.624 / 47.926 µm
```

第二组独立目标：

```text
20 µm 成功：20/20
成功率：100%
最终误差 mean/median/min/max：
9.840 / 10.160 / 7.509 / 11.298 µm
```

两组合计：

```text
39/40 = 97.5%
```

第一组唯一失败 episode 没有失控，最终稳定在约 `47.926 µm` 的平台。

## 5. 后续候选模型

### 5.1 v8 全量解冻

v8 定义为：

```text
从 v7 初始化
关闭 vision feature cache
关闭视觉编码器冻结
ResNet18 全量参与训练
```

训练结果：

| 实验 | 最佳 normalized action MAE |
|---|---:|
| v7 冻结视觉编码器 | 0.310172 |
| v8 batch 48, lr 5e-5 | 0.286544 |
| v8 batch 96, lr 1e-4 | **0.273814** |

仅看离线指标，v8 batch 96 最好。但闭环结果为：

```text
20 µm 成功：19/20
最终误差 mean/median/min/max：
11.848 / 11.117 / 9.712 / 26.791 µm
超过 30 mm 失控：0/20
safety-limited steps：0
```

与 v7 同 seed 逐 episode 配对：

```text
v8 最终误差改善：6/20
平均最终误差变化：+0.599 µm
最差最终误差：47.926 µm → 26.791 µm
```

结论：

- v8 改善了失败尾部。
- v8 没有提升成功集数量。
- v8 的平均和中位闭环误差略差于 v7。
- 离线 MAE 改善没有转化为整体闭环增益。

因此 v8 是候选模型，不是当前主模型。

### 5.2 v9 terminal-DAgger candidate

v9 的网络结构、视觉冻结策略、guided OCT 和动作采样方式与 v7 相同，新增
200 集、18,544 个有效教师帧的 terminal DAgger 数据后重新训练。该数据覆盖
20 个近目标/侧偏/过冲 profile，每个 profile 10 集；训练集合并后为 690 集、
163,097 帧。

训练使用 v7 初始化、冻结 ResNet18、视觉特征缓存、guided OCT、4 epochs、
batch 48 和 lr `5e-5`。最佳 checkpoint 为 epoch 1，离线 normalized action
MAE 为 `0.231271`，A/B/C MAE 为
`0.0116817°/0.0061930°/0.0031425 mm`。

使用与 v7/v8 完全相同的三路 seed `20271105/20271106/20271107` 和
`7/7/6` 目标分片进行闭环评估：

```text
20 µm 成功：19/20
成功率：95%
超过 30 mm 失控：0/20
safety-limited steps：0
最终误差 mean/median/min/max：
11.159 / 10.865 / 7.437 / 23.146 µm
```

与 v7 同目标逐集配对：

```text
v9 最终误差改善：5/20
恶化：15/20
平均最终误差变化：-0.091 µm
中位最终误差变化：+0.945 µm
最差最终误差：47.926 µm → 23.146 µm
```

与 v8 配对：

```text
v9 改善：9/20
恶化：7/20
持平：4/20
平均最终误差变化：-0.689 µm
最差最终误差：26.791 µm → 23.146 µm
```

结论：

- terminal DAgger 明显改善了失败尾部。
- v9 没有提升 20 µm 成功集数量。
- v9 的平均最终误差略优于 v7，但中位最终误差更差。
- 唯一失败目标停在 `23.146 µm`，最后阶段 OCT invalid、动作和速度均为零，
  没有失控或安全限位。
- v9 是更强的失败尾部候选，但尚未达到替换 v7 的工程门槛。

### 5.3 v10 temporal-conditioned candidate

v10 在 v9 基础上增加：

```text
四帧观测历史
状态差分速度
上一步实际执行动作
```

视觉编码器仍冻结，条件融合从 `1152` 扩展到
`(512+64)×4+64=2368`，动作仍预测 8 步。最佳离线 checkpoint 为 epoch 0：

```text
normalized action MAE = 0.219186
A/B/C MAE =
0.0106900° / 0.0058183° / 0.0030266 mm
```

离线性优于 v9，但完整 20 目标闭环没有提升：

| 执行方式 | 成功 | mean | median | max |
|---|---:|---:|---:|---:|
| v10 exec1 | 19/20 | 12.464 µm | 10.418 µm | 45.648 µm |
| v10 exec2 | 19/20 | 12.559 µm | 11.181 µm | 41.116 µm |

7 个目标的 action-chunk 屏幕测试中，执行 `1/2/4` 步均为 `6/7`；
中位最终误差分别为 `10.201/11.411/12.873 µm`。较长的 chunk 没有精度收益。
7 个 Diffusion 样本已改为单 batch 并行采样，并保留 OOM 顺序回退；但严格
RGB-D 和 MuJoCo 逐帧观测仍是闭环总时间主项。

结论：

- v10 改善离线动作拟合，但没有改善闭环成功率。
- v10 的失败尾部比 v9 更差。
- 执行 2 步不优于执行 1 步，执行 4 步筛选结果更差。
- 当前继续以第 1 步执行和 7-sample median 为默认控制策略。

### 5.4 Action-horizon ablation

固定 v10 的四帧历史、速度、上一步动作、guided OCT、数据、初始化和训练
预算，仅改变预测时域 `H`。完整模型训练最佳 normalized MAE：

```text
H1 = 0.223821（闭环未完整）
H2 = 0.219375（闭环未完整）
H4 = 0.246795
H8 = 0.219186
```

7 目标 screen 中四种模型均为 `6/7`，H4 的
mean/median/max=`12.609/10.094/27.691 µm` 为最好；H1/H2/H8 分别为
`16.407/11.282/49.670`、`17.211/12.079/49.068`、
`15.136/10.201/45.648 µm`。

H4 完整 20 目标闭环：

```text
success = 19/20
mean = 11.459 µm
median = 10.807 µm
max = 27.691 µm
safety-limited = 0
uncontrolled >30 mm = 0
```

与 v9 配对，H4 改善 `10/20`、恶化 `10/20`，平均最终误差增加 `0.300 µm`，
最大误差增加 `4.545 µm`。与 H8 配对，H4 改善 `9/20`、恶化 `11/20`，
但平均改善 `1.005 µm`，最大误差改善到 `27.691 µm`。

结论：

- H4 比 H8 明显改善了失败尾部。
- H4 没有超过 v9 的平均和最大误差。
- H1/H2 没有显示足以替代 H4 的 screen 收益，且完整闭环尚未完成。
- 当前 horizon 结论不是“越短越好”；H1/H2 screen 反而差于 H4。
- 所有 horizon 的共同失败仍是同一目标的 OCT invalid 后零动作停驻。

### 5.5 H4 frozen final-test result

H4 在 60 个冻结 final-test targets 上各评估 1 个 seed，使用严格 RGB-D、
MotorBackend、OCT、四帧历史、预测 4 步、执行 1 步、7-sample median、
1600 步和 20 µm 三帧停稳判据。

该历史运行使用评估契约 v2 之前的低速三帧口径，未检查编码器是否到达量化
命令目标；这里不应把旧口径成功数解释为编码器真停稳成功。

```text
20 µm success: 28/60 = 46.7%
10 µm success: 9/60 = 15.0%
5 µm success:  0/60

final error mean/median/P95/max:
622.638 / 21.295 / 332.104 / 19156.335 µm
```

失败结构：

```text
35/60 曾进入20 µm
28/60 完成停稳成功
7/60 进入20 µm但未完成三帧停稳
25/60 从未进入20 µm

最低误差分布：
0–20 µm: 35
20–30 µm: 5
30–50 µm: 12
50–100 µm: 2
100–1000 µm: 4
>=1000 µm: 2
```

32 个失败 episode 中，29 个跑满 1600 步，3 个因连续 safety-limit 早停。
目标簇依赖非常明显：g1 和 g8 分别为 `0/8`、`0/7`，g6 为 `7/7`。
两个 episode 的最终误差达到 `19.156 mm` 和 `15.734 mm`。

结论：

- H4 的固定 20 目标 `19/20` 没有转化为跨目标泛化。
- final-test 同时存在基础接近错误和终端停稳失败。
- 模型已经进入过 20 µm 的目标中仍有 7 个无法稳定保持。
- 后续优化必须同时增加目标分布覆盖并改进 terminal hold，不能只继续调
  horizon。

## 6. Baseline 定义

需要区分科学 baseline 和工程 baseline。

### 6.1 科学对照 Baseline：v4 no-DAgger Diffusion

该 baseline 用于判断 guarded diffusion、guided OCT 和 DAgger 是否有效。

```text
RGB + A/B/C + OCT
→ Diffusion Policy
→ 无 DAgger 教师纠正
```

固定 20 目标闭环结果：

```text
20 µm 成功：0/20
最终误差 mean/median：21.591 / 30.021 mm
超过 30 mm 失控：13/20
每集最低误差 mean/median：2.794 / 2.577 mm
safety-limited steps：0
```

这说明无 DAgger 策略能够接近目标，但无法稳定完成闭环。

### 6.2 当前工程 Baseline：v7

后续新模型应默认与 v7 比较：

```text
v7 = guided OCT + DAgger + 冻结视觉编码器
```

建议的验收条件：

```text
20 µm 成功率 >= 19/20
最终误差中位数 <= 约 10 µm
超过 30 mm 失控 = 0
safety-limited steps = 0
```

任何替换 v7 的新模型都必须至少满足这些条件。

## 7. DAgger 的作用

v4 到 v7 的关键变化之一是 DAgger。

流程为：

```text
当前行为策略执行
→ 工具进入接近失控或较大偏差状态
→ 严格 RGB-D 几何教师接管
→ 保存 policy/teacher/executed action
→ 只保留教师深度有效的纠正标签
→ 与 recovery 数据合并
→ 重新训练
```

DAgger 修复的主要问题：

- 扩散采样偶尔输出灾难性饱和动作。
- 策略在有效 OCT 条件下仍做出错误方向动作。
- 大范围偏差状态覆盖不足。
- 失败状态被闭环反馈放大。

DAgger 后，固定 20 目标从 `0/20` 提升到 `19/20`，失控从 `13/20`
降到 `0/20`。

## 8. Restricted Perception 路线

Restricted 策略使用：

```text
视觉特征 512
+ A/B/C 3
+ restricted vector 18
```

基础输入维度：

```text
533
```

当前选中配置：

```text
dagger_only
use_velocity = false
```

### Development 结果

修正后的 Stage 1：

```text
normal restricted vector：39/40
zero vector：10/40
same-distance mismatch vector：9/40
```

Stage 2：

```text
dagger_only：40/40
final-error P95：16.67 µm
```

Stage 3 residual-distilled：

```text
development success：40/40
residual limit：
[0.005°, 0.003°, 0.001 mm]
```

Development 上，restricted 路线可以稳定完成。

### Frozen Final Test

最终冻结测试失败：

| 阈值 | Distilled single-step | Seven-sample median |
|---|---:|---:|
| 20 µm | 20/180，11.1% | 18/60，30.0% |
| 10 µm | 14/180，7.8% | 17/60，28.3% |
| 5 µm | 7/180，3.9% | 6/60，10.0% |

180 个 final-test episode 中：

```text
20 集成功
51 集曾进入 20 µm，但未连续三帧停稳
8 集最低误差 20–30 µm
41 集最低误差 30–50 µm
45 集最低误差 50–100 µm
6 集没有从 100 µm 起点接近
9 集因 B 轴超出安全范围在初始化失败
```

51 个终端保持失败 episode：

```text
最低误差中位数：5.54 µm
最终误差中位数：96.29 µm
fine channel 可用率：99.52%
coarse fallback：从未发生
```

结论：

```text
主要问题不是 fine OCT 丢失
主要问题是终端稳定和跨目标分布泛化
```

## 9. 当前实现成熟度

| 层面 | 状态 |
|---|---|
| 场景与相机 | 可运行 v9 仿真 |
| 严格 RGB-D | 已完成并验证 |
| MotorBackend | 已完成仿真闭环 |
| 数据采集 | 支持 recovery 和 DAgger |
| 数据完整性 | 有哈希、编码器和离线重放验证 |
| 主策略训练 | 可训练、可选择 checkpoint |
| 主策略闭环 | v7 为工程 baseline；v9 为 19/20 且改善失败尾部 |
| 跨目标泛化 | restricted final test 失败 |
| 终端稳定 | 仍需改进 |
| 真实相机 | 未验证 |
| 真实 OCT | 未接入 |
| 实机运动学/动力学 | 未标定 |
| 组织接触/碰撞 | 未验证 |
| 临床安全 | 不成立 |

## 10. 当前主要限制

- 目标依赖人工橙色标靶。
- 针尖检测依赖颜色、针具尺寸和固定相机/RCM 标定。
- OCT 是理想配准仿真深度，不是真实 A-scan/B-scan。
- 主策略只看到 224×224 RGB，细节有限。
- 主策略只有两帧历史，没有 RNN/Transformer 时序模型。
- 8 步动作预测当前只执行第一步。
- 7-sample diffusion 推理较慢。
- v8 全量解冻没有带来闭环整体提升。
- v9 terminal DAgger 改善了最差最终误差，但没有提升固定目标成功数量。
- v10 四帧/速度/上一步动作没有转化为闭环收益。
- 2/4 步 action chunk 未改善 20 µm 成功率或中位精度。
- H4 final-test 只有 `28/60` 达到 20 µm，跨目标泛化明显失败。
- final-test 目标区域明显偏离 development 分布。
- 51 个 final-test episode 证明终端保持仍是核心瓶颈。

## 11. 下一步优先级

### 第一优先级：终端稳定

- 增加 20–100 µm 近目标数据。
- 对进入误差范围后的保持动作单独加权。
- 增加 hold、过冲恢复和穿越目标样本。
- 评估 terminal residual head 或显式停止判定。

### 第二优先级：目标分布覆盖

- 用新 training/DAgger 目标覆盖 final-test 类似区域。
- 不能直接用已查看的 60 个 final targets 训练。
- 修改策略后应重新冻结全新的 final test。

### 第三优先级：推理效率

- 比较执行动作块 1/2/4 步。
- 比较 `n_action_steps=1/4/8`。
- 将 7 次顺序采样改成 48 GB GPU 上的批量采样。
- 评估单步蒸馏模型作为主策略。

### 第四优先级：时序建模

- 增加观测历史到 4–5 帧。
- 显式输入速度和上一步执行动作。
- 评估 GRU、temporal convolution 或 Transformer。

### 第五优先级：实机

- 相机内参、针具、RCM 和电机动力学实机标定。
- 真实 OCT 接入及延迟/坐标配准。
- 碰撞、组织接触、力反馈和制动安全。

## 12. 当前推荐表述

### 可以写

```text
We implement a compact ResNet18-based guided OCT diffusion policy
trained with recovery and DAgger data in MuJoCo.

The current v7 policy achieves 39/40 closed-loop success at a 20 µm
threshold over two independent 20-target groups under strict RGB-D
feedback and MotorBackend execution.

The no-DAgger diffusion baseline achieves 0/20 success on the matched
20-target evaluation.
```

### 不应写

```text
The method is hardware-ready.
The method achieves clinical precision.
The model uses official LeRobot or π0.
The restricted final test passes.
```

## 相关证据

- v7 固定目标闭环：
  `outputs/evaluation_v7_guided_dagger1_cached_20261106/aggregate_summary.json`
- v7 独立重复闭环：
  `outputs/evaluation_v7_guided_20mm_repeat_20261106/aggregate_summary.json`
- v8 全量解冻闭环：
  `outputs/evaluation_v8_full_unfreeze_b96_lr1e4_20261112/aggregate_summary.json`
- v9 terminal-DAgger 闭环：
  `outputs/evaluation_v9_terminal_dagger_20261113/aggregate_summary.json`
- v10 execution-1 闭环：
  `outputs/evaluation_v10_temporal_20261113/aggregate_summary.json`
- v10 execution-2 闭环：
  `outputs/evaluation_v10_temporal_exec2_20261113/aggregate_summary.json`
- H4 action-horizon 闭环：
  `outputs/evaluation_v11_h4_20261113/aggregate_summary.json`
- H4 final-test 闭环：
  `outputs/evaluation_v11_h4_final_test_parallel8_20261113/aggregate_summary.json`
- H4 final-test 阈值与失败结构：
  `outputs/evaluation_v11_h4_final_test_parallel8_20261113/final_test_summary.json`
- 训练与验收历史：
  [`../../TEST_LOG.md`](../../TEST_LOG.md)
- Restricted 研究说明：
  [`RESTRICTED_PERCEPTION.md`](RESTRICTED_PERCEPTION.md)
- Restricted 输入流程：
  [`RESTRICTED_INPUT_PIPELINE.md`](RESTRICTED_INPUT_PIPELINE.md)
