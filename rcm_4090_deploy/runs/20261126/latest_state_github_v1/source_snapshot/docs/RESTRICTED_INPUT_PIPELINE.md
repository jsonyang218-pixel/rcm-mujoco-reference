# A/B/C 状态与受限向量输入流程

本文说明 restricted-perception 策略中 A/B/C 状态和 18 维受限向量的生成、
融合和策略输入流程，并说明它们与当前 v7/v8/v9 主策略输入的区别。

## 总览

A/B/C 状态和受限向量是两路独立输入。A/B/C 不包含在 18 维受限向量中，
而是在策略输入端与视觉特征和受限向量拼接。

```text
编码器反馈
    │
    ▼
A/B/C 状态 [3]
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
正运动学得到当前针尖         保留为策略状态输入
    │
    ▼
Coarse/Fine 测量融合
    │
    ▼
受限向量 [18]

RGB → 视觉编码器 → 视觉特征 [512]

[视觉特征 512, A/B/C 3, 受限向量 18]
    │
    ▼
策略网络
    │
    ▼
[dA_deg, dB_deg, dC_mm]
```

## 1. A/B/C 状态的来源

每个控制周期，MotorBackend 推进 50 ms 后，场景从实际编码器反馈更新：

```text
编码器计数
→ 电机执行层
→ RCMSceneV9 当前 q
→ scene.current_task()
→ [A_rad, B_rad, C_m]
```

随后转换为策略使用的单位：

```text
A_deg = A_rad × 180/π
B_deg = B_rad × 180/π
C_mm  = C_m   × 1000
```

得到：

```text
state_ABC = [A_deg, B_deg, C_mm]
shape = [3]
```

A/B/C 状态同时用于三个位置：

1. 作为策略网络的机器人本体状态输入。
2. 通过正运动学计算当前针尖位置。
3. 在受限观测融合中补偿延迟测量期间的针尖运动。

这里记录的是实际编码器反馈，不是模型命令，也不是下一时刻的真值位姿。

## 2. 正运动学与当前针尖

当前 A/B/C 状态通过 RCM 运动学计算针尖机器人坐标：

```text
current_tip_robot_mm = forward_kinematics(state_ABC)
shape = [3]
```

当前针尖用于：

- 将 coarse 目标坐标转换为目标相对针尖的位移。
- 将延迟 fine 测量对齐到当前针尖位置。
- 判断 coarse/fine 融合距离。
- 构造安全 hold/retraction 行为。

`kinematic_tip_robot_mm` 参与融合计算，但不会直接放进 18 维策略向量。

## 3. Coarse 目标向量

Coarse 感知给出目标机器人坐标：

```text
coarse_target_robot_mm
shape = [3]
```

融合器将其转换为相对当前针尖的向量：

```text
coarse_goal_vector_robot_mm
  = coarse_target_robot_mm - current_tip_robot_mm
```

含义：

```text
从当前针尖指向 coarse 目标的三维位移
```

单位是 mm。

如果 coarse 测量无效，则不生成 coarse goal vector，策略向量对应位置填零。

## 4. Fine 目标向量

Fine OCT 测量最初位于相机坐标：

```text
fine_goal_vector_camera_mm
shape = [3]
```

首先旋转到机器人坐标：

```text
fine_measured_robot
  = R_camera_to_robot × fine_goal_vector_camera_mm
```

Fine 测量可能有 1 帧延迟。设测量时刻的针尖为：

```text
tip_at_measurement
```

当前针尖为：

```text
current_tip
```

测量帧到当前帧的针尖运动为：

```text
tip_motion = current_tip - tip_at_measurement
```

对相机测量向量进行延迟对齐：

```text
fine_goal_vector_robot_mm
  = fine_measured_robot - tip_motion
```

这样得到的是目标相对于当前针尖位置的向量。

## 5. Coarse/Fine 融合

融合器根据 coarse 目标到当前针尖的距离决定 fine 权重：

```text
distance <= 1.8 mm       fine_weight = 1
distance >= 2.2 mm       fine_weight = 0
1.8 mm < distance < 2.2  linear transition
```

默认切换中心是 `2 mm`，过渡半宽是 `0.2 mm`。

目标权重还会经过一阶低通：

```text
tau = 0.1 s
dt  = 0.05 s
alpha = dt / (tau + dt)

filtered_weight += alpha × (target_weight - filtered_weight)
```

最终选择规则为：

- fine valid：进入 `FINE_TRACKING`。
- fine 短暂丢失但仍在 coast 窗口内：进入 `FINE_COAST`，使用最近 fine 向量。
- fine 不可用但 coarse 足够精确：进入 `COARSE_FALLBACK`。
- coarse/fine 都不足：进入 `PRECISION_HOLD`，并可能触发有界回撤。

受限感知的四种 fallback state：

```text
FINE_TRACKING
FINE_COAST
COARSE_FALLBACK
PRECISION_HOLD
```

具体逻辑见：

- `restricted_perception/fusion.py`
- `restricted_perception/online.py`

## 6. 18 维受限向量的定义

`RestrictedObservation.policy_vector()` 将允许进入策略的字段排成固定
18 维 `float32` 向量。

| 索引 | 字段 | 维度 | 含义 |
|---:|---|---:|---|
| 0–2 | `coarse_goal_robot_xyz_mm` | 3 | coarse 目标相对针尖向量 |
| 3–5 | `fine_goal_robot_xyz_mm` | 3 | fine 目标相对针尖向量 |
| 6 | `coarse_valid` | 1 | coarse 是否有效 |
| 7 | `fine_valid` | 1 | fine 是否有效 |
| 8 | `coarse_age_s` | 1 | coarse 测量年龄 |
| 9 | `fine_age_s` | 1 | fine 测量年龄 |
| 10 | `coarse_sigma_mm` | 1 | coarse 不确定度 |
| 11 | `fine_sigma_mm` | 1 | fine 不确定度 |
| 12 | `switch_weight` | 1 | fine 融合权重 |
| 13 | `fallback_fine_tracking` | 1 | `FINE_TRACKING` one-hot |
| 14 | `fallback_fine_coast` | 1 | `FINE_COAST` one-hot |
| 15 | `fallback_coarse_fallback` | 1 | `COARSE_FALLBACK` one-hot |
| 16 | `fallback_precision_hold` | 1 | `PRECISION_HOLD` one-hot |
| 17 | `precision_sufficient` | 1 | 当前精度是否满足要求 |
|  | **总计** | **18** | |

向量的排列为：

```text
[
  coarse_x, coarse_y, coarse_z,
  fine_x, fine_y, fine_z,
  coarse_valid, fine_valid,
  coarse_age_s, fine_age_s,
  coarse_sigma_mm, fine_sigma_mm,
  switch_weight,
  fallback_fine_tracking,
  fallback_fine_coast,
  fallback_coarse_fallback,
  fallback_precision_hold,
  precision_sufficient
]
```

## 7. 无效值和 one-hot 规则

如果 coarse 或 fine 测量无效：

```text
goal vector    = [0, 0, 0]
valid          = 0
age            = -1
sigma          = -1
```

四个 fallback state 采用 one-hot：

```text
FINE_TRACKING      → [1,0,0,0]
FINE_COAST         → [0,1,0,0]
COARSE_FALLBACK    → [0,0,1,0]
PRECISION_HOLD     → [0,0,0,1]
```

任一时刻四维中恰好一维为 1。

## 8. 哪些信息不进入受限向量

以下信息不进入 18 维策略向量：

- 仿真目标真值坐标
- 目标世界绝对位置
- coarse/fine audit bias
- 真值针尖位置
- action mask
- 回撤命令
- 教师动作
- 仿真 geom ID
- 隐藏几何重渲染结果
- 真值终点误差

这些字段可以保存在 audit 数据中，但不属于冻结的策略输入 allowlist。

`state` 和 `kinematic_tip_robot_mm` 虽然属于 policy feature allowlist，
但 `policy_vector()` 不直接包含它们。策略网络通过单独的 3 维 A/B/C 输入
接收机器人状态。

## 9. 视觉特征与受限向量的拼接

Restricted 策略的基础输入是：

```text
visual_feature      512
state_ABC             3
restricted_vector    18
```

总维度：

```text
512 + 3 + 18 = 533
```

如果启用速度输入，再增加 3 维：

```text
512 + 3 + 18 + 3 = 536
```

当前选中的 `dagger_only` 和 residual-distilled checkpoint 配置为：

```text
vision_dim     = 512
state_dim      = 3
restricted_dim = 18
use_velocity   = false
```

因此当前输入为：

```text
[visual_feature 512, state_ABC 3, restricted_vector 18]
→ [533]
```

拼接顺序为：

```text
[ image_feature[512] | state_ABC[3] | restricted_feature[18] ]
```

### 9.1 为什么不能直接送入 MLP

联合特征各列的数值范围和物理含义不同：

```text
visual feature        ResNet 激活，范围由网络学习
A/B/C                 degree / degree / mm
coarse/fine vector    mm
valid                 0 或 1
age                   s
sigma                 mm
switch weight         0 到 1
fallback one-hot      0 或 1
precision sufficient  0 或 1
```

如果直接拼接，数值方差大的列可能主导第一层线性层，造成训练不稳定，也无法
让不同字段共享合理的梯度尺度。

### 9.2 按列计算训练集统计量

标准化统计量只从训练联合特征计算：

```python
feature_mean = train_features.mean(axis=0)
feature_std = train_features.std(axis=0)
```

这里不是计算一个全局标量，而是对 533 个输入维度分别计算：

```text
feature_mean.shape = [533]
feature_std.shape  = [533]
```

每一列有独立的均值与标准差：

```text
视觉维度 0       → 独立 mean/std
视觉维度 1       → 独立 mean/std
...
视觉维度 511     → 独立 mean/std
A/B/C 维度       → 各自独立 mean/std
受限向量 18 维   → 各自独立 mean/std
```

### 9.3 标准差下限

某些字段可能长期恒定，例如某个 one-hot 在当前训练集中几乎不变。此时标准差
可能接近零。

代码使用：

```python
feature_std[feature_std < 1e-6] = 1.0
```

这样避免除零或把极小数值变化放大到不稳定范围。

### 9.4 逐元素标准化

对 batch 中每个样本、每个维度执行：

```text
normalized[i, j]
  = (feature[i, j] - feature_mean[j]) / feature_std[j]
```

形状保持不变：

```text
[B,533] → [B,533]
```

例如，如果某一列 `C_mm` 的训练均值是 `10 mm`、标准差是 `5 mm`，当前值是
`12 mm`：

```text
(12 - 10) / 5 = 0.4
```

如果 `fine_valid` 的训练均值是 `0.8`、标准差是 `0.4`：

```text
fine_valid = 1 → (1 - 0.8) / 0.4 = 0.5
fine_valid = 0 → (0 - 0.8) / 0.4 = -2.0
```

每个视觉特征维度也会单独标准化，而不是对整个 512 维视觉向量使用一个均值。

### 9.5 统计量来自哪些样本

Stage 2 的均值/方差来自当前训练集联合后的全部行：

```text
基础 recovery 数据
+ DAgger 数据
→ 拼接 vision/state/restricted
→ 按列计算 mean/std
```

验证集只使用训练集统计量，不会重新计算自己的均值/方差，避免验证信息泄漏。

DAgger replay weight 只影响 `WeightedRandomSampler` 的抽样频率，不会改变
`feature_mean` 和 `feature_std` 的计算方式。统计量是未加权按行平均。

如果启用速度输入，3 个速度维度也参与统计，总维度变为 536。

### 9.6 训练和推理必须使用同一组统计量

训练结束后，以下四个数组写入 checkpoint：

```text
feature_mean
feature_std
action_mean
action_std
```

闭环推理加载 checkpoint 后恢复相同的 buffer：

```python
model.set_normalization(
    feature_mean=...,
    feature_std=...,
    action_mean=...,
    action_std=...,
)
```

推理时不会根据当前 episode 或当前 batch 重新估计统计量，否则同一输入在不同
时刻可能得到不同标准化结果。

### 9.7 模型内部执行标准化

在线评估代码负责拼接原始特征，标准化在模型 `forward()` 内部执行：

```python
normalized = (features - self.feature_mean) / self.feature_std
```

因此输入到第一层 MLP 的是：

```text
[B,533] 原始联合特征
→ [B,533] 标准化联合特征
→ Linear(533→512)
```

### 9.8 输入标准化和 LayerNorm 的区别

两者不是同一个操作：

- 输入标准化：使用训练集的固定 533 维均值/方差，对所有样本逐列使用同一组参数。
- LayerNorm：在每次前向传播中，对单个样本的 512 维隐藏激活做归一化。

网络顺序为：

```text
固定输入标准化
→ Linear 533→512
→ SiLU
→ LayerNorm 512
→ Linear 512→512
```

LayerNorm 不会替代输入标准化。

### 9.9 标准化不裁剪离群值

当前没有对标准化后的特征执行 clip。如果闭环出现训练分布外输入，例如很大的
受限目标向量，标准化结果可能明显超出训练范围。

这种情况下主要由后续 LayerNorm、网络权重和动作限幅维持数值稳定，而不是靠
输入标准化裁剪。后续如果发现明显分布偏移，可以考虑增加固定范围的输入截断，
但当前实现没有这样做。

### 9.10 动作标准化是另一套参数

动作输出也有自己的统计量：

```text
action_mean
action_std
```

训练损失在标准化动作空间计算：

```text
predicted_normalized
  = (predicted - action_mean) / action_std

target_normalized
  = (target - action_mean) / action_std
```

网络输出后再反标准化：

```text
action = predicted_normalized × action_std + action_mean
```

特征标准化控制网络输入尺度，动作标准化控制网络输出和回归损失尺度，两者不能
混用。

### 9.11 Stage 3 residual 如何使用统计量

Stage 3 不重新拟合一套 feature mean/std，而是直接复用冻结 base Stage 2
checkpoint 的 normalization：

```text
base checkpoint normalization
→ base 网络输入标准化
→ residual 网络输入标准化
```

基础动作和残差都基于同一标准化联合特征：

```text
base_action = base_policy(normalized_joint_feature)
residual = tanh(residual_head(normalized_joint_feature))
          × residual_limit
```

因此 distillation 阶段不会因为重新计算统计量而改变 base policy 的输入空间。

## 10. Stage 2 策略网络

### 10.1 共享输入

Stage 2 和 Stage 3 使用同一个原始联合特征：

```text
joint_feature = [
    vision_feature[512],
    state_ABC[3],
    restricted_vector[18]
]
```

当前 `use_velocity=false`，形状为：

```text
[B,533]
```

Stage 2 内部使用训练集固定统计量做逐元素标准化：

```python
normalized = (features - feature_mean) / feature_std
```

### 10.2 网络结构

当前选中的 Stage 2 `dagger_only` 网络为：

```text
[B,533] raw joint feature
        │
        ▼
(feature - feature_mean) / feature_std
        │
        ▼
[B,533] normalized joint feature
        │
        ▼
Linear(533→512)
        │
        ▼
SiLU
        │
        ▼
LayerNorm(512)
        │
        ▼
Linear(512→512)
        │
        ▼
SiLU
        │
        ▼
Linear(512→3)
        │
        ▼
[B,3] normalized action head
```

### 10.3 各层作用

第一层：

```text
Linear(533→512)
```

将视觉、A/B/C 和 coarse/fine 融合信息投影到统一的 512 维隐藏空间。

```text
SiLU
```

非线性激活，避免网络退化为单一线性映射。

```text
LayerNorm(512)
```

对每个样本的 512 维隐藏激活做归一化，稳定训练。它和输入阶段的固定
`feature_mean/feature_std` 标准化不同。

第二层：

```text
Linear(512→512)
```

在隐藏空间内进一步组合特征。

输出层：

```text
Linear(512→3)
```

输出三个标准化动作值：

```text
[dA_norm, dB_norm, dC_norm]
```

### 10.4 参数数量

Stage 2 分支约为：

```text
Linear 533→512: 533 × 512 + 512 = 273,408
LayerNorm 512:  512 × 2          =   1,024
Linear 512→512: 512 × 512 + 512 = 262,656
Linear 512→3:   512 × 3 + 3     =   1,539
```

合计约：

```text
538,627 个参数
```

SiLU 没有可训练参数。

### 10.5 动作反标准化

网络直接输出的三个数是标准化动作，不是最终 degree/mm 动作。

反标准化为：

```text
base_action
  = action_head × action_std + action_mean
```

得到：

```text
base_action = [dA_base, dB_base, dC_base]
```

输出是 50 ms 控制周期的增量动作，不是绝对 A/B/C 坐标。

### 10.6 训练监督

当前选中的 `dagger_only` Stage 2 使用基础 recovery 数据和教师标注的 DAgger
数据联合训练。

损失在标准化动作空间中计算：

```text
prediction_norm
  = (base_action - action_mean) / action_std

target_norm
  = (teacher_action - action_mean) / action_std

loss = SmoothL1(prediction_norm, target_norm)
```

DAgger replay 通过 `WeightedRandomSampler` 提高教师纠正样本的采样频率。

### 10.7 Stage 2 输出边界

Stage 2 的 `Stage2ActionPolicy` 内部保留了距离缩放限幅接口，但当前
`dagger_only` 配置为：

```text
residual_distance_scale = None
use_stop_head = False
```

因此当前 Stage 2 不启用内部距离缩放限幅和 stop head。

最终动作限幅在闭环执行器中进行：

```text
A: ±0.2898°
B: ±0.1443°
C: ±0.042712 mm
```

## 11. Stage 3 residual-distilled 策略

Stage 3 residual-distilled 不替换 Stage 2，而是在冻结 Stage 2 的基础上增加
一个小型残差头。

### 11.1 双分支总览

两分支接收同一个原始联合特征，并使用同一组 Stage 2 标准化参数：

```text
                  joint_feature [B,533]
                           │
                           ▼
          (feature - base_feature_mean)
                  / base_feature_std
                           │
                           ▼
             normalized_joint [B,533]
                    │             │
                    ▼             ▼
          Stage 2 base      Stage 3 residual
         533→512→512→3       533→256→256→3
                    │             │
                    ▼             ▼
              base_action    bounded residual
                    │             │
                    └──────┬──────┘
                           ▼
                    final_action
```

### 11.2 基础分支

Stage 3 加载选中 Stage 2 checkpoint：

```python
model.base.load_state_dict(base_checkpoint["model_state_dict"])
model.base.set_normalization(**base_checkpoint["normalization"])
model.base.eval()
```

随后冻结全部 base 参数：

```python
for parameter in model.base.parameters():
    parameter.requires_grad_(False)
```

所以 Stage 3 训练期间基础分支只做前向推理，不更新权重。

基础分支输出：

```text
base_action = selected_student(
    visual_feature,
    state_ABC,
    restricted_vector
)
```

网络展开为：

```text
normalized_joint [B,533]
→ Linear 533→512
→ SiLU
→ LayerNorm
→ Linear 512→512
→ SiLU
→ Linear 512→3
→ action 反标准化
→ base_action [B,3]
```

### 11.3 残差分支

残差头结构为：

```text
normalized_joint [B,533]
→ Linear 533→256
→ SiLU
→ Linear 256→256
→ SiLU
→ Linear 256→3
→ tanh
→ 乘 residual_limit
→ correction [B,3]
```

残差头参数约为：

```text
Linear 533→256: 533 × 256 + 256 = 136,704
Linear 256→256: 256 × 256 + 256 =  65,792
Linear 256→3:   256 × 3 + 3     =     771
```

合计约：

```text
203,267 个参数
```

### 11.4 tanh 有界残差

残差头最后输出三个任意实数，先经过 `tanh`：

```text
tanh_output ∈ [-1, 1]
```

再逐轴乘以残差上限：

```text
correction_A ∈ [-0.005°, +0.005°]
correction_B ∈ [-0.003°, +0.003°]
correction_C ∈ [-0.001 mm, +0.001 mm]
```

因此无论残差网络输出多大，最终修正都不会超过这些范围。

### 11.5 零初始化

残差头最后一层的权重和 bias 初始化为零：

```python
nn.init.zeros_(self.residual[-1].weight)
nn.init.zeros_(self.residual[-1].bias)
```

因此训练开始时：

```text
correction = 0
final_action = base_action
```

Stage 3 从完全等价于 Stage 2 的策略开始，再逐渐学习小幅修正。

### 11.6 残差训练目标

Stage 3 使用七样本 median 教师动作和冻结 base 动作构造残差监督：

```text
residual_target
  = teacher_action - base_action
```

训练只优化残差头：

```text
loss = SmoothL1(
    bounded_residual(normalized_joint),
    residual_target
)
```

Base 参数固定，因此不会为了拟合蒸馏教师而破坏已经验证过的 Stage 2 行为。

### 11.7 最终动作合成

推理时：

```text
final_action
  = base_action + correction
```

展开为：

```text
final_dA = dA_base + correction_A
final_dB = dB_base + correction_B
final_dC = dC_base + correction_C
```

残差只负责终端小修正，基础接近行为仍由 Stage 2 主导。

由于误差上限很小，Stage 3 不能修复大范围方向错误。如果 Stage 2 对目标区域
给出明显错误的基础动作，残差头最多只能做 `0.005°/0.003°/0.001 mm` 的修正。

### 11.8 训练与推理模式

训练时：

- Stage 2 base 固定为 eval。
- base 参数 `requires_grad=False`。
- residual head 为 train。
- 只更新 residual 参数。

推理时：

- 整个策略切换为 eval。
- base 和 residual 都做一次前向。
- 没有随机扩散采样。
- 单步输出最终 3 维增量动作。

### 11.9 输出后的执行顺序

网络输出完成后还要经过：

```text
1. 如果 fallback_state 是 PRECISION_HOLD，动作置零
2. 对 A/B/C 做外部动作限幅
3. degree/mm 转换为 SI 单位
4. 当前 A/B/C + 增量动作
5. MotorBackend 执行 50 ms
6. 实际编码器反馈进入下一周期
```

## 12. Restricted 完整时序

每个 50 ms 周期：

```text
1. MotorBackend 推进一个控制周期
2. 读取实际编码器反馈
3. 得到 A/B/C [deg,deg,mm]
4. 正运动学计算 current_tip_robot_mm
5. 获取 coarse 目标机器人坐标
6. 获取 fine 相机坐标向量
7. 将 fine 旋转到机器人坐标
8. 补偿 fine 延迟期间的针尖运动
9. 根据距离和 valid 状态融合 coarse/fine
10. 生成 18 维受限向量
11. 获取 RGB 图像并编码为 512 维视觉特征
12. 拼接 [视觉512, A/B/C3, 受限18]
13. 特征标准化
14. 策略网络输出 [dA,dB,dC]
15. 执行动作并进入下一周期
```

## 13. 与主策略 v7/v8 的区别

主策略 v7/v8 不使用 18 维 restricted vector。

主策略每帧状态侧输入为：

```text
标准化 A/B/C             3
OCT 单位方向             3
标准化 log(distance)     1
OCT valid                1
guided base action       3
```

合计：

```text
3 + 3 + 1 + 1 + 3 = 11
```

两帧视觉编码后的融合输入为：

```text
2 × [视觉特征 512 + 状态编码 64]
= 1152
→ 512
→ 256
```

对比：

- v7/v8 主策略：RGB + A/B/C + OCT 方向/距离/valid + guided action。
- Restricted 策略：视觉特征 + A/B/C + 18 维 coarse/fine/valid/uncertainty/fallback 向量。

A/B/C 在两条链路中都是当前实际机器人状态；18 维受限向量只属于
restricted-perception 研究链路。

## 相关实现

- 受限观测数据结构：
  [`restricted_perception/schema.py`](../restricted_perception/schema.py)
- 受限特征字段名与 sidecar：
  [`restricted_perception/sidecar.py`](../restricted_perception/sidecar.py)
- 在线 coarse/fine 适配：
  [`restricted_perception/online.py`](../restricted_perception/online.py)
- Coarse/fine 融合和 fallback：
  [`restricted_perception/fusion.py`](../restricted_perception/fusion.py)
- Stage 1 restricted policy：
  [`restricted_perception/policy.py`](../restricted_perception/policy.py)
- Stage 2 policy：
  [`restricted_perception/stage2_policy.py`](../restricted_perception/stage2_policy.py)
- Restricted 闭环评估：
  [`scripts/research/restricted_perception/evaluate_restricted_policy.py`](../scripts/research/restricted_perception/evaluate_restricted_policy.py)
