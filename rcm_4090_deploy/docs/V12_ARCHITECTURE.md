# v12 输入输出与当前架构

本文描述截至 2026-11-15 的 v12 OCT-aware Diffusion Policy，包括数据构成、
策略输入输出、网络结构、闭环执行方式、关键数学公式和当前实验阻塞点。

本文验证的是 MuJoCo 仿真软件行为，不代表真实设备、物理 OCT 或临床精度。
当前所谓 OCT 输入来自注册 RGB-D 可见深度的相对导航量，不是物理 OCT。

## 1. 架构摘要

v12 当前链路可以概括为：

> 两帧显微 RGB + 两帧 A/B/C 状态 + 两帧 RGB-D 相对导航向量
> → ResNet18 条件编码 → Diffusion/DDIM 去噪
> → 8 步 A/B/C 增量动作块 → 动作安全限幅 → MotorBackend 闭环执行。

```text
Strict RGB-D sensor                         MotorBackend
  ├─ microscope RGB                         └─ actual A/B/C
  └─ visible-depth navigation
              │
              ▼
        two-frame observation
  ┌──────────────────────────────────────┐
  │ image [2, 3, 224, 224]               │
  │ state [2, 3]                         │
  │ oct   [2, 4]                         │
  └──────────────────────────────────────┘
              │
              ▼
  ResNet18 vision encoder       state + OCT MLP
       [2, 512]                    [2, 64]
              └──────── concat ─────────┘
                         │
                         ▼
                   [2, 576] → 1152
                         │
                 1152 → 512 → 256
                         │
                         ▼
       Diffusion MLP conditioned on 256-D vector
                         │
                 100-step DDIM sampling
                         │
                action chunk [8, 3]
                         │
         7 samples → per-element median
                         │
               execute first action
                         │
          clip / state safety / MotorBackend
```

网络实现位于
[train_diffusion_policy.py](../lerobot_training/train_diffusion_policy.py)，
闭环执行位于 [eval_policy.py](../lerobot_training/eval_policy.py)。

## 2. 输入与输出契约

### 2.1 训练样本

当前 v12 checkpoint 使用：

| 字段 | 形状 | 单位/含义 |
|---|---|---|
| `observation.image` | `[2, 3, 224, 224]` | 两帧 ImageNet 归一化 RGB |
| `observation.state` | `[2, 3]` | `[A_deg, B_deg, C_mm]` |
| `observation.oct` | `[2, 4]` | `[dx_mm, dy_mm, dz_mm, valid]` |
| `action` | `[8, 3]` | 未来八步 `[dA_deg, dB_deg, dC_mm]` |

其中 `valid` 是浮点形式的有效性标记：

$$
v_t =
\begin{cases}
1, & \text{RGB-D relative navigation is finite and available},\\
0, & \text{otherwise}.
\end{cases}
$$

相对导航定义为相机光学坐标系下的目标减针尖：

$$
\Delta\mathbf{p}^{cam}_t
= \mathbf{R}_{cam\leftarrow world}
\left(\mathbf{p}^{world}_{target}-\mathbf{p}^{world}_{tip,t}\right).
$$

训练数据将其乘以 `1000`，以 mm 存储。元数据将 OCT 描述为三维修正量
`[3]`；运行时会额外拼接一个 `valid` 维，因此实际网络输入为四维。

### 2.2 部署输入

闭环每个控制周期执行以下步骤：

1. 从严格 RGB-D sensor 获取当前 RGB 和相对导航量。
2. 从 MotorBackend 获取当前实际 A/B/C。
3. 将当前观测与上一帧组成 `T=2` 历史。
4. 每帧 RGB 调整为 `224×224` 并进行 ImageNet 归一化。
5. 将 `[dx,dy,dz]` 从 m 转为 mm，并附加有效性标记。

元数据还声明了 `task` 输入字段，但当前
`SimpleDiffusionPolicy.forward()` 没有显式 task 向量参数，也没有把目标
世界坐标作为条件。目标信息只通过 RGB 和 `target_minus_tip` 相对向量间接
进入策略。

以下字段不进入策略：

- 目标世界坐标；
- 终点真值误差；
- 执行来源；
- fallback 元数据；
- 源 episode 标识。

完整数据契约见
[metadata.json](../data/lerobot_v12_oct_dagger1_fixed_224/metadata.json)。

### 2.3 模型输出

模型输出为未来八步增量动作：

$$
\hat{\mathbf{A}}_t
= \left[\hat{\mathbf{a}}_t,\hat{\mathbf{a}}_{t+1},\ldots,
\hat{\mathbf{a}}_{t+7}\right]
\in \mathbb{R}^{8\times 3}.
$$

每个动作轴为：

$$
\hat{\mathbf{a}}_{t+k}
=
\begin{bmatrix}
\Delta \widehat{A}_{t+k}\\
\Delta \widehat{B}_{t+k}\\
\Delta \widehat{C}_{t+k}
\end{bmatrix}
\quad
\left(\deg,\deg,\mathrm{mm}\right).
$$

标准 v12 闭环测试预测八步，但只执行第一步，下一步重新推理。配置支持连续
执行多个动作，但当前 20 目标评估使用 `n_action_execute_steps=1`。

## 3. 数据链路

### 3.1 基础零终点误差数据

基础 v12 数据从既有专家轨迹重新采集或重标注，终点目标不是旧名义目标，
而是稳定源终点的针尖世界坐标：

$$
\mathbf{p}^{*}_{world}
= \mathbf{p}^{world}_{tip,terminal}.
$$

因此终点定义满足逐分量位相等和零欧氏终点误差：

$$
\left\|\mathbf{p}^{world}_{tip,terminal}
-\mathbf{p}^{*}_{world}\right\|_2
= 0.0\ \mu\mathrm{m}.
$$

结果：

| 项目 | 数量 |
|---|---:|
| episode | 608 |
| frame | 187,440 |

### 3.2 OCT-aware 派生集

从每个 episode 的 `oct_observations.jsonl` 读取逐帧
`navigation_depth.delta_xyz_m`，按控制帧索引对齐并转为 mm：

```python
oct_feature = np.r_[
    self.oct_navigation[hist_idx],
    float(self.oct_navigation_valid[hist_idx]),
].astype(np.float32)
obs_oct.append(torch.from_numpy(oct_feature))
```

在基础 187,440 帧中：

| 项目 | 数量 |
|---|---:|
| 有效 OCT 帧 | 182,514 |
| OCT 有效率 | 97.372% |

### 3.3 在线 DAgger round 1

round 1 使用基础 epoch 3 checkpoint、严格 RGB-D 教师、
`beta=0.3` 和 MotorBackend。执行策略为：

```python
policy_action = np.clip(policy_action, -action_limit, action_limit)

if hold_remaining > 0:
    executed_action = np.zeros(3)
    executor = "hold"
elif takeover_left > 0:
    executed_action = np.clip(
        teacher_action, -action_limit, action_limit
    )
    executor = "teacher"
elif teacher_valid and rng.random() < beta and not policy_action_limited:
    executed_action = np.clip(
        teacher_action, -action_limit, action_limit
    )
    executor = "beta_teacher"
else:
    executed_action = policy_action
    executor = "policy"
```

round 1 共 40 集、14,201 帧，其中 38 集通过严格终点检查：

| 实际执行者 | 帧数 |
|---|---:|
| teacher | 13,031 |
| hold | 798 |
| exact terminal | 304 |
| policy | 50 |
| beta teacher | 18 |

策略动作触及限幅的帧数为 12,654。该分布说明本轮数据具有有效教师终点标签，
但学生独立执行占比很低，不能把 38/40 的采集成功率直接解释为学生策略成功率。

### 3.4 最终合并数据

DAgger 成功集与基础 608 集合并后：

| 项目 | 数量 |
|---|---:|
| episode | 646 |
| frame | 200,943 |
| 有效 OCT 帧 | 195,675 |
| 训练窗口 | 195,775 |

最终数据位于
[lerobot_v12_oct_dagger1_fixed_224](../data/lerobot_v12_oct_dagger1_fixed_224)。

## 4. 网络结构

### 4.1 训练样本构建代码

数据集从连续轨迹中取出两帧历史和八帧未来动作。历史和未来必须属于同一
episode：

```python
for j in range(self.n_obs_steps):
    hist_idx = i - (self.n_obs_steps - 1 - j)
    # load image, state and OCT for hist_idx

actions = [
    torch.from_numpy(self.actions[i + j])
    for j in range(self.n_action_steps)
]

sample = {
    "observation.image": obs_images,   # [2, 3, 224, 224]
    "observation.state": obs_states,   # [2, 3]
    "observation.oct": obs_oct,        # [2, 4]
    "action": torch.stack(actions),    # [8, 3]
}
```

对应代码位于
[RCMDataset.__getitem__](../lerobot_training/train_diffusion_policy.py)。

### 4.2 视觉编码器

v12 checkpoint 没有保存特殊 `vision_mode`，因此使用默认 `global` 模式。
该模式使用 ImageNet 预训练 ResNet18，对 `224×224` 图像编码并经全连接层
输出 512 维：

```python
self.encoder = nn.Sequential(*children[:-1])
self.fc = nn.Linear(512, output_dim)

feat = self.encoder(x)
feat = feat.squeeze(-1).squeeze(-1)
feat = self.fc(feat)
```

两帧图像分别编码：

$$
\mathbf{f}^{img}_t = E_{rgb}(\mathbf{I}_t)
\in \mathbb{R}^{512}.
$$

### 4.3 状态与 OCT 编码

state 和 OCT 先标准化，再拼接为一个七维向量：

$$
\mathbf{u}_t
=
\left[
\frac{\mathbf{s}_t-\mu_s}{\sigma_s},
\frac{\mathbf{o}^{xyz}_t-\mu_o}{\sigma_o},
v_t
\right]
\in \mathbb{R}^{7}.
$$

其中 state 和相对导航均为三轴。七维向量经过线性层得到 64 维：

$$
\mathbf{f}^{state}_t
=
\mathrm{ReLU}
\left(\mathbf{W}_s\mathbf{u}_t+\mathbf{b}_s\right)
\in \mathbb{R}^{64}.
$$

对应代码：

```python
normalized_state = (
    obs_state - self.state_mean.reshape(1, 1, -1)
) / self.state_std.reshape(1, 1, -1)

oct_xyz = (
    oct_input[..., :3] - self.oct_mean[:3].reshape(1, 1, 3)
) / self.oct_std[:3].reshape(1, 1, 3)

state_input = torch.cat(
    [normalized_state, oct_xyz, oct_input[..., 3:4]],
    dim=-1,
)
state_feat = self.state_encoder(state_input)
```

当前 v12 没有启用显式速度或上一动作输入，因此
`include_velocity=false`、`include_previous_action=false`。

### 4.4 条件融合

每帧视觉和状态特征拼接：

$$
\mathbf{f}_t
=
\left[\mathbf{f}^{img}_t;\mathbf{f}^{state}_t\right]
\in \mathbb{R}^{576}.
$$

两帧展开并融合：

```python
obs_feat = torch.cat([vision_feat, state_feat], dim=-1)
obs_feat = obs_feat.reshape(B, -1)
condition = self.obs_fusion(obs_feat)
```

数学表达为：

$$
\mathbf{c}_t
=
\mathbf{W}_{c2}
\mathrm{ReLU}
\left(
\mathbf{W}_{c1}
\left[\mathbf{f}_{t-1};\mathbf{f}_t\right]
+\mathbf{b}_{c1}
\right)
+\mathbf{b}_{c2}
\in \mathbb{R}^{256}.
$$

### 4.5 Diffusion 去噪头

动作块被展平为：

$$
\mathbf{x}_0
= \operatorname{vec}(\mathbf{A}_t)
\in \mathbb{R}^{24}.
$$

去噪网络输入为当前噪声动作、条件向量和时间步：

$$
\mathbf{z}
=
\left[
\mathbf{x}_\tau;
\mathbf{c}_t;
\tau/T
\right]
\in \mathbb{R}^{24+256+1}
=\mathbb{R}^{281}.
$$

当前网络为三层 MLP：

```python
self.noise_pred_net = nn.Sequential(
    nn.Linear(action_dim * n_action_steps + 256 + 1, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, action_dim * n_action_steps),
)
```

即：

$$
\widehat{\boldsymbol{\epsilon}}_\theta
=
\mathrm{MLP}_\theta
\left(\mathbf{x}_\tau,\mathbf{c}_t,\tau\right)
\in \mathbb{R}^{24}.
$$

模型总参数量为约 12.58M。

## 5. 训练与采样数学

### 5.1 标准化

state、OCT 和动作均按训练集统计量标准化：

$$
\bar{\mathbf{s}}
=\frac{\mathbf{s}-\mu_s}{\sigma_s},
\qquad
\bar{\mathbf{o}}
=\frac{\mathbf{o}^{xyz}-\mu_o}{\sigma_o},
$$

$$
\bar{\mathbf{A}}
=\frac{\mathbf{A}-\mu_a}{\sigma_a}.
$$

动作标准差来自最终 v12 合并训练集：

```text
action_std =
[0.04901096, 0.03181447, 0.01315141]
```

分别对应 `dA_deg`、`dB_deg`、`dC_mm`。

### 5.2 扩散前向过程

使用 cosine noise schedule。定义
\(\alpha_\tau=1-\beta_\tau\) 和
\(\bar{\alpha}_\tau=\prod_{i=0}^{\tau}\alpha_i\)，前向加噪为：

$$
\mathbf{x}_\tau
=
\sqrt{\bar{\alpha}_\tau}\mathbf{x}_0
+
\sqrt{1-\bar{\alpha}_\tau}\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

### 5.3 训练损失

当前训练目标预测噪声，并按动作轴加权：

$$
\mathcal{L}_\theta
=
\frac{1}{B\cdot8\cdot3}
\sum_{b=1}^{B}
\sum_{k=1}^{8}
\sum_{j=1}^{3}
\omega_j
\left(
\epsilon_{b,k,j}
-
\widehat{\epsilon}_{\theta,b,k,j}
\right)^2.
$$

当前 v12 最终训练没有特殊 `loss_weights`，因此三轴默认权重相同：

$$
\omega_A=\omega_B=\omega_C=1.
$$

### 5.4 DDIM 推理

推理从纯噪声开始，并执行 100 次反向去噪：

```python
actions = torch.randn(B, n_action_steps * 3)

for t in reversed(range(num_diffusion_iters)):
    noise_pred = self.noise_pred_net(
        torch.cat([actions, condition, timestep_embed], dim=-1)
    )

    pred_x0 = (
        actions - torch.sqrt(1 - alpha_t) * noise_pred
    ) / torch.sqrt(alpha_t)
    pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)

    actions = (
        torch.sqrt(alpha_prev) * pred_x0
        + torch.sqrt(1 - alpha_prev) * noise_pred
    )
```

先估计干净动作：

$$
\widehat{\mathbf{x}}_0
=
\frac{
\mathbf{x}_\tau
-\sqrt{1-\bar{\alpha}_\tau}
\widehat{\boldsymbol{\epsilon}}_\theta
}{
\sqrt{\bar{\alpha}_\tau}
}.
$$

标准化空间中的预测被裁剪：

$$
\widehat{\mathbf{x}}_0
\leftarrow
\operatorname{clip}
\left(\widehat{\mathbf{x}}_0,-4,4\right).
$$

DDIM 更新为：

$$
\mathbf{x}_{\tau-1}
=
\sqrt{\bar{\alpha}_{\tau-1}}
\widehat{\mathbf{x}}_0
+
\sqrt{1-\bar{\alpha}_{\tau-1}}
\widehat{\boldsymbol{\epsilon}}_\theta.
$$

### 5.5 动作反标准化

最终动作块恢复为物理单位：

$$
\widehat{\mathbf{A}}_t
=
\mathbf{A}^{normalized}_t\odot\sigma_a+\mu_a.
$$

对应代码：

```python
actions = actions.reshape(B, self.n_action_steps, 3)
actions = (
    actions * self.action_std.reshape(1, 1, -1)
    + self.action_mean.reshape(1, 1, -1)
)
```

### 5.6 多次采样聚合

v12 闭环评估对同一观测独立采样七次：

$$
\left\{
\widehat{\mathbf{A}}^{(1)}_t,
\ldots,
\widehat{\mathbf{A}}^{(7)}_t
\right\}.
$$

对每个动作位置和每个轴逐元素取中位数：

$$
\widehat{\mathbf{A}}^{median}_{t,k,j}
=
\operatorname{median}
\left(
\widehat{A}^{(1)}_{t,k,j},
\ldots,
\widehat{A}^{(7)}_{t,k,j}
\right).
$$

然后只取当前第一步：

$$
\mathbf{a}^{policy}_t
=
\widehat{\mathbf{A}}^{median}_{t,0}.
$$

## 6. 闭环执行与安全层

### 6.1 动作与状态限制

当前动作上限来自训练数据各轴最大绝对值：

$$
\mathbf{a}_{max}
=
\begin{bmatrix}
0.2898 & 0.1443 & 0.0427
\end{bmatrix}
\quad
(\deg,\deg,\mathrm{mm}).
$$

策略动作首先裁剪：

$$
\mathbf{u}_t
=
\operatorname{clip}
\left(
\mathbf{a}^{policy}_t,
-\mathbf{a}_{max},
\mathbf{a}_{max}
\right).
$$

目标状态为：

$$
\mathbf{s}^{raw}_{t+1}
=
\mathbf{s}_t+\mathbf{u}_t.
$$

状态边界为：

$$
A\in[-30,30]^\circ,\qquad
B\in[-30,30]^\circ,\qquad
C\in[0,50]\ \mathrm{mm}.
$$

### 6.2 安全缩放

若完整动作导致场景拒绝，则依次尝试：

$$
\lambda\in
\left\{
1.0,0.5,0.25,0.125,0.0625,0
\right\}.
$$

实际执行动作写为：

$$
\mathbf{u}^{applied}_t
=
\lambda^{*}\mathbf{u}_t,
$$

其中 \(\lambda^{*}\) 是第一个使 `scene.set_task()` 成功的缩放系数。若连续
20 个控制周期发生缩放，评估以
`SAFETY_LIMITED_20_STEPS` 早停。

对应代码：

```python
action = np.clip(action, -action_limit, action_limit)
new_state_abc = state_abc + action

for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.0):
    trial_state = state_abc + scale * action
    trial_task = state_abc_mm_to_task(trial_state)
    try:
        scene.set_task(trial_task, motor_dt=control_period_s)
        applied_action = scale * action
        break
    except ValueError:
        continue
```

### 6.3 MotorBackend

推理增量转为绝对控制目标后，由整数编码器 MotorBackend 执行：

$$
\mathbf{encoder}_{target}
=
\operatorname{Quantize}
\left(
\mathbf{s}_t+\mathbf{u}^{applied}_t
\right).
$$

控制周期为 `0.05 s`，即 20 Hz。编码器到达目标并且速度低于停稳阈值后，
才允许计入成功。

### 6.4 成功条件

真值只用于评价，不用于选择动作。世界坐标误差为：

$$
e_t
=
\left\|
\mathbf{p}^{world}_{tip,t}
-
\mathbf{p}^{world}_{target}
\right\|_2.
$$

20 µm 成功条件为：

$$
e_t<0.020\ \mathrm{mm}
\quad\text{and}\quad
\left|\mathbf{v}_t\right|
\le
\left[0.05,0.05,0.01\right]
\quad
(\deg/\mathrm{s},\deg/\mathrm{s},\mathrm{mm}/\mathrm{s}),
$$

并连续保持三帧。

## 7. 当前实验结果

### 7.1 基础 v12 模型

基础 v12 epoch 3 checkpoint 在固定 development 20 目标上：

| 指标 | 结果 |
|---|---:|
| 20 µm 成功率 | `0/20` |
| 最终误差均值 | `22.495 mm` |
| 最终误差中位数 | `22.815 mm` |
| 每集最小误差均值 | `19.123 mm` |
| 触发安全限位的 episode | `19/20` |
| safety-limited steps | `4,016` |

结论：

- 不是 OCT 缺失或单位错误；OCT 输入有效率为 100%。
- 模型持续输出错误方向动作并触及 A/B/C 限幅。
- 失败属于闭环策略行为问题，不是输入管线崩溃。

证据见
[aggregate_summary.json](../outputs/evaluation_v12_oct_dev20_20261115/aggregate_summary.json)。

### 7.2 DAgger round 1 更新模型

从基础 epoch 3 checkpoint 继续训练四个 epoch，最佳 checkpoint 为 epoch 7：

| 项目 | 基础模型 | round 1 模型 |
|---|---:|---:|
| train loss | `0.150758` | `0.137604` |
| val loss | `0.144923` | `0.129817` |

离线 loss 改善不等于闭环成功。round 1 数据中的执行分布仍然明显偏向教师：

$$
\frac{N_{teacher}}{N_{total}}
=
\frac{13031}{14201}
\approx 91.76\%.
$$

学生独立执行比例仅为：

$$
\frac{N_{policy}}{N_{total}}
=
\frac{50}{14201}
\approx 0.35\%.
$$

策略动作触及限幅的比例为：

$$
\frac{12654}{14201}
\approx 89.11\%.
$$

### 7.3 最新闭环验证状态

round 1 checkpoint 的 20 目标配对闭环验证目录为
[evaluation_v12_oct_dagger1_fixed_dev20_20261115](../outputs/evaluation_v12_oct_dagger1_fixed_dev20_20261115)。

截至本文快照：

- 结果文件为 `IN_PROGRESS`；
- 只完成 1 集；
- 该集 `success=false`；
- 运行 204 步；
- 最终误差 `18.897 mm`；
- 早停原因 `SAFETY_LIMITED_20_STEPS`；
- 检查时没有对应评测进程在运行。

因此该目录只能作为中断的部分证据，不能作为完整 20 目标结果，也不能据此判断
round 1 模型是否优于基础模型。

## 8. 当前问题判断

当前主要问题不是数据格式，也不是训练代码无法运行，而是策略没有稳定学会闭环
制动和恢复：

1. 基础模型在接近目标前出现错误方向运动和动作饱和。
2. 教师可以完成严格零误差采集，但学生独立执行比例极低。
3. DAgger 数据可以减少状态分布缺口，但本轮数据仍然高度 teacher-heavy。
4. 离线 diffusion loss 改善不能约束闭环动作方向、制动和停稳行为。
5. 最新 round 1 checkpoint 尚无完整配对评测，性能结论仍未知。

当前尚未证明策略忽略了 OCT。现有证据只说明 OCT 已正确输入且有效；模型是否
真正利用 OCT，需要后续输入消融、条件敏感性测试和完整闭环对比才能确定。

## 9. 复现与证据入口

模型与训练日志：

- [round 1 checkpoint](../outputs/diffusion_policy_v12_oct_dagger1_fixed_20261115/best_model.pth)
- [training history](../outputs/diffusion_policy_v12_oct_dagger1_fixed_20261115/training_history.json)
- [training log](../outputs/diffusion_policy_v12_oct_dagger1_fixed_20261115/training.log)

基础模型闭环证据：

- [aggregate summary](../outputs/evaluation_v12_oct_dev20_20261115/aggregate_summary.json)
- [g1](../outputs/evaluation_v12_oct_dev20_20261115/g1/evaluation_results.json)
- [g2](../outputs/evaluation_v12_oct_dev20_20261115/g2/evaluation_results.json)
- [g3](../outputs/evaluation_v12_oct_dev20_20261115/g3/evaluation_results.json)
- [g4](../outputs/evaluation_v12_oct_dev20_20261115/g4/evaluation_results.json)

round 1 DAgger 证据：

- [fixed-target round 1 shards](../runs/20261115/online_dagger_v12/fixed_target_round1_shards)
- [pilot fixed reachable target](../runs/20261115/online_dagger_v12/pilot_fixed_reachable_target)

完整时间顺序实验记录见 [TEST_LOG.md](../TEST_LOG.md)。
