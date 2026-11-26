# kinematics.py G15 代码说明与模型交接

本文供其他模型或开发者在不阅读聊天记录的情况下理解代码。内容根据 2026-11-20 本地源码整理；公式描述软件行为，不代表实机定位精度或已辨识的物理动力学。

更新（2026-11-21）：当前项目已默认加载独立 G15 原文件快照 `mujoco_model/vendor/simulation_driver_g15/`，可用新进程环境变量 `RCM_DRIVER_VARIANT=g10` 回退。下文关于“当前 G10”的叙述记录整理时的状态。连续目标累积尚未接入策略评估器。

## 1. 文件与版本

分析对象：

```text
/home/him1/mujoco_bc/simulation_driver/ros_msg/src/robot_control/kinematics.py
```

源码 SHA-256：

```text
3465daf19fd3269466f90fd77c6626149ab4da8ddde612c8f3d9a12c77d06088
```

这份文件的 `CONTROL_GEAR = 15`。它提供几何、协议换算、限位检查与路点辅助函数；不实现电机响应、ROS 节点调度或控制周期。虽然导入 `ros_msg.msg.controlSignal` 以构造消息，但导入本身不会启动节点或发布命令。实际 ROS 环境需要构建生成的消息；离线测试可以使用明确标注的消息替身。

当前部署项目使用另一份固定驱动：

```text
/home/him1/mujoco_bc/rcm_4090_deploy/mujoco_model/vendor/simulation_driver/
```

截至本文整理时，该副本仍为 G10，来源锁定提交 `c2a3ffc5f0801706a48559b3f514b9d32cae0435`，29 个清单文件哈希匹配。加载器为 `mujoco_model/driver_loader.py`，仿真控制链为：

```text
MotorExecution → MotorBackend → load_driver() → vendor 中的 robot_control.kinematics
```

根目录这份 G15 文件的上传或存在，不会自动替换当前仿真的 G10 实现。

## 2. 必须先区分的变量与单位

| 变量 | 含义 | 单位/类型 |
|---|---|---|
| `robot_values = (rA, rB, rC, rE)` | 机器人编码器状态 | 整数编码器 count |
| `handle_values = (hA, hB, hC, hE)` | 协议累计手柄目标 | 消息字段为 int32 |
| `handle_delta = (dA, dB, dC, dE)` | 相对手柄目标变化量 | 整数 count；浮点接口可返回取整前的小数 |
| `position` / `junction` | 针具交接点 XYZ 反馈 | mm；与针尖位置不同 |
| `target_tip` / `delta_xyz` | 针尖目标/笛卡尔修正 | mm |
| `phi` | A 转角 | 方向公式内部使用 rad |
| `alpha` | B 连杆角 | `b_alpha_from_robot_encoder()` 返回 deg；姿态字典中的 `alpha` 为 rad |
| `theta` | 针轴相对模型 +Y 的夹角 | `alpha + 22°`；三角函数内部使用 rad |
| `tip_offset_mm` / `tip_offset_sign` | 针尖相对交接点的偏置长度/方向 | 默认 4 mm；旧针 +1，新针 −1 |

命令消息 `controlSignal` 的字段为四个 `int32 handleEncoder*` 和两个布尔方向标志。反馈 `Encoder` 另有四个 `robotEncoder*` 及 `float32 posX/Y/Z`。

**消息里的手柄字段是累计目标，不是每个收到的数据包都要再次累加的增量。** 例如当前累计目标 `hC=100`，需要增加 1 count，应发送 `101`；重复发送 `101` 不应持续增加位移。`stepped_values()` 才是在已有累计目标上添加增量。

`predicted_robot_encoders(robot_values, handle_delta)` 的输入则明确是相对某个机器人参考状态的增量。协议适配器需要保存会话锚点，将累计目标转换为相对该锚点的总增量；现有 `MotorBackend` 就采用这种语义。

## 3. 核心常量

```python
G = CONTROL_GEAR = 15
K = ENCODER_COUNTS_PER_MM = 4 * 1024 * 15.5817175
# K = 63822.71488 encoder counts/mm

DEFAULT_TIP_OFFSET_MM = 4.0
LEGACY_NEEDLE_OFFSET_SIGN = +1.0
NEW_NEEDLE_OFFSET_SIGN = -1.0
PROTOCOL_INITIAL_HANDLES = (0, 0, 0, 0)
```

G 是软件运动学换算增益。源码注释还提到设备 5 档的倍率为 7；这属于不同参数，设置 `CONTROL_GEAR=15` 不会设置设备档位。

| 常量 | 数值 | 实际含义 |
|---|---:|---|
| `MIN_CARTESIAN_STEP_UM` | 10 µm | 笛卡尔点动辅助函数允许的最小输入步长 |
| `MAX_CARTESIAN_STEP_UM` | 1000 µm | 点动输入最大步长 |
| `DEFAULT_CARTESIAN_STEP_UM` | 50 µm | 默认点动输入步长 |
| `INSERTION_STEP_MM` | 0.010 mm | 预设进退针距离，等于 10 µm |
| `ABSOLUTE_TARGET_TOL_MM` | 0.001 mm | 绝对目标相关逻辑的容差，等于 1 µm |
| `ABSOLUTE_TARGET_WAYPOINT_STEP_MM` | 0.300 mm | 默认路点分段距离 |
| `ABSOLUTE_TARGET_WAYPOINT_TOL_MM` | 0.0015 mm | 路点容差常量，等于 1.5 µm；具体使用由调用方决定 |
| `MAX_CARTESIAN_AXIS_CHANGE_DEG` | 8° | 相关换算/校验函数允许的单段针轴方向变化上限 |

10 µm 的点动输入限制、1 µm 的目标容差、一个手柄 count 的运动量、一个编码器 count 的分辨率是四个不同概念。底层 `cartesian_handle_delta_float()` 不受点动输入最小 10 µm 的统一限制。

## 4. 正运动学：编码器 → 针轴与针尖

A 编码器对应角度：

```text
phi_deg = rA × 360 / 200000
phi_rad = radians(phi_deg)
```

B 连杆的长度与角度换算：

```text
d_B_mm(alpha) = 98.48 − 60 cos(alpha)
                − sqrt(52² − (60 sin(alpha) − 16)²)

distance = rB / K
ad = 98.48 − distance
ac = hypot(ad, 16)
gamma_deg = degrees(atan2(16, ad))
beta_cos = (60² + ac² − 52²) / (2 × 60 × ac)
alpha_deg = degrees(acos(clamp(beta_cos, −1, 1))) + gamma_deg
```

`b_distance_from_alpha()` 的参数是 deg，内部先转换成 rad。

针轴方向与针尖：

```text
theta = radians(alpha_deg + 22)
u = (sin(theta) sin(phi), cos(theta), sin(theta) cos(phi))
u = normalize(u)

junction_radius_mm = rC / K − 14
tip_radius_mm = junction_radius_mm + tip_offset_sign × tip_offset_mm
model_tip_xyz_mm = tip_radius_mm × u
```

这里的模型坐标以 RCM 为参考。半径可以为负，因此逆解不能只取向量模长而忽略当前针轴方向。4 mm 是针尖相对交接点的固定偏置，并非针尖到 RCM 的固定距离。

正 C 增量使模型位置沿当前 `u` 方向变化；不应把它等同于固定世界坐标的 +Z 位移。A/B 的角度步长也不能直接当成固定微米位移，针尖位移取决于姿态和有效半径。

## 5. 逆运动学：针尖目标 → 编码器

主要接口：`inverse_robot_encoders_from_tip(target_tip, current_robot, ...)`。

1. 根据当前编码器求当前针轴方向 `u_current`。
2. 计算目标向量的模长 `radius`。
3. 目标接近零时沿用当前针轴方向，交接点半径设为 `−sign × offset`。
4. 目标非零时，通过目标单位向量与 `u_current` 的点积选择针轴同向或反向分支，并保留对应半径符号。
5. 由方向计算 `phi = atan2(x, z)`、`theta = acos(y)`、`alpha = theta − 22°`。
6. 输出取整后的 A/B/C 编码器目标。

```text
target_A = round(phi_deg × 200000 / 360)
target_B = round(K × d_B_mm(alpha_deg))
target_C = round(K × (14 + signed_junction_radius_mm))
```

返回值只有 `(A, B, C)`。新针包装函数固定 `tip_offset_sign = −1`。

`inverse_robot_encoders_from_position()` 是另一个基于位置模长和轴向偏置的接口，接近 RCM 时直接报错，不能与带符号针尖逆解不加区分地替换使用。

## 6. 手柄协议：整数 count 如何转换为 ABC 运动

### 6.1 一个手柄 count 的名义量

```text
A 编码器 counts/handle = (2 × G × 3 / 4000) × (200000 / 360)
B 连杆角 deg/handle   = G × 3 / 4000
C 编码器 counts/handle = 3 × G × 10000 / 4000
```

| 量 | G15：本文文件 | G10：当前固定仿真副本 |
|---|---:|---:|
| A 编码器 count/手柄 count | 12.5 | 8.333333… |
| A 名义角度/手柄 count | 0.0225° | 0.015° |
| B 名义连杆角/手柄 count | 0.01125° | 0.0075° |
| C 编码器 count/手柄 count | 112.5 | 75 |
| C 名义位移/手柄 count | 1.762695307 µm | 1.175130205 µm |

**这些数值是软件指令分辨率，不能写成实机定位精度。**

### 6.2 增量到编码器目标

`predicted_robot_encoders()` 的关键逻辑：

```text
A_target = round(rA + dA × A_counts_per_handle)
alpha_target = alpha(rB) + dB × B_deg_per_handle
B_target = round(K × d_B_mm(alpha_target))
C_target = round(rC + dC × C_counts_per_handle)
E_target = rE
```

B 使用角度增量后重新计算连杆编码器目标，不能直接将 B 编码器差除以固定常数当成手柄 count。

G15 的半整数会经过 Python `round()`（中点取偶数）。整数参考状态下，相邻单手柄目标的编码器差可能为：

- A：12 或 13 个编码器 count，即 0.0216° 或 0.0234°。
- C：112 或 113 个编码器 count，即约 1.754861106 或 1.770529508 µm。
- B：编码器差随连杆姿态变化。

累计目标应从统一会话锚点计算。逐包对已取整的反馈再叠加增量，可能产生与锚点算法不同的累计取整误差。

### 6.3 2 µm 与 10 µm 的 C 轴示例

当输入是沿当前针轴的 C 轴正向位移，并采用最近整数手柄目标时：

```text
名义 C 步长 = 112.5 / 63822.71488 × 1000
             = 1.762695307 µm/count

2 µm  → round(2 / 1.762695307)  = 1 count
10 µm → round(10 / 1.762695307) = 6 counts
```

因此 C 累计目标分别增加 1 或 6，名义位移约 1.762695307 或 10.576171842 µm，不是精确的 2 或 10 µm。实际编码器目标还要取整；先做编码器逆解的调用路径也有前一级量化。任意 XYZ 的 2 µm 修正可能需要 ABC 联动，不能一律解释为 C 加 1。

### 6.4 编码器反馈分辨率

- A：`360 / 200000 = 0.0018°/encoder count`。
- C：`1000 / K = 0.015668403 µm/encoder count`。
- B：因连杆非线性而变化；在 `rB=1466021` 附近，相邻编码器 count 对应约 `0.000011655°`。

编码器反馈能区分更细的状态，不等于当前整数手柄协议能直接命令这么细的终点。

### 6.5 A/B 轴运动：角度如何变成针尖位移

A、B 均改变针轴方向。固定 C 时，模型针尖到 RCM 的有符号半径为：

```text
s_mm = rC / K − 14 + tip_offset_sign × tip_offset_mm
p = s × (sin(theta) sin(phi), cos(theta), sin(theta) cos(phi))
```

这里的轴向和方向均为运动学模型坐标；MuJoCo 世界坐标还需应用安装旋转和平移。

- **A 轴**：改变 `phi`，相当于绕模型 Y 轴转动；固定 B/C 时针尖 Y 不变，X/Z 沿圆弧变化，圆弧半径为 `abs(s × sin(theta))`。
- **B 轴**：改变连杆角 `alpha`，且 `theta=alpha+22°`，因此 `delta_theta=delta_alpha`；固定 A/C 时针尖在由模型 Y 轴和当前针轴确定的平面内摆动，圆弧半径为 `abs(s)`。
- **C 轴**：固定 A/B 时改变 `s`，针尖沿当前针轴直线运动。

G15 下，手柄计数变化 `dA/dB` 对应的名义角度为：

```text
delta_phi_deg = dA × 0.0225°
delta_theta_deg = dB × 0.01125°
```

单独转动一轴的针尖起终点距离（弦长）为：

```text
A：distance_mm = 2 × abs(s × sin(theta)) × abs(sin(delta_phi_rad / 2))
B：distance_mm = 2 × abs(s) × abs(sin(delta_theta_rad / 2))
```

小角度近似下，针尖位移向量为：

```text
A：delta_p ≈ s × (sin(theta) cos(phi), 0, −sin(theta) sin(phi)) × delta_phi_rad
B：delta_p ≈ s × (cos(theta) sin(phi), −sin(theta), cos(theta) cos(phi)) × delta_theta_rad
```

例如 `phi=0`、正半径 `s>0`、常用姿态 `theta≈76.236°` 时，正 A 首先使针尖向 +X 运动；正 B 使 Y 减小、Z 增大。改变姿态或半径符号后，不能沿用固定的 XYZ 正负方向。

取 `rB=1466021`，即 `theta≈76.236040°`，按名义角度计算：

| 针尖到 RCM 的距离 | A 加 1 的针尖位移 | B 加 1 的针尖位移 |
|---|---:|---:|
| 4 mm | 约 1.525690 µm | 约 0.785398 µm |
| 0.5 mm | 约 0.190711 µm | 约 0.098175 µm |

这些是指定几何条件下的名义弦长，尚未计入整数编码器取整、动态滞后或实机误差。4 mm 在此是人为指定的针尖到 RCM 距离，不能因为 `DEFAULT_TIP_OFFSET_MM=4` 就默认真实半径等于 4 mm。

如果要求沿该轴圆弧切向约移动 2 µm，在 4 mm 半径示例中，小角度换算并取最近整数得到 A 约加 1、B 约加 3；半径变为 0.5 mm 时则约为 A 加 10、B 加 20。它们均是近似示例，不是任意 XYZ 的 2 µm 位移换算规则。一般 XYZ 目标应通过逆解计算 ABC 联动。

当 `s=0` 时，针尖位于 RCM，单独转动 A/B 不改变模型针尖位置，尽管针轴方向仍变化；当 `sin(theta)=0` 时，A 对针尖位置的瞬时影响也为零。因此 A/B 不存在全姿态通用的“每 count 固定几微米”。

## 7. 笛卡尔修正接口的完整流程

`cartesian_handle_delta_float()`：

1. 检查机器人、交接点、修正量维度，并检查交接点与修正量为有限值。
2. 用反馈交接点与针尖偏置构造反馈坐标中的 `target_tip`。
3. 在 RCM 模型坐标中构造 `model_tip + delta_xyz`，供逆解使用。
4. 逆解生成整数编码器目标，并检查针轴变化不超过 8°。
5. 将编码器差/连杆角差换算为浮点手柄增量。
6. 返回 `(target_tip, raw_handle_delta, ideal_target_robot_ABCE)`。

这种处理允许反馈坐标与模型坐标之间存在平移；前提是 XYZ 轴方向、单位一致。它不估计任意坐标旋转。不要把反馈原点偏移本身当成要执行的位移。

`cartesian_handle_target()` 在上述流程后执行：

```text
integer_delta = tuple(round(value) for value in raw_handle_delta)
target_handles = current_handles + integer_delta
```

然后检查累计手柄目标在 int32 范围内。返回的 `target_robot` 是手柄取整之前的逆解目标，不保证等于实际整数手柄可重现的编码器目标。需要评估实际目标时，应结合协议锚点和 `predicted_robot_encoders()` 重算。

文件中的函数不会保存跨调用残差。浮点接口为上层控制器提供保留小数的机会，但不会自动实现残差累计。若增加累计逻辑，应在上层增量计算阶段实现，保持累计绝对消息重发幂等。

## 8. 限位、单包限制与方向标志

机器人编码器软限位：

| 轴 | 最小值 | 最大值 |
|---|---:|---:|
| A | −74873 | 73302 |
| B | 223159 | 5023488 |
| C | 0 | 2952362 |

`validate_manual_jog()` 检查当前及预测 ABC 编码器范围；预测逻辑拒绝非零 E 增量。

`validate_cartesian_delta()` 进一步检查量化后针尖位移与针轴角度变化，默认位移上限为 1 mm，调用方可提供更小上限。并非每个换算辅助函数都会自动执行这些完整检查。

`limit_cartesian_packet_delta(delta, limits)` 是独立工具：检查整数输入、禁止 E 变化，按共同缩放系数限幅，再取整。它不保存待执行队列，不保证物理完成时间。不能因为函数存在，就认为所有控制路径都在调用它。

外围基点响应控制器 `BasePositionExperiment` 另有 A/B/C/E 每包上限 `(8, 9, 1, 0)`；响应周期由 `ResponseControl.interval_s=0.05` 定义。它们不在本文件中设置，也不能与实机 UI 的其他限制混用。

`make_control_message(values, mode)` 将四轴值转换为整数并设置标志：

- `hold`：`stepForward=False`、`stepBackward=False`。
- `forward`：`True, False`。
- `backward`：`False, True`。

辅助函数的 `hold` 名称只描述标志组合；只要累计手柄目标变化，仍可能发生运动，不等于已实施物理制动。代码对未知 mode 也返回两个 False，调用方应自行保证 mode 合法。方向标志不定义已标定的进退针速度；当前离线 `MotorBackend` 拒绝非零方向标志，仅支持显式累计 C 目标。

## 9. 路点、显示与函数索引

| 函数组 | 入口 | 作用 |
|---|---|---|
| 协议/点动 | `direction_text`, `control_direction_flags`, `make_control_message`, `stepped_values`, `c_handle_counts_for_distance` | 标志、消息、累计目标加法及距离换算 |
| 几何 | `b_distance_from_alpha`, `b_alpha_from_robot_encoder`, `needle_direction_from_robot_encoders`, `model_tip_from_robot_encoders` | 连杆、方向与针尖正解 |
| 逆解 | `inverse_robot_encoders_from_position`, `inverse_robot_encoders_from_tip`, `inverse_robot_encoders_from_new_needle_tip` | 位置/针尖到编码器目标 |
| 笛卡尔输入 | `cartesian_direction_delta`, `cartesian_step_mm_from_um` | 点动方向与整数微米输入限制 |
| 笛卡尔协议换算 | `cartesian_handle_delta_float`, `cartesian_handle_target`, `limit_cartesian_packet_delta` | 修正量转手柄及独立限幅 |
| 编码器/协议互换 | `predicted_robot_encoders`, `virtual_handles_from_robot_encoders` | 手柄增量预测、相对零点的虚拟手柄目标 |
| 校验 | `validate_manual_jog`, `validate_cartesian_delta` | 软限位、量化后位移和针轴角度检查 |
| 路径 | `plan_tip_segments`, `plan_new_needle_tip_segments`, `refine_tip_path_for_axis_limit` | 直线路点分段、按角度插入共线中点 |
| 姿态/显示 | `initial_observation_geometry`, `tip_to_display`, `relative_to_rcm`, `trajectory_to_display`, `b_axis_display_direction`, `needle_pose_from_robot_encoders`, `new_needle_pose_from_robot_encoders` | 显示几何与姿态字典 |
| 绝对位置/针段 | `absolute_tip_position_from_state`, `absolute_new_needle_position_from_state`, `new_needle_solid_segment` | 绝对坐标中的针尖与实体针段 |

路点规划保持调用方 XYZ 坐标，逆解时才平移到模型坐标。`refine_tip_path_for_axis_limit()` 可通过 `max_points` 限制点数，并在无法细分时抛异常。路点几何检查不代表完整的电机动态或轨迹执行验收。

方向键映射为：left=+X、right=−X、forward_y=+Y、backward_y=−Y、up=−Z、down=+Z。它是本代码的约定，不应按屏幕直觉重写。

## 10. 已验证的行为与局限

以下是 2026-11-20 已完成的离线检查结果，本文编写时未重跑训练或 MuJoCo 闭环：

- 22 项现有测试通过，涉及针型、验证路径、基点响应；错误/失败/跳过均为 0。
- 语法检查通过；选取的 7 个合法 B 编码器点正反换算均回到原 count。
- 测试使用离线消息替身，不验证真实 ROS 消息序列化、网络或硬件。
- 在 `robot=(0,1466021,925000,0)` 且沿当前针轴修正时，0.35 µm 返回浮点 C 增量约 `0.195555556`，整数包装函数返回 0。
- 同一状态下，0.005 µm 在逆解的编码器取整阶段已经丢失，浮点手柄接口也返回 0。因此“浮点手柄接口”不等于完全连续的逆运动学输出。
- `b_alpha_from_robot_encoder()` 缺少有限值和几何可达性前置检查：测试中 `+inf` 返回 180°、`−inf` 返回 0°，明显越界的输入也可能得到数值结果。`acos` 截断只能处理数值边界，不能代替输入合法性检查。
- 部分接口先做 `int(value)`；严格整数类型、字段完整性和 int32 合法性不能假设已由所有函数统一检查。

这些结果不证明整个工作空间、全姿态、真实设备或其他控制器都已验证，也不能据此保证达到 0.35 µm 的系统成功阈值。

机器可读证据：

```text
rcm_4090_deploy/runs/20261120/verification/kinematics_review_200233_v2/review_results.json
rcm_4090_deploy/runs/20261120/verification/kinematics_review_200233_v2/unit_tests.log
```

失败的初版测试包装脚本记录保留在同级 `kinematics_review_200156/`：其测试导入路径配置错误，未执行实际单元测试；修正包装脚本后，第二个目录中的 22 项测试通过。此问题不来自被分析源码。

## 11. 交给下一个模型时应保留的约束

1. 先核对绝对路径、SHA-256 和实际导入的 `module.__file__`，明确讨论的是 G15 文件还是当前 G10 固定副本。
2. 分清 robot encoder count、handle count、deg/rad、mm/µm、交接点/针尖、RCM 模型坐标/反馈坐标。
3. 不把 10 µm 点动下限、1 µm 容差或 1.763 µm 名义协议步距称为实测定位精度。
4. C 轴 2 µm 取整成 +1 count 是近似命令；不能把任意 XYZ 修正等同于 C 运动。
5. 浮点接口不自动累计残差；不能通过重复叠加绝对累计消息来补偿小动作。
6. G10/G15 的控制器与正向适配器必须匹配，不能只改一端。切换会改变量化终点，需要独立验证，不能默认提升精度。
7. 不能只在运行过程中修改 `CONTROL_GEAR` 就认为切换完成：若干默认参数在定义时绑定，派生常量也已计算；Python 模块还可能已被缓存导入。
8. 本次任务只编写说明；未修改源码、固定驱动、数据、模型或控制参数，也未部署 G15。
