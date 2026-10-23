# MuJoCo RCM 模型：代码证据与报告公式

核查日期：2026-10-23。范围仅限 `rcm_reference.py`、`rcm_recovered_model.py`、`config.json`、`scene.xml`；模拟代码和旧 PPT 未修改。数值验证仅使用已完成的三个姿态，见 `numeric_checks.json`。

## 1. 可放入主报告的结论

当前实现是 **3 个独立任务关节的等效 RCM 运动学参考模型**。外观机构另有 4 个从属显示关节，MuJoCo 因而有 `nq=nv=7`，但独立控制维数仍为 3，`nu=0`。任务主链为：

```text
world
  └ fixed_inclined_mount      安装 Rx(45°)，原点在 RCM
      └ A_stage              A_azimuth：hinge，局部轴 (0,−1,0)
          └ B_RCM_task_frame  B_pitch：hinge，局部轴 (1,0,0)
              └ C_carriage_and_needle
                               C_insertion：slide，局部轴 (0,0,−1)
```

A、B 是等效任务旋转轴，均在虚拟 RCM 原点；这不证明实体电机铰轴都位于 RCM。`illustrative_linkage.measured=false`，外观杆件不是已辨识的完整机构 CAD。

## 2. 坐标与单位：task ABC 不等于 JointState

MuJoCo 内部任务坐标记为 `q=(a,b,c)`，单位 `(rad,rad,m)`。界面对应 `(A_deg,B_deg,C_mm)`；源运动学 `JointState=(phi_deg,alpha_deg,length_mm)` 使用 `(deg,deg,mm)`。

```text
phi_deg   = phi0_deg   + a × 180/pi
alpha_deg = alpha0_deg + b × 180/pi
length_mm = L0_mm      + 1000 c
```

当前配置：

- 报告复位编码器 ABC = `[0,1466018,893518]` counts。
- `phi0_deg=0`；`alpha0_deg=54.23600525044604`。
- `beta0_deg=alpha0_deg−(90−22)=−13.763994749553959`。
- `theta0_deg=alpha0_deg+22=76.23600525044604`。
- `L0_mm=4`：**暂定的复位 RCM 到针尖距离，尚未完成实体测量标定**。
- 所以 task ABC=`(0°,0°,0mm)` 对应 `JointState=(0°,54.23600525044604°,4mm)`。

4 mm 远端细针段长度与暂定 4 mm RCM 到针尖距离是两个不同参数；当前数值相等不能将二者视作同一个已标定量。编码器源公式 `E_C/63822.71488−14` 在复位处约 `−1.3036e−7 mm`（一计数量化范围内约为零），并不提供当前几何采用的 4 mm 绝对针尖偏置。

## 3. 精确主链正运动学：矩阵顺序不能替换成常见 RzRy

以下统一使用米和弧度。令

```text
phi  = phi0 + a
beta = beta0 + b = alpha − 68°
L    = 0.004 + c
R_m  = Rx(45°)
p_R  = 配置中的 rcm_world_mm × 0.001 = (0,0,0)^T
```

令 `H(R)` 表示旋转的 4×4 齐次嵌入，`T(v)` 表示平移，则当前配置的精确 tip frame 变换为

```math
{}^W T_{tip}=T(p_R)\,H(R_m)\,H(R_y(-\phi))\,H(R_x(\beta))\,T_z(-L).
```

源 body 链的初始与任务项可显式拆开：

```math
T(p_R)H(R_m)H(R_y(-\phi_0))H(R_y(-a))H(R_x(\beta_0))H(R_x(b))T_z(-L).
```

`C_carriage_and_needle` 的一般固定初始平移是 `−(L0−distal_length)`，针尖 site 再位于 `−distal_length`；加上 C 的 `−c`，总平移因此为 `−(L0+c)`。当前二者均为 4 mm，carriage 初始平移为零。

记 `s_phi=sin(phi)`、`c_phi=cos(phi)`、`s_beta=sin(beta)`、`c_beta=cos(beta)`，安装前的完整矩阵为：

```math
{}^M T_{tip}=\begin{bmatrix}
c_\phi&-s_\phi s_\beta&-s_\phi c_\beta&L s_\phi c_\beta\\
0&c_\beta&-s_\beta&L s_\beta\\
s_\phi&c_\phi s_\beta&c_\phi c_\beta&-L c_\phi c_\beta\\
0&0&0&1
\end{bmatrix}.
```

针尖与指向针尖的单位针轴：

```math
d_w=R_m R_y(-\phi)R_x(\beta)(0,0,-1)^T,
\qquad p_{tip,w}=p_R+L d_w.
```

tip frame 的局部 `+z` 指向针体后方，`−z` 才是插入/指向针尖方向。主报告示意图应保留该符号。

## 4. 与源 robot 坐标的等价形式

源 `RCMModel` 取 `theta=alpha+22°`：

```math
d_r=\begin{bmatrix}\sin\theta\sin\phi\\\cos\theta\\\sin\theta\cos\phi\end{bmatrix},
\qquad p_{tip,r}=L d_r.
```

robot 到 world 转换：

```math
R_{wr}=R_m\operatorname{diag}(1,-1,-1),\quad
p_w=p_R+R_{wr}p_r,\quad
p_r=R_{wr}^T(p_w-p_R).
```

`diag(1,−1,−1)=Rx(180°)` 仅是固定坐标换基，**不是另一次物理安装旋转**。当前

```text
R_wr = [[1,0,0],
        [0,−0.7071067811865476, 0.7071067811865475],
        [0,−0.7071067811865475,−0.7071067811865476]]
```

## 5. RCM 约束与可报告的误差

理想模型约束：

```math
\|d_w\|=1,\qquad
(I-d_wd_w^T)(p_{tip,w}-p_R)=0,\qquad
p_{tip,w}-p_R=L d_w.
```

代码的针轴到 RCM 距离与任意两个轴线上点等价：

```math
e_{axis}=\|(p_R-p_{back})\times d_w\|.
```

区分两个指标：RCM site 漂移检验固定点位置；针轴到 RCM 距离检验针轴线是否穿过该点。极小数值残差只是理想运动学的解析式与 MuJoCo 一致性，不能称为实体定位精度、动力学性能或组织穿刺误差。

## 6. 编码器映射与 B 非线性

当前比例 `k_A=555.5555555555555 counts/deg`、`k_BC=63822.71488 counts/mm`，方向均 +1、origin 均 0。源 B 位移为

```math
D_B(\alpha)=98.48-60\cos\alpha-\sqrt{52^2-(60\sin\alpha-16)^2}\quad[mm].
```

当前启用复位参考，模型 count 输出：

```text
E_A = E_A0 + k_A × (a × 180/pi)
E_B = E_B0 + k_BC × [D_B(alpha0+b)−D_B(alpha0)]
E_C = E_C0 + k_BC × (1000 c)
```

公式内角度按弧度理解，若直接调用 Python 方法则参数使用 degrees。B 是非线性连杆角映射，不能写成恒定 counts/degree；B 跨过其位移转折点时可能存在多个角解，需要上一姿态/目标姿态提示来保持支路。该提示不是新增编码器或新增 DOF。编码器比例公式已获用户确认，但 config 中 `encoders.calibrated=false`，不能宣称完成硬件零偏、方向及空间标定。

## 7. 备份页可用 Jacobian

源坐标正运动学对 `(phi_rad,alpha_rad,L_mm)` 的 Jacobian 为：

```math
J_r=\begin{bmatrix}
L\sin\theta\cos\phi&L\cos\theta\sin\phi&\sin\theta\sin\phi\\
0&-L\sin\theta&\cos\theta\\
-L\sin\theta\sin\phi&L\cos\theta\cos\phi&\sin\theta\cos\phi
\end{bmatrix}.
```

若输出 world mm、输入 task `(rad,rad,m)`，则 `J_task,mm=R_wr J_r diag(1,1,1000)`；若输出 world m，则再乘 `0.001`。这个单位变换不能省略。复位工作点代入 `phi=0`、`theta=76.23600525044604°`、`L=4mm` 即可。

## 8. 从属关节与模型边界

```text
visual_B_rear         = +b
visual_B_bridge       = −b
visual_B_lower_bridge = −b
visual_B_aux          = +b
```

代码同时以 equality joint coupling 描述这些依赖，并在 `set_task` 时直接赋值。外观杆件的四个 qpos 不增加独立控制自由度。外观 `visual_scale=0.67` 只缩放说明性装置；眼球与针几何没有随其缩放。

运动更新是直接设置 qpos、清零 qvel、调用 `mj_forward`，没有在此执行 `mj_step` 动力学积分。MJCF 设置 `gravity=0`、`contact=disable`，没有 actuator；各 body 惯量只是满足编译器所需的正值占位，未测量、未用于运动。当前不能用于实体电机扭矩/摩擦、组织接触或穿刺力、动力学稳定性、真实机械限位和安全保证的结论。

`measured_limits.enabled=false`，记录的编码器窗口不是已确认的全行程/硬件安全范围。模型仍检查非负长度和 B 连杆实数可解性；界面滑条范围也不能等同于硬件可达范围。

## 9. 三姿态已完成数值核对

使用仓库已有 `.venv/bin/python` 构建 `Reference()`，直接调用 `set_task→mj_forward`，无渲染、无导航、无训练。解析式使用本文件第 3 节真实矩阵顺序。

| task A/B/C (deg,deg,mm) | 解析 tip world (mm) | FK 差范数 (mm) | 轴到 RCM 距离 (mm) |
|---|---|---:|---:|
| 0,0,0 | (0, 2.074257706609, −3.420154231402) | 2.589462819655575e−15 | 0 |
| 5,2,2 | (0.511950585785, 3.272728255044, −5.002714899567) | 2.8435583831733385e−15 | 2.220446049250313e−16 |
| −20,8,10 | (−4.764072535047, 8.261232141943, −10.249666159331) | 8.140289677804162e−15 | 2.979040983896728e−15 |

## 10. 代码来源定位

- `/workspace/rcm-mujoco-reference/rcm_reference.py:1`：运动学范围声明。
- `rcm_reference.py:20`：三个任务关节及四个从属关节定义。
- `rcm_reference.py:37`：安装矩阵 `Rz(z)Ry(y)Rx(x)`；当前 y=z=0，因此仅 Rx45。
- `rcm_reference.py:67`：mount 与 robot 坐标换基。
- `rcm_reference.py:94`：复位 phi/alpha、暂定 tip reference 及 beta 初始项。
- `rcm_reference.py:216`：task 与 JointState 映射。
- `rcm_reference.py:235`：直接 qpos 与 `mj_forward`。
- `rcm_reference.py:247`：启用复位参考的 encoder 映射。
- `rcm_reference.py:258`：B 支路解码提示。
- `rcm_reference.py:282`：robot/world 双向坐标变换。
- `rcm_reference.py:394`：FK、RCM site 漂移及轴到 RCM 距离指标。
- `rcm_reference.py:448`：运动学说明、零重力与 contact disable。
- `rcm_reference.py:641`：惯量占位声明。
- `rcm_reference.py:647`：A、B body 的初始旋转。
- `rcm_reference.py:664`：A 局部 −y 轴；`:674` B 局部 +x 轴。
- `rcm_reference.py:690`：carriage 固定平移；`:692` C 局部 −z 轴。
- `rcm_reference.py:727`：tip site 在局部 z=−distal_length。
- `rcm_reference.py:730`：等效可视杆件；`:751` equality coupling。
- `/workspace/rcm-mujoco-reference/rcm_recovered_model.py:59`：JointState 单位字段。
- `rcm_recovered_model.py:69`：三任务坐标、非完整空间连杆辨识的范围声明。
- `rcm_recovered_model.py:94`：B 非线性位移公式。
- `rcm_recovered_model.py:149`：源针轴与 FK。
- `rcm_recovered_model.py:221`：源 Jacobian，列单位 rad/rad/mm。
- `/workspace/rcm-mujoco-reference/config.json:28`：45°安装且未标定。
- `config.json:53`：细针尺寸与暂定 tip-to-RCM 的区别。
- `config.json:220`：复位 counts 与 4 mm 参考偏置未确认标记。
- `/workspace/rcm-mujoco-reference/scene.xml:72`：实际导出 body chain；`:79` A，`:85` B，`:105` C；`:149` 从属 equality。

注：数值检查调用的是 `Reference.build_xml()` 生成的现行模型。`scene.xml` 的主任务链与本报告一致；导出文件和当前生成器的部分显示组设置存在细节差异，不影响上述主链 FK。
