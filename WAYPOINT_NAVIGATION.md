# 多路径点共享排序、导航对比与数据集

## 当前视觉测量与停止条件（2026-10-22）

默认 `position_feedback="geometry"` 使用1920×1440实际渲染。针尖由可见针杆圆柱多点与实体端面/真正尖端小球重建，固定相机、RCM、针具尺寸是标定输入；目标由当前RGB、分割和深度球面拟合得到。不可观测或拟合质量不足会明确失败，不以单个锥面像素补齐尖端，也不读取动态目标/针尖真值或进行已知目标姿态的特征校准。

几何控制按测得的三维误差求耦合轴指令并限速、限加速度、接近减速；默认8 μm测量阈值和连续三帧零指令速度确认，之后用独立MuJoCo真值按<20 μm验收。C-only接近只按可控方向判断测量收敛，完整三维位置仍须通过独立验收。历史 `image_depth` 模式的像素/深度阈值和可选目标姿态标定另行保留，下方旧结果不代表新几何测量的实测性能。

默认眼底视觉伺服仍先依据已知目标构造回撤和离壁转移轨迹，RGB在该阶段只记录，最后固定方向C接近才参与视觉控制。这是混合先验流程；几何反馈取消目标姿态特征校准，并不取消转移阶段的目标先验。`--strict-visual` 已支持按输入顺序连续访问 3–5 个目标，输出 `guidance_mode` / `execution_mode` 为 `strict_visual_multitarget`：当前 RGB-D 测量生成 A/B 对准与 C-only 接近指令；两点间按上一实际观测针尖和针轴定义回撤 2 mm 的虚拟点，以视觉闭环执行后再对准下一点。该入口不做预先目标建图、逆解或路线排序。详细测量边界、遮挡模式与复现步骤见 [严格视觉导航](STRICT_VISUAL_NAVIGATION.md)。所有结果仍为理想RGB-D运动学仿真，不能称为真实RGB-only感知或硬件精度；当前尚无严格三点 3/3 完整通过报告。

```bash
python verify_visual_geometry.py --output-dir runs/geometry_checks
python verify_visual_geometry_direction.py
python verify_visual_motion.py
python verify_visual_servo.py --output-dir runs/visual_checks
python verify_vision_navigation.py --output-dir runs/vision_checks
```

> 本文下方的三目标参数和失败表是旧实验记录，不代表最新数据。新的五目标 `visual_current_target` pilot 已在 20 Hz 下完成；数据与审计报告见 [随机眼底数据集说明](RANDOM_FUNDUS_DATASET.md#无目标坐标-task-pilot-2026-10-19) 和 `runs/历史记录/05_random_datasets/pilot_n5_visual_current_target_v1/`。

2026-10-18 尺寸更正：前端细针段外径 **0.060 mm**、长度 **4 mm**，零位针尖到 RCM 的暂定偏置仍为 4 mm，近端杆体暂沿用直径 0.5 mm。当前 MJCF 将器械分成近端杆体、2.5 mm 细针管段和 1.5 mm 八边形渐缩局部针尖。旧的 0.410 mm × 12 mm 结果保留在历史记录，不代表当前模型。下方三点结果是针体可见性失败的历史实验，不能代表顶部所链接的新五目标 pilot。

## 运行

在 `rcm_mujoco_reference/` 中运行：

```powershell
.\.venv\Scripts\python.exe run_waypoint_comparison.py
```

程序无窗口执行，自动创建新的 `runs/历史记录/04_sessions/waypoint_comparison/日期时间/`。默认读取 `examples/three_waypoints.json`，包含三个无序目标，输入坐标单位 mm：P1 `[2.6400,4.5086,23.2390]`、P2 `[-0.9600,7.6648,22.5900]`、P3 `[1.9200,8.4780,22.0430]`（此处四舍五入；文件中保留完整精度）。

三个点从编译后的眼球椭球后部曲面生成，满足局部归一化 z ≥ 0.9、椭球方程值为 1；不是悬在眼球内部。模型没有独立视网膜层，因此采用眼球后部曲面作为眼底代理。改变眼球几何后，运行 `python generate_fundus_waypoints.py` 重新生成目标。运行前会检查眼底位置并保存 `fundus_target_verification.json`。

未指定 `--strict-visual` 时，默认 `--planner-feedback vision` 的规划比较流程为：相机测量目标 → 自主排序 → RRT/PRM 规划 → 每步视觉测量针尖并纠偏。目标坐标文件只布置仿真标靶和供独立验收；不会填补不可见目标。当前针体尺寸为近端杆体直径 0.5 mm、前端细针段外径 0.060 mm、长度 4 mm。尺寸更新后必须重新生成视觉测量和遮挡诊断，旧版像素数、访问顺序和成功结果不能直接沿用。

感知使用 RGB 目标检测、仿真分割和深度，根据已标定的固定相机、RCM、针具几何和球形标靶尺寸分别重建目标中心与物理针尖。深度边界异常点通过鲁棒拟合剔除；支持不足、拟合不一致或目标/针具几何不可观测都拒绝继续。几何测量不读取目标真值，也不把机器人移到目标姿态进行特征标定。它仍是借助仿真深度/分割的 RGB-D 实验，不是普通 RGB 相机上的实机系统。建图逐个呈现标靶，尚非同时检测三个病灶。

上述默认三目标规划比较只有在三个目标均有有效视觉测量后，脚本才枚举全部 6 种访问顺序，按视觉测量转换的轴目标及限速、限加速度下预计运动时间选择顺序；三种方法共享该次运行的顺序。`--strict-visual` 使用给定输入顺序，跳过此建图与排序。这些流程都不能证明学习模型同时识别或自主排序多个病灶。中途不整体回零；眼底点间必须沿针轴回撤，再离壁转向、固定方向接近。默认排序代价包括三个阶段，不宣称是实际视觉执行时间的全局最优解。

默认 `--profile precision`：视觉伺服一次、RRT 和 PRM 各三个种子（7、17、27）。建图失败时这些是共同前置条件的失败记录，不是算法搜索失败。建图成功后 RRT 扩展采样树、PRM 构建并复用路线图；路径简化仍须通过机械/离壁检查。执行时每条指令由视觉针尖误差决定并限速、限加速度；指定点及保留的路径拐点停稳。执行中目标不可见立即终止并保持，目标移动超过 0.15 mm 要求重新建图，不自动切回模型导航。立即保持只是仿真中止，不代表实机制动已验证。

当前几何视觉伺服使用8 μm测量位置阈值，接近目标时根据测量位移减速，指令速度归零且连续三帧条件满足后停止。三维真值只用于独立的<20 μm验收，不进入控制或停止条件。保留的`image_depth`模式使用各自像素/深度容差；其参数不可解释为三维几何阈值。A/B/C 的原速度上限 `[4,2,0.6]`（度/秒、度/秒、毫米/秒）及加速度上限 `[10,5,1.5]` 保持不变。

使用 `--planner-feedback model` 明确选择旧的真值模型规划；再加 `--profile legacy --targets examples/three_waypoints_interior.json` 可复现旧内部点比较。该模式不属于视觉导航，不能混入新版成功率。

可指定新的目标文件、方法或种子：

```powershell
.\.venv\Scripts\python.exe run_waypoint_comparison.py --targets examples/three_waypoints.json --seeds 7
.\.venv\Scripts\python.exe run_waypoint_comparison.py --methods visual_servo
```

目标文件仍用现有 `schema_version=1` 格式，ID 唯一。下方历史三目标比较的文件排列顺序不是执行顺序；严格模式则按文件给定的输入序列访问，可用于 3–5 点连续导航，不做自主排序。每次运行只更新内存中的目标标靶，不改写 `config.json`。

回撤距离和离壁余量是仿真参数，分别通过 `--retract-mm 2`、`--transfer-clearance-mm 1` 调整。余量用椭球几何给出的针尖到表面距离下界检查，不等于整根针的组织间隙，也不是临床安全距离。RRT/PRM 仅规划两个回撤位置之间的眼内转移段，路径简化不能删除回撤/接近步骤。视觉伺服每条指令下发前也检查近壁转向，并在成功停稳后的目标切换前执行回撤；旧三目标实验曾因遮挡失败，尚未完成点间转移。2026-10-19 五目标 pilot 已完成仿真导航，但只有部分遮挡、没有完全遮挡帧，详见本文顶部链接。

## 输出

下表主要描述默认规划比较输出；严格模式的给定顺序、视觉阶段和审计字段以 [严格视觉导航](STRICT_VISUAL_NAVIGATION.md) 及该次 `experiment_protocol.json` 为准。

| 文件/目录 | 内容 |
|---|---|
| `order_selection.json` | 6 个候选顺序、每段预计时间、选择结果 |
| `visual_target_map.json` | 视觉测量与不可见原因；建图失败时不产生访问顺序 |
| `各方法/visual_feedback.json` | 成功建图后逐条视觉测量、残差与实际轴指令 |
| `comparison.md` / `comparison.csv` | 方法汇总对比表；CSV 可用 Excel 打开 |
| `waypoint_results.csv` | 每个方法、随机种子、路径点的到达误差与时间 |
| `run_reports.json` | 每条轨迹的完整指标与通过状态 |
| `visual_servo/`、`RRT_seed_*/`、`PRM_seed_*/` | 控制报告/规划路径及统一格式的逐帧几何日志 |
| `config_snapshot.json`、`source_sha256.json` | 实验参数快照与关键代码哈希 |
| `experiment_protocol.json` | 精度模式、到达容差、逐目标停止要求 |
| `figures/fundus_transitions.png` | 眼底附近放大图：回撤、眼内转向、接近与方向箭头 |
| `figures/mujoco_paths.png` | MuJoCo 渲染的三种方法路径对照 |
| `figures/paths_3d.png`、`path_projections.png` | 仿真针尖三维路径及三个平面投影 |
| `figures/error_curves.png`、`before_after.png` | 分目标误差收敛及与旧版的时间/精度对比 |
| `dataset/manifest.json` | 字段定义、单位、轨迹索引、图像尺寸 |
| `dataset/data/episode_*.npz` | 有类型的状态、动作、任务、时间、索引等数组 |
| `dataset/data/episode_*.jsonl` | 相同数据的逐帧可读版本 |
| `dataset/images/episode_*/frame_*.png` | 原始相机 RGB 图像 |

图表同时保存 SVG。前后对比图仅在目标坐标相同时生成，眼底新任务不与旧内部目标混作同任务对比。独立重新绘图不必重新采集数据：

```powershell
.\.venv\Scripts\python.exe plot_waypoint_results.py --run-dir runs/历史记录/01_current/three_waypoint_retract_20261016 --baseline-dir runs/历史记录/02_baselines/three_waypoint_dataset_20261016
```

路径图来自保存的 MuJoCo 针尖坐标，不是示意曲线。RRT/PRM 分别展示首个随机种子的轨迹；本次无额外障碍物，简化后的两种规划路径重合。紫色区域标出眼底曲面。只有目标点落在眼底；点间先沿 C 轴回撤 2 mm，再在离壁至少 1 mm 的区域转向，最后保持 A/B 不变沿 C 轴接近。回撤和接近的重合路径在放大图中用虚线、实线及方向箭头区分。MuJoCo 图隐藏机构外壳以展示眼内路径，黄色球表示目标点，箭头表示行走方向。

## 数据集约定

| 字段 | 类型和单帧形状 | 含义 |
|---|---|---|
| `observation.images.microscope_rgb` | 解码后 uint8，`(480,640,3)` | 当前 `eye_internal` 原始 RGB；数组和 JSONL 中保存相对 PNG 路径 |
| `observation.state` | float32，`(3,)` | 动作执行前，实际 MuJoCo 的 A/B/C 任务关节位置 |
| `action` | float32，`(3,)` | 经控制/轨迹约束后，下一次实际下发的绝对位置目标 |
| `task` | 字符串 | 由 `--task-regime` 决定；两种模式及其专家信息见下文 |
| `timestamp` | float64 | 从该轨迹开始累计的仿真秒数 |
| `episode_index`、`frame_index` | int64 | 从 0 开始的轨迹/帧索引 |
| `waypoint_id` | 字符串 | 当前激活的目标标靶 |
| `is_terminal` | bool | 最后一帧标记 |
| `phase` | 字符串 | 本帧动作阶段：retract 回撤、transfer 眼内转移、approach 接近、visual_servo 视觉控制、hold 保持 |

A/B 单位为度，C 单位为毫米。位置坐标以已确认的任务零位为参考，不是原始编码器计数；action 是该坐标系中的位置目标，不是位置增量，也不是始终重复填写最终目标。

每帧 RGB 和 state 对应同一时刻、位于 action 之前。除最后一帧外，`action[i] == observation.state[i+1]`；最后一帧的 action 等于自身 state，表示保持。此关系成立于当前直接设置关节位置的运动学执行，不应假设真实电机也没有跟踪误差。

视觉图像直接来自闭环观测回调；RRT/PRM 图像在规划轨迹执行时采集。统一使用视觉传感器的渲染选项，不添加十字覆盖层。目标切换时整体移动中心球与四个定位球，观察相机保持固定。

默认名义采样间隔为 0.05 s（20 Hz）；RRT/PRM 的轨迹段端点可能产生更短间隔，因此请使用保存的 timestamp，不能按 frame_index/20 重新生成所有时间。

### 策略输入、任务模式与示教先验

`run_waypoint_comparison.py` 和 `random_fundus_dataset.py` 都提供 `--task-regime`：

- `registered_coordinate_prior`：task 文本包含假设已配准的机器人坐标。当前值来自仿真目标输入，并非实测 OCT；这是坐标先验导航任务。
- `visual_current_target`（默认）：task 文本不含目标坐标、ID、访问顺序或随机 seed，只要求到达当前呈现的标靶。随机目标、图像和状态仍来自仿真。

`dataset/policy_inputs/episode_*.npz` 是唯一约定的策略输入包，字段严格为 RGB 路径、`observation.state`、`task`；加载器必须把路径解码为图像像素，不应使用文件名作为特征。`policy_inputs_no_image/` 保留同一 state/task、仅移除 RGB；`action_labels/` 单独保存逐行对齐的 action。完整 `data/`、`targets.json`、目标坐标、`waypoint_id`、phase、visibility、访问顺序、时间戳和帧索引属于审计数据，不是策略输入。验证器检查字段白名单、两种模式的 task 文本、文件对齐和 action→下一帧 state 关系。

策略输入与专家信息不是一回事。专家可访问 RGB-D、分割、深度及仿真目标姿态。视觉伺服专家还使用已知目标姿态进行特征标定。当前眼底视觉伺服在回撤/转移阶段使用目标轴坐标生成解析运动，RGB 只被记录而不用于纠偏或可见性停机；只有最后 C 轴接近使用闭环视觉深度。因此这些动作标签来自混合先验专家，不能称为全程纯单目专家。

此处目标逐个呈现，访问顺序由程序选择并共享给各方法，不能据此宣称学习策略能同时识别并自主排序多个病灶。去图像输入包目前只是为消融实验准备；仓库尚无策略训练/推理，因此没有图像依赖或消融结果。正式检验应以目标集合为单位划分 train/validation/test，训练同构输入管线的 full-image 与 no-image 模型，再比较成功率、误差、约束违规及遮挡分层表现；零图像或打乱图像推理可辅助识别模型是否真正使用图像。即使这些实验通过，也需另设多病灶同时检测/排序任务才能支持自主排序结论。

这些文件匹配所需字段，但没有声称是已封装的 LeRobot Parquet/视频数据集。NPZ 不需要 pickle，PNG 无损保存 RGB。

### 读取样例

从项目工作目录执行以下 Python 代码：

```python
from pathlib import Path
import numpy as np
from verify_waypoint_comparison import read_png

root = Path("runs/历史记录/01_current/three_waypoint_retract_20261016/dataset")
with np.load(root / "data/episode_000000.npz", allow_pickle=False) as data:
    i = 0
    image = read_png(root / str(data["observation.images.microscope_rgb"][i]))
    state = data["observation.state"][i]
    action = data["action"][i]
    print(image.shape, image.dtype, state, action)
    print(data["task"][i], data["timestamp"][i])
```

`read_png` 支持本项目 `save_png` 生成的 RGB PNG。也可以使用自己的标准 PNG 解码库。

## 验证与解释

```powershell
.\.venv\Scripts\python.exe verify_waypoint_comparison.py --run-dir runs/历史记录/01_current/three_waypoint_retract_20261016
```

验证包括：输入顺序任意打乱后仍选择同一最优顺序、目标标靶整体移动且不改变机器人/相机姿态、RRT/PRM 在独立合成障碍物测试中确实绕行，以及路径简化不穿越测试障碍物、梯形/三角形速度轨迹的限速与连续性、数据类型、形状、索引、时间、动作对齐、图像完整性。每条轨迹抽取首/中/末图像，根据保存的 float32 状态重新渲染核对；浮点量化只允许极少边界像素差异。视觉目标切换的指令速度/加速度单独检查，精度模式还验证成功轨迹三个目标的 ≤20 μm 误差及停止确认。数据集验证 PASS 仅表示文件和记录完整，不表示失败的导航变成成功。当前还根据实际 MuJoCo 针尖位置独立检查每帧动作的离壁余量、回撤/接近时 A/B 固定、各目标的阶段顺序，并用直接连接两个眼底点的反例验证禁止滑行规则。眼底任务另外验证三个目标的曲面位置，并检查眼球中心和前表面不能被误判为眼底。

实际对比场景没有额外障碍物，几何约束是连杆有效、针尖位于参考眼球内。合成障碍物只用于规划器算法测试，不混入本次导航结果。针杆体积、入口孔径和组织接触不在本实验验收范围。

当前默认几何视觉伺服依赖仿真分割/深度和固定几何标定，不进行逐目标已知姿态特征标定；历史`image_depth`模式保留该可选先验。默认眼底视觉伺服的解析回撤/转移仍使用已知目标，不能宣称它与视觉RRT/PRM拥有完全相同的先验。视觉RRT/PRM的测量与反馈不读取目标/针尖真值，运动学用于测量到轴指令的转换及几何约束；独立验收可读取真值。旧模型导航接近零的误差不是感知精度或实机精度。所有结果均为运动学仿真，不包含真实驱动器响应。

表格中的“最大到达误差”是三个目标各自停止时误差的最大值；曲线中的“误差”是运动中针尖到当前目标的距离，开始时数千微米属于尚未到达，并不代表定位精度。仿真运动时间与脚本计算/渲染耗时也不同。数据帧数表示保存的 RGB/状态/动作样本数，不是规划迭代次数，轨迹变短后帧数会相应减少。

新版结果见 [对比表](runs/历史记录/01_current/three_waypoint_needle_011mm_20261016/comparison.md)、[视觉建图](runs/历史记录/01_current/three_waypoint_needle_011mm_20261016/visual_target_map.json) 和 [数据集索引](runs/历史记录/01_current/three_waypoint_needle_011mm_20261016/dataset/manifest.json)。旧针径与模型导航结果保留在历史记录。

| 方法 | 旧三目标实验结果 | 完整访问时间 / s | 最大到达误差 / μm |
|---|---|---:|---:|
| 视觉伺服 | 未完成：针尖不可见 | — | — |
| 视觉 + RRT（三个种子） | 未完成：针尖不可见 | — | — |
| 视觉 + PRM（三个种子） | 未完成：针尖不可见 | — | — |

该旧三目标实验共记录 7 条失败观测记录、7 帧，各次执行均在初始姿态因针尖不可见而拒绝运动；运动时间为 0。此前较大针尖标记下的可见内部点验证不能替代当时小直径针的感知验证。新五目标 pilot 是不同运行和数据版本，见本文顶部链接；不要把两组数据合并成同一次实验。

旧的内部三点精度结果（视觉 30.6 s、最大误差 13.36 μm、1861 帧）仍保存在 [旧精度实验](runs/历史记录/01_current/three_waypoint_precision_20261016/comparison.md)，但它不满足“三个点都在眼底”的新要求。

此前 39.7857 s 的[眼底直连实验](runs/历史记录/01_current/three_waypoint_fundus_20261016/comparison.md) 没有回撤约束；55.1190 s 的[回撤实验](runs/历史记录/01_current/three_waypoint_retract_20261016/comparison.md) 有回撤约束但按精确模型执行。两者都不是当前要求的视觉 RRT/PRM 导航成功记录。
