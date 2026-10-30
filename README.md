# RCM 机械臂 MuJoCo 仿真

当前默认模型为根据实物扫描、照片和运动视频修正的 **v9 模型**：白色主壳上缘保持固定，银色圆角端盖、底缘和长罩作为同一个部件升降；保留 A/B/C 编码器换算和驱动接口。整体静态姿态采用相对原扫描前端抬高约 40° 的照片估计修正，尚非实机标定。

目录已清理旧运行结果、缓存和旧预览；研究代码、示例配置、网格重建及回归基线全部保留。当前模型预览与完整闭环测试集中在 [reports/20261030](reports/20261030/README.md)，旧结果可从 [清理前版本](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/tree/42a0b2855b343d4efced99dee615cfa6a5d2be4c) 查看。

## 启动当前模型

```bash
python -m pip install -r requirements.txt
python run_reference.py
```

无窗口检查及图像导出：

```bash
python run_reference.py --no-viewer
python run_reference.py --demo abc --sample-time 5 --no-viewer
python run_reference.py --export scene.xml --no-viewer
python run_reference.py --render runs/current_model.png --no-viewer
python run_reference.py --verify
python verify_current_model.py
```

根目录 `scene.xml` 和默认 `run_reference.py` 均使用当前扫描模型；`mujoco_model/` 保存网格、建模配置、修复脚本、驱动适配、证据及验证报告。直接查看 XML 可运行 `python -m mujoco.viewer --mjcf scene.xml`；从动关节的交互同步请使用默认 Python 入口。具体参数见 `python run_reference.py --help`。

## v9 眼球注入场景

根 `scene.xml` 是不含眼球注入的机械模型基础文件。导出包含 eye shell、retina、lens、lesion 和 `eye_internal` 相机的可移植场景：

```bash
python export_v9_eye_scene.py
python -m mujoco.viewer --mjcf v9_eye_scene.xml
```

`v9_eye_scene.xml` 使用相对路径 `mujoco_model/assets` 引用扫描网格，不依赖当前机器绝对路径。

眼球场景的前端按实物照片重建为圆角白壳、金属套筒、浅色带筋连接件、六角针座和双段针具。细针长 4 mm、半径 0.03 mm，近端管长 22 mm、半径 0.25 mm；针座及壳体尺寸是照片估计，未实机标定。重建仅替换此注入场景的 C 滑台显示外形，原扫描网格保留。具体说明见 [前端建模说明](V9_FRONT_TOOL.md)。

该眼球场景的扫描外壳采用去毛边、平滑后的 `*_smooth.obj` 显示网格，原扫描及封闭基线保留；小裂隙和碎片经显示重建处理，不作为测量或碰撞几何。

俯视长罩长边已通过 `v9_alignment.json` 中约 -17.103° 的世界 Z 轴偏航对齐到世界 X 轴；三视图仍沿世界坐标轴正交投影。该刚体变换同步作用于针具、RCM、眼球和相机，原驱动安装配置与关节零位不变。

前端安装方向进一步依据原始 STL 的套筒区域拟合，结合零位俯视平行约束记录在 `v9_tool_mount.json`。针座、粗细针及虚拟 C 轴使用同一安装坐标；参考 RCM 与眼球入口同步调整，原扫描壳体的零位姿态保留。拟合及照片尺寸均未实机标定，详情见前端建模说明。

根据用户正面照片，前端白壳、金属套筒和针具采用同一中心线，壳体 XY 安装偏置为零；扫描网格面积中心不再用作偏置依据。

按新增侧照，前端白壳改为正面宽、侧面薄的带底部开口结构，针座侧视倾角暂估 55°；补建透明侧夹、下支架、银罩侧面条纹/紧固件和灰色下机壳/接口座。参数在 `v9_side_profile.json`，均为照片外观估计，仍保留针具居中和软管删除状态。

直接打开 XML 时默认显示与物理针同尺寸的 group 2 副本；传感器使用 group 3 的物理针。场景已将针尖改为朝向扫描壳体外侧，眼球、病灶和相机一并采用新的刚体坐标映射。v9 适配器将逻辑任务的 A/C 映射为反号显示关节（包含对应关节/执行器范围）；原驱动与 legacy 编码器约定保持原样。导出的 XML 原生关节滑块不是硬件指令，使用任务接口应通过 `RCMSceneV9`。

```bash
MUJOCO_GL=egl python export_v9_eye_scene.py
MUJOCO_GL=egl python verify_v9_front_tool.py
MUJOCO_GL=egl python verify_rcm_scene_v9.py
```

## 当前 v9 的视觉闭环测试

2026-10-30，直接使用 `RCMSceneV9` 的相机观测和运动接口，完成单目标从零位到目标的闭环测试：A/B/C 初始均为零，针尖距目标约 19.75 mm；679 步、33.9 秒仿真后误差降至 6.03 μm，连续 3 帧停稳，速度和加速度检查通过。控制及停止使用当前图像、理想注册深度和分割重建的几何测量，场景真值仅用于记录及独立终点验收。此结果不代表真实硬件精度，也未验收 v9 多目标、组织接触或遮挡恢复。

- [完整过程视频（41 秒，包含起止停留）](reports/20261030/v9_zero_to_target/v9_zero_to_target_full_test.mp4)
- [测试报告、轨迹与复现命令](reports/20261030/README.md)
- [当前完整三视图](reports/20261030/v9_model/three_views.png) · [前端正面](reports/20261030/v9_model/front_tool.png) · [侧照修订](reports/20261030/v9_model/side_profile.png)

该测试通过独立录制脚本直接驱动 v9；`run_visual_servo.py` 的控制后端仍为 legacy，选择 v9 viewer 只改变显示后端。

- [模型与修复说明](mujoco_model/README.md)
- [实机标定工作流](mujoco_model/CALIBRATION.md)
- [银罩整体运动修正对照](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/mujoco_model/outputs/silver_integral_comparison.gif)
- [侧视图](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/mujoco_model/outputs/static_side_reference.png) · [三视图](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/mujoco_model/outputs/three_views/three_views.png)
- [端盖整体运动验证](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/mujoco_model/outputs/silver_integral_validation.json) · [白壳修复验证](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/mujoco_model/outputs/seam_validation.json)

`simulation_driver/` 原始 29 个文件保持不变。模型加载的是 `mujoco_model/vendor/simulation_driver/` 中提交 `c2a3ffc` 的固定副本，以便复现现有验证；修改根目录驱动不会自动改变该固定副本。仿真入口不连接设备或加载实验室 ROS 网络配置。

当前模型仍为外观与运动学原型：银罩行程比例、电机一阶响应、质量惯量和 RCM 尚未实机标定，碰撞关闭，ROS 实机通信未验证。验证中的微米级误差描述的是软件模型内部一致性，不代表硬件精度。

## 历史程序入口

原模型和启动脚本保留为 `legacy_scene.xml` 与 `run_legacy_reference.py`，可复现此前的眼底导航、视觉伺服和 OCT 研究。`run_navigation.py`、`run_visual_servo.py`、`run_waypoint_comparison.py` 及 `rcm_reference.Reference` 继续使用旧研究场景；当前 v9 的独立单目标验证见上文，历史导航入口尚未切换到 v9。下面的历史结果与命令只适用于该旧场景。

---

## 历史导航与 OCT 研究记录

2026-10-22 更新：已完成按给定顺序 P1→P2→P3 的严格视觉导航和实体遮挡恢复实验。零位出发三点独立终点误差为 **6.63 / 6.09 / 6.27 μm**；遮挡恢复组为 **6.38 / 6.44 / 6.28 μm**，均到达并停稳。目标在眼球后部内表面代理上，未建立独立视网膜层。这是理想 RGB、分割和深度的仿真结果。

当前入口与限制见 [严格视觉导航](STRICT_VISUAL_NAVIGATION.md)、[OCT 仿真接口](OCT_SIMULATION.md) 和 [实验及组会材料](reports/20261022/README.md)。OCT 深度来源明确标注为 MuJoCo 理想注册深度，尚未接真实 OCT，也未验证实时 20 Hz。

严格视觉连续三点导航（依赖安装后执行）：

```bash
python run_waypoint_comparison.py --strict-visual --methods visual_servo --planner-feedback vision --targets examples/three_waypoints.json --output-dir runs/strict_three_demo
```

云端无窗口运行可先设置 `MUJOCO_GL=egl`。Windows 本地通常不需要此设置。输出目录须为新目录。`run_navigation.py` 是保留的模型导航入口；默认多方法对比仍保留历史混合流程，严格视觉实验须显式使用上面的参数。

## 安装与运行

仓库保留导航代码、配置、场景、目标示例、验证脚本和说明。Python 虚拟环境、历史运行结果、验收产物、预览图和备份不上传；下文的历史结果路径仅适用于保存过这些结果的本地目录。

在项目目录安装依赖并启动导航：

```bash
python -m pip install -r requirements.txt
python run_navigation.py
```

无窗口导航、视觉伺服和多目标对比的入口分别为：

```bash
python run_navigation.py --no-viewer
python run_visual_servo.py
python run_waypoint_comparison.py
```

可通过 `--help` 查看参数。运行结果会重新生成在 `runs/` 下，并由 `.gitignore` 排除。图形窗口需要本地桌面环境；云端运行可使用支持无窗口模式的入口，并按运行环境配置 MuJoCo 渲染。

本地已保存的汇总可通过 `runs/打开结果.html` 或 `runs/最新结果/` 查看；GitHub 仓库不包含这些文件。

历史五目标 `visual_current_target` 混合流程 pilot 及其限制保留在下文；旧失败记录也保留，最新严格视觉结果见本页开头。器械前端细针段外径 0.060 mm、长度 4 mm。点间回撤 2 mm、转向时针尖至少离壁 1 mm 的仿真约束保留。

随机眼底目标生成和 train/validation/test 数据采集见 [RANDOM_FUNDUS_DATASET.md](RANDOM_FUNDUS_DATASET.md)。

多目标共享访问顺序、视觉伺服/RRT/PRM 对比和 RGB/state/action 数据格式见 [WAYPOINT_NAVIGATION.md](WAYPOINT_NAVIGATION.md)。运行 `python run_waypoint_comparison.py` 可生成实验、对比表、数据集和路径图。默认 `--planner-feedback vision` 使用仿真 RGB、深度和标记分割，依赖相机与标靶几何标定，不是实机 RGB-only 感知。使用 `--planner-feedback model` 可复现模型目标规划基线。随机目标及数据集说明见 [RANDOM_FUNDUS_DATASET.md](RANDOM_FUNDUS_DATASET.md)。

## 当前默认视觉模式（2026-10-22）

`run_visual_servo.py` 默认使用 `position_feedback="geometry"`，相机实际渲染分辨率为 1920×1440。固定相机内外参、RCM 位置和针具尺寸作为标定输入；针轴从可见近端针杆圆柱的多点深度拟合，轴向位置由可见实体端面或真实尖端小球约束，再重建同一个物理针尖。均匀针杆不能单独确定进针位置；观测不足或拟合不可靠会报告失败，不再将单个锥面像素当作尖端。目标中心由当前 RGB、分割和深度的球面拟合得到。几何测量不读取动态针尖/目标位置，也不使用已知目标姿态做特征校准。

控制使用归一化的三轴耦合 Jacobian、限速/限加速度和接近减速。默认测得的位置误差阈值为 8 μm，并要求连续三帧误差达标且指令速度为零；结束后再用独立 MuJoCo 真值按 **<20 μm** 验收。测量阈值、停稳和独立验收是不同检查，不能只看视觉条件。这里仍是标定几何和理想 RGB-D 的运动学仿真，不是实机精度结果。历史 `image_depth` 模式保留，旧实验的像素/深度阈值及逐目标姿态标定只适用于该模式。

```bash
python run_visual_servo.py --no-viewer --output-dir runs/geometry_navigation
python verify_visual_geometry.py --output-dir runs/geometry_checks
python verify_visual_geometry_direction.py
python verify_visual_motion.py
python verify_visual_servo.py --output-dir runs/visual_checks
```

云端可在这些命令前设置 `MUJOCO_GL=egl MPLBACKEND=Agg`；图形窗口仍需桌面环境。`verify_visual_geometry.py` 检查代表姿态、隐藏尖端、轴向退化和真值接口隔离；方向检查覆盖端面方向歧义，运动检查覆盖耦合控制、停稳与失败保持。完整导航结果应以该次生成的报告为准，不能把短测试或历史多目标 pilot 的误差当作当前全周期结果。

默认多目标眼底流程的回撤/转移仍使用已知目标构造解析轨迹，仅最后固定方向的 C 轴接近使用视觉反馈；因此整个流程仍是混合先验导航。`--strict-visual` 已支持给定顺序的 3–5 点连续访问：以当前 RGB-D 测量驱动 A/B 对准、固定 A/B 的 C 轴接近，以及点间沿实测针轴回撤，不生成场景目标逆解或解析转移轨迹。它不进行自主病灶识别或访问排序，仍使用相机、RCM、器械标定和运动执行 Jacobian。默认不可观测即停止；可选的有界静态目标记忆须另行声明假设。协议、复现命令与结果判读见 [严格视觉导航](STRICT_VISUAL_NAVIGATION.md)。目前尚无已发布的严格三点 3/3 完整通过结论；下方 2026-10-19 pilot 是旧控制版本的历史记录。

## 最新多目标视觉 pilot（2026-10-19）

数据目录：`runs/历史记录/05_random_datasets/pilot_n5_visual_current_target_v1/`。使用 5 个随机眼底目标、20 Hz 采集，共 5,692 帧。三种仿真导航轨迹均 PASS；遮挡标签显示有部分针体遮挡，但没有完全遮挡帧。

| 方法 | 结果 | 帧数 | 运动时间 (s) | 最大到达误差 (μm) | 部分遮挡帧 | 完全遮挡帧 |
|---|---:|---:|---:|---:|---:|---:|
| 视觉伺服 | PASS | 1,580 | 78.739 | 5.30 | 604 | 0 |
| RRT（seed 7） | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 |
| PRM（seed 7） | PASS | 2,056 | 102.750 | 14.60 | 979 | 0 |

任务模式有两种：`registered_coordinate_prior` 会将假设已配准的机器人坐标写入任务文本（当前示例坐标来自仿真，不是实测 OCT）；默认 `visual_current_target` 的策略输入文本不含目标坐标、目标 ID 或访问顺序。这个 pilot 中标靶逐个呈现，访问顺序由脚本依据测量选定并由各方法共享，不能证明策略自主识别并排序多个病灶。

专家与策略输入也要区分：专家使用仿真分割/渲染深度和标定信息；眼底视觉伺服还按已知目标姿态标定特征，并在回撤/转移阶段使用解析运动，RGB 在该阶段只记录、不参与纠偏，最后的 C 轴接近才使用视觉深度闭环。因此示范不是全程纯单目导航。已导出去图像输入包和对应动作标签，但项目尚未实现策略训练/推理或去图像消融；该 pilot 只有一个目标集合，不是完整 train/validation/test 数据集。

字段、轨迹、输入泄漏及遮挡标签验证结果见 `runs/历史记录/05_random_datasets/pilot_n5_visual_current_target_v1/episodes/train/episode_000000/dataset_verification.json`。复验命令（从本目录运行）：

```powershell
.\.venv\Scripts\python.exe verify_waypoint_comparison.py --run-dir runs\历史记录\05_random_datasets\pilot_n5_visual_current_target_v1\episodes\train\episode_000000
```

本版默认不启用 A/B/C 位置限位，默认针尖目标改为眼底路径点 P3 `[1.92,8.478,22.043]` mm；原深部示例 `[9.5,-1.5,16]` mm 仅作为历史回归目标保留。此前记录的编码器区间仅保留为观测数据；它们不能代表全行程，也不能用来判断眼底目标不可达。

```bash
python -m pip install -r requirements.txt
python run_navigation.py
```

程序执行：瞄准 → 进针 → 停留 → C 回零 → A/B 复位。P 暂停/继续，关闭窗口保存报告和 CSV。无窗口运行：

```bash
python run_navigation.py --no-viewer
python run_navigation.py --target-robot 9.5 -1.5 16 --solve-only
python run_navigation.py --targets examples/targets.json --no-viewer
```

执行方式是运动学回放 `mj_forward`，用于核对目标求解、针轴与 RCM 的关系及导航流程。反馈控制、驱动器响应和实机通信还没有接入。完整接口说明见 [NAVIGATION.md](NAVIGATION.md)。

## 零位、针和编码器

已确认的零位 ABC 编码器为 `[0,1466018,893518]`，E 为 0。启动和 R 复位都回到这个状态，相对任务坐标为 `[0,0,0]`。

| 轴 | 换算与相对坐标 |
|---|---|
| A | `A_deg = EA*0.0018`；相对零位角度写入 `A_azimuth`，单位 rad |
| B | `dB_mm = EB/K`；经连杆公式求 alpha；`qB = radians(alpha-alpha_zero)` |
| C | `insertion_mm = (EC-893518)/K`；`qC = insertion_mm/1000`，单位 m |

`K=63822.71488 count/mm`，当前 B 零位角 `alpha_zero≈54.236005250°`。默认正向计数，原点/反向参数仍在配置文件中。

针尖到 RCM 的距离：

```text
L_tip_mm = needle_tip_distance_at_reference_mm + insertion_mm
needle_tip_distance_at_reference_mm = 4
```

4 mm 是暂定的零位针尖到 RCM 偏置，不是细针段总长或 C 轴全行程。前端细针段总长为 4 mm，因此零位时其近端变径点与 RCM 重合；C 运动时针杆、变径点和针尖一起滑动，RCM 保持固定，针尖到 RCM 的总距离 L 仍须非负。

原深部目标距 RCM 约 18.668 mm，需要相对任务 A≈30.699723°、B≈18.372721°、C≈14.668155 mm。关闭位置限位后可以执行该目标。`run_navigation.py` 使用整数编码器目标，所以终点具有量化误差。

前端细针段按用户最新尺寸设为外径 0.060 mm、半径 0.030 mm、总长 4 mm；MuJoCo 编译后的圆柱半径为 0.000030 m、半长由端点确定。模型分成近端杆体、细针段圆柱和局部针尖三段：细针段圆柱长度 2.5 mm，末端 1.5 mm 使用八边形渐缩网格近似针尖，尖端半径 0.020 mm。1.5 mm 针尖斜面长度仍是仿真暂定值，未按实物 CAD 校准。近端杆体暂沿用直径 0.5 mm、长度 22 mm。针段尺寸与零位 4 mm 针尖偏置是独立参数，尺寸来源和范围见 `needle_dimension_reference`。尖端辅助球仅用于分割观测，不代表额外实体；物理标定标志保持未标定。

## B 轴反解分支

连杆公式仍为：

```text
dB = 98.48 - 60*cos(alpha) - sqrt(52^2-(60*sin(alpha)-16)^2)
ad = 98.48-dB
ac = sqrt(ad^2+16^2)
gamma = atan2(16,ad)
beta = acos((60^2+ac^2-52^2)/(2*60*ac))
alpha = degrees(gamma+beta) 或 degrees(gamma-beta)
```

只使用 `gamma+beta` 会把一部分眼内目标反解到另一个角度。新版本会验证两个候选解是否满足原连杆公式，再按目标姿态或当前姿态选择分支。目标预检得到的角度会保存在计划中，执行和渲染不再仅凭编码器重新猜分支。

`RCMModel.alpha_from_b()` 不提供参考角时保留原程序的默认分支。`Reference.task_from_encoders(counts, reference_task=...)` 可显式提供参考姿态；省略时使用当前模型姿态。跨越 B 位移极小点时，仅靠一个编码器位置不能唯一确定分支，需保留姿态/运动历史；这次验证的是数学模型的分支处理，未验证实物能否跨越该构型。

针尖恰好位于 RCM 时保留当前 A/B 方向；目标位于机器人 Y 轴时保留当前 A 方位角。这两种情况下位置本身不能确定全部朝向，但无需拒绝位置目标。B 位移极小点及入口附近的整数计数量化会选取邻近有效计数，保持连杆有实数解、针尖总距离非负。

## 位置限位与演示幅度

`config.json` 中 `measured_limits.enabled=false`。生成的三个任务关节都是 `limited="false"`；手动操作、编码器输入和目标导航不再被原样本区间截断。

旧观测数据仍保存在 `measured_limits.encoder_counts`：

| 轴 | 旧观测区间 | 数据意义 |
|---|---|---|
| A | -3152～3759 count | 观测窗口，不限制当前位置 |
| B | 1282350～1669860 count | 观测窗口，不限制当前位置 |
| C | 895599～1027792 count | 观测窗口，不代表最大进针量 |

独立演示默认 A±5°、B±2°、C 0～2 mm，属于展示动作幅度。`demonstration.ui_ranges` 是查看器的显示区间，也不启用位置约束。速度/加速度参数继续控制轨迹播放时间；它们与位置限位是不同设置。

可选限位机制仍可通过配置启用，但当前眼球任务和验收使用无限位配置。保留有限输入、连杆几何、非负针尖距离和眼球目标范围检查。

## 三轴演示与手动操作

```bash
python run_legacy_reference.py
python run_legacy_reference.py --demo abc
python run_legacy_reference.py --auto-reach --no-viewer
python run_legacy_reference.py --pose 30.7 18.37 14.67 --no-viewer
python run_legacy_reference.py --encoders 0 1466018 893518 --no-viewer
```

| 按键 | 功能 |
|---|---|
| 1 / 2 / 3 / 4 | A / B / C 单轴演示或依次演示 |
| 左 / 右方向键 | A 减 / 加 0.5° |
| 下 / 上方向键 | B 减 / 加 0.5° |
| PageDown / PageUp | C 退 / 进 0.02 mm |
| 空格 / 回车 | 只瞄准 / 瞄准后到达目标 |
| R / P | 复位 / 暂停或继续 |
| M / V | 外壳显示 / 切换固定视角 |

手动步长在 `manual_steps` 中调整。macOS 被动窗口使用 `mjpython`。GUI 和真实键盘事件需要桌面环境，本次没有运行实时 GUI 验收。

## 场景与坐标

任务关节为 A 转向、B 摆动、C 沿针轴滑动；4 个从动关节仅用于等效连杆同步。针轴和 C 导向平行，姿态轴线都通过固定 RCM。

MJCF 长度单位为 m，输入和配置长度为 mm。世界变换：

```text
p_world_mm = R_install*diag(1,-1,-1)*p_robot_mm + rcm_world_mm
R_install = Rz(z)*Ry(y)*Rx(x)
```

参考眼球外形为 24×23×24 mm 的椭球。当前 RCM 在其表面，零位针轴指向眼球中心。默认 P3 为眼球后部表面代理上的示意目标，并非实测病灶或精确的视网膜表面。

```bash
python run_navigation.py --target-world 9.5 12.3743686708 -10.2530483272 --no-viewer
python run_legacy_reference.py --export legacy_scene.xml --no-viewer
python -m mujoco.viewer --mjcf legacy_scene.xml
```

眼球内部还提供固定虚拟相机和橙色目标标靶（中心小球及四个定位点）；全景视角中青色小球/短轴线表示相机位置与朝向，可运行仿真视觉伺服：

```bash
python run_visual_servo.py --no-viewer --output-dir runs/历史记录/04_sessions/demos/visual_servo_demo
python verify_visual_servo.py --output-dir acceptance/visual_servo
```

在有桌面环境时省略 `--no-viewer` 可直接打开 MuJoCo 窗口观察针尖运动：

```bash
python run_visual_servo.py --output-dir runs/历史记录/04_sessions/demos/visual_servo_viewer
python run_visual_servo.py --camera overview --output-dir runs/历史记录/04_sessions/demos/visual_servo_overview
python run_visual_servo.py --camera free --output-dir runs/历史记录/04_sessions/demos/visual_servo_free
```

`--camera free` 支持鼠标旋转、平移和滚轮缩放；运行中按 `F` 切换自由/固定视角，按 `V` 循环固定视角。观察相机只影响显示，不影响眼内相机视觉控制。窗口右下角始终叠加控制相机画面：青色十字为目标中心、绿色十字为针尖、白线为图像误差；左上角显示实时像素误差、深度误差和 A/B/C 状态。

视觉伺服从相机 RGB 中检测橙色目标质心，并用 MuJoCo segmentation/depth 观测目标和专用针尖标记。A/B 根据图像平面误差和有限差分 Jacobian 调整，C 根据校正后的深度误差调整；三轴均从当前姿态逐帧运动并受速度、加速度限制。停止条件只使用校正后的像素和深度误差，三维针尖真值仅用于事后验收。已知目标姿态只用于启动前标定有限标记尺寸、遮挡和像素离散造成的目标图像特征偏置，不直接设置实际 C。它仍是离线 MuJoCo 运动学仿真，不包含真实相机标定、驱动器响应或组织接触验证。

独立 XML 已按当前无限位配置重新生成。配套脚本每次从配置重建模型。原生查看器拖动从动关节时可能暂时破坏同步，配套脚本会保持运动学闭合。

## 验收

```bash
python run_legacy_reference.py --verify --report verification.json
python verify_navigation.py --output-dir acceptance
```

导航验收直接从编译后的眼球椭球生成 512 个内部点、70 个表面点及 11 个回归目标，总共 593 个目标；不再从原关节观测窗口的正解生成一批小范围点，再据此宣称覆盖眼球。每个目标都会经过求解和独立 MuJoCo 针尖位置核对；其中 49 个目标执行完整导航循环。

回归包含原深部目标、先前误拒绝的 B 分支目标、B 位移极小点、RCM、Y 轴、入口附近和眼球中心。还检查整数编码器量化、RCM 固定、针轴通过 RCM、针段长度保持配置值、连杆闭合、安装变换、速度/加速度、完整回零以及无效输入的整批预检。

`verification.json` 另有 2026 个运动学采样姿态、独立轴演示、编码器反向/原点及可选限位机制检查。结果均来自 MuJoCo 3.3.7 / Python 3.12.14，并通过 Python 3.11 语法解析。具体误差和样本数量见 JSON 报告。

采样通过反映本数学模型的覆盖和一致性；未接入组织接触、驱动器动力学或实机精度测量。当前完整循环仍在 C=0、针尖距 RCM 4 mm 时调整朝向，验收没有声称整根针或中间轨迹都避开组织。

## 文件

| 文件 | 内容 |
|---|---|
| `config.json` | 已确认零位、换算公式、针/安装参考值及默认关闭的位置限位 |
| `rcm_recovered_model.py` | 原公式和带参考角的 B 反解 |
| `rcm_reference.py` / `legacy_scene.xml` | MuJoCo 生成、运动学、编码器和场景 |
| `run_legacy_reference.py` | 单轴演示、手动控制、静态导航与导出 |
| `run_navigation.py` / `rcm_navigation.py` | 目标接口、整批预检、完整循环和日志 |
| `verify_reference.py` / `verification.json` | 运动学与兼容性验证 |
| `verify_navigation.py` / `acceptance` | 眼球空间验收、目标解和完整轨迹 |
| `NAVIGATION.md` / `examples` | 运行说明与目标输入示例 |
| `preview` / `runs` | MuJoCo 实际渲染和示例运行结果 |
| `requirements.txt` | MuJoCo 与 NumPy 依赖 |

## 针与眼球几何诊断（2026-10-09）

运行 `python verify_needle_geometry.py`，结果写入 `needle_geometry_verification.json`。`Reference.needle_eye_geometry()` 使用 MuJoCo 实际椭球的位置、姿态和半轴，与当前针轴中心线求交；不改变针具尺寸，也不启用位置限位。

| 状态 | 细段中心线在眼内 | 粗段中心线在眼内 | 入口处针段 |
|---|---:|---:|---|
| C=0 | 4 mm | 0 mm | 前端细针段/变径点在 RCM |
| C=2 mm | 4 mm | 2 mm | 近端杆体 |
| C=8 mm | 4 mm | 8 mm | 近端杆体 |
| 原深部目标 | 4 mm | 约 14.668 mm | 近端杆体 |
| C=-4 mm | 0 mm | 0 mm | 针尖回到 RCM |

上述为 4 mm 前端细针段、4 mm 零位偏置下的中心线几何结果。整根针轴向滑动，总长不随进针变化；原深部目标仍会让部分近端杆体进入眼内。诊断中的直径在变径点采用较大直径，并不代表入口可通过该尺寸。

`metrics()` 和每个到达结果增加 `needle_eye_geometry`：RCM 是否在眼球表面、针轴入口与 RCM 偏差、针尖/变径点是否在眼内、两段入眼长度、入口对应的针段。CSV 逐帧记录对应信息，导航报告增加整段轨迹的最大粗段入眼长度、最大入口偏差与针尖位于眼球外的采样数。

诊断不把目标到达 PASS 当作组织接触验收。计算对象是中心线与参考外形，尚未检测有限半径针杆、入口孔径、套管或组织接触。针尖位于眼球外的中间采样会被明确记录，但不会自动新增限位或更改导航流程。

本次完整验收轨迹中有 259 个采样被严格椭球判定标为针尖在外；根据 CSV 针尖坐标独立复核，最大沿眼球中心射线的径向超出约 0.131601 µm，发生在整数编码器量化后的表面目标。原深部目标完整循环的该计数为 0。此计数应结合超出幅度解释，不代表 259 次独立动作或接触事件。
