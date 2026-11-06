# 严格视觉多目标眼底导航

> 本文记录的历史 `--strict-visual` 协议禁用场景目标逆解，但使用仿真分割和理想深度。当前 v9 已新增不使用物体 ID 的 [严格 RGB-D 导航](STRICT_RGBD_NAVIGATION.md)，两种观测条件需分别报告。

2026-10-22：`run_waypoint_comparison.py --strict-visual` 已实现按给定输入顺序连续访问眼底目标，协议和执行模式均为 `strict_visual_multitarget`。完整三目标实跑 3/3 通过，独立终点误差为 6.634 / 6.085 / 6.268 μm；实体遮挡恢复组也为 3/3，误差为 6.382 / 6.442 / 6.283 μm。每点通过连续三帧零指令速度、独立 <20 μm 验收及全程速率约束检查。详见 [实验及组会材料](reports/20261022/README.md)。这些是固定三点的少量仿真 episode，长时间随机采集尚未开展。

这里的眼底目标位于眼球后部内表面代理上，模型没有独立视网膜层。输入提供待呈现标靶及访问顺序，实验不包含自主病灶识别、同时检测多个目标或自主排序。

## 信息与控制边界

目标中心来自当前显微镜 RGB、仿真分割和渲染深度的球面拟合。物理针尖由当前可见针杆圆柱多点及实体端面/真实尖端小球重建；均匀针杆本身不能确定进针位置。默认图像实际分辨率为 1920×1440。观测不足、拟合不可靠或测量丢失必须产生明确结果。

固定相机内外参、RCM 位置、器械及标靶尺寸作为标定输入保留。正向运动学的执行 Jacobian 将测得的误差转换成轴指令；关节状态、限速、限加速度、眼内约束和离壁安全检查仍可使用运动学。它们不提供场景目标的轴终点或预计算轨迹。严格入口安装运行时保护，禁止控制路径调用 `target_task()` 和 `nominal.inverse()`；场景目标 XYZ 只用于放置标靶及独立验收，不用于特征校准或生成运动目标。

这是标定几何与理想 RGB-D 下的运动学仿真。仿真分割、深度、标定信息属于专家感知权限，不能称为 RGB-only 导航、实机精度或组织接触安全验证。

深度现在通过 [OCT 仿真接口](OCT_SIMULATION.md) 传递，带米单位、注册相机、仿真时间戳、帧标识与有效掩码。上述四组多目标实验早于该包装接口接入，使用相同的理想渲染深度；其原始数据不能追溯改称真实 OCT 或含完整深度序列。

## 连续访问的视觉阶段

| 阶段 | 可动轴与视觉目标 | 完成条件 |
|---|---|---|
| `visual_align` | C 固定，A/B 对准。按实测目标相对 RCM 的方向和当前实测针尖到 RCM 的半径构造虚拟点，每帧重新测量与纠偏 | 可控方向测量误差满足该阶段阈值，连续三帧零指令速度；当前对准阈值为 1 μm |
| `visual_approach` | A/B 固定，C-only 接近当前测得目标 | 可控投影误差 <8 μm，连续三帧零指令速度；随后独立真值三维误差 <20 μm |
| `visual_retract` | 到下一目标前，保持上一标靶。取上一停稳时实际观测的针尖和针轴，沿该轴后退 2 mm 构造虚拟点，固定 A/B，以当前针尖测量闭环控制 C | 可控投影误差满足阈值并连续三帧零指令速度；完成后才呈现、测量和对准下一目标 |

回撤虚拟点来自上一实际测量，不读取场景目标逆解，也不调用参数化模型轨迹。对准和回撤的虚拟点是运动阶段目标；只有真实标靶处的接近完成才计入目标到达。C-only 的投影误差不能证明完整三维误差合格，独立验收会拒绝横向偏差过大的结果。

默认回撤 2 mm、转向时保守针尖离壁余量 1 mm，可分别用 `--retract-mm` 和 `--transfer-clearance-mm` 调整。这些是仿真参数，不是整根器械的组织间隙或临床安全距离。严格运行的 `config_snapshot.json` 移除旧 `navigation.fundus_transition`，在 `navigation.strict_visual_navigation` 保存测量驱动阶段的回撤与安全配置。

## 遮挡模式

默认 `navigation.visual_servo.occlusion_handling="stop"`，不可观测即失败并保持；不会用场景真值补齐观测。仿真保持不能代表实机制动已经验证。

显式设为 `static_memory_reobserve` 才启用静态目标记忆和有界重观测。它假设相机固定、目标静止；目标需先获得稳定观测，记忆最长 1 s、期间针尖累计移动不超过 0.5 mm，正常记忆跟踪仍要求新的可靠针尖测量。每个视觉阶段允许至多 3 次重观测用的受限 C 轴回撤，每次至多 4 s、0.5 mm，重新获得连续 3 帧有效观测后再恢复跟踪。该回撤使用受限编码器动作，不算目标到达，也不能预测不可见针尖。隐藏期间的目标运动无法即时检测；本功能不提供动态目标或任意完全遮挡的恢复保证。

报告必须保留实际 `occlusion_handling`、目标来源和记忆年龄，不能将记忆位置标成当前观测。不同遮挡模式的结果应分别报告。

## 复现与验证

从仓库根目录、使用已安装依赖的 `.venv` 运行。云端无窗口渲染可先设置 `MUJOCO_GL=egl MPLBACKEND=Agg`；输出目录应为新目录。

```bash
.venv/bin/python run_waypoint_comparison.py \
  --strict-visual --methods visual_servo --planner-feedback vision \
  --targets examples/three_waypoints.json --output-dir runs/strict_three_demo
```

此命令按文件输入顺序逐个呈现三点。采用其他 3–5 点序列时替换 `--targets`。默认任务为 `visual_current_target`，策略任务文本不含目标坐标、ID、顺序或 seed。相机和分割渲染可能使全流程耗时较长；过程中第一点的接近或单阶段通过不代表整集通过。

以下短验证检查控制、阶段组织和导出契约，不能代替完整导航：

```bash
.venv/bin/python verify_visual_geometry_direction.py
.venv/bin/python verify_visual_motion.py
.venv/bin/python verify_multitarget_visual.py
.venv/bin/python verify_target_tracking.py --output runs/strict_checks/target_tracking.json
.venv/bin/python verify_occlusion_fixture.py --output-dir runs/strict_checks/occlusion_fixture
.venv/bin/python verify_dataset_collection.py
```

`verify_multitarget_visual.py` 的普通短测使用控制器替身验证阶段、给定顺序、接口隔离及动作对齐；不能据此报告渲染导航精度。遮挡夹具验证使用独立物理几何干预；无渲染调度测试可加 `--skip-render`，必须标明跳过了渲染检查。

实际运行结束后，再检查该次结果：

```bash
.venv/bin/python verify_waypoint_comparison.py --run-dir runs/strict_three_demo
```

| 证据 | 能支持的结论 |
|---|---|
| 短测试通过 | 被覆盖的测量/控制接口、阶段组织或数据契约成立 |
| `dataset_verification.json` 为 PASS | 该次文件、对齐及审计检查通过；失败导航仍是失败 |
| `run_reports.json` 中每点 PASS、停稳与独立 <20 μm 验收通过，且全程约束通过 | 该次给定序列的严格仿真导航成功 |
| FAIL、NOT_RUN、进程中断或仍在运行 | 应分别报告失败原因、未尝试目标和未完成状态，不能补算成功 |

重点核对 `experiment_protocol.json`、`config_snapshot.json`、`source_sha256.json`、`order_selection.json`、`visual_servo/report.json` 与 `run_reports.json`。严格顺序报告应为 `GIVEN_TARGET_ORDER_NO_SCENE_ROUTE_PLANNING`，无预先目标测量建图或候选路线。数据审计中的 `perception.*` 记录测量来源、控制状态、记忆年龄、估计/独立真实误差、当前目标是否观测到，以及控制帧号和遮挡事件；这些字段不进入 `policy_inputs`、`policy_inputs_no_image` 或 `action_labels`。

## 建设与分层评价框架

本说明参考用户指定的 [Supervisor-Skills `benchmark-paper-template`](https://github.com/HKUSTDial/Supervisor-Skills/blob/a20d76ef3979b804d7dbf599d577826957e0a8a9/skills/benchmark-paper-template/SKILL.md)，固定版本 `a20d76ef3979b804d7dbf599d577826957e0a8a9`。引用其 Construction Pipeline 和 Evaluation Framework 组织原则，用于区分可复现的数据建设、质量控制和细粒度能力边界；它不是本项目的导航实现或性能证据来源。

后续建设按目标集合和重置姿态冻结场景、配置原始字节与哈希，再采集对齐 RGB/state/action 和独立审计标签；同一目标集合的所有方法、seed 必须留在同一 split。当前优先完成严格导航，尚未宣称长采集、策略训练或去图像消融已经完成。

评价分别报告逐目标到达与停稳、整集完成率、真实终点误差、限速/离壁违规、失败阶段及原因；遮挡按可见、部分/完全遮挡和其他不可观测条件分层，另列记忆使用与重观测结果。同一目标可经历多种可见性条件，各组不必互斥。当前仿真结果的适用范围限于给定顺序、标定 RGB-D 和眼球后部表面代理。
