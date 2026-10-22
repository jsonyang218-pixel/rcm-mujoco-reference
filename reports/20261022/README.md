# 2026-10-22 实验记录与组会材料

已完成给定三个眼底目标的连续严格视觉导航、数据采集和实体遮挡恢复。目标位于仿真眼球后部内表面代理；没有独立视网膜层。所有精度均是理想 RGB、分割、配准深度与运动学控制下的仿真结果。

## 已完成实验

| 实验 | 成功/计划目标 | P1 / P2 / P3 独立终点误差 μm | 数据帧 |
|---|---|---|---|
| 零位起始，严格视觉 | 3/3 | 6.634 / 6.085 / 6.268 | 1,289 |
| 实体遮挡，失观测即停止 | 0/3 | P1 失败，P2/P3 未运行 | 29 |
| 相同实体遮挡，静态记忆及重观测 | 3/3 | 6.382 / 6.442 / 6.283 | 669 |
| 新 OCT 仿真接口，P1 闭环接近 | 1/1 | 6.382 / — / — | 97 |

成功要求独立三维误差 <20 μm、连续三帧零指令速度以及运动约束检查通过。遮挡配对组从先前真实视觉回撤记录的同一姿态重置，并使用相同球体遮挡计划：P1 接近阶段第20帧开始持续8帧，P2 第20帧开始持续35帧。恢复组共24帧使用有界历史目标测量，完成一次回撤重观测；针尖始终要求当前可靠测量。目标固定、相机固定是本模式的假设。

失观测即停止的对照组保留失败结果，其仿真瞬停也未通过独立加速度检查；数据完整性通过不代表安全制动通过。自然针遮挡和独立实体球引起的不可见分别计数。这里只采集了少量固定目标 episode，没有完成随机大规模数据集、策略训练或泛化评估。

## 组会材料

- [完整中文报告](meeting/report.md)
- [六页离线 HTML 幻灯片](meeting/slides.html)：下载后用浏览器打开，方向键切换，可打印为 PDF。
- [指标与协议 JSON](meeting/metrics.json)、[逐目标 CSV](meeting/endpoint_metrics.csv)
- [严格三点预览视频](meeting/strict_three_preview.mp4)：按名义20 fps播放，不表示计算或设备实时速率。
- [误差图](meeting/figures/endpoint_errors.png)、[实际执行轨迹](meeting/figures/executed_strict_path.png)、[遮挡恢复时间图](meeting/figures/timeline_strict_occlusion_recovery_ep000000.png)

`meeting/evidence/` 包含配置、控制报告、原始 JSONL、小型 NPZ 和代码快照；`rawframes/` 保存11张未经修改的代表性 RGB 图像。完整四组数据共3,133帧、约935 MiB，完整 RGB 库仍在云工作区 `runs/multitarget_occlusion_20261022/`，不包含在 GitHub 下载包中。[原始目录索引](meeting/DATASET_LINKS.md) 是云工作区路径，下载到本机后需要重新采集或另行复制完整数据。策略输入仍为 RGB/state/task，深度和真值审计不进入这些输入包。

旧多目标实验的部分源文件未在采集时快照，材料如实列出了11项当前源码副本与采集哈希的差异，不能将当前副本视为当时源码。旧四组使用渲染深度，发生在 OCT 包装接口接入前，不包含原始 OCT 扫描或完整深度序列。

## 新 OCT 接口闭环验证

[OCT 闭环审计](oct_p1_final/oct_closed_loop_verification.json) 和 [导航数据审计](oct_p1_final/dataset_verification.json) 均为 PASS。本次从此前 P1 实际视觉回撤位置出发，运行 A/B 视觉对准及 C 视觉接近，8+89=97帧，仿真时间4.75秒，P1真实终点误差6.381807 μm。这不是重新完成一整轮零位三目标实验。

每帧深度均经 `SyntheticOCTProvider` 传递，并记录明确的仿真来源、唯一帧 ID、米单位、有效掩码和带阶段起始偏移的仿真时间。[逐阶段记录](oct_p1_final/visual_servo/P1/visual_approach/oct_observations.jsonl) 可复核。各阶段最后一帧 NPZ/JSON 随此目录提供，可用 `RegisteredOCTFrame.load()` 加载；全程原始深度序列没有导出。

实测渲染耗时中位数0.471秒/帧，最大0.578秒/帧；这只是本次机器上的渲染耗时，不是设备延迟，也不支持实时20 Hz结论。接口不是 OCT 成像物理仿真，没有真实 A/B-scan、设备 SDK、眼底层分割或硬件标定。接真实设备前仍需建立相机/探头注册、设备时钟和延迟接口，见 [OCT 接口说明](../../OCT_SIMULATION.md)。

该闭环的16份源码快照均与采集哈希匹配。[快照说明](oct_p1_final/snapshot_status.json) 记录了随后独立修复的一个恢复分支：虚拟回撤目标下，短暂针尖丢失后只需重获针尖即可恢复，不应要求标靶可见；该修复由完整10项恢复回归覆盖，本次 `stop` 模式闭环未执行该分支。

## 复现

从仓库根目录安装 `requirements.txt`。云端先设置 `MUJOCO_GL=egl`，本地 Windows 一般无需该变量。下列输出目录必须尚不存在。

```bash
python run_waypoint_comparison.py --strict-visual --methods visual_servo --planner-feedback vision --targets examples/three_waypoints.json --output-dir runs/strict_three_demo
python verify_waypoint_comparison.py --run-dir runs/strict_three_demo
```

实体遮挡恢复配对条件：

```bash
python run_waypoint_comparison.py --strict-visual --methods visual_servo --planner-feedback vision --targets examples/three_waypoints.json --config reports/20261022/meeting/evidence/run_configuration/recovery_config.json --start-task 6.481389210293787 2.847750084902797 17.8199835578327 --output-dir runs/recovery_demo
python verify_waypoint_comparison.py --run-dir runs/recovery_demo
```

停止对照将 `recovery_config.json` 换成同目录的 `stop_occlusion_config.json`，使用新的输出目录；预期导航失败，退出码为1。短 OCT 闭环：

```bash
python run_waypoint_comparison.py --strict-visual --methods visual_servo --planner-feedback vision --targets reports/20261022/oct_p1_final/targets.json --config reports/20261022/oct_p1_final/config_snapshot.json --start-task 6.481389210293787 2.847750084902797 17.8199835578327 --output-dir runs/oct_p1_demo
python verify_waypoint_comparison.py --run-dir runs/oct_p1_demo
```

本次新检查：OCT接口数值10组、实际渲染传感器12组、恢复控制10项均通过；既有运动控制8项、阶段组织4项及实际视觉伺服回归也通过。检查脚本为 `verify_oct_interface.py`、`verify_oct_sensor.py`、`verify_occlusion_recovery.py` 等。检查覆盖的具体范围见各自代码及此目录结果 JSON。

数据建设和分层评价组织参考用户指定的 [Supervisor-Skills / benchmark-paper-template](https://github.com/HKUSTDial/Supervisor-Skills/blob/a20d76ef3979b804d7dbf599d577826957e0a8a9/skills/benchmark-paper-template/SKILL.md)，仅作为组织框架。
