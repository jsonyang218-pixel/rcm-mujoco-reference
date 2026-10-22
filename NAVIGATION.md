# 无限位眼球目标导航

## 启动

进入解压后的 `rcm_mujoco_reference` 文件夹，激活现有 Python 环境：

```bash
python run_navigation.py
python run_navigation.py --no-viewer
```

默认目标恢复为原深部示例：机器人坐标 `[9.5,-1.5,16]` mm。程序执行瞄准、进针、停留、C 回零、A/B 复位；P 暂停/继续，关闭窗口保存 CSV 和报告。提前关闭标为 ABORTED。

A/B/C 位置限位默认关闭。旧观测的 C 约 2 mm 区间不再限制导航；单轴 C 演示的 2 mm 仅是展示幅度。零位编码器保持 `[0,1466018,893518]`。

## 输入目标

输入坐标单位为 mm，目标指的是显示针尖：

```bash
python run_navigation.py --target-robot 9.5 -1.5 16
python run_navigation.py --target-robot 9.5 -1.5 16 --solve-only
python run_navigation.py --target-world 9.5 12.3743686708 -10.2530483272 --no-viewer
python run_navigation.py --targets examples/targets.json --no-viewer
```

`--solve-only` 返回目标解，不改变模型姿态或标记。结果包含坐标、位置限位状态、相对 A/B/C、整数编码器目标、B 反解候选角、选定的任务姿态和量化误差。

```json
{
  "schema_version": 1,
  "frame": "robot",
  "targets": [
    {"id": "deep_target", "point_mm": [9.5, -1.5, 16]}
  ]
}
```

`frame` 为 `robot` 或 `world`，单个目标可以覆盖文件默认值。id 须唯一。整批目标预检通过才执行；存在无效目标时整批不启动。

## 针尖偏置与 B 分支

针尖距 RCM 为 `4+C_relative_mm`。因此深部示例需要 C≈14.668 mm，而不是 18.668 mm；4 mm 已包含在针尖几何中。C 也允许负值，用于总距离小于 4 mm 的入口附近目标；总距离仍不得为负。

B 连杆位移不是全局单值的角度函数。代码验证 `gamma+beta` 和 `gamma-beta` 两个解，再按目标/当前姿态选取分支。完整计划保存已选定的任务姿态，执行或渲染时不会从零位重新反解出错误的分支。

```python
from rcm_reference import Reference
from rcm_navigation import build_plan, execute_plan, solve_target

reference = Reference()
solution = solve_target(reference, [9.5, -1.5, 16], frame="robot")
plan = build_plan(reference, [{"id": "deep_target", "frame": "robot", "point_mm": [9.5, -1.5, 16]}])
recorder = execute_plan(reference, plan)
recorder.write("runs/历史记录/04_sessions/demos/my_navigation")
```

直接解码编码器可调用 `reference.task_from_encoders(counts, reference_task=previous_or_requested_q)`，q 使用 rad、rad、m。省略参考姿态时使用当前模型姿态。只给编码器值不能唯一确定 B 的两个分支，实机分支识别仍需运动历史/额外测量与机构核对。

目标恰在 RCM 时保留当前 A/B；在机器人 Y 轴时保留当前 A 方位角，避免把方向不唯一误判为位置不可达。

## 完整流程

每个目标依次执行：

1. C 保持相对零位，A/B 瞄准。
2. A/B 固定，C 到达针尖目标。
3. 停留。
4. A/B 固定，C 返回相对零位。
5. C 固定，A/B 返回零位。

从非零初始姿态启动时先沿当前针轴回 C 零位，再复位 A/B。C=0 时针尖仍距 RCM 4 mm；它不是完全拔针状态。

五次平滑插值保持段首、段末速度与加速度为零。`navigation.max_velocity_*` 和 `max_acceleration_*` 控制时间分配，继续保留；关闭位置限位不改变这些速度设置。

## 验收与拒绝

```bash
python verify_navigation.py --output-dir acceptance
python run_navigation.py --targets examples/rejected_targets.json --no-viewer
```

新版验收直接采样眼球空间：512 个内部点、70 个表面点、11 个回归目标，共 593 个目标全部进行目标求解和独立 MuJoCo 位置核对。其中 49 个目标完整执行导航与回零。覆盖深部目标、入口附近、两种 B 分支和 B 位移极小点。

旧的“原深部目标应因 C 超限被拒绝”测试已删除。拒绝示例改成眼球外目标；非有限坐标和维度不对的输入也仍被拒绝。

默认情况下不产生 `AXIS_A_LIMIT`、`AXIS_B_LIMIT` 或 `AXIS_C_LIMIT`。可选位置限位若人为重新启用，这些拒绝原因仍有效。数学几何、整数编码器误差和目标位于参考眼球内的检查保留。

## 渲染与日志

## 眼内相机视觉伺服

场景包含固定在眼球空腔内的 `eye_internal` 虚拟相机；全景视角中用青色小球和短轴线标出相机位置与朝向。相机配置位于
`config.json -> reference_scene.eye_camera`，默认 640x480、朝向橙色目标标靶。标靶由中心小球和四个定位点组成，使针尖覆盖中心后仍可观测目标质心。
视觉伺服使用 RGB 阈值检测目标质心，并用 MuJoCo 几何分割和深度观测目标与专用针尖标记。A/B 通过有限差分图像 Jacobian 调整，C 通过校正后的深度误差调整；三轴均受速度和加速度限制。停止条件只使用图像和深度，MuJoCo 三维针尖真值只用于事后验收。启动前会在已知目标姿态标定目标图像特征偏置，但不会用该姿态直接设置实际 C。该流程是 MuJoCo 仿真闭环，不代表真实相机、驱动器或组织接触验证。

```bash
python run_visual_servo.py --no-viewer --output-dir runs/历史记录/04_sessions/demos/visual_servo_demo
python verify_visual_servo.py --output-dir acceptance/visual_servo
```

视觉伺服默认不限制运行步数（`navigation.visual_servo.max_iterations=null`），持续运行至视觉到达条件满足、观测丢失或约束触发。关闭可视化窗口可中止控制并保存报告；无窗口运行可用 Ctrl+C 中断。精度模式不再覆盖最大步数。

要直接观察运动过程，在有桌面显示环境的机器上运行：

```bash
python run_visual_servo.py --output-dir runs/历史记录/04_sessions/demos/visual_servo_viewer
python run_visual_servo.py --camera overview --output-dir runs/历史记录/04_sessions/demos/visual_servo_overview
python run_visual_servo.py --camera free --output-dir runs/历史记录/04_sessions/demos/visual_servo_free
```

默认窗口使用 `eye_internal` 相机；使用 `--camera overview` 可从外部全景观察同一段视觉伺服过程，使用 `--camera free` 可直接鼠标旋转、平移和缩放。运行时按 `F` 在自由/固定相机间切换，按 `V` 循环固定相机。右下角实时眼内画面用青色十字标目标、绿色十字标针尖、白线显示图像误差，左上角显示像素误差、深度误差和三轴状态。控制始终由眼内相机完成；视觉伺服完成后保持最终姿态，关闭窗口结束程序。

输出包含 `visual_servo_report.json`、`visual_servo.csv`、原始相机 PNG 和带十字/误差线的 `guidance_*.png`。报告的
`feedback_controller_used` 为 `true`，`execution_mode` 为
`visual_servo_kinematic`；默认目标通过时最终针尖误差应低于 150 um。

每次运行写入：

| 文件 | 内容 |
|---|---|
| `target_solutions.json` | 所有目标、姿态与整数编码器解 |
| `navigation_plan.json` | 目标姿态、阶段、时间和轨迹设置 |
| `trajectory.csv` | 每个采样时刻的姿态、针尖、变径点、误差与速度 |
| `navigation_report.json` | 到达、RCM/连杆误差、位置限位状态与回零 |

```bash
python run_navigation.py --no-viewer --output-dir runs/历史记录/04_sessions/demos/deep_demo --render preview/deep_target.png --camera needle_close
```

无窗口 Linux 可使用 `MUJOCO_GL=egl`。报告的 `execution_mode` 为 `kinematic_mj_forward`，不是驱动器反馈。当前参考眼球与针的外形未完成实物标定，也没有组织碰撞验收。眼球表面采样用于验证参考椭球，不能等同于精确视网膜模型。

## 针与入口的逐帧诊断

```bash
python verify_needle_geometry.py
```

`needle_geometry_verification.json` 包含零位、进针 2 mm、针尖回到 RCM、原深部目标、合成眼球错位及刚体安装变换检查。到达深部目标时细段在眼内仍为 4 mm，粗段在眼内约 14.668 mm。

每个到达结果和 `navigation_report.json` 记录 `needle_eye_geometry`，CSV 记录入口针段、变径点轴向位置、粗/细段中心线入眼长度、入口与 RCM 偏差及针尖是否在眼内。这些字段供核对模型和轨迹使用；它们不启用新的位置限位。当前孔径/接触检验状态为 `entry_aperture_validated=false`。
