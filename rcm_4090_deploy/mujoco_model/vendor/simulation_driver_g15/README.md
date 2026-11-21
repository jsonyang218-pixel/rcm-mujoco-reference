# MuJoCo 仿真闭环控制包

本目录提供在 MuJoCo 中完成实验所需的 ROS 1 Noetic 消息、反馈控制器和结果导出模块。实验控制环只使用仿真话题：

- `/simulation/controlSignal`：控制器发送给 MuJoCo 的 `ros_msg/controlSignal`
- `/simulation/RobotState`：MuJoCo 返回给控制器的 `ros_msg/Encoder`

真实机器人话题不属于 MuJoCo 实验输入，实验期间不得向 `/robot/controlSignal` 发布命令。

## 与实机源码同步（2026-11-20）

同步基准为仓库提交 `749575d`。`kinematics.py` 和公共导出与
`src/ros_msg/src/robot_control/` 一致：换算增益 `CONTROL_GEAR=15`、浮点逆解、
整数累计手柄目标、软限位、针型偏置和 E 轴禁用规则共用同一实现。
消息字段、单位、方向未变；两端消息定义与 ROS MD5 一致。

MuJoCo 适配器的正向手柄增量换算也须匹配 G=15，尤其 C 轴每手柄计数为
112.5 个机器人编码器计数。G 是软件运动学换算参数，不会设置设备速度档位；
适配器若仍采用 G=10，会出现命令与模型响应比例不一致。

本包提供实机同款响应实验引擎，保持 `ResponseControl.interval_s=0.05`
（20 Hz）、基点响应单包 A/B/C/E 上限 `8/9/1/0` 和反馈超时停止。
实机 UI 笛卡尔控制环的 47.2 Hz、`96/16/64/0` 上限仅适用于该 UI；
本包不包含 UI，该参数不应用到响应实验。同步的单包等比例限幅函数是公共工具，
响应引擎仍按实机 `BasePositionExperiment` 的独立规则限幅。

保留仿真专用话题参数和无 UI 的源码指纹清单。两个实验脚本直接运行时默认使用
`/simulation/` 话题及 `data_source=simulation`；启动文件启用 `--simulation-only`，
同时检查反馈、控制话题经 ROS 重映射后的实际名称，防止重映射到实机话题。
导出元数据包含 G 值、实际话题、数据来源和源码指纹。显式自定义话题时也应始终带
`--simulation-only`；未启用该选项时仍保留自定义话题兼容能力。

## 安装

将 `ros_msg/` 放入 MuJoCo 所在 catkin 工作区的 `src/`，然后构建：

```bash
source /opt/ros/noetic/setup.bash
cd <mujoco-catkin-workspace>
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

本包与实机包同名，须使用独立 catkin 工作区，不可在同一 `src/` 中放两份 `ros_msg`。
安装时仅分发网络模板，机器专用 `config/ros_network.env` 不会复制到 install 空间。

如果 MuJoCo 和 ROS Master 不在同一台电脑，在 MuJoCo 电脑设置：

```bash
export ROS_MASTER_URI=http://<robot-host>:11311
export ROS_IP=<mujoco-host-ip>
unset ROS_HOSTNAME
```

本机实际地址写入 `ros_msg/config/ros_network.env`（不入库）；模板见 `ros_msg/config/ros_network.env.example`。

## MuJoCo 节点接口

MuJoCo 适配节点负责：

1. 订阅 `/simulation/controlSignal`；
2. 根据命令推进 MuJoCo 模型；
3. 按 `ros_msg/Encoder` 发布 `/simulation/RobotState`；
4. 保持消息字段、单位和编码器方向与 `msg/Encoder.msg` 一致。

MuJoCo 适配节点未发布仿真反馈时，控制器会因反馈超时停止。

## 启动实验

```bash
roslaunch ros_msg mujoco_response.launch \
  output:=/tmp/mujoco-response \
  feedback_topic:=/simulation/RobotState \
  control_topic:=/simulation/controlSignal
```

启动文件固定带有 `--simulation-only`，并拒绝把控制话题改回真实机器人命名空间。实验脚本也支持显式覆盖：

```bash
rosrun ros_msg base_position_response.py \
  --output /tmp/mujoco-response \
  --feedback-topic /simulation/RobotState \
  --control-topic /simulation/controlSignal \
  --simulation-only \
  --execute --confirm-safe
```

`base_position_response.py --kind sine` 为基点 XYZ 正弦位置闭环；
`base_sine_response.py` 为 A/B/C 手柄输入正弦，两者实验口径不同。
`--simulation-only` 要求反馈和控制的实际话题均位于 `/simulation/` 命名空间。

## 运行前检查

```bash
rostopic type /simulation/RobotState
rostopic type /simulation/controlSignal
rostopic info /simulation/RobotState
rostopic info /simulation/controlSignal
rostopic info /robot/controlSignal
```

实验控制器应订阅 `/simulation/RobotState`，MuJoCo 应订阅 `/simulation/controlSignal`。`/robot/controlSignal` 在实验期间不得有本实验节点发布者。

`feedback_probe.py` 仅作为独立的只读诊断工具保留；它不参与 MuJoCo 闭环，也不会发布控制命令。

## 离线验证

在独立仿真工作区构建并加载环境后运行：

```bash
catkin_make run_tests
catkin_test_results
python3 -m py_compile src/ros_msg/scripts/*.py src/ros_msg/src/robot_control/*.py src/ros_msg/test/*.py
python3 src/ros_msg/test/validate_simulation_response.py --output /tmp/simulation-gate3-unique
```

最后一条命令自行启动仅绑定回环地址的 ROS Master，运行模拟反馈和真实 CLI，
覆盖只读零控制发布、正常闭环、保持抖动、反馈超时、跳变、操作者停止和 RCM 预检拒绝，
核对导出及停止后无新运动命令。输出目录必须尚不存在。
该验证无需 MuJoCo，不连接硬件；不能替代 MuJoCo 适配器和模型的联调验收。

本次验证记录见 [同步验证记录](同步验证记录_20261120.md)。
