# MuJoCo 仿真闭环控制包

本目录提供在 MuJoCo 中完成实验所需的 ROS 1 Noetic 消息、反馈控制器和结果导出模块。实验控制环只使用仿真话题：

- `/simulation/controlSignal`：控制器发送给 MuJoCo 的 `ros_msg/controlSignal`
- `/simulation/RobotState`：MuJoCo 返回给控制器的 `ros_msg/Encoder`

真实机器人话题不属于 MuJoCo 实验输入，实验期间不得向 `/robot/controlSignal` 发布命令。

## 安装

将 `ros_msg/` 放入 MuJoCo 所在 catkin 工作区的 `src/`，然后构建：

```bash
source /opt/ros/noetic/setup.bash
cd <mujoco-catkin-workspace>
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

如果 MuJoCo 和 ROS Master 不在同一台电脑，在 MuJoCo 电脑设置：

```bash
export ROS_MASTER_URI=http://192.168.1.104:11311
export ROS_IP=192.168.1.105
unset ROS_HOSTNAME
```

已配置网络可直接使用 `ros_msg/config/ros_network.env`；模板保留在 `ros_msg/config/ros_network.env.example`。

## MuJoCo 节点接口

MuJoCo 适配节点负责：

1. 订阅 `/simulation/controlSignal`；
2. 根据命令推进 MuJoCo 模型；
3. 按 `ros_msg/Encoder` 发布 `/simulation/RobotState`；
4. 保持消息字段、单位和编码器方向与 `msg/Encoder.msg` 一致。

MuJoCo 适配节点未发布仿真反馈时，控制器会因反馈超时停止。

## 启动实验

```bash
roslaunch ros_msg mujoco_response.launch \\
  output:=/tmp/mujoco-response \\
  feedback_topic:=/simulation/RobotState \\
  control_topic:=/simulation/controlSignal
```

启动文件固定带有 `--simulation-only`，并拒绝把控制话题改回真实机器人命名空间。实验脚本也支持显式覆盖：

```bash
rosrun ros_msg base_position_response.py \\
  --output /tmp/mujoco-response \\
  --feedback-topic /simulation/RobotState \\
  --control-topic /simulation/controlSignal \\
  --simulation-only \\
  --execute --confirm-safe
```

正弦响应使用 `base_sine_response.py`，参数规则相同。`--simulation-only` 要求控制话题位于 `/simulation/` 命名空间，避免误发实机命令。

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
