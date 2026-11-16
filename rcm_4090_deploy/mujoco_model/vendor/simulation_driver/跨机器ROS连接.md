# MuJoCo 电脑的 ROS 连接

本文件只描述 MuJoCo 仿真电脑如何连接 ROS Master。所有实验操作在 MuJoCo 中完成，闭环不读取真实机器反馈，也不向真实机器控制话题发布命令。

## 网络环境

如果 ROS Master 和 MuJoCo 在不同电脑：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://192.168.1.104:11311
export ROS_IP=192.168.1.105
unset ROS_HOSTNAME
```

两台电脑应位于同一可互通网络，且 `192.168.1.105` 是 MuJoCo 电脑能被 ROS Master 访问的地址。ROS 1 节点除 `11311` 外还会使用动态 TCP 端口，网络策略必须允许两台电脑互通。

## 构建

```bash
cd <mujoco-catkin-workspace>
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

消息必须由这份包重新生成：

```bash
rosmsg show ros_msg/Encoder
rosmsg show ros_msg/controlSignal
rosmsg md5 ros_msg/Encoder
rosmsg md5 ros_msg/controlSignal
```

## MuJoCo 话题检查

启动 MuJoCo ROS 适配节点后检查：

```bash
rostopic type /simulation/RobotState
rostopic type /simulation/controlSignal
rostopic echo -n 1 /simulation/RobotState
rostopic info /simulation/controlSignal
```

预期类型分别是：

```text
ros_msg/Encoder
ros_msg/controlSignal
```

## 启动 MuJoCo 实验

```bash
roslaunch ros_msg mujoco_response.launch \\
  output:=/tmp/mujoco-response \\
  feedback_topic:=/simulation/RobotState \\
  control_topic:=/simulation/controlSignal
```

控制器只读取 `/simulation/RobotState`，只向 `/simulation/controlSignal` 发布。不要使用 `/robot/RobotState` 作为 MuJoCo 闭环输入，也不要向 `/robot/controlSignal` 发布。

## 故障判断

- `/simulation/RobotState` 没有消息：MuJoCo 适配节点尚未发布反馈。
- `/simulation/controlSignal` 没有订阅者：MuJoCo 适配节点尚未启动或话题名不一致。
- 消息 MD5 不一致：两端没有使用同一份 `Encoder.msg` 和 `controlSignal.msg`。
- 控制器报 `feedback_timeout`：仿真反馈频率过低、话题错误或 MuJoCo 节点已停止。
