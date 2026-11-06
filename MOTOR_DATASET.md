# 当前 v9 的电机执行与数据集采集

`collect_v9_dataset.py` 使用当前照片前端/眼球模型和原始 `eye_internal` 相机，直接执行单目标视觉闭环并保存 RGB/state/action episode。实际运动经过离线 `MotorBackend`，没有失败后直接写目标位姿的回退。

当前新采集固定采用 [严格 RGB-D 观测](STRICT_RGBD_NAVIGATION.md)：RGB 计算掩码，正常可见深度参与几何估计，仿真 ID 与隐藏物体重渲染不进入导航。2026-10-30 的 683 帧试采是此前的仿真 ID 辅助版本，不应追溯标成当前严格 RGB-D 数据。

## 采集

从仓库根目录执行，输出目录必须尚不存在：

```bash
python -m pip install -r requirements.txt
MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa python collect_v9_dataset.py --output-dir runs/v9_motor_dataset --episodes 1
python verify_v9_dataset.py --run-dir runs/v9_motor_dataset
```

Windows/macOS 按当地 MuJoCo 图形环境运行，无需照抄 Linux 的 EGL 环境变量。默认从 A/B/C 零位出发，访问配置中的单个目标；每集独立重置，默认最多 1600 次观测。采集和 PNG 写盘可能需要数分钟，20 Hz 指仿真时钟，不保证实际计算达到实时速率。

增加独立 episode、使用可复现的随机眼后部目标：

```bash
python collect_v9_dataset.py --output-dir runs/v9_motor_random --episodes 10 --random-targets --seed 20261030
```

`--start-task A_DEG B_DEG C_MM` 可指定每集的初始化姿态；这是开始采集之前的显式模拟器重置，不作为运动标签。非零起点会写入每集报告，不能将近目标试采称为零位完整任务。目标不可见、不可达或达到迭代上限时，保留失败轨迹并返回非零退出码；训练时须筛选成功及符合所需运动质量要求的 episode。

## 执行链

```text
视觉测量误差 → 有速率约束的累计目标 q_command
  → Reference 编码器换算
  → 驱动的 virtual_handles_from_robot_encoders（包含齿轮比例与 B 非线性）
  → MotorBackend.consume（累计绝对 handles，ABCE int32）
  → MotorBackend.step（默认 2 ms 子步，推进完整控制周期）
  → 实际 robot 编码器 → task_from_encoders
  → 当前 v9 的 qpos / qvel → mj_forward → 下一帧图像
```

默认控制周期 50 ms，电机一阶滞后 τ=40 ms，速度参数沿用离线模型的 A/B/C 10°/s、10°/s、2 mm/s。它们是选定的模拟参数，未由真实电机辨识。指令的亚 handle 修正会累计，避免小修正反复被量化舍掉；控制器同时读取实际反馈，停止条件要求视觉误差达标、零指令速度和连续编码器停稳。

`rcm_motor_execution.MotorExecution` 可挂接到 `Reference` 或 `RCMSceneV9`。v9 内部用于坐标计算的隐藏 `Reference` 始终不推进电机，避免重复执行一次命令。只在显式执行 tick 中推进时间，运动学查询不推进时钟。`reset_motor_time()` 会保持当前位置、重置后端时间戳并重新建立累计 handle 锚点。

常规单目标入口现在默认使用 **v9 控制 + MotorBackend**：

```bash
python run_visual_servo.py --no-viewer --max-iterations 1600 --output-dir runs/v9_motor_servo
```

根目录 `config.json` 没有 `motor_execution.enabled=true`，只表示直接构造 `Reference()` 或 `RCMSceneV9()` 时不自动启动电机。正式视觉伺服入口默认选择 `--motion-backend motor`，在创建场景后显式调用 `enable_motor_execution()`；采集入口也在 episode 初始化完成后显式开启。因此，判断正式入口是否使用电机，应检查实际对象和报告中的 `motor_execution.backend`、`control_steps`，不能只看根配置里的自动开启开关。

直接调用 Python API 时，需要显式开启，例如：

```python
from rcm_scene_v9 import RCMSceneV9

scene = RCMSceneV9()
try:
    scene.enable_motor_execution()
    # q 的单位为 rad、rad、m；以下调用推进一个 50 ms 电机控制周期。
    scene.set_task([0.01, -0.005, 0.0005], motor_dt=0.05)
    print(scene.motor_execution.summary())
finally:
    scene.close()
```

也可以向构造器传入含 `motor_execution.enabled=true` 的独立配置，使用自动开启方式；两种方式选择一种，重复开启会报错。全局修改根配置还会影响运动学研究入口，其中的规划探测及测试不能据此自动迁移到电机执行。

复现旧研究时显式选择 `--control-backend legacy --motion-backend kinematic`。原 RRT/PRM、多目标规划器与随机数据集研究入口仍保留其运动学语义；它们会拒绝打开电机执行的配置，避免把规划用位姿探测误作真实动作。当前新增采集入口验证的是独立单目标 episode，尚未验证这些规划器的电机执行版本。

## 文件与时间对应

- `dataset/images/episode_XXXXXX/frame_XXXXXX.png`：未经叠加的原始 1920×1440 RGB。
- `dataset/data/*.jsonl`、`*.npz`：完整样本及离线审计字段。
- `dataset/policy_inputs/*.npz`：仅图像路径、实际 A/B/C 状态和任务文本。
- `dataset/policy_inputs_no_image/*.npz`：仅状态和任务文本。
- `dataset/action_labels/*.npz`：经过协议量化、实际发送的绝对 A/B/C 目标。
- `episode_XXXXXX/motor_steps.jsonl`：请求目标、handles、量化后的目标编码器、前后实际编码器、前后实际位姿及子步时钟。
- `episode_XXXXXX/frame_audit.jsonl`：观测时刻、实际编码器、RGB 像素 SHA-256 和对应的运动区间编号。
- 每集还包含控制报告、轨迹和最终模拟深度帧；根目录保存冻结配置、源码哈希和采集协议。

在时刻 t 保存图像和实际状态 s(t)，动作 a(t) 是随后控制区间的协议目标；下一次观测得到 s(t+dt)。有电机滞后时 **a(t) 不等于 s(t+dt)**。终止帧的动作标签为当前位置保持，不虚构后续运动。单位为度、度、毫米。

数据为 PNG + JSONL + NumPy NPZ，自带字段契约，尚不是 LeRobot 容器。真值误差、目标坐标、深度、访问顺序和可见性审计不进入模型输入。原始 RGB 与策略输入没有指导十字线、显示放大针体或其他视频覆盖物。

## 验证及范围

```bash
MUJOCO_GL=egl python verify_motor_execution.py
python verify_v9_motor_collection.py
python verify_v9_dataset.py --run-dir runs/v9_motor_dataset
```

验证逐区间重放原始 handle 指令和电机子步，检查编码器、图像哈希、时间、NPZ/JSONL、动作标签及策略输入字段。它会拒绝将下一实际状态冒充动作，也区分“数据完整”与“导航成功”。成功 episode 还需通过独立终点误差和编码器停稳检查。

报告分别列出指令速率检查和实际轨迹的最大速度/加速度。该一阶电机模型没有独立的加速度限制，协议/编码器量化也可能使实际差分速率超过控制器名义门限；验证报告会如实标记，不能将指令限速通过等同于实际电机限速通过。需要严格运动约束的数据可据这些字段筛选；本次不宣称物理制动、组织接触、真实 OCT 或硬件性能有效。

原始采集数据保存在忽略的 `runs/`，不会自动塞入 GitHub 项目下载包。可单独打包数据，项目保留代码及精简验证记录。

已完成从零位出发的 683 帧试采，最终误差 5.867 μm；实际名义速率门限未全部通过，限制与证据见 [本次试采记录](reports/20261030/v9_motor_dataset/README.md)。
