# OCT 注册深度仿真接口

当前已接入的是 **MuJoCo 理想注册深度接口**，用于先验证眼底 RGB 与深度的接口、坐标、时间戳和控制数据链路，再替换成真实设备适配器。来源固定标注为 `mujoco_ideal_registered_oct_simulation`；没有连接 OCT 设备、调用设备 SDK 或读取真实 OCT 数据。

它不是 OCT 成像物理仿真：不生成 A-scan/B-scan、反射强度、散斑、视网膜层、组织穿透或真实设备噪声。当前 `oct_depth` 历史变量名指的是理想深度图，不能据此称为真实 OCT 测量。

## 当前图像和深度的含义

`VisualServoSensor` 分别渲染 RGB、包含眼底的场景深度、排除眼底背景的深度、包含眼底的可见性 segmentation。`SyntheticOCTProvider.capture()` 只接收这两张已有深度图，不读取关节真值、目标真值或 MuJoCo site 坐标。

| 字段 | 含义 |
| --- | --- |
| `scene_surface_depth_m` | 包含眼底的正常场景第一可见表面轴向深度，包括可见针、目标和眼底。 |
| `registered_feature_depth_m` | 额外关闭眼底背景渲染所得的理想深度，供可见针/目标的几何回投使用。该处理不代表真实 OCT 能穿透眼底或保证在遮挡时仍可测。 |
| `scene_valid_mask` / `feature_valid_mask` | 捕获时的数值有效性：有限、正值且在配置深度界限内。mask 本身不提供针/目标语义，也不保证物理 OCT 可测性。 |
| `depth_unit` / `depth_convention` | `m` / `positive_camera_optical_axis_z_not_ray_range`：米，沿相机光轴的正 z 距离，**不是沿像素射线的斜距**。 |
| `camera_name` / `registration_frame` | 注册坐标系对应的相机名，默认 `eye_internal`。当前注册由同一 MuJoCo 相机的像素坐标直接对应，不等于真实设备标定。 |

输入必须已经使用米单位；接口不会根据量级猜测毫米，也不做隐式单位换算。两张 map 必须为同一非空二维尺寸，输出为各自独立的 `float32` 数组；不会改写传入数组。深度 map 中的 NaN、Inf、零和负值会被有效 mask 排除。可选 `min_valid_depth_m` / `max_valid_depth_m` 以严格不等式过滤近裁剪、远裁剪或无返回值，原 map 值仍保留。MuJoCo 远裁剪值若为有限正数，默认规则仍会判为数值有效；调用方应传入裁剪界限，并继续按可见性和 feature segmentation 取样。

原有 observation 的 `oct_depth` 可继续指向 `frame.registered_feature_depth_m`，避免改变已有几何估计器的输入约定。数组保持可写以支持现有传感器因果扰动测试；有效 mask 描述捕获时的数据，主动修改 map 后应重新计算 mask 或重新 capture，不能把修改后的缓存当作新采集。

## 已接入的控制链路

`VisualServoSensor.observe()` 已实际调用 provider；`depth` / `oct_depth` 分别直接引用 frame 的场景/特征深度图，几何回投也检查 `feature_valid_mask`。默认有效上界为渲染 `zfar × 0.9999`，保守排除最后 0.01% 的深度范围，以免 MuJoCo reverse-Z 舍入使无返回背景略低于远裁剪值而被误判。原深度数组不改写；标定与真实可见性仍需分割判断。

严格几何控制器逐次提供仿真时间，并将阶段起始时间加到时间戳上。相同仿真时刻的重采会显式标记 `timestamp_repeated=true`。独立调用传感器时，默认使用合成序号时间；元数据区分这两种来源，均不宣称是设备时钟。

每个控制阶段保存 `oct_observations.jsonl`（时间、来源、相对深度审计）及 `final_oct_frame.npz` / `.json`（该阶段最后一帧完整深度和 mask）。报告含文件路径与最终帧元数据。默认 RGB/state/action 数据集仍不含全程深度序列，policy 输入也没有自动加入深度。旧实验不会补写这些新字段或文件。

`oct_navigation_depth` 由当前可见目标和针尖的几何测量计算，采用光学坐标 x 向右、y 向下、z 向前，差值定义为 `target - tip`。光轴深度差与沿针轴距离分别记录；目标不可见、针尖不可重建或有效深度不足时为 `None`。遮挡记忆另标为历史测量，不能视作新 OCT 观测。

## API

```python
from rcm_oct import SyntheticOCTProvider, RegisteredOCTFrame, relative_depth

provider = SyntheticOCTProvider(
    nominal_period_s=0.05,
    min_valid_depth_m=0.0,
    max_valid_depth_m=None,  # 可改为有依据的渲染远裁剪界限，单位米
)
frame = provider.capture(
    scene_depth_m,
    feature_depth_m,
    timestamp_s=iteration * 0.05,  # 明确是仿真时间
    camera_name="eye_internal",
    metadata={"acquisition_reason": "controller_observation"},
)
z_m = frame.sample((u, v), map="feature")
small_json_metadata = frame.metadata()
saved_path = frame.save("runs/oct_example/frame_0000.npz")
same_frame = RegisteredOCTFrame.load(saved_path)
```

`OCTFrame` 是 `RegisteredOCTFrame` 的别名。`sample()` 接受整数 `(u, v)`，可选 `map="feature"` 或 `"scene"`；无效像素、越界、亚像素坐标及未知 map 会明确拒绝，不做静默截断、舍入或插值。

`save()` 写压缩数值 NPZ 和同名 JSON；NPZ 中同时包含 Unicode JSON 元数据，不使用 pickle。`load()` 使用 `allow_pickle=False`，验证仿真来源和有效 mask。本版加载器只支持当前仿真格式；真实来源的存档读取需要随设备适配扩展来源与标定验证。

## 时间与来源

每个 provider 有随机唯一 `stream_id`；`sequence_id` 从零单调递增，`frame_id` 包含 stream 和 sequence，重建传感器不会把新流第一帧误当作上一流第一帧。失败的 capture 不消耗序号。

`timestamp_s` 属于 `clock_domain="simulation"`，要求非负且不倒退；不是设备采样时间，也不是系统 UTC。`acquired_monotonic_s` 记录本机单调 wall clock 的采集标记，默认 `time.monotonic()`；可传入同一时钟域的明确采集时刻，但不能在同一流内倒退。渲染耗时需要独立记录，例如 metadata 中的 `render_latency_s`，不能由仿真时间差推算。

静态 survey 或同一控制迭代的有限差分可能冻结仿真时间。第二次及后续同一时间的 capture 必须显式设置 `allow_repeated_timestamp=True`，建议在 metadata 写明原因。该帧仍有新序号，但 `timestamp_repeated=true`、`fresh_simulation_timestamp=false`，不能作为时间推进后的新扫描声称有新的时域信息。

所有当前帧明确记录：

- `is_simulation=true`、`raw_oct_available=false`、`realtime_validated=false`。
- `source="mujoco_ideal_registered_oct_simulation"`。
- `nominal_period_s=0.05` 对应名义 20 Hz，且 `nominal_rate_is_wall_clock_guarantee=false`。

0.05 秒只控制仿真调度；本环境 CPU 渲染约每秒数帧，不能宣称实机或 wall-clock 已达到 20 Hz。实际速度应使用该次运行记录的 wall-clock 耗时。当前没有经过真实设备精度、端到端延迟或闭环实时性验证。

## 相对深度的符号

`relative_depth(tip_xyz_camera, target_xyz_camera, *, source_flags=None, needle_axis_camera=None)` 接收同一相机坐标系内、以米为单位的已估计 XYZ，不读取动态真值。输出始终定义为 `target_minus_tip`：

- `axial_depth_difference_m = target_z - tip_z`，正值表示目标沿相机正 z 方向更深。
- `delta_xyz_m = target_xyz - tip_xyz`，`euclidean_distance_m` 是两点三维距离。
- `along_needle_difference_m` 仅在提供明确正方向的针轴时计算，为差向量在单位针轴上的有符号投影；不等同于光轴深度差。

`source_flags` 由调用方写入两点的观测/记忆来源，用来保留数据依据；它不会把记忆点自动升级为当前直接观测。函数不会修改输入点或轴向量。

## 接真实设备时需要补齐

`OCTSource` 协议要求采集方法返回统一的 `RegisteredOCTFrame`。真实适配器应先完成实际 OCT 数据采集、针/目标或表面分割、米单位换算和到相机的空间配准，再输出注册深度及 mask；单独一张原始 B-scan 不符合当前控制输入约定。未来适配器必须按实际来源设置 provenance，不能沿用当前仿真来源或仿真时钟标记。

接设备前需要明确设备型号和 SDK/驱动/API、采集格式与扫描几何、深度定义及组织折射率修正、像素/扫描轴和相机坐标约定、内外参及标定版本/误差、设备时间戳与时钟同步、实际采样率、曝光/扫描时间、传输和处理延迟、丢帧/过期判定、原始数据保存方式，以及无返回/遮挡/低质量数据的有效性标准。随后才能验证注册误差、真实闭环延迟和精度；这些工作当前尚未执行。

## 可复现数值检查

```bash
cd /workspace/rcm-mujoco-reference
.venv/bin/python verify_oct_interface.py
.venv/bin/python verify_oct_sensor.py
```

输出保存在 `runs/multitarget_occlusion_20261022/oct_interface_checks/`。10 组检查覆盖复制与米单位、有效性和裁剪界、非法取样、尺寸/类型、时钟/序号、流唯一标识、明确仿真声明、无 pickle 存档往返、相对深度符号和输入不变。该检查验证接口数值约定，不是设备连接或物理 OCT 成像验证。

另有实际 EGL 渲染的 12 组传感器检查全部通过，结果保存在 `oct_sensor_checks/`：包括三点静态坐标与深度符号、远裁剪背景拒绝、真实球体遮挡、针杆支持不足、provider 全无效、重采时钟、深度扰动影响估计和真实帧保存读取。这是静态传感器验收，闭环导航结果单独列于 [实验记录](reports/20261022/README.md)。
