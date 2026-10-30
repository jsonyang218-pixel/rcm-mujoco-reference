# RCM 迁移接口审计（阶段一）

范围：`rcm-mujoco-reference-main`。目标是把依赖 legacy `Reference` 的研究模块，
通过适配器切换到 `mujoco_model` 的 v9 扫描模型，legacy 全程保留作对照。

## 1. `Reference` 实际公开接口（legacy，`rcm_reference.py`）

注意：规划文档里假设的 `set_abc / get_abc / tip_position_world / render_eye_camera /
eye_center_world / target_position_world / encoder_to_abc / abc_to_encoder /
reset_pose` 这些名字 **不是** 实际 API。实际公开方法是：

| 方法 | 语义 |
|---|---|
| `__init__(config=None)` | 读 `config.json`，构建 legacy MJCF 与 MuJoCo 模型 |
| `build_xml()` | 生成含眼内场景的 legacy MJCF 字符串 |
| `state_from_task(q)` | task 坐标（A/B rad、C m）→ `JointState`（phi/alpha 度、length mm） |
| `task_from_state(state)` | 逆变换 |
| `validate_task(q)` | 校验 A/B/C task 坐标在限位内并可逆 |
| `set_task(q)` | 写入 ABC 关节并 `mj_forward` |
| `current_task()` | 返回当前 ABC task 坐标（rad/rad/m） |
| `encoders(q=None)` | task → 编码器计数（3 元组） |
| `task_from_encoders(values, reference_task=None)` | 编码器 → task（B 分支用提示角消歧） |
| `robot_to_world(p_mm)` | 机器人参考系 → world（mm） |
| `world_to_robot(p_mm)` | 逆变换 |
| `set_target_world(point_mm)` | 平移 lesion 靶点与测量 site |
| `target_task(validate=True)` | 当前靶点逆解为 task 坐标 |
| `site_mm(name)` | 读取 named site 的 world 坐标（mm） |
| `camera_id(name="eye_internal")` | 相机 id |
| `render_camera(name="eye_internal", mode="rgb")` | 渲染 rgb/depth/segmentation |
| `camera_pose(name="eye_internal")` | 相机 world 位置与旋转矩阵 |
| `needle_eye_geometry()` | 针中心线与眼球椭球求交（诊断） |
| `metrics()` | 汇总 tip/target/encoder/FK 一致性等指标 |

研究模块还会直接读取以下属性（适配器必须保留）：`config`、`model`、`data`、`sites`、
`eye_camera`、`rcm_mm`、`xml`。

## 2. 研究模块实际调用面（只列非快照目录 `reports/`、`source_snapshot/`）

| 模块 | 用到的 `Reference` 接口 |
|---|---|
| `rcm_fundus.py` | `state_from_task` `robot_to_world` `world_to_robot` |
| `rcm_navigation.py` | `encoders` `task_from_encoders` `state_from_task` `task_from_state` `validate_task` `robot_to_world` `world_to_robot` `current_task` `set_target_world` `set_task` `metrics` |
| `rcm_vision_navigation.py` | `camera_id` `world_to_robot` `current_task` `state_from_task` `task_from_state` `validate_task` `set_target_world` `set_task` |
| `rcm_visual_servo.py` | `camera_id` `set_task` `state_from_task` `robot_to_world` `current_task` `validate_task` `target_task` `metrics`；`VisualServoSensor` 直接读 `model.geom(...)`、`data`、`eye_camera` |
| `rcm_multitarget_visual.py` | `current_task` `set_target_world` `metrics` |
| `rcm_sampling_planners.py` | `validate_task` `state_from_task` `robot_to_world` |
| `rcm_occlusion_fixture.py` | `camera_id` |
| `rcm_target_tracking.py` / `rcm_waypoint_dataset.py` | 主要经 metrics/当前姿态间接使用 |

`verify_*` 还用到 `site_mm`、`camera_pose`、`needle_eye_geometry`、`render_camera`。
`rcm_visual_geometry.NeedleGeometry.from_reference` 强依赖下列 geom/site 名：
`lesion_marker`、`needle_proximal`、`needle_tip_marker`、`needle_distal`、
`needle_distal_tip`、相机 `eye_internal`、`config['needle']` 中的
`proximal_radius_mm / proximal_length_mm / distal_length_mm / tip_marker_radius_mm`。

## 3. v9（`mujoco_model`）现状

- 入口 `run_model.py`；场景对象是 `model_sync.DriverModel`。
- `DriverModel` 以编码器计数驱动，`sync(robot)` 写入 ABC 关节并算
  `diagnostics`（`A_deg / B_relative_deg / C_axial_mm / tip_world_m / junction_world_m`）。
- `MotorBackend`（`motor_backend.py`）是离线协议电机，`tip_mm()` 用 vendor FK。
- `DriverModel` **不读根 `config.json`**。它读 `assets/scan_template.xml`、
  `installation_config.json`、`silver_motion.py` 和 vendor 驱动
  `vendor/simulation_driver/ros_msg/src/robot_control/kinematics.py`。

## 4. `scene.xml` 眼内场景缺口（已确认）

- 根 `scene.xml`（v9 根场景）只有 `overview` 相机、机架、针具和
  `phantom / phantom_pad / target_tape_*` 标定假体，**没有**：
  - 眼球 sphere/ellipsoid geom（`eye_reference`）
  - 眼底靶点 site/marker（`lesion` / `lesion_marker` / `lesion_display_marker`）
  - 眼内相机 `<camera name="eye_internal">` 及其内参
- legacy `Reference` 动态生成的场景里有完整眼内场景，但其坐标在 legacy 机器人参考系，
  **不能**直接复制到 v9 世界系。

## 5. 坐标与运动学结论（已用真实实现数值验证）

1. legacy `rcm_recovered_model.py` 与 v9 vendor `kinematics.py` 使用同一套编码器
   常数和同一套针轴/B 连杆代数：
   - `a_counts_per_degree = 555.555...`
   - `bc_counts_per_mm = 63822.71488`
   - `b_displacement(alpha) = 98.48 - 60 cos(alpha) - sqrt(52^2 - (60 sin(alpha)-16)^2)`
   - `needle_axis` 与 `needle_direction` 公式一致（`theta = alpha + 22`）
2. legacy `encoders(home)` 与 v9 `HOME=(0,1466018,893518,0)` 完全一致；legacy
   `alpha_home` 与 v9 `alpha0` 同为 `54.23600525044604` 度。
3. 对 A∈[-10,10]°、B∈[-5,5]°、C∈[0,8]mm 的 9×9×9 网格，两套局部针尖向量
   最坏差 **0.00016565 mm（约 0.166 µm）**，来源是编码器整数取整，非模型差。
4. 世界系映射 `v9_world = rcm_world + robot_to_world @ (p_legacy_mm * 0.001)`
   能把 legacy 局部坐标精确投影到 v9 世界（最坏同样 0.166 µm）。该映射即补眼内
   场景时使用的坐标变换。

## 6. 阶段二建议

眼内场景不能直接复制坐标，也不能只读根 `config.json` 就想驱动 v9。适配器应在
运行时把 `config.json` 的 `reference_scene` 局部坐标用第 5 条映射注入 v9 MJCF；
`eye_center_v9` 初始等于 legacy 局部坐标（经同一映射），实机标定后再单独更新。
`VisualServoSensor`/`NeedleGeometry` 强依赖的 geom/site 名也需要在注入时补齐，
否则阶段四的完整视觉伺服无法构造。
