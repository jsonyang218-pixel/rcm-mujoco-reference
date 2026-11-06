# 当前 v9 的严格 RGB＋深度导航

本入口保留现有 `MotorBackend` 执行架构，导航观测仅为当前 RGB 与同一可见场景的配准深度。控制不读取 MuJoCo 物体 ID 掩码，不通过隐藏针具、目标或眼底后重新渲染来补齐观测。

## 使用

```bash
MUJOCO_GL=egl python run_visual_servo.py --no-viewer --max-iterations 1200 --output-dir runs/strict_rgbd_servo
MUJOCO_GL=egl python collect_v9_dataset.py --output-dir runs/strict_rgbd_dataset --episodes 1 --max-iterations 1200
python verify_v9_dataset.py --run-dir runs/strict_rgbd_dataset
```

v9 单目标启动器默认选择 `strict_rgbd`，新采集固定使用此模式。`simulation_ids` 仅作为显式旧对照保留。旧研究的 `--strict-visual` 表示禁用目标逆解，仍可能使用模拟分割，不等同于本协议。

## 信息边界

允许当前 RGB/可见表面深度、固定相机/RCM/器械尺寸标定、实际编码器和执行 Jacobian。场景真值仅用于日志和停止后的独立验收，不生成目标或决定停止。

`rcm_rgb_features.RGBFeatureDetector` 只接收 RGB，输出橙色目标、蓝灰近端管、浅色细针与绿色标记的布尔掩码，没有模型 ID、深度或动态位姿输入，也不伪造 MuJoCo ID 图。

`rcm_strict_rgbd.StrictRGBDSensor` 仅渲染一次 RGB 和一次深度，两次使用相同光学场景并保留眼底背景。外部查看专用的外壳/放大辅助物不进入两个光学通道；不会单独在深度通道移除可遮挡目标的光学物体。两个兼容深度字段使用同一张正常可见深度图，来源明确标记为仿真，不能称为真实 OCT。

几何估计只消费这些像素及深度点。观测不足、无效深度或几何退化时停止，不读隐藏目标、不靠目标记忆继续运动、不接受 `measurement_goal/target_transform` 注入世界坐标。RGB 无法可靠判断目标消失是否由针造成，遮挡原因保持 unknown，不通过隐藏针重渲染给控制补信息。

## 人工标记假设

当前是受控标靶实验：假设一个橙色人工标靶、已知颜色的针具及绿色针尖标记。标靶标定半径 0.15 mm；绿色球半径采用配置 0.10 mm，它不是 0.02 mm 金属尖端本身。颜色阈值尚未验证真实显微图像、复杂背景、反光和白平衡泛化。

接近/目标位姿可以通过可见近端管后端面重建针尖，绿色标记完全不可见时仍可测量。零位和 C=8 mm 的端面尚不可靠，目前仍依赖可见绿色标记球。不能据此声称实现了无标记真实针尖或病灶自主识别。

## OCT 接入

`measure(rgb, depth, timestamp, metadata)` 支持外部注册数组；`frame_provider.capture(timestamp_s=...)` 可提供同步 RGB、米单位深度与真实来源/时钟元数据，外部路径不需要 MuJoCo 渲染。

真实 OCT 的 A/B-scan 不能直接当成此深度图。设备接入仍需特征深度提取、相机/探头配准、时间同步、延迟和无效返回处理。外部数组接口测试不表示已接通真实 OCT。

## 验证

```bash
MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa python verify_strict_rgbd.py --output-dir runs/strict_rgbd_checks
```

检查代表位姿精度、冻结数组后禁止动态真值 API、缺失或污染分割字段无效、RGB 目标消失后不能靠深度补标签、无效深度、深度因果性、同场景两次渲染、外部注册帧及禁止目标注入。

数据完整性、视觉到达和实际运动约束分开判定。此前电机试采的实际速率超限未在本次感知改动中修正。

已完成从零位出发的 682 帧完整闭环及 10 项观测隔离测试，证据见 [2026-10-31 验证](reports/20261031/strict_rgbd/README.md)。
