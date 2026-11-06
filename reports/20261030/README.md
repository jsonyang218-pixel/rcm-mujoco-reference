# v9 照片建模与零位出发视觉闭环测试

本目录保存当前 v9 的外观预览，以及直接使用 `RCMSceneV9` 完成的单目标闭环录制。测试没有预先将针送至目标附近，也没有使用 legacy 相机反馈。

## 模型预览

- [完整三视图](v9_model/three_views.png)
- [前端正面：针具居中](v9_model/front_tool.png)
- [根据侧照修订后的模型](v9_model/side_profile.png)

模型入口为根目录 `v9_eye_scene.xml`，重建与尺寸假设见 [前端说明](../../V9_FRONT_TOOL.md)。外壳、安装倾角和 RCM 仍未实机标定。

## 测试结果

[完整过程视频](v9_zero_to_target/v9_zero_to_target_full_test.mp4) 为 1600×1080、H.264、20 fps、41.1 秒；按 1× 仿真时间播放，并保留开始和结束的停留画面。相机画面在实际闭环运行过程中每 4 步采样一次。

| 项目 | 结果 |
| --- | --- |
| 初始 A / B / C | 0° / 0° / 0 mm |
| 初始针尖到目标距离 | 19.749 mm |
| 控制后端与相机 | 当前 `RCMSceneV9` / `eye_internal` |
| 控制周期 | 0.05 s 仿真时间 |
| 观测次数 / 终点仿真时间 | 679 / 33.9 s |
| 最终独立针尖定位误差 | 6.033 μm（验收阈值 20 μm） |
| 最终视觉估计误差 | 7.029 μm |
| 连续停稳 | 3 帧，最终指令速度为零 |
| 速度 / 加速度限制 | PASS |
| 结果 | PASS：`POSITION_ERROR_AND_SETTLED` |

- [零位画面](v9_zero_to_target/start.png) · [结束画面](v9_zero_to_target/final.png)
- [原始测试报告](v9_zero_to_target/visual_servo_report.json)
- [A/B/C 与误差轨迹 CSV](v9_zero_to_target/visual_servo.csv)
- [视频与运动检查记录](v9_zero_to_target/video_validation.json)
- [模拟深度观测元数据](v9_zero_to_target/oct_observations.jsonl)
- 最后一帧模拟深度保存在 `final_oct_frame.npz`，配套说明为 `final_oct_frame.json`。

起点的细针尖位于眼内 4 mm 的参考零位，距眼后部目标仍约 19.75 mm。眼球透明化、近景放大的彩色针尖/目标/RCM 标记、黄色针体强调、相机固定裁剪及误差线仅用于视频显示，不参与控制。

控制和停止使用可见图像、理想注册深度与分割拟合的针尖及目标位置；针具/相机标定和运动 Jacobian 是已知输入。真值仅用于日志及独立终点验收。仿真时间不代表实际渲染达到实时 20 Hz。本次没有验证真实电机、组织接触、真实 OCT、v9 多目标或遮挡恢复；默认 `run_visual_servo.py` 的控制后端仍是 legacy。

## 复现

在仓库根目录安装 `requirements.txt` 后，指定一个尚不存在的输出目录：

```bash
MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa python reports/20261030/v9_zero_to_target/record_full_test.py --output-dir runs/v9_full_reproduction
```

需要系统 `ffmpeg`、Pillow、NumPy 和 MuJoCo；主 Python 依赖已在根目录要求中。脚本直接执行闭环并同步录制，沿用配置中的速度、加速度及观测质量门限，只将最大迭代次数设为 1000；没有 180 秒墙钟终止条件。运行可能需要数分钟。输出目录另含首尾原始相机图像、完整轨迹、模拟深度最终帧和逐次观测元数据。
