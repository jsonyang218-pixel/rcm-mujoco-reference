# MuJoCo 机械臂 v9：银色圆角端盖与长罩整体运动

根据用户补充的视频，银色长罩左端的圆角封头、顶面、长侧面及端盖底缘属于同一个整体。本版将错误留在固定白壳上的端盖面归回银罩运动体，消除升降时端盖残留、分开的现象。

## 本次修正

- 1,797 个源三角面从固定壳归入银罩：主端盖 1,238 面，端盖底缘 559 面。
- 端盖范围沿银罩自身三维轴定义，并通过底缘法向排除白壳屋面；不是按屏幕视角裁切。
- 端盖和长罩现在属于同一个 `silver_linear_stage`，没有增加关节或执行器。
- v8 归回固定白壳的 537 个面全部保留，白壳上缘不会再次被银罩带走。
- 240,000 个简化扫描源三角面完整保留，只改部件归属和随部件使用的材质；没有新增封闭块或改变原外形。
- 静态安装修正仍为相对旧模型 -40°（前端抬高约 40°），A/B/C 零位、关节轴、软件限位、驱动换算及 29 个原驱动文件保留。

## 查看

- `outputs/silver_before_after_raised.png`：升起状态的端盖修正前后对照。
- `outputs/silver_before_after_home.png`：相同零位对照。
- `outputs/silver_integral_comparison.gif` / `.mp4`：左侧上一版，右侧本版。
- `outputs/silver_integral_motion.gif` / `.mp4`：仅显示本版整体升降。
- `outputs/static_side_reference.png`、`static_overview.png`：整机静态图。
- `outputs/three_views/three_views.png` / `.pdf`：三视图。
- `outputs/driver_motion.gif` / `.mp4`：保留的离线电机接口下 A/B/C 联动演示。
- `model.xml` / `model_new.xml`：新针具模型；`model_legacy.xml`：另一针具模型。

局部动图为约 ±5° 的 B 轴仿真扫动，用于检查端盖是否随罩体整体运动。视频参考支持部件刚体归属，尚未标定银罩真实行程、方向或驱动比例；本版沿用之前的滑台映射。联动演示约按 4 倍仿真时间播放。

## 运行

```bash
python -m pip install -r requirements.txt
python repair_silver_cap.py
python build_static_pose.py
python render_seam_preview.py
python preview_static_pose.py
python run_driver_demo.py
python render_three_views.py
```

本目录的 `repair_seam.py` 兼容入口也会执行当前完整修正，不会撤销端盖归属。修复从包内固定的 v8 归属快照构建，可重复执行。

从仓库根目录可直接运行 `python run_reference.py`；本节脚本在 `mujoco_model/` 目录中执行。Linux 渲染默认使用 EGL，Windows/macOS 使用 GLFW，也可自行设置 `MUJOCO_GL`。MuJoCo viewer 可直接打开 `model.xml`。MP4 使用 ffmpeg 从 GIF 转换，Python 渲染脚本无需 ffmpeg。

调整安装角时修改 `installation_config.json` 的 `pitch_y_deg`，然后重跑构建与预览。每次从未安装模板生成，不会累计旋转。`python preview_static_pose.py --pitch -35` 仅生成临时预览，不改配置及模型；不带参数运行恢复默认预览。

## 验证与来源

```bash
python validate_silver_integral.py
python validate_seam.py
python validate_geometry.py
python validate_installation.py
python validate_frames.py
python validate_driver_integration.py
```

报告保存在 `outputs/*validation.json`。端盖专项验证逐面核对源三角面、仅指定归属变化和白壳保护面，并检查两种针具共 70 个姿态（含限位角点）中端盖与长罩的相对距离、关节参数和驱动反馈。`validate_seam.py` 同时检查原白壳修复仍有效。

`assets/silver_cap_faces.npy` 为本次端盖归属索引，`assets/returned_shell_faces.npy` 为之前受保护的白壳索引。`silver_cap_definition.json` 保存端盖局部轴，底缘法向规则另见 `repair_silver_cap.py` 与 `silver_cap_repair.json`。`evidence/v8_meshes/`、`v8_segmentation.npz` 保存本次修正前的基线；视频及观察图也在 `evidence/`。验证不依赖包外旧版本目录。

原驱动来源为提交 `c2a3ffc`，SHA-256 由 `source_manifest.json` 核对。本目录已导入仓库作为当前默认模型；未连接实机或改写 Python/ROS 控制源码，修复前基线保存在 evidence/。

当前仍是外观与运动学原型。照片估计的安装角、真实 RCM、银色滑轨参数、质量惯量、摩擦、电机动态及 ROS 网络传输尚待实机验证；碰撞关闭。原扫描的其他毛边或开口未全面重建，参考板高度和靶体台尺寸仍为展示假设。

仓库根目录的 `scene.xml` 指向本目录资产。`vendor/simulation_driver/` 是提交 `c2a3ffc` 的固定副本；根目录 `simulation_driver/` 保持原样，二者在本次导入时逐项核对 SHA-256 一致。历史导航/OCT 仍使用根目录的旧研究 API，未声称已迁移到本扫描模型。
