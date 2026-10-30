# 当前 MuJoCo 扫描模型与重建工具

当前保留机械基础场景 `../scene.xml` 和包含照片前端、眼球及相机的 `../v9_eye_scene.xml`。查看照片修订后的建模请打开后者；详细参数见 [前端说明](../V9_FRONT_TOOL.md)，最新三视图与完整视觉伺服视频见 [2026-10-30 交付](../reports/20261030/README.md)。

v5–v8 的历史 XML、网格和实现快照、旧版修复/对照脚本，以及重复导出 `model*.xml` 已移除。[清理前版本](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/tree/64d866b104b519e7f0b26eb81913f26f98259080/mujoco_model) 保留完整历史。当前加载器读取 `assets/scan_template.xml`；它是仍在使用的未安装坐标模板，内容与清理前运行模板一致。

## 运行

从仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python run_reference.py
python export_v9_eye_scene.py
python -m mujoco.viewer --mjcf v9_eye_scene.xml
```

`run_reference.py` 是无眼球基础模型的离线运动查看器；`RCMSceneV9` 在同一机械模型上注入照片前端和眼球场景。相机、针具、眼球与 RCM 的映射及模型假设见前端说明。研究导航和视觉伺服源码保留，默认历史研究入口仍动态构造其参考场景，不依赖已删除的 XML 导出。

## 保留的资产

- `assets/*_smooth.obj` 和 `photo_*_side_r3.obj`：当前眼球场景使用的外壳和前端。
- `assets/*_enclosed.obj`：基础模型使用的闭合外皮，也是平滑构建的输入。
- 分件源 OBJ、`fixed_housing_clean.obj`、`registered_scan_world.obj`、`segmentation.npz`：当前网格构建、源面覆盖和归属验证的输入。
- `silver_cap_faces.npy`、`returned_shell_faces.npy` 及对应 JSON：银罩端盖与白壳保护面的当前归属记录。
- `registration.json`、`source_scan_info.json`、`enclosure_reference.json`：扫描配准、来源和原始 STL 校验信息。
- `evidence/` 中的实物照片、参考运动视频及观察记录：建模依据。
- `vendor/simulation_driver/`：提交 `c2a3ffc` 的固定驱动副本，`source_manifest.json` 保存其 SHA-256。根目录的原始驱动也保留。

分件源网格及构建输入仍用于复验和重建，因此保留。已删除的完整旧扫描副本和参考网格不参与当前运行；原始用户 STL 位于仓库外，需要重新拟合时按前端说明传入路径。

## 当前验证

从仓库根目录执行：

```bash
python run_reference.py --verify
python verify_current_model.py
python verify_enclosed_shells.py
python verify_smooth_shells.py
python verify_v9_alignment.py
python verify_v9_front_tool.py
python verify_v9_side_profile.py
python verify_rcm_scene_v9.py
```

Linux 云端渲染命令设置 `MUJOCO_GL=egl`。网格检查需安装根目录 `requirements-mesh.txt` 的可选依赖。

`--verify` 检查当前两种针具配置的编码器/FK、27 个姿态的源网格归属与挂载、闭合外壳、照片前端的 27 个任务姿态及导出 XML 的眼球入口。银罩的 1,797 个端盖面必须属于同一个滑台，537 个白壳保护面必须保留在白壳；分件应覆盖全部 240,000 个简化扫描面。原始驱动的 29 个文件仍逐项核对哈希。

它不再执行 v6–v8 实现之间的差分验证，不能把这些当前检查当成重新完成旧版对照实验。生成报告写入临时目录或显式输出位置；`outputs/` 仅保留目录说明，不提交新的临时结果。

## 重建与预览

从仓库根目录安装可选网格依赖后：

```bash
python mujoco_model/clean_display_mesh.py
python mujoco_model/build_enclosed_shells.py
python mujoco_model/smooth_display_shells.py
python mujoco_model/build_side_profile.py
python export_v9_eye_scene.py
python mujoco_model/render_three_views.py --background black --robot-only --output-dir runs/base_three_views
```

这些脚本可能重写独立显示网格，重建后应重新运行对应检查。`render_three_views.py` 显示基础模型；照片前端与眼球的当前预览见交付目录。`build_static_pose.py` 可以按需生成基础场景的 `model*.xml`，这些重复导出不再纳入版本控制。

`run_driver_demo.py`、`motor_backend.py` 和 `ros_sim_bridge.py` 保留驱动相关研究用途。常规离线运行不连接设备、不加载实验室 ROS 网络设置；真实 ROS 通信仍未验证。

## 标定与限制

实机采集流程见 [CALIBRATION.md](CALIBRATION.md)，工具为 `calibrate_model.py`。仿真安装角、真实 RCM、银罩传动、质量惯量、摩擦及电机动态仍未实机标定；碰撞关闭。显示修补和平滑不代表机械加工、壁厚或组织接触几何，验证中的微米级数值不代表硬件精度。
