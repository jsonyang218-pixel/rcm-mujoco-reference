# v9 视觉窗口前端显示修复

视觉伺服窗口隐藏 group 1 的环境和示意连杆；照片重建的白壳、套管、针座与侧面补件此前也使用 group 1，因此被一起隐藏。修复将这 63 件机器人外观归入 group 0，保留其位置、尺寸、材料及运动归属。

- [前端修复前后](front_before_after.png) · [整机修复前后](full_before_after.png)
- [实际窗口选项及 4 项回归](viewer_verification.json)
- [严格 RGB-D 的 10 项回归](strict_rgbd_verification.json)
- [同模型、同姿态和相机的像素对照](visibility_comparison.json)
- [静态 XML 仅修改 63 处分组的核查](xml_comparison.json)

对照图的上排为零位，下排为用户截图的近似姿态 A=4.97°、B=-7.22°、C=19.29 mm。左右两列保持相同模型、姿态、相机、光照和 `[1,0,1,1,1,1]` 窗口过滤，只切换照片件的 group 1/0。图中针具未放大；照片前端和侧面补件仍是既有照片估计外形，扫描主体未重新建模。

真实 `run_viewer()` 的初始化和每帧过滤在无 GUI 的窗口边界替身下验证，实际可见像素由 MuJoCo 渲染检查。零位与截图近似姿态下，严格控制 RGB 和正常深度在修复前后逐元素完全一致。感知回归 10/10、窗口回归 4/4、前端 27 姿态、侧照几何及轴线对齐检查全部通过。本轮验证显示修复与观测隔离，未复跑完整导航，也不代表真实硬件或 OCT 验证。

复现：

```bash
MUJOCO_GL=egl python verify_v9_viewer.py --output-dir runs/viewer_checks
MUJOCO_GL=egl python verify_strict_rgbd.py --output-dir runs/viewer_strict_checks
MUJOCO_GL=egl python verify_v9_front_tool.py
python verify_v9_side_profile.py
python verify_v9_alignment.py
```

更新后关闭旧窗口，在新项目目录运行 `python run_visual_servo.py --camera overview`，以重新加载完整 v9 眼球及照片前端。
