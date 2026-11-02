# 修正上一版仍然空缺的侧面承托结构

本轮修改前：`55e28b4`。用户指出薄银色背板后方仍然空缺；本轮补的是这个区域的实体承托结构。先前 `reports/20261102/redframe/` 是历史版本，最新图片在本目录。

- [前后对照（左：55e28b4；右：本轮）](local_before_after.png)
- [当前完整侧视](three_views/side.png)
- [当前完整三视图](three_views/three_views.png) · [PDF](three_views/three_views.pdf)
- [局部结构、框外保持与间隙检查](local_verification.json)
- [严格 RGB-D](strict_rgbd_verification.json) · [电机回归](motor_verification.txt)

新增封闭白色承托体，连接原银帽下缘及银色背板背面，随 B 部件运动。它由针座局部 X/Z 六边形轮廓沿 Y 挤出；轴向 45–119 mm、深度 X=-52 至 -22 mm、横向 Y=±24 mm，均为照片估计，未进行实机标定。C 轴白壳、夹具、针座、针具，以及框外主壳/长罩/底座没有变更。新增显示几何质量为零，保持原编译质量惯量；碰撞仍关闭。

验证不仅检查薄板，而是检查报告空腔内的 3 条侧面首射线均命中新承托体；承托体为闭合正体积实体。45 组 B/C 全关节行程组合检查包含承托体和薄板，距 C 组件的最小几何间隙仍为 3.5 mm。原始 `ffd7b15` 框外 MJCF/网格/配置/质量惯量比较仍通过，没有重复应用 38 mm 调整。

复现（仓库根目录，项目虚拟环境，`MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa`）：

```bash
.venv/bin/python verify_v9_redframe.py --baseline reports/20261102/redframe/baseline.json --report runs/gap-verification.json
.venv/bin/python verify_strict_rgbd.py --output-dir runs/gap-checks
.venv/bin/python verify_motor_execution.py
.venv/bin/python render_v9_three_views.py --output-dir runs/gap-three-views
git show 55e28b4:v9_eye_scene.xml > /tmp/rcm-gap-before.xml
.venv/bin/python render_v9_redframe_comparison.py --baseline-xml /tmp/rcm-gap-before.xml --baseline-label 55e28b4 --output-dir runs/gap-comparison
```

严格 RGB-D 仍为人工标记加理想仿真深度；本轮没有重跑完整导航、没有修复历史速度/加速度超限、没有接入真实 OCT。未发布云环境。
