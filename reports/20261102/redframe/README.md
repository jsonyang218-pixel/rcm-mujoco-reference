# 红框局部修订（2026-11-02）

基线：`ffd7b15`，分支 `codex/v9-integration-20261030`。仅改变照片红框内的显示结构，尺寸仍为照片估计，未经实机标定。

- [同视角局部前后对照](local_before_after.png)
- [当前完整三视图](after/three_views.png) · [PDF](after/three_views.pdf)
- [当前前端组件正面](after/front_assembly.png) · [当前侧视](after/side.png)
- [局部范围及间隙验证](local_verification.json)
- [严格 RGB-D](strict_rgbd_verification.json) · [窗口显示分组](viewer_verification.json)
- [正面装配回归](front_registration_verification.json) · [电机回归](motor_verification.txt)
- [修改前基线快照](baseline.json) · [实际渲染文件/网格哈希与相机](after/view_parameters.json)

新增背板随 B 运动；白壳、透明夹、套管和针座仍随 C 运动。背板上端与原银色导向件重叠连接，C 组件与背板留有最小 3.5 mm 几何间隙。白壳仅下端 3 mm 内的深度侧角圆滑化，保留原网格资产及全部上部顶点。交付 XML 另逐项保留基线的框外节点文本，避免跨平台浮点格式化引入无关差异。上银帽、主壳、长罩、底座及现有 38 mm 基准、RCM、眼球、相机、物理针、关节和电机参数保持。

针座曾由显示网格推算质量惯量，本次将 `ffd7b15` 的推算结果显式冻结；没有把新增显示细节的质量引入电机模型。范围回归逐项比较 MJCF 与编译后质量惯量，不以图片相似代替验证。

## 复现

从仓库根目录，使用项目虚拟环境和 `MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa MPLBACKEND=Agg`。常规运行不需要重新生成网格。

```bash
.venv/bin/python verify_v9_redframe.py --baseline reports/20261102/redframe/baseline.json --report runs/redframe-verification.json
.venv/bin/python verify_v9_front_registration.py --output-dir runs/redframe-checks
.venv/bin/python verify_strict_rgbd.py --output-dir runs/redframe-checks
.venv/bin/python verify_v9_viewer.py --output-dir runs/redframe-checks
.venv/bin/python verify_motor_execution.py
.venv/bin/python render_v9_three_views.py --output-dir runs/redframe-three-views
# 重建圆角资产（仅此局部新资产）：
.venv/bin/python mujoco_model/build_redframe_case.py
# 重建前后对照：
git show ffd7b15:v9_eye_scene.xml > /tmp/rcm-redframe-baseline.xml
.venv/bin/python render_v9_redframe_comparison.py --baseline-xml /tmp/rcm-redframe-baseline.xml --output-dir runs/redframe-comparison
```

网格射线/包含检查还使用 trimesh 的可选 `rtree`；既有侧面网格回归也需要它。当前云环境已安装 shapely、mapbox-earcut、rtree、networkx；最终圆角构建使用原有端面三角形分层连接，不依赖 Boolean 切面。项目依赖声明未改动。

结果：局部检查 PASS，45 组 B/C 全行程组合；上端内部连接探针 27/27；装配 6/6、严格 RGB-D 10/10、窗口分组 4/4、电机 7/7 PASS，前端 27 姿态与侧面回归 PASS。完整三视图同零位、正交、统一比例；针具没有放大。局部对照为前端组件显示，不能代替完整三视图。

碰撞仍关闭；尚未标定硬件。未重跑完整导航、未重采数据集、未验证真实 OCT，历史速度/加速度超限问题保持未解决。本轮未发布云环境。
