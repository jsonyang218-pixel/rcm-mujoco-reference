# 当前 v9：正面照片修订后的完整三视图

本轮修正了照片白壳偏低、旧扫描前面板/左夹重复遮挡以及白壳、套管、下支架比例。针具、套管和白壳沿同一中心轴；透明夹位于照片左侧。所有图均由当前真实 `RCMSceneV9` 的零位导出。

- [完整三视图 PNG](three_views.png) · [PDF](three_views.pdf)
- 单图：[正视](front.png) · [侧视](side.png) · [俯视](top.png)
- 按照片方向核对：[银帽及前端组件正面](front_assembly.png) · [针座局部正面](front_detail.png)
- [相机、比例、坐标检查和源码/网格哈希](view_parameters.json)
- [全部回归摘要](regression_summary.json) · [正面装配检查](front_registration_verification.json)
- [严格 RGB-D 检查](strict_rgbd_verification.json) · [窗口检查](viewer_verification.json)

## 三视图方向

三张完整视图使用世界坐标的相互正交方向，共享 A=0°、B=0°、C=0 mm 和 **0.387825376 mm/px**：正视沿 +X 看 −X，侧视沿 −Y 看 +Y，俯视沿 +Z 看 −Z。整机保持真实安装姿态，俯视针轴及长罩沿水平 X，侧视针轴向下约 55°。同一组几何没有为各视图分别摆正或重排，也没有为了外观删掉后方扫描主体/基座。

`front_assembly.png` 与 `front_detail.png` 属于明确标注的局部图，按针座局部 +X 观看、+Z 朝上，与用户正面照片对应；它们与世界三视的相机基准不同。针管没有按显示倍率放大。

## 建模与标定范围

白壳估计尺寸为厚 28、宽 76、高 72 mm，底部拱口半径 12 mm；外观套管半径 7.2 mm、长 26 mm，下支架宽 68 mm，双导杆间距 58 mm。它们来自照片比例估计，不是加工尺寸。物理针仍为近端 22 mm/半径 0.25 mm、远端 4 mm/半径 0.03 mm。

旧扫描前面板/左夹由闭合派生上帽/导轨资产替换，原扫描源保持。装配轴向原点沿既有针座 +Z 修正 38 mm，将原 60 mm 示意针的连接基准换算到 22 mm 物理近端针；眼球、相机和虚拟 RCM 同步，保留扫描表面的零位世界位置。该原点和外形均未经实机标定，旧版本的世界坐标应结合其自己的标定快照使用。

验证包括：前端 27 姿态、闭合/开孔、世界轴线/眼表 RCM、正面首射线无旧壳遮挡、严格感知 10 项、窗口 4 项以及电机执行 7 项。本轮未复跑完整导航或重新采集训练集；既有人工标记、理想仿真深度、实际速率超限的范围继续适用。

## 复现

```bash
MUJOCO_GL=egl python render_v9_three_views.py --output-dir runs/current_three_views --size 1200
MUJOCO_GL=egl python verify_v9_front_registration.py
MUJOCO_GL=egl python verify_v9_front_tool.py
python verify_v9_side_profile.py
python verify_v9_alignment.py
```

渲染时的 Git 提交仅作为基线，未提交修订通过 `source_tree_dirty` 明确标记；文件与网格 SHA-256 记录实际渲染版本。
