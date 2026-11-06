# v9 眼球场景的照片前端

交付入口为 `v9_eye_scene.xml`；`export_v9_eye_scene.py` 会从
`rcm_scene_v9.py` 和 `v9_front_tool.py` 重建同一场景。

前端按用户提供的实物照片分层重建：圆角白色壳体、两侧导杆与支架、
暖灰金属套筒、浅色连接颈与环筋、六角针座、渐缩出口、近端粗管和远端细针。
按用户标注移除侧面示意软管。

除已给定的针具尺寸外，其他尺寸和颜色均为照片估计，不能作为加工尺寸。
针具保持近端长 22 mm / 直径 0.5 mm，远端长 4 mm / 直径 0.06 mm。
照片不能约束亚毫米针尖细节，仍沿用配置的 1.5 mm 渐缩尖端和尖端标记；
它们的传感器语义保持不变。没有用显示倍率夸大针管。

## 方向与运动

针座使用 STL 套筒拟合得到的 `needle_mount` 坐标系。重建的针座沿该坐标系 +Z 连接到壳体，粗管
从 +22 mm 到交界 0，细针从 0 到 -4 mm，朝外延伸。零位交界与 RCM
重合，RCM 位于椭球面上，细针位于眼内。针座与粗细两段针均跟随
`C_translation`，扫描主壳体与原始资产没有改写。

为保持共享逻辑任务的 FK 与编码器接口，注入场景使用
`R_new = R_yaw @ R_install @ R_mount @ R_robot_to_installation @ diag(1, -1, -1)`
的坐标转换，并将 A/C 的显示关节反号，B 不变。虚拟关节轴应用相同的
`R_mount`，C 平移轴与针具同轴。眼球、病灶和相机统一应用该坐标转换；关节/执行器
范围同步镜像。这是仿真呈现转换，不是硬件驱动标定或电机方向更改。
`DriverModel` 和 legacy `Reference` 保留原始约定；不要把导出 XML 的
原生滑块读数直接发送给设备。

### STL 套筒轴线配准

`mujoco_model/fit_v9_tool_mount.py` 直接读取上传 STL 的 5,056,338 个三角面，
核对 SHA-256 后，以现有配准矩阵投影到装配坐标。对前端金属套筒区域的
三角面面积加权，作稳健圆柱拟合。当前版本将侧照估计的 55° 针座倾角与
“零位俯视针轴与长罩长边平行”共同作为轴向约束，再拟合套筒位置和半径。
扫描点不完整，拟合不是独立的平行度测量或实机标定。

结果保存在 `v9_tool_mount.json`，包括区域范围、拟合点数、半径、残差、
安装旋转及参考原点。侧照约束版本的套筒拟合半径约 4.55 mm，径向残差
中位数约 0.56 mm、90 分位约 2.73 mm；这是扫描区域的轴线诊断，
外观套筒当前按正面照片比例使用估计半径 7.2 mm。
轴线先外推至装配坐标 Z=125 mm 的参考平面
（约 X=456.108、Y=71.232 mm），再沿针座 +Z 加上 38 mm 的轴向装配修正。
该修正将源模型 60 mm 示意针的连接基准换算为 22 mm 物理近端针，
使照片前端回到原头部的安装位置；它不是实测 RCM 标定。
参考平面点与当前原点分别保存在 `rcm_fit_plane_registered_m` 和
`rcm_registered_m`，当前后者约为 `[460.158,64.822,162.236] mm`。
原虚拟原点为 [456,70,125] mm；调整原点时补偿扫描部件的局部偏置，使三个
扫描外壳的零位世界姿态保持一致。

用户提供的清晰正面照片确认：白色前壳、套筒与针具居中。
`case_center_xy_tool_m` 因此为 `[0,0]`，导杆及下支架也随壳体居中。
同源 `C_carriage.obj` 的面积中心仅作为诊断记录；扫描表面缺失及
不对称会使面积中心偏移，不能用它推断实物的安装偏置。重复拟合会保留
这条照片给定的居中约束。

重新拟合需提供原始 STL：

```bash
python mujoco_model/fit_v9_tool_mount.py --stl /path/to/Fused_Project_20261028093616.stl
MUJOCO_GL=egl python export_v9_eye_scene.py
MUJOCO_GL=egl python verify_v9_alignment.py
```

改变安装俯仰或整体偏航后，需要重新拟合前端坐标，防止套用旧配准。

### 侧照修订

`v9_side_profile.json` 保存侧照外形估计：针座向下倾角 55°（目测不确定度约
±7°），白壳侧厚 28 mm。依据正面照，白壳正面宽修正为 76 mm、轴向高 72 mm，
与保留的扫描银帽宽度接近；套管、68 mm 下支架及对称导杆按相同照片比例修订。
针座和白壳依旧居中，
针具的 0.5/0.06 mm 外径、22/4 mm 长度保持不变。

`build_side_profile.py` 生成独立显示资产：带 24 mm 宽底部拱形开口的白壳、
有中心通孔的下支架及照片估计的灰色下机壳。透明侧夹、紧固件、银罩条纹和
螺钉、灰壳接口座与警示图形属于外观补建。银罩刻线投射到原曲面，随银罩运动；
下机壳随 A 阶段运动，前端组件随 C 轴运动。已删除的软管保持删除。
这些新增部件均在机器人外观 group 0，不进入针尖/目标传感器，碰撞和实际线缆连接未建模。

修改侧照倾角后先重新拟合；修改壳体尺寸后需重建网格，再重新导出：

```bash
python mujoco_model/fit_v9_tool_mount.py --stl /path/to/Fused_Project_20261028093616.stl
python mujoco_model/build_v9_front_display.py
python mujoco_model/build_side_profile.py
MUJOCO_GL=egl python export_v9_eye_scene.py
python verify_v9_side_profile.py
MUJOCO_GL=egl python verify_v9_front_tool.py
MUJOCO_GL=egl python verify_rcm_scene_v9.py
```

网格构建与开口检查使用 `requirements-mesh.txt` 的可选工具依赖。
随模型保存的网格可直接由普通运行入口加载，不要求运行时安装网格构建工具。

重建壳体与连接件在 group 0，传感器过滤它们；物理针在 group 3，默认
viewer 使用 group 2 的同尺寸副本。原有遮挡针管的 `axis_aid` 已从此
注入场景移除。碰撞仍关闭，没有宣称组织接触或刚体动力学有效。

视觉伺服窗口关闭 group 1 以隐藏环境和示意连杆。照片前端及侧面补件使用
group 0，确保隐藏环境时白壳、套筒、针座和支架仍完整显示；严格 RGB-D
观测继续只使用 group 3/4/5。可用 `python verify_v9_viewer.py` 检查实际
窗口选项、模型可见性和光学观测隔离。

### 正面装配与旧扫描前壳替换

`build_v9_front_display.py` 从 `distal_guide_smooth.obj` 派生闭合上帽/导轨网格，
以当前针座坐标 Z=117 mm 为替换边界，移除会遮住新白壳的旧前面板和旧左夹。
派生文件为 `distal_guide_front_display.obj`；原扫描资产保留，保留部分的零位
世界位置保持。切面和源/输出哈希记录在 `mujoco_model/v9_front_display_build.json`。
白壳、套管和针具跟随 C，银帽/导轨仍跟随原 B 部件，左侧透明夹保持独立外观。

轴向基准修正同时更新场景中的虚拟 RCM、眼球和相机位置；逻辑 A/B/C、
编码器接口和物理针尺寸保持一致。旧版本的世界坐标及标定快照属于其原版本，
新采集应使用当前模型与标定；不能将本次外观修正表述为实机标定。

使用 `render_v9_three_views.py` 导出当前实际模型的完整世界三视图：正视沿
世界 +X、侧视沿世界 −Y、俯视沿世界 +Z，三者同零位、同尺度且相互正交。
`front_assembly.png` 和 `front_detail.png` 另沿针座局部 +X 核对照片正面，
分别标注为前端组件和针座局部图，不能混作世界工程三视。

```bash
MUJOCO_GL=egl python render_v9_three_views.py --output-dir runs/current_three_views --size 1200
MUJOCO_GL=egl python verify_v9_front_registration.py
```

## 俯视轴向对齐

v9 眼球场景按 `v9_alignment.json` 围绕安装支点增加世界 Z 轴偏航
约 -17.103°，使长罩在俯视图中的长边沿世界 X 轴水平。此变换写入
`v9_eye_scene.xml`，不是对渲染图片做旋转。机器人、针座、眼球、RCM、
病灶及相机使用一致的坐标变换；支撑面仍保持水平，原驱动安装配置及
源网格不变。原照片估计的 -40° 安装俯仰角保留。这是显示坐标对齐，
不是实机安装标定。

`verify_v9_alignment.py` 从实际导出的长罩网格检查俯视主方向，并检查
针轴平行、套筒与 C 轴同轴、扫描外壳零位姿态、支撑面水平、RCM 位于眼表及
针尖位于眼内。需要网格工具依赖。

## 扫描来源

本次上传的五个 7z 分卷已成功解压，得到
`Fused_Project_20261028093616.stl`，252,816,984 字节。
SHA-256：`1a005c8579cfce3db6a61a4620f75ba6cdba9ac5d84816b50e13c596b544d579`。
它与 `mujoco_model/enclosure_reference.json` 登记的源文件完全一致；
因此复用仓库既有配准和扫描主体，无需再次简化或替换整机网格。
分卷和解压文件保留在仓库外的 `/workspace/scan_reference_20261028` 及附件目录。

## 扫描外壳去毛边

此眼球场景使用 `assets/*_smooth.obj` 显示外壳，原始扫描及
`*_enclosed.obj` 均保留。`mujoco_model/smooth_display_shells.py` 以
0.8 mm 体素重建封闭外皮，填平小裂隙、移除孤立碎片，再进行曲面平滑。
薄小纹理、窄凹槽及推定补面属于显示重建，不再代表原扫描逐面几何；
不能用作机械加工、组织接触或碰撞标定。关节、针具、眼球和相机坐标不变。

重建依赖 `requirements-mesh.txt`。生成和平滑验证命令：

```bash
python mujoco_model/smooth_display_shells.py
MUJOCO_GL=egl python export_v9_eye_scene.py
python verify_smooth_shells.py
```

各部件的源/输出 SHA-256、封闭性、体积和表面偏移记录于
`mujoco_model/smooth_shells_report.json`。

## 检查

`verify_v9_front_tool.py` 检查 27 个 A/B/C 组合、逻辑 FK 映射、默认
可见针管尺寸、C 滑台归属，以及实际导出 XML 的眼球入口关系。
`verify_rcm_scene_v9.py` 另检查编码器、坐标传感器、眼内 RGB-D/分割和
C=8 mm 的视觉针尖观测；这不是完整闭环导航或实机验证。
