# 2026-10-23 组会：多目标严格视觉控制与实体遮挡

> 历史记录：旧运行产物已清理，源码与复现配置仍保留。图表、日志和当时数据目录请参阅[清理前归档](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/tree/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022)，文中本地数据路径不表示当前仍有这些文件。

生成时间：2026-10-22T12:40:36.551544+00:00。记录状态：**完整最终证据**。所有结果数字读取本地最终 JSON/JSONL；本脚本不启动导航、不补写结果。

## 真实结果与分母

| 实验 | 落盘状态/episode结果 | 成功/计划 | 成功/已尝试 | 最大已尝试终点误差 µm | 帧 |
|---|---|---|---|---|---|
| 混合三点对照（场景解析转移 + 视觉接近） | COMPLETE / PASS | 3/3 | 3/3 | 6.301 | 1146 |
| 严格 RGB-D 三点（零位起始） | COMPLETE / PASS | 3/3 | 3/3 | 6.634 | 1289 |
| 严格视觉 + 实体遮挡 / stop | COMPLETE / FAIL | 0/3 | 0/1 | 1504.051 | 29 |
| 严格视觉 + 实体遮挡 / static_memory_reobserve | COMPLETE / PASS | 3/3 | 3/3 | 6.442 | 669 |

成功定义：真实终点误差严格小于 20 µm，且该目标 phase report 记录至少 3 帧零命令速度停稳；失败 episode、未运行目标均保留。成功/计划将失败后未尝试目标计入分母；成功/已尝试只描述实际尝试，二者不能互换。结果 FAIL 表示实验结局，不表示材料构建失败。

![True endpoint errors](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/endpoint_errors.png)

真实误差仅用于独立验收。回撤和方向对准的控制误差指各自视觉构造的阶段目标，不能把它解释为到物理标靶的真实误差。

## 混合三点对照（场景解析转移 + 视觉接近）

独立复核（与导航结果分开）：完整性总结果=PASS；schema=PASS；动作对齐=PASS；状态重渲次数=20；重渲最大不同像素=0；独立边界速率=PASS。

| 目标 | 尝试 | 记录结果/原因 | 真实误差 µm | 零速帧 | 独立成功 |
|---|---|---|---|---|---|
| P3 | True | PASS / POSITION_ERROR_AND_SETTLED | 6.218 | 3 | True |
| P2 | True | PASS / POSITION_ERROR_AND_SETTLED | 6.225 | 3 | True |
| P1 | True | PASS / POSITION_ERROR_AND_SETTLED | 6.301 | 3 | True |

导出帧 visibility：`{"clear": 995, "partial_needle": 151, "full_needle": 0, "not_visible": 0, "unknown": 0}`；控制阶段计数：`{"memory_frame_count": null, "recovery_attempt_count": null, "recovery_success_count": null}`。这些计数不直接等价于成功率。

![Time series](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/timeline_baseline_three_ep000000.png)



## 严格 RGB-D 三点（零位起始）

独立复核（与导航结果分开）：完整性总结果=PASS；schema=PASS；动作对齐=PASS；状态重渲次数=19；重渲最大不同像素=0；独立边界速率=PASS。

| 目标 | 尝试 | 记录结果/原因 | 真实误差 µm | 零速帧 | 独立成功 |
|---|---|---|---|---|---|
| P1 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.634 | 3 | True |
| P2 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.085 | 3 | True |
| P3 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.268 | 3 | True |

导出帧 visibility：`{"clear": 1061, "partial_needle": 228, "full_needle": 0, "not_visible": 0, "unknown": 0}`；控制阶段计数：`{"memory_frame_count": null, "recovery_attempt_count": null, "recovery_success_count": null}`。这些计数不直接等价于成功率。

![Time series](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/timeline_strict_three_ep000000.png)



## 严格视觉 + 实体遮挡 / stop

独立复核（与导航结果分开）：完整性总结果=PASS；schema=PASS；动作对齐=PASS；状态重渲次数=6；重渲最大不同像素=0；独立边界速率=FAIL。

**速率失败必须保留**：失去目标后仿真执行瞬停；独立审计最大 ABC 加速度为 `[[0.01220703125, 0.2905845642089844, 12.000274658203113]]` deg/s²、deg/s²、mm/s²。完整性 PASS 不代表运动速率 PASS，不能称为有界安全制动。

| 目标 | 尝试 | 记录结果/原因 | 真实误差 µm | 零速帧 | 独立成功 |
|---|---|---|---|---|---|
| P1 | True | FAIL / TARGET_NOT_VISIBLE | 1504.051 | 0 | False |
| P2 | False | NOT_RUN / PREVIOUS_VISUAL_PHASE_FAILED | — | — | False |
| P3 | False | NOT_RUN / PREVIOUS_VISUAL_PHASE_FAILED | — | — | False |

导出帧 visibility：`{"clear": 28, "partial_needle": 0, "full_needle": 0, "not_visible": 1, "unknown": 0}`；控制阶段计数：`{"memory_frame_count": null, "recovery_attempt_count": null, "recovery_success_count": null}`。这些计数不直接等价于成功率。

![Time series](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/timeline_strict_occlusion_stop_ep000000.png)



## 严格视觉 + 实体遮挡 / static_memory_reobserve

独立复核（与导航结果分开）：完整性总结果=PASS；schema=PASS；动作对齐=PASS；状态重渲次数=30；重渲最大不同像素=0；独立边界速率=PASS。

| 目标 | 尝试 | 记录结果/原因 | 真实误差 µm | 零速帧 | 独立成功 |
|---|---|---|---|---|---|
| P1 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.382 | 3 | True |
| P2 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.442 | 3 | True |
| P3 | True | PASS / VISUAL_ARRIVAL_AND_STOP | 6.283 | 3 | True |

导出帧 visibility：`{"clear": 398, "partial_needle": 228, "full_needle": 0, "not_visible": 43, "unknown": 0}`；控制阶段计数：`{"memory_frame_count": 24, "recovery_attempt_count": 1, "recovery_success_count": 1}`。这些计数不直接等价于成功率。

![Time series](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/timeline_strict_occlusion_recovery_ep000000.png)



## 已执行轨迹与协议可比性

![Executed tip path from independent evaluation](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/executed_strict_path.png)

严格三点图取 `visual_servo/trajectory.csv` 的已执行 world XYZ，目标散点取验收报告；这是 independent evaluation 的执行轨迹，不是模型规划路径。展示跨点轴向回撤、转向、接近的实际日志。

混合对照访问顺序为 P3→P2→P1，严格三点为 P1→P2→P3，且控制权限不同。不能把总运动时间差单独归因于严格视觉控制。

[严格三点派生视频预览（若文件已生成）](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/strict_three_preview.mp4)：20 fps 展示预览；事件与端点的精确时间以原始 JSONL timestamp 为准。预览不替代 raw RGB、state/action 或真实误差验收。

## 实体球计划事件

| 事件 | 目标/phase | 相对phase开始frame | 持续frame | 名义持续s |
|---|---|---|---|---|
| P1_short_full | P1 / visual_approach | 20 | 8 | 0.400 |
| P2_long_full | P2 / visual_approach | 20 | 35 | 1.750 |

实体球 radius=0.3 mm，ray_fraction=0.75。表格是计划 schedule；实际不可见帧和恢复控制状态见前述时间图与 CSV。

## 设计与控制权限

| 目标 | robot XYZ mm | 归一化 posterior Z | 椭球方程值 | 径向表面差 µm | 验证 |
|---|---|---|---|---|---|
| P1 | 2.640000000, 4.508556435, 23.238969794 | 0.970360758 | 0.999999999999999 | 7.354e-12 | PASS |
| P2 | -0.960000000, 7.664776747, 22.589962903 | 0.980408078 | 0.999999999999998 | 9.091e-12 | PASS |
| P3 | 1.920000000, 8.477994401, 22.043001744 | 0.952260469 | 0.999999999999999 | 7.408e-12 | PASS |

眼底定义来自 `fundus_target_verification.json`：Compiled eye ellipsoid posterior cap as fundus proxy; no separate retinal layer。椭球表面约束与后部帽区域是几何证明，不是实际视网膜组织或病灶测量。

严格三点从零位按 P1→P2→P3 给定顺序逐个呈现目标。视觉观测构造轴向回撤、AB 方向对准、C 接近；不预计算目标 IK 路径。运行时禁止以 `target_task` / `nominal.inverse` 提供场景目标给控制；允许已知相机、RCM、尺寸、执行 Jacobian 与眼壁安全约束。场景 XYZ 仅用于放置 marker 和独立验收。当前实验没有验证同一画面中的多病灶识别与自主排序。混合对照包含场景解析转移，单独报告。

遮挡 stop/recovery 配对协议检查：**True**。相同三目标、相同 reset、相同实体球 schedule；完整差异见 `metrics.json/paired_protocol`。reset 来源为 `retracted_start.json` 所指严格三点 P1 实际视觉回撤记录，不是目标 IK。实体球半径、ray fraction、事件开始和时长取各自 protocol；计划 schedule 不保证失败提前终止的 episode 实际看满全部遮挡。

默认 stop 不启用记忆。opt-in 静态目标记忆需成熟的 3 次观测，受 1 s / 0.5 mm 限制，重获偏移超过 40 µm 拒绝；恢复最多 3 次，每次 4 s / 0.5 mm C 回撤。当前针尖必须始终为 fresh 观测。该设计假设相机固定、目标静态；这些限制是设计条件，非统计置信界或硬件性能。实际记忆帧数、恢复尝试/成功数从 phase report 读取。

自然针遮挡的 `partial_needle/full_needle` 与实体球造成、无针反事实中仍不存在的 `not_visible` 分开计数。无针渲染只用于离线诊断，不能作为控制输入；可见像素和 event id 用于核查计划遮挡与实际曝光。

## 数据规模与动作审计

| 实验 | episode | JSONL行 | PNG | 全dataset MiB | 本脚本审计 |
|---|---|---|---|---|---|
| 混合三点对照（场景解析转移 + 视觉接近） | 1 | 1146 | 1146 | 341.90 | True |
| 严格 RGB-D 三点（零位起始） | 1 | 1289 | 1289 | 384.36 | True |
| 严格视觉 + 实体遮挡 / stop | 1 | 29 | 29 | 8.64 | True |
| 严格视觉 + 实体遮挡 / static_memory_reobserve | 1 | 669 | 669 | 199.97 | True |

名义采样 20 Hz；图像与当前 state 在 timestamp t 先于 action，action 是下一帧实际 state 的绝对 A/B/C 命令。阶段端点或失败观测可以产生短于 0.05 s 的间隔。逐 episode 审计 float32 action[:-1] 与 next-state 精确相等、terminal hold、PNG存在、policy/action NPZ 对齐和九个 perception 字段；具体误差与 dt 范围见 metrics.json。混合旧数据若无 perception 明示 LEGACY_UNAVAILABLE，不补写。

Policy 输入严格为 RGB路径、state、task；action 监督单列保存，phase/visibility/perception/场景坐标均为审计字段。RGB路径必须解码为图像，不能把文件名作为特征。paired no-image bundle 仅说明接口可做消融，当前没有训练/验证/测试 policy 对比。

这是固定目标、给定顺序、少量相关 episode 的 pilot，尚非可训练数据集。不能按相邻帧随机拆分；未来按完整 episode/场景/目标布局分组拆分，并独立留出遮挡设置及失败案例，先扩大独立场景和运行数量。失败 episode 保留，禁止只筛选 PASS。

## 原始图像证据

![Raw RGB montage](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/raw/42a0b2855b343d4efced99dee615cfa6a5d2be4c/reports/20261022/meeting/figures/raw_frame_montage.png)

截图是控制器所用 raw RGB PNG 的逐字节副本；拼图仅按显示比例缩放，未裁剪、未叠加标注或修改像素。每张原始图、SHA256、帧索引和可见性/控制状态见 rawframes.json。Viewer 的放大针/目标辅助不属于这些 raw 传感器截图。

## 可复现、证据包与局限

生成命令：`python runs/multitarget_occlusion_20261022/build_meeting_record.py`。仅写当前运行目录下 meeting/，不改源码和原始实验。`--allow-incomplete` 仅用于过程 review，材料会显著标明未完成。`metrics.json` 保存逐目标结果、动作审计、完整协议与源文件哈希，CSVs 方便复核；每张图另存 PDF。`slides.html` 为 6 页自包含 HTML，可离线打开，用方向键/空格切换并打印 PDF。

`evidence_light.tar.gz`（除非 --no-archive）保留 reports/configs/CSVs、源代码快照与哈希、完整 JSONL/小型 NPZ、关键 raw PNG、脚本和组会材料；不含完整 RGB 图像库。完整数据位置见 DATASET_LINKS.md。源代码快照与采集时 hash 不一致时会单独标明，不能把当前快照冒充采集版本。

没有硬件执行、物理制动/接触验证、真实 OCT 噪声、真实眼动/组织变形或临床精度结论。理想仿真深度、分割与固定标定是专家权限，RGB-only 学习能力尚未验证。单次少量 episode 无法估计泛化成功率或统计显著性。

组织框架参考：[Supervisor-Skills / benchmark-paper-template](https://github.com/HKUSTDial/Supervisor-Skills/blob/a20d76ef3979b804d7dbf599d577826957e0a8a9/skills/benchmark-paper-template/SKILL.md)，固定 commit `a20d76ef3979b804d7dbf599d577826957e0a8a9`。仅借用数据采集与分层评价组织方法；不是性能结果来源，不套用 LLM benchmark 结论，不引用不存在论文。

## 深度传感器版本边界

本报告四组多目标 run 在 OCT 包装接口接入前完成，使用相同的 MuJoCo 理想配准深度与几何分割权限。保存的传感器图像是 raw RGB；这些旧数据不含 raw OCT，不能追溯改称 OCT 闭环实验。后续 OCT 接口/传感器闭环验证须单独编号与报告，且仍需区分仿真理想深度与真实 OCT 噪声。
