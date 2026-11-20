# 680 步切档点三轴诊断与下阶段破局规划

基于 [`be0da86`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/be0da86d32102035cce7c9817ab30aa1d16f9c3e)（[overnight_fast5_20261120_v1_partial10/SUMMARY.md](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/be0da86d32102035cce7c9817ab30aa1d16f9c3e/rcm_4090_deploy/github_reports/overnight_fast5_20261120_v1_partial10/SUMMARY.md)）的最新状态制定。当前 C 轴大回撤已消除，但所有轨迹在退出满速饱和段（step ~680）后稳定在 **`20–130 µm`** 水平振荡带（`minimum_mean = 27.852 µm`，`strict_success = 0/10`）。本规划先用 15 分钟三轴定量诊断锁定 `20–50 µm` 地板的主导轴与物理/模型成因，再按判据分流至对应修复方案。

## 阶段 A：680 步切档点与稳态带三轴定量诊断（零训练，~15 min）

直接复用 `step_002000.pth`（`I0`）已记录的 `E00, E02, E06, E08` 逐帧日志（不重跑闭环 rollout），分别分析两个时间窗口：
- **窗口 1（切档瞬间）**：`step 640–800`，观察针尖退出 `+0.030 mm/step` 饱和区、拐入水平带时 A/B/C 三轴的瞬时响应；
- **窗口 2（稳态地板）**：`step 1000–1600`，量化后半程 600 步内托住 `20–130 µm` 地板的真实来源。

### 必算 4 项定量指标

| 指标项 | 计算方式 | 判别意义 |
| :--- | :--- | :--- |
| **1. 三轴针尖位移贡献（µm）** | 在 `step 700–1600` 用当前位姿雅可比 `J(q)` 的三列分别计算 `‖J_A ΔA‖`、`‖J_B ΔB‖`、`‖J_C ΔC‖` 的中位数与均值 | 直接锁定这 `20–60 µm` 误差里 A、B、C 谁占大头 |
| **2. 静差（Bias）vs 振荡（Std）** | 统计 `step 1000–1600` 各轴残差 `state - target` 的均值、标准差与过零次数 | 区分“单轴偏在一侧不过零”还是“围绕 0 高频抖动” |
| **3. 死区与最小步进量化** | 统计 `step 700–1600` 各轴 `|applied_cmd| > 0` 但 `|actual_change| < 1e-6` 的帧占比，以及非零实际步长的最小量子对应的针尖位移（µm） | 排查 A 轴阶梯平台等执行器死区是否已形成物理下界 |
| **4. 策略 vs 教师指令幅值比** | 在同状态下重算单步零初速教师指令（`zero_vel_teacher`），计算 `median(|policy_cmd|) / median(|teacher_cmd|)` | 衡量全局线性输出尺度是否比近目标所需步长粗一个量级 |

### 阶段 A 产出清单
- 目录：`rcm_4090_deploy/github_reports/diag_20261120_step680_axes/`
- 文件：
  - `AXIS_FLOOR_DIAGNOSTICS.md` 与 `axis_floor_summary.json`（含上述 4 项指标汇总表与自动分流建议）；
  - `{E00,E02,E06,E08}_step640_800.png`（3 行 A/B/C × 2 列：左列带符号残差，右列策略指令 / 实际变化 / 零初速教师）；
  - `axis_contribution_bar.png`（各目标 `step 700–1600` 的 A/B/C 针尖误差贡献堆叠柱状图）。

## 阶段 B：基于阶段 A 证据的自动分流方案

完成阶段 A 后，严格按实测数据选择唯一主攻方向（不并行混改）：

| 诊断结果特征 | 根因判定 | 执行方案 |
| :--- | :--- | :--- |
| **情况 1**：某轴（重点看 A 轴）零位移死区帧占比高，或最小非零实际步进折算针尖位移 `≥ 5 µm` | **执行器/仿真死区构成物理下界** | 暂停模型微调，先核查 `MotorBackend` 步进分辨率、死区与摩擦/量化参数，并用纯专家控制器在同场景跑 5 目标验证 `< 0.35 µm` 物理可达性。 |
| **情况 2**：某一轴残差均值明显偏离 0 且几乎不过零（稳态静差），其余两轴已 `< 5 µm` | **单轴近零区标签/分布偏置** | 仅针对该轴审计清洗后数据集中 `< 50 µm` 区的标签均值与符号分布，补齐对称近零样本后按 8% 低比例回放单变量复测。 |
| **情况 3（最高概率）**：三轴均围绕 0 抖动且 `|policy_cmd| / |teacher_cmd| >> 1`，无明显物理死区阻断 | **全局输出尺度过粗（输出端下界）** | 启动**局部尺度两阶段 Fine DiT**（优先）或**动作 `symlog` 独立重训**，把 `step 680` 之后的输出满量程缩小 1–2 个数量级。 |

### 若命中情况 3：局部尺度两阶段 Fine DiT 落地规范

1. **Coarse 基线选定**：待当前运行中的 `dev20` 终审收尾后，在 `task2_step4000`（`53.950 µm`）与 `step2000` 之间选终值更稳者作为 Coarse 前置模型。
2. **Fine 局部数据集与统计量**：仅抽取清洗后训练集中近目标片段（例如 `error < 300 µm` 的 Plan-2 D1/D2/D3 与原集近目标帧），**单独重新统计局部 `mean / std`**（保持相同 State-Attention DiT 架构与 `valid_window_indices` 契约，不复用全行程大方差统计量）。
3. **接口与归因严格隔离**（针对 [eval_policy.py](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/experiment/experience-replay-20261118/rcm_4090_deploy/lerobot_training/eval_policy.py) 现有约束）：
   - 新增独立的 `Fine DiT` 加载分支（不混用现有 `oct_guided_residual_regression` 路径）；
   - 实现显式**进出滞回状态机**（如进入 `< 200 µm`、退出 `> 400 µm`）；
   - 为**解析编码器终端修正（terminal refinement）增设独立开关**，首轮评估**强制关闭**解析修正，纯测 Fine DiT 网络净增益。
4. **分层评估**：先跑固定 **5 目标（`E00, E02, E06, E08, E10`）× `1,000` 步**快筛；确认 `Min Mean` 压进个位数微米或出现 `encoder_settled > 0` 后，再跑全量 **20 目标 × `1,600` 步**终审。

## 执行检查清单

- [ ] 阶段 A：导出 `step2000` 在 `step 640–800` 与 `step 1000–1600` 的三轴定量指标与图表并推送仓库。
- [ ] 收口当前 `dev20` 终审结果，对比 `task2_step4000` 选定 Coarse 基线。
- [ ] 根据 `axis_floor_summary.json` 明确归入情况 1 / 2 / 3，再启动对应修复分支。
