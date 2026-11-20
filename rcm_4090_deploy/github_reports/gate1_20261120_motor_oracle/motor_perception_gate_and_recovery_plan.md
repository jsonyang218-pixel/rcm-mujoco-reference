# 执行链与感知硬地板排查及后续策略恢复规划

基于 [`33d4fd7`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/33d4fd7ec71dd608981ef768388295b544cf6d7c)（[AXIS_FLOOR_DIAGNOSTICS.md](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/33d4fd7ec71dd608981ef768388295b544cf6d7c/rcm_4090_deploy/github_reports/diag_20261120_step680_axes/AXIS_FLOOR_DIAGNOSTICS.md)）的诊断结论制定。在当前协议与感知条件下，已发现两道独立于神经网络的硬地板（整数 `handle` 协议可达下限 `0.88–3.20 µm` + A 轴单 handle `5.45–5.91 µm` / 42%–45% 死区，以及稳态 OCT 约 `9.10–9.72 µm` 系统偏差）。**在两道底层门禁收口前，暂停一切神经网络微调与 Fine DiT 训练。**

## 总体路线（先修地基，后动策略）

| 阶段 | 任务名称 | 预计耗时 | 评估范围 | 核心通过门禁 |
| :--- | :--- | :---: | :--- | :--- |
| **Gate 1** | `MotorBackend` 执行链与整数 `handle` 可达性实测 | ~20 min | 5 目标 Oracle 闭环 | 明确物理下限；若为协议 bug 则修复至 Oracle `5/5 ≤ 0.35 µm` |
| **Gate 2** | 稳态 `~9.1–9.7 µm` OCT 感知偏差溯源与消除 | ~30 min | 5 目标 RGB-D 教师闭环 | 拆清目标估计 `9.2–9.6 µm` 偏差来源；确认 RGB-D 教师闭环上限 |
| **Phase 3** | 策略端复测与 C 轴静差（`-20 ~ -48 µm`）攻关 | 按需启动 | 5 目标快筛 → 20 目标终审 | 仅在 Gate 1 & Gate 2 收口后启动 |

## Gate 1：`MotorBackend` 执行链与整数 `handle` 可达性验证（立即执行）

### 1a. 5 目标 Oracle 真值闭环基准
- **目标集合**：固定 `E00, E02, E06, E08, E10`（seeds `42, 44, 48, 50, 52`，`1,600` 步，`0.35 µm` + 连续 3 帧停稳门槛）。
- **控制方式**：在原封不动的 `MotorBackend` 上，直接运行**真值几何教师控制器（Oracle，不走 RGB-D/OCT 估计）**，彻底剥离感知误差与神经网络误差。
- **必记指标**：
  - 5 目标各自终值误差、最低误差、`encoder_settled` 命中数；
  - 对比 `33d4fd7` 逆解邻域 `±2 handle` 下限（`E00: 0.884`、`E02: 1.593`、`E06: 3.197`、`E08: 1.172`、`E10: 1.510 µm`）；
  - A/B/C 三轴在 `step 700–1600` 的非零指令无位移占比与最小实际步进。

### 1b. 协议审计与分流处置
结合 [`motor_protocol_audit/`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/tree/33d4fd7ec71dd608981ef768388295b544cf6d7c/rcm_4090_deploy/github_reports/diag_20261120_step680_axes/motor_protocol_audit) 核查 `MotorBackend` 的 handle 取整、小数指令是否被每步直接截断丢弃（导致 A 轴 42%–45% 死区），以及真实硬件步距定义：
- **分支 1A（软件/仿真协议 bug）**：例如每步把 `< 1 handle` 的增量直接归零丢弃而未做残差累积，或仿真接口误用了过粗的 handle 步距 → 修复协议后重跑 1a，要求 **Oracle 5/5 达到 `≤ 0.35 µm` 且连续 3 帧停稳**。
- **分支 1B（真实硬件物理步距约束）**：若真实电机 A 轴单 handle 物理分辨率确为 `~5.5–5.9 µm` → 不强迫模型对抗物理定律，执行以下二选一并写入冻结协议文档：
  1. 重新构造**同时满足整数 `encoder` 与整数 `handle` 均可达 `≤ 0.35 µm`** 的新目标清单；
  2. 或按真实 handle 格点分辨率同步标定可实现的工程成功门槛。
- **产出目录**：`rcm_4090_deploy/github_reports/gate1_20261120_motor_oracle/`（含 `GATE1_MOTOR_AUDIT.md`、`oracle_dev5_summary.json`、`oracle_axes_E00_E08.png`）。

## Gate 2：稳态 `~9.1–9.7 µm` OCT 感知系统偏差溯源

此前 [`26981e4`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/26981e4312503546b7e45915f15ccb1b8143afda) 已测出针尖估计误差仅 **`0.58–0.81 µm`**，而**目标估计误差高达 `9.22–9.55 µm`**；本次 [`33d4fd7`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/33d4fd7ec71dd608981ef768388295b544cf6d7c) 再次确认稳态 OCT 向量与真值偏差中位数为 **`9.10–9.72 µm`**。这说明主要感知偏置高度集中在**目标估计端**而非针尖端。

### 2a. 目标估计 `~9.5 µm` 偏差分解
在 `E00, E02, E06, E08, E10` 的稳态帧上，对 `target_world_mm_est - target_world_mm_true` 与 `oct_nav - true_nav` 做三项拆解：
1. **分量分解**：拆成世界系 `(dx, dy, dz)` 及相机系 `(横向像素 u/v 误差, 光轴深度 z 误差)`，明确 `~9.5 µm` 是恒定系统偏置（固定常向量）还是随机噪声；
2. **模块归因**：分别检查 [strict_rgbd.py](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/experiment/state-attention-v1-20261117/rcm_4090_deploy/rcm/strict_rgbd.py) 中 `RGBFeatureDetector` 目标质心掩码、`SyntheticOCTProvider` 表面深度采样（是否打在目标表面而非目标中心）、以及相机内参/外参投影；
3. **验证是否影响训练标签**：确认历史训练集和 Plan-2 中存储的 `oct_navigation_mm` 是否同样带有该 `~9.5 µm` 偏置。

### 2b. Strict RGB-D 教师 5 目标闭环基准
- 在通过 Gate 1 的执行链上，运行 **Strict RGB-D 教师** 5 目标闭环，直接对比 **Oracle 教师 vs RGB-D 教师**：
  - 若 `~9.5 µm` 为可修复的深度/掩码中心偏移，修复后重跑 2b，验证稳态 OCT 偏差压降至 `< 0.5 µm` 且 RGB-D 教师逼近 Oracle 上限；
  - 若为传感器固有分辨率极限，则将 RGB-D 教师实测终值正式锁定为**纯视觉/OCT 策略的理论上限**。
- **产出目录**：`rcm_4090_deploy/github_reports/gate2_20261120_oct_bias/`（含 `GATE2_OCT_BIAS.md`、`oct_bias_breakdown.json`、`rgbd_vs_oracle_dev5.png`）。

## Phase 3：地基收口后的策略恢复与 C 轴静差攻关

仅当 **Gate 1 与 Gate 2 均有明确书面结论与基线数据**后，再进入策略端：

1. **零训练基线重测**：直接拿当前最佳 Coarse 模型 `outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth`（`53.950 µm`）在修好后的执行链/感知链上跑一次 5 目标闭环，剥离底层修复带来的免费增益，看清策略本身的真实残余误差。
2. **定位 C 轴 `-20 ~ -48 µm` 稳态静差成因**：
   - 在 `33d4fd7` 中，策略在 C 轴的中位指令幅值仅为零初速教师的 **`0.28–0.30` 倍**，且在 E00/E02/E08 稳态 600 步完全不过零。
   - 核查清洗后数据集中 `< 50 µm` 区间 `C-only overlay` 替换后的 `dC` 幅值分布，确认是否存在“近目标正向标签过小/零动作占比过高”导致策略提前熄火。
3. **单变量启动下阶段模型方案**：
   - 若地基已可达 `< 0.35 µm` 且策略在 `< 50 µm` 仍受限于全局尺度，则按严格隔离规范启动 **局部尺度两阶段 Fine DiT**（独立加载入口、进出滞回切换、**首轮关闭解析终端修正**纯测网络净增益）；
   - 全程继续执行 **5 目标 × 1,000 步快筛 → 胜出后 20 目标 × 1,600 步终审**。

## 执行检查清单

- [ ] **Gate 1a**：跑完 5 目标 Oracle 教师闭环，确认现有 `MotorBackend` 物理下限。
- [ ] **Gate 1b**：定位并决策整数 `handle` 协议（修复 bug 或重标可达目标/门槛）。
- [ ] **Gate 2a**：拆解 `~9.1–9.7 µm` OCT 稳态偏差（重点查目标估计 `9.22–9.55 µm` 来源）。
- [ ] **Gate 2b**：跑完 5 目标 Strict RGB-D 教师闭环，锁定感知端理论上限。
- [ ] **Phase 3**：Gate 1/2 收口后零训练重测 `task2_step4000`，再决定是否重启策略训练。
