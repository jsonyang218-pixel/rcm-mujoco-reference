> 2026-11-21 修订：当前执行依据为 [P2 配对诊断执行卡](rcm_4090_deploy/docs/CONTROL_P2_EXECUTION_PLAN_20261121.md)。以下保留为历史草案；其中21 mm步距、硬件倍率、门禁与输出路径以修订版为准。先执行P2与S1–S3诊断，微调另需完整训练卡。

# 最新闭环执行规划（P2 连续累积 + 精简 S1–S3 快筛 + Phase 3 精调）

基于分支 [`feature/g15-control-single-packet-20261121`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/tree/feature/g15-control-single-packet-20261121)（提交 [`91427cf`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/91427cfdd255578104b1f26b3eaf2cd138909665)）与 [`7f6b07c`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/commit/7f6b07c7fbb6ad3a174d3a391630b69c0b441864) 制定。对原 [`CONTROL_G15_MIGRATION_PLAN_20261121.md`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/feature/g15-control-single-packet-20261121/rcm_4090_deploy/docs/CONTROL_G15_MIGRATION_PLAN_20261121.md) 做了两项关键修正：**① 解除 S2 必炸的 `10/10 ≤ 0.35 µm` 停机门禁；② 用 5 目标快筛替代 460 分钟全量矩阵，把单轮验证压缩到 ~1.5 小时内。**

## 当前基线与精度口径

- **驱动与分辨率事实**：`G15` 已完成默认切换（`RCM_DRIVER_VARIANT=g10` 保留回滚），但眼底 `~21 mm` 深度下单 handle 步距 **`G15`（A `~8.4` / B `~4.6` / C `1.763 µm`）比 `G10`（A `~5.6` / B `~3.1` / C `1.175 µm`）粗 1.5 倍**；真机 level-5 实际硬件倍率为 `7`（A `~3.9` / B `~2.15` / C `0.823 µm`，可按需新增 `g7` 变体）。
- **感知状态**：Gate 2a 已通过 `offsamples=0 + 112/64` 将 OCT 误差中位数压至 **`0.077 µm`**；历史冻结训练集 `< 50 µm` 仍残留旧 `~9.5 µm` 渲染偏置待覆盖。
- **策略基线**：当前最佳模型为 `outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth`（20 目标终值均值 `53.950 µm`，`I0` 臂）。
- **统一评估标尺**：不再以物理不可达的 `0.35 µm` 作为阻断门槛，统一输出 **绝对终值均值/中位数/最大值（µm）**、**`≤ 5 µm` 与 `≤ 10 µm` 命中率**，以及 **相对同挡位真值 Oracle 的超额误差**。

## 总体执行顺序（预计总耗时 ~3.5–4.5 小时）

| 阶段 | 任务内容 | 规模与耗时 | 核心目标 | 放行条件 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1（P2 落地）** | 实现 `rcm/control_target.py` 连续意图目标累积 + 抗饱和 | 单元测试 (~20 min) | 根治 `eval_policy.py` 每拍吞小指令导致的 A 轴 42%–45% 死区 | 单元测试通过 + 累计 `handle` 重发幂等 |
| **Step 2（S1 + S2）** | 3 目标 smoke + 5 目标教师闭环（Oracle & 新渲染 RGB-D） | 5 目标 × 1,600 步 (~30 min) | 锁定 `G10` / `G15` 物理天花板并完成 Gate 2b 验证 | **无执行异常即放行**（不卡 `0.35 µm` 停机） |
| **Step 3（S3 快筛）** | 零训练策略 5 目标快筛（`V0` vs `V3` vs `V2`） | 5 目标 × 1,000 步 (~40 min) | 实测 P2 连续累积收益及 `G10` vs `G15` 策略差距 | 锁定最优控制/驱动变体 `BEST_VARIANT` |
| **Step 4（Phase 3）** | 训练集近目标新渲染 OCT 覆盖 + C 轴静差修复 + 8% 回放微调 | 5 目标 × 1,000 步快筛 (~1.5 h) | 消除 `~9.5 µm` 旧 OCT 偏差与 C 轴 `-20 ~ -48 µm` 静差 | 快筛终值显著优于 Step 3 基线（目标 `< 15 / 10 µm`） |
| **Step 5（终审）** | 胜出候选全量 20 目标正式终审（条件触发） | 20 目标 × 1,600 步 (~1.0 h) | 正式归档 `E00–E19` 终值均值、`≤ 10/5 µm` 命中率与对比图 | 仅当 Step 4 快筛明确胜出时执行 |

### Step 1：落地 P2 连续意图目标累积（`rcm/control_target.py`）

- 严格沿用 [`CONTROL_G15_MIGRATION_PLAN_20261121.md`](https://github.com/jsonyang218-pixel/rcm-mujoco-reference/blob/feature/g15-control-single-packet-20261121/rcm_4090_deploy/docs/CONTROL_G15_MIGRATION_PLAN_20261121.md) 的 P2 设计：
  - 维护浮点 ABC 意图目标 `q_intent`（`deg, deg, mm`），每拍计算 `q_trial = q_intent + scale * clipped_policy_delta`；
  - 抗饱和领先量上限取速度上限 × 周期（A/B `0.5°`、C `0.1 mm`），单包手柄增量默认限制在 **`|Δhandle| ≤ (32, 32, 32)`**；
  - 采用“只读预演 → 一次提交”，拒绝候选不得推进时间或改写状态；重置/换目标/故障时清零意图并重新锚定；
  - 保持 `previous_action` 为本拍接受的 ABC 命令增量，累计整数 `handle` 重发幂等。

### Step 2：S1 Smoke 与 S2 教师物理天花板 + Gate 2b（解除 `0.35 µm` 死锁）

- **S1（Smoke）**：索引 `0/1/2` 各跑 `120` 步，验证 `V0`（G10 无累积）、`V3`（G10 + P2）、`V2`（G15 + P2）执行、限位与复位链路正常。
- **S2（5 目标教师基准 + Gate 2b）**：在固定 5 目标（`E00, E02, E06, E08, E10`，seeds `42, 44, 48, 50, 52`，`1,600` 步）+ `offsamples=0 & 112/64` 新渲染下：
  - 分别跑 **Oracle 真值教师** 与 **Strict RGB-D 教师**（对比 `G10` 与 `G15`）；
  - 确认新渲染下 RGB-D 教师追平同挡位 Oracle 天花板（验证 Gate 2b）；
  - **门禁调整**：只要运行无异常、日志完整，即直接放行 Step 3，不因 `0.35 µm` 格点不可达而中止。
- **产出目录**：`rcm_4090_deploy/github_reports/g15_20261121_s1_s2_teachers/`。

### Step 3：S3 零训练策略 5 目标快筛（`V0` vs `V3` vs `V2`）

- **配置**：不重训，使用 `task2_step4000/step_004000.pth`（固定 `I0` 臂 + 新渲染），在固定 5 目标（`E00, E02, E06, E08, E10`）× `1,000` 步上对比：
  - `V0`：G10 + 原 `actual + action`（对照基线）；
  - `V3`：G10 + P2 连续累积（单独验证 P2 收益）；
  - `V2`：G15 + P2 连续累积（评估 G15 下的实际策略表现）。
- **核心观测**：对比三组的 5 目标终值均值、A 轴非零指令无位移占比（验证 42%–45% 死区是否大幅消除）及 C 轴稳态静差；选出更优的挡位与控制配置进入 Step 4（若后续确认真机最终跑在 level-5，可在此同步补测 `g7 + P2`）。
- **产出目录**：`rcm_4090_deploy/github_reports/g15_20261121_s3_policy_fast5/`。

### Step 4：训练集新渲染 OCT 覆盖 + C 轴静差修复与 8% 低比例微调

1. **新渲染 OCT 覆盖层**：对 `data/plan2_plus_original_train_224_20261119` 的近目标帧（`< 200 µm`）用 `offsamples=0 + 112/64` 重算干净 `oct_navigation_mm`（抽样校验中位误差 `< 0.2 µm`，不改写原文件）。
2. **修复 C 轴 `-20 ~ -48 µm` 静差**：核查 `< 50 µm` 区间 `C-only overlay` 的正向 `dC` 幅值分布，若因零初速替换导致步长过小而提前熄火，在 `< 50 µm` 欠插区改用限加速度前的安全正向期望幅值。
3. **8% 低比例微调与 5 目标快筛**：从 `task2_step4000/step_004000.pth` 热启动（`dagger_repeat=1`，8% 清洗回放 + 92% 原始，`4 × 1,000` batches，保存 `step_002000` 与 `step_004000`），用 Step 3 选出的最优变体跑 5 目标 × `1,000` 步快筛。
- **产出目录**：`rcm_4090_deploy/github_reports/g15_20261121_step4_clean_oct_replay/`。

### Step 5：全量 20 目标终审（条件触发）

- 仅当 Step 4 快筛终值均值明显优于 Step 3 零训练基线时，对胜出 checkpoint 跑满 **20 目标（`E00–E19`）× `1,600` 步**正式终审，输出 `summary.json`、`episodes.csv`、`error_curves_all.png`、`best_and_final_error.png` 及 `≤ 10 µm` / `≤ 5 µm` 达标统计。

## 验收检查清单

- [ ] **Step 1**：`rcm/control_target.py` 单元测试通过，累计 `handle` 重发幂等。
- [ ] **Step 2**：G10 / G15 教师物理天花板落盘，新渲染 Strict RGB-D 教师追平 Oracle（Gate 2b 通过）。
- [ ] **Step 3**：对比 `V0 / V3 / V2`，确认 P2 连续累积对 A 轴死区和终值误差的净改善。
- [ ] **Step 4**：近目标 OCT 覆盖层中位误差 `< 0.2 µm`，C 轴稳态静差收敛，5 目标快筛逼近 `< 15 / 10 µm`。
- [ ] **Step 5**：20 目标全量终审归档，汇报绝对终值与相对 Oracle 的超额误差。
