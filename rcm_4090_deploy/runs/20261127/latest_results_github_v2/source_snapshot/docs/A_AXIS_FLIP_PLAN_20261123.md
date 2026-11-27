# A 轴反向偏置审计与条件修复执行约定

本轮依据工作区根目录 `规划4.txt`，基点 `dad76c522f7e74f3a86b56d41f6c89488ae878ff`，
使用独立分支 `fix/a-axis-flip-20261123`。原数据、旧 task2 权重、新 cleanOCT 权重和驱动副本均保留。

## 固定条件

- 阶段、训练、episode、worker 初始化均无 wall-clock 超时；出现异常、NaN/Inf、非法状态即记录并停止，不自动重试。
- 全部快筛为 E00/E02/E06/E08/E10，种子 42/44/48/50/52，1600 次真实控制动作，50 ms。
- NVIDIA GPU 0，RTX 4090；CUDA 推理/训练，NVIDIA EGL 渲染；质量 0/112/64。
- 0.35 µm 与连续 3 帧编码器停稳仅用于严格成功记录，不使本轮评估停机。合法限幅也不因连续次数终止；全部动作尺度被拒或超出既有 30 mm 安全误差包络属于无效评估。
- 全量标签审计覆盖源数据所有原始有效 OCT<200 µm 帧。几何教师必须使用真实重渲染的有效观测，不伪造可见性或拟合质量。
- 原始 RGB 未随历史漂移轨迹保存；9 快照采用已验收的固定重建输入，双方完全相同，不声称与原始 RGB 逐像素相等。

## 审计分流与单变量

original/plan2 的数据来源分界为元数据中的 episode 839，非旧训练参数的 760。
逐轴反向比例以全量 near 桶中零初速与限加速度前几何期望均非零的合格帧为分母，同时记录所有近目标帧分母、教师无效数、历史速度可用数和惯性重放兼容数。
A/B 任一反向比例 ≥20% 命中 A。B 的“显著更少”冻结为旧 A≤3/9、新 A≥7/9，且 RGB/OCT 来源不同或边界 OCT 跳变中位数增加。两者命中时 B 优先；均未命中就跳过依赖阶段。

A 只替换历史有限差分速度重放能够重现惯性拖反、且零初速安全的轴，独立 overlay，保持其余轴与实际执行动作历史。回放窗口必须经过 `valid_window_indices` 校验且包含被修正标签。
B 固定 50 条训练来源目标，排除 development 目标；取源轨迹第一个 150–200 µm 有效状态，缺失时取第一个 <200 µm 有效状态，旧模型固定闭环 1600 步。
仅保留同一 `sensor.observe()` 中近目标有效 RGB/OCT/零初速 ABC 标签，实际执行动作另存历史；保留源帧与时间间隙，不拼接伪连续窗口。

训练从旧 task2 step4000 热启动，92% 为未修改合并训练集的全部有效窗口，8% 为独立纠偏池，dagger_repeat=1。
4×1000 batches，batch16，每轮精确 14720 原始样本和 1280 纠偏样本；保留原归一化、学习率和 CUDA FP16 AMP。
完整冻结验证集，不截断 validation batches。每100 batches 原子保存含 optimizer/scaler/RNG 的续跑状态，已完成 epoch 不重训。

## 评估门禁与记录

终值取第1600次动作后误差，最小值包含最终状态；末段漂移取终值减 trace[1000] 的动作前误差，门禁使用带符号均值，并同时报告绝对漂移均值。
命中数为最终误差 ≤5/10/30 µm。严格成功单独报告，不以曾短暂命中替代终点连续停稳。

先比较 step2000/4000 的 V0，同条件终值均值较低者复检9快照。
方向改善门禁冻结为 A反向≤1/9、toward>75%、末段带符号漂移<15 µm，全部通过才跑 I_med5 和 V3。
最优完整 fast5 终值均值<35 µm 且末段漂移<15 µm 才启动 dev20，20目标×1600步，每5个完成项写快照。

每集保存 `E*.json`，调度器随即写 `partial_summary.json`、`reuse_receipt.json` 和文件哈希凭据。
重启复用必须匹配数据、checkpoint、代码、场景、目标、种子与全部证据哈希；未完成 shard 归档后重做，异常不自动调参或重复实验。

报告使用任务书指定 `github_reports/audit_20261123_a_axis_flip/` 与 phase2/3/4 目录；
新数据使用 `data/a_axis_fix_20261123_<A或B>_v1/`，新权重使用 `outputs/a_axis_fix_20261123_<A或B>_v1/`。
流程日志、预检和卡放入 `runs/20261123/a_axis_flip_workflow_v4/`，验证放入 `runs/20261123/verification/a_axis_flip_v1/` 及后续验证版本。
所有可报告结果追加 `TEST_LOG.md`。流程结束发布代码、结构化审计、评估、日志和根 `SUMMARY.md`；大权重和原始纠偏 RGB 保留本机并登记 SHA256，不混入 Git 普通大文件。

```bash
env -u PYTHONHOME -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID=0 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -E -B -m scripts.research.a_axis_flip_pipeline launch \
  --output-dir runs/20261123/a_axis_flip_workflow_v4
```

只在新 GNOME Terminal 启动一次，日志同时显示并保存。终端启动失败不后台回退；助手在启动后不等待或轮询，用户再次询问时读一次状态。
外部中断后由相同环境运行 `run --output-dir ...` 续跑；运行中不得重复启动。

结果仅验证软件/MuJoCo 仿真，不代表临床精度或硬件就绪。
