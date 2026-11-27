# clean OCT 最佳权重 dev20 授权确认

2026-11-21 用户在得知 fast5 最佳均值45.84530045175171 µm、原 <45 µm 门禁未通过后，
指示“差不多,继续”。本卡将其记录为本次 dev20 的显式用户放行；原 fast5 结果、
原门禁未通过状态、原 workflow 跳过 dev20 的记录全部保留。

假设：clean OCT/C标签修正后的最佳 step2000 + V0/G10 在完整 development20 上的表现
需要直接确认。仅执行这20个目标，不重新训练或改变标签，也不扩展到 final_test。
固定权重为 `outputs/clean_oct_cbias_20261121_resume_v2/step_002000.pth`，
架构 State-Attention v1，训练数据 `data/plan2_plus_original_train_224_20261119`，
OCT/C覆盖层 `data/clean_oct_cbias_20261121_resume_v2`，验证集保持隔离。

冻结目标清单 `configs/v12_reachable_development_targets_0p35um.json` 的索引0–19，
seeds42–61，每目标最多1600步；V0/G10反馈、I0、Flow4、samples1、execute1、
50ms控制周期与原驱动不变。严格成功判定 ≤0.35 µm 且连续3帧编码器停稳。
fast5的5目标/1000步结果和历史教师fast5均不是本次20目标/1600步配对基线，
缺同协议dev20教师时教师超额误差留空；不自行加跑基线。

`configs/clean_oct_dev20_user_authorization_20261121_v1.json` 固定原fast5摘要SHA256、
权重SHA256、variant、实际45.8453 µm以及用户指示。默认 <45 门禁继续生效，
仅显式传入这个匹配凭据时允许执行本次确认；运行 admission.json 和最终summary记录
`original_gate_passed=false` 与 `user_authorized_exception=true`。

使用物理GPU0 RTX4090的CUDA推理、NVIDIA EGL渲染，4路起步、最多8路，保留资源准入。
先前1000步单目标约360–440秒，按1600步和4路5批估计约50–65分钟，
实际取决于进程竞争和提前停止；不保证结束时间。单目标上限900秒、计算阶段80分钟、
workflow外层85分钟，额外5分钟用于初始化、诊断图和结果保存。
遇到异常或超时记录部分成果，停止，不自动调参或重试。

结果目录 `runs/20261121/clean_oct_dev20_authorized_v1/`，子阶段 `dev20/`；
完整目标覆盖、流程状态、各阈值命中和严格成功率分别报告。结果追加 `TEST_LOG.md`。
启动前检查设备、架构、SHA256、目标/种子/参数、新路径和放行凭据；
在新的GNOME Terminal启动一次，真实日志显示并落盘。启动后不轮询，用户询问时只查一次。
所有结果用于软件和MuJoCo仿真验证。
