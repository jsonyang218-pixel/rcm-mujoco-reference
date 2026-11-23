# Clean OCT：诊断与9采样闭环结果

本次上传包含最近的代码、实验卡、训练记录、完整dev20、A/B诊断、单因素3目标闭环、原始轨迹及TEST_LOG。
保留原始文件内容与SHA256；绝对路径记录对应原机器，仓库路径去掉 `/home/him1/mujoco_bc/` 前缀。

## 最近的单因素闭环

流程3/3完成，约11分25秒；Flow4、execute1、G10 feedback、1600步、50ms保持原协议。
唯一候选改动：每步1次采样 → 9次并行采样，沿用逐分量torch.median。
E00/E08/E10，种子42/50/52，step_002000同一权重；完整同身份基线复用。

| 指标 | 基线 | 9采样候选 |
|---|---:|---:|
| 平均终点误差 µm | 127.641903 | 113.314311 |
| 末200步平均误差 µm | 111.339233 | 107.350691 |
| 末段漂移绝对值均值 µm | 74.964477 | 27.775900 |
| 严格成功 | 0/3 | 0/3 |

终点均值下降11.22%，三个平均指标门禁通过。E08终点增加17.010574µm，不能忽略单目标恶化。
这只是3目标smoke筛选；未执行该候选screen10/dev20，不外推总体成功率。真值不用于动作选择。
严格成功维持≤0.35µm及连续3帧编码器停稳。采样数改变随机数消耗，不声称相同噪声流。

## 前置记录与限制

完整单采样dev20均值104.285155µm、严格成功0/20；与本次3目标均值分母不同，不能直接混比。
A阶段20目标40段完成；B阶段9快照288采样完成，原种子动作9/9精确复现。
7/9快照平均A方向与教师参考相反；一阶朝向测量目标109/288。原RGB未保存，不声称像素完全一致。
这些是诊断线索，未证明驱动bug，也未证明增加采样足以消除方向偏差。
早期fast5分支失败和后续dev20用户例外授权均保留；早期中断工作流不伪装为成功。

## 文件入口

- `rcm_4090_deploy/docs/CLEAN_OCT_MEDIAN9_SMOKE_20261122.md`：冻结协议和门禁。
- `rcm_4090_deploy/runs/20261122/diagnosis/clean_oct_terminal_ablation_v1/REPORT.md`：逐目标对照。
- `rcm_4090_deploy/runs/20261122/analysis/clean_oct_drift_interpretation_v1/REPORT.md`：诊断解读。
- `rcm_4090_deploy/runs/20261121/clean_oct_dev20_authorized_v1/dev20/summary.json`：完整20目标基线。
- `rcm_4090_deploy/outputs/clean_oct_cbias_20261121_resume_v2/`：训练配置、日志、历史和清单。
- `rcm_4090_deploy/TEST_LOG.md`：保留远端历史并追加本地新记录；`local_TEST_LOG.md`保存本地原件。
- 本目录 `publication_manifest.json`：全部发布文件（自身除外）的字节数及SHA256。

权重 `step_002000.pth` SHA256：`705dc36de56d3efd659fd9d0c47f33bd91d8a9c3c6e5ea32e5dbcbcec0266523`。
模型文件约415MB，超过GitHub普通单文件限制，沿用之前发布方式记录身份，不上传模型二进制。
原训练/验证数据集不重复上传；本轮clean OCT/C标签覆盖层和审计结果包含在内。
发布准备属于CPU文件I/O；原实验使用物理GPU0 RTX4090 CUDA/EGL。所有结论仅限软件/MuJoCo仿真。
