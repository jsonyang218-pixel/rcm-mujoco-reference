# Motor / perception recovery gate — 2026-11-20

真值几何教师完成 **5/5**，严格成功 **0/5**。流程完成与门禁通过分开报告；本轮 workflow 因严格门禁失败而标为 `failed`，没有训练、改目标、放宽门槛或自动重试。

- [Gate 1 电机审计](GATE1_MOTOR_AUDIT.md)
- [五目标机器可读摘要](oracle_dev5_summary.json)
- [E00/E08 三轴轨迹](oracle_axes_E00_E08.png)
- [Gate 2 离线 OCT 归因与渲染修复](../gate2_20261120_oct_bias/README.md)
- [完整 Oracle 轨迹与电机指令](evidence/oracle_trajectories.tar.gz)
- [工作流终态](evidence/workflow_status.json)、[实际实验卡](evidence/experiment_card.json)

协议：E00/E02/E06/E08/E10，seeds42/44/48/50/52，1600步，50ms，≤0.35µm且连续3帧编码器停稳。原 MotorBackend；本轮为真值反馈几何教师，保留 command_q/速度累积，不走 RGB-D/OCT。

旧的最近协议端点 Oracle 仅作辅助参照（见 evidence/expert_summary.json），不是本轮几何教师。本轮几何教师存在持续运动；瞬态反馈状态可能优于已搜索的静态端点，不能据此推翻端点参照，也不能把局部搜索当作全局物理下限证明。

Gate 1b 缺真实硬件 handle 步距标定，未选择硬件条件分支；Gate 2b、Phase 3 和神经网络训练保持暂停。只验证软件/MuJoCo仿真。
