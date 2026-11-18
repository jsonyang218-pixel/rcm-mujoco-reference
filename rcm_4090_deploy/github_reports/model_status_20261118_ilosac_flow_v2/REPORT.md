# state_attention_ilosac_flow_v2

运行目录：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_ilosac_flow_v2_20261118_1`

模型权重：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_ilosac_flow_v2_20261118_1/training/best_model.pth`

| 指标 | 结果 |
|---|---:|
| 成功目标 | 0/20 |
| 最终误差均值 | 162.215 μm |
| 最终误差中位数 | 148.591 μm |
| 最坏最终误差 | 342.324 μm |
| 退出时编码器停稳 | 0/20 |
| 最后一轮训练 loss | 0.118527 |
| 最后一轮验证 loss | 0.148435 |

最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。
