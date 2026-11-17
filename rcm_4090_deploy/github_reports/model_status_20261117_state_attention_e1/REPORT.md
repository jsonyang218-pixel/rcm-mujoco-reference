# state_attention_v1_continue

运行目录：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_v1_continue_20261117_1`

模型权重：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_v1_continue_20261117_1/training/best_model.pth`

| 指标 | 结果 |
|---|---:|
| 成功目标 | 0/20 |
| 最终误差均值 | 978.815 μm |
| 最终误差中位数 | 637.927 μm |
| 最坏最终误差 | 2548.990 μm |
| 退出时编码器停稳 | 0/20 |
| 最后一轮训练 loss | 0.108947 |
| 最后一轮验证 loss | 0.147718 |

最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。
