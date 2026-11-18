# state_attention_kv8

运行目录：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_kv8_20261117_1`

模型权重：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_kv8_20261117_1/training/best_model.pth`

| 指标 | 结果 |
|---|---:|
| 成功目标 | 0/20 |
| 最终误差均值 | 1513.620 μm |
| 最终误差中位数 | 1713.822 μm |
| 最坏最终误差 | 2983.236 μm |
| 退出时编码器停稳 | 0/20 |
| 最后一轮训练 loss | 0.131388 |
| 最后一轮验证 loss | 0.174219 |

最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。
