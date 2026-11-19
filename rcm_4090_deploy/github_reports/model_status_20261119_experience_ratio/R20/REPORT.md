# state_attention_experience_ratio_r20_20261118

运行目录：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_experience_ratio_20261118_1/r20`

模型权重：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/state_attention_experience_ratio_20261118_1/r20/training/step_004000.pth`

| 指标 | 结果 |
|---|---:|
| 成功目标 | 0/20 |
| 最终误差均值 | 589.481 μm |
| 最终误差中位数 | 227.022 μm |
| 最坏最终误差 | 2202.496 μm |
| 退出时编码器停稳 | 0/20 |
| 最后一轮训练 loss | 0.116947 |
| 最后一轮验证 loss | 0.148117 |

最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。
