# dit_rectified_flow_plan2_plus_original200

运行目录：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/evaluation_dit_rectified_flow_plan2_plus_original200_20261119_dev20_0p35um_shards12`

模型权重：`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/dit_rectified_flow_plan2_plus_original200_20261119_1/best_model.pth`

| 指标 | 结果 |
|---|---:|
| 成功目标 | 0/20 |
| 最终误差均值 | 1151.149 μm |
| 最终误差中位数 | 1392.104 μm |
| 最坏最终误差 | 1952.920 μm |
| 退出时编码器停稳 | 0/20 |

最后一轮 loss 不一定对应被选中的 best checkpoint；权重的 epoch 与 loss 见 summary.json 的 evaluation_contract。轨迹最小误差沿用动作前记录口径。
