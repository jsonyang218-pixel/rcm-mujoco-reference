# G15 单包最小步长测试

2026-11-21；流程完成，7/7 工况通过。本测试测量模型默认零位的单包指令步长，不是实机定位精度或任意 XYZ 方向的分辨率。

从机器人编码器 `ABCE=[0,1466018,893518,0]`、legacy 针具、针尖超过 RCM 约 4 mm 开始。每个工况重新初始化，仅在 t=0 接受一次累计绝对手柄目标，然后只推进电机及 MuJoCo 反馈，不重发、不累积、不强制写入目标位姿。测试 A/B/C 各 ±1 和零指令对照；0.1 手柄计数另做拒绝检查。

| +1 手柄计数 | 编码器变化 | 实际轴运动量 | 50 ms 针尖位移 | 停稳针尖位移 |
|---|---:|---:|---:|---:|
| A | +12 | 0.0216° | 1.098496 µm | 1.464662 µm |
| B | +965 | 0.011246240° | 0.560588 µm | 0.785136 µm |
| C | +112 | 1.754861 µm | 1.253472 µm | 1.754861 µm |

默认零位三个单轴中，测试到的最小非零稳定位移为 B 轴约 0.785136 µm。A/C 与名义步长略有差别是整数编码器取整造成的；改变会话起点、当前姿态或针尖距 RCM 后，应重新换算。50 ms 的较小位移是尚未完成的跟踪过程，不能作为独立可控的最小步长。

保留当前动力学：τ=40 ms，速度 A/B=10°/s、C=2 mm/s，积分步长 2 ms，watchdog=0.5 s。观察周期 50 ms；在量化目标处连续三帧编码器不变的确认时间分别为 A=0.3 s、B=0.5 s、C=0.4 s。零指令位移为 0；正反向均完成；不接受分数手柄；没有关闭超时或用重复包续期。

G15 的源文件 SHA-256：`3465daf19fd3269466f90fd77c6626149ab4da8ddde612c8f3d9a12c77d06088`。本次使用 CPU 完成小规模标量运动学和 MuJoCo 前向计算，无渲染、训练、网络推理或硬件通信。未执行导航 screen10/dev20，未评估 0.35 µm 导航严格成功率，也不据此声明比 G10 更精确。

- [summary.json](summary.json)：汇总和逐工况结果。
- [preflight.json](preflight.json)：参数、门禁、预算及输入指纹。
- [run.log](run.log)、[command_receipt.json](command_receipt.json)：实际运行输出与命令。
- `A/B/C_plus1/minus1.json`、`hold_zero.json`：各工况明细；对应 `*_trace.jsonl` 保存全部 250 个积分步。
- [measure.py](measure.py)：实际执行的测试脚本。
- [G15 切换验收](../g15_switch_v2/acceptance.json)：G15 项目回归 22 项、G10 回退 7 项、协议/几何/入口检查通过。此前旧协议测试的 20 ms 到位假设失败保存在 `../g15_switch_v1/`，修正测试假设后的检查见 `../g15_switch_v2/`。

在 `rcm_4090_deploy/` 使用部署虚拟环境复现；必须选择一个尚未使用的新输出目录，例如：

```bash
MUJOCO_GL=egl MPLBACKEND=Agg RCM_DRIVER_VARIANT=g15 .venv/bin/python -E -B \
  runs/20261121/verification/g15_single_packet_v1/measure.py \
  --output-dir runs/20261121/verification/g15_single_packet_repeat_v1
```

完整评估和训练仍遵循项目实验 SOP；本测试不改变旧 Gate1 的失败结论。策略连续目标累积尚未实施。
