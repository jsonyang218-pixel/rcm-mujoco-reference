# v9 电机执行采集试验

此试验直接使用当前 `RCMSceneV9`，所有采集运动经 `MotorBackend` 协议、一阶滞后及实际编码器反馈执行；没有将目标位姿直接写入场景的回退。

| 项目 | 实测结果 |
| --- | --- |
| 初始化 A/B/C | 0° / 0° / 0 mm |
| 初始目标距离 | 约 19.75 mm |
| 数据帧 / 电机控制区间 | 683 / 682 |
| 仿真周期 / 仿真时长 | 50 ms / 34.1 s |
| 电机子步 / 一阶时间常数 | 2 ms / 40 ms |
| 最终独立终点误差 | 5.867 μm |
| 停稳 | 连续 3 帧，实际编码器已到达协议量化后的目标 |
| 图像、编码器、动作及电机重放 | PASS |
| 存在电机滞后的区间 | 678 |

- [采集完整性和实际速率审计](dataset_verification.json)
- [终点与执行摘要](pilot_result.json)
- [数据清单](dataset_manifest.json)
- [冻结配置](config_snapshot.json) · [采集协议与源码哈希](collection_protocol.json)

此目录仅保存小型审计记录。全部原始 RGB、JSONL/NPZ、实际电机区间、逐帧哈希、源代码快照及完整控制报告位于云工作区 `runs/v9_motor_dataset_home_20261030`，可单独下载其数据 ZIP；原始图片未纳入 GitHub 项目下载包。

复现采集：

```bash
MUJOCO_GL=egl MESA_SHADER_CACHE_DIR=/tmp/rcm-mesa python collect_v9_dataset.py --output-dir runs/v9_motor_reproduction --episodes 1 --max-iterations 1200
python verify_v9_dataset.py --run-dir runs/v9_motor_reproduction
```

数据输入只有原始 RGB、实际 A/B/C 状态和任务文本；动作是下一控制区间的协议目标。电机滞后使动作与下一实际状态不同，这种区别由重放验证，不能用下一状态冒充动作。细节见 [采集说明](../../../MOTOR_DATASET.md)。

## 运动质量范围

终点成功和采集完整性均通过，但当前实际轨迹未满足全部控制器名义速率门限。按采集状态差分得到的最大速度为 A/B/C **4.140°/s、2.060°/s、0.610 mm/s**，最大加速度约 **14.400°/s²、5.823°/s²、1.749 mm/s²**。这些值和失败标记已保存在验证报告，没有按“指令速率通过”替代实际速率检查。

当前离线电机的独立速度参数为 10°/s、10°/s、2 mm/s，未实现独立加速度限幅，也没有真实电机辨识。此样例用于验证执行和数据采集链；若训练集要求严格满足控制器名义运动门限，应根据报告排除此 episode。这里没有物理制动、接触动力学、真实 OCT 或临床精度结论。
