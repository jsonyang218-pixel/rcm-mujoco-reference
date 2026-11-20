# Gate 2a — OCT 偏差归因与仿真渲染修复

五目标×4稳态姿态共20帧，使用同状态/同目标的三种渲染对照：

|变体|目标误差中位数 µm|OCT误差中位数 µm|OCT最大误差 µm|
|---|---:|---:|---:|
|original|9.514390|9.317193|9.767218|
|single_sample|1.060336|1.063502|1.203021|
|corrected|0.073006|0.076533|0.114764|

`single_sample` 仅将 offsamples 从4改为0；`corrected` 再将球面网格从28切片/16层提高至112切片/64层。RGB检测器、标定和球心拟合公式不变。修复位于仿真帧提供器，未添加经验真值偏移；外部 RGB-D 数组估计路径不变。静态校准/数组回放测试禁止访问动态真值。

这只是离线帧改善。**Gate 2b RGB-D 教师闭环未运行**，因为 [Gate 1](../gate1_20261120_motor_oracle/README.md) 严格成功0/5且协议尚未收口。`rgbd_vs_oracle_dev5.png` 中只有实测 Oracle 数据，并标注 RGB-D 未运行；不能作为闭环改善或理论上限证据。

- [完整报告](GATE2_OCT_BIAS.md)、[60帧分量与数据审计摘要](oct_bias_breakdown.json)
- [离线三种渲染对照](offline_oct_attribution.png)
- [目标来源与覆盖](evidence/training_target_provenance.json)、[逐帧历史 OCT 偏差](evidence/stored_oct_errors.npz)
- [21项检查的证据目录](evidence/verification/)

训练数据目标可追溯1001/1039个 episode：Original801/839，Plan2 200/200。检查了 298,978 个有效观察；缺目标的38个 Original episode明确排除，另跳过无效 OCT。近目标<50µm时，Original/Plan2存储OCT偏差中位数分别为 9.510454/9.502314µm。冻结数据未改写；观察偏差不等同于已证明动作标签损坏。

证据附 E00/E08 step1200 的原始/修复 RGB-D 帧；其余原始帧保留在本地运行目录，完整文件指纹见 `artifact_inventory.json`。探针中一次单位传入错误及一次浮点精确相等预检失败均保留，纠正原因见验证日志；它们没有启动门禁重跑。

源码：`../../rcm/strict_rgbd.py`、`../../scripts/research/motor_perception_gates.py`；卡：`../../configs/experiment_card_motor_perception_gate_20261120_v1.json`。仅为仿真证据，不代表真实 OCT、硬件或临床精度。
