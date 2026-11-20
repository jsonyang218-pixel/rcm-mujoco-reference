# 五目标纯专家物理可达性验证

Exact-target oracle with original MotorBackend; no learned policy or perception. Physical feasibility only, not a deployable expert or policy comparison.

流程完成：5/5；严格成功：0/5；终值均值：1.671093 µm。
同一 home/session anchor；E00/E02/E06/E08/E10；seed=42+index；50 ms；每目标最多 1600 步；0.35 µm 且连续 3 帧编码器停稳。

|目标|终值 µm|最小值 µm|停稳|严格成功|
|---|---|---|---|---|
|E00|0.884319|0.770849|True|False|
|E02|1.592676|1.592676|True|False|
|E06|3.196515|3.173370|True|False|
|E08|1.171820|1.171820|True|False|
|E10|1.510134|1.487608|True|False|

这是使用真值目标的编码器 oracle 上界诊断，未使用网络或感知反馈，不得与 learned policy 成功率混淆。
无参数更改、无重试、无目标清单改写。物理门禁失败则不启动 Fine DiT 或数据回放训练。仅验证 MuJoCo 软件仿真。
