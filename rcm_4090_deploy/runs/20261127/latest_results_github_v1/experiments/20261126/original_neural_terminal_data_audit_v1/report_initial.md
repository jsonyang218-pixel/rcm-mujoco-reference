# 原神经末端最小实验：现有数据与训练链路离线核查

**为什么没有形成“同误差、不同速度”的独立检验：** 本轮未构造成对、按输入运动方向分层验收的同误差分支，且把“本拍执行后距离减小”当作 approaching，混淆了输入时速度与动作结果。
实际100个首动作窗含 4 个输入时远离、96 个输入时靠近；不是完全没有远离输入。初始远离第0步的真实历史已齐全、也被选中，故“history尚未有效，远离已经结束”的假设被否定。

**时序/标签是否存在已证实缺陷：** 未发现保存的 t、t−1、label t、horizon16 起点或0.05s受理执行间隔错位。已证实的是分类语义缺陷和覆盖不足：等距离也被原 else 分支叫作 receding；停稳而受理C非零不满足原 strict hold。
每段末15个状态因未来不足16步不能作为首动作窗，但它们仍在较早窗口的未来监督中。不能把“未作为首窗”写成“全部未训练”。

**当前唯一最值得做的下一步（仅建议）：** 先补齐并验收连续视觉教师数据中的同误差异速配对及可作首窗的保持段，再考虑任何训练。必须从真实前缀开始记录history，使history有效后仍保留实际远离/制动；每个目标锚点后保留至少16步真实连续监督。保持原速度定义和homogeneous目标登记，不用改定义/改ID解决。完整最小规格见 `recommended_collection_spec_not_executed.json`；本次未执行。

## 范围与证据

CPU确定性文件复算；4 条原始teacher episode、160 个teacher状态、100 个原保存窗口。模型推理0次、策略重采样0次、梯度更新0次、新仿真0周期。
逐状态证据：`raw_state_evidence.csv/json`；前缀：`prefix_history_evidence.json`；逐窗：`window_evidence.csv/json`；所有状态丢弃原因：`all_candidate_states_and_exclusions.csv`。

## 全40步分类、连续区间与真实方向

原类定义保持不动：先判 abs(label_C)<=1e−9mm 且 abs(pre_actual_vC)<=1e−6mm/s 为hold；再判前/后signed OCT Z反号为crossing；再判 post_error<pre_error 为approaching；其余均receding。
因此它描述的是执行区间结果，不是输入时刻的径向运动；最后一个分支还包含 post==pre。物理说明另外列 incoming_phase 与 outgoing_distance_phase，未改原标签、gate或速度。
incoming速度仍取记录的真实编码器上一执行区间有限差分。方向用相邻同源signed OCT向量模长变化说明；teacher step>=1另有独立真值距离差核对。step0缺前缀独立site XYZ真值，故其incoming truth径向值留空，使用已有signedOCT/编码器方向为证。

|episode|原分类（完整40步）|输入运动（完整40步）|原分类连续区间（含端点）|
|---|---|---|---|
|E40_below_recede|{"approaching": 25, "receding": 15}|{"receding": 1, "approaching": 25, "stationary": 14}|approaching 0–24；receding 25–39|
|E40_above_recede|{"approaching": 27, "receding": 13}|{"receding": 1, "approaching": 26, "ambiguous": 1, "stationary": 12}|approaching 0–26；receding 27–39|
|E41_below_recede|{"approaching": 25, "receding": 15}|{"receding": 1, "approaching": 25, "stationary": 14}|approaching 0–24；receding 25–39|
|E41_above_recede|{"approaching": 27, "receding": 13}|{"receding": 1, "approaching": 27, "stationary": 12}|approaching 0–26；receding 27–39|

原始 strict hold 的确没有采到：tail虽误差稳定/电机不动，但受理C仍非零；不能说 strict hold 只是被过滤。原始真远离的执行后区间也不能由片段名证明，须查看距离变化。
片段名 below_recede/above_recede 指固定前缀制造的入场状态，不保证随后教师40步都在远离。实际teacher第一拍已经使距离减小，但输入仍包含前一拍朝外速度。

|episode|已保留的异速近邻原始步|真值距离差µm|signed OCT向量差µm|实际C速度µm/s|教师受理Cµm|
|---|---|---:|---:|---|---|
|E40_below_recede|[0, 1]|1.034110089|1.033380783|[-43.87152764136204, 20.682291602333592]|[1.5564236179321789, 4.7841145801169604]|
|E40_above_recede|[0, 1]|1.034110088|1.039362186|[43.87152764136204, -20.682291602333592]|[-1.5564236179321789, -4.7841145801169604]|
|E41_below_recede|[0, 1]|1.034093218|1.014991430|[-43.87152764122326, 20.6822916022642]|[1.5564236179391178, 4.784114580113491]|
|E41_above_recede|[0, 1]|1.034093223|1.043498496|[43.87152764129265, -20.682291602333592]|[-1.5564236179356483, -4.7841145801169604]|

这些是少量描述性近邻，并非相同RGB/OCT/ABC的独立速度干预配对；不据此声称已完成同误差异速检验。

## 40步到25窗的完整过程

- observations 共42帧：索引0/1是实际前缀motor24/25，索引2对应teacher step0（motor26）。两个prefix是history，不是teacher标签。
- teacher候选step s=0..39；原代码恰好 `range(25)`，与40−16+1相符。窗口历史帧为[s+1,s+2]，输入截止为step s；label为teacher[s:s+16]。
- step0有前缀index1可用，实际全100窗均无history缺失。不存在额外warmup丢step0。
- step0..24有16步真实未来；step25..39分别只剩15..1步，未padding。每段15个、共60个只作为首窗被排除；无有效首窗遗漏的证据。
- 已选窗的两帧RGB均以CPU按原PIL/ToTensor/Normalize路径独立重处理，与全部保存图像张量逐元素核对；原OCT有效位、索引、动作16步连续性同步核对。
- 没有按类别筛选：原代码先收所有25窗，分类只写元数据且不进入loss或采样权重。尾部与输入运动分层覆盖不足是实际数据/固定horizon边界问题，不能称为class筛选bug。

## 时序、标签、受理与实际位移

共 112 项原轨迹核对、600 项逐窗张量核对，容差外错位/张量错配均0。逐项单位、容差、最大差与逐元素差异见 `raw_alignment_checks.json`、`saved_input_checks.json`。
原NPZ/traj受理标签最大差 0 deg/deg/mm；历史上一指令最大差 0；motor请求回推指令最大差 3.46944695e-15 deg/deg/mm。
pre时间相对motor.start最大差 0s；执行间隔相对0.05s最大差 1.80411242e-16s；post评分对下一pre评分最大差 0µm。
label t 是当拍受理的增量请求；不是“执行后的编码器位移”。Motor G10量化/执行动力学可使两者不同，尤其尾部受理非零但真实编码器零运动；这是原控制接口语义，不可偷偷把label换成实际位移。
ABC命令单位为度/度/mm；送Motor时乘[π/180,π/180,0.001]。motor输出actual velocity是(actual_after−actual_before)/duration；输入t使用的是上一执行区间实际速度，上一受理指令来自t−1。
原网络实际速度输入来自float32 ABC相减/0.05，与原生产代码一致；相对float64真实速度最大差A/B/C=[0.0, 0.0, 3.6783631811587725e-05]（deg/deg/mm每秒）。这是已有float32量化差，不冒称为零差或改变实际速度定义。
归一化统计before/after相同；label和物理预测按(mean,std)正/反变换复算，数值误差按明确1e−15容差记录。没有保存采样latent，故不宣称重新验证采样内部的每个反归一化数值。
模型观测只含真实RGB2/ABC/signedOCT/有限差分速度/上一受理指令；真值距离仅进评分、离线分组元数据，不是模型输入。训练不会读取未来图像或未来状态。

## 训练链路：已证实、设计选择与未知原因

- 已证实分类语义错误：输入远离与本拍靠近被合并，等距离停稳尾部被称receding。报告保留原组再另列物理说明，没有重写旧指标。
- 已证实链路一致：同一个原架构条件构建与归一化；原入口step0/短门禁bit-exact和after checkpoint加载复用已有证据；本次没有模型实例化、前向或策略采样。
- 原配方设计选择：loss为全部16步×3轴的normalized flow velocity MSE，首步名义权重1/16；部署是同一模型4步flow/K1只执行首步。训练连续随机tau/noise与4步数值采样不是相同计算，但符合原rectified-flow配方。
- 条件中的速度/上一受理指令是当前向量，模型对两帧广播；训练与生产一致。图像编码器冻结/eval、GroupNorm20无BN漂移；其62个张量before/after再次逐位核对。
- CPU种子重放训练抽样，与129次记录全部一致；128次实际更新的512个首步样本槽中，输入远离仅19槽。见 `training_sampling_evidence.csv`。
- 训练FP16/autocast、评估FP32是原配方；覆盖不均、horizon目标与首动作验收差异均是待验证原因，不足以从loss下降与MAE变差的相关性推断因果。

|既有指标复算|C首动作MAE µm before→after|flow loss before→after|
|---|---:|---:|
|E40训练|1.4598114 → 1.8401981|0.014810521 → 0.010742446|
|E41留出|1.4682645 → 1.9467759|0.015690573 → 0.012941935|

原gate未修改，128更新负结果不外推为架构不可能。

## 证据局限

- Intermediate independent MuJoCo needle-site XYZ were not saved. Audit recomputes all stored pre/post distance differences and final XYZ norm, but cannot independently reconstruct each intermediate3D norm without additional kinematics/simulation; none was run.
- No terminal post-action RGB/OCT after step39; final crossing cannot be directly assessed from measured OCT. Original fallback copies last OCT; explicitly flagged.
- Simulation acquisition timestamps are recorded; true camera latency/hardware clock synchronization and biological/physical robot dynamics are not identified.
- RGB shape/dtype and all selected history tensors were independently reprocessed on CPU; this does not independently establish needle visibility or recalibrate OCT3D geometry.
- Saved predictions are physical units; latent normalized sampled trajectories are absent. Normalization inverse checked algebraically and static code checked; sampler not rerun.
- Original raw strict hold group is zero because accepted C remains nonzero even when encoders settle. Physical settled low-error tail exists, but cannot be relabeled as a demonstrated zero-command hold under the original definition.
- Near-error opposite-velocity neighbors exist but are few and not independent matched same-observation branches; no exact equal-signed-error velocity intervention was tested.
- Flow-loss improvement versus first-action degradation is a measured association. Horizon16 objective/exec1 evaluation, coverage imbalance and FP16/FP32 difference are candidate explanations, not causal proof.

## 文件与执行验收

输入文件 107 个，恢复前/后SHA-256全部一致；含原脚本、原始数据、before/after checkpoint、旧metrics及恢复报告。唯一新产物目录为本目录；部署根TEST_LOG按授权追加。
每项检查完成后立即写本目录及部署根TEST_LOG并fsync；最终 `test_log_write_verification.json` 会读回核对每项记录确实存在。
确定性CPU核查耗时 4.207s（小于60s，不含阅读编写）；命令见 `command.json`，脚本见 `audit.py`。
本次只建议下一步，未采集、未训练、未推理、未仿真、未修改生产代码或原产物、未上传。
