# 眼内 RCM 机器人模仿学习：从视觉伺服原型到可发表的顶会路线

检索日期：2026-11-04

## 摘要

当前仓库已经形成一条较完整的研究链：RCM 眼内注射 MuJoCo 场景、严格 RGB-D/理想注册 OCT 观测、A/B/C 三轴视觉闭环、电机一阶滞后、随机眼底目标、成功与近目标恢复示范，以及 RGB + state + OCT navigation 的条件扩散策略。仓库目前最值得利用的不是“再做一个更大的端到端模型”，而是尚未被充分解决的闭环问题：教师在近目标处减速、回撤和停稳，但策略输入没有实际速度、上一动作或明确的停止状态，训练分布又被远距离正向进针帧主导，因此离线动作 MAE 较低并没有转化为闭环成功。

本次检索找到了直接相关的 IROS/ICRA/RSS 论文和两篇 2025 年 Science Robotics 工作。最直接的顶会工作是 IROS 2024 的眼科机器人 apprentice，它用强化学习与模仿学习、专家在环示范和课程学习处理白内障手术阶段；最接近任务与安全约束的是 ICRA 2023 的视网膜针导航、ICRA 2024 的带约束腹腔镜模仿学习，以及 2025 年 Science Robotics 的自主视网膜静脉插管。通用方法中，ACT、Diffusion Policy、3D Diffusion Policy 和 π0 分别支持动作分块、多模态动作生成、三维条件化和 flow matching，但没有证据表明它们可以自动解决本项目的速度不可观测、近目标制动和 OCT 配准问题。

建议路线是：先重建并合并完整的恢复数据，再把策略从“图像 + state + OCT 的单步回归”改为“历史速度/动作 + OCT 目标误差 + 动作分块 + 停止头”的闭环策略；随后用自动 DAgger/IntervenGen 式补数据覆盖策略自己的失败状态；最后在模仿策略外保留 RCM 投影、速度限制和不确定时回退到几何控制器。论文级创新更适合定位为安全约束下的 OCT-aware hierarchical/residual imitation learning，而不是单次 diffusion loss 的改进。

## 1. 研究问题

本报告回答三个问题。

**RQ1：哪些顶会或顶刊工作与当前的眼科 RCM 机器人、RGB-D/OCT 视觉伺服和模仿学习最相关？**

**RQ2：当前仓库的训练与闭环证据暴露了什么瓶颈，哪些表面指标不能继续作为优化目标？**

**RQ3：下一步应如何设计数据、策略、安全机制和评测，才能从“能收集轨迹”推进到可信的模仿学习结果，并形成 ICRA/IROS/CoRL 级别的方法贡献？**

## 2. 方法与范围

先审计仓库中的 `README.md`、`STRICT_RGBD_NAVIGATION.md`、`OCT_SIMULATION.md`、`MOTOR_DATASET.md`、`RANDOM_FUNDUS_DATASET.md`、训练脚本、数据元数据和最近评估结果；再以 Crossref、OpenAlex、arXiv 和 Europe PMC 对标题、作者、会议/期刊和摘要进行交叉核验。检索词覆盖四组视角：

- 眼科、视网膜、眼内注射、RCM 与自主针导航；
- OCT、RGB-D、几何视觉伺服与手术机器人安全控制；
- 显微/眼科机器人中的 imitation learning、offline RL 和 surgeon-in-the-loop；
- 通用 robot imitation learning、action chunking、diffusion policy、flow matching、DAgger 与 failure recovery。

纳入标准是：与当前任务或方法至少有一个直接重合，并且有可核验的论文记录。顶会/顶刊论文与 arXiv 预印本分开标注。报告只使用标题和摘要能够支持的判断，不从摘要外推具体实现或性能数字。该检索是面向研究决策的快速综述，不等同于系统综述或元分析。

## 3. 文献全景

### 3.1 最直接的顶会/顶刊工作

| 工作 | 发表 | 与本项目的重合 | 应借鉴什么 | 不能直接证明什么 |
|---|---|---|---|---|
| Zhang et al., *Autonomous Needle Navigation in Retinal Microsurgery: Evaluation in ex vivo Porcine Eyes* [1] | ICRA 2023 | 视网膜针导航、几何估计、安全轨迹和离体猪眼 | 将安全约束与终点误差分开评价 | 其方法以机会约束 MPC 为主，不是模仿学习 |
| Kim et al., *Towards Autonomous Retinal Microsurgery Using RGB-D Images* [2] | IEEE RA-L 2024 | 显微镜 RGB + OCT、工具/组织分割、亚视网膜注射 | 把 RGB、OCT、运动学约束放进同一自主系统，并分别报告感知、控制和试验结果 | 不是通过示范学习得到的策略；不能把仿真 RGB-D 结果直接迁移 |
| Zhang et al., *Deep learning-based autonomous retinal vein cannulation in ex vivo porcine eyes* [3] | Science Robotics 2025 | 自主视网膜静脉插管、CNN 方向/接触/穿刺识别、iOCT | 把学习限制在可验证子问题，同时保留手术流程和安全监督 | 其 90% 和动态 83% 来自特定离体猪眼流程，不是本项目 20 μm 仿真精度 |
| Gomaa et al., *Towards a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning* [4] | IROS 2024 | 眼科显微手术、模仿学习、强化学习、医生在环、个性化 | 用专家纠正和课程学习处理不同医生风格与微手术阶段 | 主要是在仿真白内障手术阶段验证，不是 RCM 针尖 20 μm 定位 |
| Jiang et al., *Adaptive Motion Scaling for Robot-Assisted Microsurgery Based on Hybrid Offline Reinforcement Learning and Damping Control* [5] | ICRA 2024 | 显微手术、离线数据驱动、运动缩放与阻尼 | 将数据驱动模块与确定性控制结合，而不是替代全部控制 | 评价的是运动缩放，不是完整眼内目标到达 |
| Kawaharazuka et al., *Robotic Constrained Imitation Learning for the Peg Transfer Task in Fundamentals of Laparoscopic Surgery* [6] | ICRA 2024 | 器械穿过固定支点的约束、单目图像、手术模仿学习 | 从专家动作提取运动约束，再围绕约束生成数据 | 任务和尺度与眼内注射不同；摘要不证明可实现 20 μm |

这一组文献支持一个清楚判断：眼科自主机器人已经大量采用“深度学习感知 + 显式几何/控制约束”的组合，而不是只依赖端到端 BC。当前仓库若把神经网络直接接到 A/B/C，而把 RCM、速度、接触和停止逻辑全部交给模型隐式学习，反而比已有强基线承担了更多风险。

### 3.2 外科模仿学习与失败恢复

| 工作 | 发表 | 核心信号 | 对本项目的直接启示 |
|---|---|---|---|
| Kim et al., *SRT-H: A hierarchical framework for autonomous surgery via language-conditioned imitation learning* [10] | Science Robotics 2025 | 高层语言规划 + 低层轨迹策略，显式处理长期任务和低层错误恢复 | 单目标注射暂时不需要语言 VLA，但“高层阶段/停止/回退”和“低层连续动作”应分开 |
| Hoque et al., *IntervenGen: Interventional Data Generation for Robust and Data-Efficient Robot Imitation Learning* [7] | IROS 2024 | 从少量人类干预自动扩展纠错数据，覆盖分布漂移状态 | 可在仿真中从策略失败点自动调用几何专家补数据，先做低成本 DAgger |
| Lee et al., *Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation* [8] | ICRA 2025 | 使用 diffusion 训练目标估计不确定性，机器人主动请求专家纠正 | 为“何时回退/请求专家”提供了比固定阈值更合理的设计方向 |
| Dai et al., *RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning* [9] | ICRA 2025 | 自动生成失败恢复轨迹、细粒度语言监督、VLM supervisor + actor | 当前不必上 VLM，但应显式标注 overshoot、visibility loss、rank deficiency 和 recovery phase |
| Kim et al., *Surgical Robot Transformer (SRT): Imitation Learning for Surgical Tasks* [18] | arXiv 2024，非顶会 | 将手术演示转成相对末端运动并训练语言条件 Transformer | 可作为手术动作表示的工程参考，不能当作已发表顶会结果 |

这组工作共同指向同一个原则：手术模仿学习的关键数据不是更多的“专家正常前进帧”，而是策略实际会到达的偏差、遮挡、过冲和恢复状态。只增加随机接近目标的成功轨迹，不能自动解决闭环分布漂移。

### 3.3 通用模仿学习策略

| 工作 | 发表 | 方法贡献 | 对本项目是否应立即采用 |
|---|---|---|---|
| Chi et al., *Diffusion Policy* [11] | RSS 2023 | 条件扩散生成动作序列，receding horizon，处理多模态动作分布 | 保留作为对照；当前失败首先不是生成模型容量问题 |
| Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* [12] | RSS 2023 | ACT 用 action chunking 降低高精度任务中的复合误差 | 很适合当前低维连续控制，尤其是 20 Hz 下的动作平滑和短时预测 |
| Ze et al., *3D Diffusion Policy* [13] | RSS 2024 | 用稀疏点云的三维表示提高空间泛化和数据效率 | 若未来加入真实 OCT B-scan 或点云，可考虑；当前已有低维 OCT 目标误差向量，不需要为了“3D”而先引入点云网络 |
| Black et al., *π0* [14] | RSS 2025 | VLM + flow matching，面向大规模多机器人数据 | 当前单任务、小数据和 20 Hz 安全闭环不适合从零用 VLA；可在多目标/多阶段和高层语义明确后再评估 |
| Prasad et al., *Consistency Policy* [17] | RSS 2024 | 通过一致性蒸馏加速 diffusion policy 推理 | 在扩散策略闭环成功后用于降延迟，不应优先于观测状态和停止逻辑修复 |
| Bharadhwaj et al., *RoboAgent* [15] | ICRA 2024 | 语义增强与 action chunking 提高数据效率 | 可借鉴图像/目标增强，但微米级定位仍需要专门的几何验证 |
| O'Neill et al., *Open X-Embodiment* [16] | ICRA 2024 | 多机器人数据标准化与大模型正迁移 | 说明数据格式和跨机器人数据有价值，但不能证明眼内显微动作可跨本体泛化 |

当前任务的动作空间只有三轴，且可见 OCT 目标误差已把三维目标关系压缩成低维条件。扩散模型的多模态优势只有在确实存在多种有效策略时才会成为主要优势；若失败来自缺少速度、停止状态和恢复样本，换更大的生成模型不会修复因果信息缺失。

## 4. 当前仓库证据说明了什么

### 4.1 已具备的研究资产

现有基础数据为 200 个训练 episode、40 个验证 episode 和 40 个测试 episode，分别有 136,348、27,290 和 27,317 帧。训练和验证的导航成功数为 192/200 与 39/40。原始轨迹很长，成功 episode 通常从约 20 mm 开始，接近目标后只有很少的帧进入 20 μm 区域。

第一批近目标恢复数据的三个 split 各有 20 个 episode，共 580、556 和 534 帧，分别成功 20、19、19 个，负 `dC` 已进入数据。转换后的数据能够证明反向动作存在，但尚未证明新的策略已学会恢复。

第二批恢复数据已完成采集并通过当前目录下的数据完整性验证：主 train 180 集/5,094 帧/175 成功，train_a 45 集/1,260 帧/43 成功，train_b 45 集/1,271 帧/44 成功，val 40 集/1,154 帧/40 成功，test 40 集/1,136 帧/39 成功。合并第一批后，原始 run 目录对应 train 290 集、val 60 集、test 60 集，共 11,585 帧和 399 个成功导航。关键事实是：这些 run 数据已通过采集完整性验证，但尚未全部转换并合并进当前 `data/` 训练集；当前训练格式中的恢复数据仍主要是第一批 20/20/20。

### 4.2 离线指标为何没有转化为闭环成功

最近的结果暴露出三个彼此独立的问题。

第一，输入缺少教师控制状态。当前策略主要使用 RGB、A/B/C state 和 OCT 目标误差，没有显式输入实际速度、上一动作或累计阶段。几何教师的 A/B/C 指令依赖误差、速度、加速度限制、遮挡记忆和停稳条件；只给当前位置与目标误差时，同一个观测在“接近目标”和“从过冲位置回撤”时可能对应相反动作，策略无法稳定区分。

第二，数据分布被远距离进针支配。原始训练集 136,348 帧的 `dC` 全部非负，意味着原轨迹只展示正向插入，没有展示越过目标后的负向回撤。近目标区域虽然存在小幅正向减速，但在全局均匀采样中占比很低。策略因此更容易学到稳定的正向节奏，而不是学习误差趋近零时制动。

第三，评价指标选错。回归基线的归一化离线动作 MAE 曾降到约 0.105，物理 MAE 很小，但闭环仍达到约 19.39 mm 最终误差；扩散策略最佳归一化 MAE 约 0.668，闭环最佳真值误差约 1.25 mm，随后继续过冲到最终 13.65 mm。这两个结果不能互相直接比较，因为模型、采样器、归一化和数据不同；它们共同说明离线动作误差无法区分“大部分时间稳定前进”和“关键几步是否正确制动”。

因此下一步不应只调学习率、batch size 或 diffusion step。应先让策略看得见闭环所需的隐藏状态，再让训练集覆盖策略会进入的错误状态，最后才比较网络族。

## 5. 与现有工作相比，真正的研究空白

本次检索没有发现一篇顶会或顶刊论文同时满足以下全部条件：

- 眼内或视网膜 RCM 机器人；
- 真实或注册的 OCT 几何观测作为策略条件；
- 从专家示范训练连续动作策略；
- 在闭环中显式保持 RCM、速度、接触和停稳约束；
- 用策略自身失败分布构造恢复数据；
- 在独立测试目标、遮挡和运动扰动下报告 20 μm 级结果。

已有工作分别覆盖其中一部分：Zhang et al. [1-3] 覆盖眼内机器人和 OCT/安全控制，Gomaa et al. [4] 覆盖眼科模仿学习，Kawaharazuka et al. [6] 覆盖支点约束模仿学习，Hoque et al. [7]、Lee et al. [8] 和 Dai et al. [9] 覆盖失败恢复，ACT/Diffusion Policy [11-13] 覆盖通用动作生成。这个组合空缺可以支撑方法研究，但论文不能只写“把 diffusion policy 用在 RCM 上”，因为这会表现为已有模块的工程拼接。

更有潜力的贡献应回答一个现有文献尚未回答的问题：**在固定远心点和有限 OCT 观测下，如何让模仿策略既保留几何控制器的安全性与停稳行为，又能从视觉/OCT 误差和失败恢复中学习传统解析控制不擅长的部分？**

## 6. 推荐的模仿学习方案

### 阶段 0：先固定评测协议

在训练任何新模型前冻结独立测试集和评价口径：

- 按目标/episode 分组划分，不能按帧随机划分；
- test 只用于最终评价，不参与特征选择或阈值调整；
- 主指标为最终独立 MuJoCo 位姿误差 `<20 μm` 且连续停稳，不接受只满足视觉误差；
- 同时报告 `<100 μm` 成功率、平均/中位误差、成功时间、路径长度、最大 RCM 偏差、超调量、最大速度/加速度、安全干预次数、不可观测中止率和失败原因；
- 至少用固定随机种子重复三组，报告均值和置信区间，避免单个 smoke episode 决定方向；
- 将离线 MAE 仅作为诊断指标，不能作为主结果。

当前先把两批恢复数据转换并合并，得到约 490/100/100 个 train/val/test episode。若第二批转换后图像缓存需要重建，也应从合并后的 640×480 数据统一生成，避免训练与推理预处理不一致。

### 阶段 1：建立可观测的强行为克隆基线

先保留现有扩散策略作为对照，同时新增一个结构更简单、输入更完整的基线：

```text
输入：
  RGB 当前帧 + 1 张历史帧
  A/B/C 当前 state
  实际或由真实编码器差分得到的 velocity
  上一控制周期实际动作
  OCT navigation: target - tip 的三维相机坐标差、距离、valid mask
  可选：历史 OCT 误差、可见性/几何质量标志

输出：
  短动作块 [ΔA, ΔB, ΔC]_{t:t+H}
  停止概率 p_stop
  可选：距离分层或剩余时间预测
```

推荐先比较三个模型，而不是直接扩大 diffusion：

1. **强 MLP/GRU 单步或 4-step 回归基线**：确认新增速度和动作历史是否解决闭环漂移；
2. **ACT 风格 8-step 动作分块策略**：利用短时动作生成降低复合误差，但每个控制周期仍接收新观测并做 receding-horizon 重规划；
3. **相同输入下的 Diffusion Policy**：保持原采样器和归一化，只改变观测、数据和采样权重，形成严格消融。

停止不能用后处理阈值完全替代策略学习。建议增加二分类停止头，与动作头联合训练；样本包含“达到 20 μm 且编码器停稳”的正例，以及刚进入阈值但仍有速度的负例。执行时可使用动作分块，但不能把预测的动作块盲执行到底，必须在出现不可见、误差反向或不确定度升高时立即重规划。

### 阶段 2：按闭环难度重采样，而不是按帧均匀采样

训练采样需要覆盖四类状态：

- 远距离接近；
- 中距离对准；
- 20 μm 到 800 μm 的减速与停稳；
- 越过目标、侧向偏差、遮挡、方向不可观测和恢复。

可采用距离分层的分层 batch、按 episode 等权、困难状态过采样或 focal-style 动作损失，但不能只提高近目标权重；若没有负动作和过冲状态，过采样只会重复同一正向动作。第二批恢复数据应先作为新训练集验证这一点。

### 阶段 3：用自动 DAgger 覆盖策略自己的错误状态

在仿真中，几何视觉控制器可以作为脚本专家。每一轮：

1. 用当前策略在 train 目标上闭环 rollout；
2. 记录策略偏离目标、进入遮挡、减速失败或接近安全边界的时刻；
3. 从这些状态调用几何专家或 MPC 给出纠正动作；
4. 使用 IntervenGen 式方法从少量纠正状态扩展相邻扰动；
5. 将纠正 rollout 加入训练集，并在固定 validation 上检查是否损害原有成功轨迹；
6. 重复到 test 上的失败类型稳定，而不是只到训练 loss 下降。

这不等于“把专家全部轨迹再采一遍”。关键是让训练分布覆盖策略自己产生的状态。若要加入机器人主动请求专家，Diff-DAgger 的不确定性思路可以作为第二阶段：先按固定规则回退，再研究不确定性门控能否减少不必要的专家介入。

### 阶段 4：使用安全残差策略，而不是完全替代几何控制器

建议优先试验以下结构：

```text
安全基础控制器：当前几何视觉伺服或 MPC
学习残差：只修正 C 轴近目标制动、A/B 小偏差和遮挡恢复
动作投影：RCM 约束、速度/加速度限制、安全包络
门控：不可见、几何质量差、不确定性高时残差=0，回退基础控制器
```

这种 residual imitation learning 有三点优势：保留已通过验证的 RCM 与停稳机制；将学习重点限制在真正困难的近目标段；更容易做消融，证明收益不是来自重新实现了控制器。它比“策略直接输出绝对 A/B/C”更适合作为医疗机器人的第一版学习组件。

### 阶段 5：再讨论层次化与 VLA

SRT-H [10] 表明高层规划和低层轨迹策略能处理长时程手术及错误恢复，但其任务是离体胆囊切除，不是 OCT 眼内注射。π0 [14] 需要大规模多任务数据，当前 490 个单任务 episode 不足以支撑从零训练通用 VLA。

更合理的层次边界是：

- 高层：选择病灶/目标、决定访问顺序、判断接近/回撤/恢复阶段；
- 低层：在 20 Hz 下根据 RGB-D/OCT 输出安全动作块；
- 安全层：RCM、速度、可见性、接触风险和停止条件始终独立于学习策略。

只有当多目标、访问顺序和不同病灶类型真正成为数据变化来源时，语言条件或 VLA 才可能带来可验证增益。当前阶段把 π0/RT-2 类模型引入主控制回路，复杂度和数据需求都高于可解释的收益。

## 7. 论文实验矩阵

最低限度应包含以下消融：

| 实验 | 比较 | 要回答的问题 |
|---|---|---|
| 观测消融 | RGB + state；RGB + state + OCT；OCT + state；加入 velocity/上一动作 | 增益到底来自视觉、OCT 还是可观测的教师状态 |
| 数据消融 | 原始数据；首批恢复；完整恢复；自动 DAgger | 近目标制动是否需要恢复数据，还是简单输入修复已足够 |
| 策略族 | MLP/GRU BC；ACT；Diffusion Policy；安全残差 BC | 多模态生成能力是否必要 |
| 停止机制 | 固定误差阈值；学习停止头；学习停止头 + 停稳约束 | 学习停止是否优于后处理 |
| 安全机制 | 无投影；RCM 投影；RCM + 速度/加速度；完整安全回退 | 成功率与安全约束的权衡 |
| 鲁棒性 | 相机/深度噪声、OCT 偏置、延迟、遮挡、运动扰动、电机参数变化 | 方法是否只适配理想仿真 |
| 实时性 | diffusion 100 steps、蒸馏/flow、单步或 ACT | 20 Hz 名义周期下推理是否可用 |

测试报告至少应包含 60 个独立随机目标、3 个随机种子和按失败类型分层的结果。只展示最好成功轨迹，或把一次闭环 smoke 与离线验证集合并叙述，都不足以支撑顶会主张。

## 8. 建议的论文叙事

当前最值得验证的候选题目是：

**Safety-Constrained OCT-Aware Imitation Learning for RCM-Constrained Intraocular Injection**

或中文表述：

**面向 RCM 眼内注射的安全约束 OCT 感知模仿学习**

贡献不应写成“首次使用 diffusion policy”，而应围绕三个可实证问题组织：

1. 注册 OCT 目标误差与历史速度/动作如何使 20 μm 级近目标制动从离线可模仿变成闭环可执行；
2. 自动失败状态生成如何用远少于完整重新采集的数据覆盖过冲、遮挡与恢复；
3. RCM 安全残差与不确定性回退如何在提高成功率的同时保持固定远心点、速度限制和安全中止。

若最终实验显示简单 ACT/GRU 已达到与 diffusion 相同的闭环成功率，并且 residual + fallback 显著优于端到端绝对动作，那么论文的价值反而更清楚：核心贡献是安全、可观测性和数据构造，而不是模型复杂度。

## 9. 开放问题

- 当前 OCT 是 MuJoCo 理想注册深度，不是真实 A-scan/B-scan、散斑、组织层或折射校正数据。真实 OCT 下的分割误差、扫描延迟和配准漂移仍是未解决证据缺口。
- 当前动作标签是协议目标，不是下一实际状态；电机一阶滞后使 action-state 关系非马尔可夫。必须把速度和动作历史纳入观测，或显式训练动力学/延迟模型。
- 当前恢复数据动作方向覆盖已有改善，但其规模和场景仍来自单一眼球/单目标几何。多目标顺序、组织接触、虹膜/镜头遮挡和真实呼吸运动尚未验证。
- 目前没有公开、统一的 RCM 眼内注射 imitation learning benchmark。论文需要自己建立按目标分组的训练/验证/测试协议，并避免测试目标泄漏。
- 若迁移到硬件，必须单独验证 RCM 注册误差、相机-机器人标定、OCT 时钟、执行器延迟、力/接触安全和失败制动；仿真三维误差不能作为临床精度证据。

## 10. 结论

**RQ1 的答案**：有直接相关论文，但不存在一个已经完整覆盖“OCT 条件 + RCM 眼内注射 + 模仿学习 + 失败恢复 + 20 μm 闭环”的现成顶会论文。最直接的顶会入口是 IROS 2024 的眼科 robotic apprentice 和 ICRA 2024 的约束模仿学习；最强的手术任务证据来自 ICRA 2023 视网膜针导航、RA-L 2024 RGB-D/OCT 自主亚视网膜注射，以及 Science Robotics 2025 的自主视网膜静脉插管。通用策略应以 ACT、Diffusion Policy、3D Diffusion Policy 和 π0 为方法参照，而不是把它们直接当作医疗任务有效性的证据。

**RQ2 的答案**：当前瓶颈是闭环可观测性和数据覆盖，不是网络规模。教师状态中的速度、上一动作和停止阶段没有进入模型；原始动作只有正向进针；离线 MAE 被大量稳定前进帧掩盖；一次随机目标 smoke 不能代表策略质量。

**RQ3 的答案**：先完成恢复数据转换与合并，建立带历史速度、动作和停止头的强 BC/ACT 基线；再用主动失败采样和自动 DAgger 补充策略自身错误状态；最后将学习模块限制为 RCM 安全控制器的残差，并做完整闭环、消融、鲁棒性和真实感降级实验。若这一路线能在独立 60 目标测试集上稳定达到 20 μm 停稳，同时证明恢复数据、历史状态和安全残差各自的贡献，才具备形成顶会方法论文的证据基础。

## References

[1] P. Zhang, J. W. Kim, P. Gehlbach, I. Iordachita, M. Kobilarov, "Autonomous Needle Navigation in Retinal Microsurgery: Evaluation in ex vivo Porcine Eyes," 2023 IEEE International Conference on Robotics and Automation, 2023.

[2] J. W. Kim, S. Wei, P. Zhang, P. Gehlbach, J. U. Kang, I. Iordachita, M. Kobilarov, "Towards Autonomous Retinal Microsurgery Using RGB-D Images," IEEE Robotics and Automation Letters, 2024.

[3] P. Zhang, P. Gehlbach, R. H. Taylor, I. Iordachita, M. Kobilarov, "Deep learning-based autonomous retinal vein cannulation in ex vivo porcine eyes," Science Robotics, 2025.

[4] A. Gomaa, B. Mahdy, N. Kleer, A. Krüger, "Towards a Surgeon-in-the-Loop Ophthalmic Robotic Apprentice using Reinforcement and Imitation Learning," 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024.

[5] P. Jiang, W. Li, Y. Li, D. Zhang, "Adaptive Motion Scaling for Robot-Assisted Microsurgery Based on Hybrid Offline Reinforcement Learning and Damping Control," 2024 IEEE International Conference on Robotics and Automation, 2024.

[6] K. Kawaharazuka, K. Okada, M. Inaba, "Robotic Constrained Imitation Learning for the Peg Transfer Task in Fundamentals of Laparoscopic Surgery," 2024 IEEE International Conference on Robotics and Automation, 2024.

[7] R. Hoque, A. Mandlekar, C. Garrett, K. Goldberg, D. Fox, "IntervenGen: Interventional Data Generation for Robust and Data-Efficient Robot Imitation Learning," 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2024.

[8] S.-W. Lee, X. Kang, Y.-L. Kuo, "Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation," 2025 IEEE International Conference on Robotics and Automation, 2025.

[9] Y. Dai, J. Lee, N. Fazeli, J. Chai, "RACER: Rich Language-Guided Failure Recovery Policies for Imitation Learning," 2025 IEEE International Conference on Robotics and Automation, 2025.

[10] J. W. Kim, J.-T. Chen, P. Hansen, L. X. Shi, et al., "SRT-H: A hierarchical framework for autonomous surgery via language-conditioned imitation learning," Science Robotics, 2025.

[11] C. Chi, S. Feng, Y. Du, Z. Xu, et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," Robotics: Science and Systems XIX, 2023.

[12] T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," Robotics: Science and Systems XIX, 2023.

[13] Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, "3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations," Robotics: Science and Systems XX, 2024.

[14] K. Black, N. Brown, D. Driess, A. Esmail, et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," Robotics: Science and Systems XXI, 2025.

[15] H. Bharadhwaj, J. Vakil, M. Sharma, A. Gupta, et al., "RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking," 2024 IEEE International Conference on Robotics and Automation, 2024.

[16] A. O'Neill, A. Rehman, A. Maddukuri, A. Gupta, et al., "Open X-Embodiment: Robotic Learning Datasets and RT-X Models," 2024 IEEE International Conference on Robotics and Automation, 2024.

[17] A. Prasad, K. Lin, J. Wu, L. Zhou, J. Bohg, "Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation," Robotics: Science and Systems XX, 2024.

[18] J. W. Kim, T. Z. Zhao, S. Schmidgall, A. Deguet, et al., "Surgical Robot Transformer (SRT): Imitation Learning for Surgical Tasks," arXiv:2407.12998, 2024.
