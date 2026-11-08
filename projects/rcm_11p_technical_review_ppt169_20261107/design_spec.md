<!-- ppt-master-schema: design-spec/v1 -->
# RCM 眼内注射机器人 11 页技术汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | RCM 眼内注射机器人 11 页技术汇报 |
| Canvas Format | ppt169, 1280 × 720 |
| Page Count | 11 |
| Primary Language | zh-CN |
| Target Audience | 课题组技术评审教师与同学，熟悉机器人控制、模仿学习和实验评估，但需要快速核对本阶段机制、证据与风险 |
| Communication Intent | 先报告开发集成效与最终测试失败，再解释闭环、学习与蒸馏机制，最后提出下一阶段优先级并暴露尚未补齐的证据 |
| Desired Audience Outcome | 评审者认可“学习 base 是主要增益、扩散残差是有限辅助、外围目标外推与终端保持仍是主问题”，并据此讨论协议修正和模型改造 |
| Core Message / Ask / Action | 20 µm 闭环控制已在开发集达成，但外围目标外推尚未成立；先修测试协议与终端保持，再增加模型复杂度 |
| Delivery Context | 现场主讲式技术评审，约 25 分钟，页数按 1/2/3/3/3/3/3/2/2/2/1 分钟分配 |
| Artifact Afterlife | 组内复盘、后续实验排期与答辩材料复用；讲者备注保留证据口径和风险边界 |
| Reading Mode | presentation |
| Content Strategy | 证据优先：按用户确认的证据优先版叙事重组来源，不改变数值、样本口径或仿真边界 |
| Design Style | 自定义“证据校准台”：结论式大标题、超大关键数字、细网格与少量深蓝结构带，青绿只标焦点；国自然风格启发但不复制官方模板 |
| AI Image Acquisition Path | not applicable |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — 用户明确要求详细讲者备注 |
| Custom Animations | disabled — 用户明确要求无动画 |
| Narration Audio | disabled — 用户明确要求无旁白 |
| Created Date | 2026-11-07 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 72 px 左右安全边距，56 px 上边距，48 px 下边距 |
| Content Area | x=72..1208, y=56..672 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: Conclusion-first engineering review: open with the global verdict, explain each mechanism as a claim-evidence pair, and close by converting the failure evidence into ordered protocol and controller actions.
- **Visual style**: custom
- **Visual style Behavior**: Sharp, scientific-review geometry on a white field: thin 1 px grids, pale blue-gray measurement surfaces, deep-blue structural bands, restrained 16 px corners, generous whitespace, and one teal focal accent per page; no shadows, decoration gradients, 3D, or dense card walls.
- **Theme**: 20 µm calibration gate and three-stage evidence chain
- **Tone**: precise, candid, evidence-led, non-promotional

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | Primary page field |
| Secondary background | #EAF2F7 | Measurement surfaces and quiet blocks |
| Primary | #0B4F8A | Titles, structural bands, primary data series |
| Accent | #14988F | Single page focal accent and current claim |
| Secondary accent | #78A83D | Positive evidence and completed development gate |
| Body text | #172633 | Main prose and labels |
| Secondary text | #526571 | Annotations, sources, and explanatory text |
| Divider | #CBD9E2 | Hairlines, grid, and table rules |
| Warning | #D97706 | Limited support, distribution shift, and caution |
| Negative | #B42318 | Failure and unresolved terminal behavior |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | Neutral humanist sans with strong technical hierarchy | Microsoft YaHei | Arial | sans-serif |
| Body | Neutral humanist sans, high legibility at projection size | Microsoft YaHei | Arial | sans-serif |
| Data | Tabular numerals for comparison and latency | Microsoft YaHei | Consolas | monospace |
| Formula | Mathematical and symbolic content | Microsoft YaHei | Cambria Math | serif |

- **Typography upgrade (Reference)**: Target-machine substitution is unnecessary; Microsoft YaHei, Arial, Consolas, and Cambria Math are the export-safe named faces.
- **Role rationale**: Data uses Consolas for aligned numerals; Formula uses Cambria Math where native text math is selected.
- **Title stack**: Microsoft YaHei, Arial
- **Body stack**: Microsoft YaHei, Arial
- **Data stack**: Microsoft YaHei, Consolas
- **Formula stack**: Microsoft YaHei, Cambria Math

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 28 |
| Title | 44 |
| Subtitle | 32 |
| Annotation | 22 |
| Footer | 20 |
| Hero number | 96 |
| Section label | 24 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: Claim first, then one dominant diagram or evidence surface, then a compact editorial takeaway
- **Composition tendency**: Large-margin single-focus pages; two-panel evidence pages; left-aligned claims with a meaningful right-side calibration field
- **Cross-page continuity**: A thin horizontal 20 µm reference line, small square section markers, and page-local teal focus recur as one calibration motif
- **Spacing posture**: Open; dense pages use alignment and rules instead of nested containers
- **Spacing anchors**: 72 px page margin, 24 px block gap, 32 px column gutter, 16 px corner radius, 1.35 body leading

## VI. Icon Usage Specification

- **Primary bundled library**: tabler-outline
- **Stroke Width**: 2

| Icon Path | Suitable Scenarios |
| --- | --- |
| icons/tabler-outline/eye.svg | 感知、视觉与手术场景 |
| icons/tabler-outline/target.svg | 目标、20 µm 判据与定位 |
| icons/tabler-outline/route.svg | 控制流程、闭环与阶段路线 |
| icons/tabler-outline/cpu.svg | 策略网络、推理与蒸馏 |
| icons/tabler-outline/database.svg | 数据采集、缓存与标签 |
| icons/tabler-outline/git-branch.svg | DAgger 接管、分叉与版本 |
| icons/tabler-outline/chart-bar.svg | 实验证据与比较 |
| icons/tabler-outline/chart-line.svg | 误差与分布趋势 |
| icons/tabler-outline/alert-triangle.svg | 外推、保持失败与风险 |
| icons/tabler-outline/circle-check.svg | 已达成门槛与通过项 |
| icons/tabler-outline/clock.svg | 20 Hz、延迟与时间预算 |
| icons/tabler-outline/git-compare.svg | 三种基与消融对照 |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P03 | table | table/comparison_matrix | Compare the three base-role inputs, formulas, purposes, and development outcomes |
| P08 | chart | chart/horizontal_bar_chart | Compare development successes across the teacher, DAgger-only, and added modules |
| P09 | chart | chart/horizontal_bar_chart | Compare final-test success with displayed sample denominators |
| P10 | chart | chart/horizontal_bar_chart | Show the six mutually exclusive final-test outcome counts |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## IX. Content Outline

### Part 1: 任务、闭环与判断

#### Slide 01 - 开发达标，最终外推未达标

- **Audience move**: 从“是否已做出可汇报的闭环结果”转为“开发集已达标，但最终测试暴露外推缺口”
- **Relationships**: 20 µm 判据为 membership 基准；开发集 40/40 与最终测试 20/180 为直接 contrast；三条数字共同 parent 于同一个仿真结论
- **Composition**: 三个超大数字沿一条 20 µm 校准线展开，左下写结论，右下写仿真边界
- **Title**: 20 µm 闭环控制已在开发集达成，外围泛化仍未成立
- **Core message**: 开发集证明机制有效，最终测试证明当前结果不能外推为部署能力
- **Content**: 20 µm · 连续 3 帧停稳；20 Hz · 单步闭环；开发 40/40 → 最终 20/180；MuJoCo 仿真与模拟 OCT 边界
- **Cover impact (binding)**: 用“40/40 → 20/180”的反差作为开场钩子

#### Slide 02 - 受限感知闭环

- **Audience move**: 从“只有 RGB 或只有 OCT”转为“远场粗测与近场细测通过 2 mm 门控组成连续 20 Hz 闭环”
- **Relationships**: RGB → coarse target → fusion → policy → incremental action → scene feedback 为 ordered sequence；OCT 为 fine input；四种失效状态为 fallback membership
- **Composition**: 横向闭环主干，2 mm 切换点作为中央分隔；下方只保留回退状态带
- **Title**: 受限感知把远场 RGB 与近场 OCT 接成 20 Hz 闭环
- **Core message**: 系统把不同距离的测量权限和失效处理固化为可审计状态机
- **Content**: RGB-D coarse 0–30 mm；OCT fine 1 µm–3.5 mm；2 mm ±0.2 mm 过渡；0.1 s 低通融合；FINE_TRACKING / FINE_COAST / COARSE_FALLBACK / PRECISION_HOLD；0.5 s 后有限回退

### Part 2: 三种基、教师与学生

#### Slide 03 - 三种基的职责

- **Audience move**: 从“三个 controller 都在算动作”转为“三种基使用不同信息权限和不同饱和顺序，只有学习 base 是最终学生主干”
- **Relationships**: 几何教师、解析 base、学习 base 为 role contrast；每一行输入、公式、用途、开发结果为 link
- **Composition**: 三列角色卡转为无卡墙的三行比较表，青绿只标“学习 base”
- **Title**: 三种“基”承担不同职责，不能混为一种控制器
- **Core message**: 特权目标、受限 OCT 与 DAgger 学生是三种信息条件和训练角色，不等价于同一控制器
- **Content**: 几何教师 · 特权目标误差 · clip(K·J⁺e, ±L) · 数据采集诊断 · 8/40；解析 base · 受限 OCT 误差 · K·clip(J⁺e, ±L) · 动作先验 · 40/40 引导策略；学习 base · 512 视觉 + 21 状态/受限特征 · DAgger MLP · 最终学生主干 · 40/40
- **Mathematical content**: a_{\mathrm{geo}}=\operatorname{clip}(KJ^{+}e_{\mathrm{target}},\pm L);\quad a_{\mathrm{base}}=K\operatorname{clip}(J^{+}e_{\mathrm{OCT}},\pm L)
- **Visualization**: three-row comparison matrix with role, information permission, formula, and outcome
- **Native-ready**: `base-role-matrix=yes`

#### Slide 04 - 扩散教师

- **Audience move**: 从“扩散模型直接预测动作”转为“教师只学习解析比例动作附近的有界残差”
- **Relationships**: two observations concatenate into action encoder output; noisy action plus condition through denoiser to residual sequence
- **Composition**: two-stage native architecture: observation tower left, denoising network right; cache boundary shown by a thin rule
- **Title**: 扩散教师只学习比例动作附近的有界残差
- **Core message**: 教师把动作先验留给解析 base，把模型容量用于有限修正
- **Content**: 11 维状态输入 · 2 帧融合；每帧 512 + 64 = 576 → 1152 → 512 → 256；动作噪声 24 + 条件 256 + 时间 1 = 281 → 512 → 512 → 24；100 步 DDIM；单样本 x₀ 截断 ±4；七样本取中位

#### Slide 05 - 距离门控残差

- **Audience move**: 从“残差大小固定”转为“远场放开、近场收缩，终端权限只有 10%”
- **Relationships**: base action plus clipped residual equals final action; distance s(d) parents residual bound; valid OCT gates authority to zero
- **Composition**: one equation crossing a distance scale; near/far brackets and axis-limit labels form the measurement field
- **Title**: 距离门控只在远场放开残差，近场把权限收至 10%
- **Core message**: 残差预算随距离单调收缩，近目标时解析先验获得最高权限
- **Content**: OCT 方向 · log d · valid 编码；K=0.15；λrel=0.001；L=[0.2898, 0.1443, 0.0427]；s(d)=clamp(d/1 mm, 0.1, 1)；最终动作轴限幅
- **Mathematical content**: a=a_{\mathrm{base}}(d)+\operatorname{clip}(r_\theta(x),\pm s(d)L);\quad s(d)=\operatorname{clamp}(d/1\,\mathrm{mm},0.1,1)

#### Slide 06 - 单步学生

- **Audience move**: 从“学生也是扩散模型”转为“学生是 DAgger 单步 MLP，叠加一个很小的无门控微调项”
- **Relationships**: 512 visual + 18 restricted + 3 state = 533 input membership; base MLP and residual MLP branch in parallel and sum to action
- **Composition**: shared 533 input rail splits into a wide base branch and narrow residual branch; parameter counts anchor the comparison
- **Title**: 学生是 DAgger 单步 MLP 加一个很小的微调项
- **Core message**: 单次推理的速度来自缓存特征后的轻量 MLP，而不是教师结构的直接继承
- **Content**: 533 维输入；base 533→512→512→3；residual 533→256→256→3；741,894 参数 = 538,627 + 203,267；残差固定限幅 [0.005°, 0.003°, 0.001 mm]；无距离门控

### Part 3: 训练闭环与证据

#### Slide 07 - 数据闭环与蒸馏

- **Audience move**: 从“单次监督训练”转为“接管数据闭环贡献主要增益，蒸馏把七次采样压成单次推理”
- **Relationships**: geometric collection → DAgger takeover/relabel → seven-sample median labels → student distillation is ordered parent-child pipeline; latency is a separate measurement endpoint
- **Composition**: four-step horizontal data pipeline with a return loop under DAgger; latency receipt isolated at right
- **Title**: 数据闭环贡献主要增益，蒸馏把七次采样压成单次推理
- **Core message**: 关键变化的是数据分布与单步化，不是继续增加网络深度
- **Content**: 几何教师采集；DAgger 接管与重标注；七样本中位标签；学生蒸馏；策略头延迟 0.384 ms P50；七样本教师 288.7 ms P50；两者口径不同

#### Slide 08 - 开发集证据

- **Audience move**: 从“附加模块可能继续提升”转为“DAgger-only 已解决开发集，额外模块没有超过它”
- **Relationships**: five variant bars share a 40-episode denominator; DAgger-only is maximum; v7 pre-correction is excluded as a gray provenance reference
- **Composition**: dominant horizontal bar evidence with the selected result as the single teal focus and a gray v7 side note
- **Title**: DAgger-only 已解决开发集，额外模块没有超过它
- **Core message**: 学习 base 对闭环结果的贡献最大，后续附加模块没有形成开发集增益
- **Content**: 几何教师 8/40；DAgger-only 40/40；Velocity 34/40；Residual cap 38/40；Stop head 38/40；灰框：预修正 v7 39/40
- **Visualization**: horizontal bars for geometry, DAgger-only, velocity, residual cap, and stop head; a separate gray provenance note carries pre-correction v7
- **Native-ready**: `development-success-bars=yes`

#### Slide 09 - 独立测试

- **Audience move**: 从“最终失败可能来自感知噪声”转为“最终目标整体位于开发分布之外，外推失败是首要结构因素”
- **Relationships**: development and final polar ranges are disjoint; final-test success for distilled and seven-sample median share the same frozen target set but different episode counts
- **Composition**: left final-test counts, right polar-angle distribution; a gap between bands makes non-overlap explicit
- **Title**: 独立测试失败，首先因为目标完全位于开发分布之外
- **Core message**: 最终目标位于开发集未覆盖的外围后极角带，现有开发结论不能外推到该区域
- **Content**: 最终 20 µm：蒸馏 20/180，七样本中位 18/60；开发极角 1.71–25.76°，最终 28.10–36.86°；归一化后极坐标无交集；细通道偏置未整体上移
- **Visualization**: paired success-count bars with sample denominators; custom polar-band comparison is a qualitative child visual
- **Native-ready**: `final-success-bars=yes`

#### Slide 10 - 失败机制

- **Audience move**: 从“失败都归因于 OCT 不可用”转为“多数 episode 已进入 20 µm，随后没有保持住，细通道在场”
- **Relationships**: six outcome classes sum to 180; reached-but-not-held is largest; fine-channel availability and fallback are diagnostic conditions linked to that class
- **Composition**: left complete 180-episode outcome bars, right three-beat stability narrative: reach → lose hold → drift
- **Title**: 多数失败已经进入 20 µm，却没有保持住
- **Core message**: 终端策略稳定性比细通道可用性更接近当前主因
- **Content**: 20 成功；51 到达未保持；8 为 20–30 µm；41 为 30–50 µm；45 为 50–100 µm；6 未接近；9 初始化不安全；51 组最低误差中位 5.54 µm、最终误差中位 96.29 µm；细通道有效 99.52%；coarse fallback 0%
- **Visualization**: six outcome classes sum to 180; custom three-beat stability strip is a qualitative child visual
- **Native-ready**: `failure-mode-bars=yes`

### Part 4: 下一步

#### Slide 11 - 先修协议和保持

- **Audience move**: 从“继续堆叠模型模块”转为“先修复测试有效性与终端控制，再补外围对照和图像消融”
- **Relationships**: protocol repair and terminal control are immediate prerequisites; outer-band retraining follows; teacher/image ablations are next evidence links
- **Composition**: ordered three-step roadmap with immediate actions left and deferred evidence right; close on a single falsifiable next gate
- **Title**: 下一阶段先修协议和终端保持，再增加模型复杂度
- **Core message**: 下一步先让测试能区分模型能力，再让终端动作学会停在 20 µm，之后才值得扩大模型
- **Content**: 安全初始化筛选；按极角重划测试；外围目标补训；几何教师外围对照；图像分支闭环消融；显式终端保持；下一门槛仍以 20 µm 连续 3 帧停稳为判据
- **Closing impact (binding)**: 以“先修协议，再修保持；其余模型增量之后再做”收束

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: 每页说明要观众带走的判断、对应证据口径、公式与实验事实、已知限制和口头过渡；不得把开发集与最终测试样本口径混写，不得声称真实设备或临床精度
- **Total duration**: 约 25 分钟
- **Notes style**: 正式、技术评审式、直接说明证据与风险
- **Presentation purpose**: 报告开发进展与最终失败，解释机制，并推动下一阶段协议和终端控制决策
