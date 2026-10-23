# MuJoCo建模PPT：独立文字、数学与源码一致性审查

日期：2026-10-23。审查对象为独立7页生成器`build_modeling_deck.py`；本次未检查旧π0汇报，未修改任何PPT或生成器，未运行导航、训练或动力学步进。仅写入本审查文件。核对源码为`/workspace/rcm-mujoco-reference/rcm_reference.py`、`rcm_recovered_model.py`与`config.json`。

最终打包版本PPTX SHA-256（由root提供的最终版本记录）：`2919163ee05ef8f5ae9625e629a198087f5f0d267970f409a3a0d2ab1abb18b8`。最终修订将公式改为原生上下标（W左上标表示世界系），未改变数学内容；第5页和第7页单位说明采纳如下。本轮仅更新审查记录，按打包阶段要求未新增审查或试验。

## 结论与必须修改项

7页正文中的主链局部轴、零位偏置、45°安装旋转、4 mm暂定参考长度、7个qpos与3个独立任务自由度的区分，以及名义B映射与外观杆件的分离均与源码一致。**所有必须修项均已关闭；无剩余必须修改的文字、运动学公式或源码引用项。** 第6页三姿态数值证据已核验通过，来源函数名已在生成器及实际PPTX修正；第7页B逆映射分支限制已加入生成器和现有PPTX。

| 级别 | 页码 | 问题 | 要求 |
|---|---|---|---|
| MINOR，已关闭 | 6 | 初稿source写`state_from_task、apply_task、inspect`，但源码没有`apply_task`或`inspect`。 | 生成器与实际PPTX（SHA `40c0e0d677ffa3a6901c87f59ad44a1a8b7563c71d7ec1496b4cad23851cc53d`）均已改成`state_from_task、set_task、metrics`。`set_task`第235–242行负责写qpos、从属关节、置零qvel和`mj_forward`；`metrics`第394行起计算几何指标。 |
| MINOR，已关闭 | 7 | 初稿source写`rcm_reference.py:apply_task及build_xml`。 | 生成器和实际PPTX均已改为`set_task`。 |
| 证据核验，已关闭 | 6 | 首轮数值文件未落盘，原暂不验收。 | 现已读取`evidence/numeric_checks.json`及`model_evidence.md`，确为3个姿态；FK差范数最大`8.140289677804162e−15 mm`，针轴到RCM距离最大`2.979040983896728e−15 mm`。JSON明确坐标显示值已四舍五入、原始差范数另存；仅作为理想软件模型浮点一致性，与第6页文字一致。 |
| MINOR，已关闭 | 6 | 原caption称源码哈希见numeric_checks，但JSON原无哈希字段。 | `numeric_checks.json`已新增`source_files_sha256`，含config与两个源码文件；逐一与磁盘重算SHA-256一致，caption定位现正确。 |

本次不判断投影字号、遮挡、公式字体或PPT原生对象行为；这些需在新7页PPTX生成后单独渲染检查。

## 逐页核验

| 页码 | 结论 | 源码证据与边界 |
|---|---|---|
| 1 | 通过 | 明确为独立建模讲述；静态外视图未被作为新增导航实验。与素材`model_assets/render_metadata.json`中`physics_steps=0`、`navigation_executed=false`一致。 |
| 2 | 通过 | 场景组成与`Reference.build_xml`一致。源码任务关节为3个，`DEPENDENT_JOINTS`为4个显示关节；当前通过`set_task`写qpos并`mj_forward`，无actuator。外观`measured=false`和`visual_scale=0.67`明确，不推断实机尺寸或动力学。 |
| 3 | 通过 | `A_azimuth`为hinge、轴`0 -1 0`；`B_pitch`为hinge、轴`1 0 0`；`C_insertion`为slide、轴`0 0 -1`（`rcm_reference.py`第647–692行）。A/B body原点均在RCM，展开图已声明父子层级不表示轴空间分离，也不声称真实电机轴位于RCM。4个visual_B关节按`+1,-1,-1,+1`随b赋值。 |
| 4 | 通过 | 图使用安装系M，A轴为−Y_M，B轴为`Ry(−a)X_M`，针轴为`Ry(−a)Rx(β0+b)(−Z)`；当前φ0=0使图示写法与完整模型一致。实线与参考虚线均过同一个原点，C方向沿当前针轴，L标为L0+c。图示不按比例，未冒充实体几何。 |
| 5 | 通过 | 变换链`T(p_RCM) Rx(45°) Ry(−φ) Rx(β) Tz(−L)`与`build_xml`固定安装、A预旋转、B预旋转和C平移一致。φ0=0°、β0≈−13.764°、L0=4 mm为当前配置的正确值；L0的暂定与未实测边界在页面及讲稿均保留。源机器人系换基`diag(1,−1,−1)`未被误当成额外物理安装。 |
| 6 | 公式、数值与函数名通过 | d由正交旋转作用于单位向量得到，`||(I−ddᵀ)(p_tip−p_RCM)||`为点到针轴直线的垂距；理想模型中`p_tip−p_RCM=Ld`，故该值为0。3姿态证据与“浮点精度量级”表述相符。限定为软件/理想模型一致性，未映射为实机精度。 |
| 7 | 通过 | B公式准确使用名义60/52/16/98.48 mm参数与源码正根分支；明确s_B不是任务角增量b，且与外观70/48/12 mm等示意参数分离。现已说明逆解需分支约束/前姿态。无实机惯量、重力/接触关闭、没有actuator与动力学积分的当前运行方式表述符合源码。 |

## 数学交叉核对

直接从当前配置和名义B逆映射重新计算零位，而非照抄PPT常量：

- 编码器参考ABC=`[0,1466018,893518]`；B比例为63822.71488 counts/mm。
- φ0=0°；α0=54.23600525044604°。
- 源模型针轴使用θ=α+22°；MuJoCo局部X俯仰对应β=θ−90°，故β0=−13.763994749553959°。
- L0=`encoder_reference.needle_tip_distance_at_reference_mm`=4 mm=0.004 m，且`physical_needle_reference_confirmed=false`。这与精细针段长度也为4 mm是两个独立含义，不能互相视为实测校准；当前讲稿未混同。

源名义模型针轴是

`d_robot=[sin(α+22°)sinφ, cos(α+22°), sin(α+22°)cosφ]ᵀ`。

应用源码`R_robot_to_world=Rx(45°)diag(1,−1,−1)`后，与PPT的

`d_W=Rx(45°)Ry(−φ)Rx(β)[0,0,−1]ᵀ`，其中`β=α+22°−90°`，完全等价。当前φ=φ0+a、β=β0+b、L=L0+c；a/b按rad、c按m代入时，所有常量也要统一到rad/m。

名义B执行器公式为

`s_B=98.48−60cosα−sqrt(52²−(60sinα−16)²)` mm，

与`RCMModel.b_displacement`逐项一致；α在该Python接口使用度数，并在内部转换到弧度。该式是恢复自原程序的名义机构映射，不是MuJoCo外观显示杆件的闭链约束。

## 可选改进

1. **已采纳并关闭：**第7页现已补充“位移逆解需选定分支”，讲稿加入β=α−68°及前一姿态条件。与源码`alpha_candidates_from_b`和`reference_alpha_deg`选支一致。
2. **已采纳并关闭：**第5页最终版本将L0写成`0.004 m（4 mm）`，并补充代入前统一SI单位。β0的角度与弧度转换说明保留在讲述条件中。
3. **已采纳并关闭：**第7页最终图注补充“α接口用°、计算转rad”，避免与第5页内部a/b(rad)混淆。
4. 第4页蓝色a转角弧使用绕−Y的参考−Z投影，而灰色参考针已含β0。可在讲述时说转角弧为轴向示意，或令转角弧以含β0的参考针方向起始，使弧与参考姿态直观对应；当前两者都绕正确的轴，未构成公式错误。

## 范围、版本与计时

生成器为7页，预计15+45+55+75+75+60+55=380秒（6:20），与讲稿总时长说明一致。此处不代表已实测口述时间。

本次读取的源码哈希与静态渲染元数据一致：

| 文件 | SHA-256 |
|---|---|
| `rcm_reference.py` | `f7388260747c99f9f9851300c71c05c4878a6ffadd6f18fdb05acb1cf7a105c7` |
| `rcm_recovered_model.py` | `9c23a20a18305cd1b4718f9266fa87fbe337c868a88b94b242bd4dbf7a870271` |
| `config.json` | `0b739b29415fde416af85313caf4e2dd2391e2cda0e2e4bf3326055e882fd5de` |

静态图仅对独立内存场景进行白底、颜色、相机和显示组设置；元数据声明几何不变、原scene/config不变。PPT已说明显示标记和外观示意，未把放大针具或外观比例当作物理测量。
