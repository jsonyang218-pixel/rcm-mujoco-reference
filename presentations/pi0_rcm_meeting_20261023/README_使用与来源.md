# π0论文阅读与眼底机器人导航策略学习进展

本材料根据用户确认的逐页方案制作，沿用用户上传PPT的16:9画布、蓝色渐变封面、浅蓝页眉、徽章、天印湖科创学院logo与页脚样式。原模板文件未修改，原有研究正文未作为本次进度依据。

## 文件与使用

- `pi0_rcm_navigation_meeting_20261023.pptx`：16页可编辑演示文稿；正文13页，备份3页。
- `pi0_rcm_navigation_meeting_20261023.pdf`：排版预览与演示备用文件。
- `speaker_notes_讲稿与计时.md`：逐页讲稿、计划时长及证据来源；讲稿也写入PPT讲者备注。
- 正文计划时长为13分30秒，π0第3–5页共4分钟，留1分30秒用于翻页、指图和补充。此为计划计时，非已完成口头试讲。
- 标题、文字、表格和方法图节点/连线是原生PowerPoint对象。原论文图和数值图作为有来源的图片插入；实验图由真实CSV/JSON确定性绘制，保留生成脚本与数据来源。
- 字体为Noto Sans SC。完整资料包提供常规/粗体字体与OFL许可证；缺字体时可安装后重新打开。PDF已按该字体渲染。
- 素材与脚本用于追溯和复绘；完整导航采集目录及用户原始模板不在资料包中。生成脚本中的采集路径需按本地文件位置调整，不能将它称为无外部依赖的一键复现包。直接编辑PPTX不需要运行脚本。

## π0论文来源

Black et al. **π0: A Vision-Language-Action Flow Model for General Robot Control**. 2024. arXiv:2410.24164.

- 用户指定原站：https://www.physicalintelligence.company/download/pi0.pdf
- 原站本环境访问受限，使用两个公开GitHub镜像的实际PDF；两份字节一致。镜像及SHA256、裁切页码和坐标见`paper/assets.json`、`paper/paper_evidence.md`。
- 第3页采用Figure3原图；第5页采用Figure13下半图，保留原图例、纵轴和Packing Eggs例外。纵轴为每任务10次的平均任务进度，不是成功率。
- π0动作块H=50、10步Flow Matching积分和机器人执行频率分别说明；本项目未实现π0或Flow Matching。

## 项目证据

- 固定三点与旧遮挡配对：`reports/20261022/meeting/`；三点6.634/6.085/6.268 μm，恢复组最大6.442 μm，均为专家控制器。
- 补充包发布提交：`10c465607cceea3a298a360746087b58053f920e`，分支`experiment/oct-three-group-79afacc`。旧primary与接口接入后的local_diagnostic分开；新本地C三点10–11 μm。归档审计PASS不改变A/B失败结局，不反向建立统一采集版本出处。
- 随机示教与训练：`runs/random_policy_overnight_20261022/`；19次任务16合格、26,705帧，布局分组12/2/2；合格示教64目标最大14.981 μm。
- 学习策略仅离线一步动作评估：RGB＋状态MSE0.1203，对照0.2067，下降41.8%；仅2个测试episode。C轴动作MAE没有改善，策略闭环未运行。
- OCT为MuJoCo理想注册深度接口，没有真实OCT/A/B-scan；完整深度仅保留各阶段末帧，序列元数据另存JSONL。

## 制作与检查方法

已按下列软件/方法资料组织制作与审查，它们不是本项目性能结果来源：

1. K-Dense Inc. **Scientific Slides**, v1.12, Scientific Agent Skills，commit `593377bb048752abb0152deab9fbb1083a6087df`。
   https://github.com/K-Dense-AI/scientific-agent-skills/tree/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-slides
2. HKUSTDial. **Supervisor-Skills**，`tech-paper-template`与`figure-designer`，commit `a20d76ef3979b804d7dbf599d577826957e0a8a9`。
   https://github.com/HKUSTDial/Supervisor-Skills/tree/a20d76ef3979b804d7dbf599d577826957e0a8a9

检查范围为页数/计时、文本和代码/数据对应、原图出处、独立逻辑审查，以及LibreOffice导出PDF后的逐页视觉检查。详细发现、修复和复查记录见随附检查报告。未进行现场投影或实际口头试讲。
