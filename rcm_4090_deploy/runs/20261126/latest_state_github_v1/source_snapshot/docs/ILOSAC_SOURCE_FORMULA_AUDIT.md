# ILOSAC 源码与公式核查

核查日期：2026-11-18。

集成状态：本文、原始论文和机器可读来源状态已接入项目的 `docs/`，作为
ILOSAC/THMM 复现边界与命名依据。当前项目仍没有已验证的官方 ILOSAC
源码，现有 replay actor-critic 不应改名为 ILOSAC。

**结果：公式已取得并核对原始 PDF；这篇 ILOSAC 的官方算法源码仍未确认。**
本包包含论文原始 PDF、公式对应关系和公开仓库核查记录，不包含已验证的 ILOSAC 官方实现。
目前没有将新模块接入 RCM 项目，也没有将现有 Replay 重命名为 ILOSAC。

## 论文来源

Youchao Zhang, Fanghao Wang, Xiangyu Guo, Yibin Ying, Mingchuan Zhou,
Zhongliang Jiang, Alois Knoll.
Robotic Cell Micromanipulation Skill Learning via Imitation-Enhanced Reinforcement Learning.
CAAI Transactions on Intelligence Technology, 11(1), 123–136.
DOI: https://doi.org/10.1049/cit2.70076

原文：
https://hub.hku.hk/bitstream/10722/368297/1/content.pdf

项目内
[ILOSAC_cit2.70076_original.pdf](references/ILOSAC_cit2.70076_original.pdf)
为同一原始文件，未修改页面内容。机器可读来源状态见
[ILOSAC_source_status_20261118.json](references/ILOSAC_source_status_20261118.json)。

## 已核实的公开代码来源

从同作者 SIRIL 论文所链接的公开账号 `zycrobot` 追溯，枚举了 31 个公开仓库，
进一步检查了四个相关候选的文件树、README 或 SAC 文件。未审阅这 31 个仓库的全部文件内容。

| 候选 | 核查结果 | 能否作为本篇 ILOSAC 官方源码 |
|---|---|---|
| https://github.com/zycrobot/SIRIL | README 对应另一篇 SIRIL 工作；文件树没有发现 ILOSAC/THMM 命名实现 | 当前不能 |
| https://github.com/zycrobot/sofa_cell_micromanipulation | README 为 SOFA 细胞微操作仿真场景 | 当前不能 |
| https://github.com/zycrobot/cleanrl | CleanRL 分支；已检查 `cleanrl/sac_continuous_action.py`，没有核实到论文特有约束实现或论文出处 | 当前不能 |
| https://github.com/zycrobot/CHAOtlab | README 为另一软件项目，未建立与本论文的对应关系 | 当前不能 |

“未确认”不等于证明作者未公开代码，也不排除其他账号、历史版本、附件或非公开发布。
本次排除了把相关仓库或通用 SAC 分支直接标成 ILOSAC 官方实现的做法。

论文的数据可用性声明写明数据可向通讯作者合理索取；这不等于已经提供代码仓库。

## 公式到模块的对应

| 论文位置 | 实现对象 | 当前 RCM Replay 状态 |
|---|---|---|
| 第 5 页，式(7)–(10) | 高斯隐状态模型、前向/后向推断及 EM 参数估计 | 没有 THMM |
| 第 5–6 页，式(11)–(16) | 解码、阶段后验、持续时间预测及约束状态集合 | 没有该约束模块 |
| 第 7 页，式(17)–(18) | 价值学习与熵温度更新 | 有双 Q，但没有原文的 SAC 熵温度更新 |
| 第 7 页，式(19)–(20) | 专家策略分布 KL 约束及策略更新 | 以专家 Flow matching 约束替代，尚非原方法 |
| 第 6 页，Algorithm 2；第 7 页对应说明 | 约束惩罚、约束放宽及回放更新 | 当前为旧数据离线回放，没有原文的完整在线约束流程 |

页码均为 PDF 页面上印刷的页码，原文件共 14 页。

### 关键公式

式(12)：

\[
h_{t,i}=\frac{\alpha_{t,i}}{\sum_{k=1}^{K}\alpha_{t,k}}.
\]

式(13)的持续时间预测递推按原文记号为：

\[
\alpha_{t,i}=\sum_{j=1}^{K}\sum_{s=1}^{t-1}
\alpha_{t-s,j}a_{j,i}\mathcal N(s\mid\mu_i^S,\Sigma_i^S).
\]

式(19)：

\[
D_{KL}(\pi_z\Vert\pi_\phi)
=\sum_j\pi_z(j)\log\frac{\pi_z(j)}{\pi_\phi(j)}.
\]

式(20)：

\[
L_\pi(\phi)=\mathbb E_{s_i\sim D}
[\alpha\log\pi_\phi(a_i\mid s_i)-Q_\theta(s_i,a_i)]
+\lambda_{KL}^{t}D_{KL}(\pi_z\Vert\pi_\phi).
\]

这里的 \(\pi_z\) 是示范策略分布。当前代码的 Flow matching MSE 不是该 KL，
Flow 的初始高斯噪声也不能直接当作最终动作分布的均值和协方差。

## 接入前尚需确认的四项

1. **式(17)下方的 TD 目标。** 原 PDF 第 7 页印刷的目标 \(y_i\) 在折扣后的软 Q 项外
   带平方。常见 SAC 目标不含这个平方；需要通过作者源码或说明确认，不能默默改掉再声称严格复现。
   对照资料：https://spinningup.openai.com/en/latest/algorithms/sac.html 。
2. **式(7)的下标与转移项。** 前向/后向及边缘后验部分有下标/记号疑点。
   实现时应逐项记录采用的解释，不能将标准 HMM 实现自动视为作者原实现。
3. **约束集合成员判定。** Algorithm 2 使用 \(s\in S_{THMM}\)，但将连续观测映射到
   该集合的可执行阈值/判定细节尚未从源码核实。自行添加马氏距离阈值属于实现选择。
4. **示范分布与衰减时钟。** 需确认 \(\pi_z\) 的参数化、协方差含义，以及
   \(\lambda_{KL}^{t}\)、\(\lambda_s^t\) 的 t 是 episode、交互步还是梯度步。

这些是实现来源与可复现性问题，不构成“论文一定错误”的结论。

## 项目中的贡献应如何区分

当前约 171 µm 的结果仍归属于已经运行的 Flow Replay 适配方案。
今后取得原实现或完成有明确假设记录的论文复现后，应另外建立版本与实验记录。
论文模块作为复用部分；状态注意力、与 DiT/Flow 的结合或其他新增机制作为待验证贡献。
取得作者源码和完成公式复现，都不能单独证明新增机制有效。

若需要向作者索取实现，原文通讯邮箱为 `mczhou@zju.edu.cn`。
应同时索取 ILOSAC/THMM 源码、运行配置，以及上述公式与约束判定的说明。
本次没有向作者发送消息。
