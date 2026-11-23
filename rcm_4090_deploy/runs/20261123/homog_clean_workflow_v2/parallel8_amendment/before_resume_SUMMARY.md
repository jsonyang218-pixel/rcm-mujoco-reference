# 同源新渲染近目标回放 — 2026-11-23

流程状态：interrupted。

基点 `71040dd33a48644da662dc92ad28053a2a326871`；独立分支 `feature/clean-render-homog-replay-20261123`。

全部阶段、训练、初始化及单episode均无wall-clock时长限制；0.35µm只用于诊断，不提前停止。

92%池仅使用原始episode0–838完整有效轨迹窗口，排除全部Plan-2窗口；8%为本轮同源新包。

结构化报告见 `rcm_4090_deploy/github_reports/homog_20261123_s*/`。

权重、优化器断点及原始纠偏RGB保留本机；发布路径/大小/SHA256清单。代码、配置、日志及结构化证据上传GitHub。

仅验证软件与MuJoCo仿真，不代表临床精度或硬件就绪。
