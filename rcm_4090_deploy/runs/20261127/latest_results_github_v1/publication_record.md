

### 2026-11-27T09:10:38+08:00 最新结果GitHub归档验证（latest_results_github_v1）

- 用户授权：上传最新结果到GitHub；无新增训练、推理、仿真或重试。
- 版本/目的：归档2026-11-27T09:09:16+08:00当前部署源码与11组2026-11-26/27实验；基于分支experiment/output-head-sanity-20261125、提交3a5bd63af89d21249be78b9a5ca64a95fda35648。
- 命令：python3 -E -B runs/20261127/latest_results_github_v1/verify_snapshot.py --snapshot-dir runs/20261127/latest_results_github_v1。
- 来源载荷1451文件、39356778字节；JSON/JSONL解析、Python AST、逐文件SHA、捕获稳定性和禁止文件/凭据模式检查通过；25个已归档最新输入指纹逐字节匹配。完整归档计数见manifest.json/verification.json。
- 最新实验：dit_relative_v2_resume/candidate1，实际3000更新，E40/E42训练、E41开发；近目标4状态最后200周期≤20µm为4/4；标准E00/E02/E06/E08/E10、种子42/44/48/50/52、1600周期、50ms、K1/exec1/4flow步，最后200周期≤20µm为1/5，最终误差均值33.521926µm。complete_standard_negative，总体目标未达到；不等同原0.35µm验收成功。
- 模型：原v5 init SHA6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11；最终checkpoint SHA748d950a31c1d2529af8f0fb18a5d90933de16b723e7dc4dc55ae20f9096ce94，本次实际文件重新计算通过。数据清单/源指纹随各实验归档。
- 排除946项：权重、数组、原始轨迹/观测/动作载荷、图像视频和缓存；清单见capture_inventory.json。
- 输出：runs/20261127/latest_results_github_v1/；验证结果verification.json；主TEST_LOG快照截止6318703字节。日志和源文件原始换行/历史空白按原样保留。
- 结果：本地归档验证通过；推送后另记远端SHA核对回执。软件/MuJoCo仿真范围。
