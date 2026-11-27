

### 2026-11-27T10:04:13+08:00 最新v3状态GitHub归档验证（latest_results_github_v2）

- 授权/目的：用户“最新状态上传到github”；归档2026-11-27T10:02:43+08:00。无新增训练、推理、仿真或重试。
- 基于分支experiment/output-head-sanity-20261125，父提交584456f3cc309f44f6c30af8eeafc0c4bea1c017；前一归档runs/20261127/latest_results_github_v1/保留。
- 命令：python3 -E -B runs/20261127/latest_results_github_v2/verify_snapshot.py --snapshot-dir runs/20261127/latest_results_github_v2。
- 来源载荷737文件、22,519,478字节；JSON/JSONL解析、Python AST、文件SHA/大小、捕获稳定性、禁止文件与凭据模式检查通过；25个输入来源项分别注明当前捕获/既有归档/vendor复核或历史大二进制未重新散列。两个候选实际权重SHA重算通过；完整计数见manifest.json/verification.json。
- 实验：dit_quantization_recovery_v3；E40/E42采集92/92分支；candidate1训练2400更新、E41近目标3/4；按卡一次修复1200更新后candidate2近目标4/4。训练未使用E41/标准目标，E41是已使用的开发目标而非独立测试。
- 标准五目标：E00/E02/E06/E08/E10，seed42/44/48/50/52，各1600周期、50ms、G10、K1/exec1/4flow步、原限幅及≤100µm测量OCT切换；同v2卡片控制字段一致。最后200动作后三维误差均≤20µm（含最终动作）为5/5；最终误差均值8.887796µm，v2已存对照33.521926µm、1/5。此为本卡20µm门槛成功，不等同原0.35µm停稳验收，也未运行screen10/dev20/final_test。
- 选中模型candidate2 SHAdae7252fe5ee1af6a35d39974b395b674f9b6607175b8e6d984d5331812cd843；初始化v2 SHA748d950a31c1d2529af8f0fb18a5d90933de16b723e7dc4dc55ae20f9096ce94；数据及可用指纹随experiment_card/input_fingerprints/relative_window_manifest归档。候选1失败及预检异常原样保留。
- 输出：runs/20261127/latest_results_github_v2/；主TEST_LOG截点6,608,613字节，415排除项记录在capture_inventory.json；大权重、原始轨迹、图像视频、数组和缓存留本地；源码与历史换行/空白按原样保留。
- 结果：本地归档验证通过，推送后另记远端SHA回执；软件/MuJoCo仿真范围。
