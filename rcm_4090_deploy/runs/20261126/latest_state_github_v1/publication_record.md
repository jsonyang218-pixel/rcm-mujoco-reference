

### 2026-11-26T10:17:09+08:00 最新状态 GitHub 上传前验证（latest_state_github_v1）

- 目的：按用户“最新状态上传到github”归档 2026-11-26T10:15:08+08:00 的部署状态；无新增训练、评估或重试。
- 基于分支 experiment/output-head-sanity-20261125，父提交 b362ea79e00a24bbc1ec68199e78ac4656e59463；输出 runs/20261126/latest_state_github_v1/。
- 命令：python3 -E -B runs/20261126/latest_state_github_v1/verify_snapshot.py --snapshot-dir runs/20261126/latest_state_github_v1。
- 来源载荷674文件、16,128,331字节；275份JSON、2份JSONL解析通过，297份Python AST通过；来源SHA-256全部一致，捕获期间无单文件修改；完整归档另含本次元数据和验证器，计数见manifest.json/verification.json。
- 实验索引：output_head_sanity recovery_v2已完成但fast5门槛失败；learned-terminal局部恢复仅局部可行；standard2_v1协议失败、protocol_v2与standard3保留逐目标真值指标及严格成功0/5。
- 最新original_neural_terminal_pilot_v1：E40训练/E41留出、v5 init SHA 6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11；128更新后留出C MAE 1.4682645→1.9467759µm，离线门槛失败，未启动闭环；结束状态写入另有重复time关键字技术异常，均原样保留。
- 数据/权重：身份和现有SHA见各experiment_card.json/input_fingerprints.json/training_result.json；不上传权重、原始轨迹、数组、图像、视频或缓存；76个排除项见capture_inventory.json。
- 结果：上传前本地完整性验证通过；推送结果以GitHub远端提交SHA比对为准。TEST_LOG.md快照截点5939110字节，不覆写历史日志；此条是截点后的独立追加记录。MuJoCo仿真范围。
