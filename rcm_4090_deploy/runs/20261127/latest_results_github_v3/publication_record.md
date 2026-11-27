

### 2026-11-27T11:03:22+08:00 最新冻结泛化状态GitHub归档验证（latest_results_github_v3）

- 授权/目的：用户“最新状态上传到github”；截点2026-11-27T11:02:15+08:00；基于experiment/output-head-sanity-20261125、父提交4c8fb3c518215491a9460e7b6b1c0bf6bd732d13。未启动训练、评估、推理、仿真或重试。
- 命令：python3 -E -B runs/20261127/latest_results_github_v3/verify_snapshot.py --snapshot-dir runs/20261127/latest_results_github_v3。
- 来源载荷545文件、37,510,336字节；捕获源逐字节与稳定性重新核对，归档JSON/JSONL、Python AST、文件大小/SHA和禁止文件/凭据模式检查通过；111个实验卡输入指纹复核通过。near/far实际模型文件流式SHA重算通过。
- 实验dit_v3_generalization_v1：冻结v3 candidate2 SHAdae7252fe5ee1af6a35d39974b395b674f9b6607175b8e6d984d5331812cd843 + far原v5 SHA6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11；训练更新0，10个不同空间目标G00–G09×2采样种子1926112701/1926112702，共20/20回合完成，各1600周期、50ms、K1/exec1/4flow步、原限幅/安全预览/G10、测量OCT≤100µm后单向近场切换，无历史重置；teacher/geometry在线调用均0。
- 结果：最后200动作后三维误差均≤20µm且含最终动作的保持门槛19/20（95%）；全部20回合最终误差均值16.410332µm，最大217.021852µm，最后200最大223.751665µm。唯一失败G02_s1926112702，近场动作0、far动作1599、启动零动作1、无无效后评分传感器周期；不删失败回合或更换目标。
- 对照/边界：不存在这些同目标历史v3基线，不作配对改善声明；是10个空间目标的20回合而非20独立目标，非SOP screen10/dev20或历史final_test重跑，也非原0.35µm停稳验收。仅有限冻结仿真证据。
- 数据/来源：冻结方案、目标分离与历史使用清单完整保留，history_inventory/history_points较大JSON作为来源审计上传；62个排除记录（含整sensor子树）见capture_inventory.json；权重、原始轨迹、数组、图像视频和缓存留本地。
- 输出runs/20261127/latest_results_github_v3/；主TEST_LOG固定6,688,664字节截点。本次元数据大小守卫误判已散列权重后，复用实际哈希并重核全部源字节，未触及实验；记录preparation_recovery.json。
- 状态：本地完整性验证通过；推送后另记远端SHA核对回执。软件/MuJoCo仿真范围。
