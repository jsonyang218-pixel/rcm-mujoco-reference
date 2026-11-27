# Frozen vision terminal head v1：报告恢复与最小工程修复

恢复结论：实现完成、原流程实际启动、FP32 parity 通过；tinyfit 在固定 10 个 E40 窗口上已完成 512 次更新，但原科学门未通过。正式训练 0 次，offline 未执行，closedloop 跳过。原始 technical_failure 记录保留；恢复状态独立见 recovered_workflow_status.json。没有 20um 闭环结果，不能宣布成功或失败。

## 数值门与真实训练记录

逐条核对 tinyfit_updates.jsonl：512 条，update 1–512 连续；每条 batch_indices 均为 [244, 278, 312, 572, 606, 640, 0, 21, 329, 349]。与原 select_tiny 函数及原 manifest 交叉核对，6 个 receding 独立 episode 锚点，加 below/above 的 approaching/settled 各 1，共 10 窗口；全部 E40。该指标仅代表这 10 个窗口，不是全 657 个训练窗口指标，也不是 E41 offline。

| 更新 | normalized RMSE A/B/C | C MAE (um) | normalized MSE loss |
|---|---|---:|---:|
| 128 | 0.0566899702 / 0.0427899249 / 0.3714595139 | 0.6996613741 | 0.047675631940 |
| 256 | 0.0550001636 / 0.0387217253 / 0.3698408306 | 0.6936801076 | 0.047102216631 |
| 384 | 0.0541632250 / 0.0365148000 / 0.3606490195 | 0.6706553102 | 0.044778238982 |
| 512 | 0.0520442948 / 0.0330292471 / 0.1176008359 | 0.2584018111 | 0.005876499228 |

原门保持：各轴 normalized RMSE <= 0.1 且 C MAE <= 0.2um。

- 最终 A/B/C normalized RMSE：[0.05204429477453232, 0.03302924707531929, 0.11760083585977554]，轴门结果 true / true / false。
- 从保存 RMSE 独立平方复算各轴 normalized MSE：[0.002708608618578412, 0.0010909311623624879, 0.01382995659491787]；均值 0.005876498791953，与保存 loss 0.005876499228179 的差在 FP32 舍入范围。
- 最终物理 MAE A/B/C：[4.377235654828837e-06, 2.7378353024687385e-06, 0.00025840182206593454] deg/deg/mm。保存 C MAE 为 0.258401811123um；从保存 mm 值乘 1000 复算为 0.258401822066um，两者舍入一致，均大于 0.2um。
- 由原 scales 与 normalized RMSE 换算物理 RMSE：[5.204429345978285e-06, 3.3029246240930327e-06, 0.00027030047407347255] deg/deg/mm。
- updates 的 loss 是更新前值（第 512 次 0.005666422192007），train_metrics 是更新后固定全批量评估值；不把它们当同一时点。
- 这是保存聚合值的独立算术复核。没有最终逐窗口预测/残差，无法独立复算逐样本指标，不伪称已从最终模型重算。

## 技术错误与最小修复

真实 failure.json 的第一异常位于 pilot.py:300：p.detach().cpu()-initial[n]。initial 来自 CUDA head 的 state_dict 深拷贝；当前权重转 CPU 后相减触发设备错误。except 在第 313 行重复该错误，掩盖第一次诊断保存。第 304 行尚未执行的 tinyfit 失败保存有同样问题。

仅将三处 reporting/save 表达式中的 initial[n] 改为 initial[n].detach().cpu()，不改学习、数据、门、推理路径。修改前 SHA256：ba24cb9786dfb596ba7a1c29a1d2adce5c0f51a1157d2b4095823d8cb2a3917c；修改后：8eb20218759e2780da61857848dab4fa52041c3c52774dc6604a8ac1c473b4ba。最小 diff 为 pilot_reporting_only.diff，修复前源码及原 failure/summary/metrics 的副本位于 original_snapshot/。这修复的是报告及异常记录设备错误，不能称训练算法 bug 已修复，也不能保证未执行路径没有其他问题。现有 traceback 未显示另一个训练失败原因；未来路径未运行，不作无错担保。

验证只使用人工小张量：CPU 与 RTX 4090 CUDA 两种设备。旧版三处表达式在 CUDA case 全部复现设备错误；新版三处差值精确正确。隔离执行实际修复后的失败保存语句与 except handler，两份合成诊断文件保存后读回通过，原始合成异常仍被重新抛出。无网络实例、无优化器实例/更新、无 backward、无实际数据前向。pilot.py 只做语法检查，未导入或实际运行。合成文件仅在 synthetic_tensor_test/ 下，绝非原训练梯度或训练权重。

## 产物可恢复性与身份

initial_head.pth 存在，CPU 只读载入核对总元素 112323；spatial_softmax_cache.pt 两个 split 均为 [657,1536]。preflight、参数清单、shapes 与实验卡一致：新 head 112323 参数、20 状态维、1536 视觉维；parity 12 个实际窗口通过，覆盖两 split、两 side、三类。最大 feature 差 1.8998980522e-7，最大 normalized action 差 2.5324666808e-8，production_get_action 差为 0（均来自既有记录，未重跑）。

tinyfit_result.json、tinyfit_checkpoint.pth、tinyfit_failure_gradients_weights.pt、tinyfit_interrupted_gradients_weights.pt 均不存在。实际最终训练权重、梯度、优化器状态和权重差未保存，无法从初始权重/标量指标/缓存恢复。没有伪造张量，没有回放 512 更新。

原 input_hashes.json 的 243 项修复前全部匹配（包含生产源码、原权重、数据、任务输入及其来源）；原视觉 checkpoint SHA256：6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11。修复后 242 项匹配，唯一有意变化为任务 pilot.py 三处报告表达式。生产代码、视觉权重、数据、原门、原指标文件身份均未变。原有任务文件除 pilot.py 与追加的任务 TEST_LOG.md 外均逐字节哈希不变；部署根 TEST_LOG.md 也追加了恢复记录，不能声称所有文件未变。原 task_status/summary/failure/report 和 stale running 的 tinyfit_progress 原样保留，解释以独立恢复状态为准。

## 预算、记录与边界

沿用 900 秒：原 preparation 5.239828s + 原 runtime 5.929314s = 已用 11.169143s。budget_usage 的分阶段值已包含在 runtime 中，不重复计费。本恢复从 2026-11-26 14:24:17 UTC 起保守计入全部实际工作墙钟时间（包括读写），未计原运行结束到恢复开始的空闲间隔。恢复计费截点 240.625892s，余量 648.204965s；不新增候选、不重置预算。详见 recovery_budget.json。

每项检查完成后均立即追加任务及部署根 TEST_LOG 并精确读回；偏移及字节数见 test_log_readback_receipts.jsonl。恢复辅助脚本曾因缺失 pathlib.Path import 停在首次快照后；已补 import，只从未完成身份检查处继续，未覆盖已有快照、未运行科研流程。

本恢复新增训练更新 0、模型前向 0、网络动作采样 0、仿真 0、GNOME 启动 0、子代理 0、上传 0。

## 仅基于既有曲线的建议

保持原门并停在 tinyfit，不启动正式训练、offline 或闭环。128→512 的保存指标显示三轴 RMSE 和 C MAE 均下降，尤其 384→512 的 C 误差明显下降，但 512 时仍未达到原门。短训未过门不足以说明架构学不会，也不能据此断言泛化或闭环表现；目前只应归档这条曲线、数值门失败与独立技术报告错误，不自动增加训练步数、调整阈值或生成新候选。
