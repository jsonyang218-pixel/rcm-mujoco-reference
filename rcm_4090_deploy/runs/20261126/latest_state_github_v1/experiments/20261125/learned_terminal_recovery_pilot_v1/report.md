# 学习终末恢复最小实验 v1 — GNOME 启动失败

2026-11-25 22:41:48 +08:00，GNOME 返回 `Cannot open display`（exit 1）。未启动真实数据补采、ridge 拟合或闭环；没有新模型、模型哈希或学生精度结果。未运行 dev20 / final_test，未发布，未覆盖结果。

已完成：代码与实验卡；既有完全相同数据/评测复用核查；8 段既有教师执行数据的动作/时序/实际执行后状态审计。训练目标来自独立登记的 homogeneous 20261123 collection：E40/E41/E42；隔离目标 E43。没有将其冒称为旧 frozen split 的目标 ID。

计划固定：补采10条40周期纯训练教师段；共18训练段（含原始侧向恢复与稀疏驻留）；标准化12输入、39系数ridge，α=0.01，只拟合一次。E43三个起点分别学生/教师各240周期。全部末200≤20µm且教师可行才记局部候选。未完成的计划不算实验结果。

失败证据：launch_receipt.json、workflow_status.json、launch_failure_report.json。复用证据：reuse_audit.json、input_fingerprints.json。输入与隔离：experiment_card.json。代码：pilot.py、student_policy.py、launch.sh、launch_once.py。

按项目SOP，GNOME打不开则停止，不后台回退、不自动重试。当前运行命名空间的 nvidia-smi 亦不能连接驱动；宿主GPU/进程预检未执行，不能声称已经核对宿主运行进程。未对R2 fast5和r2_hold220_planner_v1启动、停止或写入。
