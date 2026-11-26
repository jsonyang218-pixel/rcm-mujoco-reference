# 学习终末恢复最小实验 v1

流程完成；局部候选：True。学生末200全≤20µm：3/3；教师诊断：3/3。

训练：既有训练登记 E40/E41/E42；整目标隔离 E43。此处 ID 归属 homogeneous 20261123 collection registry，未冒用旧 frozen split ID。
模型 SHA256：`8d0679fc706c8a1a384977c59af5ec131e22e281afa9d02bd2f6625baccfbfc1`。12输入、39个学习系数/截距，标准化 ridge α=0.01，仅拟合一次。动作单位 dA[deg], dB[deg], dC[mm]/50ms。

| 起点 | 控制器 | 最长连续≤20周期 | 末200最大µm | 末步3D µm | 首次退出周期/秒 |
|---|---|---:|---:|---:|---|
|below_recede|student|232|6.535071|1.881930|无（未进入时见指标）|
|below_recede|teacher|234|3.052929|3.052929|无（未进入时见指标）|
|above_recede|student|233|6.394534|5.459913|无（未进入时见指标）|
|above_recede|teacher|234|3.039659|3.039659|无（未进入时见指标）|
|lateral|student|237|5.992781|2.148769|无（未进入时见指标）|
|lateral|teacher|231|19.057664|19.057664|无（未进入时见指标）|

所有原始失败与侧向挑战均保留。教师仅作诊断正对照，不计学生成功。未运行dev20、final_test或常规初始位置独立全程。
实际耗时 581.61s；单进程顺序执行，未进行超参搜索或后续扩展。

查看：summary.json、local_metrics.json、model.json、data_audit.json、segment_audit.json、reuse_audit.json、control_boundary_audit.json、host_preflight.json、workflow_status.json、trajectories/。
model.json 给出全部原单位/标准化系数与输入；local_metrics.json 给出首次退出动作，轨迹给出逐周期全部ABC动作。
