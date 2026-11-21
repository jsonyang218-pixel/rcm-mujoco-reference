# GPU 并行 P2 诊断：完整结果与原始证据

本轮三组策略和最终比较已完成，三组共约32分50秒；每组各5目标×1000步。
并行流程完成；本轮P2平均误差未改善；严格成功每组0/5。三项结论分别记录。

| 变体 | 控制 | 平均终值误差 µm | 最大误差 µm | 严格成功 |
|---|---|---:|---:|---:|
| V0 | G10 feedback（原反馈基线） | 63.137605 | 72.408199 | 0/5 |
| V3 | G10 continuous（P2） | 117.733648 | 172.836239 | 0/5 |
| V2 | G15 continuous（P2） | 100.457812 | 186.376458 | 0/5 |

按预登记规则，V0是本轮均值最低的变体。不能把流程执行成功称为控制精度提升，
也不能把严格成功未下降（0仍为0）解释为达到0.35µm稳定定位。
本轮没有训练和微调；历史约54µm结果不是本轮同协议配对基线。

完整数据保持目录结构：

- `runs/20261121/control_p2_parallel_v1/parallel/s3_compare/`：最终JSON、逐目标CSV、两张比较图。
- `runs/20261121/control_p2_parallel_v1/parallel/s3_v0/`、`s3_v3/`、`s3_v2/`：15个episode原始JSON、逐拍电机JSONL和日志。
- `runs/20261121/control_p2_parallel_v1/parallel/reuse_receipt.json`：教师与冒烟阶段复用凭据，逐文件SHA-256。
- `runs/20261121/control_p2_diagnostics_v3/`：完整教师原始来源及因并行迁移中断的串行策略；不混入并行统计。
- `runs/20261121/control_p2_diagnostics_v2/`：此前30分钟超时现场。
- `runs/20261121/diagnostics/`、`runs/20261121/verification/p2_parallel_preflight_v1/`：状态诊断、预检、调度器验证及授权启动记录。
- `scripts/research/control_p2_parallel.py`：三独立进程CUDA并行调度器。
- `configs/experiment_card_control_p2_parallel_20261121_v1.json`：可审查执行卡。
- `docs/CONTROL_P2_GPU_PARALLEL_20261121.md`：执行范围、复用、预算和停止条件。

教师（每组5目标×1600步）：G10 RGB-D均值1.700127µm；G15 RGB-D均值2.525820µm；
两组Gate2b通过，严格成功均0/5。教师结果不等同于学习策略表现。
与Oracle比较时，策略使用同1000步前缀，而非1600步终值。

固定目标0/2/6/8/10，种子42/44/48/50/52，50ms控制周期，Flow4/单样本/执行首动作。
三组进程共享物理GPU0进行CUDA推理与EGL渲染；CPU几何与标量驱动不变。
并行资源竞争会影响延迟；没有完整配对串行计时，不能声称实测三倍加速。
这是fast5诊断，不是正式screen10/dev20或硬件精度验证。

固定414733455字节checkpoint本次不上传，路径和SHA-256保存在pinned_assets.json。
原始日志保持字节；TEST_LOG.md保留GitHub前缀并追加本地分叉记录，
local_TEST_LOG.md保存本地原始字节。
publication_manifest.json记录其余全部上传文件的大小与SHA-256，不包含自身。
