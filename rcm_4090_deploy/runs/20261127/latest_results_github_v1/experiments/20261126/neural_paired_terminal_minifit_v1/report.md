# 配对末端数据与原神经最小训练验收

**数据门：passed；训练：completed（真实optimizer updates=128）；离线改善：failed。**

流程状态：completed_offline_failed。仅诊断；没有执行神经闭环、dev5/dev20或20µm神经成功验收。
E40为训练登记，E41为已经参与选择的整目标开发留出，不是全新独立测试。原网络、教师、驱动未改；初始化只用6cc389df原权重。

上轮history假设已否定：初始4/100远离输入被保留，旧outcome分类掩盖输入方向；尾15步不能作为首窗但仍为未来监督。本轮分别记录输入实际方向与执行后结果。

配对仅使用预注册24条真实前缀的首个教师时刻，30/50/70µm；门禁不放宽、不搜索。每时间戳真实调用原传感器；静止画面像素可以相同，但没有复制帧充当未来。
物理保持是独立XYZ误差≤20µm且所有实际编码器不动；非零受理C的停稳单列，不能冒称原strict hold，也不改教师标签。

训练首窗按接近/远离/停稳各1/3，再按侧、episode均衡，有放回。重复抽样不计作独立采集。数据量、覆盖和采样均发生改变，不能宣称单因素因果。

实际有效首动作窗数量：{"E40": {"approaching": 205, "settled": 446, "receding": 6}, "E41": {"approaching": 210, "settled": 441, "receding": 6}}

|E41开发留出门禁|before|after|通过|
|---|---:|---:|---|
|E41_equal_motion_C_MAE_um|1.782976886|1.781243252|True|
|E41_receding_C_MAE_um|2.182694579|2.258334309|False|
|E41_nonzero_teacher_C_direction|0.7066210046|0.6270928463|False|
|E41_all_A_MAE_deg|0.004375809155|0.005963873618|False|
|E41_all_B_MAE_deg|0.003248998283|0.003338244117|False|
|E41_settled_C_MAE_um|1.594160707|1.429499328|True|

训练及开发各组ABC MAE/C方向准确率、所有4seed及flow loss独立存于offline；flow loss不作为晋级门。

## 产物

- 数据/时间/配对/保持：data/、raw_state_evidence.csv/json、paired_anchor_evidence.csv/json、hold_gate_evidence.json、all_alignment_checks.json。
- 窗口与身份：window_manifest.json、real_input_windows.pt、input_fingerprints.json、input_hash_verification.json。
- 神经链路：new_data_input_first_action_parity.json、parameters.json、before.pth、after.pth、training.jsonl、training_result.json。
- 离线：offline/及offline_*_per_window_seed.csv；空组显式0/null。
- 启动与日志：experiment_card.json、gnome_launch_receipt.json、actual_start_receipt.json、terminal.log、TEST_LOG.md、test_log_receipts.jsonl。

失败阶段后未生成的产物保持缺失；以task_status/summary真实状态为准。每个报告阶段的TEST_LOG写入都在锁内fsync并立即按字节读回。
只验证软件与MuJoCo，不代表临床或真实硬件结论。离线通过也只建议下一轮，不执行闭环；离线负结果不证明原架构不可能。

机器可读完整结果：
```json
{
  "status": "completed_offline_failed",
  "time": "2026-11-26T15:06:53.023056+08:00",
  "data_gate": "passed",
  "training": "completed",
  "offline_improvement": "failed",
  "optimizer_updates": 128,
  "closed_loop_started": false,
  "closed_loop_authorized": false,
  "diagnostic_only": true,
  "neural_20um_claim": false,
  "development_not_independent_test": true,
  "distribution_and_data_both_changed": true,
  "unrequested_retry": false,
  "offline_gate": {
    "status": "failed",
    "time": "2026-11-26T15:06:52.518629+08:00",
    "checks": [
      {
        "name": "E41_equal_motion_C_MAE_um",
        "before": 1.7829768858583912,
        "after": 1.781243252445988,
        "passed": true,
        "required": "after < before"
      },
      {
        "name": "E41_receding_C_MAE_um",
        "before": 2.1826945788537464,
        "after": 2.258334308862686,
        "passed": false,
        "required": "after < before"
      },
      {
        "name": "E41_nonzero_teacher_C_direction",
        "before": 0.70662100456621,
        "after": 0.6270928462709284,
        "passed": false,
        "required": "after >= before"
      },
      {
        "name": "E41_all_A_MAE_deg",
        "before": 0.004375809155153353,
        "after": 0.0059638736176007194,
        "passed": false,
        "required": "after <= before"
      },
      {
        "name": "E41_all_B_MAE_deg",
        "before": 0.00324899828322466,
        "after": 0.003338244116706579,
        "passed": false,
        "required": "after <= before"
      },
      {
        "name": "E41_settled_C_MAE_um",
        "before": 1.5941607070793218,
        "after": 1.429499327991783,
        "passed": true,
        "required": "after <= before"
      }
    ],
    "numeric_tolerance": 0.0,
    "E41_is_development_not_independent_test": true,
    "closed_loop_allowed": false,
    "flow_loss_not_gate": true
  },
  "teacher_20um_is_data_acceptance_only": true,
  "next_step": "只建议下一轮；本轮不做闭环",
  "artifacts": {
    "data_gate.json": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/neural_paired_terminal_minifit_v1/data_gate.json",
    "raw_data_gate.json": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/neural_paired_terminal_minifit_v1/raw_data_gate.json",
    "training_result.json": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/neural_paired_terminal_minifit_v1/training_result.json",
    "offline_gate.json": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/neural_paired_terminal_minifit_v1/offline_gate.json",
    "input_hash_verification.json": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/neural_paired_terminal_minifit_v1/input_hash_verification.json"
  },
  "actual_data_counts": {
    "E40": {
      "approaching": 205,
      "settled": 446,
      "receding": 6
    },
    "E41": {
      "approaching": 210,
      "settled": 441,
      "receding": 6
    }
  },
  "training_result": {
    "status": "completed",
    "time": "2026-11-26T15:05:50.828620+08:00",
    "optimizer_updates": 128,
    "attempts": 129,
    "AMP_skips": 1,
    "optimizer_state_steps_min": 128,
    "optimizer_state_steps_max": 128,
    "seed": 20261126,
    "before_sha256": "6cc389df3786f1f34c43cf65d6dc17fe910ab342fae8d84c8e37cea6a7db6c11",
    "after_sha256": "e35d0a1cf8a7769f3170d0028c09542c36562d7fbbe8ff8ad47e3c47d4a50090",
    "frozen_vision_tensors_bit_exact": 62,
    "all_normalization_buffers_unchanged": true,
    "successful_draw_class_counts": {
      "settled": 175,
      "receding": 154,
      "approaching": 183
    },
    "successful_draw_class_proportions": {
      "approaching": 0.357421875,
      "receding": 0.30078125,
      "settled": 0.341796875
    },
    "successful_draw_episode_counts": {
      "E40_above_30_approaching": 100,
      "E40_below_30_receding": 46,
      "E40_above_70_receding": 31,
      "E40_below_30_approaching": 106,
      "E40_below_70_receding": 48,
      "E40_below_50_receding": 37,
      "E40_above_70_approaching": 18,
      "E40_above_50_approaching": 14,
      "E40_above_30_receding": 39,
      "E40_above_50_receding": 44,
      "E40_below_50_approaching": 9,
      "E40_below_70_approaching": 20
    },
    "independent_collected_windows": 657,
    "draws_are_not_independent_new_data": true,
    "sampler_trace_replay": "exact",
    "initial_flow_loss": 0.027012770995497704,
    "final_flow_loss": 0.005084624979645014
  },
  "total_runtime_including_preflight_s": 402.9492959129857,
  "budget_accounting": [
    {
      "time": "2026-11-26T15:04:48.443793+08:00",
      "stage": "collection_and_data_gate",
      "elapsed_s": 273.7329872759874,
      "registered_cap_s": 600
    },
    {
      "time": "2026-11-26T15:05:48.557890+08:00",
      "stage": "offline_before",
      "elapsed_s": 60.113855581963435,
      "registered_cap_s": 180
    },
    {
      "time": "2026-11-26T15:05:50.976359+08:00",
      "stage": "training",
      "elapsed_s": 2.4182210119906813,
      "registered_cap_s": 120
    },
    {
      "time": "2026-11-26T15:06:52.518541+08:00",
      "stage": "offline_after",
      "elapsed_s": 61.527575849962886,
      "registered_cap_s": 119.88608007505536
    }
  ]
}
```
