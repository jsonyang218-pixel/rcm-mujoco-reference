# task2_step2000: completed

```json
{
  "output": "runs/20261119/overnight_c_fix_symlog_081234/task2_step2000",
  "checkpoint": "outputs/overnight_c_fix_symlog_081234/task2_train/step_002000.pth",
  "arm": "I0",
  "protocol": "development E00-E19, seed 42+index, 1600 steps, 0.35 um",
  "target_count": 20,
  "base_seed": 42,
  "max_steps": 1600,
  "error_threshold_um": 0.35,
  "flow_steps": 4,
  "samples": 1,
  "final_mean_um": 61.994802515134914,
  "final_median_um": 61.33628736511069,
  "final_min_um": 36.36983277166861,
  "minimum_mean_um": 31.20141174197486,
  "canonical_success_count": 0,
  "encoder_settled_at_exit_count": 0,
  "encoder_settled_near_target_frames": 1,
  "inference_median_ms": 9.062160015106201,
  "inference_p90_ms": 9.321791648864746,
  "inference_timing": "CUDA events: sample and aggregate, excluding initial history fill and host transfer",
  "near_target_ab_mean_change_deg": [
    0.006476823841017499,
    0.0038017866548822592
  ],
  "near_target_ab_pairs": 18720,
  "c_retraction": {
    "E08": {
      "minimum_step": 1579,
      "C_residual_at_min_um": -15.417708285369258,
      "C_retreat_after_80_steps_um": null,
      "post_min_negative_commands": 19,
      "post_min_under_target_negative_commands": 19,
      "post_min_command_frames": 21
    },
    "E00": {
      "minimum_step": 1002,
      "C_residual_at_min_um": -29.205902686911145,
      "C_retreat_after_80_steps_um": 26.056553738381893,
      "post_min_negative_commands": 51,
      "post_min_under_target_negative_commands": 51,
      "post_min_command_frames": 80
    },
    "E02": {
      "minimum_step": 1499,
      "C_residual_at_min_um": -22.625173540724575,
      "C_retreat_after_80_steps_um": -0.7364149282658161,
      "post_min_negative_commands": 33,
      "post_min_under_target_negative_commands": 33,
      "post_min_command_frames": 80
    },
    "E06": {
      "minimum_step": 1183,
      "C_residual_at_min_um": -45.17200506780128,
      "C_retreat_after_80_steps_um": 20.21223952045048,
      "post_min_negative_commands": 46,
      "post_min_under_target_negative_commands": 46,
      "post_min_command_frames": 80
    }
  },
  "result_status": "completed",
  "commands": [
    {
      "stage": "task2_step2000",
      "time": "2026-11-19T09:15:56.190720-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix_eval",
        "--checkpoint",
        "outputs/overnight_c_fix_symlog_081234/task2_train/step_002000.pth",
        "--target-manifest",
        "configs/v12_reachable_development_targets_0p35um.json",
        "--output-dir",
        "runs/20261119/overnight_c_fix_symlog_081234/task2_step2000",
        "--arm",
        "I0"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/task2_step2000.log"
    }
  ]
}
```
