# eval_smoke: completed

```json
{
  "checkpoint": "outputs/state_attention_experience_ratio_20261118_1/r0_natural/training/step_004000.pth",
  "output": "runs/20261119/overnight_c_fix_symlog_081234/eval_smoke",
  "steps": 2,
  "arm": "I3",
  "target_id": "target_d30721d6bc45",
  "seed": 42,
  "latency_ms": [
    null,
    174.63177490234375
  ],
  "result_status": "passed",
  "commands": [
    {
      "stage": "eval_smoke",
      "time": "2026-11-19T08:12:41.009829-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix_eval",
        "--checkpoint",
        "outputs/state_attention_experience_ratio_20261118_1/r0_natural/training/step_004000.pth",
        "--target-manifest",
        "configs/v12_reachable_development_targets_0p35um.json",
        "--output-dir",
        "runs/20261119/overnight_c_fix_symlog_081234/eval_smoke",
        "--arm",
        "I3",
        "--smoke"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/eval_smoke.log"
    }
  ]
}
```
