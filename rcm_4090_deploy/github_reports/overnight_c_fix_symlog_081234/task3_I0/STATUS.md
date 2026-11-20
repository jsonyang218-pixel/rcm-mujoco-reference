# task3_I0: interrupted

```json
{
  "output": "runs/20261119/overnight_c_fix_symlog_081234/task3_I0",
  "checkpoint": "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth",
  "arm": "I0",
  "protocol": "development E00-E19, seed 42+index, 1600 steps, 0.35 um",
  "reason": "Command '['.venv/bin/python', '-m', 'scripts.research.overnight_cfix_eval', '--checkpoint', 'outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth', '--target-manifest', 'configs/v12_reachable_development_targets_0p35um.json', '--output-dir', 'runs/20261119/overnight_c_fix_symlog_081234/task3_I0', '--arm', 'I0']' timed out after 67.55822070105933 seconds",
  "commands": [
    {
      "stage": "task3_I0",
      "time": "2026-11-19T15:11:26.573224-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix_eval",
        "--checkpoint",
        "outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth",
        "--target-manifest",
        "configs/v12_reachable_development_targets_0p35um.json",
        "--output-dir",
        "runs/20261119/overnight_c_fix_symlog_081234/task3_I0",
        "--arm",
        "I0"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/task3_I0.log"
    }
  ]
}
```
