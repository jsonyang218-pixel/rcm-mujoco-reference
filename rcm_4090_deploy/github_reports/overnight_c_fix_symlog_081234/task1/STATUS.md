# task1: completed

```json
{
  "dataset": "data/plan2_plus_original_train_224_20261119",
  "output": "runs/20261119/overnight_c_fix_symlog_081234/task1",
  "candidate_count": 11400,
  "replayed_count": 11400,
  "reproduced_count": 11400,
  "dragged_negative_fraction": 0.9674561403508772,
  "threshold_reached": true,
  "failure_count": 0,
  "commands": [
    {
      "stage": "task1",
      "time": "2026-11-19T08:12:46.184653-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix_data",
        "audit",
        "--dataset-dir",
        "data/plan2_plus_original_train_224_20261119",
        "--output-dir",
        "runs/20261119/overnight_c_fix_symlog_081234/task1"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/task1.log"
    }
  ]
}
```
