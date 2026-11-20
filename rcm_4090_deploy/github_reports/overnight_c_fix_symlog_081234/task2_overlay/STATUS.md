# task2_overlay: completed

```json
{
  "output": "data/overnight_c_fix_symlog_081234_overlay/label_override.npz",
  "dataset": "data/plan2_plus_original_train_224_20261119",
  "source_dataset": "data/plan2_plus_original_train_224_20261119",
  "audit": "runs/20261119/overnight_c_fix_symlog_081234/task1/contamination_audit.json",
  "updated_action_rows": 11029,
  "executed_action_unchanged": true,
  "valid_windows_2x16": 301634,
  "commands": [
    {
      "stage": "task2_overlay",
      "time": "2026-11-19T09:13:13.569757-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix_data",
        "prepare-clean",
        "--dataset-dir",
        "data/plan2_plus_original_train_224_20261119",
        "--output-dir",
        "data/overnight_c_fix_symlog_081234_overlay",
        "--audit-csv",
        "runs/20261119/overnight_c_fix_symlog_081234/task1/contamination_audit.csv",
        "--audit-json",
        "runs/20261119/overnight_c_fix_symlog_081234/task1/contamination_audit.json"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/task2_overlay.log"
    }
  ]
}
```
