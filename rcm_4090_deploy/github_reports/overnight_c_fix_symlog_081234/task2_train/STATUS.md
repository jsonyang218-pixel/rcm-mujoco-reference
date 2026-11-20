# task2_train: completed

```json
{
  "checkpoint": "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth",
  "dataset": "data/plan2_plus_original_train_224_20261119",
  "validation": "data/lerobot_rcm_val_acthist_20261117_224",
  "overlay": "data/overnight_c_fix_symlog_081234_overlay/label_override.npz",
  "output": "outputs/overnight_c_fix_symlog_081234/task2_train",
  "seed": 20261118,
  "sampling_counts": {
    "original": 58880,
    "correction": 5120
  },
  "train_loss": 0.116777173448354,
  "val_loss": 0.14800261818676544,
  "batches_seen": 4000,
  "optimizer_updates": 3998,
  "amp_skipped_batches": 2,
  "replay_labels": "C-only overlay",
  "commands": [
    {
      "stage": "task2_train",
      "time": "2026-11-19T09:13:13.935069-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "lerobot_training.dit_experience_replay",
        "train",
        "--data-dir",
        "data/plan2_plus_original_train_224_20261119",
        "--val-data-dir",
        "data/lerobot_rcm_val_acthist_20261117_224",
        "--init-from",
        "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth",
        "--output-dir",
        "outputs/overnight_c_fix_symlog_081234/task2_train",
        "--mode",
        "ratio",
        "--correction-ratio",
        "0.08",
        "--epochs",
        "4",
        "--steps-per-epoch",
        "1000",
        "--checkpoint-every-steps",
        "1000",
        "--label-overlay",
        "data/overnight_c_fix_symlog_081234_overlay/label_override.npz"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/task2_train.log"
    }
  ]
}
```
