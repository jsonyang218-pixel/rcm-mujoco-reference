# preflight: completed

```json
{
  "output": "runs/20261119/overnight_c_fix_symlog_081234/preflight.json",
  "dataset": "data/plan2_plus_original_train_224_20261119",
  "checkpoints": [
    "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth",
    "outputs/state_attention_experience_ratio_20261118_1/r0_natural/training/step_004000.pth"
  ],
  "gpu": "NVIDIA GeForce RTX 4090",
  "assets": {
    "data/plan2_plus_original_train_224_20261119/data/dataset.npz": "9d80523f519faf4e00bbaef3d30e29dec1401181f3dcddd438c5087c05eb54db",
    "data/plan2_plus_original_train_224_20261119/metadata.json": "7f02f265acadf965a9dc0de29b2058cfccc240ea39d06dc88050179d0ae2698d",
    "outputs/state_attention_experience_ratio_20261118_1/r0_natural/training/step_004000.pth": "b504146485c38c1c26bd960d8aa831beba0df7e564f1f20267862a79b7dc1733",
    "outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth": "d1e2527fe087ad1f619450bdd83eac24e63d1e3a0d96186dc2cdd849fdfbc424",
    "configs/v12_reachable_development_targets_0p35um.json": "fcc4872a9d5b58bb7158b36d3c0ab6d7452e4d8d747317dfb077b6e7bbc815c7"
  },
  "original_windows": 220816,
  "correction_and_plan2_windows": 80818,
  "window_diagnostics": {
    "candidate_windows": 319681,
    "trajectory_boundary_excluded": 16599,
    "source_frame_gap_excluded": 1420,
    "time_interval_excluded": 0,
    "invalid_input_excluded": 28,
    "valid_windows": 301634
  },
  "overlay_arrays_isolated": true,
  "correction_samples_per_epoch": [
    1280,
    1280,
    1280,
    1280
  ],
  "total_samples_per_epoch": 16000,
  "r0_final_mean_um": 247.93630506807492,
  "c_reference_path": "outputs/state_attention_v1_c_retraction_20261119_1/results.json",
  "c_reference_retreat_um": {
    "r0": {
      "E08": 119.33055518430535,
      "E00": 71.49492165257598
    },
    "plan2": {
      "E08": 414.0532105173875,
      "E00": 475.09730754969223
    }
  },
  "result_status": "PASS",
  "commands": [
    {
      "stage": "preflight",
      "time": "2026-11-19T08:12:34.131643-07:00",
      "argv": [
        ".venv/bin/python",
        "-m",
        "scripts.research.overnight_cfix",
        "preflight",
        "--output-dir",
        "runs/20261119/overnight_c_fix_symlog_081234"
      ],
      "log": "runs/20261119/overnight_c_fix_symlog_081234/preflight.log"
    }
  ]
}
```
