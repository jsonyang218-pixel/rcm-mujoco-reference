# DiT quantization recovery v3

{
  "workflow_status": "complete_strict_pass",
  "goal_achieved": true,
  "actual_start": {
    "time": "2026-11-27T09:26:09.184380+08:00",
    "pid": 374062,
    "ppid": 374059,
    "actual_start": true,
    "output_dir": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261127/dit_quantization_recovery_v3",
    "source": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261127/dit_relative_v2_resume",
    "budget_s": 2700,
    "concurrency_max": 12,
    "interpreter": "/home/him1/mujoco_bc/rcm_4090_deploy/.venv/bin/python",
    "launch_method": "GNOME new window, FIFO one-shot acknowledgement",
    "simulation_only": true
  },
  "simulation_only": true,
  "collection": {
    "completed": 92,
    "planned": 92,
    "only_targets": [
      "E40",
      "E42"
    ]
  },
  "initial_near_E41": {
    "candidate": "candidate1",
    "completed": 4,
    "strict20um_pass": 3,
    "denominator": 4,
    "ratios_last200_max_vs_saved_original_exec1": [
      0.31040702859279673,
      0.27935959330126725,
      0.24921073142201444,
      0.3205608024269592
    ],
    "substantial_improvement": true,
    "all_passed": false,
    "scope": "E41 development only; excluded from train and all scale estimation"
  },
  "repaired_near_E41": {
    "candidate": "candidate2",
    "completed": 4,
    "strict20um_pass": 4,
    "denominator": 4,
    "ratios_last200_max_vs_saved_original_exec1": [
      0.006881105544679177,
      0.005128783536681938,
      0.18698633383749264,
      0.014267720417893577
    ],
    "substantial_improvement": true,
    "all_passed": true,
    "scope": "E41 development only; excluded from train and all scale estimation"
  },
  "selected_candidate": "candidate2",
  "near_E41": {
    "candidate": "candidate2",
    "completed": 4,
    "strict20um_pass": 4,
    "denominator": 4,
    "ratios_last200_max_vs_saved_original_exec1": [
      0.006881105544679177,
      0.005128783536681938,
      0.18698633383749264,
      0.014267720417893577
    ],
    "substantial_improvement": true,
    "all_passed": true,
    "scope": "E41 development only; excluded from train and all scale estimation"
  },
  "standard_five": {
    "completed": 5,
    "denominator": 5,
    "strict20um_pass": 5,
    "cycles_each": 1600,
    "paired": {
      "E00": {
        "v2": {
          "completed_cycles": 1600,
          "endpoint_um": 46.567830434903776,
          "first_enter20_post_cycle": null,
          "last200_max_post_error_um": 46.567830434903776,
          "longest_le20_cycles": 0,
          "terminal_le20_cycles": 0,
          "last200_includes_final_post": true,
          "strict20um_pass": false
        },
        "v3": {
          "completed_cycles": 1600,
          "endpoint_um": 9.405437049497493,
          "first_enter20_post_cycle": 750,
          "last200_max_post_error_um": 15.326873446967616,
          "longest_le20_cycles": 851,
          "terminal_le20_cycles": 851,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        }
      },
      "E02": {
        "v2": {
          "completed_cycles": 1600,
          "endpoint_um": 15.421148206631083,
          "first_enter20_post_cycle": 666,
          "last200_max_post_error_um": 15.421148206631083,
          "longest_le20_cycles": 935,
          "terminal_le20_cycles": 935,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        },
        "v3": {
          "completed_cycles": 1600,
          "endpoint_um": 9.731133609426777,
          "first_enter20_post_cycle": 656,
          "last200_max_post_error_um": 17.09834143689879,
          "longest_le20_cycles": 941,
          "terminal_le20_cycles": 941,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        }
      },
      "E06": {
        "v2": {
          "completed_cycles": 1600,
          "endpoint_um": 31.04965603182882,
          "first_enter20_post_cycle": null,
          "last200_max_post_error_um": 31.04965603182882,
          "longest_le20_cycles": 0,
          "terminal_le20_cycles": 0,
          "last200_includes_final_post": true,
          "strict20um_pass": false
        },
        "v3": {
          "completed_cycles": 1600,
          "endpoint_um": 13.091027774278427,
          "first_enter20_post_cycle": 969,
          "last200_max_post_error_um": 13.091027774278427,
          "longest_le20_cycles": 632,
          "terminal_le20_cycles": 632,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        }
      },
      "E08": {
        "v2": {
          "completed_cycles": 1600,
          "endpoint_um": 38.592537304154675,
          "first_enter20_post_cycle": null,
          "last200_max_post_error_um": 38.592537304154675,
          "longest_le20_cycles": 0,
          "terminal_le20_cycles": 0,
          "last200_includes_final_post": true,
          "strict20um_pass": false
        },
        "v3": {
          "completed_cycles": 1600,
          "endpoint_um": 1.1099645285131694,
          "first_enter20_post_cycle": 662,
          "last200_max_post_error_um": 13.614177041238202,
          "longest_le20_cycles": 939,
          "terminal_le20_cycles": 939,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        }
      },
      "E10": {
        "v2": {
          "completed_cycles": 1600,
          "endpoint_um": 35.978458963537875,
          "first_enter20_post_cycle": null,
          "last200_max_post_error_um": 35.978458963537875,
          "longest_le20_cycles": 0,
          "terminal_le20_cycles": 0,
          "last200_includes_final_post": true,
          "strict20um_pass": false
        },
        "v3": {
          "completed_cycles": 1600,
          "endpoint_um": 11.101417510320822,
          "first_enter20_post_cycle": 771,
          "last200_max_post_error_um": 14.422003051101344,
          "longest_le20_cycles": 830,
          "terminal_le20_cycles": 830,
          "last200_includes_final_post": true,
          "strict20um_pass": true
        }
      }
    }
  },
  "time": "2026-11-27T09:55:29.320149+08:00",
  "elapsed_s": 1760.1358030170086,
  "batch_limit_s": 2700
}
