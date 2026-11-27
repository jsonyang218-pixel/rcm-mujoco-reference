# DiT relative v2 resume

{
  "workflow_status": "complete_standard_negative",
  "goal_achieved": false,
  "attempt": "20261127_083029",
  "simulation_only": true,
  "source": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/dit_relative_terminal_v2",
  "output_dir": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261127/dit_relative_v2_resume",
  "budget_s": 1919,
  "actual_start": {
    "time": "2026-11-27T08:30:29.589581+08:00",
    "pid": 367289,
    "ppid": 367286,
    "actual_start": true,
    "attempt": "20261127_083029",
    "output_dir": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261127/dit_relative_v2_resume",
    "source": "/home/him1/mujoco_bc/rcm_4090_deploy/runs/20261126/dit_relative_terminal_v2",
    "budget_s": 1919,
    "interpreter": "/home/him1/mujoco_bc/rcm_4090_deploy/.venv/bin/python",
    "concurrency_max": 12,
    "old_diagnostics_skipped": true
  },
  "near_E41": {
    "candidate": "candidate1",
    "completed": 4,
    "strict20um_pass": 4,
    "denominator": 4,
    "ratios_last200_max_vs_saved_original_exec1": [
      0.03339543033427413,
      0.021903307413796314,
      0.1129111947290396,
      0.10017771276823297
    ],
    "substantial_improvement": true,
    "all_passed": true,
    "scope": "E41 development only; excluded from train and all scale estimation"
  },
  "C_bias": {
    "old_four_same_side": true,
    "new_four_same_side": true,
    "disappearance_of_74um_C_bias": true,
    "interpretation": "Bias disappearance requires bounded independent postXYZ, not merely changed mean command; joint data/input feasibility only.",
    "arms": {
      "below_approaching": {
        "new": {
          "last200_mean_error_xyz_um": [
            1.2161919159439094,
            0.12172675551980205,
            -1.6103324028904353
          ],
          "last200_mean_OCT_xyz_um": [
            -0.14172746159601957,
            0.03815062518697232,
            -1.9347526831552386
          ],
          "last200_mean_accepted_C_um": 0.31022786686662585,
          "last200_max_postXYZ_um": 2.0216603639621513
        },
        "old": {
          "last200_mean_error_xyz_um": [
            -44.62074086478509,
            4.4616273934095485,
            58.69469581997947
          ],
          "last200_mean_OCT_xyz_um": [
            12.545244232751429,
            -3.2505910308100283,
            72.79956746846437
          ],
          "last200_mean_accepted_C_um": -0.06695321337019777,
          "last200_max_postXYZ_um": 75.45723938536605
        },
        "measured_C_zero_crossings": [
          {
            "cycle": 23,
            "OCT_C_before_um": 0.5048359162174165,
            "OCT_C_after_um": -0.4059106868226081,
            "accepted_C_before_um": 0.9402259020134807,
            "accepted_C_after_um": 0.6567751406691968
          }
        ]
      },
      "below_receding": {
        "new": {
          "last200_mean_error_xyz_um": [
            0.8564901910403933,
            0.15578378443237284,
            -1.1370934370233954
          ],
          "last200_mean_OCT_xyz_um": [
            -0.02044262146227993,
            0.0023810614493413595,
            -1.3439985923469067
          ],
          "last200_mean_accepted_C_um": 0.3965445645735599,
          "last200_max_postXYZ_um": 1.4320703611771872
        },
        "old": {
          "last200_mean_error_xyz_um": [
            -44.46782030256543,
            4.447148674933423,
            58.4935069912018
          ],
          "last200_mean_OCT_xyz_um": [
            12.50511787366122,
            -3.2418824429623783,
            72.55567789077759
          ],
          "last200_mean_accepted_C_um": -0.04055002540553687,
          "last200_max_postXYZ_um": 74.95585298848292
        },
        "measured_C_zero_crossings": [
          {
            "cycle": 25,
            "OCT_C_before_um": 0.14478024968411773,
            "OCT_C_after_um": -0.1068896526703611,
            "accepted_C_before_um": 0.753018248360604,
            "accepted_C_after_um": 0.8363726665265858
          }
        ]
      },
      "above_approaching": {
        "new": {
          "last200_mean_error_xyz_um": [
            4.472439108951676,
            -0.1865789798980444,
            -5.894390409537209
          ],
          "last200_mean_OCT_xyz_um": [
            -1.0477072792127728,
            0.28927274979650974,
            -7.266995497047901
          ],
          "last200_mean_accepted_C_um": -0.5685053349588998,
          "last200_max_postXYZ_um": 7.4014432105545795
        },
        "old": {
          "last200_mean_error_xyz_um": [
            -44.344527803463336,
            4.435475180154587,
            58.33129810882241
          ],
          "last200_mean_OCT_xyz_um": [
            12.472749333828688,
            -3.2320792856626213,
            72.34947919845581
          ],
          "last200_mean_accepted_C_um": -0.006351112283482507,
          "last200_max_postXYZ_um": 75.7236009221961
        },
        "measured_C_zero_crossings": []
      },
      "above_receding": {
        "new": {
          "last200_mean_error_xyz_um": [
            4.1222032190262325,
            -0.15341818858871648,
            -5.433605100733985
          ],
          "last200_mean_OCT_xyz_um": [
            -0.9372216300107539,
            0.23442809469997883,
            -6.70670997351408
          ],
          "last200_mean_accepted_C_um": -0.46719659309019335,
          "last200_max_postXYZ_um": 6.822034953755547
        },
        "old": {
          "last200_mean_error_xyz_um": [
            -44.620030927159746,
            4.461560175589092,
            58.693761795699686
          ],
          "last200_mean_OCT_xyz_um": [
            12.548001888208091,
            -3.2524434698279947,
            72.79898315668106
          ],
          "last200_mean_accepted_C_um": -0.05101335977542476,
          "last200_max_postXYZ_um": 75.34756110823923
        },
        "measured_C_zero_crossings": []
      }
    }
  },
  "standard_five": {
    "completed": 5,
    "denominator": 5,
    "cycles_each": 1600,
    "strict20um_pass": 1
  },
  "time": "2026-11-27T08:51:45.319276+08:00",
  "elapsed_s": 1275.7297055920353
}
