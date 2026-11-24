# Frozen-Backbone v5 three-hour sprint and dev20 validation

Status: `completed`. Base: `f49a3a27eb6debf2e03ba8776cb4387d407ada7f`. Branch: `experiment/frozen-backbone-v5-20261124`.

Scope: software and MuJoCo simulation only. This is not a clinical or hardware-readiness result.

## Budget and execution

- Hard budget: 180 minutes.
- No-new-task cutoff: T+165 minutes.
- Reserved close-out: 10 minutes.
- Elapsed: 149.27 minutes.

## Baseline and Phase 0

- Baseline 9-snapshot toward fraction: 0.3993; axis reverse counts: `{'A': 7, 'B': 4, 'C': 6}`.
- Baseline dev20 final/min mean: 58.8867/30.8695 um; <=30/10/5 hit rates: 1/20, 0/20, 0/20.
- Fallback F dev20 final/min mean: 51.5813/31.3898 um; <=30/10/5 hit rates: 0/20, 0/20, 0/20.
- Late 1000--1600 diagnosis: `oscillation`; action-norm ratio 0.9687, late-error ratio 0.9735, axis flip rates `{'A': 0.4307178631051753, 'B': 0.4187813021702838, 'C': 0.2638564273789649}`.

## Dataset sampling evidence

| Arm | Mode | T3 weight | Nonzero weight fraction | Empirical T3 stationary fraction | Theoretical weighted fraction |
|---|---|---:|---:|---:|---:|
| V5-A | `t3_only` | 0.2500 | 1.0000 | 0.9745 | 0.9661 |
| V5-B | `executed_only` | 0.2500 | 1.0000 | 0.9745 | 0.0893 |
| V5-C | `t3_only` | 0.1000 | 1.0000 | 0.9745 | 0.9272 |

`nonzero_weight_fraction` only proves that stationary frames retained a nonzero weight. It is not evidence that their empirical action share was calibrated.

## Phase 1 fast5

| Arm/step | Direction gate | Toward fraction | Reverse counts | Final mean um | Min mean um | Strict | <=30/10/5 | Drift um |
|---|---|---:|---|---:|---:|---:|---|---:|
| A/2000 | True | 0.4028 | `{'A': 7, 'B': 4, 'C': 6}` | 57.0910 | 33.6590 | 0/5 | 1/0/0 | 23.0800 |
| A/4000 | True | 0.4062 | `{'A': 7, 'B': 4, 'C': 6}` | 54.2218 | 32.5563 | 0/5 | 0/0/0 | 15.1955 |
| B/4000 | True | 0.4167 | `{'A': 6, 'B': 4, 'C': 6}` | 58.6424 | 44.7511 | 0/5 | 0/0/0 | 23.7890 |

## Top-1 and dev20

- Top-1: `V5-A` step 4000 (`/home/him1/mujoco_bc/rcm_4090_deploy/outputs/homog_label_fix_20261124_v5/homog_label_fix_20261124_v5_phase1_A/step_004000.pth`). Ranking: strict success descending, final mean ascending, min mean ascending, absolute drift ascending, toward fraction descending.
- dev20 status: `skipped_time`. Metric gate: `True`.
- Reason: fewer than 45.0 minutes remain before T+165

## Interpretation

Only relative development-target comparisons are reported. Failed gates were not retried or retuned, and no final-test or hardware claim is made.
