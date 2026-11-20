# Overnight C-label / OCT residual experiment

**Status: partial.** Tasks 1 and 2 completed. The 7-hour time limit interrupted Task 3 during its first target; Tasks 4 and 5 did not start.

The full audit reproduced all 11,400 eligible original labels. Of those, 11,029 (96.75%) met the acceleration-limited drag criterion, exceeding the 60% gate. The best cleaned-replay checkpoint (4,000 batches, 8% replay) reduced the mean final error from 247.94 to 53.95 um (78.2% lower than frozen R0), with improvements on all 20 paired targets. No target met the 0.35 um and three-consecutive-settled-frames success criterion.

Local evidence index: `runs/20261119/overnight_c_fix_symlog_081234`. Checkpoints, datasets, and raw logs are not included. MuJoCo simulation only.

- [preflight](preflight/STATUS.md): completed
- [eval_smoke](eval_smoke/STATUS.md): completed
- [task1](task1/STATUS.md): completed; reproduced 11400/11400 original labels; drag fraction 96.75%
- [task2_overlay](task2_overlay/STATUS.md): completed
- [task2_train](task2_train/STATUS.md): completed; labels: C-only overlay; 3998/4000 optimizer updates (AMP)
- [task2_step2000](task2_step2000/STATUS.md): completed
- [task2_step4000](task2_step4000/STATUS.md): completed
- [task2_selection](task2_selection/STATUS.md): completed; selected `outputs/overnight_c_fix_symlog_081234/task2_train/step_004000.pth` (replay best 53.950 um; R0 247.936 um)
- [task3_I0](task3_I0/STATUS.md): interrupted at the experiment deadline; no complete evaluation. There are no formal I1/I3 comparison or OCT symlog results.

## Complete closed-loop evaluations (E00-E19, seeds 42-61)

All rows use I0 and a 1,600-step budget per target. The latency metrics cover GPU sampling and aggregation, not rendering or host transfer.

| Stage | Final mean (um) | Mean trajectory minimum (um) | A/B frame jitter (deg) | Settled at exit | GPU median / P90 (ms) |
|---|---:|---:|---:|---:|---:|
| Frozen R0 | 247.936 | 67.474 | n/a | 0/20 | n/a |
| [Replay 2,000 batches](task2_step2000/summary.json) | 61.995 | 31.201 | 0.00648/0.00380 | 0/20 | 9.06 / 9.32 |
| [Replay 4,000 batches](task2_step4000/summary.json) | 53.950 | 30.397 | 0.00638/0.00372 | 0/20 | 9.10 / 9.32 |

[2,000-batch target table](task2_step2000/episodes.csv) · [4,000-batch target table](task2_step4000/episodes.csv) · [4,000-batch error curves](task2_step4000/error_curves_all.png) · [4,000-batch best and final error](task2_step4000/best_and_final_error.png)

## Needle-tip trajectories

These figures compare frozen R0 and both cleaned-replay checkpoints using the existing I0 rollouts. CPU forward kinematics reconstructed 96,060 recorded poses, including each final pose. All reconstructed distances and final tip coordinates matched the evaluation records. No rollouts were rerun.

- **E00-E19 3D overview:** target-relative coordinates in millimeters, with a shared scale and view. [PNG](trajectories/needle_tip_trajectories_3d.png) | [PDF](trajectories/needle_tip_trajectories_3d.pdf)
- **E08, E00, E02, and E06 close-ups:** equal-scale projections within +/-150 micrometers on each axis, alongside the 3D error over the final 20 seconds. [PNG](trajectories/needle_tip_near_target.png) | [PDF](trajectories/needle_tip_near_target.pdf)

## C retreat 80 steps after each trajectory minimum (um)

Positive means retraction. `n/a` means no comparable 80-step observation.

| Stage | E08 | E00 | E02 | E06 |
|---|---:|---:|---:|---:|
| Frozen reference: r0 | 119.33 | 71.49 | n/a | n/a |
| Frozen reference: plan2 | 414.05 | 475.10 | n/a | n/a |
| task2_step2000 | n/a | 26.06 | -0.74 | 20.21 |
| task2_step4000 | n/a | n/a | 42.54 | 4.89 |

The 2,000-batch model reduced E00 retreat relative to R0. For the selected 4,000-batch model, E08 and E00 reached their minima too late to measure the next 80 steps; this is not evidence that retreat was eliminated. See the [2,000-batch](task2_step2000/c_min_aligned.png) and [4,000-batch](task2_step4000/c_min_aligned.png) diagnostic plots.
