# Frontiers 2022 DDPG + DiT-Flow/RCM — 2026-11-18

## Scope

This branch adds an independent implementation of Algorithm 1 and Equations
(5)-(7) from Chen et al., *Deep Reinforcement Learning Based Trajectory
Planning Under Uncertain Constraints*, Frontiers in Neurorobotics 2022,
DOI `10.3389/fnbot.2022.883562`.

It is **not the authors' official source code** and is not a reproduction of
the paper's original PyBullet/Panda experiment. The actor is the project's
state-attention DiT rectified-Flow policy with one fixed Flow-noise tensor.
Training uses stored RCM transitions, a single Q network, the paper target
reward, and the deterministic policy gradient. The obstacle reward is disabled
because the RCM recordings do not contain obstacle distance.

## Result

The implementation and pipeline completed, but strict closed-loop acceptance
failed decisively.

| Metric | Replay v1 baseline | Frontiers DDPG |
|---|---:|---:|
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | 171.028 µm | **15,909.594 µm** |
| Final error median | 161.439 µm | **13,546.894 µm** |
| Final error minimum | 92.671 µm | 11,030.918 µm |
| Final error maximum | 269.203 µm | **23,944.568 µm** |
| Per-episode minimum mean | 77.841 µm | **12,334.938 µm** |
| Safety-limited steps | 0 | **108** |

All 20 paired targets worsened. Mean final error increased by
`15,738.567 µm`; median final error increased by `13,385.456 µm`. No target
entered the `0.35 µm` success gate, and only `5 / 20` episodes ended with the
encoder settled.

## Training diagnosis

The best checkpoint was selected at epoch 0 with validation Flow loss
`1.103461`. Validation loss then rose to `13.099834`, `13.174330`, and
`16.239620` on epochs 1-3. Policy-Q mean rose from `0.021241` to `0.673852`
while replay reward stayed approximately `-0.021`, and closed-loop final errors
remained in millimetres. This is consistent with a critic/value mismatch and
out-of-distribution actor extrapolation, but the experiment does not establish
causality for either failure mode.

## Verification

- Frontiers focused tests: 5 / 5 passed.
- Replay regression tests: 6 / 6 passed.
- Full training: 4 x 1,000 updates, 3,750 actor updates.
- Closed-loop protocol: 20 development targets, seed 42, 1,600 steps,
  0.05 s control period, one executed action per model call.
- Best checkpoint SHA-256:
  `5559f5b3f1b986aceeda6aab76d64694880d931a58f9c8b4e65124ffdc51be4d`.

## Evidence

- [Pipeline report](REPORT.md)
- [Paired baseline comparison](COMPARISON.md)
- [Machine-readable summary](summary.json)
- [Per-target metrics](episodes.csv)
- [All error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired error vs Replay v1](evaluation/plots/final_error_vs_replay_v1.png)
- [Training result](training/result.json)
- [Frontiers reward and DDPG notes](../../FRONTIERS2022_DDPG.md)

The 554 MB checkpoint and raw datasets are intentionally excluded from Git.
Results are from MuJoCo simulation only and do not establish hardware or
clinical performance.
