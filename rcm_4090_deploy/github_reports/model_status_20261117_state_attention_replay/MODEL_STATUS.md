# State-attention replay actor-critic — 2026-11-17

## Result

Replay actor-critic is the strongest final-error model so far, but it still
does not meet the strict terminal success requirement.

| Metric | v1 baseline | Replay |
|---|---:|---:|
| Best validation loss | 0.149865 | **0.148059** |
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | 434.397 µm | **171.028 µm** |
| Final error median | 258.933 µm | **161.439 µm** |
| Final error maximum | 2128.198 µm | **269.203 µm** |
| Per-episode minimum mean | 93.680 µm | **77.841 µm** |
| Final error ≤ 300 µm | 12 / 20 | **20 / 20** |
| Paired final error | — | 19 improved, 1 worsened |
| Safety-limited steps | 0 | 0 |

The model never crosses the 0.35 µm gate, even transiently. It also improves
17/20 targets relative to the E1 fixed-budget control.

## Training

The replay pool contains 225,614 valid contiguous transitions from the existing
acthistory training data. No new online rollouts or validation data enter the
replay buffer. Training uses expert Flow supervision plus twin-Q guidance on
the first action of a differentiable four-step sample.

| Epoch | Flow train loss | Validation loss | Critic loss | Policy Q mean |
|---:|---:|---:|---:|---:|
| 0 | 0.118485 | 0.148653 | 0.076534 | -2.2558 |
| 1 | 0.118097 | 0.148461 | 0.011947 | -4.7274 |
| 2 | 0.117015 | 0.148512 | 0.014281 | -7.6047 |
| 3 | 0.116647 | **0.148059** | 0.019713 | -10.4546 |

Policy-Q scale drifts increasingly negative while the normalized Q objective
stays near its 0.1 cap. This is a critic-scale stability warning, not evidence
of optimum convergence.

## Evidence

- [Pipeline report](REPORT.md)
- [v1 comparison](COMPARISON.md)
- [Machine-readable summary](summary.json)
- [Per-target metrics](episodes.csv)
- [All error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired error vs v1](evaluation/plots/final_error_vs_v1.png)
- [Training result](training/result.json)
- [Training history](training/training_history.json)
- [Replay manifest](training/replay_manifest.json)
- [Training configuration](training/training_config.json)
- [Shard logs](evaluation/shard_logs.txt)

The 555 MB checkpoint is intentionally excluded from Git.

All results are MuJoCo simulations and do not establish hardware accuracy or
clinical validity.
