# DiT DAgger2 acthistory status — 2026-11-17

## Result

The action-history data-contract repair is implemented and verified, but the
retrained policy did not improve closed-loop performance.

| Item | Result |
|---|---:|
| Retrained checkpoint | epoch 3 |
| Training loss | 0.135912 |
| Validation loss | 0.161541 |
| Previous DAgger2 validation loss | 0.158247 |
| 0.35 µm canonical successes | 0 / 20 |
| Encoder-settled successes | 0 / 20 |
| Final error mean / median | 1737.487 / 2012.021 µm |
| Per-episode minimum error mean | 143.065 µm |
| Low-speed exit frames | 2 / 20 |
| Encoder-settled exit frames | 0 / 20 |
| Safety-limited steps | 0 |

The result is `FAIL`.

## Data and contract repair

- Expert supervision is stored in `action`.
- The command actually accepted for execution is stored in
  `executed_action`.
- Velocity, previous-action features and action chunks only use
  `executed_action` for history.
- `source_frame_index` and `timestamp_s` preserve original time continuity.
- Windows do not cross filtered expert-invalid gaps.
- Legacy teacher data missing the v2 history fields fails fast.
- Closed-loop success requires the error gate and three consecutive
  `MotorExecution.settled()` frames.

The migrated training set contains 839 episodes and 234,316 frames. Round 2
contains 79 valid trajectories, 15,883 retained frames and 5,617 policy
execution frames. All 5,614 valid policy frames had different teacher and
accepted actions. Three invalid expert frames create three protected gaps.

## Comparison with old DAgger2

The evaluation protocol and 20 development targets are identical.

| Metric | Old DAgger2 | acthistory DAgger2 |
|---|---:|---:|
| Final error mean | 1029.075 µm | 1737.487 µm |
| Final error median | 668.484 µm | 2012.021 µm |
| Per-episode minimum mean | 68.675 µm | 143.065 µm |
| Canonical successes | 0 / 20 | 0 / 20 |

Paired final-error comparison: 5 episodes improved and 15 worsened. Mean and
median final error regressed by 708.412 µm and 635.352 µm respectively.

## Plots

- [All 20 error trajectories](evidence/error_curves_all.png)
- [Per-target small multiples](evidence/error_curves_small_multiples.png)
- [Best and final error per target](evidence/best_and_final_error.png)
- [Paired final error vs old DAgger2](evidence/final_error_vs_old.png)

## Interpretation

The result cannot be attributed to the stricter success definition because
both the old and new models had zero successes. The acthistory model reaches
local minima as low as 37.391 µm, but its final error is generally higher and
it never reaches the 0.35 µm terminal gate.

All numbers are MuJoCo simulation results, not hardware accuracy or clinical
validity.
