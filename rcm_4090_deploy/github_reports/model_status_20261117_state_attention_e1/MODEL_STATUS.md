# State-attention v1 continue-training E1 — 2026-11-17

## Result

E1 continued the frozen state-attention v1 policy for four epochs from the v1
checkpoint with the same data and protocol. Validation loss improved, but
closed-loop final error regressed. The experiment is `FAIL`; v1 remains the
selected model.

| Metric | v1 baseline | E1 |
|---|---:|---:|
| Best validation loss | 0.149865 | **0.146707** |
| Last validation loss | 0.149865 | 0.147718 |
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | **434.397 µm** | 978.815 µm |
| Final error median | **258.933 µm** | 637.927 µm |
| Final error maximum | **2128.198 µm** | 2548.990 µm |
| Per-episode minimum mean | 93.680 µm | **72.531 µm** |
| Safety-limited steps | 0 | 0 |
| Paired final error | — | 7 improved, 13 worsened |

The candidate's final-error mean worsened by 544.418 µm and its median
worsened by 378.994 µm relative to v1. The lower validation loss therefore does
not translate into better terminal convergence.

## Training

E1 started from
`outputs/dit_rectified_flow_state_attention_v1_acthist_20261117/best_model.pth`.
The new attention path used learning rate `1e-5`; the backbone and QKV
projections used `1e-6`. Other settings matched the frozen pipeline config.

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 0.111892 | 0.147184 |
| 1 | 0.110727 | **0.146707** |
| 2 | 0.110032 | 0.147519 |
| 3 | 0.108947 | 0.147718 |

Best checkpoint:

```text
outputs/state_attention_v1_continue_20261117_1/training/best_model.pth
```

SHA-256:

```text
4bc962b328ac57cc9f00e137f7866afe28ce5aa3592c3d166c525619d5f0610b
```

The full 396 MB checkpoint is intentionally excluded from Git.

## Evaluation

The run used the frozen 20-target development manifest and the same strict
protocol as v1: 1600 steps, 0.35 µm threshold, three consecutive settled motor
frames, one action sample, first-step execution, and four flow-integration
steps. All 10 shards exited successfully.

Evidence:

- [Pipeline summary](summary.json)
- [Baseline comparison](comparison.json)
- [Markdown comparison](COMPARISON.md)
- [Per-target metrics](episodes.csv)
- [Training stage result](training/result.json)
- [Training report](training/REPORT.md)
- [Shard logs](evaluation/shard_logs.txt)
- [Cumulative progress reports](evaluation/progress)
- [All 20 E1 error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired final error vs v1](evaluation/plots/final_error_vs_v1.png)

## Pipeline note

The initial progress implementation counted in-progress shard files as
complete. The source pipeline was fixed to count only shards whose process
exited normally. E1 used its pre-start source snapshot, so the incorrect
intermediate artifacts are preserved under `_pre_fix` names. Final E1 metrics
were unaffected, and the complete `after_01_shards` through
`after_10_shards` progress set was regenerated with the fixed logic.

## Decision

- Keep `state_attention_v1_20261117` as the selected model and frozen
  baseline.
- Use E1 as the fixed-budget control for the KV8 experiment.
- Do not rank models by validation loss alone.

All results are MuJoCo simulations and do not establish hardware accuracy or
clinical validity.
