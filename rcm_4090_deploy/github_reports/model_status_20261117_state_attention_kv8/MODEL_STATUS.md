# State-attention KV8 — 2026-11-17

## Result

KV8 keeps all 16 action queries and outputs while aggregating K, V, and bias
key dimensions from 16 to 8 by adjacent-pair averaging. It uses the same
parameters and starts from the v1 checkpoint. The experiment is `FAIL`: it
improves the best transient error slightly but worsens final error.

| Metric | v1 baseline | KV8 |
|---|---:|---:|
| Best validation loss | **0.149865** | 0.174219 |
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | **434.397 µm** | 1513.620 µm |
| Final error median | **258.933 µm** | 1713.822 µm |
| Final error maximum | **2128.198 µm** | 2983.236 µm |
| Per-episode minimum mean | 93.680 µm | **80.416 µm** |
| Paired final error | — | 4 improved, 16 worsened |
| Safety-limited steps | 0 | 0 |

Training completed four epochs with 1000 steps each and no persistent parameter
increase. Q/K/V projections remain full-sized; compression changes attention
computation rather than model capacity.

## Validation

- Q and output remain 16 tokens.
- K, V, and bias key dimension are 8.
- v1 checkpoint strict-loads into KV8.
- Parameter count remains 34,538,115.
- GPU forward output is finite.
- Checkpoint metadata records `state_conditioned_temporal_attention_kv8_v1`,
  eight KV tokens, and fixed adjacent-mean aggregation.

## Evidence

- [Pipeline report](REPORT.md)
- [v1 comparison](COMPARISON.md)
- [Machine-readable summary](summary.json)
- [Per-target metrics](episodes.csv)
- [All error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired error vs v1](evaluation/plots/final_error_vs_v1.png)
- [Training stage result](training/result.json)
- [Shard logs](evaluation/shard_logs.txt)

The 414 MB checkpoint is intentionally excluded from Git.
