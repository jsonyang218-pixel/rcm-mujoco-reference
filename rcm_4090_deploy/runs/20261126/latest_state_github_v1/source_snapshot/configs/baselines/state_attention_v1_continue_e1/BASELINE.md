# State-attention v1 continue-training E1 baseline

Baseline ID: `state_attention_v1_continue_e1_20261117`

Status: **FROZEN**

E1 is the same-budget continuation control for KV8. It uses the same v1
initial checkpoint, dataset, four-epoch budget, learning rates, seed, target
manifest, strict terminal gate, and shard layout as E2.

| Metric | E1 |
|---|---:|
| Best epoch | 1 |
| Best validation loss | 0.1467070205 |
| Last validation loss | 0.1477184232 |
| Canonical settled successes | 0 / 20 |
| Final error mean | 978.815008 µm |
| Final error median | 637.927069 µm |
| Final error maximum | 2548.989683 µm |
| Per-episode minimum error mean | 72.530569 µm |
| Safety-limited steps | 0 |

Frozen artifacts:

- [Pipeline configuration](pipeline_config.json)
- [Machine-readable baseline](BASELINE.json)
- [Evaluation summary](evaluation_summary.json)
- [Training history](training_history.json)

KV8 comparisons must use E1 for evaluating the effect of compression. The
frozen v1 baseline remains the primary product-level comparison. Do not use E1
successor metrics as a substitute for both comparisons.
