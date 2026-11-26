# State-attention v1 baseline

Baseline ID: `state_attention_v1_20261117`

Status: **FROZEN**

This is the first saved state-attention model version and the primary
development comparison baseline for all later versions.

## Training baseline

| Item | Result |
|---|---:|
| Architecture | `state_conditioned_temporal_attention_v1` |
| Best epoch | 3 |
| Best validation loss | 0.1498646501 |
| Warm-start validation loss | 0.1615405332 |
| Validation improvement | 7.227835% |
| New attention parameters | 83,136 |

## Closed-loop baseline

The evaluation uses 20 frozen development targets under the strict terminal
protocol.

| Metric | Result |
|---|---:|
| Canonical settled successes | 0 / 20 |
| Final error mean | 434.397058 µm |
| Final error median | 258.933493 µm |
| Final error minimum | 134.900926 µm |
| Final error maximum | 2128.197890 µm |
| Per-episode minimum error mean | 93.680006 µm |
| Safety-limited steps | 0 |

## Frozen artifacts

- [Pipeline configuration](pipeline_config.json)
- [Machine-readable baseline](BASELINE.json)
- [Evaluation summary](evaluation_summary.json)
- [Training history](training_history.json)

Exact hashes for the model source, checkpoint, datasets, target manifest, and
evaluation evidence are stored in `BASELINE.json`.

## Comparison rules

Later versions must use the same development target IDs and episode indices.
Comparisons require the same target manifest SHA-256 and evaluation protocol.
The primary metrics are:

- canonical settled successes;
- final error mean and median;
- per-episode minimum error mean;
- paired improved and worsened target counts;
- safety-limited steps.

Paired improvement is defined as:

```text
v1_final_error_um - candidate_final_error_um
```

A positive value is an improvement. Any later pipeline run that configures
this baseline must emit `comparison.json` and `COMPARISON.md`.

Do not rewrite these frozen files. Add a new versioned baseline instead.
