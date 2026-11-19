# State-Attention v1 + Plan-2 Data — 2026-11-19

## Scope

This run continues the state-conditioned temporal attention v1 architecture
from the frozen v1 checkpoint and trains on the previous 839-episode dataset
plus the accepted 200 plan-2 expert-correction trajectories.

| Data | Episodes | Frames |
|---|---:|---:|
| Previous training set | 839 | 234,316 |
| Plan-2 accepted trajectories | 200 | 85,381 |
| Merged training set | **1,039** | **319,697** |

The merged dataset contains 301,634 valid training windows. Validation uses the
existing 40-episode, 26,650-window v1 validation set. Normalization statistics
are inherited from the initial v1 checkpoint.

## Model

- Architecture: `state_conditioned_temporal_attention_v1`
- Parameters: 34.538115M
- Initial checkpoint: frozen state-attention v1
- Best checkpoint SHA-256:
  `6a8572a6f2646f2fc9f0525cff3a9b667c6e9962a162c8c86151137f30dacd4c`
- Training: 4 epochs, batch size 64, seed 20261116
- Plan-2 sampling: `dagger_episode_start=839`, `dagger_repeat=10`

## Training result

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 0.094976 | 0.151036 |
| 1 | **0.090775** | **0.149849** |
| 2 | 0.089364 | 0.150341 |
| 3 | 0.088714 | 0.149910 |

Best epoch is 1 with validation loss `0.149849`, narrowly better than the
frozen v1 best validation loss `0.149865`. The final-training loss continues to
decrease, but validation loss does not improve after epoch 1.

## Closed-loop result

The checkpoint was evaluated on the first 10 development targets with seed 42,
1,600 control steps, and the `0.35 um` plus three-consecutive-settled-frame
success gate.

| Metric | Frozen v1, first 10 | New v1 + plan 2 |
|---|---:|---:|
| Canonical successes | 0 / 10 | 0 / 10 |
| Final error mean | **351.163 um** | 3,458.685 um |
| Final error median | **258.933 um** | 4,308.960 um |
| Final error minimum | 134.901 um | **287.678 um** |
| Final error maximum | 2,128.198 um | 6,447.226 um |
| Final error <= 300 um | 6 / 10 | 1 / 10 |
| Final error <= 100 um | 0 / 10 | 0 / 10 |
| Encoder settled at exit | 0 / 10 | 0 / 10 |
| Safety-limited steps | 0 | 0 |

Only 2 of 10 targets improve relative to frozen v1; 8 worsen. The final-error
mean increases by `3,107.522 um`. The candidate reaches `34.009 um` on one
target but loses the target and ends at `287.678 um`; no target enters the
strict `0.35 um` gate.

All 10 evaluation shards completed. The automated pipeline then rejected the
run while comparing it against the frozen 20-target baseline because the
candidate contains episode indices 0-9 and the baseline contains 0-19. The
reported comparison therefore uses the exact target-ID-matching first 10
frozen-v1 records.

## Conclusion

The plan-2 merged dataset produces a normal offline validation curve for the
state-attention v1 architecture, but the closed-loop result is a severe
regression. This checkpoint is rejected as a deployment candidate. Improved
Flow validation loss again does not predict terminal closed-loop accuracy.

## Evidence

- [Evaluation report](evaluation/REPORT.md)
- [Machine-readable evaluation summary](evaluation/summary.json)
- [Per-target metrics](evaluation/episodes.csv)
- [Compressed raw trajectories](evaluation/trajectories.tar.gz)
- [All error curves](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Training curve](training/training_curves.png)
- [Training summary](training/training_summary.json)
- [10-target evaluation config](configs/state_attention_v1_plan2_plus_original200_eval_dev10_shards10_20261119.json)

The 396 MiB checkpoint and raw image datasets are excluded from Git. Results
are MuJoCo simulation only and do not establish hardware or clinical accuracy.
