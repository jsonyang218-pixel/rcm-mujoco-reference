# Original DiT-Flow with Previous + Plan-2 Data — 2026-11-19

## Scope

This run fine-tunes the original DiT rectified-Flow architecture from the
existing v1 checkpoint. It keeps the original Flow-matching objective and uses
the previous 839-episode training set plus the accepted 200 plan-2 trajectories.

| Data | Episodes | Frames |
|---|---:|---:|
| Previous training set | 839 | 234,316 |
| Plan-2 accepted trajectories | 200 | 85,381 |
| Merged training set | **1,039** | **319,697** |

The merged dataset has 301,634 valid training windows. Validation uses the
existing 26,650-window v1 validation set.

## Training

Checkpoint normalization statistics are inherited from the initial v1
checkpoint, rather than recomputed from the new data.

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 0.117040 | 0.148789 |
| 1 | 0.113280 | 0.147219 |
| 2 | 0.111231 | 0.147659 |
| 3 | **0.110118** | **0.146885** |

Best validation loss is `0.146885`, about 9.1% lower than the initial
checkpoint's `0.161541`. The best checkpoint SHA-256 is
`4e4b2865fc9939a22ff7d5f6953d2a378af9d4baeead887fd053ff32a243d275`.

The earlier new-data-only run improved training loss but worsened validation
loss from 0.189130 to 0.213401. It is retained as a negative diagnostic and is
not used for the closed-loop result.

## Closed-loop result

The merged checkpoint was evaluated on the fixed 20 development targets with
seed 42, 1,600 steps, and the `0.35 um` plus three-settled-frame success gate.

| Metric | State-attention v1 | Original DiT-Flow + plan 2 |
|---|---:|---:|
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | **434.397 um** | 1,151.149 um |
| Final error median | **258.933 um** | 1,392.104 um |
| Final error minimum | 134.901 um | **83.444 um** |
| Final error maximum | 2,128.198 um | **1,952.920 um** |
| Final error <= 300 um | **12 / 20** | 4 / 20 |
| Per-episode minimum mean | 93.680 um | **72.032 um** |
| Encoder settled at exit | not published | **0 / 20** |
| Safety-limited steps | 0 | 0 |

Only 5 targets improved relative to v1 and 15 worsened. The final-error mean
increased by `716.752 um`. The lower transient minimum error indicates that
the policy can approach targets more closely, but it does not retain the target
or settle.

## Conclusion

Adding the previous data fixed the catastrophic validation-loss behavior of the
new-data-only run and produced a normal offline curve. However, lower Flow loss
did not translate into better closed-loop terminal accuracy. The original
architecture with this merged dataset is rejected for the next deployment
candidate.

## Evidence

- [Closed-loop report](REPORT.md)
- [v1 comparison](COMPARISON.md)
- [Machine-readable summary](summary.json)
- [Per-target metrics](episodes.csv)
- [All error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Training curve](training/training_curves.png)
- [Training summary](training/training_summary.json)

The 414 MB checkpoint and original/raw datasets are excluded from Git. Results
are MuJoCo simulation only and do not establish hardware or clinical accuracy.
