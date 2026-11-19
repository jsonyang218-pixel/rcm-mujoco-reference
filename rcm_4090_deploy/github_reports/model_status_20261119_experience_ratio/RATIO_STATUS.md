# Correction-Experience Sampling Ratio Results — 2026-11-19

## Scope

Three policies started from the same frozen state-attention v1 checkpoint and
used the same 4 x 1,000 update budget, validation data, seed, and 20-target
closed-loop protocol. The only intended variable was the probability of
sampling a teacher-corrected window.

| Arm | Correction probability |
|---|---:|
| R0 | Natural ratio, 6.5987% |
| R10 | 10% |
| R20 | 20% |

## Data-contract correction

The action-window boundary bug was fixed. For 20 rows, two observation-history
rows and a 16-action block, the verifier now returns exactly four valid
windows. Training and auditing share one function and produce the same index
SHA-256:

`3f852d9c7b296cdcc4c4abbca6a12e0e099050bc5bbfc4254aae979830fb927d`

The shared index set contains 220,816 valid windows. Exclusions are 13,408 for
episode boundaries, 48 source-frame gaps, 0 time anomalies, and 28 invalid
inputs. The natural correction ratio is 6.5987%.

## Zero-update reproduction

- 220 model tensors are bit-identical after a nominal one-step zero-learning
  update.
- Fixed-noise action output is bit-identical in the direct model check.
- Three closed-loop targets have bit-identical states, errors, velocities and
  settle flags across old and new evaluators.
- Raw action records differ by at most `2.015e-6`, below the `1e-5` audit
  tolerance; motor quantization produces identical motion.

## Closed-loop results

| Metric | R0 natural | R10 | R20 |
|---|---:|---:|---:|
| Correction probability | 6.5987% | 10% | 20% |
| Canonical successes | 0 / 20 | 0 / 20 | 0 / 20 |
| Final error mean | **247.936 um** | 980.699 um | 589.481 um |
| Final error median | **170.736 um** | 549.848 um | 227.022 um |
| Final error P90 | **566.363 um** | 2,435.211 um | 1,430.254 um |
| Final error maximum | **658.363 um** | 2,521.840 um | 2,202.496 um |
| Encoder settled at exit | 0 / 20 | 0 / 20 | 0 / 20 |
| Safety-limited steps | 0 | 0 | 0 |
| Improved vs v1 | **19 / 20** | 9 / 20 | 11 / 20 |
| Worsened vs v1 | **1 / 20** | 11 / 20 | 9 / 20 |

Against v1, R0 improves the final-error mean by `186.461 um`. R10 worsens it
by `546.302 um`, and R20 worsens it by `155.084 um`.

## Conclusion

Raising correction sampling above the natural ratio is harmful at this data
scale and update budget. R0 is retained for new expert-corrected collection.
All ratios still fail the strict `0.35 um` plus three-settled-frame gate, and
none settles at exit. Sampling ratio alone does not solve retention or
terminal stopping.

## Evidence

- [R0 report](R0/REPORT.md)
- [R10 report](R10/REPORT.md)
- [R20 report](R20/REPORT.md)
- [A/B/C/R ratio comparison](RATIO_COMPARISON.md)
- [Machine-readable comparison](ratio_comparison.json)
- [R0 trajectories](R0/evaluation/plots/error_curves_all.png)
- [R10 trajectories](R10/evaluation/plots/error_curves_all.png)
- [R20 trajectories](R20/evaluation/plots/error_curves_all.png)

Checkpoints are excluded from Git. Results are MuJoCo simulation only.
