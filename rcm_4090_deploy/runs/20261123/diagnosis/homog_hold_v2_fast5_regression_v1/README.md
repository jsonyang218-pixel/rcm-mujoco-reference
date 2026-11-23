# Homogeneous replay hold-v2 fast5 regression diagnosis

Date: 2026-11-23 (Asia/Shanghai)

## Outcome

The repaired collection workflow completed all four configured collection,
training, and fast5 stages. The strict model-improvement gate failed and the
conditional dev20 stage was correctly skipped. No process remained active.

The three-target real RGB-D/CUDA hold-transition smoke passed:

| Target | Final error (um) | Held frames | Max held error (um) | All held motor records motionless |
|---|---:|---:|---:|---|
| E40 | 4.12475 | 200 | 4.12475 | true |
| E41 | 4.00974 | 200 | 4.00974 | true |
| E42 | 4.21097 | 200 | 4.21097 | true |

Collection audit and training also passed:

- 60/60 trajectories; 42,399 frames; 41,422 valid windows.
- 200 independently sampled frames; OCT/truth median error 0.078638 um.
- A/B/C reverse fraction 0/0/0; valid-window fraction 1.0.
- T3 held frames below 5 um: 4000/4000.
- Four epochs x 1000 batches, batch 16, CUDA AMP, exact 8% correction sampling.
- Validation loss moved from 0.14809153 to 0.14729503.

## Fast5 comparison

All arms use the same five development targets, seeds, 1600 actions, render,
teacher snapshots, and stopping protocol.

| Arm | Final mean (um) | Absolute drift 1000 to 1600 (um) | Strict success |
|---|---:|---:|---:|
| Initial task2 step4000 | 65.5693 | 22.6132 | 0/5 |
| New step2000 | 187.9443 | 32.4441 | 0/5 |
| New step4000 | 244.8281 | 35.9779 | 0/5 |

The selected new checkpoint was step2000, but it was 122.375 um worse in final
mean than the paired initial checkpoint. The direction gate also failed:
A/B/C snapshot reversals were 4/4/3 and the first-order-toward fraction was
0.489583, below the required greater-than-0.70. The best checkpoint did not
trigger the optional arms, and dev20 was skipped by the existing gate.

## Evidence for the regression

The early approach is almost unchanged: at steps 0, 100, and 400 the error
curves and raw action RMS values are nearly identical across arms. The failure
appears after conventional control has reduced the error, during the terminal
fine-correction phase.

Late C-axis raw-action RMS:

| Arm | Steps 800-1200 (mm/step) | Steps 1200-1600 (mm/step) |
|---|---:|---:|
| Initial task2 step4000 | 0.0019052 | 0.0018946 |
| New step2000 | 0.0026522 | 0.0027628 |
| New step4000 | 0.0030905 | 0.0030212 |

The new step2000 and step4000 policies therefore retain about 39% and 63% more
late C-axis command magnitude. Their action means are near zero, but their
variance keeps moving the tip around the target. The frozen direction audit
also found reversals on every axis and a toward fraction below chance-level
stability.

The direct supervision defect is in T3. The settlement controller starts
issuing exact zero actions once a below-5-um candidate is encoder-settled, while
the stored zero-initial-velocity geometry label remains nonzero. In the first
400 frames of the 20 T3 trajectories, 7,695 of 8,000 frames (96.19%) have a
nonzero label but an exact-zero executed action. The model is consequently
trained to keep making micro-corrections from a state at which the collector
has already stopped.

T2 has a smaller but related magnitude mismatch. Its label norm averages
0.000924 while the executed teacher-command norm averages 0.006243 (label is
14.8% of executed); direction remains consistent, with median cosine 0.930.

## Interpretation and next gate

This result does not reject the hold repair: hold transitions and motor
settlement passed all checks. It rejects the supervision produced after that
repair. Increasing the correction ratio or changing the checkpoint budget
would train against the same contradiction and is not supported by this
evidence.

The next experiment should make terminal supervision and executed behavior
consistent. A concrete candidate is a new version that records explicit
`settlement_ready` provenance and trains zero terminal labels from the first
ready frame, while excluding any pre-ready transition whose action history
cannot be reconstructed. The T2 teacher-label/execution magnitude mismatch
should be checked in the same preflight. Then run a fresh fast5 screen with the
same targets, seeds, 1600 actions, and gates. Any change to label semantics,
correction ratio, frozen modules, or checkpoint schedule is a new experiment
and requires explicit authorization under the repository SOP.

All findings here concern software and MuJoCo simulation only. They do not
establish clinical accuracy or hardware readiness.
