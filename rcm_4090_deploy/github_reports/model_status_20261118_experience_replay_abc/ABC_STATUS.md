# Expert-Corrected Experience Replay A/B/C — 2026-11-18

## Scope

This branch implements the project plan for state-attention DiT rectified-Flow
training with expert-corrected history replay. It keeps the existing 16-step
action blocks and adds no critic, reward loss, progress loss, or action penalty.

| Arm | Training data | Batch organization |
|---|---|---|
| A | Original supervised windows | Uniform original batches |
| B | Original + teacher corrections | Uniform over the static union |
| C | Original + teacher corrections | Exactly 8 original + 8 correction |

All arms start from the state-attention v1 checkpoint and use 4 x 1,000 updates
with the same validation and closed-loop protocol.

## Data audit

The action-block contract passed:

- 234,316 frames and 839 episodes;
- 219,967 valid two-history, sixteen-action windows;
- `action` is the expert supervision label;
- `executed_action` is the motion-history and execution record;
- no cross-episode or source-frame-gap windows;
- 14,489 valid teacher-corrected action windows;
- teacher near-target coverage: 2,048 windows below 30 um, 2,863 at
  30-100 um, and 5,268 at 100-300 um.

## Final A/B/C result

All three arms completed the same 20 development targets. None reached the
strict `0.35 um` plus three-settled-frames success gate.

| Metric | State-attention v1 | A original | B static | C balanced |
|---|---:|---:|---:|---:|
| Canonical successes | 0 / 20 | 0 / 20 | 0 / 20 | 0 / 20 |
| Final error mean | 434.397 um | 1,087.778 um | **207.290 um** | 1,486.514 um |
| Final error median | 258.933 um | 883.749 um | **141.257 um** | 1,637.206 um |
| Final error P90 | 757.163 um | 2,196.504 um | **510.406 um** | 2,661.735 um |
| Final error max | 2,128.198 um | 2,466.592 um | **677.360 um** | 2,915.647 um |
| Final error <= 300 um | 12 / 20 | 6 / 20 | **17 / 20** | 4 / 20 |
| Per-episode minimum mean | 93.680 um | **63.988 um** | **63.011 um** | 84.064 um |
| Encoder settled at exit | not published | 0 / 20 | 0 / 20 | 0 / 20 |
| Safety-limited steps | 0 | 0 | 0 | 0 |

Against the frozen v1 result:

- A: 7 improved, 13 worsened, final-error mean worse by `653.381 um`;
- B: **18 improved, 2 worsened**, final-error mean better by `227.107 um`;
- C: 4 improved, 16 worsened, final-error mean worse by `1,052.117 um`.

Against A, B improved 18 targets and worsened 2. Against B, C improved only 3
targets and worsened 17. The experiment therefore identifies the added expert
correction data as useful, while the hard 50/50 replay organizer is harmful in
this configuration.

## Retention and settling

- B entered 300 um on 20/20 targets, but left the interval after first entry
  on 15/20; mean longest in-interval run was `667.9` frames.
- B entered 100 um on 19/20 targets, but left the interval after first entry
  on 19/20; mean longest run was `29.5` frames.
- A entered 100 um on 16/20 and C on 15/20.
- Only one target in A and one in B briefly entered 30 um; none in C did.
- No arm entered the `0.35 um` gate or ended with a settled encoder.

This separates two remaining problems: static correction merging improves
terminal error, but the policy still cannot retain fine error or stop.

## Interpretation

Validation Flow loss alone did not rank the policies: A had the lowest value
(`0.144550`), but B produced the best closed-loop result. The useful change was
the additional teacher-corrected data. The exact 50/50 organizer over-weighted
correction windows and degraded the policy, so it should not be carried into
the next round unchanged.

The next experiment should use B-style static merging with newly collected
near-target, overshoot-recovery, and hold trajectories, while adding explicit
retention and stop criteria. The 50/50 historical replay mechanism is rejected
for the current data ratio and update budget.

## Evidence

- [A pipeline report](A_ORIGINAL/REPORT.md)
- [A baseline comparison](A_ORIGINAL/COMPARISON.md)
- [A machine-readable summary](A_ORIGINAL/summary.json)
- [A error trajectories](A_ORIGINAL/evaluation/plots/error_curves_all.png)
- [A small multiples](A_ORIGINAL/evaluation/plots/error_curves_small_multiples.png)
- [B pipeline report](B_STATIC/REPORT.md)
- [B machine-readable summary](B_STATIC/summary.json)
- [C pipeline report](C_BALANCED/REPORT.md)
- [C machine-readable summary](C_BALANCED/summary.json)
- [A/B/C comparison](ABC_COMPARISON.md)
- [A/B/C machine-readable comparison](abc_comparison.json)
- [Contract audit](contract/experience_replay_contract.json)
- [Balanced replay GPU smoke](smoke/training.log)
- [B training result](B_STATIC/training/result.json)
- [B status](B_STATIC/status_final.json)
- [C training result](C_BALANCED/training/result.json)
- [C status](C_BALANCED/status_final.json)

The 414 MB checkpoints, raw data, and full trajectory JSON are excluded from
Git. Results are from MuJoCo simulation only and do not establish hardware or
clinical performance.
