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

## Completed result: A original-only continuation

A completed all 20 development targets but failed the closed-loop objective.

| Metric | State-attention v1 | A original |
|---|---:|---:|
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | 434.397 um | **1,087.778 um** |
| Final error median | 258.933 um | **883.749 um** |
| Final error max | 2,128.198 um | **2,466.592 um** |
| Final error <= 300 um | 12 / 20 | **6 / 20** |
| Per-episode minimum mean | 93.680 um | **63.988 um** |
| Encoder settled at exit | not published | **0 / 20** |
| Safety-limited steps | 0 | 0 |

Training validation Flow loss improved from `0.149865` to `0.144550`, while
paired closed-loop final error worsened on 13 targets and improved on 7.
Transient minimum error improved from `93.680` to `63.988 um`, but the policy
did not retain the target. This is evidence against using validation Flow loss
as the primary experiment selector.

## Partial status at upload

- B training completed: best validation loss `0.145008` at epoch 3.
- B closed-loop evaluation was still running at upload time.
- C had not started because the pipeline runs arms sequentially.
- No B/C performance claim is made in this report.

## Evidence

- [A pipeline report](A_ORIGINAL/REPORT.md)
- [A baseline comparison](A_ORIGINAL/COMPARISON.md)
- [A machine-readable summary](A_ORIGINAL/summary.json)
- [A error trajectories](A_ORIGINAL/evaluation/plots/error_curves_all.png)
- [A small multiples](A_ORIGINAL/evaluation/plots/error_curves_small_multiples.png)
- [Contract audit](contract/experience_replay_contract.json)
- [Balanced replay GPU smoke](smoke/training.log)
- [B training result](B_STATIC/training/result.json)
- [B status snapshot](B_STATIC/status_asof.json)

The 414 MB checkpoints, raw data, and full trajectory JSON are excluded from
Git. Results are from MuJoCo simulation only and do not establish hardware or
clinical performance.
