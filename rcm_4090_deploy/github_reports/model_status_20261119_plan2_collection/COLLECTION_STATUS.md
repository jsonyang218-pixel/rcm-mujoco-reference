# Plan 2 Expert-Correction Collection — 2026-11-19

## Dataset design

The accepted plan contains 280 independent trajectories over 70 new target
seeds:

| Category | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| D1 near-target correction | 70 | 14 | 14 | 98 |
| D2 overshoot/oscillation recovery | 50 | 10 | 10 | 70 |
| D3 settling/hold | 50 | 10 | 10 | 70 |
| D4 full approach | 30 | 6 | 6 | 42 |
| Total | 200 | 40 | 40 | 280 |

Target seeds are isolated by split: 50 training targets, 10 validation targets,
and 10 test targets. The student checkpoint is R0 natural-ratio
`step_004000.pth`, SHA-256
`b504146485c38c1c26bd960d8aa831beba0df7e564f1f20267862a79b7dc1733`.

## Pilot acceptance

The accepted pilot contains 20 trajectories and 8,538 frames:

- D1: 7
- D2: 5
- D3: 5
- D4: 3

Teacher correction labels are present, actual executed actions remain separate,
and incomplete/empty trajectories are not counted as successful samples.

The original D3 pilot used mixed policy/teacher control and did not improve
reliably. D3 was re-collected with expert-dominant `beta=1.0`. Its accepted
trajectories reduce error from `20-77 um` to `0.016-4.081 um`, demonstrating
that the expert provides useful settling/hold corrections.

## Current collection state

The remaining 180 training trajectories have started:

- D1: 63
- D2: 45
- D3: 45
- D4: 27

Validation and test collection follow after training completion. The raw image
datasets are intentionally excluded from Git; only collection plans, manifests,
status, and summaries are uploaded.

## Failed and superseded attempts

- `data/experience_replay_plan2_20261118`: planning metadata was written before
  the output-root creation guard; no trajectories were produced.
- `data/experience_replay_plan2_20261118_retry1`: collection stopped before any
  trajectory because DAgger loaded the base DiT class for a state-attention
  checkpoint. Architecture dispatch was added.
- The first D3 pilot used `beta=0.5`; 2 of 5 trajectories worsened. It is
  retained for audit but superseded by the `beta=1.0` D3 pilot.
- Empty-frame D3 perturbations were replaced with fixed lateral perturbations;
  the failed attempts retain their manifests and are not counted as data.

## Evidence

- [Pilot plan](plans/pilot_plan.json)
- [Expert-dominant D3 plan](plans/d3_beta1_plan.json)
- [Remaining training plan](plans/remaining_plan.json)
- [Compact collection summary](plan2_collection.json)
- [Pilot run manifests](pilot_manifests/)
