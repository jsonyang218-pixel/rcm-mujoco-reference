# Homogeneous label-fix v4 six-hour gated experiment

Status: `completed`. Base: `703003bc7595318709097faa184f147d2386554f`. Branch: `experiment/overnight-label-fix-v4-20261124`.

Scope: software and MuJoCo simulation only; this is not clinical or hardware evidence.

## Baseline

Reused validated task2 fast5 final mean: 65.569288 um (0/5 strict).

## Gates

- `phase0`: `completed`

- `phase1`: `none`

- `phase2`: `skipped_no_phase1_candidate`

- `phase3`: `skipped_gate`

- `fallback`: `completed`

## Best fast5 result

- Checkpoint: `None`
- Final mean: 54.313674 um
- Minimum/global mean: 33.249772 um
- Strict success: 0/5

## dev20

- Executed: `False`
- Reason: `Phase 2 checkpoint gate did not pass`

## Recommendation

Continue only with candidates that show direction consistency and reduced drift on the same evaluator. Retain the frozen v3/baseline assets and treat any non-improvement as a negative label-repair result rather than retuning the gate.
