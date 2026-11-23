# Homogeneous replay label repair and retraining — 2026-11-23

Workflow status: `completed`. Base: `cd22e10471f76807e2b1ce307db5d38903b40d30`. Branch: `feature/homog-label-fix-20261123`.

All images and OCT observations were reused. Only the action supervision array changed.

No wall-clock timeout was configured. Training/inference resume state is written atomically.

## stage1

```json
{
  "status": "completed"
}
```

## stage2

```json
{
  "status": "completed"
}
```

## stage3

```json
{
  "status": "completed",
  "stage3_checks_passed": false,
  "dev20_gate_passed": false,
  "best_combination": "V0_step4000",
  "optional_arms_status": "skipped_trigger_failed"
}
```

## stage4

```json
{
  "status": "skipped",
  "reason": "Stage3 late-C/direction/improvement or final<40um/late-drift gate failed"
}
```

Baseline fast5 final mean: 65.569288 um.
Best fast5 final mean: 141.550792 um.
Best absolute late drift: 45.678054 um.
Strict success: 0/5.

Weights and raw correction RGB remain local with path/size/SHA256 inventories in the publication record.

This result validates software and MuJoCo behavior only; it is not clinical or hardware accuracy evidence.
