# Homogeneous replay label repair and retraining — 2026-11-23

Workflow status: `failed`. Base: `cd22e10471f76807e2b1ce307db5d38903b40d30`. Branch: `feature/homog-label-fix-20261123`.

All images and OCT observations were reused. Only the action supervision array changed.

No wall-clock timeout was configured. Training/inference resume state is written atomically.

## failure

```json
{
  "status": "failed",
  "reason": "TypeError(\"'tuple' object is not a mapping\")"
}
```

Weights and raw correction RGB remain local with path/size/SHA256 inventories in the publication record.

This result validates software and MuJoCo behavior only; it is not clinical or hardware accuracy evidence.
