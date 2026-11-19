# Plan 2 Expert-Correction Collection

Plan 2 targets 280 independent trajectories over 70 new targets:

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| D1 near-target correction | 70 | 14 | 14 |
| D2 overshoot/oscillation recovery | 50 | 10 | 10 |
| D3 settling and hold | 50 | 10 | 10 |
| D4 full approach | 30 | 6 | 6 |

The collection driver is:

```bash
python scripts/datasets/collect_experience_replay_plan2.py \
  --model outputs/state_attention_experience_ratio_20261118_1/r0_natural/training/step_004000.pth \
  --dataset-root data/<run> \
  --stage pilot|d3-pilot|remaining|validation|test|all
```

Use `--dry-run` to inspect all commands and trajectory counts. Each invocation
uses explicit target-seed files and never reuses validation or test targets for
training.

The collector stores expert correction labels, actual executed actions, label
validity, OCT values, terminal audits, and image references per episode.
Validation and test partitions remain isolated from training.

The first D3 pilot showed that mixed policy/teacher collection did not improve
settling reliably. D3 now uses expert-dominant `beta=1.0`; D1, D2, and D4 keep
the policy/teacher mixture. Raw RGB-D datasets are excluded from normal Git
storage.
