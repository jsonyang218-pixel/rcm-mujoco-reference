# Experiment data — DiT DAgger2 acthistory

## Scope

This report contains the compact evaluation and metadata needed to reproduce
the 2026-11-17 acthistory experiment. Raw RGB-D images, the full 790 MB
training directory and the 414 MB checkpoint are intentionally excluded from
ordinary Git.

## Dataset contract v2

The training dataset adds three row-aligned arrays:

- `executed_action`: accepted delta command in A deg, B deg, C mm.
- `source_frame_index`: original frame index before invalid-frame filtering.
- `timestamp_s`: original simulation timestamp.

`action` remains the expert supervision target. History-enabled training
requires all three new arrays and fails on old teacher data without them.

## Files

- `evidence/acthist_dit_rf_dagger2_0p35um.json`: complete 20-target summary
  and paired old-model comparison.
- `evidence/acthist_dit_rf_dagger2_training_history.json`: four training
  epoch losses.
- `evidence/dagger2_acthist_migration.json`: row-level migration validation.
- `evidence/train_metadata.json`: full migrated training metadata.
- `evidence/validation_metadata.json`: upgraded validation metadata.
- `evidence/*.png`: closed-loop error plots.
- `archives/evaluations_dit_rf_dagger2_acthist_035um.tar.gz`: all 20 episode
  trajectories and the summary.
- `archives/acthist_training_and_migration.tar.gz`: training history,
  migration evidence and dataset metadata.

## Evaluation protocol

```text
targets          20 v12 development targets
reachable        reachable_target_robot_mm from frozen manifest
control period   0.05 s
max steps        1600
success gate     0.35 um
action samples   1
action execution first step of a 16-step action chunk
flow integration 4 steps
backend          MotorBackend
success stop     MotorExecution.settled() for 3 consecutive frames
```

The run used 10 parallel shards of two targets each on one RTX 4090.
