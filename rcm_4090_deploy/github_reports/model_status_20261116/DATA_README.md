# Experiment Data Package

This directory contains the compact evidence used for model selection and
failure analysis. All results are from MuJoCo simulation.

## Included

- `evaluation_metrics.csv`: compact cross-model comparison.
- `evidence/`: strict summaries, aggregate summaries, training history, and
  selected phase-transition plots.
- `archives/evaluations_diffusion_20um.tar.gz`: raw v8-v12 Diffusion Policy
  evaluation JSON for the `20 um` protocol.
- `archives/evaluations_flow_strict.tar.gz`: raw Flow, Flow+DAgger2, residual, and
  reachable-target evaluation JSON for `0.1 um` / `0.35 um` protocols.
- `archives/evaluations_dit_rf_035um.tar.gz`: raw pure-DiT and DiT+DAgger1
  evaluation JSON.
- `archives/training_histories.tar.gz`: training loss history for the main model
  checkpoints.
- `archives/dataset_metadata.tar.gz`: episode/frame counts, strict-zero metadata,
  source provenance, and strict audit for the main training datasets.
- `archives/collection_manifests.tar.gz`: DAgger round 1 and round 2 collection
  manifests, including failed episodes and teacher-executor statistics.
- `archives/SHA256SUMS`: archive integrity hashes.

The raw `evaluation_results.json` files contain per-step states, actions,
velocities, errors, OCT navigation values, and evaluation contracts.

## Not included in Git

- RGB-D image datasets and episode `.npz` payloads, approximately 14 GB.
- Checkpoints and optimizer states, approximately 4.4 GB.
- Raw DAgger rollout directories, approximately 29 GB.
- Generated `outputs/`, `data/`, and `runs/` trees.

These files exceed normal GitHub repository limits. Model checkpoints should be
published as GitHub Release assets, Git LFS objects, or an external model/data
registry. Raw datasets require an external artifact store.

## Archive verification

```bash
cd rcm_4090_deploy/github_reports/model_status_20261116/archives
sha256sum -c SHA256SUMS
tar -tzf evaluations_flow_strict.tar.gz | head
```

## Interpretation boundary

The `20 um`, `0.1 um`, and `0.35 um` experiments use different acceptance
gates. Compare absolute endpoint errors separately from binary success counts.
No result in this package establishes physical OCT accuracy, hardware safety,
or clinical readiness.
