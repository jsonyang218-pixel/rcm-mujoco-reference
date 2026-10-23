# OCT Three-Group Supplement

This directory is a local, descriptive simulation pilot supplement for the strict visual A/B/C episode set.

> Provenance warning: the requested `79afacc` commit could not be fetched or verified. The checkout has no `.git` metadata, so this package uses a clearly labelled local fallback.

## Evidence separation

The primary A/B/C episodes are copied from `reports/20261022/meeting/evidence/`. They were recorded before the OCT wrapper was connected. The local replay directories under `runs/oct_three_group_79afacc_20261022/` are diagnostic only and are not substituted for primary results.

| Group | Primary outcome | Targets | Error summary |
| --- | --- | ---: | --- |
| A: no occlusion | 3/3 successful | 3/3 attempted | 6.634 um, 6.085 um, 6.268 um |
| B: occlusion stop | 0/3 successful | 1/3 attempted | not computed (incomplete) |
| C: occlusion recovery | 3/3 successful | 3/3 attempted | 6.382 um, 6.442 um, 6.283 um |

A and C have three-point means/maxima only because all three targets succeeded. B retains P1 `FAIL: TARGET_NOT_VISIBLE` and P2/P3 `NOT_RUN`; its independent acceleration audit failed and is not described as safe braking.

## OCT interpretation

The OCT-like records are `mujoco_ideal_registered_oct_simulation` depth in metres. They are ideal rendered depth registered to a MuJoCo camera, not physical OCT, and do not validate hardware registration or real-time 20 Hz operation. Full depth is retained only as one final frame per phase; the JSONL audit preserves the sequence metadata.

## Figures and raw data

Figures are exported as both PNG and PDF. Figure 5 uses four byte-preserved C P2 RGB snapshots. Figure 6 shows C P3 final RGB, `registered_feature_depth_m`, and `feature_valid_mask` with the frame ID and simulation warning in the caption.

The figure generation follows the Scientific Visualization and Matplotlib skills' integrity guidance (raw data retained, missing states explicit, units labelled). For the skill reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills*, arXiv:2609.00065, DOI 10.48550/arXiv.2609.00065.

## Reproduction

Run `python reports/20261022/oct_three_group_79afacc/verify_supplement.py` from the repository root to audit this published branch copy. The retained `build_supplement.py` is a provenance build tool; regenerating the package from scratch requires the original local evidence and `runs/oct_three_group_79afacc_20261022/` directories, which are not committed to this branch. It does not run navigation or RRT/PRM.

Do not infer hardware performance, real-time behavior, generalization, training performance, or statistical significance from this three-episode pilot.
