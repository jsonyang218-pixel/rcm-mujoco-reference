# ILOSAC-inspired Flow v2 — 2026-11-18

## Scope

This is an audited, explicitly approximate extension of Flow Replay
actor-critic. It is **not an ILOSAC reproduction**. The implementation retains
the state-attention DiT actor, expert rectified-Flow supervision, replay,
twin-Q critics, and Q guidance through differentiable four-step sampling.

The v2 additions are:

- a frozen demonstration-policy action anchor as an explicit KL proxy;
- TD-target clipping to `[-5, 5]` to control critic-scale drift.

THMM, duration modeling, online `S_THMM` membership/expansion, Gaussian policy
entropy temperature, and exact demonstration KL are not implemented.

## Result

The strict task result remains `FAIL` because no target reaches the 0.35 µm
gate with three settled motor frames. Final-error metrics are the best so far.

| Metric | v1 baseline | Replay v1 | ILOSAC-inspired v2 |
|---|---:|---:|---:|
| Best validation loss | 0.149865 | 0.148059 | 0.148258 |
| Canonical successes | 0 / 20 | 0 / 20 | 0 / 20 |
| Final error mean | 434.397 µm | 171.028 µm | **162.215 µm** |
| Final error median | 258.933 µm | 161.439 µm | **148.591 µm** |
| Final error maximum | 2128.198 µm | **269.203 µm** | 342.324 µm |
| Per-episode minimum mean | 93.680 µm | 77.841 µm | **63.193 µm** |
| Final error ≤ 300 µm | 12 / 20 | **20 / 20** | 19 / 20 |
| Safety-limited steps | 0 | 0 | 0 |

Against v1, 18 targets improve and 2 worsen. Final-error mean and median improve
by 272.182 µm and 81.240 µm. Compared with Replay v1, v2 improves the mean,
median, and transient minimum, but has one more target above 300 µm and a worse
maximum.

## Q-scale stabilization

The previous Replay v1 policy-Q mean drifted from -2.256 to -10.455. With the
demonstration anchor and TD clipping, v2 drifts from -2.031 to -3.712.

| Epoch | Policy Q mean | Demo anchor loss | Validation loss |
|---:|---:|---:|---:|
| 0 | -2.031 | 0.002414 | 0.149015 |
| 1 | -3.031 | 0.002448 | **0.148258** |
| 2 | -3.474 | 0.002392 | 0.148312 |
| 3 | -3.712 | 0.002460 | 0.148435 |

The stabilization helped training diagnostics but did not produce a formal
terminal success.

## Evidence

- [Pipeline report](REPORT.md)
- [v1 comparison](COMPARISON.md)
- [Machine-readable summary](summary.json)
- [Per-target metrics](episodes.csv)
- [All error trajectories](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired error vs v1](evaluation/plots/final_error_vs_v1.png)
- [Training result](training/result.json)
- [Replay/assumption manifest](training/replay_manifest.json)
- [Training configuration](training/training_config.json)
- [Shard logs](evaluation/shard_logs.txt)

The 555 MB checkpoint is intentionally excluded from Git. Official ILOSAC
source status remains `not_verified`; the formula audit and paper are included
under `docs/`.
