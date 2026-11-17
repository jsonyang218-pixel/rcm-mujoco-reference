# State-conditioned temporal attention v1 — 2026-11-17

## Status

The state-conditioned temporal attention policy is implemented, integration
tested, and fine-tuned from the existing DiT acthistory checkpoint. Offline
validation loss improved relative to the warm-start baseline.

The 20-target, 0.35 µm closed-loop development evaluation is complete. The
policy improves final error substantially but does not satisfy the strict
terminal success requirement: the result is `FAIL` with 0/20 successes.

| Item | Result |
|---|---:|
| Architecture | `state_conditioned_temporal_attention_v1` |
| Warm-start checkpoint | acthistory DiT, epoch 3 |
| Warm-start validation loss | 0.1615405332 |
| Fine-tuned epoch | 3 |
| Fine-tuned training loss | 0.1142573576 |
| Fine-tuned validation loss | 0.1498646501 |
| Validation improvement | 7.227835% |
| New attention parameters | 83,136 |
| Training duration | 979.608 s |
| Closed-loop result | `FAIL` |
| Canonical successes | 0 / 20 |
| Final error mean / median | 434.397 / 258.933 µm |
| Per-episode minimum error mean | 93.680 µm |
| Paired final-error improvement | 18 improved / 2 worsened |
| Safety-limited steps | 0 |

## Closed-loop result

The evaluation completed in 3,099 seconds with all 10 shards exiting
successfully. No episode reached the 0.35 µm gate, even transiently. The best
per-episode minimum error was 45.103 µm, so the model approaches the target
but remains outside the strict threshold.

| Metric | acthistory baseline | state-attention v1 |
|---|---:|---:|
| Canonical successes | 0 / 20 | 0 / 20 |
| Final error mean | 1737.487 µm | **434.397 µm** |
| Final error median | 2012.021 µm | **258.933 µm** |
| Per-episode minimum mean | 143.065 µm | **93.680 µm** |
| Safety-limited steps | 0 | 0 |

Pairwise final-error comparison: 18 episodes improved and 2 worsened. Mean
and median final errors improved by 1303.090 µm and 1668.789 µm respectively.
This is a clear improvement in trajectory error, but not a successful policy
under the required settled-terminal gate.

Evaluation evidence:

- [Strict 20-target summary](evaluation/strict_summary_vs_acthist.json)
- [Evaluation status](evaluation/evaluation_status.json)
- [All 20 error curves](evaluation/plots/error_curves_all.png)
- [Per-target small multiples](evaluation/plots/error_curves_small_multiples.png)
- [Best and final error](evaluation/plots/best_and_final_error.png)
- [Paired final error vs acthistory](evaluation/plots/final_error_vs_acthist.png)

## Training

The model adds a state-conditioned additive bias to action-token
self-attention. The bias is generated from the existing geometry condition and
four relative-time templates per attention head. Visual cross-attention,
normalization, rectified-flow loss, and the sampler remain unchanged.

Training used the acthistory DAgger2 dataset with 220,844 windows and an
independent validation set with 26,650 windows.

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 0.120944 | 0.151613 |
| 1 | 0.117699 | 0.150786 |
| 2 | 0.115918 | 0.150879 |
| 3 | 0.114257 | 0.149865 |

The complete history and log are in:

- [training_history.json](training/training_history.json)
- [Training log](training/training_log.txt)
- [checkpoint_metadata.json](training/checkpoint_metadata.json)

The 396 MB `best_model.pth` is intentionally excluded from Git. Its SHA-256 is
recorded in `checkpoint_metadata.json`; the checkpoint remains in the local
output directory and should use GitHub Releases or an external artifact store
if distribution is required.

## Verification

- Original acthistory checkpoint strict-loads into the enhanced policy.
- Initial enhanced output matches the original checkpoint exactly with the new
  gate path at zero.
- Real-checkpoint GPU forward and backward passes are finite.
- The new gate receives nonzero gradient.
- `eval_policy.PolicyEvaluator` loads the enhanced checkpoint on CUDA.
- Eight real training samples complete FP16 AMP training with adaptive scaling.
- The repository unit suite passes 15/15.

Evidence:

- [Integration smoke](verification/integration_smoke.txt)
- [AMP optimizer step](verification/amp_step.txt)
- [Trained checkpoint check](verification/checkpoint_check.txt)
- [Unit tests](verification/unittest.txt)

## Closed-loop protocol

The evaluation uses the frozen `v12_reachable_development_targets_0p35um.json`
manifest and the same strict protocol as the baseline:

- 20 development targets in 10 parallel two-target shards;
- 1600-step limit;
- 0.35 µm success threshold with three consecutive settled motor frames;
- one action sample and first-step execution;
- 4-step rectified flow;
- 50 ms control period;
- no fine policy or terminal refinement.

The complete evaluation run and logs are recorded in:

- [Evaluation launch log](evaluation/eval_launch.txt)
- [Evaluation summary](evaluation/summary.txt)
- [Plot generation log](evaluation/plot_generation.txt)
- [evaluation_status.json](evaluation/evaluation_status.json)

All reported values are MuJoCo simulation results. They do not establish
hardware accuracy or clinical validity.
