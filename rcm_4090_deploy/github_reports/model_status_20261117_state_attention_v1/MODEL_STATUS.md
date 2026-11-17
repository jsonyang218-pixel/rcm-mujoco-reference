# State-conditioned temporal attention v1 — 2026-11-17

## Status

The state-conditioned temporal attention policy is implemented, integration
tested, and fine-tuned from the existing DiT acthistory checkpoint. Offline
validation loss improved relative to the warm-start baseline.

The 20-target, 0.35 µm closed-loop development evaluation was started after
training and is still running at the time of this repository snapshot. The
closed-loop metrics are therefore pending and must not be inferred from the
offline validation result.

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
| Closed-loop evaluation | running |

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
- [training.log](training/training.log)
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

- [Integration smoke](verification/integration_smoke.log)
- [AMP optimizer step](verification/amp_step.log)
- [Trained checkpoint check](verification/checkpoint_check.log)
- [Unit tests](verification/unittest.log)

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

Launch status is recorded in:

- [eval_launch.log](evaluation/eval_launch.log)
- [evaluation_status.json](evaluation/evaluation_status.json)

All reported training values are MuJoCo simulation results. They do not
establish hardware accuracy, clinical validity, or closed-loop success.
