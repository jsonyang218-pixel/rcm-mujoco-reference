# Expert-Corrected Experience Replay

This experiment keeps the state-attention DiT rectified-Flow actor and the
existing 16-step action-block contract. Reward and next-state values are not
used in the loss. Policy-visited states are paired with expert correction
labels, while deployment history continues to use the action that was actually
executed.

## Contract

- `action` is the flow-matching supervision label.
- `executed_action` is the action used for motion history and replay
  diagnostics.
- A training window must stay inside one episode and use consecutive source
  frames at the 50 ms control period.
- Teacher-invalid or discontinuous samples are excluded rather than assigned
  the failed policy action as a target.
- Raw images are referenced by their existing dataset paths; index replay does
  not duplicate image files.

Run the contract audit with:

```bash
python scripts/verification/verify_experience_replay_contract.py \
  --data-dir data/lerobot_v12_oct_dit_rf_dagger2_acthist_20261116_224 \
  --output-dir runs/20261118/verification/experience_replay_contract
```

## A/B/C comparison

All arms start from the state-attention v1 checkpoint, use the same validation
set, execute 4 x 1,000 training updates, and evaluate the same 20 development
targets.

| Arm | Training data | Batch organization |
|---|---|---|
| A | Original supervised windows | Uniform original batches |
| B | Original + expert corrections | Uniform over the static union |
| C | Original + expert corrections | Exactly 8 original + 8 correction per 16-sample batch |

For arm C, the two equal half-batches give the original and correction halves
equal Flow-loss weight without adding a new loss term.

Run the three pipelines sequentially with an explicit destination:

```bash
bash scripts/run_experience_replay_abc.sh \
  outputs/state_attention_experience_replay_abc_YYYYMMDD_1
```

## Initial hyperparameters

These are project choices to be tested, not paper parameters:

- batch size 16, validation batch 64;
- backbone learning rate `1e-6`;
- state-attention learning rate `1e-5`;
- 4 epochs, each 1,000 updates;
- uniform sampling with replacement;
- rectified-Flow matching only;
- no critic, reward, progress loss, or action penalty.

## Acceptance

Closed-loop acceptance remains the existing `0.35 um` error gate with three
settled encoder frames. Report final-error mean, median, P90, maximum, paired
changes, interval-entry and retention behavior, exit speed, settle rate, and
safety-limited steps. Training loss alone is not evidence that replay helped.

## Sampling-ratio result

The first corrected-window ratio experiment kept the same initialization,
validation set, update budget, seed, and 20-target closed-loop protocol:

| Arm | Correction ratio | Final error mean | Improved vs v1 |
|---|---:|---:|---:|
| R0 | 6.5987% natural | 247.936 um | 19 / 20 |
| R10 | 10% | 980.699 um | 9 / 20 |
| R20 | 20% | 589.481 um | 11 / 20 |

Increasing the correction ratio above the natural rate was harmful. R0 is the
selected ratio for the next data collection. All arms still failed the strict
success gate and none ended with a settled encoder, so the remaining blocker is
fine retention and stopping behavior rather than correction frequency alone.
