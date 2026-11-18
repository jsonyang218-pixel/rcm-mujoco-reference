# Documentation Index

Start with the root `README.md` for the current model and primary commands.
This directory contains focused supporting documents.

| Document | Use it for |
|---|---|
| [MOTOR_DATASET.md](MOTOR_DATASET.md) | Motor execution, dataset fields, collection, and validation |
| [NAVIGATION.md](NAVIGATION.md) | Legacy navigation interfaces, targets, and acceptance checks |
| [RESTRICTED_PERCEPTION.md](RESTRICTED_PERCEPTION.md) | Restricted-observation research pipeline |
| [RESTRICTED_INPUT_PIPELINE.md](RESTRICTED_INPUT_PIPELINE.md) | A/B/C state, coarse/fine fusion, 18-D restricted vector, and policy inputs |
| [IMITATION_LEARNING_STATUS.md](IMITATION_LEARNING_STATUS.md) | Current imitation-learning maturity, baselines, closed-loop metrics, and limitations |
| [POLICY_IO_ARCHITECTURE.md](POLICY_IO_ARCHITECTURE.md) | Current v9 input tensors, encoders, guided base, diffusion residual, action aggregation, and closed-loop output |
| [V12_ARCHITECTURE.md](V12_ARCHITECTURE.md) | v12 OCT-aware input/output contract, data pipeline, diffusion equations, closed-loop execution, and current blockers |
| [REPLAY_ACTOR_CRITIC.md](REPLAY_ACTOR_CRITIC.md) | Flow-compatible replay, twin-Q training, expert Flow regularization, and deployment scope |
| [ILOSAC_SOURCE_FORMULA_AUDIT.md](ILOSAC_SOURCE_FORMULA_AUDIT.md) | Original-paper formula audit, official-source status, and ILOSAC reproduction boundaries |
| [IMAGE_PROCESSING_PIPELINE.md](IMAGE_PROCESSING_PIPELINE.md) | RGB capture, resizing, detection, training transforms, and ResNet visual encoding |
| [OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md) | Required destinations and naming for generated artifacts |
| [../TEST_LOG.md](../TEST_LOG.md) | Timestamped acceptance and regression evidence |
| [../mujoco_model/README.md](../mujoco_model/README.md) | Mesh, model, driver, and calibration details |
| [../lerobot_training/README.md](../lerobot_training/README.md) | Policy training, evaluation, and data conversion |

Run moved Python workflows from the project root as modules. For example:

```bash
cd /home/him1/mujoco_bc/rcm_4090_deploy
source .venv/bin/activate
python -m scripts.verification.verify_current_model
python -m scripts.datasets.collect_v9_dataset --help
```
