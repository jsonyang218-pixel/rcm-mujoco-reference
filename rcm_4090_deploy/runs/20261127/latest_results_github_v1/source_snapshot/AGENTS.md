# Repository Guidelines

## Project Structure & Module Organization

The deployable project is in `rcm_4090_deploy/`; run commands from that directory.

- The importable `rcm/` package implements modeling, navigation, visual servoing, and perception; keep `run_reference.py` as the single root quick-start entry point.
- `scripts/` groups dataset, application, research, tooling, and verification workflows; run these from `rcm_4090_deploy/` with `python -m scripts...`.
- `docs/` contains focused supporting documentation; start with `docs/README.md`.
- `restricted_perception/` contains the importable restricted-observation package.
- `lerobot_training/` contains dataset conversion, training, and evaluation tools.
- `mujoco_model/` contains model builders, assets, validation tools, and the pinned `vendor/simulation_driver/` copy.
- `configs/` holds experiment and split definitions; `data/` and `outputs/` hold checked-in datasets and evaluation artifacts.

Do not rewrite data manifests, frozen outputs, or the vendored driver unless the task explicitly requires it.

## GPU Execution Policy

Prefer the NVIDIA GeForce RTX 4090 for CUDA-capable work. Run project commands
inside the deployment virtual environment, which pins physical GPU 0:

```bash
cd rcm_4090_deploy
source .venv/bin/activate
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Use CUDA for model training, inference, and other compute-heavy tensor work.
Use `MUJOCO_GL=egl` for headless MuJoCo rendering so it runs through NVIDIA EGL;
keep GLFW for interactive MuJoCo windows. Preserve the training entry point's
CUDA AMP defaults unless a numerical-parity comparison requires otherwise.

Fall back to CPU only when the GPU is unavailable, the task is small or
I/O-bound enough that CPU is clearly more appropriate, or the user explicitly
requests CPU. Do not silently replace the CUDA-enabled PyTorch installation
with a CPU-only build. State any CPU fallback and its reason in the result.

## Output Organization Policy

Every generated artifact must have an explicit, purpose-specific destination.
Do not write generated files to the repository root, source-package
directories, or arbitrary working-directory locations.

- Runtime runs, smoke tests, verification reports, renders, videos, logs, and
  debug artifacts go under `runs/<YYYYMMDD>/<purpose>/`, for example
  `runs/20261112/verification/strict_rgbd/`.
- Retained training, inference, and experiment deliverables go under
  `outputs/<experiment>/`. Use a version or date in the experiment name when
  results may be compared or regenerated.
- Generated datasets go under `data/<dataset>/`.
- Human-authored documentation goes under `docs/`; do not place generated
  reports there.
- Disposable probes may use `/tmp/rcm_<purpose>_<unique>/` and must be removed
  after use unless the user asks to retain them.
- Never place JSON, CSV, NPY/NPZ, PNG, MP4, HTML, logs, checkpoints, or caches
  in `rcm/`, `scripts/`, `mujoco_model/`, `restricted_perception/`,
  `lerobot_training/`, or the deployment root unless the task explicitly
  requires a checked-in source asset.
- Pass an explicit `--output-dir` or equivalent path for every run. If a tool
  only writes to its current directory or a default location, add or use an
  output option, or run it inside a dedicated temporary directory.
- Do not overwrite an existing result directory unless the tool has an explicit
  resume/overwrite workflow. Record the chosen destination in the final
  response and append meaningful acceptance results to `TEST_LOG.md`.

## Experiment Workflow Policy

Apply the [Chinese Experiment SOP](rcm_4090_deploy/docs/EXPERIMENT_SOP.md)
to every experiment request, whether it arrives as a chat sentence, an
experiment card, or a multi-stage plan. No explicit SOP opt-in is needed.

- **Plan only:** document the hypothesis, main change, data and checkpoints,
  baseline, evaluation tier, budget, gates, and output paths. Do not launch
  any experiment.
- **Execute an existing plan:** treat it as the card. Preserve its parameters,
  order, and branch conditions. Preflight assets, configuration, baseline, and
  destinations; ask only about blocking gaps or conflicts. Automatically run
  the authorized stages. Record failed gates and skip dependent work; do not
  change parameters, retry, or expand scope without authorization.
- **New training and test round:** after preflight, launch once in a new GNOME
  Terminal. Training and validation lead into closed-loop evaluation when the
  gates pass. Show and save real logs and print stage result paths. If the
  terminal cannot open, report the error; never fall back to a background job.
  After launch, do not wait for the run, poll it, or send unsolicited stage
  updates. Read one status snapshot only when the user messages again.

Use the SOP's smoke, fixed 10-target screen, and full 20-target development
confirmation tiers. Reuse only complete, provenance-matched evaluations.
Report workflow completion, metric improvement, and strict success separately.

## Experiment Result Recording Policy

Every experiment result must be recorded. This includes dataset collection,
training, offline evaluation, closed-loop evaluation, ablations, smoke tests,
baseline comparisons, and failed or interrupted runs.

- Record every completed, failed, or interrupted experiment in `TEST_LOG.md`
  immediately after the run reaches a reportable state. Do not leave results
  only in console output, chat, temporary files, or an unindexed artifact.
- Each record must include a timestamp, experiment/version or purpose,
  command or config reference, dataset and checkpoint identifiers where
  applicable, result/status, key metrics or failure reason, and the output
  directory or machine-readable result path.
- Preserve structured evidence in the designated `runs/`, `outputs/`, or
  `data/` location. For interrupted or partially completed runs, record the
  completed scope and mark the result explicitly as partial or interrupted;
  never present it as a final result.
- Paired or baseline comparisons must record the baseline model, seeds,
  target split, evaluation protocol, and comparison metrics.
- Never overwrite or silently rewrite an earlier experiment record. Add a new
  entry that references the earlier result when correcting an interpretation.

## Build, Test, and Development Commands

Install the core or optional mesh dependencies:

```bash
cd rcm_4090_deploy
python -m pip install -r requirements.txt
python -m pip install -r requirements-mesh.txt  # optional
```

Run the current model locally or headlessly:

```bash
python run_reference.py
MUJOCO_GL=egl python run_reference.py --no-viewer
python run_reference.py --verify
python -m scripts.verification.verify_current_model
```

Use `MUJOCO_GL=egl` and `MPLBACKEND=Agg` on headless Linux. Supporting
workflows live under `scripts/` and are invoked as modules from
`rcm_4090_deploy/`; see `scripts/README.md` for the category index. The
unit-suite command recorded in `TEST_LOG.md` is
`python -m unittest discover -v -s tests`, but `tests/` is absent from this
deployment checkout. Run the relevant standalone module under
`scripts/verification/` until that directory is restored. Syntax-check edits
with `python -m py_compile path/to/file.py`.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Match the surrounding style; older scripts are more compact than newer typed modules. Include units in identifiers such as `_mm`, `_m`, `_deg`, and `_rad`. No formatter, linter, or type-checker is configured.

## Testing Guidelines

Most regression checks are standalone scripts in `scripts/verification/`.
Name new classes `*Tests` and methods `test_*`; run the narrowest applicable
check first, then the related model or navigation verification. No coverage
threshold is configured. Follow the Experiment Result Recording Policy for
every run, including non-acceptance and interrupted experiments.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace. Use short imperative subjects, for example `Fix strict RGB-D tip visibility check`, with one logical change per commit. Pull requests should explain the change, list verification commands and results, note model/data compatibility impacts, and link relevant issues. Include screenshots or rendered comparisons for visual-model changes.

## Simulation Safety

Results validate software behavior, not clinical accuracy or hardware readiness. Keep control inputs explicit, write new results to a new output directory, and state simulation assumptions in reports.
