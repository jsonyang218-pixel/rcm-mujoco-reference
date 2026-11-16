# Script Index

The root-level `run_reference.py` is the single quick-start entry point.
Navigation, visual-servo, waypoint, legacy, dataset, research, and verification
workflows live here and are invoked as modules from the project root:

```bash
cd /home/him1/mujoco_bc/rcm_4090_deploy
source .venv/bin/activate
python -m scripts.verification.verify_current_model
```

| Directory | Contents |
|---|---|
| `datasets/` | Dataset collection, random target generation, and waypoint generation |
| `applications/` | Navigation, visual-servo, waypoint, legacy, and occlusion applications |
| `tools/` | Rendering, plotting, migration validation, and maintenance utilities |
| `verification/` | Standalone model, navigation, sensor, and policy checks |
| `research/navigation/` | Retained multi-target navigation experiments |
| `research/restricted_perception/` | Stage 0-4 restricted-perception pipeline |

Use `python -m ...` rather than executing a moved file path directly. Module
execution puts the project root on `sys.path`, which preserves imports of the
`rcm` package, `restricted_perception`, and training helpers.

All scripts that generate artifacts must receive an explicit output directory.
Use `runs/YYYYMMDD/<purpose>/` for transient or verification output,
`outputs/<experiment>/` for retained experiment deliverables, and
`data/<dataset>/` for datasets. See `docs/OUTPUT_LAYOUT.md`.
