# Output Layout Policy

Generated artifacts must be grouped by purpose. They must not be written to the
deployment root or mixed into source packages.

| Artifact type | Destination |
|---|---|
| Smoke tests, verification reports, renders, videos, logs, debug output | `runs/YYYYMMDD/<purpose>/` |
| Retained checkpoints and experiment deliverables | `outputs/<experiment>/` |
| Generated datasets | `data/<dataset>/` |
| Human-authored documentation | `docs/` |
| Disposable probes | `/tmp/rcm_<purpose>_<unique>/`, removed after use |

Rules:

1. Every command must use an explicit output directory.
2. Use a new directory unless the command supports a deliberate resume or
   overwrite workflow.
3. Keep a report, summary, or manifest in each retained run directory.
4. Never put generated JSON, CSV, arrays, images, videos, logs, checkpoints,
   or caches in the deployment root or source packages.
5. Record meaningful acceptance runs in `TEST_LOG.md`.

Example:

```bash
cd /home/him1/mujoco_bc/rcm_4090_deploy
source .venv/bin/activate
mkdir -p runs/20261112/verification/strict_rgbd
python -m scripts.verification.verify_strict_rgbd \
  --output-dir runs/20261112/verification/strict_rgbd
```
