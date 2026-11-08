# Branch Archive Cleanup

This branch keeps its GitHub-generated source archive below 100 MB by omitting
generated artifacts that can be recreated from the retained source, configs,
and datasets.

Removed from the current tree:

- `outputs/` training checkpoints and evaluation results
- root-level temporary trajectories and training/install logs
- intermediate presentation backups and preview-server logs

Source code, tests, configs, model assets, reports, and final presentation
deliverables remain in version control.
