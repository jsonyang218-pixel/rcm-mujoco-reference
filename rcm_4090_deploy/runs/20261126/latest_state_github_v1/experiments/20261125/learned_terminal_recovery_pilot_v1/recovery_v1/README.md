# Authorized launch recovery v1

Original files in the parent directory remain byte-for-byte unchanged. The original failed status/receipt/report remain historical failure evidence.

Recovery/runtime state and experiment results: `run/workflow_status.json`, `run/report.md`, `run/summary.json`, `run/local_metrics.json`, `run/model.json`, `run/trajectories/`. Startup receipts: `first_actual_start.json`, `actual_start_receipt.json`, `gnome_launch_receipt.json`. Software errors: `software_failure.json`, `run/failure.json`, `terminal.log`.

No recipe, model math, target, sampling, budget, or gate is changed. The frozen pilot module runs with only its OUT directory redirected here. The parent project TEST_LOG.md and deployment TEST_LOG.md receive appended records.
