# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python project for RCM kinematics, MuJoCo simulation, visual servoing, OCT simulation, and waypoint navigation. Core modules use the `rcm_*.py` naming pattern; executable entry points are `run_*.py`. Verification programs live at the repository root as `verify_*.py`. Keep reusable logic in the matching `rcm_` module and CLI parsing in its runner. `config.json` and `scene.xml` define the default mechanism and scene, while `examples/` contains target fixtures. Research utilities belong under `research/`; curated experiment evidence belongs under `reports/YYYYMMDD/`. Do not commit ignored local output such as `runs/`, `acceptance/`, `preview/`, virtual environments, or generated `verification*.json` files.

## Build, Test, and Development Commands

Use Python 3.11 or newer and run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_navigation.py --no-viewer
.\.venv\Scripts\python.exe run_visual_servo.py --no-viewer
.\.venv\Scripts\python.exe -m compileall -q .
```

The first two commands create the environment. The runners exercise headless navigation and visual control; `compileall` provides a quick syntax check. Use `--help` for scenario-specific options. GUI runs require a desktop and should not be the only validation evidence.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Prefer `pathlib.Path`, type hints, small pure helpers, and explicit units in names such as `_mm`, `_deg`, and `_um`. Preserve module docstrings and keep safety assumptions or simulation limitations explicit. No formatter or linter is configured, so avoid unrelated reformatting.

## Testing Guidelines

Tests use standalone verification scripts and Python `unittest`; files are named `verify_<feature>.py` and test classes end in `Tests` or `Checks`. Run the narrowest relevant check first, then broader checks when shared kinematics or control code changes:

```powershell
.\.venv\Scripts\python.exe verify_visual_motion.py
.\.venv\Scripts\python.exe verify_navigation.py --output-dir acceptance
.\.venv\Scripts\python.exe run_reference.py --verify --report verification.json
```

After every test run, append the command, outcome, date, and important limits or artifacts to `TEST_LOG.md`. Never describe a simulation result as hardware validation.

## Commit & Pull Request Guidelines

This checkout contains no Git history, so no established commit format can be inferred. Use concise, imperative subjects such as `Fix visual depth rejection`. Pull requests should explain the affected workflow, list exact verification commands and outcomes, link relevant issues, and include report paths or screenshots only when behavior or rendering changed. Call out configuration, coordinate-frame, unit, and safety-assumption changes explicitly.
