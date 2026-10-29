#!/usr/bin/env python3
"""Launch the current scanned MuJoCo arm; legacy research has its own entry."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "mujoco_model"))
from run_model import main

if __name__ == "__main__":
    raise SystemExit(main())
