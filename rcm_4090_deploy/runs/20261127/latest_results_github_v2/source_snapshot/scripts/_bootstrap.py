"""Shared project paths for scripts run as modules from the project root."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
