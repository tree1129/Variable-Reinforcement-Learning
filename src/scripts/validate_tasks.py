#!/usr/bin/env python3
"""Validate the independent Case 4–7 task definitions."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the script work both as `python scripts/validate_tasks.py` and with PYTHONPATH=src.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from tree_reinforcement_learning.cli import validate  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(validate())
