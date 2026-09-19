#!/usr/bin/env python3
"""Train SAC/PPO for the independent Case 4–7 environment."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the script work when launched from the project root without PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
src = PROJECT_ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from tree_reinforcement_learning.cli import train  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(train())
