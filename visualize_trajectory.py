"""Forwards to scripts/visualize_trajectory.py (canonical copy lives there)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "visualize_trajectory.py"
    if not target.is_file():
        sys.exit(f"Missing {target}")
    runpy.run_path(str(target), run_name="__main__")
