"""
Root-level server entry-point required by openenv validate.

This module re-exports the FastAPI application from the package and
provides a callable main() function so the openenv validator finds
a proper server/app.py with a runnable entry-point.
"""

from __future__ import annotations

import os

import uvicorn
from support_triage_env.server.app import app  # noqa: F401 — re-export


def main() -> None:
    """Start the Support Triage environment server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "support_triage_env.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()