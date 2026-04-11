"""
OpenEnv multi-mode validator expects this file at server/app.py.

The real FastAPI app lives in support_triage_env.server.app.
"""

from __future__ import annotations

from support_triage_env.server.app import app, main as _serve_main

__all__ = ["app", "main"]


def main() -> None:
    """Uvicorn entry (openenv validate requires def main in this file)."""
    _serve_main()


if __name__ == "__main__":
    main()
