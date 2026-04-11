"""FastAPI entrypoint for Support Triage OpenEnv."""

from __future__ import annotations

import os

from fastapi.responses import RedirectResponse
from openenv.core.env_server import create_app

try:
    # when running as package
    from support_triage_env.models import SupportTriageAction, SupportTriageObservation
    from support_triage_env.server.triage_environment import SupportTriageEnvironment
except ImportError:
    # fallback (local dev)
    from models import SupportTriageAction, SupportTriageObservation
    from server.triage_environment import SupportTriageEnvironment

app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage_env",
)


@app.get("/", include_in_schema=False, response_model=None)
def _space_root():
    # This route is registered after create_app() and overrides OpenEnv's default
    # root. When the Gradio UI is enabled, send judges to the Playground; otherwise
    # keep a tiny JSON index for API-only runs.
    if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes"):
        return RedirectResponse(url="/web/")
    return {
        "service": "support_triage_env",
        "health": "/health",
        "docs": "/docs",
        "reset": "POST /reset",
        "playground": "set ENABLE_WEB_INTERFACE=true then open /web/",
    }


def main() -> None:
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()