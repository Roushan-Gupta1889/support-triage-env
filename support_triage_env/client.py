"""WebSocket client for Support Triage environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SupportTriageAction, SupportTriageObservation, SupportTriageState


class SupportTriageEnv(EnvClient[SupportTriageAction, SupportTriageObservation, SupportTriageState]):
    """Async client; use ``async with`` or ``await env.connect()``."""

    def _step_payload(self, action: SupportTriageAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, data: Dict[str, Any]) -> StepResult[SupportTriageObservation]:
        obs = SupportTriageObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward"),
            done=data.get("done", False),
        )

    def _parse_state(self, data: Dict[str, Any]) -> SupportTriageState:
        return SupportTriageState(**data)
