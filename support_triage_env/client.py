"""WebSocket client for Support Triage environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import SupportTriageAction, SupportTriageObservation, SupportTriageState


class SupportTriageEnv(EnvClient[SupportTriageAction, SupportTriageObservation, SupportTriageState]):
    """Async client; use ``async with`` or ``await env.connect()``."""

    def _step_payload(self, action: SupportTriageAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, data: Dict[str, Any]) -> StepResult[SupportTriageObservation]:
        # Safely extract observation sub-dict — server may vary shape
        obs_data = data.get("observation", {})

        # Unify reward and done from root payload first, then observation fallback
        reward = data.get("reward", obs_data.get("reward", 0.0))
        done = data.get("done", obs_data.get("done", False))

        obs = SupportTriageObservation(
            ticket_subject=obs_data.get("ticket_subject", ""),
            ticket_body=obs_data.get("ticket_body", ""),
            task_name=obs_data.get("task_name", ""),
            instruction=obs_data.get("instruction", ""),
            feedback=obs_data.get("feedback", ""),
            submission_json=obs_data.get("submission_json", "{}"),
            step_index=obs_data.get("step_index", 0),
            max_steps=obs_data.get("max_steps", 8),
            grader_score=obs_data.get("grader_score"),
            last_action_error=obs_data.get("last_action_error"),
            rubric_reward=obs_data.get("rubric_reward"),
            done=done,
            reward=float(reward),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, data: Dict[str, Any]) -> SupportTriageState:
        return SupportTriageState(
            episode_id=data.get("episode_id") or data.get("session_id", ""),
            step_count=data.get("step_count", 0),
            task_name=data.get("task_name", ""),
            ticket_id=data.get("ticket_id", ""),
        )
