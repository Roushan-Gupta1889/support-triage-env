"""Typed Action, Observation, and State models for the support ticket triage environment."""

from __future__ import annotations

from typing import Literal, Optional, Set, Dict, Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

TaskName = Literal["ticket_category", "ticket_priority", "full_resolution", "escalation_detection"]

VALID_CATEGORIES = ("billing", "technical", "account")
VALID_PRIORITIES = ("low", "medium", "high")
VALID_ESCALATION = ("yes", "no")  # should ticket be escalated to human agent?


class SupportTriageAction(Action):
    """Structured triage fields; submit any subset each step (merged into running submission)."""

    category: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(VALID_CATEGORIES)}",
    )
    priority: Optional[str] = Field(
        default=None,
        description=f"One of: {', '.join(VALID_PRIORITIES)}",
    )
    reply: Optional[str] = Field(
        default=None,
        description="Draft customer-facing reply for full_resolution task.",
    )
    escalate: Optional[str] = Field(
        default=None,
        description="For escalation_detection task: 'yes' if ticket needs human agent, 'no' otherwise.",
    )
    tool_call: Optional[str] = Field(
        default=None,
        description="Name of the tool to invoke (e.g., 'check_customer_tier', 'check_system_status')."
    )
    tool_args: Optional[str] = Field(
        default=None,
        description="JSON string of arguments for the tool."
    )


class SupportTriageObservation(Observation):
    """Observation shown to the agent after reset or each step."""

    ticket_subject: str = Field(..., description="Ticket subject line")
    ticket_body: str = Field(..., description="Ticket body text")
    task_name: str = Field(..., description="Active task id")
    instruction: str = Field(..., description="What must be satisfied to finish the episode")
    feedback: str = Field(..., description="Feedback on the last action")
    submission_json: str = Field(
        default="{}",
        description="JSON snapshot of merged category/priority/reply so far",
    )
    step_index: int = Field(0, ge=0, description="Current step index in the episode")
    max_steps: int = Field(8, ge=1, description="Maximum steps before forced termination")
    grader_score: Optional[float] = Field(
        default=None,
        description="Final task grader score in [0,1] when episode is done",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Validation or execution error for the last action (for inference logging)",
    )
    rubric_reward: Optional[float] = Field(
        default=None,
        description="RFC 004 compliant temporally-discounted rubric score",
    )


class SupportTriageState(State):
    """Server-side episode metadata."""

    task_name: str = Field(default="", description="Active task")
    ticket_id: str = Field(default="", description="Selected ticket id")
    last_grader_score: Optional[float] = Field(
        default=None,
        description="Last computed grader score",
    )
    used_tools: Set[str] = Field(
        default_factory=set,
        description="Tracks the set of tools successfully called by the agent",
    )
    customer_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fetched customer information from mock database",
    )
    system_status: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fetched system status from mock database",
    )
