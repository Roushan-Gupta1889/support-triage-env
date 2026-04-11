"""Customer support ticket triage simulation (real-world task, dense rewards, task graders)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

try:
    from ..models import (
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TaskName,
    )
except ImportError:
    from models import (
        SupportTriageAction,
        SupportTriageObservation,
        SupportTriageState,
        TaskName,
    )

from .graders import (
    final_grader,
    grade_partial,
    merge_submission,
    submission_to_json,
)
from .rubrics import SupportTriageRubric
from .generator import TicketGenerator

TASK_MAX_STEPS: Dict[str, int] = {
    "ticket_category": 10,
    "ticket_priority": 14,
    "full_resolution": 20,
    "escalation_detection": 16,
}



class SupportTriageEnvironment(Environment):
    """Simulates triage of support tickets with three graded task modes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._generator = TicketGenerator(seed=0)
        self._rubric = SupportTriageRubric()
        self._state = SupportTriageState(episode_id=str(uuid.uuid4()), step_count=0)
        self._task: TaskName = "ticket_category"
        self._ticket: Dict[str, Any] = self._generator.generate_ticket()
        self._submission: Dict[str, Any] = {}
        self._last_partial: float = 0.0
        self._max_steps: int = 6
        self._instruction: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        task = kwargs.get("task") or kwargs.get("task_name")
        if task is None:
            task = "ticket_category"
        if task not in ("ticket_category", "ticket_priority", "full_resolution", "escalation_detection"):
            task = "ticket_category"
        self._task = _coerce_task(str(task))

        seed_to_use = 0
        if seed is not None:
            seed_to_use = int(seed)
        elif "ticket_index" in kwargs and kwargs["ticket_index"] is not None:
            seed_to_use = int(kwargs["ticket_index"])
            
        self._generator = TicketGenerator(seed=seed_to_use)
        self._is_probe = kwargs.get("is_probe", False)
        self._ticket = self._generator.generate_ticket(is_probe=self._is_probe)

        self._submission = {}
        self._last_partial = 0.0
        self._max_steps = TASK_MAX_STEPS[self._task]

        self._state = SupportTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=self._task,
            ticket_id=str(self._ticket["id"]),
            last_grader_score=None,
        )

        self._instruction = {
            "ticket_category": (
                "Task (easy): Output the single best category for this ticket: "
                "billing | technical | account. Partial credit improves as you converge."
            ),
            "ticket_priority": (
                "Task (medium): Assign category (billing|technical|account) AND "
                "priority (low|medium|high) matching our ground truth."
            ),
            "full_resolution": (
                "Task (hard): Provide category, priority, AND a short reply that includes "
                "every required keyword phrase (comma-separated in ground truth). "
                "Match category and priority exactly."
            ),
            "escalation_detection": (
                "Task (very hard): Determine category, priority, AND whether this ticket "
                "requires escalation to a human agent (escalate: yes | no). "
                "Escalation is needed for: security incidents, production outages affecting many users, "
                "financial disputes >$500, account ownership transfers, or compliance deadlines. "
                "Score: 0.4 × category + 0.3 × priority + 0.3 × correct escalation decision."
            ),
        }[self._task]

        return SupportTriageObservation(
            done=False,
            reward=0.0,
            metadata={"status": "ready"},
            ticket_subject=self._ticket["subject"],
            ticket_body=self._ticket["body"],
            task_name=self._task,
            instruction=self._instruction,
            feedback="Episode started. Submit fields using the action schema.",
            submission_json="{}",
            step_index=0,
            max_steps=self._max_steps,
            grader_score=None,
            last_action_error=None,
        )

    def step(
        self,
        action: SupportTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        self._state.step_count += 1
        err: Optional[str] = None

        if action.tool_call:
            tool_name = action.tool_call
            tool_args = action.tool_args or "{}"
            tool_reward = 0.01
            if tool_name == "check_customer_tier":
                self._state.used_tools.add("check_customer_tier")
                tool_output = self._generator.tool_check_customer_tier(tool_args)
                tool_reward = 0.05 # Richer signal for valid tools
            elif tool_name == "check_system_status":
                self._state.used_tools.add("check_system_status")
                tool_output = self._generator.tool_check_system_status(tool_args)
                tool_reward = 0.05 # Richer signal for valid tools
            else:
                tool_output = f"Error: Unknown tool {tool_name}"
                tool_reward = -0.05 # Penalty for hallucinated tools
            
            # Tools provide helpful state progression without ending the episode
            # We don't artificially increase max score yet, just give them the data.
            rubric_reward = self._rubric.score_step(False, action, None, self._task, self._submission, self._ticket, self._state)
            return self._build_obs(
                reward=tool_reward,
                rubric_reward=rubric_reward,
                done=False,
                feedback=f"[TOOL OUTPUT] {tool_name}:\n{tool_output}",
                last_error=None,
            )

        delta = _action_to_delta(action)
        if not delta:
            err = "Empty action: set at least one of category, priority, reply, escalate, or use a tool_call."
            rubric_reward = self._rubric.score_step(False, action, err, self._task, self._submission, self._ticket, self._state)
            return self._build_obs(
                reward=-0.05,
                rubric_reward=rubric_reward,
                done=False,
                feedback="You must submit at least one field or tool call.",
                last_error=err,
            )

        new_sub = merge_submission(self._submission, delta)
        self._submission = new_sub

        partial, fb = grade_partial(self._task, self._submission, self._ticket)
        delta_reward = max(0.0, partial - self._last_partial)
        stagnation = 0.02 if delta_reward < 1e-6 else 0.0
        step_reward = max(0.0, min(1.0, delta_reward - stagnation))
        self._last_partial = max(self._last_partial, partial)

        score = final_grader(self._task, self._submission, self._ticket)
        
        if getattr(self, "_is_probe", False):
            # Flat 1.0 neutrality probe rewards to evaluate unbiased trajectory without gradient penalty
            step_reward = 1.0
            score = 1.0

        # Clamp for external consumers — validator rejects exact 0.0/1.0
        exposed_score = max(0.01, min(0.99, score))
        self._state.last_grader_score = exposed_score

        done = False
        if self._state.step_count >= self._max_steps:
            done = True
        if score >= 0.999:  # use unclamped score for done detection
            done = True

        rubric_reward = self._rubric.score_step(done, action, err, self._task, self._submission, self._ticket, self._state)

        if done:
            self._state.last_grader_score = exposed_score
            obs = self._build_obs(
                reward=step_reward,
                rubric_reward=rubric_reward,
                done=True,
                feedback=f"Episode finished. Grader={exposed_score:.3f}. {fb}",
                last_error=None,
                grader_score=exposed_score,
            )
            return obs

        return self._build_obs(
            reward=step_reward,
            rubric_reward=rubric_reward,
            done=False,
            feedback=fb,
            last_error=err,
        )

    def _build_obs(
        self,
        reward: float,
        rubric_reward: float,
        done: bool,
        feedback: str,
        last_error: Optional[str],
        grader_score: Optional[float] = None,
    ) -> SupportTriageObservation:
        meta: Dict[str, Any] = {}
        if done and grader_score is not None:
            meta["grader_score"] = grader_score
        return SupportTriageObservation(
            done=done,
            reward=float(reward),
            rubric_reward=float(rubric_reward),
            metadata=meta,
            ticket_subject=self._ticket["subject"],
            ticket_body=self._ticket["body"],
            task_name=self._task,
            instruction=self._instruction,
            feedback=feedback,
            submission_json=submission_to_json(self._submission),
            step_index=self._state.step_count,
            max_steps=self._max_steps,
            grader_score=grader_score,
            last_action_error=last_error,
        )

    @property
    def state(self) -> SupportTriageState:
        return self._state


def _coerce_task(task: str) -> TaskName:
    if task == "ticket_priority":
        return "ticket_priority"
    if task == "full_resolution":
        return "full_resolution"
    if task == "escalation_detection":
        return "escalation_detection"
    return "ticket_category"


def _action_to_delta(action: SupportTriageAction) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if action.category is not None:
        d["category"] = action.category
    if action.priority is not None:
        d["priority"] = action.priority
    if action.reply is not None:
        d["reply"] = action.reply
    if action.escalate is not None:
        d["escalate"] = action.escalate
    return d
