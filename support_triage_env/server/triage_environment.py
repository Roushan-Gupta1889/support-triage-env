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

# Fixed corpus — deterministic grading (no external data).
# 15 diverse real-world support tickets across billing, technical, account.
TICKET_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "TK-1001",
        "subject": "Double charge on Pro plan",
        "body": "I was billed twice this month for the Pro subscription. Please fix and refund the duplicate.",
        "category": "billing",
        "priority": "high",
        "reply_keywords": "sorry, refund",
        "requires_escalation": True,  # financial dispute — needs human
    },
    {
        "id": "TK-1002",
        "subject": "Cannot log in after password reset",
        "body": "The login page returns an error after I reset my password. I need access today.",
        "category": "technical",
        "priority": "medium",
        "reply_keywords": "sorry, troubleshoot",
        "requires_escalation": False,
    },
    {
        "id": "TK-1003",
        "subject": "Update billing email address",
        "body": "Please change my billing contact email from old@corp.com to new@corp.com.",
        "category": "account",
        "priority": "low",
        "reply_keywords": "confirm, updated",
        "requires_escalation": False,
    },
    {
        "id": "TK-1004",
        "subject": "Invoice PDF missing line items",
        "body": "Last invoice PDF is blank for line items. Need a corrected invoice for accounting.",
        "category": "billing",
        "priority": "medium",
        "reply_keywords": "invoice, apologize",
        "requires_escalation": False,
    },
    {
        "id": "TK-1005",
        "subject": "API RATE LIMITS BLOCKING PRODUCTION — CRITICAL",
        "body": "Our integration has been hitting 429 errors since yesterday. THIS IS DOWN IN PRODUCTION FOR 500 CUSTOMERS. We need an emergency exemption NOW.",
        "category": "technical",
        "priority": "high",
        "reply_keywords": "investigate, sorry",
        "requires_escalation": True,  # production outage + large customer
    },
    {
        "id": "TK-1006",
        "subject": "Request for annual invoice for tax purposes",
        "body": "Could you please send me an annual summary invoice for 2024 for our tax filing?",
        "category": "billing",
        "priority": "low",
        "reply_keywords": "confirm, send",
        "requires_escalation": False,
    },
    {
        "id": "TK-1007",
        "subject": "Account hacked — unauthorized transactions",
        "body": "URGENT: Someone has accessed my account and made unauthorized purchases. I need this stopped immediately and all charges reversed.",
        "category": "account",
        "priority": "high",
        "reply_keywords": "sorry, secure, refund",
        "requires_escalation": True,  # security incident
    },
    {
        "id": "TK-1008",
        "subject": "Slow dashboard load times",
        "body": "My dashboard takes 15-20 seconds to load, started about 3 days ago. Other pages are fine.",
        "category": "technical",
        "priority": "medium",
        "reply_keywords": "investigate, sorry",
        "requires_escalation": False,
    },
    {
        "id": "TK-1009",
        "subject": "Cancel subscription and refund remaining balance",
        "body": "I'd like to cancel my annual subscription effective immediately and request a prorated refund for the unused months.",
        "category": "billing",
        "priority": "medium",
        "reply_keywords": "confirm, refund",
        "requires_escalation": False,
    },
    {
        "id": "TK-1010",
        "subject": "Add team members to our enterprise account",
        "body": "We need to add 5 new team members to our Enterprise plan. Please advise on the process.",
        "category": "account",
        "priority": "low",
        "reply_keywords": "confirm, assist",
        "requires_escalation": False,
    },
    {
        "id": "TK-1011",
        "subject": "Data export failing with timeout error",
        "body": "When I try to export our 3-year dataset the job fails after 30 minutes with a timeout. We have a compliance deadline next week.",
        "category": "technical",
        "priority": "high",
        "reply_keywords": "investigate, sorry",
        "requires_escalation": True,  # compliance deadline
    },
    {
        "id": "TK-1012",
        "subject": "Wrong plan tier charged this month",
        "body": "I was charged for Business tier but I downgraded to Starter last month. Please correct and refund the difference.",
        "category": "billing",
        "priority": "medium",
        "reply_keywords": "sorry, refund",
        "requires_escalation": False,
    },
    {
        "id": "TK-1013",
        "subject": "Request to transfer account ownership",
        "body": "Our company was acquired. I need to transfer the account owner from john@oldco.com to sarah@newco.com and update all billing details.",
        "category": "account",
        "priority": "medium",
        "reply_keywords": "confirm, assist",
        "requires_escalation": True,  # ownership transfer — legal/security risk
    },
    {
        "id": "TK-1014",
        "subject": "Webhook delivery failures",
        "body": "Our webhooks stopped arriving about 2 hours ago. We have checked our endpoint and it is returning 200 OK. Please investigate your delivery system.",
        "category": "technical",
        "priority": "high",
        "reply_keywords": "investigate, sorry",
        "requires_escalation": False,
    },
    {
        "id": "TK-1015",
        "subject": "How do I update my credit card?",
        "body": "I got a new credit card and need to update payment method before next billing cycle.",
        "category": "account",
        "priority": "low",
        "reply_keywords": "confirm, assist",
        "requires_escalation": False,
    },
]

TASK_MAX_STEPS: Dict[str, int] = {
    "ticket_category": 6,
    "ticket_priority": 10,
    "full_resolution": 14,
    "escalation_detection": 8,
}



class SupportTriageEnvironment(Environment):
    """Simulates triage of support tickets with three graded task modes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = SupportTriageState(episode_id=str(uuid.uuid4()), step_count=0)
        self._task: TaskName = "ticket_category"
        self._ticket: Dict[str, Any] = TICKET_CORPUS[0]
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

        idx = 0
        if seed is not None:
            idx = int(seed) % len(TICKET_CORPUS)
        elif "ticket_index" in kwargs and kwargs["ticket_index"] is not None:
            idx = int(kwargs["ticket_index"]) % len(TICKET_CORPUS)
        self._ticket = TICKET_CORPUS[idx]

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

        delta = _action_to_delta(action)
        if not delta:
            err = "Empty action: set at least one of category, priority, reply."
            return self._build_obs(
                reward=-0.05,
                done=False,
                feedback="You must submit at least one field.",
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
        # Clamp for external consumers — validator rejects exact 0.0/1.0
        exposed_score = max(0.01, min(0.99, score))
        self._state.last_grader_score = exposed_score

        done = False
        if self._state.step_count >= self._max_steps:
            done = True
        if score >= 0.999:  # use unclamped score for done detection
            done = True

        if done:
            self._state.last_grader_score = exposed_score
            obs = self._build_obs(
                reward=step_reward,
                done=True,
                feedback=f"Episode finished. Grader={exposed_score:.3f}. {fb}",
                last_error=None,
                grader_score=exposed_score,
            )
            return obs

        return self._build_obs(
            reward=step_reward,
            done=False,
            feedback=fb,
            last_error=err,
        )

    def _build_obs(
        self,
        reward: float,
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
