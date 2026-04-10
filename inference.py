"""
Baseline inference for Support Triage OpenEnv — OpenAI-compatible chat API.

Required env (hackathon):
  API_BASE_URL, MODEL_NAME, HF_TOKEN (API key)
Optional:
  IMAGE_NAME or LOCAL_IMAGE_NAME — Docker image for from_docker_image()
  SUPPORT_TRIAGE_BASE_URL — HTTP URL of running env (skips Docker when set)
  SUPPORT_TRIAGE_TASKS — comma-separated tasks (default: all three)
  SUPPORT_TRIAGE_SEED — int seed for ticket selection (default: 0)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from support_triage_env import SupportTriageAction, SupportTriageEnv
from support_triage_env.server.graders import extract_json_from_text

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("SUPPORT_TRIAGE_BENCHMARK", "support_triage_env")
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
BASE_URL = os.getenv("SUPPORT_TRIAGE_BASE_URL") or "https://roushan1889-support-triage-env.hf.space"
DEFAULT_TASKS = ("ticket_category", "ticket_priority", "full_resolution", "escalation_detection")
TASKS = tuple(
    t.strip()
    for t in os.getenv("SUPPORT_TRIAGE_TASKS", ",".join(DEFAULT_TASKS)).split(",")
    if t.strip()
)
SEED = int(os.getenv("SUPPORT_TRIAGE_SEED", "0"))

MAX_STEPS_CAP = 16
TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.75

# Valid values shown to model in system prompt for grounding
VALID_CATEGORIES = ("billing", "technical", "account")
VALID_PRIORITIES = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer support triage agent. Your job is to classify support tickets.

    RULES:
    1. Respond with EXACTLY ONE JSON object — no markdown, no extra text.
    2. Valid category values: billing, technical, account
    3. Valid priority values: low, medium, high
       - If you don't know the priority, USE YOUR TOOLS first! 
       - Tool: {"tool_call": "check_customer_tier", "tool_args": "{\\"customer_id\\": \\"CUST-XXXX\\"}"} 
       - Tool: {"tool_call": "check_system_status", "tool_args": "{}"}
    4. For escalation_detection task, also include "escalate": "yes" or "no"
       Escalate to human (yes) when: security breach, production outage for many users,
       financial disputes, legal/ownership transfers, or compliance deadlines.
       Enterprise customers usually require escalation for non-trivial issues.
    5. IMPORTANT: After each step you receive feedback. If feedback says your answer
       is WRONG or earns no reward, you MUST try a DIFFERENT value next turn.
       NEVER repeat a value that has already been rejected.
    6. If you invoke a tool_call, DO NOT output category/priority in the same step.
    7. Base your answer on the NATURE of the request and the tool feedback.

    Example tool call:
      {"tool_call": "check_customer_tier", "tool_args": "{\\"customer_id\\": \\"CUST-1234\\"}"}

    Example final outputs:
      Simple: {"category":"account","priority":"low"}
      With reply: {"category":"billing","priority":"high","reply":"We are sorry; we will refund you."}
      With escalation: {"category":"technical","priority":"high","escalate":"yes"}
    """
).strip()


def build_initial_message(obs: Any) -> str:
    """First user message — full ticket context + task instruction."""
    return textwrap.dedent(
        f"""
        === TICKET ===
        Subject: {obs.ticket_subject}
        Body: {obs.ticket_body}

        === TASK ===
        Task ID: {obs.task_name}
        Instruction: {obs.instruction}
        Step: {obs.step_index + 1}/{obs.max_steps}

        Analyze the ticket carefully and respond with the correct JSON fields.
        """
    ).strip()


def build_followup_message(obs: Any, last_reward: float) -> str:
    """Subsequent user messages — include feedback and push model to fix mistakes."""
    if last_reward <= 0.0:
        reward_comment = (
            "ZERO reward — your last answer was INCORRECT. "
            "You MUST change at least one field. Think carefully: what does this ticket actually need?"
        )
    elif last_reward < 0.5:
        reward_comment = (
            "Partial credit. Some fields are still wrong. "
            "Read the feedback and improve your answer."
        )
    else:
        reward_comment = "Good progress! Refine if needed to reach a perfect score."

    return textwrap.dedent(
        f"""
        Feedback from last step: {reward_comment}

        === ENVIRONMENT FEEDBACK ===
        {obs.feedback}

        === YOUR CURRENT SUBMISSION ===
        {obs.submission_json}

        Step: {obs.step_index}/{obs.max_steps}
        Valid categories: {", ".join(VALID_CATEGORIES)}
        Valid priorities: {", ".join(VALID_PRIORITIES)}

        Respond with an improved JSON — fix any incorrect fields.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Stateful episode agent — maintains conversation history per episode
# ---------------------------------------------------------------------------

class EpisodeAgent:
    """One agent instance per episode; retains full conversation history."""

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._last_action_text: Optional[str] = None
        self._stagnation_count: int = 0

    def get_action(self, obs: Any, last_reward: Optional[float] = None) -> SupportTriageAction:
        # Build next user message
        if last_reward is None:
            user_msg = build_initial_message(obs)
        else:
            user_msg = build_followup_message(obs, last_reward)
        self.history.append({"role": "user", "content": user_msg})

        # Escalate temperature when stuck repeating a wrong answer
        temperature = TEMPERATURE
        if self._stagnation_count >= 2:
            temperature = min(0.9, TEMPERATURE + 0.2 * self._stagnation_count)
            hint = (
                f"[HINT] You have submitted the same answer {self._stagnation_count} "
                f"consecutive times with no improvement. STOP. Pick a completely "
                f"different category from: {list(VALID_CATEGORIES)}."
            )
            self.history.append({"role": "user", "content": hint})

        try:
            comp = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.history,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            text = (comp.choices[0].message.content or "").strip()
        except Exception:
            # Silently fallback per hackathon stdout rules, no debug prints allowed.
            text = '{"category":"technical","priority":"medium"}'

        # Track stagnation
        if text == self._last_action_text:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
        self._last_action_text = text

        # Append assistant reply to history for next turn
        self.history.append({"role": "assistant", "content": text})

        data = extract_json_from_text(text)
        return SupportTriageAction(
            category=data.get("category"),
            priority=data.get("priority"),
            reply=data.get("reply"),
            escalate=data.get("escalate"),
            tool_call=data.get("tool_call"),
            tool_args=data.get("tool_args"),
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def action_to_log_str(action: SupportTriageAction) -> str:
    d: Dict[str, Any] = {}
    if action.category is not None:
        d["category"] = action.category
    if action.priority is not None:
        d["priority"] = action.priority
    if action.reply is not None:
        d["reply"] = action.reply
    if action.escalate is not None:
        d["escalate"] = action.escalate
    if action.tool_call is not None:
        d["tool_call"] = action.tool_call
    if action.tool_args is not None:
        d["tool_args"] = action.tool_args
    raw = json.dumps(d, ensure_ascii=False)
    raw = re.sub(r"\s+", " ", raw)
    return raw[:500]


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_one_task(client: OpenAI, task: str) -> None:
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env: Optional[SupportTriageEnv] = None
    result: Optional[Any] = None

    try:
        if BASE_URL:
            env = SupportTriageEnv(base_url=BASE_URL.rstrip("/"))
            await env.connect()
        elif IMAGE_NAME:
            env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
        else:
            raise RuntimeError(
                "Set SUPPORT_TRIAGE_BASE_URL (deployed Space) or IMAGE_NAME for local Docker."
            )

        result = await env.reset(task=task, seed=SEED)
        agent = EpisodeAgent(client)
        last_reward: Optional[float] = None

        if result.done:
            gs = getattr(result.observation, "grader_score", None)
            score = float(gs) if gs is not None else 0.0
            success = score >= SUCCESS_THRESHOLD
        else:
            for step in range(1, MAX_STEPS_CAP + 1):
                action = agent.get_action(result.observation, last_reward)
                result = await env.step(action)

                rw = float(result.reward or 0.0)
                rewards.append(rw)
                last_reward = rw
                steps_taken = step
                last_err = getattr(result.observation, "last_action_error", None)

                log_step(
                    step=step,
                    action=action_to_log_str(action),
                    reward=rw,
                    done=result.done,
                    error=last_err,
                )

                if result.done:
                    gs = getattr(result.observation, "grader_score", None)
                    score = float(gs) if gs is not None else 0.0
                    success = score >= SUCCESS_THRESHOLD
                    break

    except Exception:
        # Silently fail episode per hackathon stdout rules
        success = False
        score = 0.0
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close(): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "HF_TOKEN environment variable is required but not set. "
            "Set it before running: set HF_TOKEN=your_token"
        )
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_one_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
