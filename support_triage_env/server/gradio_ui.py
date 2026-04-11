"""Custom Gradio UI for the support triage environment."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import gradio as gr

from openenv.core.env_server.gradio_ui import get_gradio_display_title
from openenv.core.env_server.types import EnvironmentMetadata


FIELD_PLACEHOLDERS = {
    "category": "e.g. technical / billing / account",
    "priority": "e.g. low / medium / high",
    "reply": "e.g. We are checking your issue...",
    "escalate": "e.g. true / false",
    "tool_call": "e.g. check_customer_tier / check_system_status",
    "tool_args": 'e.g. {"customer_id":"CUST-9382"}',
}


def _escape_md(text: Any) -> str:
    return re.sub(r"([\\`*_\{\}\[\]()#+\-.!|~>])", r"\\\1", str(text))


def _status_summary(obs: Dict[str, Any], reward: Optional[float]) -> str:
    feedback = str(obs.get("feedback") or "")
    if reward is not None and reward >= 0.99:
        return "Category correct" if "category" in feedback.lower() else "Strong step"
    if reward is not None and reward > 0:
        return "Partial progress"
    if obs.get("last_action_error"):
        return "Action needs correction"
    return "Waiting for input"


def _format_observation(data: Dict[str, Any]) -> str:
    obs = data.get("observation", {}) if isinstance(data.get("observation"), dict) else {}
    reward = data.get("reward")
    done = data.get("done")
    summary = _status_summary(obs, reward)

    lines: List[str] = []
    lines.append("# Playground")
    lines.append("")
    lines.append(
        "Fill the fields and click Step to simulate agent decision and receive reward feedback."
    )
    lines.append("")

    task_label = obs.get("task_label") or obs.get("task_name") or "Unknown task"
    lines.append(f"**Task:** `{_escape_md(task_label)}`")
    if reward is not None:
        lines.append(f"**Reward:** `{float(reward):.2f}`")
    if done is not None:
        lines.append(f"**Done:** `{done}`")
    lines.append(f"**Status:** `{_escape_md(summary)}`")
    lines.append("")

    subject = obs.get("ticket_subject")
    body = obs.get("ticket_body")
    if subject:
        lines.append("**Ticket Subject**")
        lines.append("")
        lines.append(_escape_md(subject))
        lines.append("")
    if body:
        lines.append("**Ticket Body**")
        lines.append("")
        lines.append(_escape_md(body))
        lines.append("")

    instruction = obs.get("instruction")
    if instruction:
        lines.append("**Task Guidance**")
        lines.append("")
        lines.append(_escape_md(instruction))
        lines.append("")

    feedback = obs.get("feedback")
    if feedback:
        lines.append("**Feedback**")
        lines.append("")
        for line in str(feedback).splitlines():
            if line.strip():
                lines.append(f"- {_escape_md(line)}")
        lines.append("")

    reward_explanation = obs.get("reward_explanation")
    if reward_explanation:
        lines.append("**Reward Breakdown**")
        lines.append("")
        lines.append("```text")
        lines.append(str(reward_explanation))
        lines.append("```")
        lines.append("")

    submission_json = obs.get("submission_json")
    if submission_json:
        lines.append("**Submission Snapshot**")
        lines.append("")
        lines.append("```json")
        lines.append(str(submission_json))
        lines.append("```")

    return "\n".join(lines)


def _readme_section(metadata: Optional[EnvironmentMetadata]) -> str:
    if not metadata or not metadata.readme_content:
        return "*No README available.*"
    return metadata.readme_content


def build_support_triage_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Optional[EnvironmentMetadata],
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: Optional[str] = None,
) -> gr.Blocks:
    """Build a judge-friendly Gradio app for support triage."""

    if is_chat_env:
        raise ValueError("Support triage UI expects structured form fields, not chat mode.")

    display_title = get_gradio_display_title(metadata, fallback=title)
    readme_content = _readme_section(metadata)

    async def reset_env():
        try:
            data = await web_manager.reset_environment()
            return (
                _format_observation(data),
                json.dumps(data, indent=2),
                "Environment reset successfully. Review the task label and ticket, then submit a step.",
            )
        except Exception as exc:
            return ("", "", f"Error: {exc}")

    async def step_form(*values):
        try:
            action_data: Dict[str, Any] = {}
            for index, field in enumerate(action_fields):
                if index >= len(values):
                    break
                value = values[index]
                if value is None or value == "":
                    continue
                action_data[field["name"]] = value

            data = await web_manager.step_environment(action_data)
            return (
                _format_observation(data),
                json.dumps(data, indent=2),
                "Step complete. Reward feedback has been updated below.",
            )
        except Exception as exc:
            return ("", "", f"Error: {exc}")

    def get_state_sync():
        try:
            return json.dumps(web_manager.get_state(), indent=2)
        except Exception as exc:
            return f"Error: {exc}"

    with gr.Blocks(title=display_title) as demo:
        with gr.Row():
            with gr.Column(scale=1, elem_classes="col-left"):
                if quick_start_md:
                    with gr.Accordion("Quick Start", open=True):
                        gr.Markdown(quick_start_md)
                with gr.Accordion("README", open=False):
                    gr.Markdown(readme_content)

            with gr.Column(scale=2, elem_classes="col-right"):
                obs_display = gr.Markdown(
                    value="# Playground\n\nClick **Reset** to load a ticket."
                )
                gr.Markdown(
                    "Fill the fields and click **Step** to simulate agent decision and receive reward feedback."
                )

                with gr.Group():
                    step_inputs = []
                    for field in action_fields:
                        name = field["name"]
                        label = name.replace("_", " ").title()
                        placeholder = FIELD_PLACEHOLDERS.get(name, field.get("placeholder", ""))
                        input_component = gr.Textbox(
                            label=label,
                            placeholder=placeholder,
                            lines=3 if name in {"reply", "tool_args"} else 1,
                        )
                        step_inputs.append(input_component)

                    with gr.Row():
                        step_btn = gr.Button("Step", variant="primary")
                        reset_btn = gr.Button("Reset", variant="secondary")
                        state_btn = gr.Button("Get state", variant="secondary")
                    status = gr.Textbox(label="Status", interactive=False)
                    raw_json = gr.Code(
                        label="Raw JSON response",
                        language="json",
                        interactive=False,
                    )

        reset_btn.click(fn=reset_env, outputs=[obs_display, raw_json, status])
        step_btn.click(
            fn=step_form,
            inputs=step_inputs,
            outputs=[obs_display, raw_json, status],
        )
        state_btn.click(fn=get_state_sync, outputs=[raw_json])

    return demo
