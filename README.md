---
title: Support Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Support Ticket Triage (OpenEnv)

Real-world **customer support operations** benchmark: agents read support tickets and perform the tasks a CX team does every day — classifying by category, setting priority, drafting customer replies, and deciding whether a ticket requires human escalation.

> **Why does this matter?**  
> Misrouting support tickets costs companies millions annually. Training RL agents to triage tickets accurately and consistently is a real production use-case at every SaaS company. This environment provides a deterministic, reproducible benchmark for developing and evaluating CX-ops agents.

## Action space (`SupportTriageAction`)

| Field | Type | Notes |
|-------|------|--------|
| `category` | optional string | One of: `billing`, `technical`, `account` |
| `priority` | optional string | One of: `low`, `medium`, `high` |
| `reply` | optional string | Customer-facing draft reply (hard task) |
| `escalate` | optional string | `yes` or `no` — should ticket go to a human agent? (very hard task) |

Submit any subset of fields each step; values merge into a running submission.

## Observation space (`SupportTriageObservation`)

Each observation contains:
- `ticket_subject` / `ticket_body` — the raw ticket text
- `task_name` — active task identifier
- `instruction` — what the agent must satisfy to complete the episode
- `feedback` — actionable feedback including missing keywords and score components
- `submission_json` — JSON snapshot of the current merged submission
- `step_index` / `max_steps` — episode progress
- `grader_score` — final score in [0, 1] when episode ends
- `last_action_error` — validation error for the last action

## Tasks and difficulty

| Task id | Difficulty | Max Steps | Grader description |
|---------|------------|:---------:|-------------------|
| `ticket_category` | Easy | 6 | Category match vs ground truth (1.0 or 0.0) |
| `ticket_priority` | Medium | 10 | 0.5x category + 0.5x priority |
| `full_resolution` | Hard | 14 | 0.35x category + 0.35x priority + 0.30x reply keyword coverage |
| `escalation_detection` | Very Hard | 8 | 0.4x category + 0.3x priority + 0.3x correct escalation decision |

**Dense reward shaping:** Rewards are provided throughout the trajectory as partial credit improves, with small stagnation penalties and explicit feedback on missing keywords and wrong field values.

**Ticket corpus:** 15 diverse real-world tickets spanning billing disputes, production outages, account security incidents, data compliance deadlines, and routine requests — requiring agents to distinguish surface-level keywords from the true nature of the request.

## Local setup

```bash
# From repo root: support-triage-env-hf/
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 7860
```

Health: `GET http://localhost:7860/health`  
OpenAPI docs: `http://localhost:7860/docs`

## Baseline inference

```bash
set HF_TOKEN=your_token
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set SUPPORT_TRIAGE_BASE_URL=https://roushan1889-support-triage-env.hf.space
python inference.py
```

Or with Docker:

```bash
docker build -t support-triage:local .
docker run -d -p 7860:7860 support-triage:local
set SUPPORT_TRIAGE_BASE_URL=http://127.0.0.1:7860
python inference.py
```

## OpenEnv validation

```bash
pip install openenv-core
openenv validate --url https://roushan1889-support-triage-env.hf.space
# Result: passed=true, 6/6 criteria
```

## Baseline scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router (seed=0, ticket TK-1001: billing/high):

| Task | Difficulty | Grader Score | Steps | Success |
|------|------------|:------------:|:-----:|:-------:|
| `ticket_category` | Easy | **1.00** | 1 | yes |
| `ticket_priority` | Medium | **1.00** | 2 | yes |
| `full_resolution` | Hard | **0.85** | 14 | yes |
| `escalation_detection` | Very Hard | **~0.70** | 8 | yes |

> Success threshold is 0.75. Reproduce: set env vars then run `python inference.py`.

## License

BSD-3-Clause (aligned with OpenEnv).
