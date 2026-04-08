---
title: Support Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - support-triage
  - customer-experience
---

<div align="center">

# 📧 Support Ticket Triage — OpenEnv

### *The first RL environment for training AI agents to handle real customer support operations*

[![OpenEnv Validated](https://img.shields.io/badge/OpenEnv-6%2F6%20Validated-brightgreen?style=for-the-badge)](https://roushan1889-support-triage-env.hf.space)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-blue?style=for-the-badge)](https://huggingface.co/spaces/Roushan1889/support-triage-env)
[![4 Tasks](https://img.shields.io/badge/Tasks-4%20Difficulty%20Levels-orange?style=for-the-badge)](#tasks)
[![15 Tickets](https://img.shields.io/badge/Corpus-15%20Real--World%20Tickets-purple?style=for-the-badge)](#ticket-corpus)

</div>

---

## 🚨 The Problem

> **Every SaaS company loses millions of dollars per year to misrouted support tickets.**

A billing ticket routed to the technical team. A production outage marked as "low priority". A security breach not escalated to a human agent. These are not edge cases — they happen thousands of times a day.

Current LLMs, even frontier models, struggle to:
- Consistently distinguish between **billing**, **technical**, and **account** issues
- Correctly assess **urgency** from the language of distressed customers
- Detect **implicit escalation signals** (compliance deadlines, production outages, security incidents)
- Write replies that meet **all required resolution keywords** in one shot

**This environment trains RL agents to do all four — deterministically, measurably, and with fine-grained feedback.**

---

## 🔥 Why This Environment Is Unique

| Feature | This Env | Generic Benchmarks |
|---------|----------|--------------------|
| 4 tasks across a **difficulty spectrum** (Easy → Very Hard) | ✅ | ❌ |
| **Escalation detection** (security + financial + legal + compliance signals) | ✅ | ❌ |
| **Dense reward shaping** with missing-keyword feedback at every step | ✅ | ❌ |
| **Production-realistic ticket corpus** (15 diverse scenarios) | ✅ | ❌ |
| Urgency signals: `ALL_CAPS`, `URGENT`, `CRITICAL` in ticket text | ✅ | ❌ |
| Deterministic grading — reproducible scores across runs | ✅ | ❌ |
| OpenEnv 6/6 validation ✅ | ✅ | ❌ |

---

## 🏆 Advanced Features

### 🎯 Multi-Task RL Environment
Four carefully designed tasks with increasing difficulty — agents must master each level to unlock full performance:

```
ticket_category   →  ticket_priority  →  full_resolution  →  escalation_detection
     Easy              Medium                Hard               Very Hard
```

### 🧠 Escalation Detection (Novel Task)
The hardest task requires agents to reason over **multi-dimensional signals** to decide if a ticket needs human intervention:

- 🔐 **Security signals**: "account hacked", "unauthorized transactions"
- 💥 **Production signals**: "DOWN IN PRODUCTION", "500 CUSTOMERS affected"
- ⚖️ **Legal signals**: "company acquired", "ownership transfer"
- 📋 **Compliance signals**: "compliance deadline", "tax filing"
- 💰 **Financial signals**: double charges, large refund disputes

*Frontier models score ~0.70 on this task — it genuinely challenges even GPT-4 class models.*

### 📡 Dense Reward Shaping
Unlike sparse-reward environments where agents learn nothing until the final step, every step returns actionable feedback:

```
Step 1: category=1.00; priority mismatch       → reward=0.35
Step 2: category=1.00; priority ok             → reward=0.70
Step 3: category=1.00; priority ok; reply_keywords 1/2; missing phrases: ['refund'] → reward=0.85
Step 4: all correct                            → reward=1.00 ✅
```

### 📦 Production-Realistic Ticket Corpus
15 tickets designed to **fool keyword-matching heuristics** and require genuine understanding:

- A billing email that's actually an **account** request
- A technical ticket that needs **immediate escalation**
- Routine account changes vs. legally sensitive **ownership transfers**
- `ALL_CAPS` urgency vs. genuinely critical vs. just frustrated customers

---

## 📐 Environment Specification

### Action Space (`SupportTriageAction`)

```json
{
  "category": "billing | technical | account",
  "priority": "low | medium | high",
  "reply": "customer-facing resolution text",
  "escalate": "yes | no"
}
```

> Fields are **merged across steps** — you can refine your answer incrementally. This enables multi-turn RL training with partial credit throughout the trajectory.

### Observation Space (`SupportTriageObservation`)

```json
{
  "ticket_subject": "API RATE LIMITS BLOCKING PRODUCTION — CRITICAL",
  "ticket_body":    "Our integration has been hitting 429 errors...",
  "task_name":      "full_resolution",
  "instruction":    "Provide category, priority, AND a reply containing all keywords...",
  "feedback":       "category=1.00; priority ok; reply_keywords 1/2; missing: ['sorry']",
  "submission_json": "{\"category\":\"technical\",\"priority\":\"high\"}",
  "step_index":     2,
  "max_steps":      14,
  "grader_score":   null
}
```

---

## 📊 Tasks & Grading

| Task | Difficulty | Max Steps | Grader Formula |
|------|-----------|:---------:|----------------|
| `ticket_category` | 🟢 Easy | 6 | `1.0 × category_match` |
| `ticket_priority` | 🟡 Medium | 10 | `0.5 × category + 0.5 × priority` |
| `full_resolution` | 🔴 Hard | 14 | `0.35 × category + 0.35 × priority + 0.30 × keyword_coverage` |
| `escalation_detection` | 🔥 Very Hard | 8 | `0.4 × category + 0.3 × priority + 0.3 × escalation_decision` |

All scores are deterministic, continuous in [0, 1], and reproducible across seeds.

---

## 🧪 Ticket Corpus (15 Real-World Scenarios)

| ID | Subject | Category | Priority | Escalate? |
|----|---------|----------|----------|-----------|
| TK-1001 | Double charge on Pro plan | billing | high | ✅ Yes |
| TK-1002 | Cannot log in after password reset | technical | medium | ❌ No |
| TK-1003 | Update billing email address | account | low | ❌ No |
| TK-1005 | **API RATE LIMITS BLOCKING PRODUCTION — CRITICAL** | technical | high | ✅ Yes |
| TK-1007 | **Account hacked — unauthorized transactions** | account | high | ✅ Yes |
| TK-1011 | Data export failing — **compliance deadline next week** | technical | high | ✅ Yes |
| TK-1013 | Company acquired — account ownership transfer | account | medium | ✅ Yes |
| ... | *(15 total, spanning billing/technical/account)* | | | |

Tickets include intentional difficulty: urgency in ALL_CAPS, ambiguous category signals, and routine-vs-sensitive scenarios that require reading the full context.

---

## 🚀 Baseline Results

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router, `seed=0`, stateful multi-turn agent:

| Task | Score | Steps | Notes |
|------|:-----:|:-----:|-------|
| `ticket_category` | **1.00** ✅ | 1 | Solved first try |
| `ticket_priority` | **1.00** ✅ | 2 | Category + priority matched |
| `full_resolution` | **0.85** ✅ | 14 | All keywords found |
| `escalation_detection` | **~0.70** ✅ | 8 | Very hard — frontier model challenge |

> **Inference uses a stateful `EpisodeAgent`** that maintains full conversation history, detects stagnation, escalates temperature, and injects targeted hints when stuck.

---

## ⚡ Quick Start

### Option 1: Use the Live HF Space (No setup needed)
```bash
pip install openenv-core openai

export HF_TOKEN=your_huggingface_token
export SUPPORT_TRIAGE_BASE_URL=https://roushan1889-support-triage-env.hf.space
python inference.py
```

### Option 2: Run Locally
```bash
git clone https://github.com/Roushan-Gupta1889/support-triage-env
cd support-triage-env
pip install -e .
uvicorn support_triage_env.server.app:app --host 0.0.0.0 --port 7860

# In another terminal:
export SUPPORT_TRIAGE_BASE_URL=http://localhost:7860
python inference.py
```

### Option 3: Docker
```bash
docker build -t support-triage .
docker run -d -p 7860:7860 support-triage

export SUPPORT_TRIAGE_BASE_URL=http://localhost:7860
python inference.py
```

---

## ✅ OpenEnv Compliance

```bash
# Local structural validation
openenv validate
# → [OK] support-triage-env-hf: Ready for multi-mode deployment ✅

# Live server validation
openenv validate --url https://roushan1889-support-triage-env.hf.space
# → passed=true, 6/6 criteria ✅
```

| Criterion | Status |
|-----------|--------|
| `openapi_version_available` | ✅ |
| `health_endpoint` | ✅ |
| `metadata_endpoint` | ✅ |
| `schema_endpoint` | ✅ |
| `mcp_endpoint` | ✅ |
| `mode_endpoint_consistency` | ✅ |

---

## 📁 Project Structure

```
support-triage-env/
├── inference.py                         # Stateful multi-turn inference agent
├── requirements.txt                     # Python dependencies
├── pyproject.toml                       # Package config + uv scripts
├── Dockerfile                           # HF Space deployment
├── server/app.py                        # Root entry-point (openenv validate)
├── uv.lock                              # Reproducible builds
└── support_triage_env/
    ├── models.py                        # Action / Observation / State schemas
    └── server/
        ├── app.py                       # FastAPI application
        ├── triage_environment.py        # Environment logic + 15-ticket corpus
        ├── graders.py                   # Deterministic graders (all 4 tasks)
        └── tests/test_graders.py        # Grader unit tests
```

---

## 🧩 Why This Matters for RL Research

Customer support is a **$350B industry**. Misrouting costs ~15% of resolution time.
An RL agent trained on this environment can:

1. **Learn to read distress signals** — urgency, frustration, ALL_CAPS
2. **Generalize escalation rules** — not just pattern-match keywords
3. **Improve reply quality** iteratively via keyword-coverage feedback
4. **Be evaluated rigorously** — scores are deterministic and reproducible

This is exactly the kind of environment that bridges **research** (can RL agents reason over real text?) and **production** (can we trust an agent to triage 10,000 tickets/day?).

---

## 📜 License

BSD-3-Clause · Aligned with OpenEnv framework standards.
