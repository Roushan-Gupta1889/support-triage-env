"""Deterministic task graders mapping submissions to scores in (0, 1) exclusive."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, cast

try:
    from ..models import TaskName, VALID_CATEGORIES, VALID_PRIORITIES, VALID_ESCALATION
except ImportError:
    from models import TaskName, VALID_CATEGORIES, VALID_PRIORITIES, VALID_ESCALATION


def _norm(s: str | None) -> str:
    if s is None:
        return ""
    return s.strip().lower()


def _clamp(score: float) -> float:
    """Clamp score to open interval (0, 1) — validator rejects exact 0.0 and 1.0."""
    return max(0.01, min(0.99, score))


def _parse_keywords(s: str) -> List[str]:
    if not s:
        return []
    return [w.strip().lower() for w in s.split(",") if w.strip()]


def grade_ticket_category(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Easy: exact category match only."""
    cat = _norm(submission.get("category"))
    if cat not in VALID_CATEGORIES:
        return 0.0
    return 1.0 if cat == _norm(ground_truth.get("category")) else 0.0


def grade_ticket_priority(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Medium: 0.5 category + 0.5 priority."""
    c = grade_ticket_category(submission, ground_truth)
    pr = _norm(submission.get("priority"))
    if pr not in VALID_PRIORITIES:
        pscore = 0.0
    else:
        pscore = 1.0 if pr == _norm(ground_truth.get("priority")) else 0.0
    return 0.5 * c + 0.5 * pscore


def grade_full_resolution(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Hard: category + priority + reply keywords (all weighted)."""
    c = 1.0 if _norm(submission.get("category")) == _norm(ground_truth.get("category")) and _norm(submission.get("category")) in VALID_CATEGORIES else 0.0
    p = 1.0 if _norm(submission.get("priority")) == _norm(ground_truth.get("priority")) and _norm(submission.get("priority")) in VALID_PRIORITIES else 0.0
    reply = (submission.get("reply") or "").casefold()
    required = _parse_keywords(ground_truth.get("reply_keywords", ""))
    if not required:
        rscore = 1.0 if (submission.get("reply") or "").strip() else 0.0
    else:
        hits = sum(1 for phrase in required if phrase and phrase in reply)
        rscore = hits / len(required)
    return 0.35 * c + 0.35 * p + 0.30 * rscore


def grade_partial(
    task: TaskName,
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Tuple[float, str]:
    """Incremental [0,1] score for reward shaping (dense signal)."""
    if task == "ticket_category":
        s = submission.get("category")
        if s is None or _norm(s) == "":
            return _clamp(0.0), "Provide category."
        if _norm(s) not in VALID_CATEGORIES:
            return _clamp(0.05), f"Invalid category; use one of: {list(VALID_CATEGORIES)}."
        g = grade_ticket_category(submission, ground_truth)
        return _clamp(g), "Category correct." if g >= 1.0 else "Category does not match ticket."

    if task == "ticket_priority":
        has_c = bool(submission.get("category") and _norm(submission.get("category")))
        has_p = bool(submission.get("priority") and _norm(submission.get("priority")))
        if not has_c and not has_p:
            return _clamp(0.0), "Submit category and/or priority."
        score = 0.0
        parts: List[str] = []
        if has_c:
            score += 0.5 * grade_ticket_category(submission, ground_truth)
            parts.append("category evaluated")
        if has_p:
            pr = _norm(submission.get("priority"))
            if pr not in VALID_PRIORITIES:
                score += 0.05
                parts.append("invalid priority label")
            else:
                ok = pr == _norm(ground_truth.get("priority"))
                score += 0.5 * (1.0 if ok else 0.0)
                parts.append("priority ok" if ok else "priority mismatch")
        return _clamp(score), "; ".join(parts)

    if task == "escalation_detection":
        parts = []
        total = 0.0
        if submission.get("category"):
            gc = grade_ticket_category(submission, ground_truth)
            total += 0.4 * gc
            parts.append(f"category={gc:.2f}")
        if submission.get("priority"):
            pr = _norm(submission.get("priority"))
            if pr in VALID_PRIORITIES:
                ok = pr == _norm(ground_truth.get("priority"))
                total += 0.3 * (1.0 if ok else 0.0)
                parts.append("priority ok" if ok else "priority mismatch")
        if submission.get("escalate"):
            esc = _norm(submission.get("escalate"))
            gt_esc = "yes" if ground_truth.get("requires_escalation") else "no"
            if esc in VALID_ESCALATION:
                ok = esc == gt_esc
                total += 0.3 * (1.0 if ok else 0.0)
                parts.append(f"escalation {'correct' if ok else f'wrong (expected {gt_esc})'}")
            else:
                parts.append(f"invalid escalate value; use 'yes' or 'no'")
        if not parts:
            return _clamp(0.0), "Provide category, priority, and/or escalate (yes/no)."
        return _clamp(total), "; ".join(parts)

    # full_resolution — weights match grade_full_resolution: 0.35 / 0.35 / 0.30
    parts = []
    total = 0.0
    if submission.get("category"):
        gc = grade_ticket_category(submission, ground_truth)
        total += 0.35 * gc
        parts.append(f"category={gc:.2f}")
    if submission.get("priority"):
        pr = _norm(submission.get("priority"))
        if pr in VALID_PRIORITIES:
            ok = pr == _norm(ground_truth.get("priority"))
            total += 0.35 * (1.0 if ok else 0.0)
            parts.append("priority ok" if ok else "priority mismatch")
    if submission.get("reply"):
        reply = (submission.get("reply") or "").casefold()
        req = _parse_keywords(ground_truth.get("reply_keywords", ""))
        if req:
            matched = [phrase for phrase in req if phrase and phrase in reply]
            missing = [phrase for phrase in req if phrase and phrase not in reply]
            hits = len(matched)
            total += 0.30 * (hits / len(req))
            kw_msg = f"reply_keywords {hits}/{len(req)}"
            if missing:
                kw_msg += f"; missing phrases: {missing} — add these to your reply"
            else:
                kw_msg += "; all keywords present"
            parts.append(kw_msg)
    if not parts:
        return _clamp(0.0), "Provide category, priority, and/or reply."
    return _clamp(total), "; ".join(parts)


def grade_escalation_detection(submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Very hard: 0.4 for correct category + 0.3 for correct priority + 0.3 for correct escalation decision."""
    c = grade_ticket_category(submission, ground_truth)
    pr = _norm(submission.get("priority"))
    p = 1.0 if pr in VALID_PRIORITIES and pr == _norm(ground_truth.get("priority")) else 0.0
    esc = _norm(submission.get("escalate"))
    gt_esc = "yes" if ground_truth.get("requires_escalation") else "no"
    e = 1.0 if esc in VALID_ESCALATION and esc == gt_esc else 0.0
    return 0.4 * c + 0.3 * p + 0.3 * e


def final_grader(task: TaskName, submission: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    if task == "ticket_category":
        return _clamp(grade_ticket_category(submission, ground_truth))
    if task == "ticket_priority":
        return _clamp(grade_ticket_priority(submission, ground_truth))
    if task == "escalation_detection":
        return _clamp(grade_escalation_detection(submission, ground_truth))
    return _clamp(grade_full_resolution(submission, ground_truth))


def normalize_submission_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Merge pydantic / dict action into plain dict."""
    out: Dict[str, Any] = {}
    if raw.get("category") is not None:
        out["category"] = cast(str, raw["category"])
    if raw.get("priority") is not None:
        out["priority"] = cast(str, raw["priority"])
    if raw.get("reply") is not None:
        out["reply"] = cast(str, raw["reply"])
    if raw.get("escalate") is not None:
        out["escalate"] = cast(str, raw["escalate"])
    return out


def merge_submission(
    prior: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    m = dict(prior)
    for k, v in delta.items():
        if v is not None:
            m[k] = v
    return m


def submission_to_json(sub: Dict[str, Any]) -> str:
    return json.dumps(sub, ensure_ascii=False, sort_keys=True)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON object extraction from an LLM response."""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}
