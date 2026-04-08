"""Deterministic grader + environment unit tests."""

from __future__ import annotations

import pytest

from support_triage_env.server.graders import (
    extract_json_from_text,
    final_grader,
    grade_full_resolution,
    grade_partial,
    grade_ticket_category,
    grade_ticket_priority,
    merge_submission,
)
from support_triage_env.server.triage_environment import SupportTriageEnvironment
from support_triage_env.models import SupportTriageAction

# ---------------------------------------------------------------------------
# Ground-truth fixtures
# ---------------------------------------------------------------------------

GT_BILLING = {
    "category": "billing",
    "priority": "high",
    "reply_keywords": "sorry, refund",
}

GT_TECHNICAL = {
    "category": "technical",
    "priority": "medium",
    "reply_keywords": "sorry, troubleshoot",
}

GT_ACCOUNT = {
    "category": "account",
    "priority": "low",
    "reply_keywords": "confirm, updated",
}


# ---------------------------------------------------------------------------
# ticket_category (easy) — grade_ticket_category
# ---------------------------------------------------------------------------

class TestTicketCategory:
    def test_perfect_match(self):
        assert grade_ticket_category({"category": "billing"}, GT_BILLING) == 1.0

    def test_wrong_category(self):
        assert grade_ticket_category({"category": "technical"}, GT_BILLING) == 0.0

    def test_invalid_category(self):
        assert grade_ticket_category({"category": "unknown"}, GT_BILLING) == 0.0

    def test_empty_category(self):
        assert grade_ticket_category({}, GT_BILLING) == 0.0

    def test_case_insensitive(self):
        assert grade_ticket_category({"category": "BILLING"}, GT_BILLING) == 1.0

    def test_all_valid_categories(self):
        for cat in ("billing", "technical", "account"):
            gt = {"category": cat, "priority": "low", "reply_keywords": ""}
            assert grade_ticket_category({"category": cat}, gt) == 1.0


# ---------------------------------------------------------------------------
# ticket_priority (medium) — grade_ticket_priority
# ---------------------------------------------------------------------------

class TestTicketPriority:
    def test_both_correct(self):
        assert grade_ticket_priority(
            {"category": "billing", "priority": "high"}, GT_BILLING
        ) == 1.0

    def test_only_category_correct(self):
        assert grade_ticket_priority(
            {"category": "billing", "priority": "low"}, GT_BILLING
        ) == 0.5

    def test_only_priority_correct(self):
        assert grade_ticket_priority(
            {"category": "technical", "priority": "high"}, GT_BILLING
        ) == 0.5

    def test_both_wrong(self):
        assert grade_ticket_priority(
            {"category": "account", "priority": "low"}, GT_BILLING
        ) == 0.0

    def test_invalid_priority(self):
        result = grade_ticket_priority(
            {"category": "billing", "priority": "urgent"}, GT_BILLING
        )
        assert result == 0.5  # category correct (0.5) + invalid priority (0.0)

    def test_missing_both(self):
        assert grade_ticket_priority({}, GT_BILLING) == 0.0


# ---------------------------------------------------------------------------
# full_resolution (hard) — grade_full_resolution
# ---------------------------------------------------------------------------

class TestFullResolution:
    def test_perfect_score(self):
        score = grade_full_resolution(
            {"category": "billing", "priority": "high", "reply": "We are sorry; we will refund you."},
            GT_BILLING,
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_all_wrong(self):
        score = grade_full_resolution(
            {"category": "account", "priority": "low", "reply": "nothing relevant"},
            GT_BILLING,
        )
        assert score == pytest.approx(0.0, abs=0.02)

    def test_weights_sum(self):
        # category + priority correct, no reply → 0.35 + 0.35 = 0.70
        score = grade_full_resolution(
            {"category": "billing", "priority": "high", "reply": ""},
            GT_BILLING,
        )
        assert score == pytest.approx(0.70, abs=0.01)

    def test_reply_keyword_partial(self):
        # one of two keywords present
        score = grade_full_resolution(
            {"category": "billing", "priority": "high", "reply": "we will refund you"},
            GT_BILLING,
        )
        # 0.35 + 0.35 + 0.30 * (1/2) = 0.85
        assert score == pytest.approx(0.85, abs=0.01)

    def test_missing_fields(self):
        score = grade_full_resolution({}, GT_BILLING)
        assert score == 0.0


# ---------------------------------------------------------------------------
# grade_partial — dense reward shaping
# ---------------------------------------------------------------------------

class TestGradePartial:
    # --- easy ---
    def test_easy_correct(self):
        s, msg = grade_partial("ticket_category", {"category": "billing"}, GT_BILLING)
        assert s == 1.0

    def test_easy_wrong(self):
        s, msg = grade_partial("ticket_category", {"category": "technical"}, GT_BILLING)
        assert s == 0.0

    def test_easy_invalid(self):
        s, msg = grade_partial("ticket_category", {"category": "nope"}, GT_BILLING)
        assert s == pytest.approx(0.05, abs=0.01)

    def test_easy_no_submission(self):
        s, msg = grade_partial("ticket_category", {}, GT_BILLING)
        assert s == 0.0

    # --- medium ---
    def test_medium_both_correct(self):
        s, msg = grade_partial(
            "ticket_priority",
            {"category": "billing", "priority": "high"},
            GT_BILLING,
        )
        assert s == pytest.approx(1.0, abs=0.01)

    def test_medium_partial_category_only(self):
        s, msg = grade_partial(
            "ticket_priority", {"category": "billing"}, GT_BILLING
        )
        assert s == pytest.approx(0.5, abs=0.01)

    def test_medium_no_submission(self):
        s, msg = grade_partial("ticket_priority", {}, GT_BILLING)
        assert s == 0.0

    # --- hard ---
    def test_hard_perfect(self):
        s, msg = grade_partial(
            "full_resolution",
            {"category": "billing", "priority": "high", "reply": "sorry for the refund issue"},
            GT_BILLING,
        )
        assert s == pytest.approx(1.0, abs=0.01)

    def test_hard_no_submission(self):
        s, msg = grade_partial("full_resolution", {}, GT_BILLING)
        assert s == 0.0

    def test_hard_weights_consistent_with_final(self):
        """Partial grader and final grader must agree when all fields are provided."""
        sub = {"category": "billing", "priority": "high", "reply": "sorry for the refund issue"}
        partial_s, _ = grade_partial("full_resolution", sub, GT_BILLING)
        final_s = final_grader("full_resolution", sub, GT_BILLING)
        # Both should be close (partial may be ≤ final in edge cases)
        assert abs(partial_s - final_s) < 0.05


# ---------------------------------------------------------------------------
# merge_submission helper
# ---------------------------------------------------------------------------

class TestMergeSubmission:
    def test_empty_merge(self):
        assert merge_submission({}, {"category": "billing"}) == {"category": "billing"}

    def test_override(self):
        result = merge_submission({"category": "billing"}, {"category": "technical"})
        assert result["category"] == "technical"

    def test_none_values_ignored(self):
        result = merge_submission({"category": "billing"}, {"category": None})
        # None values must NOT overwrite existing
        assert result["category"] == "billing"

    def test_additive(self):
        result = merge_submission({"category": "billing"}, {"priority": "high"})
        assert result == {"category": "billing", "priority": "high"}


# ---------------------------------------------------------------------------
# extract_json_from_text
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        assert extract_json_from_text('{"category":"billing"}') == {"category": "billing"}

    def test_json_in_prose(self):
        text = 'Here is my answer: {"category": "technical", "priority": "medium"} thanks!'
        result = extract_json_from_text(text)
        assert result["category"] == "technical"

    def test_empty_string(self):
        assert extract_json_from_text("") == {}

    def test_no_json(self):
        assert extract_json_from_text("no json here at all") == {}

    def test_markdown_fenced(self):
        text = '```json\n{"category": "account"}\n```'
        # regex won't strip fences but JSON parse may still work on trimmed
        result = extract_json_from_text(text)
        # should at least not crash
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SupportTriageEnvironment — integration tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def _make_env(self):
        env = SupportTriageEnvironment()
        return env

    def test_reset_returns_observation(self):
        env = self._make_env()
        obs = env.reset(seed=0, task="ticket_category")
        assert obs.task_name == "ticket_category"
        assert obs.step_index == 0
        assert not obs.done
        assert obs.grader_score is None

    def test_reset_selects_ticket_by_seed(self):
        env = self._make_env()
        obs0 = env.reset(seed=0, task="ticket_category")
        obs2 = env.reset(seed=2, task="ticket_category")
        assert obs0.ticket_subject != obs2.ticket_subject

    def test_step_correct_action_easy(self):
        env = self._make_env()
        env.reset(seed=0, task="ticket_category")  # TK-1001, billing
        obs = env.step(SupportTriageAction(category="billing"))
        assert obs.reward > 0

    def test_step_wrong_action_easy(self):
        env = self._make_env()
        env.reset(seed=0, task="ticket_category")
        obs = env.step(SupportTriageAction(category="account"))
        assert obs.reward == 0.0

    def test_empty_action_penalty(self):
        env = self._make_env()
        env.reset(seed=0, task="ticket_category")
        obs = env.step(SupportTriageAction())  # all None
        assert obs.reward < 0

    def test_done_on_perfect_score(self):
        env = self._make_env()
        env.reset(seed=0, task="ticket_category")  # billing
        obs = env.step(SupportTriageAction(category="billing"))
        assert obs.done
        assert obs.grader_score == pytest.approx(1.0, abs=0.01)

    def test_max_steps_terminates_episode(self):
        env = self._make_env()
        env.reset(seed=0, task="ticket_category")
        obs = None
        for _ in range(20):
            obs = env.step(SupportTriageAction(category="account"))  # always wrong
            if obs.done:
                break
        assert obs is not None and obs.done

    def test_state_property(self):
        env = self._make_env()
        env.reset(seed=1, task="ticket_priority")
        state = env.state
        assert state.task_name == "ticket_priority"
        assert state.ticket_id != ""

    def test_full_resolution_hard_task(self):
        """Hard task must produce a grader_score when done."""
        env = self._make_env()
        env.reset(seed=0, task="full_resolution")  # TK-1001, billing/high/"sorry, refund"
        obs = None
        for _ in range(15):
            obs = env.step(SupportTriageAction(
                category="billing",
                priority="high",
                reply="We are sorry; we will process a full refund.",
            ))
            if obs.done:
                break
        assert obs is not None and obs.done
        assert obs.grader_score is not None
        assert obs.grader_score >= 0.7  # well above chance
