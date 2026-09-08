"""Tests for selective evidence expansion. No retrieval, no model, no API."""

from __future__ import annotations

import pytest

from scrag.correct import BASELINE_K, CORRECTION_K, select_evidence
from scrag.retrieval import Scored
from scrag.schema import Paragraph
from scrag.verify import Verdict

RANKED = [Scored(Paragraph(f"P{i}", (f"body {i}",)), 1.0 - i / 10) for i in range(10)]


def verdict(sufficient: bool) -> Verdict:
    return Verdict(key="k", qid="q", sufficient=sufficient, confidence=0.9, reason="r")


def failed() -> Verdict:
    return Verdict(key="k", qid="q", sufficient=None, confidence=None, reason="",
                   error="429 RESOURCE_EXHAUSTED")


class TestSufficientEvidence:
    def test_keeps_the_top_two(self):
        assert len(select_evidence(RANKED, verdict(True))) == BASELINE_K

    def test_returns_exactly_the_existing_top_two(self):
        titles = [s.paragraph.title for s in select_evidence(RANKED, verdict(True))]
        assert titles == ["P0", "P1"]


class TestInsufficientEvidence:
    def test_expands_to_five(self):
        assert len(select_evidence(RANKED, verdict(False))) == CORRECTION_K

    def test_preserves_ranking_order(self):
        titles = [s.paragraph.title for s in select_evidence(RANKED, verdict(False))]
        assert titles == ["P0", "P1", "P2", "P3", "P4"]

    def test_the_original_top_two_are_never_dropped(self):
        expanded = select_evidence(RANKED, verdict(False))
        assert [s.paragraph.title for s in expanded[:2]] == ["P0", "P1"]

    def test_scores_are_carried_through_unchanged(self):
        assert [s.score for s in select_evidence(RANKED, verdict(False))] == [
            s.score for s in RANKED[:5]
        ]


class TestUndecidedVerdicts:
    """A failed verification is not evidence of insufficiency."""

    def test_failed_verdict_raises(self):
        with pytest.raises(ValueError, match="decided verdict"):
            select_evidence(RANKED, failed())

    def test_missing_verdict_raises(self):
        with pytest.raises(ValueError, match="decided verdict"):
            select_evidence(RANKED, None)

    def test_error_message_names_the_cause(self):
        with pytest.raises(ValueError, match="429"):
            select_evidence(RANKED, failed())


class TestBoundaries:
    def test_fewer_candidates_than_the_expansion_target(self):
        short = RANKED[:3]
        assert len(select_evidence(short, verdict(False))) == 3, "returns what exists"

    def test_fewer_candidates_than_the_baseline(self):
        assert len(select_evidence(RANKED[:1], verdict(True))) == 1

    def test_empty_candidate_list(self):
        assert select_evidence([], verdict(False)) == []

    def test_does_not_mutate_the_input(self):
        before = list(RANKED)
        select_evidence(RANKED, verdict(False))
        assert RANKED == before

    def test_returns_a_new_list(self):
        result = select_evidence(RANKED, verdict(True))
        result.append("x")
        assert len(RANKED) == 10


class TestNoHiddenWork:
    def test_is_a_pure_function_of_ranking_and_verdict(self):
        """Same inputs, same output — no retrieval, scoring, or model call inside."""
        first = select_evidence(RANKED, verdict(False))
        second = select_evidence(RANKED, verdict(False))
        assert [s.paragraph.title for s in first] == [s.paragraph.title for s in second]

    def test_sufficient_and_insufficient_differ_only_in_length(self):
        kept = select_evidence(RANKED, verdict(True))
        expanded = select_evidence(RANKED, verdict(False))
        assert [s.paragraph.title for s in expanded[: len(kept)]] == [
            s.paragraph.title for s in kept
        ], "expansion is a superset; it never reorders"
