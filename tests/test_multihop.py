"""Tests for two-hop retrieval. The embedding model is never loaded."""

from __future__ import annotations

import pytest

from amrag import dense
from amrag.multihop import expansion_text, multihop_retrieve
from amrag.schema import Example, Paragraph

from tests.test_dense import fake_encode


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(dense, "encode", fake_encode)
    monkeypatch.setattr(dense, "_cache", None)


@pytest.fixture
def example():
    # The fake encoder scores on the words depp / burton / arkansas. The question
    # mentions only "depp", so hop 1 finds the Ed Wood paragraph; that paragraph's
    # text mentions "Burton", which is what lets hop 2 reach the Tim Burton paragraph.
    return Example(
        qid="q1",
        question="unused",
        paragraphs=(
            Paragraph("Ed Wood (film)", ("Starring Johnny Depp, directed by Burton.",)),
            Paragraph("Woodson, Arkansas", ("A place in Arkansas.",)),
            Paragraph("Tim Burton", ("Burton is a filmmaker.",)),
        ),
    )


class TestExpansionText:
    @pytest.fixture
    def paragraph(self):
        return Paragraph("Tim Burton", ("He is a filmmaker.", " He directed Ed Wood."))

    def test_title_mode(self, paragraph):
        assert expansion_text(paragraph, "who directed", "title") == "Tim Burton"

    def test_paragraph_mode_is_title_plus_body(self, paragraph):
        text = expansion_text(paragraph, "who directed", "paragraph")
        assert text.startswith("Tim Burton")
        assert "He directed Ed Wood." in text

    def test_sentence_mode_picks_the_question_relevant_sentence(self, paragraph):
        text = expansion_text(paragraph, "who directed Ed Wood", "sentence")
        assert "He directed Ed Wood." in text
        assert "He is a filmmaker." not in text

    def test_sentence_mode_keeps_the_title(self, paragraph):
        assert expansion_text(paragraph, "directed", "sentence").startswith("Tim Burton")

    def test_unknown_mode_is_rejected(self, paragraph):
        with pytest.raises(ValueError, match="unknown expansion mode"):
            expansion_text(paragraph, "q", "nonsense")


class TestTwoHopRetrieval:
    def test_hop2_reaches_a_paragraph_the_question_alone_does_not(self, example):
        # Dense on the bare question ranks Tim Burton below Arkansas.
        single = [s.paragraph.title for s in dense.dense_retrieve(example, "depp", 2)]
        assert single[1] != "Tim Burton"

        two_hop = [s.paragraph.title for s in multihop_retrieve(example, "depp", 2)]
        assert two_hop == ["Ed Wood (film)", "Tim Burton"]

    def test_hop1_result_is_always_first(self, example):
        first = dense.dense_retrieve(example, "depp", 1)[0].paragraph.title
        assert multihop_retrieve(example, "depp", 3)[0].paragraph.title == first

    def test_hop1_paragraph_is_not_repeated(self, example):
        titles = [s.paragraph.title for s in multihop_retrieve(example, "depp", 3)]
        assert len(titles) == len(set(titles))

    def test_returns_every_candidate_when_k_is_large(self, example):
        assert len(multihop_retrieve(example, "depp", 99)) == 3

    def test_k_limits_results(self, example):
        assert len(multihop_retrieve(example, "depp", 1)) == 1

    def test_k_of_zero_returns_nothing(self, example):
        assert multihop_retrieve(example, "depp", 0) == []

    def test_is_deterministic(self, example):
        first = [s.paragraph.title for s in multihop_retrieve(example, "depp", 3)]
        second = [s.paragraph.title for s in multihop_retrieve(example, "depp", 3)]
        assert first == second

    def test_expansion_mode_changes_the_hop2_query(self, example, monkeypatch):
        seen = []
        original = dense.dense_retrieve

        def spy(ex, query, k):
            seen.append(query)
            return original(ex, query, k)

        monkeypatch.setattr("amrag.multihop.dense_retrieve", spy)
        multihop_retrieve(example, "depp", 2, expansion="title")

        assert seen[0] == "depp", "hop 1 uses the bare question"
        assert seen[1] == "depp Ed Wood (film)", "hop 2 appends the hop-1 title"

    def test_matches_the_retriever_shape(self, example):
        results = multihop_retrieve(example, "depp", 2)
        assert all(isinstance(r.paragraph, Paragraph) for r in results)
        assert all(isinstance(r.score, float) for r in results)


class TestRouting:
    """Hop 2 runs only when dense's rank-2/rank-3 margin is small."""

    def _example_with_scores(self, scores):
        """An example whose dense scores are exactly `scores`, via a stub retriever."""
        from amrag.retrieval import Scored

        paragraphs = tuple(
            Paragraph(f"P{i}", (f"body {i}",)) for i in range(len(scores))
        )
        example = Example(qid="q1", question="unused", paragraphs=paragraphs)
        ranked = [Scored(p, s) for p, s in zip(paragraphs, scores)]
        return example, ranked

    def test_wide_margin_skips_hop_two(self, monkeypatch):
        from amrag import multihop

        example, ranked = self._example_with_scores([0.9, 0.8, 0.5])  # m23 = 0.30
        monkeypatch.setattr(multihop, "dense_retrieve", lambda e, q, k: ranked[:k])
        monkeypatch.setattr(
            multihop, "multihop_retrieve",
            lambda *a, **kw: pytest.fail("hop 2 must not run when dense is confident"),
        )

        result = multihop.routed_retrieve(example, "q", k=2)
        assert [r.paragraph.title for r in result] == ["P0", "P1"]

    def test_narrow_margin_runs_hop_two(self, monkeypatch):
        from amrag import multihop

        example, ranked = self._example_with_scores([0.9, 0.8, 0.79])  # m23 = 0.01
        called = []
        monkeypatch.setattr(multihop, "dense_retrieve", lambda e, q, k: ranked[:k])
        monkeypatch.setattr(
            multihop, "multihop_retrieve",
            lambda *a, **kw: called.append(True) or ranked[:2],
        )

        multihop.routed_retrieve(example, "q", k=2)
        assert called == [True]

    def test_threshold_boundary_is_inclusive_of_dense(self, monkeypatch):
        from amrag import multihop

        # m23 exactly at the threshold -> keep dense.
        example, ranked = self._example_with_scores([0.9, 0.8, 0.78])
        assert multihop.score_margin(ranked) == pytest.approx(multihop.ROUTING_THRESHOLD)
        monkeypatch.setattr(multihop, "dense_retrieve", lambda e, q, k: ranked[:k])
        monkeypatch.setattr(
            multihop, "multihop_retrieve",
            lambda *a, **kw: pytest.fail("at the threshold the dense result is kept"),
        )
        assert len(multihop.routed_retrieve(example, "q", k=2)) == 2

    def test_score_margin_uses_ranks_two_and_three(self):
        from amrag import multihop

        _, ranked = self._example_with_scores([0.9, 0.5, 0.4])
        # Not 0.9 - 0.5; the rank-1/rank-2 gap is deliberately not the signal.
        assert multihop.score_margin(ranked) == pytest.approx(0.1)

    def test_fewer_than_three_candidates_skips_hop_two(self):
        from amrag import multihop

        _, ranked = self._example_with_scores([0.9, 0.8])
        assert multihop.score_margin(ranked) == float("inf")

    def test_results_are_correctly_formed(self, example):
        results = __import__("amrag.multihop", fromlist=["x"]).routed_retrieve(example, "depp", k=2)
        assert len(results) == 2
        assert all(isinstance(r.paragraph, Paragraph) for r in results)
        assert all(isinstance(r.score, float) for r in results)
        assert len({r.paragraph.title for r in results}) == 2

    def test_no_llm_calls(self, example):
        """Routing uses only retrieval scores, so a pipeline built on it reports zero."""
        from amrag.harness import Prediction, aggregate, evaluate
        from amrag.multihop import routed_retrieve

        def pipeline(ex):
            hits = routed_retrieve(ex, ex.question, k=2)
            return Prediction(answer="", supporting_facts=[(h.paragraph.title, 0) for h in hits])

        summary = aggregate(evaluate(pipeline, [example]))
        assert summary["llm_calls_total"] == 0
