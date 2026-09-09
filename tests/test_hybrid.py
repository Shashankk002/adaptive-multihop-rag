"""Tests for RRF hybrid retrieval. The real embedding model is never loaded."""

from __future__ import annotations

import pytest

from amrag import dense, hybrid
from amrag.hybrid import hybrid_retrieve, rrf_scores
from amrag.schema import Example, Paragraph

from tests.test_dense import fake_encode


class TestRRFScores:
    def test_hand_calculated_scores(self):
        # "a" is rank 1 then rank 2; "b" is rank 2 then rank 1. With rrf_k = 1:
        #   a = 1/(1+1) + 1/(1+2) = 0.5     + 0.3333 = 0.83333
        #   b = 1/(1+2) + 1/(1+1) = 0.3333  + 0.5    = 0.83333
        scores = rrf_scores([["a", "b"], ["b", "a"]], rrf_k=1)
        assert scores["a"] == pytest.approx(1 / 2 + 1 / 3)
        assert scores["b"] == pytest.approx(1 / 3 + 1 / 2)

    def test_single_ranking_is_monotonically_decreasing(self):
        scores = rrf_scores([["a", "b", "c"]], rrf_k=1)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_agreement_preserves_the_ranking(self):
        scores = rrf_scores([["a", "b", "c"], ["a", "b", "c"]], rrf_k=1)
        assert sorted(scores, key=lambda t: -scores[t]) == ["a", "b", "c"]

    def test_fusion_is_symmetric_when_unweighted(self):
        first = rrf_scores([["a", "b"], ["b", "a"]], rrf_k=5)
        second = rrf_scores([["b", "a"], ["a", "b"]], rrf_k=5)
        assert first == second

    def test_rrf_k_of_zero_is_pure_reciprocal_rank(self):
        scores = rrf_scores([["a", "b"]], rrf_k=0)
        assert scores["a"] == pytest.approx(1.0)
        assert scores["b"] == pytest.approx(0.5)


class TestRRFConstantBehaviour:
    """Why the constant matters: it decides whether being rank 1 in one list beats
    being mid-rank in both. This is the whole reason we did not keep the default 60."""

    # A is ranked (1, 5); B is ranked (2, 3).
    #   rrf_k=1  -> A = 1/2 + 1/6 = 0.6667  vs  B = 1/3 + 1/4 = 0.5833   -> A
    #   rrf_k=60 -> A = 1/61 + 1/65 = 0.03178 vs B = 1/62 + 1/63 = 0.03200 -> B
    RANKINGS = [["A", "B", "C", "D", "E"], ["C", "D", "B", "E", "A"]]

    def test_small_constant_rewards_a_first_place_finish(self):
        scores = rrf_scores(self.RANKINGS, rrf_k=1)
        assert scores["A"] > scores["B"]

    def test_large_constant_rewards_consistent_mid_ranking(self):
        scores = rrf_scores(self.RANKINGS, rrf_k=60)
        assert scores["B"] > scores["A"]


class TestHybridRetrieve:
    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(dense, "encode", fake_encode)
        monkeypatch.setattr(dense, "_cache", None)

    @pytest.fixture
    def example(self):
        return Example(
            qid="q1",
            question="unused",
            paragraphs=(
                Paragraph("Ed Wood (film)", ("Starring Johnny Depp.",)),
                Paragraph("Tim Burton", ("Burton directed it.",)),
                Paragraph("Woodson, Arkansas", ("A place in Arkansas.",)),
            ),
        )

    def test_returns_the_retriever_shape(self, example):
        results = hybrid_retrieve(example, "depp", k=2)
        assert len(results) == 2
        assert all(isinstance(r.paragraph, Paragraph) for r in results)
        assert all(isinstance(r.score, float) for r in results)

    def test_scores_are_non_increasing(self, example):
        scores = [r.score for r in hybrid_retrieve(example, "depp burton", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_every_candidate_is_scored(self, example):
        # Complete rankings are fused, so nothing is dropped before fusion.
        assert len(hybrid_retrieve(example, "depp", k=99)) == 3

    def test_k_limits_results(self, example):
        assert len(hybrid_retrieve(example, "depp", k=1)) == 1

    def test_k_of_zero_returns_nothing(self, example):
        assert hybrid_retrieve(example, "depp", k=0) == []

    def test_agreement_between_retrievers_is_ranked_first(self, example):
        # Both BM25 and the fake encoder favour the Arkansas paragraph for "arkansas".
        assert hybrid_retrieve(example, "arkansas", k=1)[0].paragraph.title == "Woodson, Arkansas"

    def test_is_deterministic(self, example):
        first = [r.paragraph.title for r in hybrid_retrieve(example, "depp", k=3)]
        second = [r.paragraph.title for r in hybrid_retrieve(example, "depp", k=3)]
        assert first == second

    def test_ties_keep_candidate_order(self, example):
        # A query matching nothing leaves both retrievers in candidate order, so the
        # fused ranking must be candidate order too.
        titles = [r.paragraph.title for r in hybrid_retrieve(example, "zzz", k=3)]
        assert titles == ["Ed Wood (film)", "Tim Burton", "Woodson, Arkansas"]

    def test_rrf_k_is_wired_through(self, example, monkeypatch):
        seen = {}

        def spy(rankings, rrf_k):
            seen["rrf_k"] = rrf_k
            return rrf_scores(rankings, rrf_k)

        monkeypatch.setattr(hybrid, "rrf_scores", spy)
        hybrid_retrieve(example, "depp", k=1, rrf_k=17)
        assert seen["rrf_k"] == 17
