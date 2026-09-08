"""Tests for cross-encoder reranking. Neither model is ever loaded."""

from __future__ import annotations

import pytest

from scrag import dense, rerank
from scrag.rerank import rerank_retrieve
from scrag.schema import Example, Paragraph

from tests.test_dense import fake_encode

# Hand-assigned relevance, keyed by a word in the passage. Dense (via fake_encode)
# ranks "depp" first for the query below; the fake reranker disagrees, so any test
# that sees Tim Burton on top proves the reranker's ordering actually won.
FAKE_RELEVANCE = {"Burton directed": 9.0, "Starring Johnny Depp": 5.0, "place in Arkansas": 1.0}


def fake_score_pairs(query: str, texts: list[str]) -> list[float]:
    scores = []
    for text in texts:
        scores.append(next((v for k, v in FAKE_RELEVANCE.items() if k in text), 0.0))
    return scores


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(dense, "encode", fake_encode)
    monkeypatch.setattr(dense, "_cache", None)
    monkeypatch.setattr(rerank, "score_pairs", fake_score_pairs)


@pytest.fixture
def example():
    return Example(
        qid="q1",
        question="unused",
        paragraphs=(
            Paragraph("Ed Wood (film)", ("Starring Johnny Depp.",)),
            Paragraph("Tim Burton", ("Burton directed it.",)),
            Paragraph("Woodson, Arkansas", ("A place in Arkansas.",)),
        ),
    )


class TestReranking:
    def test_reranker_overrides_the_dense_ordering(self, example):
        # Dense puts "Ed Wood (film)" first for "depp"; the reranker prefers Tim Burton.
        assert dense.dense_retrieve(example, "depp", 3)[0].paragraph.title == "Ed Wood (film)"
        assert rerank_retrieve(example, "depp", k=1, depth=3)[0].paragraph.title == "Tim Burton"

    def test_full_reranked_order(self, example):
        titles = [r.paragraph.title for r in rerank_retrieve(example, "depp", k=3, depth=3)]
        assert titles == ["Tim Burton", "Ed Wood (film)", "Woodson, Arkansas"]

    def test_scores_are_the_reranker_scores(self, example):
        assert [r.score for r in rerank_retrieve(example, "depp", k=3, depth=3)] == [9.0, 5.0, 1.0]

    def test_scores_are_non_increasing(self, example):
        scores = [r.score for r in rerank_retrieve(example, "depp", k=3, depth=3)]
        assert scores == sorted(scores, reverse=True)


class TestDepth:
    def test_depth_limits_what_the_reranker_can_see(self, example):
        # Dense ranks Arkansas last for "depp", so depth=2 hides it from the reranker
        # and it can never be returned — the Case A ceiling, in miniature.
        titles = [r.paragraph.title for r in rerank_retrieve(example, "depp", k=3, depth=2)]
        assert "Woodson, Arkansas" not in titles
        assert len(titles) == 2

    def test_depth_larger_than_the_candidate_set(self, example):
        assert len(rerank_retrieve(example, "depp", k=99, depth=99)) == 3

    def test_k_limits_the_returned_results(self, example):
        assert len(rerank_retrieve(example, "depp", k=2, depth=3)) == 2

    def test_k_of_zero_returns_nothing(self, example):
        assert rerank_retrieve(example, "depp", k=0, depth=3) == []


class TestPairFormatting:
    """The highest-value test here: a formatting slip is silent and costs accuracy."""

    def test_pairs_are_raw_question_and_title_plus_body(self, example, monkeypatch):
        seen = {}

        def spy(query, texts):
            seen["query"] = query
            seen["texts"] = texts
            return fake_score_pairs(query, texts)

        monkeypatch.setattr(rerank, "score_pairs", spy)
        rerank_retrieve(example, "who directed it", k=1, depth=3)

        assert seen["query"] == "who directed it", "the query must be passed raw"
        assert not seen["query"].startswith(dense.QUERY_PREFIX), "no bge prefix"
        assert "Tim Burton Burton directed it." in seen["texts"], "title + body, as indexed"


class TestDeterminism:
    def test_repeated_calls_agree(self, example):
        first = [r.paragraph.title for r in rerank_retrieve(example, "depp", k=3, depth=3)]
        second = [r.paragraph.title for r in rerank_retrieve(example, "depp", k=3, depth=3)]
        assert first == second

    def test_ties_keep_the_dense_shortlist_order(self, example, monkeypatch):
        monkeypatch.setattr(rerank, "score_pairs", lambda q, texts: [0.0] * len(texts))

        dense_order = [s.paragraph.title for s in dense.dense_retrieve(example, "depp", 3)]
        reranked = [r.paragraph.title for r in rerank_retrieve(example, "depp", k=3, depth=3)]
        assert reranked == dense_order


class TestInterfaceContract:
    def test_matches_the_retriever_shape(self, example):
        results = rerank_retrieve(example, "depp", k=2)
        assert len(results) == 2
        assert all(isinstance(r.paragraph, Paragraph) for r in results)
        assert all(isinstance(r.score, float) for r in results)
