"""Tests for dense retrieval.

The real model is never loaded and nothing is downloaded: `encode` is replaced with a
tiny deterministic fake, so the suite stays offline and fast.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from amrag import dense
from amrag.dense import CacheMismatch, dense_retrieve
from amrag.schema import Example, Paragraph

# A three-word vocabulary. Each text becomes its normalised word-count vector, so
# similarities are predictable enough to assert exact rankings.
VOCAB = ["depp", "burton", "arkansas"]


def fake_encode(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        lowered = text.lower()
        v = np.array([float(lowered.count(w)) for w in VOCAB], dtype=np.float32)
        norm = np.linalg.norm(v)
        vectors.append(v / norm if norm else v)
    return np.stack(vectors)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Every test in this module uses the fake encoder and an empty cache."""
    monkeypatch.setattr(dense, "encode", fake_encode)
    monkeypatch.setattr(dense, "_cache", None)


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


class TestRetrieval:
    def test_ranks_the_semantically_closest_paragraph_first(self, example):
        assert dense_retrieve(example, "depp", k=1)[0].paragraph.title == "Ed Wood (film)"

    def test_a_different_query_reorders_the_results(self, example):
        assert dense_retrieve(example, "arkansas", k=1)[0].paragraph.title == "Woodson, Arkansas"

    def test_scores_are_non_increasing(self, example):
        scores = [r.score for r in dense_retrieve(example, "depp burton", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_identical_text_scores_one(self, example):
        # "burton" appears in both title and body of paragraph 2, so its vector is
        # parallel to the query vector and cosine similarity is exactly 1.
        assert dense_retrieve(example, "burton", k=1)[0].score == pytest.approx(1.0)

    def test_k_limits_results(self, example):
        assert len(dense_retrieve(example, "depp", k=2)) == 2

    def test_k_larger_than_candidate_set_returns_everything(self, example):
        assert len(dense_retrieve(example, "depp", k=99)) == 3

    def test_k_of_zero_returns_nothing(self, example):
        assert dense_retrieve(example, "depp", k=0) == []

    def test_ties_keep_candidate_order(self, example):
        # No vocabulary overlap, so every score is 0 and original order must survive.
        titles = [r.paragraph.title for r in dense_retrieve(example, "zzz", k=3)]
        assert titles == ["Ed Wood (film)", "Tim Burton", "Woodson, Arkansas"]

    def test_is_deterministic(self, example):
        first = [r.paragraph.title for r in dense_retrieve(example, "depp burton", k=3)]
        second = [r.paragraph.title for r in dense_retrieve(example, "depp burton", k=3)]
        assert first == second

    def test_returns_the_same_shape_as_bm25(self, example):
        from amrag.retrieval import bm25_retrieve

        dense_result = dense_retrieve(example, "depp", k=2)
        bm25_result = bm25_retrieve(example, "depp", k=2)

        assert len(dense_result) == len(bm25_result)
        for a, b in zip(dense_result, bm25_result):
            assert isinstance(a.paragraph, Paragraph) and isinstance(b.paragraph, Paragraph)
            assert isinstance(a.score, float) and isinstance(b.score, float)


class TestQueryPrefix:
    def test_prefix_is_applied_to_the_query_only(self, monkeypatch, example):
        seen = []

        def recording_encode(texts):
            seen.extend(texts)
            return fake_encode(texts)

        monkeypatch.setattr(dense, "encode", recording_encode)
        dense_retrieve(example, "who directed it", k=1)

        queries = [t for t in seen if t.startswith(dense.QUERY_PREFIX)]
        assert queries == [dense.QUERY_PREFIX + "who directed it"]
        # No passage text may carry the prefix.
        passages = [t for t in seen if "Starring Johnny Depp" in t]
        assert passages and all(not p.startswith(dense.QUERY_PREFIX) for p in passages)


class TestCache:
    def test_build_then_load_round_trip(self, tmp_path, monkeypatch):
        path = tmp_path / "emb.npz"
        count = dense.build_cache(["alpha", "beta", "alpha"], path=path)

        assert count == 2, "duplicates are stored once"
        meta = dense.load_cache(path)
        assert meta["model"] == dense.MODEL_NAME
        assert meta["revision"] == dense.MODEL_REVISION
        assert meta["dimension"] == dense.DIMENSION
        assert meta["count"] == 2

    def test_cached_vectors_are_served_without_encoding(self, tmp_path, monkeypatch):
        path = tmp_path / "emb.npz"
        dense.build_cache(["alpha depp"], path=path)
        dense.load_cache(path)

        def explode(texts):
            raise AssertionError(f"should not encode cached text: {texts}")

        monkeypatch.setattr(dense, "encode", explode)
        assert dense.embed(["alpha depp"]).shape == (1, 3)

    def test_missing_text_is_encoded_and_added(self, tmp_path, monkeypatch):
        path = tmp_path / "emb.npz"
        dense.build_cache(["alpha"], path=path)
        dense.load_cache(path)

        result = dense.embed(["burton"])

        assert result.shape == (1, 3)
        assert dense._key("burton") in dense._cache

    def test_refuses_a_cache_from_a_different_revision(self, tmp_path):
        path = tmp_path / "emb.npz"
        np.savez(
            path,
            keys=np.array(["abc"]),
            vectors=np.zeros((1, 3), dtype=np.float32),
            meta=np.array(json.dumps({"model": dense.MODEL_NAME, "revision": "deadbeef"})),
        )
        with pytest.raises(CacheMismatch, match="deadbeef"):
            dense.load_cache(path)

    def test_refuses_a_cache_from_a_different_model(self, tmp_path):
        path = tmp_path / "emb.npz"
        np.savez(
            path,
            keys=np.array(["abc"]),
            vectors=np.zeros((1, 3), dtype=np.float32),
            meta=np.array(json.dumps({"model": "other/model", "revision": dense.MODEL_REVISION})),
        )
        with pytest.raises(CacheMismatch, match="other/model"):
            dense.load_cache(path)

    def test_key_is_the_exact_text(self):
        assert dense._key("alpha") == dense._key("alpha")
        assert dense._key("alpha") != dense._key("alpha ")
        assert dense._key("Title body") != dense._key("body")
