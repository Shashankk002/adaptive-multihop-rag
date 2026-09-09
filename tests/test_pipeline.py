"""Tests for the end-to-end pipeline. No API calls: the caches are pre-populated."""

from __future__ import annotations

import pytest

from amrag import answer as answer_mod
from amrag import correct, dense, pipeline, verify
from amrag.data_loader import parse_hotpotqa
from amrag.harness import Prediction, aggregate, evaluate

from tests.test_dense import fake_encode


@pytest.fixture
def example(raw_dataset):
    parsed, _ = parse_hotpotqa(raw_dataset)
    return parsed[0]          # bridge-0001, 3 candidate paragraphs


@pytest.fixture
def wired(monkeypatch, example):
    """Pre-populate both caches so `run` needs no provider at all.

    Returns a helper that seeds a verdict and an answer for the evidence the
    pipeline will actually select.
    """
    monkeypatch.setattr(dense, "encode", fake_encode)
    monkeypatch.setattr(dense, "_cache", None)
    monkeypatch.setattr(verify, "append_cache", lambda *a, **k: None)
    monkeypatch.setattr(answer_mod, "append_cache", lambda *a, **k: None)

    def seed(sufficient: bool, text: str = "Tim Burton"):
        from amrag.multihop import routed_retrieve

        ranked = routed_retrieve(example, example.question, len(example.paragraphs))
        top2 = [s.paragraph for s in ranked[: correct.BASELINE_K]]
        verdict = verify.Verdict(
            key=verify.cache_key(example.question, top2), qid=example.qid,
            sufficient=sufficient, confidence=0.9, reason="r",
        )
        evidence = [s.paragraph for s in correct.select_evidence(ranked, verdict)]
        record = answer_mod.Answer(
            key=answer_mod.cache_key(example.question, evidence),
            qid=example.qid, text=text,
        )
        monkeypatch.setattr(pipeline, "_verdicts", {verdict.key: verdict})
        monkeypatch.setattr(pipeline, "_answers", {record.key: record})
        return evidence

    return seed


class TestComposition:
    def test_returns_a_prediction(self, wired, example):
        wired(sufficient=True)
        result = pipeline.run(example)

        assert isinstance(result, Prediction)
        assert result.answer == "Tim Burton"
        assert result.llm_calls == 2

    def test_llm_calls_is_the_logical_requirement(self, wired, example):
        """Both components were cache-served, yet the pipeline still reports 2."""
        wired(sufficient=True)
        assert pipeline.run(example).llm_calls == pipeline.LLM_CALLS == 2

    def test_answer_text_comes_from_the_generator(self, wired, example):
        wired(sufficient=True, text="Peter Chelsom")
        assert pipeline.run(example).answer == "Peter Chelsom"


class TestEvidenceSelection:
    def test_sufficient_keeps_the_top_two(self, wired, example):
        evidence = wired(sufficient=True)
        assert len(evidence) == correct.BASELINE_K == 2

        titles = {t for t, _ in pipeline.run(example).supporting_facts}
        assert titles == {p.title for p in evidence}

    def test_insufficient_expands_to_correction_k(self, wired, example):
        evidence = wired(sufficient=False)
        # The fixture has only 3 candidates, so expansion is capped by what exists.
        assert len(evidence) == min(correct.CORRECTION_K, len(example.paragraphs))
        assert len(evidence) > correct.BASELINE_K

        titles = {t for t, _ in pipeline.run(example).supporting_facts}
        assert titles == {p.title for p in evidence}

    def test_correction_k_is_still_five(self):
        assert correct.CORRECTION_K == 5, "frozen; changing it invalidates Phase 9"


class TestSupportingFacts:
    def test_covers_every_sentence_of_the_selected_evidence(self, wired, example):
        evidence = wired(sufficient=False)
        expected = {
            (p.title, i) for p in evidence for i in range(len(p.sentences))
        }
        assert set(pipeline.run(example).supporting_facts) == expected

    def test_contains_no_unselected_paragraph(self, wired, example):
        evidence = wired(sufficient=True)
        selected = {p.title for p in evidence}
        unselected = {p.title for p in example.paragraphs} - selected

        titles = {t for t, _ in pipeline.run(example).supporting_facts}
        assert not (titles & unselected)


class TestFailurePropagation:
    def test_a_failed_verdict_propagates(self, monkeypatch, example):
        monkeypatch.setattr(dense, "encode", fake_encode)
        monkeypatch.setattr(dense, "_cache", None)
        monkeypatch.setattr(pipeline, "_verdicts", None)
        monkeypatch.setattr(pipeline, "_answers", None)
        monkeypatch.setattr(verify, "append_cache", lambda *a, **k: None)
        monkeypatch.setattr(
            verify, "verify",
            lambda q, p, qid="": verify.Verdict(key="k", qid=qid, sufficient=None,
                                                confidence=None, reason="", error="429"),
        )
        with pytest.raises(ValueError, match="decided verdict"):
            pipeline.run(example)

    def test_a_failed_generation_propagates(self, wired, example, monkeypatch):
        wired(sufficient=True)
        monkeypatch.setattr(pipeline, "_answers", {})      # force a cache miss
        monkeypatch.setattr(answer_mod, "answer",
                            lambda q, p: (_ for _ in ()).throw(answer_mod.AnswerError("boom")))
        with pytest.raises(answer_mod.AnswerError, match="boom"):
            pipeline.run(example)

    def test_failures_never_become_an_answer(self, wired, example, monkeypatch):
        wired(sufficient=True)
        monkeypatch.setattr(pipeline, "_answers", {})
        monkeypatch.setattr(answer_mod, "answer",
                            lambda q, p: (_ for _ in ()).throw(answer_mod.AnswerError("boom")))
        with pytest.raises(answer_mod.AnswerError):
            pipeline.run(example)   # no Prediction is produced at all


class TestHarnessCompatibility:
    def test_runs_through_the_evaluation_harness(self, wired, example):
        wired(sufficient=True)
        results = evaluate(pipeline.run, [example])

        assert len(results) == 1
        assert results[0].qid == example.qid
        assert results[0].predicted_answer == "Tim Burton"
        assert results[0].gold_answer == example.answer
        assert results[0].llm_calls == 2

    def test_aggregates_without_special_casing(self, wired, example):
        wired(sufficient=True)
        summary = aggregate(evaluate(pipeline.run, [example]))

        assert summary["n_questions"] == 1
        assert summary["llm_calls_total"] == 2
        assert "answer_em" in summary and "sp_f1" in summary

    def test_a_correct_answer_scores_one(self, wired, example):
        wired(sufficient=True, text=example.answer)
        assert evaluate(pipeline.run, [example])[0].answer_em == 1.0


class TestNoDuplicateWork:
    def test_retrieval_runs_exactly_once(self, wired, example, monkeypatch):
        wired(sufficient=False)
        calls = []
        original = pipeline.routed_retrieve

        def counting(ex, query, k):
            calls.append(k)
            return original(ex, query, k)

        monkeypatch.setattr(pipeline, "routed_retrieve", counting)
        pipeline.run(example)

        assert len(calls) == 1, "retrieval must not be repeated for the expansion"

    def test_cache_hits_make_no_component_call(self, wired, example, monkeypatch):
        wired(sufficient=True)
        monkeypatch.setattr(verify, "verify",
                            lambda *a, **k: pytest.fail("verifier called despite a cache hit"))
        monkeypatch.setattr(answer_mod, "answer",
                            lambda *a, **k: pytest.fail("generator called despite a cache hit"))
        assert pipeline.run(example).answer == "Tim Burton"
