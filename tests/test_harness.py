"""Tests for the evaluation harness. In-memory examples only."""

from __future__ import annotations

import time

import pytest

from scrag.data_loader import parse_hotpotqa
from scrag.harness import Prediction, aggregate, evaluate


@pytest.fixture
def examples(raw_dataset):
    parsed, _ = parse_hotpotqa(raw_dataset)
    return parsed


def perfect_pipeline(example):
    """Returns the gold answer and gold supporting facts."""
    return Prediction(
        answer=example.answer,
        supporting_facts=list(example.supporting_facts),
        llm_calls=2,
    )


def wrong_pipeline(example):
    return Prediction(answer="completely wrong", supporting_facts=[("Nowhere", 0)])


class TestPerfectPipeline:
    def test_scores_are_all_one(self, examples):
        results = evaluate(perfect_pipeline, examples)

        assert len(results) == 2
        for result in results:
            assert result.answer_em == 1.0
            assert result.answer_f1 == 1.0
            assert result.sp_precision == 1.0
            assert result.sp_recall == 1.0
            assert result.sp_f1 == 1.0

    def test_records_ids_and_answers(self, examples):
        results = evaluate(perfect_pipeline, examples)

        assert [r.qid for r in results] == ["bridge-0001", "comparison-0001"]
        assert results[0].predicted_answer == "Tim Burton"
        assert results[0].gold_answer == "Tim Burton"


class TestWrongPipeline:
    def test_scores_are_all_zero(self, examples):
        results = evaluate(wrong_pipeline, examples)

        for result in results:
            assert result.answer_em == 0.0
            assert result.answer_f1 == 0.0
            assert result.sp_f1 == 0.0

    def test_partially_correct_prediction(self, examples):
        def partial(example):
            # Right answer, but only one of the two gold supporting facts.
            return Prediction(
                answer=example.answer,
                supporting_facts=[example.supporting_facts[0]],
            )

        result = evaluate(partial, examples)[0]

        assert result.answer_em == 1.0
        assert result.sp_precision == 1.0
        assert result.sp_recall == 0.5
        assert result.sp_f1 == pytest.approx(2 / 3)


class TestRecordedFields:
    def test_latency_is_recorded(self, examples):
        def slow(example):
            time.sleep(0.01)
            return Prediction(answer=example.answer)

        results = evaluate(slow, examples)

        for result in results:
            assert result.latency >= 0.01

    def test_llm_calls_are_recorded(self, examples):
        results = evaluate(perfect_pipeline, examples)
        assert [r.llm_calls for r in results] == [2, 2]

    def test_llm_calls_default_to_zero(self, examples):
        results = evaluate(wrong_pipeline, examples)
        assert [r.llm_calls for r in results] == [0, 0]


class TestAggregation:
    def test_perfect_run(self, examples):
        summary = aggregate(evaluate(perfect_pipeline, examples))

        assert summary["n_questions"] == 2
        assert summary["answer_em"] == 1.0
        assert summary["answer_f1"] == 1.0
        assert summary["sp_f1"] == 1.0
        assert summary["llm_calls_total"] == 4
        assert summary["llm_calls_mean"] == 2.0
        assert summary["latency_total"] > 0

    def test_means_are_averaged_over_questions(self, examples):
        def half_right(example):
            # Correct on the bridge question only.
            if example.qid == "bridge-0001":
                return Prediction(example.answer, list(example.supporting_facts))
            return Prediction("wrong")

        summary = aggregate(evaluate(half_right, examples))

        assert summary["answer_em"] == 0.5
        assert summary["sp_f1"] == 0.5

    def test_empty_results(self):
        assert aggregate([]) == {"n_questions": 0}


class TestPipelineIndependence:
    """The harness must work with any callable, with no knowledge of what is inside."""

    def test_accepts_a_plain_function_returning_a_constant(self, examples):
        results = evaluate(lambda _: Prediction("yes"), examples)

        # "yes" is the gold answer for the comparison question only.
        assert [r.answer_em for r in results] == [0.0, 1.0]

    def test_supporting_facts_may_be_plain_tuples(self, examples):
        def tuples(example):
            return Prediction(example.answer, [("Ed Wood (film)", 0), ("Tim Burton", 1)])

        assert evaluate(tuples, examples)[0].sp_f1 == 1.0
