"""Evaluation harness.

Runs a pipeline over a set of examples and scores each answer. The harness knows
nothing about retrieval, verification, or LLMs — a pipeline is just a function from an
Example to a Prediction. That is what lets BM25, dense, hybrid, and the full adaptive
pipeline all be measured by the same code.

    def my_pipeline(example: Example) -> Prediction:
        ...

    results = evaluate(my_pipeline, examples)
    print(aggregate(results))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from amrag.metrics import answer_f1, exact_match, supporting_fact_prf
from amrag.schema import Example, SupportingFact

Fact = SupportingFact | tuple[str, int]


@dataclass
class Prediction:
    """What a pipeline returns for one question.

    `llm_calls` is reported by the pipeline rather than measured here: only the
    pipeline knows what counts as a call, and the harness must not learn.
    """

    answer: str
    supporting_facts: list[Fact] = field(default_factory=list)
    llm_calls: int = 0


@dataclass
class QuestionResult:
    qid: str
    predicted_answer: str
    gold_answer: str
    answer_em: float
    answer_f1: float
    sp_precision: float
    sp_recall: float
    sp_f1: float
    latency: float
    llm_calls: int


Pipeline = Callable[[Example], Prediction]


def evaluate(pipeline: Pipeline, examples: Sequence[Example]) -> list[QuestionResult]:
    """Run `pipeline` on every example and score the predictions."""
    results = []
    for example in examples:
        start = time.perf_counter()
        prediction = pipeline(example)
        latency = time.perf_counter() - start

        gold_answer = example.answer or ""
        sp = supporting_fact_prf(prediction.supporting_facts, example.supporting_facts)
        results.append(
            QuestionResult(
                qid=example.qid,
                predicted_answer=prediction.answer,
                gold_answer=gold_answer,
                answer_em=exact_match(prediction.answer, gold_answer),
                answer_f1=answer_f1(prediction.answer, gold_answer),
                sp_precision=sp.precision,
                sp_recall=sp.recall,
                sp_f1=sp.f1,
                latency=latency,
                llm_calls=prediction.llm_calls,
            )
        )
    return results


def aggregate(results: Sequence[QuestionResult]) -> dict[str, float]:
    """Mean scores across questions, plus the totals that show what a run cost."""
    n = len(results)
    if n == 0:
        return {"n_questions": 0}

    def mean(field_name: str) -> float:
        return sum(getattr(r, field_name) for r in results) / n

    return {
        "n_questions": n,
        "answer_em": mean("answer_em"),
        "answer_f1": mean("answer_f1"),
        "sp_precision": mean("sp_precision"),
        "sp_recall": mean("sp_recall"),
        "sp_f1": mean("sp_f1"),
        "latency_mean": mean("latency"),
        "latency_total": sum(r.latency for r in results),
        "llm_calls_mean": mean("llm_calls"),
        "llm_calls_total": sum(r.llm_calls for r in results),
    }
