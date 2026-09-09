"""End-to-end pipeline: retrieve, verify, expand if needed, answer.

    Example -> routed retrieval -> verifier -> evidence selection -> answer

This module only wires the components together. Every decision stays where it already
lives: the routing threshold in `multihop`, the expansion size in `correct`, prompts
and models in `verify` and `answer`. Nothing here changes their behaviour.

Verdicts and answers are served from the on-disk caches when present, following the
same pattern as `dense.embed`: a cache hit costs no API call, a miss calls the
component and records the result, so a long run resumes where it stopped.
"""

from __future__ import annotations

from amrag import answer as answer_mod
from amrag import correct, verify
from amrag.harness import Prediction
from amrag.multihop import routed_retrieve
from amrag.schema import Example, Paragraph

LLM_CALLS = 2
"""One verification, one generation. Correction adds none. This is what the pipeline
*requires*; a cached run may make no network calls at all and still report 2."""

_verdicts: dict | None = None
_answers: dict | None = None


def load_caches() -> None:
    """Load the verdict and answer caches so repeated runs skip completed work."""
    global _verdicts, _answers
    _verdicts = verify.load_cache()
    _answers = answer_mod.load_cache()


def _verdict(example: Example, evidence: list[Paragraph]) -> verify.Verdict:
    key = verify.cache_key(example.question, evidence)
    if _verdicts is not None and key in _verdicts and _verdicts[key].ok:
        return _verdicts[key]
    verdict = verify.verify(example.question, evidence, qid=example.qid)
    verify.append_cache(verdict)
    if _verdicts is not None:
        _verdicts[key] = verdict
    return verdict


def _answer(example: Example, evidence: list[Paragraph]) -> str:
    key = answer_mod.cache_key(example.question, evidence)
    if _answers is not None and key in _answers and _answers[key].ok:
        return _answers[key].text
    text = answer_mod.answer(example.question, evidence)
    record = answer_mod.Answer(key=key, qid=example.qid, text=text)
    answer_mod.append_cache(record)
    if _answers is not None:
        _answers[key] = record
    return text


def run(example: Example) -> Prediction:
    """Answer `example` from its own candidate paragraphs.

    Raises if verification or generation fails: a failed component is not an answer,
    and coercing one would hide it from the metrics.
    """
    ranked = routed_retrieve(example, example.question, len(example.paragraphs))
    verdict = _verdict(example, [s.paragraph for s in ranked[: correct.BASELINE_K]])
    evidence = [s.paragraph for s in correct.select_evidence(ranked, verdict)]
    text = _answer(example, evidence)

    return Prediction(
        answer=text,
        # Every sentence of the selected evidence. We never built sentence selection,
        # so supporting-fact precision is deliberately low; this reports what the
        # pipeline actually commits to rather than an empty, vacuous set.
        supporting_facts=[
            (paragraph.title, i)
            for paragraph in evidence
            for i in range(len(paragraph.sentences))
        ],
        llm_calls=LLM_CALLS,
    )
