"""Selective evidence expansion.

HotpotQA distractor fixes each question's candidate set at ~10 paragraphs, and every
gold paragraph is always among them. So correction cannot fetch new documents — when
the verifier reports the evidence is insufficient, the only useful move is to look
further down the ranking we already have.

This module orchestrates only: it does not retrieve, score, or call a model. It takes
a ranked candidate list and a verdict, and returns the evidence to use.
"""

from __future__ import annotations

from typing import Sequence

from amrag.retrieval import Scored
from amrag.verify import Verdict

BASELINE_K = 2
CORRECTION_K = 5
"""Frozen from the TUNE gold-pair sweep, then confirmed against Phase 9 answer quality:
on the 200-question answer-dev set k=5 beat k=3 by +0.061 F1 (95% CI [+0.021, +0.103]),
and holding retrieval fixed it read slightly *better* rather than worse, so the extra
context carries no measured distraction penalty. EM alone could not separate them
(+0.030, p=0.13), partly because EM scores "Carbon County" for "Carbon" as a miss.

One round only: with a fixed candidate set there is nothing a second round could add
that a larger k does not already reach."""


def select_evidence(ranked: Sequence[Scored], verdict: Verdict) -> list[Scored]:
    """Top-2 normally; top-5 when the verifier says the evidence is insufficient.

    Raises on an undecided verdict rather than guessing. A failed verification is not
    evidence of insufficiency, and quietly expanding on it would inflate the measured
    correction rate with cases the verifier never actually judged.
    """
    if verdict is None or not verdict.ok:
        raise ValueError(
            "select_evidence needs a decided verdict; got "
            f"{'None' if verdict is None else 'a failed verdict: ' + (verdict.error or 'unknown')}"
        )
    k = BASELINE_K if verdict.sufficient else CORRECTION_K
    return list(ranked[:k])
