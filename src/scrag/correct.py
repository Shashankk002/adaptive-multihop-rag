"""Self-correction: selective evidence expansion.

HotpotQA distractor fixes each question's candidate set at ~10 paragraphs, and every
gold paragraph is always among them. So correction cannot fetch new documents — when
the verifier reports the evidence is insufficient, the only useful move is to look
further down the ranking we already have.

This module orchestrates only: it does not retrieve, score, or call a model. It takes
a ranked candidate list and a verdict, and returns the evidence to use.
"""

from __future__ import annotations

from typing import Sequence

from scrag.retrieval import Scored
from scrag.verify import Verdict

BASELINE_K = 2
CORRECTION_K = 5
"""Frozen from the TUNE sweep. One round only: with a fixed candidate set there is
nothing a second round could add that a larger k does not already reach."""


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
