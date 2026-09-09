"""HotpotQA evaluation metrics.

These follow the official `hotpot_evaluate_v1.py` conventions exactly. Reproducing the
official numbers matters more than improving on them: a metric of our own invention
would make our results incomparable to every published HotpotQA baseline.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, NamedTuple, Sequence

from amrag.schema import SupportingFact

# HotpotQA scores these answers all-or-nothing. Without this, predicting "yes" against
# the gold answer "yes he did" would earn partial F1 credit for a wrong yes/no call.
_ALL_OR_NOTHING = {"yes", "no", "noanswer"}

_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    The step order is the official one. Punctuation is removed before articles so that
    "the-film" becomes "thefilm" rather than losing "the" as a standalone word.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between the normalized prediction and gold answer."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if pred_norm in _ALL_OR_NOTHING or gold_norm in _ALL_OR_NOTHING:
        return float(pred_norm == gold_norm)

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    # Counter intersection matches repeated tokens by count, so predicting "new new
    # york" against "new york" is penalised rather than treated as a perfect match.
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class PRF(NamedTuple):
    precision: float
    recall: float
    f1: float


Fact = SupportingFact | tuple[str, int]


def supporting_fact_prf(
    predicted: Iterable[Fact],
    gold: Iterable[Fact],
) -> PRF:
    """Precision, recall and F1 over (title, sentence index) pairs.

    Compared as sets, per the official script: predicting the same sentence twice is
    neither rewarded nor punished.
    """
    pred_set = {_as_pair(f) for f in predicted}
    gold_set = {_as_pair(f) for f in gold}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return PRF(precision, recall, f1)


def _as_pair(fact: Fact) -> tuple[str, int]:
    if isinstance(fact, SupportingFact):
        return (fact.title, fact.sent_id)
    return tuple(fact)


# --- Retrieval metrics -----------------------------------------------------------
# Paragraph-level, and deliberately separate from the answer and supporting-fact
# metrics above: those score sentences, these score which paragraphs were retrieved.


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of gold paragraph titles found in the top k."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(gold_set & set(retrieved[:k])) / len(gold_set)


def all_gold_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """1.0 if every gold paragraph is in the top k, else 0.0.

    At k=2 this is the headline Phase 2 number: a multi-hop question cannot be
    answered from one of its two gold paragraphs.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return float(gold_set <= set(retrieved[:k]))


def first_missed_gold_rank(
    retrieved: Sequence[str], gold: Iterable[str], k: int
) -> int | None:
    """1-based rank of the best-ranked gold paragraph that fell outside the top k,
    or None if none were missed. Says how near a miss was."""
    gold_set = set(gold)
    for rank, title in enumerate(retrieved[k:], start=k + 1):
        if title in gold_set:
            return rank
    return None
