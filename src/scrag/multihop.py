"""Two-hop retrieval: retrieve, expand the query, retrieve again.

HotpotQA's second gold paragraph is often unreachable from the question alone — it is
reachable only through something the first paragraph says. Dense retrieval already
finds one gold paragraph for 89% of bridge questions, so hop 1 needs no help; the whole
deficit is hop 2. Adding hop-1 text to the query supplies the missing context.

Exactly two hops, with no loop or termination rule: every question in the dataset has
exactly two gold paragraphs, so a third hop would be solving a problem that is not here.
"""

from __future__ import annotations

from scrag.dense import dense_retrieve
from scrag.retrieval import Scored, bm25_scores, indexed_text
from scrag.schema import Example, Paragraph

DEFAULT_EXPANSION = "paragraph"

ROUTING_THRESHOLD = 0.020
"""Run hop 2 only when the dense rank-2/rank-3 score gap is below this.

A wide gap means dense has cleanly separated a pair of paragraphs from the rest and a
second hop has nothing to add — expanding the query there mostly does damage, which is
what sinks 2-hop-everywhere on comparison questions. Frozen from a TUNE sweep (n=1000);
the stable region was 0.010-0.030 and this is its midpoint. Note the rank-1/rank-2 gap
does *not* work as this signal: it is mildly anti-predictive.
"""


def expansion_text(paragraph: Paragraph, question: str, mode: str) -> str:
    """The hop-1 text appended to the question to form the hop-2 query."""
    if mode == "title":
        return paragraph.title
    if mode == "paragraph":
        return indexed_text(paragraph)
    if mode == "sentence":
        # The single sentence most lexically similar to the question. Cheaper and
        # narrower than the whole paragraph, and it reuses the Phase 2 scorer.
        scores = bm25_scores(question, list(paragraph.sentences))
        best = max(range(len(scores)), key=lambda i: scores[i])
        return f"{paragraph.title} {paragraph.sentences[best]}"
    raise ValueError(f"unknown expansion mode: {mode}")


def score_margin(ranked: list[Scored]) -> float:
    """Gap between the 2nd and 3rd dense scores. Infinite with fewer than 3
    candidates, where there is no third paragraph to be confused with."""
    return ranked[1].score - ranked[2].score if len(ranked) > 2 else float("inf")


def routed_retrieve(
    example: Example, query: str, k: int, expansion: str = DEFAULT_EXPANSION
) -> list[Scored]:
    """Two hops only when dense looks unsure; otherwise the dense ranking as-is."""
    ranked = dense_retrieve(example, query, len(example.paragraphs))
    if score_margin(ranked) >= ROUTING_THRESHOLD:
        return ranked[:k]
    return multihop_retrieve(example, query, k, expansion)


def multihop_retrieve(
    example: Example, query: str, k: int, expansion: str = DEFAULT_EXPANSION
) -> list[Scored]:
    """Rank candidates by two hops of dense retrieval."""
    first = dense_retrieve(example, query, 1)[0]

    hop2_query = f"{query} {expansion_text(first.paragraph, query, expansion)}"
    rest = [
        s for s in dense_retrieve(example, hop2_query, len(example.paragraphs))
        if s.paragraph.title != first.paragraph.title
    ]
    return [first, *rest][:k]
