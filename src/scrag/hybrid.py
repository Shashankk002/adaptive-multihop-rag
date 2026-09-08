"""Hybrid retrieval: Reciprocal Rank Fusion of BM25 and dense rankings.

RRF combines rankings rather than scores, which sidesteps the fact that BM25 scores are
unbounded and corpus-dependent while cosine scores live in [-1, 1]. No normalisation
step, no calibration, one parameter.

Both retrievers rank every candidate paragraph, so each paragraph has a rank in both
lists and the usual "document missing from one ranking" case cannot arise.
"""

from __future__ import annotations

from typing import Sequence

from scrag.dense import dense_retrieve
from scrag.retrieval import Scored, bm25_retrieve
from scrag.schema import Example

DEFAULT_RRF_K = 1
"""Chosen by sweeping DEV both@2; see PROJECT_PLAN.md.

The published default of 60 is tuned for ~1000-document TREC lists. Over 10 candidates
it flattens RRF into "average rank": weights span only 1/61 to 1/70. A small constant
keeps the gap between rank 1 and rank 2 meaningful, which matters because when one
retriever misses a gold paragraph the other often has it ranked first.
"""


def rrf_scores(rankings: Sequence[Sequence[str]], rrf_k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion score for every item, summed over rankings.

        RRF(d) = sum over rankings of 1 / (rrf_k + rank(d))    (rank is 1-based)
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def hybrid_retrieve(
    example: Example, query: str, k: int, rrf_k: int = DEFAULT_RRF_K
) -> list[Scored]:
    """Top-k paragraphs by RRF over the complete BM25 and dense rankings."""
    paragraphs = example.paragraphs
    n = len(paragraphs)
    rankings = [
        [s.paragraph.title for s in bm25_retrieve(example, query, n)],
        [s.paragraph.title for s in dense_retrieve(example, query, n)],
    ]
    scores = rrf_scores(rankings, rrf_k)
    # Stable sort, so ties keep candidate order — the same rule the other retrievers use.
    order = sorted(range(n), key=lambda i: -scores[paragraphs[i].title])
    return [Scored(paragraphs[i], scores[paragraphs[i].title]) for i in order[:k]]
