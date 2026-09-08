"""Cross-encoder reranking of the dense shortlist.

The bi-encoder in `dense.py` embeds question and paragraph separately, so paragraph
vectors can be precomputed. A cross-encoder instead reads the pair together and emits
one relevance score, which is more accurate and far more expensive — nothing can be
cached, since every question makes a different pair. Hence the standard two stages:
dense narrows 10 candidates to `depth`, the cross-encoder orders those.

Scores are raw logits, unbounded and not comparable to cosine or RRF scores. They are
used for ranking only, which is all our metrics consume.
"""

from __future__ import annotations

from scrag.dense import dense_retrieve
from scrag.retrieval import Scored, indexed_text
from scrag.schema import Example

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEVICE = "mps"  # 2.4x faster than CPU here, with identical top-k rankings
DEFAULT_DEPTH = 5

_model = None


def get_reranker():
    """Load the cross-encoder once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME, revision=MODEL_REVISION, device=DEVICE)
    return _model


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """Relevance of each text to the query.

    The query is passed raw: the bge instruction prefix belongs to the bi-encoder and
    would be a formatting error here, since MS MARCO cross-encoders are trained on
    bare query/passage pairs.
    """
    scores = get_reranker().predict(
        [(query, text) for text in texts], batch_size=32, show_progress_bar=False
    )
    return [float(s) for s in scores]


def rerank_retrieve(
    example: Example, query: str, k: int, depth: int = DEFAULT_DEPTH
) -> list[Scored]:
    """Dense-retrieve `depth` candidates, then reorder them with the cross-encoder."""
    shortlist = [s.paragraph for s in dense_retrieve(example, query, depth)]
    scores = score_pairs(query, [indexed_text(p) for p in shortlist])
    # Stable sort, so ties keep the dense shortlist order.
    order = sorted(range(len(shortlist)), key=lambda i: -scores[i])
    return [Scored(shortlist[i], scores[i]) for i in order[:k]]
