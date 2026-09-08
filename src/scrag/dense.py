"""Dense retrieval with sentence embeddings.

Same interface as BM25 (`Retriever`), same indexed text (title + body), so the two are
directly comparable. Similarity is a dot product over L2-normalised vectors, which is
cosine similarity; with ~10 candidates per question there is nothing for an index
structure to accelerate.

Embeddings are cached to a single .npz keyed by the SHA1 of the exact text embedded,
because a full DEV sweep otherwise costs several minutes on every rerun.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scrag.retrieval import Scored, indexed_text
from scrag.schema import Example

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEVICE = "mps"  # 2.5x faster than CPU here, with identical top-k rankings
DIMENSION = 384

# bge models are trained for asymmetric retrieval: the query gets an instruction
# prefix, the passages do not. Leaving it off measurably hurts accuracy.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CACHE_PATH = Path("data/processed/embeddings_bge-small-en-v1.5.npz")

_model = None
_cache: dict[str, np.ndarray] | None = None


class CacheMismatch(RuntimeError):
    """The cache on disk was built with a different model or revision."""


def get_model():
    """Load the model once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            MODEL_NAME, revision=MODEL_REVISION, device=DEVICE
        )
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Run the model. Returns L2-normalised vectors, so dot product == cosine."""
    return get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )


def embed(texts: list[str]) -> np.ndarray:
    """Vectors for `texts`, served from the cache where possible."""
    if _cache is None:
        return encode(texts)

    missing = [t for t in texts if _key(t) not in _cache]
    if missing:
        for text, vector in zip(missing, encode(missing)):
            _cache[_key(text)] = vector
    return np.stack([_cache[_key(t)] for t in texts])


def dense_retrieve(example: Example, query: str, k: int) -> list[Scored]:
    """Top-k candidate paragraphs for the query, best first."""
    paragraphs = example.paragraphs
    vectors = embed([indexed_text(p) for p in paragraphs])
    query_vector = embed([QUERY_PREFIX + query])[0]
    scores = vectors @ query_vector
    # Stable sort, so tied paragraphs keep candidate order and runs are reproducible.
    order = sorted(range(len(paragraphs)), key=lambda i: -scores[i])
    return [Scored(paragraphs[i], float(scores[i])) for i in order[:k]]


# --- Cache -----------------------------------------------------------------------


def _key(text: str) -> str:
    """Identity is the exact text embedded, so a different indexing choice (say
    body-only) simply produces different keys and can never collide."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_cache(texts: list[str], path: Path = CACHE_PATH) -> int:
    """Embed every unique text and write the cache. Returns the number stored."""
    unique = sorted(set(texts))
    vectors = encode(unique)
    meta = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "dimension": DIMENSION,
        "device": DEVICE,
        "count": len(unique),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        keys=np.array([_key(t) for t in unique]),
        vectors=vectors.astype(np.float32),
        meta=np.array(json.dumps(meta)),
    )
    return len(unique)


def load_cache(path: Path = CACHE_PATH) -> dict:
    """Load the cache into memory. Refuses a cache built by a different model."""
    global _cache
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    if (meta["model"], meta["revision"]) != (MODEL_NAME, MODEL_REVISION):
        raise CacheMismatch(
            f"{path} holds {meta['model']}@{meta['revision'][:8]}, "
            f"expected {MODEL_NAME}@{MODEL_REVISION[:8]}"
        )
    _cache = dict(zip(data["keys"].tolist(), data["vectors"]))
    return meta
