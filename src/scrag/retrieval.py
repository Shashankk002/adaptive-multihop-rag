"""BM25 retrieval over one question's candidate paragraphs.

HotpotQA distractor gives each question its own 10-paragraph candidate set, so there is
no global index: each question is scored as its own tiny corpus, built and discarded.

Written out rather than taken from a library because the corpus is ten short documents
(no library's performance work is relevant here) and because the IDF variant below is a
decision we want visible rather than inherited.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, NamedTuple, Sequence

from scrag.schema import Example, Paragraph

K1 = 1.5  # term-frequency saturation
B = 0.75  # length normalisation strength


class Scored(NamedTuple):
    paragraph: Paragraph
    score: float


Retriever = Callable[[Example, str, int], list[Scored]]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. No stemming or stopword list — BM25's IDF already
    discounts common terms, and each extra step is an unmeasured guess."""
    return re.findall(r"\w+", text.lower())


def indexed_text(paragraph: Paragraph) -> str:
    """Title plus body. 65% of HotpotQA questions name a gold paragraph's title
    verbatim, so dropping titles would discard the strongest lexical cue available."""
    return f"{paragraph.title} {paragraph.text}"


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Okapi BM25 score of every document against the query."""
    docs = [tokenize(d) for d in documents]
    n = len(docs)
    if n == 0:
        return []

    counts = [Counter(d) for d in docs]
    avg_len = sum(len(d) for d in docs) / n

    scores = [0.0] * n
    for term in set(tokenize(query)):
        df = sum(1 for c in counts if term in c)
        if df == 0:
            continue
        # The +1 keeps IDF positive. With only ~10 documents the textbook form goes
        # negative for any term in more than half of them, which would penalise a
        # paragraph for containing a query word.
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, counted in enumerate(counts):
            tf = counted[term]
            if tf:
                length_norm = 1 - B + B * len(docs[i]) / avg_len
                scores[i] += idf * tf * (K1 + 1) / (tf + K1 * length_norm)
    return scores


def bm25_retrieve(example: Example, query: str, k: int) -> list[Scored]:
    """Top-k candidate paragraphs for the query, best first."""
    paragraphs = example.paragraphs
    scores = bm25_scores(query, [indexed_text(p) for p in paragraphs])
    # sorted() is stable, so tied paragraphs keep their candidate order and the
    # ranking is reproducible across runs.
    order = sorted(range(len(paragraphs)), key=lambda i: -scores[i])
    return [Scored(paragraphs[i], scores[i]) for i in order[:k]]
