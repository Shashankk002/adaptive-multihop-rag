"""Dataset-agnostic representation of a multi-hop QA example.

Nothing in this module knows about HotpotQA, or about any other benchmark. Loaders
translate their own formats into these types; every downstream stage (retrieval,
reranking, verification, answering) consumes only these types. Adding a new dataset
means adding a loader, not changing anything here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionType(str, Enum):
    """How a multi-hop question is composed.

    BRIDGE     — the answer to one hop is needed to ask the next.
    COMPARISON — two independently retrievable facts are compared.
    OTHER      — a dataset uses a label we do not model yet.
    """

    BRIDGE = "bridge"
    COMPARISON = "comparison"
    OTHER = "other"

    @classmethod
    def parse(cls, raw: str | None) -> "QuestionType":
        if raw is None:
            return cls.OTHER
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.OTHER


@dataclass(frozen=True)
class Paragraph:
    """One candidate paragraph, kept split into sentences.

    Sentence granularity is preserved because supporting facts are labelled per
    sentence; the verification stage needs that granularity.
    """

    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        """The paragraph as a single string."""
        return "".join(self.sentences)

    def sentence(self, index: int) -> str | None:
        """Sentence at `index`, or None if the index is out of range."""
        if 0 <= index < len(self.sentences):
            return self.sentences[index]
        return None


@dataclass(frozen=True)
class SupportingFact:
    """A gold evidence sentence, addressed by paragraph title and sentence index."""

    title: str
    sent_id: int


@dataclass(frozen=True)
class Example:
    """A single multi-hop question with its candidate evidence.

    `answer` and `supporting_facts` are absent on unlabelled splits (e.g. HotpotQA's
    held-out test set), so both are optional/empty rather than required.
    """

    qid: str
    question: str
    paragraphs: tuple[Paragraph, ...]
    supporting_facts: tuple[SupportingFact, ...] = ()
    answer: str | None = None
    question_type: QuestionType = QuestionType.OTHER
    difficulty: str | None = None

    @property
    def is_labelled(self) -> bool:
        return self.answer is not None and bool(self.supporting_facts)

    @property
    def gold_titles(self) -> frozenset[str]:
        """Titles of the paragraphs containing gold evidence."""
        return frozenset(fact.title for fact in self.supporting_facts)

    def paragraph_by_title(self, title: str) -> Paragraph | None:
        for paragraph in self.paragraphs:
            if paragraph.title == title:
                return paragraph
        return None

    def supporting_sentences(self) -> tuple[str, ...]:
        """The gold evidence sentences, in the order the facts are listed.

        Facts whose sentence index does not resolve are skipped; the dataset contains
        a small number of these. `LoadReport` counts them at load time.
        """
        found: list[str] = []
        for fact in self.supporting_facts:
            paragraph = self.paragraph_by_title(fact.title)
            if paragraph is None:
                continue
            sentence = paragraph.sentence(fact.sent_id)
            if sentence is not None:
                found.append(sentence)
        return tuple(found)
