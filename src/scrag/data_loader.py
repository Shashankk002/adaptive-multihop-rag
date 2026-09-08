"""HotpotQA (distractor setting) loading.

This is the only module that understands HotpotQA's on-disk shape. It reads the
official JSON and produces `scrag.schema` records; everything downstream is
dataset-agnostic. To add 2WikiMultihopQA or MuSiQue later, add a sibling loader —
do not generalise this one.

The HotpotQA distractor format is a JSON list of objects:

    {
      "_id": str,
      "question": str,
      "answer": str,                       # absent on the held-out test split
      "type": "bridge" | "comparison",
      "level": "easy" | "medium" | "hard",
      "context": [[title, [sentence, ...]], ...],   # 10 paragraphs: 2 gold, 8 distractors
      "supporting_facts": [[title, sent_id], ...]   # absent on the held-out test split
    }
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from scrag.schema import Example, Paragraph, QuestionType, SupportingFact


class HotpotQAFormatError(ValueError):
    """The input does not match the HotpotQA distractor format."""


@dataclass
class LoadReport:
    """What the loader saw, including anomalies it tolerated.

    The dataset is not perfectly clean: a small number of supporting facts point at
    sentence indices that do not exist. Those are counted here rather than dropped
    quietly, so a later stage measuring evidence recall knows its ceiling is not 1.0.
    """

    n_examples: int = 0
    n_paragraphs: int = 0
    n_supporting_facts: int = 0
    type_counts: Counter[str] = field(default_factory=Counter)
    unresolved_facts: list[tuple[str, str, int]] = field(default_factory=list)
    unlabelled: int = 0

    @property
    def n_unresolved_facts(self) -> int:
        return len(self.unresolved_facts)

    def summary(self) -> str:
        types = ", ".join(f"{k}={v}" for k, v in sorted(self.type_counts.items()))
        return (
            f"{self.n_examples} examples | {self.n_paragraphs} paragraphs | "
            f"{self.n_supporting_facts} supporting facts | {types} | "
            f"unlabelled={self.unlabelled} | unresolved facts={self.n_unresolved_facts}"
        )


def load_hotpotqa(
    path: str | Path,
    *,
    strict: bool = False,
    limit: int | None = None,
) -> list[Example]:
    """Load a HotpotQA distractor JSON file into `Example` records.

    `strict` raises on data anomalies (unresolved supporting facts) instead of
    recording them. `limit` reads only the first N examples, for smoke runs.

    Use `load_hotpotqa_with_report` when the anomaly counts matter.
    """
    examples, _ = load_hotpotqa_with_report(path, strict=strict, limit=limit)
    return examples


def load_hotpotqa_with_report(
    path: str | Path,
    *,
    strict: bool = False,
    limit: int | None = None,
) -> tuple[list[Example], LoadReport]:
    """As `load_hotpotqa`, but also returns a `LoadReport` of what was seen."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on a corrupt file
        raise HotpotQAFormatError(f"{path} is not valid JSON: {exc}") from exc
    return parse_hotpotqa(raw, strict=strict, limit=limit)


def parse_hotpotqa(
    raw: Any,
    *,
    strict: bool = False,
    limit: int | None = None,
) -> tuple[list[Example], LoadReport]:
    """Parse already-decoded HotpotQA JSON. Kept separate from IO so it is testable
    against in-memory fixtures."""
    if not isinstance(raw, list):
        raise HotpotQAFormatError(
            f"expected a JSON list of examples, got {type(raw).__name__}"
        )

    report = LoadReport()
    examples: list[Example] = []
    for position, record in enumerate(_head(raw, limit)):
        example = _parse_record(record, position=position, strict=strict, report=report)
        examples.append(example)

    report.n_examples = len(examples)
    return examples, report


def _head(items: list[Any], limit: int | None) -> Iterable[Any]:
    return items if limit is None else items[:limit]


def _parse_record(
    record: Any,
    *,
    position: int,
    strict: bool,
    report: LoadReport,
) -> Example:
    if not isinstance(record, dict):
        raise HotpotQAFormatError(
            f"example at position {position}: expected an object, "
            f"got {type(record).__name__}"
        )

    qid = _require_str(record, "_id", position)
    question = _require_str(record, "question", position)
    paragraphs = _parse_context(record.get("context"), qid=qid)
    facts = _parse_supporting_facts(record.get("supporting_facts"), qid=qid)

    answer = record.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise HotpotQAFormatError(f"example {qid}: 'answer' must be a string")

    example = Example(
        qid=qid,
        question=question,
        paragraphs=paragraphs,
        supporting_facts=facts,
        answer=answer,
        question_type=QuestionType.parse(record.get("type")),
        difficulty=record.get("level"),
    )

    _record_stats(example, strict=strict, report=report)
    return example


def _parse_context(context: Any, *, qid: str) -> tuple[Paragraph, ...]:
    if context is None:
        raise HotpotQAFormatError(f"example {qid}: missing 'context'")
    if not isinstance(context, list):
        raise HotpotQAFormatError(f"example {qid}: 'context' must be a list")

    paragraphs: list[Paragraph] = []
    for entry in context:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise HotpotQAFormatError(
                f"example {qid}: each context entry must be [title, sentences]"
            )
        title, sentences = entry
        if not isinstance(title, str):
            raise HotpotQAFormatError(f"example {qid}: context title must be a string")
        if not isinstance(sentences, list) or not all(
            isinstance(s, str) for s in sentences
        ):
            raise HotpotQAFormatError(
                f"example {qid}: context sentences for {title!r} must be a list of strings"
            )
        paragraphs.append(Paragraph(title=title, sentences=tuple(sentences)))
    return tuple(paragraphs)


def _parse_supporting_facts(facts: Any, *, qid: str) -> tuple[SupportingFact, ...]:
    if facts is None:
        return ()
    if not isinstance(facts, list):
        raise HotpotQAFormatError(f"example {qid}: 'supporting_facts' must be a list")

    parsed: list[SupportingFact] = []
    for entry in facts:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise HotpotQAFormatError(
                f"example {qid}: each supporting fact must be [title, sent_id]"
            )
        title, sent_id = entry
        if not isinstance(title, str) or not isinstance(sent_id, int):
            raise HotpotQAFormatError(
                f"example {qid}: supporting fact must be [str, int], got "
                f"[{type(title).__name__}, {type(sent_id).__name__}]"
            )
        parsed.append(SupportingFact(title=title, sent_id=sent_id))
    return tuple(parsed)


def _record_stats(example: Example, *, strict: bool, report: LoadReport) -> None:
    report.n_paragraphs += len(example.paragraphs)
    report.n_supporting_facts += len(example.supporting_facts)
    report.type_counts[example.question_type.value] += 1
    if not example.is_labelled:
        report.unlabelled += 1

    for fact in example.supporting_facts:
        paragraph = example.paragraph_by_title(fact.title)
        if paragraph is not None and paragraph.sentence(fact.sent_id) is not None:
            continue
        if strict:
            raise HotpotQAFormatError(
                f"example {example.qid}: supporting fact "
                f"({fact.title!r}, {fact.sent_id}) does not resolve to a sentence"
            )
        report.unresolved_facts.append((example.qid, fact.title, fact.sent_id))


def _require_str(record: dict, key: str, position: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise HotpotQAFormatError(
            f"example at position {position}: missing or non-string {key!r}"
        )
    return value
