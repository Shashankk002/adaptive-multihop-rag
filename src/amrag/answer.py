"""Grounded answer generation.

One Gemini call per question: the question and the selected evidence go in, a short
answer comes out. The model is told to answer only from the supplied text and to say
INSUFFICIENT when it cannot — that abstention is the signal that distinguishes a
grounded answer from one recalled from pretraining.

Plain text, not JSON: HotpotQA answers are two words at the median, so a schema would
add a parse-failure mode and wasted tokens for no gain. Generation is kept separate
from verification — different prompt, different call, different cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from amrag.schema import Paragraph
from amrag.verify import retry_after  # shared parsing of Google's RetryInfo hint

MODEL = "gemini-3.1-flash-lite"
PROMPT_VERSION = "v1"
SEED = 20260908
MAX_OUTPUT_TOKENS = 64
CACHE_PATH = Path("data/processed/answer_cache.jsonl")

INSUFFICIENT = "INSUFFICIENT"

SYSTEM_INSTRUCTION = """\
Answer the question using ONLY the documents provided below.

Rules:
- Do not use outside knowledge. Do not rely on facts you happen to know about the
  subject.
- Return the answer only. No explanation, no full sentence.
- If the question is a yes/no question, answer exactly "yes" or "no".
- If the documents do not contain enough information to answer, return exactly:
  INSUFFICIENT"""

RETRY_DELAYS = (2, 8, 30)
MAX_RETRY_DELAY = 60.0


class AnswerError(RuntimeError):
    """Generation failed and produced no usable answer."""


def build_prompt(question: str, paragraphs: Sequence[Paragraph]) -> str:
    """Question plus the evidence, in ranking order."""
    blocks = [f"Question: {question}"]
    for i, paragraph in enumerate(paragraphs, start=1):
        blocks.append(
            f"Document {i}\nTitle: {paragraph.title}\nText: {paragraph.text.strip()}"
        )
    return "\n\n".join(blocks)


# --- Provider ---------------------------------------------------------------------

_client = None


def _generate(system: str, prompt: str) -> str:
    """Send one request and return the raw response text. Tests replace this."""
    global _client
    from google import genai
    from google.genai import types

    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY from the environment

    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            seed=SEED,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text or ""


def _is_permanent(exc: Exception) -> bool:
    """A 4xx other than 429 fails identically on every attempt."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return status is not None and 400 <= status < 500 and status != 429


def answer(question: str, paragraphs: Sequence[Paragraph]) -> str:
    """The answer to `question` from `paragraphs` alone, or "INSUFFICIENT".

    Raises AnswerError when generation fails, rather than returning something that
    could be mistaken for an answer.
    """
    prompt = build_prompt(question, paragraphs)

    last_error = ""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            text = _generate(SYSTEM_INSTRUCTION, prompt).strip()
            if not text:
                raise ValueError("empty generation")
            return text
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:1500]
            if _is_permanent(exc):
                break
            if attempt < len(RETRY_DELAYS):
                hinted = retry_after(exc)
                delay = RETRY_DELAYS[attempt] if hinted is None else hinted
                time.sleep(min(delay, MAX_RETRY_DELAY))

    raise AnswerError(last_error)


# --- Cache ------------------------------------------------------------------------


@dataclass
class Answer:
    """One cached generation. `text` is None when generation failed."""

    key: str
    qid: str
    text: str | None
    model: str = MODEL
    prompt_version: str = PROMPT_VERSION
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.text is not None


def cache_key(question: str, paragraphs: Sequence[Paragraph]) -> str:
    """Identity is the exact prompt sent. Keying on content rather than on a policy
    name means two policies that select the same evidence share one generation."""
    payload = f"{MODEL}\n{PROMPT_VERSION}\n{build_prompt(question, paragraphs)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def generate(question: str, paragraphs: Sequence[Paragraph], qid: str = "") -> Answer:
    """`answer` as a cacheable record: a failure becomes a record, not an exception."""
    key = cache_key(question, paragraphs)
    try:
        return Answer(key=key, qid=qid, text=answer(question, paragraphs))
    except AnswerError as exc:
        return Answer(key=key, qid=qid, text=None, error=str(exc))


def load_cache(path: Path = CACHE_PATH) -> dict[str, Answer]:
    """Read the whole cache. Later rows win, so a re-run can supersede a failure."""
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["key"]] = Answer(**row)
    return cache


def append_cache(record: Answer, path: Path = CACHE_PATH) -> None:
    """Append one record immediately, so an interrupted run resumes where it stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")
