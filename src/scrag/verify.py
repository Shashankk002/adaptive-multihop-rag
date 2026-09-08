"""Evidence verification: does the retrieved evidence answer the question?

One Gemini call per question: the question plus the retrieved paragraphs go in, a
SUFFICIENT/INSUFFICIENT verdict comes out. `_generate` is the only place that talks to
a provider, so swapping models or vendors means replacing one function.

Verdicts are cached to disk because the model behind `MODEL` is a moving alias, not a
pinned snapshot: unlike the embedding model (pinned to a git revision) we cannot
reproduce a Gemini call exactly. The cache, plus the recorded model string and date,
is therefore the reproducible artifact.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "v1"
"""Bump on any prompt or schema change. It is part of the cache key, so old and new
verdicts can never silently mix."""

SEED = 20260908
CACHE_PATH = Path("data/processed/verify_cache.jsonl")

SYSTEM_INSTRUCTION = """\
You judge whether retrieved documents contain enough information to answer a question.

Rules:
- Judge ONLY the text provided below. Do not use outside knowledge, and do not rely on
  facts you happen to know about the subject.
- Evidence is SUFFICIENT only if the answer can be derived entirely from the provided
  text, combining documents where that is needed.
- If answering would require any fact not stated in the text, the evidence is
  INSUFFICIENT — even when the documents are clearly on-topic.
- Being related to the question is not the same as answering it."""

# `reason` is generated first on purpose: with thinking disabled it acts as a minimal
# chain of thought before the model commits to a verdict.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["reason", "sufficient", "confidence"],
    "propertyOrdering": ["reason", "sufficient", "confidence"],
}


def build_prompt(question: str, paragraphs: list) -> str:
    """The user-content half of the request. Deliberately says nothing about how many
    documents to expect — that would leak HotpotQA's structure."""
    blocks = [f"Question: {question}"]
    for i, paragraph in enumerate(paragraphs, start=1):
        blocks.append(
            f"Document {i}\nTitle: {paragraph.title}\nText: {paragraph.text.strip()}"
        )
    return "\n\n".join(blocks)


@dataclass
class Verdict:
    """One cached verification. `sufficient` is None when the call failed."""

    key: str
    qid: str
    sufficient: bool | None
    confidence: float | None
    reason: str
    model: str = MODEL
    prompt_version: str = PROMPT_VERSION
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.sufficient is not None


def cache_key(question: str, paragraphs: list) -> str:
    """Identity is the exact prompt sent, plus the model and prompt version."""
    payload = f"{MODEL}\n{PROMPT_VERSION}\n{build_prompt(question, paragraphs)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cache(path: Path = CACHE_PATH) -> dict[str, Verdict]:
    """Read the whole cache. Later rows win, so a re-run can supersede a failure."""
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cache[row["key"]] = Verdict(**row)
    return cache


def append_cache(verdict: Verdict, path: Path = CACHE_PATH) -> None:
    """Append one verdict immediately, so an interrupted run resumes where it stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(verdict)) + "\n")


# --- Provider ---------------------------------------------------------------------

_client = None


def _generate(system: str, prompt: str) -> str:
    """Send one request and return the raw response text.

    The only provider-specific code in the project. Tests replace it.
    """
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
            # No thinking configuration: this model answers directly, and a binary
            # sufficiency judgement does not need a reasoning budget.
            max_output_tokens=256,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    return response.text or ""


# --- Verification -----------------------------------------------------------------

RETRY_DELAYS = (2, 8, 30)
"""Backoff before each retry when the API gives no hint. Bounded by MAX_RETRY_DELAY."""

MAX_RETRY_DELAY = 60.0
MAX_CONSECUTIVE_429 = 5
"""After this many questions in a row end in a rate limit, the quota is gone rather
than momentarily busy, and further calls only waste time. The runner checks
`quota_exhausted()` and stops; the cache makes the run resumable later."""

_consecutive_429 = 0


def quota_exhausted() -> bool:
    """True once the circuit breaker has tripped. Reset with `reset_quota_state`."""
    return _consecutive_429 >= MAX_CONSECUTIVE_429


def reset_quota_state() -> None:
    global _consecutive_429
    _consecutive_429 = 0


def _status(exc: Exception) -> int | None:
    return getattr(exc, "code", None) or getattr(exc, "status_code", None)


def retry_after(exc: Exception) -> float | None:
    """Seconds the API asked us to wait, from the Retry-After header or RetryInfo.

    Google returns the structured hint inside `error.details`; the header is not
    always present. Honouring it beats guessing at a backoff.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass

    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in details.get("error", {}).get("details", []):
            delay = item.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    pass
    return None


def quota_violations(exc: Exception) -> list[dict]:
    """The QuotaFailure entries Google attaches to a 429, if any — which quota was
    hit, and its value. Recorded so a run can report why it stopped."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return []
    for item in details.get("error", {}).get("details", []):
        if "QuotaFailure" in str(item.get("@type", "")):
            return item.get("violations", [])
    return []


def _is_permanent(exc: Exception) -> bool:
    """True for failures that retrying cannot fix.

    A bad key or a withdrawn model fails identically on every attempt, so retrying
    only multiplies the wasted time by the length of the backoff ladder. 429 is the
    exception: it is a 4xx that clears on its own.
    """
    status = _status(exc)
    if status is None:
        return False
    return 400 <= status < 500 and status != 429


def _parse(raw: str, key: str, qid: str) -> Verdict:
    """Turn a raw response into a Verdict, or raise ValueError."""
    data = json.loads(raw)
    confidence = float(data["confidence"])
    return Verdict(
        key=key,
        qid=qid,
        sufficient=bool(data["sufficient"]),
        confidence=min(1.0, max(0.0, confidence)),
        reason=str(data.get("reason", ""))[:500],
    )


def verify(question: str, paragraphs: list, qid: str = "") -> Verdict:
    """Judge whether `paragraphs` contain enough information to answer `question`.

    Retries transient failures, then records the failure rather than guessing: a
    Verdict with `sufficient=None` is excluded from metrics, never counted as a label.
    """
    key = cache_key(question, paragraphs)
    prompt = build_prompt(question, paragraphs)

    global _consecutive_429

    last_error = ""
    rate_limited = False
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            verdict = _parse(_generate(SYSTEM_INSTRUCTION, prompt), key, qid)
            _consecutive_429 = 0
            return verdict
        except Exception as exc:  # network, rate limit, truncation, bad JSON
            last_error = f"{type(exc).__name__}: {exc}"[:1500]
            rate_limited = _status(exc) == 429
            if _is_permanent(exc):
                break
            if attempt < len(RETRY_DELAYS):
                hinted = retry_after(exc)
                delay = RETRY_DELAYS[attempt] if hinted is None else hinted
                time.sleep(min(delay, MAX_RETRY_DELAY))

    if rate_limited:
        _consecutive_429 += 1

    return Verdict(key=key, qid=qid, sufficient=None, confidence=None, reason="",
                   error=last_error)
