"""Tests for the verification cache and prompt construction.

No Gemini API calls: nothing here touches the network or needs a key.
"""

from __future__ import annotations

import pytest

from amrag import verify
from amrag.verify import Verdict, append_cache, build_prompt, cache_key, load_cache
from amrag.schema import Paragraph

PARAGRAPHS = [
    Paragraph("Tim Burton", ("He directed Ed Wood.",)),
    Paragraph("Ed Wood (film)", ("A 1994 film.",)),
]


class TestPrompt:
    def test_contains_question_and_both_documents(self):
        prompt = build_prompt("Who directed Ed Wood?", PARAGRAPHS)
        assert "Question: Who directed Ed Wood?" in prompt
        assert "Title: Tim Burton" in prompt
        assert "Title: Ed Wood (film)" in prompt
        assert "He directed Ed Wood." in prompt

    def test_documents_are_numbered_in_ranking_order(self):
        prompt = build_prompt("q", PARAGRAPHS)
        assert prompt.index("Document 1") < prompt.index("Document 2")
        assert prompt.index("Tim Burton") < prompt.index("Ed Wood (film)")

    def test_leaks_no_dataset_structure(self):
        prompt = (build_prompt("q", PARAGRAPHS) + verify.SYSTEM_INSTRUCTION).lower()
        for leak in ("two documents", "both paragraphs", "hotpot", "supporting fact",
                     "bridge", "comparison", "gold"):
            assert leak not in prompt

    def test_handles_a_single_document(self):
        assert "Document 2" not in build_prompt("q", PARAGRAPHS[:1])


class TestCacheKey:
    def test_same_input_gives_the_same_key(self):
        assert cache_key("q", PARAGRAPHS) == cache_key("q", PARAGRAPHS)

    def test_different_question_gives_a_different_key(self):
        assert cache_key("q", PARAGRAPHS) != cache_key("other", PARAGRAPHS)

    def test_document_order_changes_the_key(self):
        assert cache_key("q", PARAGRAPHS) != cache_key("q", PARAGRAPHS[::-1])

    def test_prompt_version_is_part_of_the_key(self, monkeypatch):
        before = cache_key("q", PARAGRAPHS)
        monkeypatch.setattr(verify, "PROMPT_VERSION", "v2")
        assert cache_key("q", PARAGRAPHS) != before

    def test_model_is_part_of_the_key(self, monkeypatch):
        before = cache_key("q", PARAGRAPHS)
        monkeypatch.setattr(verify, "MODEL", "other-model")
        assert cache_key("q", PARAGRAPHS) != before


class TestCacheIO:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "verify.jsonl"
        v = Verdict(key="k1", qid="q1", sufficient=True, confidence=0.9, reason="ok")
        append_cache(v, path)

        loaded = load_cache(path)
        assert loaded["k1"].sufficient is True
        assert loaded["k1"].confidence == 0.9
        assert loaded["k1"].qid == "q1"

    def test_missing_file_is_an_empty_cache(self, tmp_path):
        assert load_cache(tmp_path / "absent.jsonl") == {}

    def test_appends_accumulate(self, tmp_path):
        path = tmp_path / "verify.jsonl"
        append_cache(Verdict("k1", "q1", True, 0.9, "a"), path)
        append_cache(Verdict("k2", "q2", False, 0.4, "b"), path)
        assert set(load_cache(path)) == {"k1", "k2"}

    def test_a_later_row_supersedes_an_earlier_one(self, tmp_path):
        # So a re-run can replace a recorded failure without rewriting the file.
        path = tmp_path / "verify.jsonl"
        append_cache(Verdict("k1", "q1", None, None, "", error="timeout"), path)
        append_cache(Verdict("k1", "q1", True, 0.8, "recovered"), path)

        assert load_cache(path)["k1"].sufficient is True

    def test_failed_verdicts_round_trip_as_none(self, tmp_path):
        path = tmp_path / "verify.jsonl"
        append_cache(Verdict("k1", "q1", None, None, "", error="MAX_TOKENS"), path)

        v = load_cache(path)["k1"]
        assert v.sufficient is None
        assert v.error == "MAX_TOKENS"
        assert not v.ok, "a failed verdict must never look like a usable one"

    def test_ok_flag(self):
        assert Verdict("k", "q", True, 0.9, "r").ok
        assert not Verdict("k", "q", None, None, "", error="boom").ok


class TestVerify:
    """The provider is faked, so no API call and no key are needed."""

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr(verify.time, "sleep", lambda _: None)

    def _fake(self, monkeypatch, *responses):
        """Replace the provider with one that returns each response in turn."""
        calls = []

        def fake_generate(system, prompt):
            calls.append((system, prompt))
            result = responses[min(len(calls) - 1, len(responses) - 1)]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(verify, "_generate", fake_generate)
        return calls

    GOOD = '{"reason": "Both facts are present.", "sufficient": true, "confidence": 0.9}'

    def test_parses_a_well_formed_response(self, monkeypatch):
        self._fake(monkeypatch, self.GOOD)
        v = verify.verify("q", PARAGRAPHS, qid="q1")

        assert v.sufficient is True
        assert v.confidence == 0.9
        assert v.reason == "Both facts are present."
        assert v.qid == "q1"
        assert v.ok

    def test_insufficient_verdict(self, monkeypatch):
        self._fake(monkeypatch, '{"reason": "no", "sufficient": false, "confidence": 0.7}')
        assert verify.verify("q", PARAGRAPHS).sufficient is False

    def test_sends_the_system_instruction_and_prompt(self, monkeypatch):
        calls = self._fake(monkeypatch, self.GOOD)
        verify.verify("Who directed Ed Wood?", PARAGRAPHS)

        system, prompt = calls[0]
        assert system == verify.SYSTEM_INSTRUCTION
        assert prompt == verify.build_prompt("Who directed Ed Wood?", PARAGRAPHS)

    def test_key_matches_the_cache_key(self, monkeypatch):
        self._fake(monkeypatch, self.GOOD)
        v = verify.verify("q", PARAGRAPHS)
        assert v.key == cache_key("q", PARAGRAPHS)

    def test_confidence_is_clamped(self, monkeypatch):
        self._fake(monkeypatch, '{"reason": "r", "sufficient": true, "confidence": 4.2}')
        assert verify.verify("q", PARAGRAPHS).confidence == 1.0

    def test_retries_then_succeeds(self, monkeypatch):
        calls = self._fake(monkeypatch, RuntimeError("429 rate limit"), self.GOOD)
        v = verify.verify("q", PARAGRAPHS)

        assert v.sufficient is True
        assert len(calls) == 2, "one retry, then success"

    def test_persistent_failure_is_recorded_not_guessed(self, monkeypatch):
        calls = self._fake(monkeypatch, RuntimeError("503 unavailable"))
        v = verify.verify("q", PARAGRAPHS, qid="q1")

        assert v.sufficient is None, "a failure must never become a verdict"
        assert v.confidence is None
        assert not v.ok
        assert "503" in v.error
        assert len(calls) == len(verify.RETRY_DELAYS) + 1

    def test_malformed_json_is_a_failure_not_a_verdict(self, monkeypatch):
        self._fake(monkeypatch, "not json at all")
        v = verify.verify("q", PARAGRAPHS)
        assert v.sufficient is None
        assert not v.ok

    def test_truncated_json_is_a_failure(self, monkeypatch):
        self._fake(monkeypatch, '{"reason": "cut off mid')
        assert verify.verify("q", PARAGRAPHS).sufficient is None

    def test_missing_required_field_is_a_failure(self, monkeypatch):
        self._fake(monkeypatch, '{"reason": "r", "confidence": 0.5}')
        assert verify.verify("q", PARAGRAPHS).sufficient is None

    def test_empty_response_is_a_failure(self, monkeypatch):
        self._fake(monkeypatch, "")
        assert verify.verify("q", PARAGRAPHS).sufficient is None

    def test_no_api_key_is_needed_for_any_of_this(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        self._fake(monkeypatch, self.GOOD)
        assert verify.verify("q", PARAGRAPHS).ok


class FakeAPIError(Exception):
    """Mimics google.genai's ClientError/ServerError, which carry a `code`."""

    def __init__(self, code: int, message: str = ""):
        super().__init__(f"{code} {message}")
        self.code = code


class TestFailFast:
    """Permanent 4xx failures must not burn the retry ladder — a bad key or a
    withdrawn model fails identically on every attempt."""

    @pytest.fixture(autouse=True)
    def no_sleep(self, monkeypatch):
        monkeypatch.setattr(verify.time, "sleep", lambda _: None)

    def _raising(self, monkeypatch, exc):
        calls = []

        def fake_generate(system, prompt):
            calls.append(prompt)
            raise exc

        monkeypatch.setattr(verify, "_generate", fake_generate)
        return calls

    @pytest.mark.parametrize(
        "code,label",
        [(401, "auth"), (403, "permission denied"), (404, "model not found"),
         (400, "invalid argument")],
    )
    def test_permanent_errors_are_tried_once(self, monkeypatch, code, label):
        calls = self._raising(monkeypatch, FakeAPIError(code, label))

        v = verify.verify("q", PARAGRAPHS, qid="q1")

        assert len(calls) == 1, f"{label} must not be retried"
        assert v.sufficient is None
        assert not v.ok
        assert str(code) in v.error

    def test_rate_limit_is_still_retried(self, monkeypatch):
        calls = self._raising(monkeypatch, FakeAPIError(429, "rate limit"))
        verify.verify("q", PARAGRAPHS)
        assert len(calls) == len(verify.RETRY_DELAYS) + 1, "429 clears on its own"

    def test_server_errors_are_still_retried(self, monkeypatch):
        calls = self._raising(monkeypatch, FakeAPIError(503, "unavailable"))
        verify.verify("q", PARAGRAPHS)
        assert len(calls) == len(verify.RETRY_DELAYS) + 1

    def test_errors_without_a_status_code_are_retried(self, monkeypatch):
        # e.g. a socket timeout — no way to know it is permanent.
        calls = self._raising(monkeypatch, ConnectionError("connection reset"))
        verify.verify("q", PARAGRAPHS)
        assert len(calls) == len(verify.RETRY_DELAYS) + 1

    def test_malformed_json_is_still_retried(self, monkeypatch):
        # Not an API error at all; a retry may well succeed.
        calls = []

        def fake_generate(system, prompt):
            calls.append(prompt)
            return "not json"

        monkeypatch.setattr(verify, "_generate", fake_generate)
        verify.verify("q", PARAGRAPHS)
        assert len(calls) == len(verify.RETRY_DELAYS) + 1

    @pytest.mark.parametrize("code", [400, 404])
    def test_permanent_failure_can_be_superseded_later(self, monkeypatch, tmp_path, code):
        """A recorded 4xx stays retryable: it is not `ok`, so the runner re-calls it."""
        path = tmp_path / "verify.jsonl"
        self._raising(monkeypatch, FakeAPIError(code))
        failed = verify.verify("q", PARAGRAPHS, qid="q1")
        verify.append_cache(failed, path)

        assert not verify.load_cache(path)[failed.key].ok


class FakeResponse:
    def __init__(self, headers=None):
        self.headers = headers or {}


class FakeQuotaError(Exception):
    """Mimics google.genai's ClientError, which carries .code, .details, .response."""

    def __init__(self, code=429, retry_delay=None, header=None, violations=None):
        self.code = code
        self.response = FakeResponse({"Retry-After": header} if header else {})
        details = []
        if retry_delay:
            details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": retry_delay})
        if violations:
            details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                            "violations": violations})
        self.details = {"error": {"code": code, "details": details}}
        # google.genai.errors.APIError puts the details into the message itself,
        # which is how the quota payload reaches our stored error string.
        super().__init__(f"{code} RESOURCE_EXHAUSTED. {self.details}")


class TestRetryAfter:
    def test_reads_the_retry_after_header(self):
        assert verify.retry_after(FakeQuotaError(header="17")) == 17.0

    def test_falls_back_to_retry_info_in_details(self):
        assert verify.retry_after(FakeQuotaError(retry_delay="42s")) == 42.0

    def test_header_wins_over_details(self):
        exc = FakeQuotaError(header="5", retry_delay="99s")
        assert verify.retry_after(exc) == 5.0

    def test_returns_none_when_no_hint_is_given(self):
        assert verify.retry_after(FakeQuotaError()) is None
        assert verify.retry_after(ConnectionError("reset")) is None

    def test_ignores_an_unparseable_hint(self):
        assert verify.retry_after(FakeQuotaError(header="soon")) is None

    def test_extracts_quota_violations(self):
        exc = FakeQuotaError(violations=[{"quotaId": "GenerateRequestsPerMinute",
                                          "quotaValue": "10"}])
        assert verify.quota_violations(exc)[0]["quotaId"] == "GenerateRequestsPerMinute"

    def test_no_violations_on_a_plain_error(self):
        assert verify.quota_violations(ConnectionError("reset")) == []


class TestBackoffAndCircuitBreaker:
    @pytest.fixture(autouse=True)
    def clean_state(self, monkeypatch):
        verify.reset_quota_state()
        self.slept = []
        monkeypatch.setattr(verify.time, "sleep", self.slept.append)
        yield
        verify.reset_quota_state()

    def _always(self, monkeypatch, exc):
        monkeypatch.setattr(verify, "_generate",
                            lambda system, prompt: (_ for _ in ()).throw(exc))

    def test_uses_the_api_hint_instead_of_the_default_ladder(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError(retry_delay="7s"))
        verify.verify("q", PARAGRAPHS)
        assert self.slept == [7.0, 7.0, 7.0], "the hint replaces the default backoff"

    def test_falls_back_to_the_default_ladder(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError())
        verify.verify("q", PARAGRAPHS)
        assert self.slept == list(verify.RETRY_DELAYS)

    def test_a_long_hint_is_capped(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError(retry_delay="3600s"))
        verify.verify("q", PARAGRAPHS)
        assert all(s == verify.MAX_RETRY_DELAY for s in self.slept)

    def test_retries_are_bounded_per_question(self, monkeypatch):
        calls = []
        monkeypatch.setattr(verify, "_generate",
                            lambda s, p: calls.append(1) or (_ for _ in ()).throw(FakeQuotaError()))
        verify.verify("q", PARAGRAPHS)
        assert len(calls) == len(verify.RETRY_DELAYS) + 1

    def test_breaker_trips_after_consecutive_rate_limits(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError())
        for _ in range(verify.MAX_CONSECUTIVE_429):
            assert not verify.quota_exhausted()
            verify.verify("q", PARAGRAPHS)
        assert verify.quota_exhausted(), "the runner should now stop cleanly"

    def test_a_success_resets_the_breaker(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError())
        for _ in range(verify.MAX_CONSECUTIVE_429 - 1):
            verify.verify("q", PARAGRAPHS)

        monkeypatch.setattr(verify, "_generate", lambda s, p:
                            '{"reason": "r", "sufficient": true, "confidence": 0.9}')
        assert verify.verify("q", PARAGRAPHS).ok
        assert not verify.quota_exhausted()

    def test_non_quota_failures_do_not_trip_the_breaker(self, monkeypatch):
        self._always(monkeypatch, ConnectionError("reset"))
        for _ in range(verify.MAX_CONSECUTIVE_429 + 2):
            verify.verify("q", PARAGRAPHS)
        assert not verify.quota_exhausted(), "only rate limits count"

    def test_permanent_errors_do_not_trip_the_breaker(self, monkeypatch):
        self._always(monkeypatch, FakeQuotaError(code=404))
        for _ in range(verify.MAX_CONSECUTIVE_429 + 2):
            verify.verify("q", PARAGRAPHS)
        assert not verify.quota_exhausted()

    def test_error_text_keeps_the_quota_details(self, monkeypatch):
        exc = FakeQuotaError(violations=[{"quotaId": "GenerateRequestsPerDay"}])
        self._always(monkeypatch, exc)
        v = verify.verify("q", PARAGRAPHS)
        assert "GenerateRequestsPerDay" in v.error, "details must survive truncation"


class TestProductionConfig:
    """Pins the approved Gemini configuration. These constants were agreed after a
    quota investigation; a silent edit would invalidate every cached verdict."""

    def test_model_and_prompt_version(self):
        assert verify.MODEL == "gemini-3.5-flash-lite"
        assert verify.PROMPT_VERSION == "v1"
        assert verify.SEED == 20260908

    def test_generate_sends_the_approved_config(self, monkeypatch):
        captured = {}

        class FakeModels:
            def generate_content(self, *, model, contents, config):
                captured.update(model=model, contents=contents, config=config)
                return type("R", (), {"text": '{"reason":"r","sufficient":true,"confidence":1}'})()

        monkeypatch.setattr(verify, "_client", type("C", (), {"models": FakeModels()})())

        verify._generate("SYS", "PROMPT")
        cfg = captured["config"]

        assert captured["model"] == "gemini-3.5-flash-lite"
        assert captured["contents"] == "PROMPT"
        assert cfg.system_instruction == "SYS"
        assert cfg.temperature == 0
        assert cfg.seed == verify.SEED
        assert cfg.max_output_tokens == 256
        assert cfg.response_mime_type == "application/json"
        assert cfg.response_schema == verify.RESPONSE_SCHEMA
        assert cfg.thinking_config is None, "this model takes no thinking configuration"

    def test_switching_model_changes_every_cache_key(self, monkeypatch):
        before = cache_key("q", PARAGRAPHS)
        monkeypatch.setattr(verify, "MODEL", "gemini-3.6-flash")
        assert cache_key("q", PARAGRAPHS) != before, (
            "verdicts from one model must never be served for another"
        )
