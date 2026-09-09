"""Tests for grounded answer generation. No Gemini calls; the provider is faked."""

from __future__ import annotations

import json

import pytest

from amrag import answer as ans
from amrag.answer import Answer, AnswerError, append_cache, build_prompt, cache_key, load_cache
from amrag.schema import Paragraph

PARAGRAPHS = [
    Paragraph("Ed Wood (film)", ("A 1994 film directed by Tim Burton.",)),
    Paragraph("Tim Burton", ("He is an American filmmaker.",)),
]


@pytest.fixture
def fake(monkeypatch):
    """Replace the provider; returns the queued responses in turn."""

    def install(*responses):
        calls = []

        def fake_generate(system, prompt):
            calls.append((system, prompt))
            result = responses[min(len(calls) - 1, len(responses) - 1)]
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(ans, "_generate", fake_generate)
        monkeypatch.setattr(ans.time, "sleep", lambda _: None)
        return calls

    return install


class TestPrompt:
    def test_contains_the_question_and_every_document(self):
        prompt = build_prompt("Who directed Ed Wood?", PARAGRAPHS)
        assert "Question: Who directed Ed Wood?" in prompt
        assert "Title: Ed Wood (film)" in prompt
        assert "Title: Tim Burton" in prompt
        assert "A 1994 film directed by Tim Burton." in prompt

    def test_documents_keep_ranking_order(self):
        prompt = build_prompt("q", PARAGRAPHS)
        assert prompt.index("Document 1") < prompt.index("Document 2")
        assert prompt.index("Ed Wood (film)") < prompt.index("Tim Burton")

    def test_reversed_evidence_produces_a_different_prompt(self):
        assert build_prompt("q", PARAGRAPHS) != build_prompt("q", PARAGRAPHS[::-1])

    def test_system_instruction_states_the_grounding_rules(self):
        text = ans.SYSTEM_INSTRUCTION
        assert "ONLY the documents" in text
        assert "outside knowledge" in text
        assert "INSUFFICIENT" in text

    def test_leaks_no_gold_information(self):
        combined = (build_prompt("q", PARAGRAPHS) + ans.SYSTEM_INSTRUCTION).lower()
        for leak in ("gold", "supporting fact", "bridge", "comparison", "hotpot"):
            assert leak not in combined


class TestGeneration:
    def test_returns_a_plain_text_answer(self, fake):
        fake("Tim Burton")
        assert ans.answer("q", PARAGRAPHS) == "Tim Burton"

    def test_strips_surrounding_whitespace(self, fake):
        fake("  Tim Burton \n")
        assert ans.answer("q", PARAGRAPHS) == "Tim Burton"

    def test_yes_answer(self, fake):
        fake("yes")
        assert ans.answer("q", PARAGRAPHS) == "yes"

    def test_no_answer(self, fake):
        fake("no")
        assert ans.answer("q", PARAGRAPHS) == "no"

    def test_insufficient_is_returned_verbatim(self, fake):
        fake("INSUFFICIENT")
        assert ans.answer("q", PARAGRAPHS) == ans.INSUFFICIENT

    def test_sends_the_system_instruction_and_prompt(self, fake):
        calls = fake("Tim Burton")
        ans.answer("Who directed it?", PARAGRAPHS)

        system, prompt = calls[0]
        assert system == ans.SYSTEM_INSTRUCTION
        assert prompt == build_prompt("Who directed it?", PARAGRAPHS)

    def test_does_not_mutate_the_evidence(self, fake):
        fake("Tim Burton")
        before = list(PARAGRAPHS)
        ans.answer("q", PARAGRAPHS)
        assert PARAGRAPHS == before


class TestFailureHandling:
    def test_empty_generation_is_retried_then_fails(self, fake):
        calls = fake("")
        with pytest.raises(AnswerError, match="empty generation"):
            ans.answer("q", PARAGRAPHS)
        assert len(calls) == len(ans.RETRY_DELAYS) + 1

    def test_whitespace_only_generation_is_a_failure(self, fake):
        fake("   \n  ")
        with pytest.raises(AnswerError):
            ans.answer("q", PARAGRAPHS)

    def test_transient_error_is_retried_then_succeeds(self, fake):
        calls = fake(RuntimeError("503 unavailable"), "Tim Burton")
        assert ans.answer("q", PARAGRAPHS) == "Tim Burton"
        assert len(calls) == 2

    def test_permanent_4xx_is_not_retried(self, fake):
        exc = RuntimeError("404 not found")
        exc.code = 404
        calls = fake(exc)
        with pytest.raises(AnswerError, match="404"):
            ans.answer("q", PARAGRAPHS)
        assert len(calls) == 1

    def test_rate_limit_is_retried(self, fake):
        exc = RuntimeError("429 rate limit")
        exc.code = 429
        calls = fake(exc)
        with pytest.raises(AnswerError):
            ans.answer("q", PARAGRAPHS)
        assert len(calls) == len(ans.RETRY_DELAYS) + 1

    def test_failure_never_becomes_an_answer(self, fake):
        fake(RuntimeError("boom"))
        record = ans.generate("q", PARAGRAPHS, qid="q1")

        assert record.text is None
        assert not record.ok
        assert "boom" in record.error

    def test_success_becomes_a_usable_record(self, fake):
        fake("Tim Burton")
        record = ans.generate("q", PARAGRAPHS, qid="q1")

        assert record.text == "Tim Burton"
        assert record.ok
        assert record.qid == "q1"
        assert record.key == cache_key("q", PARAGRAPHS)


class TestCacheKey:
    def test_same_inputs_give_the_same_key(self):
        assert cache_key("q", PARAGRAPHS) == cache_key("q", PARAGRAPHS)

    def test_different_question_changes_the_key(self):
        assert cache_key("q", PARAGRAPHS) != cache_key("other", PARAGRAPHS)

    def test_different_evidence_changes_the_key(self):
        assert cache_key("q", PARAGRAPHS) != cache_key("q", PARAGRAPHS[:1])

    def test_evidence_order_changes_the_key(self):
        assert cache_key("q", PARAGRAPHS) != cache_key("q", PARAGRAPHS[::-1])

    def test_model_is_part_of_the_key(self, monkeypatch):
        before = cache_key("q", PARAGRAPHS)
        monkeypatch.setattr(ans, "MODEL", "other-model")
        assert cache_key("q", PARAGRAPHS) != before

    def test_prompt_version_is_part_of_the_key(self, monkeypatch):
        before = cache_key("q", PARAGRAPHS)
        monkeypatch.setattr(ans, "PROMPT_VERSION", "v2")
        assert cache_key("q", PARAGRAPHS) != before

    def test_identical_evidence_shares_one_key_across_policies(self):
        """Two policies selecting the same evidence must reuse one generation."""
        policy_a = PARAGRAPHS[:2]
        policy_b = list(PARAGRAPHS[:2])
        assert cache_key("q", policy_a) == cache_key("q", policy_b)


class TestCacheIO:
    def test_miss_then_hit(self, tmp_path, fake):
        path = tmp_path / "answers.jsonl"
        assert cache_key("q", PARAGRAPHS) not in load_cache(path), "miss on empty cache"

        fake("Tim Burton")
        append_cache(ans.generate("q", PARAGRAPHS, qid="q1"), path)

        cache = load_cache(path)
        assert cache[cache_key("q", PARAGRAPHS)].text == "Tim Burton"

    def test_miss_when_the_evidence_differs(self, tmp_path, fake):
        path = tmp_path / "answers.jsonl"
        fake("Tim Burton")
        append_cache(ans.generate("q", PARAGRAPHS, qid="q1"), path)

        assert cache_key("q", PARAGRAPHS[:1]) not in load_cache(path)

    def test_missing_file_is_an_empty_cache(self, tmp_path):
        assert load_cache(tmp_path / "absent.jsonl") == {}

    def test_a_later_row_supersedes_an_earlier_failure(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        append_cache(Answer(key="k", qid="q", text=None, error="429"), path)
        append_cache(Answer(key="k", qid="q", text="Tim Burton"), path)

        assert load_cache(path)["k"].text == "Tim Burton"

    def test_failed_records_round_trip_as_unusable(self, tmp_path):
        path = tmp_path / "answers.jsonl"
        append_cache(Answer(key="k", qid="q", text=None, error="429"), path)

        record = load_cache(path)["k"]
        assert record.text is None
        assert not record.ok

    def test_stores_model_and_prompt_version(self, tmp_path, fake):
        path = tmp_path / "answers.jsonl"
        fake("Tim Burton")
        append_cache(ans.generate("q", PARAGRAPHS), path)

        row = json.loads(path.read_text().splitlines()[0])
        assert row["model"] == ans.MODEL
        assert row["prompt_version"] == ans.PROMPT_VERSION


class TestProductionConfig:
    def test_pinned_constants(self):
        assert ans.MODEL == "gemini-3.1-flash-lite"
        assert ans.SEED == 20260908
        assert ans.MAX_OUTPUT_TOKENS == 64
        assert ans.PROMPT_VERSION == "v1"

    def test_generate_sends_the_approved_config(self, monkeypatch):
        captured = {}

        class FakeModels:
            def generate_content(self, *, model, contents, config):
                captured.update(model=model, contents=contents, config=config)
                return type("R", (), {"text": "Tim Burton"})()

        monkeypatch.setattr(ans, "_client", type("C", (), {"models": FakeModels()})())
        ans._generate("SYS", "PROMPT")
        cfg = captured["config"]

        assert captured["model"] == "gemini-3.1-flash-lite"
        assert cfg.temperature == 0
        assert cfg.seed == ans.SEED
        assert cfg.max_output_tokens == 64
        assert cfg.response_schema is None, "plain text, not JSON"
        assert cfg.response_mime_type is None
        assert cfg.thinking_config is None
