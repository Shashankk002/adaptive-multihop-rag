"""Tests for HotpotQA parsing. Fixtures only — never the downloaded dataset."""

from __future__ import annotations

import copy
import json

import pytest

from amrag.data_loader import (
    HotpotQAFormatError,
    load_hotpotqa,
    load_hotpotqa_with_report,
    parse_hotpotqa,
)
from amrag.schema import QuestionType


class TestFieldPreservation:
    def test_preserves_every_field_of_a_bridge_example(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])

        assert example.qid == "bridge-0001"
        assert example.question.startswith("Who directed the film")
        assert example.answer == "Tim Burton"
        assert example.question_type is QuestionType.BRIDGE
        assert example.difficulty == "hard"
        assert len(example.paragraphs) == 3
        assert len(example.supporting_facts) == 2

    def test_preserves_candidate_paragraphs_in_order_with_sentences_split(
        self, bridge_record
    ):
        (example,), _ = parse_hotpotqa([bridge_record])

        titles = [p.title for p in example.paragraphs]
        assert titles == ["Ed Wood (film)", "Tim Burton", "Woodson, Arkansas"]
        assert example.paragraphs[0].sentences == (
            "Ed Wood is a 1994 American biographical film.",
            " It stars Johnny Depp as the filmmaker Ed Wood.",
        )

    def test_paragraph_text_rejoins_sentences_losslessly(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])

        original = "".join(bridge_record["context"][0][1])
        assert example.paragraphs[0].text == original

    def test_supporting_facts_keep_title_and_sentence_index(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])

        assert [(f.title, f.sent_id) for f in example.supporting_facts] == [
            ("Ed Wood (film)", 0),
            ("Tim Burton", 1),
        ]

    def test_comparison_type_is_parsed(self, comparison_record):
        (example,), _ = parse_hotpotqa([comparison_record])
        assert example.question_type is QuestionType.COMPARISON

    def test_unknown_type_falls_back_to_other(self, bridge_record):
        bridge_record["type"] = "some-future-label"
        (example,), _ = parse_hotpotqa([bridge_record])
        assert example.question_type is QuestionType.OTHER


class TestDerivedAccessors:
    def test_gold_titles_are_the_supporting_fact_titles(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])
        assert example.gold_titles == {"Ed Wood (film)", "Tim Burton"}
        assert "Woodson, Arkansas" not in example.gold_titles

    def test_supporting_sentences_resolve_to_the_labelled_text(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])
        assert example.supporting_sentences() == (
            "Ed Wood is a 1994 American biographical film.",
            " He directed Ed Wood in 1994.",
        )

    def test_paragraph_lookup_by_title_returns_none_when_absent(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])
        assert example.paragraph_by_title("Tim Burton") is not None
        assert example.paragraph_by_title("Nonexistent") is None

    def test_labelled_flag(self, bridge_record):
        (example,), _ = parse_hotpotqa([bridge_record])
        assert example.is_labelled

        unlabelled = copy.deepcopy(bridge_record)
        del unlabelled["answer"]
        del unlabelled["supporting_facts"]
        (example,), _ = parse_hotpotqa([unlabelled])
        assert not example.is_labelled


class TestUnlabelledSplit:
    def test_missing_answer_and_facts_are_tolerated(self, bridge_record):
        del bridge_record["answer"]
        del bridge_record["supporting_facts"]

        (example,), report = parse_hotpotqa([bridge_record])

        assert example.answer is None
        assert example.supporting_facts == ()
        assert len(example.paragraphs) == 3
        assert report.unlabelled == 1


class TestDataAnomalies:
    """The real dataset contains supporting facts that do not resolve. The loader
    records them; it does not crash, and it does not hide them."""

    def test_out_of_range_sentence_index_is_reported_not_dropped(self, bridge_record):
        bridge_record["supporting_facts"] = [["Ed Wood (film)", 99]]

        (example,), report = parse_hotpotqa([bridge_record])

        assert report.unresolved_facts == [("bridge-0001", "Ed Wood (film)", 99)]
        assert len(example.supporting_facts) == 1, "the fact is kept, not silently removed"
        assert example.supporting_sentences() == ()

    def test_strict_mode_raises_on_an_unresolved_fact(self, bridge_record):
        bridge_record["supporting_facts"] = [["Ed Wood (film)", 99]]

        with pytest.raises(HotpotQAFormatError, match="does not resolve"):
            parse_hotpotqa([bridge_record], strict=True)

    def test_clean_data_produces_no_warnings_in_strict_mode(self, raw_dataset):
        examples, report = parse_hotpotqa(raw_dataset, strict=True)
        assert len(examples) == 2
        assert report.n_unresolved_facts == 0


class TestMalformedInput:
    def test_non_list_input_is_rejected(self):
        with pytest.raises(HotpotQAFormatError, match="expected a JSON list"):
            parse_hotpotqa({"_id": "x"})

    def test_missing_id_is_rejected(self, bridge_record):
        del bridge_record["_id"]
        with pytest.raises(HotpotQAFormatError, match="_id"):
            parse_hotpotqa([bridge_record])

    def test_missing_question_is_rejected(self, bridge_record):
        del bridge_record["question"]
        with pytest.raises(HotpotQAFormatError, match="question"):
            parse_hotpotqa([bridge_record])

    def test_missing_context_is_rejected(self, bridge_record):
        del bridge_record["context"]
        with pytest.raises(HotpotQAFormatError, match="missing 'context'"):
            parse_hotpotqa([bridge_record])

    def test_malformed_context_entry_is_rejected(self, bridge_record):
        bridge_record["context"] = [["title only"]]
        with pytest.raises(HotpotQAFormatError, match=r"\[title, sentences\]"):
            parse_hotpotqa([bridge_record])

    def test_non_string_sentences_are_rejected(self, bridge_record):
        bridge_record["context"] = [["Title", ["ok", 42]]]
        with pytest.raises(HotpotQAFormatError, match="list of strings"):
            parse_hotpotqa([bridge_record])

    def test_malformed_supporting_fact_is_rejected(self, bridge_record):
        bridge_record["supporting_facts"] = [["Ed Wood (film)", "zero"]]
        with pytest.raises(HotpotQAFormatError, match=r"\[str, int\]"):
            parse_hotpotqa([bridge_record])


class TestReport:
    def test_counts_across_a_small_dataset(self, raw_dataset):
        _, report = parse_hotpotqa(raw_dataset)

        assert report.n_examples == 2
        assert report.n_paragraphs == 6
        assert report.n_supporting_facts == 4
        assert report.type_counts == {"bridge": 1, "comparison": 1}
        assert report.unlabelled == 0

    def test_summary_is_a_readable_line(self, raw_dataset):
        _, report = parse_hotpotqa(raw_dataset)
        assert "2 examples" in report.summary()


class TestFileLoading:
    def test_round_trip_through_a_json_file(self, tmp_path, raw_dataset):
        path = tmp_path / "hotpot_fixture.json"
        path.write_text(json.dumps(raw_dataset), encoding="utf-8")

        examples = load_hotpotqa(path)

        assert [e.qid for e in examples] == ["bridge-0001", "comparison-0001"]

    def test_limit_reads_a_prefix_only(self, tmp_path, raw_dataset):
        path = tmp_path / "hotpot_fixture.json"
        path.write_text(json.dumps(raw_dataset), encoding="utf-8")

        examples, report = load_hotpotqa_with_report(path, limit=1)

        assert len(examples) == 1
        assert report.n_examples == 1

    def test_unicode_survives_the_round_trip(self, tmp_path, bridge_record):
        bridge_record["context"][0][1][0] = "Ed Wood – a 1994 film."
        path = tmp_path / "unicode.json"
        path.write_text(json.dumps(bridge_record and [bridge_record]), encoding="utf-8")

        examples = load_hotpotqa(path)

        assert "–" in examples[0].paragraphs[0].sentences[0]
