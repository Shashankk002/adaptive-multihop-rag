"""Tests for the HotpotQA metrics. Hard-coded examples only — no dataset access."""

from __future__ import annotations

import pytest

from scrag.metrics import answer_f1, exact_match, normalize_answer, supporting_fact_prf
from scrag.schema import SupportingFact


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Tim Burton", "tim burton"),
            ("TIM BURTON", "tim burton"),
            ("Tim Burton.", "tim burton"),
            ("(Tim) Burton!", "tim burton"),
            ("The Beatles", "beatles"),
            ("A Clockwork Orange", "clockwork orange"),
            ("an apple", "apple"),
            ("  extra   spaces  ", "extra spaces"),
            ("", ""),
        ],
    )
    def test_normalization_cases(self, raw, expected):
        assert normalize_answer(raw) == expected

    def test_articles_are_removed_only_as_whole_words(self):
        # "Thermal" must not become "rmal" by stripping a leading "the".
        assert normalize_answer("Thermal") == "thermal"
        assert normalize_answer("Canada") == "canada"

    def test_punctuation_is_removed_before_articles(self):
        # Hyphen removal joins the words, so "the" is no longer a standalone article.
        assert normalize_answer("the-film") == "thefilm"


class TestExactMatch:
    def test_identical_strings_match(self):
        assert exact_match("Tim Burton", "Tim Burton") == 1.0

    def test_case_and_punctuation_differences_still_match(self):
        assert exact_match("tim burton.", "Tim Burton") == 1.0

    def test_article_differences_still_match(self):
        assert exact_match("The Beatles", "Beatles") == 1.0

    def test_different_answers_do_not_match(self):
        assert exact_match("Tim Burton", "Ed Wood") == 0.0

    def test_partial_answer_is_not_an_exact_match(self):
        assert exact_match("Burton", "Tim Burton") == 0.0

    def test_empty_prediction(self):
        assert exact_match("", "Tim Burton") == 0.0

    def test_both_empty_match(self):
        assert exact_match("", "") == 1.0


class TestAnswerF1:
    def test_identical_answers_score_one(self):
        assert answer_f1("Tim Burton", "Tim Burton") == 1.0

    def test_no_overlap_scores_zero(self):
        assert answer_f1("Ed Wood", "Tim Burton") == 0.0

    def test_partial_overlap(self):
        # pred 1 token, gold 2 tokens, 1 shared -> P=1.0, R=0.5, F1=2/3
        assert answer_f1("Burton", "Tim Burton") == pytest.approx(2 / 3)

    def test_extra_tokens_reduce_precision(self):
        # pred 3 tokens, gold 2, 2 shared -> P=2/3, R=1.0, F1=0.8
        assert answer_f1("director Tim Burton", "Tim Burton") == pytest.approx(0.8)

    def test_repeated_tokens_are_counted_by_multiplicity(self):
        # "new" twice in pred, once in gold -> overlap 2 of 3 pred tokens, not 2 of 2.
        assert answer_f1("new new york", "new york") == pytest.approx(0.8)

    def test_repeated_tokens_in_gold(self):
        # pred 2 tokens, gold 3 ("new new york"), overlap 2 -> P=1.0, R=2/3, F1=0.8
        assert answer_f1("new york", "new new york") == pytest.approx(0.8)

    def test_empty_prediction_scores_zero(self):
        assert answer_f1("", "Tim Burton") == 0.0

    def test_empty_gold_scores_zero(self):
        assert answer_f1("Tim Burton", "") == 0.0

    def test_both_empty_scores_zero(self):
        # No overlapping tokens exist, so F1 is 0 — not 1. This matches the official
        # script, and gold answers are never empty in HotpotQA anyway.
        assert answer_f1("", "") == 0.0

    def test_normalization_applies_before_scoring(self):
        assert answer_f1("The BEATLES!", "beatles") == 1.0


class TestYesNoAnswers:
    """HotpotQA scores yes/no answers all-or-nothing."""

    def test_correct_yes(self):
        assert answer_f1("yes", "yes") == 1.0

    def test_yes_against_no_scores_zero(self):
        assert answer_f1("yes", "no") == 0.0

    def test_yes_earns_no_partial_credit_against_a_longer_gold(self):
        # Without the guard this would score partial F1 for a token overlap.
        assert answer_f1("yes", "yes he did") == 0.0

    def test_longer_prediction_against_yes_gold_scores_zero(self):
        assert answer_f1("yes he did", "yes") == 0.0


class TestSupportingFacts:
    def test_perfect_prediction(self):
        gold = [("Tim Burton", 1), ("Ed Wood (film)", 0)]
        assert supporting_fact_prf(gold, gold) == (1.0, 1.0, 1.0)

    def test_no_overlap(self):
        result = supporting_fact_prf([("Other", 0)], [("Tim Burton", 1)])
        assert result == (0.0, 0.0, 0.0)

    def test_partial_overlap(self):
        # 1 of 2 predicted correct, 1 of 3 gold found -> P=0.5, R=1/3, F1=0.4
        predicted = [("Tim Burton", 1), ("Wrong", 0)]
        gold = [("Tim Burton", 1), ("Ed Wood (film)", 0), ("Ed Wood (film)", 1)]
        result = supporting_fact_prf(predicted, gold)
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(1 / 3)
        assert result.f1 == pytest.approx(0.4)

    def test_over_prediction_keeps_recall_high_and_drops_precision(self):
        predicted = [("A", 0), ("B", 0), ("C", 0), ("D", 0)]
        gold = [("A", 0), ("B", 0)]
        result = supporting_fact_prf(predicted, gold)
        assert result.precision == pytest.approx(0.5)
        assert result.recall == 1.0
        assert result.f1 == pytest.approx(2 / 3)

    def test_sentence_index_must_match_not_just_the_title(self):
        result = supporting_fact_prf([("Tim Burton", 0)], [("Tim Burton", 1)])
        assert result == (0.0, 0.0, 0.0)

    def test_duplicate_predictions_are_collapsed(self):
        predicted = [("Tim Burton", 1), ("Tim Burton", 1)]
        gold = [("Tim Burton", 1)]
        assert supporting_fact_prf(predicted, gold) == (1.0, 1.0, 1.0)

    def test_empty_prediction(self):
        result = supporting_fact_prf([], [("Tim Burton", 1)])
        assert result == (0.0, 0.0, 0.0)

    def test_empty_gold(self):
        result = supporting_fact_prf([("Tim Burton", 1)], [])
        assert result == (0.0, 0.0, 0.0)

    def test_both_empty(self):
        assert supporting_fact_prf([], []) == (0.0, 0.0, 0.0)

    def test_accepts_supporting_fact_objects_from_the_loader(self):
        predicted = [SupportingFact("Tim Burton", 1)]
        gold = [("Tim Burton", 1)]
        assert supporting_fact_prf(predicted, gold) == (1.0, 1.0, 1.0)
