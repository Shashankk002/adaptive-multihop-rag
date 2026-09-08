"""Tests for BM25 retrieval. Hand-built fixtures only — no dataset access."""

from __future__ import annotations

import math

import pytest

from scrag.metrics import all_gold_at_k, first_missed_gold_rank, recall_at_k
from scrag.retrieval import B, K1, bm25_retrieve, bm25_scores, indexed_text, tokenize
from scrag.schema import Example, Paragraph


def make_example(*paragraphs: tuple[str, list[str]]) -> Example:
    return Example(
        qid="q1",
        question="unused",
        paragraphs=tuple(Paragraph(t, tuple(s)) for t, s in paragraphs),
    )


class TestTokenizer:
    def test_lowercases_and_splits_on_non_word_characters(self):
        assert tokenize("Tim Burton's Ed-Wood (1994)!") == [
            "tim", "burton", "s", "ed", "wood", "1994",
        ]

    def test_empty_string(self):
        assert tokenize("") == []


class TestIndexedText:
    def test_title_is_included_with_the_body(self):
        paragraph = Paragraph("Tim Burton", ("He directed Ed Wood.",))
        assert indexed_text(paragraph) == "Tim Burton He directed Ed Wood."


class TestHandCalculatedScore:
    """One score computed from the formula by hand, to prove this is really BM25."""

    def test_matches_a_score_worked_out_by_hand(self):
        # Three documents, all the same length (2 tokens), so avg_len = 2 and the
        # length normalisation factor is exactly 1 for each.
        documents = ["alpha beta", "alpha gamma", "delta epsilon"]

        # Query "beta": df = 1, n = 3.
        #   idf         = log(1 + (3 - 1 + 0.5) / (1 + 0.5)) = log(1 + 2.5/1.5)
        #   length_norm = 1 - 0.75 + 0.75 * (2 / 2) = 1
        #   tf          = 1
        #   score       = idf * 1 * (K1 + 1) / (1 + K1 * 1)
        idf = math.log(1 + 2.5 / 1.5)
        expected = idf * (K1 + 1) / (1 + K1)

        scores = bm25_scores("beta", documents)

        assert scores[0] == pytest.approx(expected)
        assert scores[1] == 0.0
        assert scores[2] == 0.0

    def test_idf_stays_positive_for_a_term_in_every_document(self):
        # The textbook IDF would be negative here; ours must not be.
        scores = bm25_scores("alpha", ["alpha", "alpha", "alpha"])
        assert all(s > 0 for s in scores)


class TestScoringProperties:
    def test_term_frequency_saturates(self):
        # Four occurrences score less than four times one occurrence.
        one = bm25_scores("alpha", ["alpha filler filler filler"])[0]
        four = bm25_scores("alpha", ["alpha alpha alpha alpha"])[0]
        assert four > one
        assert four < 4 * one

    def test_shorter_document_scores_higher_at_equal_term_frequency(self):
        scores = bm25_scores("alpha", ["alpha", "alpha padding padding padding"])
        assert scores[0] > scores[1]

    def test_length_normalisation_is_actually_applied(self):
        assert B > 0, "a zero b would disable length normalisation"

    def test_rarer_term_outweighs_a_common_one(self):
        documents = ["common rare", "common x", "common y", "common z"]
        rare_only = bm25_scores("rare", documents)
        common_only = bm25_scores("common", documents)
        assert rare_only[0] > common_only[0]


class TestRetrieve:
    @pytest.fixture
    def example(self):
        return make_example(
            ("Tim Burton", ["He directed Ed Wood in 1994."]),
            ("Ed Wood (film)", ["A 1994 biographical film starring Johnny Depp."]),
            ("Woodson, Arkansas", ["A census-designated place in Arkansas."]),
        )

    def test_ranks_the_matching_paragraph_first(self, example):
        results = bm25_retrieve(example, "Johnny Depp", k=3)
        assert results[0].paragraph.title == "Ed Wood (film)"

    def test_scores_are_non_increasing(self, example):
        scores = [r.score for r in bm25_retrieve(example, "1994 film", k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_k_limits_the_number_of_results(self, example):
        assert len(bm25_retrieve(example, "1994", k=1)) == 1
        assert len(bm25_retrieve(example, "1994", k=2)) == 2

    def test_k_larger_than_the_candidate_set_returns_everything(self, example):
        assert len(bm25_retrieve(example, "1994", k=99)) == 3

    def test_k_of_zero_returns_nothing(self, example):
        assert bm25_retrieve(example, "1994", k=0) == []

    def test_out_of_vocabulary_query_scores_zero_without_failing(self, example):
        results = bm25_retrieve(example, "zzzz qqqq", k=3)
        assert len(results) == 3
        assert all(r.score == 0.0 for r in results)

    def test_empty_query_scores_zero_without_failing(self, example):
        assert all(r.score == 0.0 for r in bm25_retrieve(example, "", k=3))

    def test_ties_keep_candidate_order(self, example):
        # Nothing matches, so every score is 0 and the original order must survive.
        titles = [r.paragraph.title for r in bm25_retrieve(example, "zzzz", k=3)]
        assert titles == ["Tim Burton", "Ed Wood (film)", "Woodson, Arkansas"]

    def test_ranking_is_deterministic_across_calls(self, example):
        first = [r.paragraph.title for r in bm25_retrieve(example, "1994 film", k=3)]
        second = [r.paragraph.title for r in bm25_retrieve(example, "1994 film", k=3)]
        assert first == second

    def test_title_match_alone_is_enough_to_retrieve(self, example):
        # "Woodson" appears only in a title, not in any body text.
        results = bm25_retrieve(example, "Woodson", k=1)
        assert results[0].paragraph.title == "Woodson, Arkansas"


class TestRetrievalMetrics:
    RETRIEVED = ["A", "B", "C", "D"]
    GOLD = {"A", "C"}

    def test_recall_at_k(self):
        assert recall_at_k(self.RETRIEVED, self.GOLD, 1) == 0.5
        assert recall_at_k(self.RETRIEVED, self.GOLD, 2) == 0.5
        assert recall_at_k(self.RETRIEVED, self.GOLD, 3) == 1.0
        assert recall_at_k(self.RETRIEVED, self.GOLD, 4) == 1.0

    def test_all_gold_at_k(self):
        assert all_gold_at_k(self.RETRIEVED, self.GOLD, 2) == 0.0
        assert all_gold_at_k(self.RETRIEVED, self.GOLD, 3) == 1.0

    def test_all_gold_is_stricter_than_recall(self):
        # Half the gold retrieved still means the question is unanswerable.
        assert recall_at_k(self.RETRIEVED, self.GOLD, 2) == 0.5
        assert all_gold_at_k(self.RETRIEVED, self.GOLD, 2) == 0.0

    def test_first_missed_gold_rank(self):
        assert first_missed_gold_rank(self.RETRIEVED, self.GOLD, 2) == 3
        assert first_missed_gold_rank(self.RETRIEVED, self.GOLD, 3) is None

    def test_first_missed_gold_rank_when_gold_is_never_retrieved(self):
        assert first_missed_gold_rank(["X", "Y"], {"A"}, 1) is None

    def test_empty_gold_scores_zero(self):
        assert recall_at_k(self.RETRIEVED, set(), 2) == 0.0
        assert all_gold_at_k(self.RETRIEVED, set(), 2) == 0.0

    def test_nothing_retrieved(self):
        assert recall_at_k([], self.GOLD, 2) == 0.0
        assert all_gold_at_k([], self.GOLD, 2) == 0.0
