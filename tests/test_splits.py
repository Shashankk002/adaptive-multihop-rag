"""Tests for the Phase 7 verifier evaluation split.

Selection logic is tested against synthetic examples, so the suite needs no dataset.
The frozen artifact is checked against the committed split files only — those are id
lists, not data. The one check that needs question types skips without the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scrag import splits
from scrag.schema import Example, Paragraph, QuestionType
from scrag.splits import (
    SplitError,
    VERIFY_TUNE_PER_TYPE,
    VERIFY_TUNE_SIZE,
    load_split_ids,
    read_split,
    select_verify_tune_ids,
    write_verify_tune_split,
)

SPLITS_DIR = Path("data/splits")


def make_examples(n_bridge: int, n_comparison: int) -> list[Example]:
    """A stand-in TUNE with the real one's 805/195 shape."""
    made = []
    for kind, count in ((QuestionType.BRIDGE, n_bridge), (QuestionType.COMPARISON, n_comparison)):
        for i in range(count):
            made.append(
                Example(
                    qid=f"{kind.value}-{i:04d}",
                    question="q",
                    paragraphs=(Paragraph("T", ("body",)),),
                    question_type=kind,
                )
            )
    return made


@pytest.fixture
def tune():
    return make_examples(805, 195)


class TestSelectionLogic:
    def test_selects_exactly_500(self, tune):
        assert len(select_verify_tune_ids(tune, [e.qid for e in tune])) == VERIFY_TUNE_SIZE

    def test_quotas_are_403_bridge_and_97_comparison(self, tune):
        by_id = {e.qid: e for e in tune}
        chosen = select_verify_tune_ids(tune, [e.qid for e in tune])

        counts = {t: 0 for t in VERIFY_TUNE_PER_TYPE}
        for qid in chosen:
            counts[by_id[qid].question_type] += 1

        assert counts[QuestionType.BRIDGE] == 403
        assert counts[QuestionType.COMPARISON] == 97

    def test_result_is_a_subset_of_the_pool(self, tune):
        pool = {e.qid for e in tune}
        assert set(select_verify_tune_ids(tune, pool)) <= pool

    def test_ids_are_unique_and_sorted(self, tune):
        chosen = select_verify_tune_ids(tune, [e.qid for e in tune])
        assert len(set(chosen)) == len(chosen)
        assert chosen == sorted(chosen)

    def test_ignores_examples_outside_the_pool(self, tune):
        """Questions that are not in TUNE must never be selected."""
        pool = [e.qid for e in tune]
        outsiders = make_examples(50, 50)
        for e in outsiders:
            e.__dict__["qid"] = "outsider-" + e.qid

        chosen = select_verify_tune_ids(tune + outsiders, pool)

        assert not any(qid.startswith("outsider-") for qid in chosen)

    def test_selection_is_deterministic(self, tune):
        pool = [e.qid for e in tune]
        assert select_verify_tune_ids(tune, pool) == select_verify_tune_ids(tune, pool)

    def test_input_order_does_not_matter(self, tune):
        """Candidates are sorted before sampling, so file order cannot leak in."""
        pool = [e.qid for e in tune]
        assert select_verify_tune_ids(tune, pool) == select_verify_tune_ids(tune[::-1], pool)

    def test_a_different_seed_gives_a_different_selection(self, tune):
        pool = [e.qid for e in tune]
        base = select_verify_tune_ids(tune, pool)
        other = select_verify_tune_ids(tune, pool, seed=splits.SPLIT_SEED + 100)

        assert other != base
        assert len(other) == VERIFY_TUNE_SIZE, "still a valid split, just a different draw"

    def test_too_few_candidates_is_an_error(self):
        small = make_examples(10, 10)
        with pytest.raises(SplitError, match="cannot draw"):
            select_verify_tune_ids(small, [e.qid for e in small])


class TestFrozenArtifact:
    """Properties of the committed split file itself."""

    @pytest.fixture
    def frozen(self):
        path = SPLITS_DIR / "verify_tune_ids.json"
        if not path.exists():
            pytest.skip("frozen split not present")
        return read_split(path)

    def test_holds_exactly_500_unique_ids(self, frozen):
        assert len(frozen.ids) == VERIFY_TUNE_SIZE
        assert len(set(frozen.ids)) == VERIFY_TUNE_SIZE

    def test_seed_is_recorded(self, frozen):
        assert frozen.seed == splits.SPLIT_SEED + 3

    def test_is_a_subset_of_tune(self, frozen):
        assert set(frozen.ids) <= load_split_ids("tune")

    def test_has_no_overlap_with_test(self, frozen):
        assert not (set(frozen.ids) & load_split_ids("test"))

    def test_load_split_ids_returns_the_frozen_ids(self, frozen):
        loaded = load_split_ids("verify_tune")
        assert loaded == frozenset(frozen.ids)
        assert len(loaded) == VERIFY_TUNE_SIZE

    def test_ids_look_like_hotpotqa_ids(self, frozen):
        assert all(len(qid) == 24 and qid.isalnum() for qid in frozen.ids)

    def test_stratification_matches_the_quotas(self, frozen):
        """Needs the dataset for question types, so it skips on a clean checkout."""
        dataset = Path("data/raw/hotpot_dev_distractor_v1.json")
        if not dataset.exists():
            pytest.skip("dataset not downloaded")
        from collections import Counter

        from scrag.data_loader import load_hotpotqa

        by_id = {e.qid: e for e in load_hotpotqa(dataset)}
        counts = Counter(by_id[qid].question_type.value for qid in frozen.ids)

        assert counts["bridge"] == 403
        assert counts["comparison"] == 97


class TestOverwriteGuard:
    def test_refuses_to_overwrite_without_force(self, tmp_path, tune):
        (tmp_path / "tune_ids.json").write_text(
            json.dumps({"name": "tune", "seed": 1, "source": "s", "description": "d",
                        "n_ids": len(tune), "ids": [e.qid for e in tune]}),
            encoding="utf-8",
        )
        write_verify_tune_split(tune, splits_dir=tmp_path)

        with pytest.raises(SplitError, match="refusing to overwrite"):
            write_verify_tune_split(tune, splits_dir=tmp_path)

    def test_force_allows_a_deliberate_rewrite(self, tmp_path, tune):
        (tmp_path / "tune_ids.json").write_text(
            json.dumps({"name": "tune", "seed": 1, "source": "s", "description": "d",
                        "n_ids": len(tune), "ids": [e.qid for e in tune]}),
            encoding="utf-8",
        )
        first = write_verify_tune_split(tune, splits_dir=tmp_path)
        again = write_verify_tune_split(tune, splits_dir=tmp_path, force=True)

        assert again.ids == first.ids, "same seed, same draw"
