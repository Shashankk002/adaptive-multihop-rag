"""In-memory HotpotQA fixtures.

These are hand-written and deliberately tiny. No test in this suite reads the
downloaded dataset — the suite must pass on a clean checkout with an empty data/.
"""

from __future__ import annotations

import copy

import pytest

BRIDGE_RECORD = {
    "_id": "bridge-0001",
    "question": "Who directed the film that starred Johnny Depp as Ed Wood?",
    "answer": "Tim Burton",
    "type": "bridge",
    "level": "hard",
    "supporting_facts": [["Ed Wood (film)", 0], ["Tim Burton", 1]],
    "context": [
        [
            "Ed Wood (film)",
            [
                "Ed Wood is a 1994 American biographical film.",
                " It stars Johnny Depp as the filmmaker Ed Wood.",
            ],
        ],
        [
            "Tim Burton",
            [
                "Timothy Walter Burton is an American filmmaker.",
                " He directed Ed Wood in 1994.",
            ],
        ],
        ["Woodson, Arkansas", ["Woodson is a census-designated place in Arkansas."]],
    ],
}

COMPARISON_RECORD = {
    "_id": "comparison-0001",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
    "context": [
        ["Scott Derrickson", ["Scott Derrickson is an American director."]],
        ["Ed Wood", ["Edward Davis Wood Jr. was an American filmmaker."]],
        ["Tyler Bates", ["Tyler Bates is an American musician."]],
    ],
}


@pytest.fixture
def bridge_record() -> dict:
    return copy.deepcopy(BRIDGE_RECORD)


@pytest.fixture
def comparison_record() -> dict:
    return copy.deepcopy(COMPARISON_RECORD)


@pytest.fixture
def raw_dataset(bridge_record, comparison_record) -> list[dict]:
    return [bridge_record, comparison_record]
