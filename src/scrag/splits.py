"""Frozen evaluation splits.

This module assigns every question in the dataset to exactly one of:

    TEST  — 150 stratified ids, frozen. Never tuned against.
    DEV   — everything else. Defined by exclusion, so it is never materialised as a
            second copy of the data.
    SMOKE — 100 ids drawn from DEV, for fast iteration.

Only question ids are stored. The dataset itself is never copied, moved, or modified;
splits are applied by filtering examples at load time.

Selection is deterministic: candidate ids are sorted before sampling, so the result
depends on the seed and the set of ids, not on the order they appear in the file.

Generate or verify from the command line:

    .venv/bin/python -m scrag.splits generate
    .venv/bin/python -m scrag.splits verify
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scrag.data_loader import load_hotpotqa
from scrag.schema import Example, QuestionType

# --------------------------------------------------------------------------------
# Frozen parameters. Changing any of these invalidates every result ever reported
# against these splits, so they are constants, not arguments.
# --------------------------------------------------------------------------------

SPLIT_SEED = 20260908
"""Fixed seed for split generation (the date the splits were frozen)."""

TEST_SIZE = 150
TEST_PER_TYPE = {QuestionType.BRIDGE: 75, QuestionType.COMPARISON: 75}
SMOKE_SIZE = 100

DEFAULT_DATASET = Path("data/raw/hotpot_dev_distractor_v1.json")
DEFAULT_SPLITS_DIR = Path("data/splits")
TEST_FILENAME = "test_ids.json"
SMOKE_FILENAME = "smoke_ids.json"


class SplitError(RuntimeError):
    """A split could not be generated, or a generated split failed verification."""


@dataclass(frozen=True)
class SplitFile:
    """The on-disk form of a split: metadata plus a sorted list of ids."""

    name: str
    seed: int
    source: str
    description: str
    ids: tuple[str, ...]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "source": self.source,
            "description": self.description,
            "n_ids": len(self.ids),
            "ids": list(self.ids),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "SplitFile":
        return cls(
            name=payload["name"],
            seed=payload["seed"],
            source=payload["source"],
            description=payload["description"],
            ids=tuple(payload["ids"]),
        )


# --------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------


def _sorted_ids(examples: Iterable[Example]) -> list[str]:
    """Ids in a canonical order, so sampling does not depend on file order."""
    return sorted(e.qid for e in examples)


def select_test_ids(examples: Sequence[Example], *, seed: int = SPLIT_SEED) -> list[str]:
    """Stratified sample of TEST ids: a fixed quota per question type.

    Stratifying keeps the bridge/comparison balance from drifting with the sample, so
    per-type results on TEST are comparable rather than dominated by bridge questions
    (which are ~80% of the dataset).
    """
    rng = random.Random(seed)
    chosen: list[str] = []
    for question_type, quota in sorted(TEST_PER_TYPE.items(), key=lambda kv: kv[0].value):
        pool = _sorted_ids(e for e in examples if e.question_type is question_type)
        if len(pool) < quota:
            raise SplitError(
                f"cannot draw {quota} {question_type.value} ids: only {len(pool)} exist"
            )
        chosen.extend(rng.sample(pool, quota))
    return sorted(chosen)


def select_smoke_ids(
    examples: Sequence[Example],
    test_ids: Iterable[str],
    *,
    seed: int = SPLIT_SEED,
) -> list[str]:
    """Sample SMOKE ids from DEV only.

    Drawn proportionally to DEV's own type composition, so a smoke run is a miniature
    of DEV rather than a lucky or unlucky slice of it. A separate RNG stream keeps this
    selection independent of how many draws TEST happened to make.
    """
    excluded = set(test_ids)
    dev = [e for e in examples if e.qid not in excluded]
    if len(dev) < SMOKE_SIZE:
        raise SplitError(f"DEV holds only {len(dev)} examples; need {SMOKE_SIZE}")

    rng = random.Random(seed + 1)
    pools = {
        question_type: _sorted_ids(e for e in dev if e.question_type is question_type)
        for question_type in sorted({e.question_type for e in dev}, key=lambda t: t.value)
    }

    # Largest-remainder apportionment, so the quotas sum to exactly SMOKE_SIZE.
    exact = {t: len(pool) * SMOKE_SIZE / len(dev) for t, pool in pools.items()}
    quota = {t: int(v) for t, v in exact.items()}
    for question_type, _ in sorted(
        exact.items(), key=lambda kv: (-(kv[1] - int(kv[1])), kv[0].value)
    ):
        if sum(quota.values()) >= SMOKE_SIZE:
            break
        quota[question_type] += 1

    chosen: list[str] = []
    for question_type in sorted(pools, key=lambda t: t.value):
        chosen.extend(rng.sample(pools[question_type], quota[question_type]))
    return sorted(chosen)


def dev_ids(examples: Sequence[Example], test_ids: Iterable[str]) -> list[str]:
    """DEV is defined by exclusion — every id that is not in TEST."""
    excluded = set(test_ids)
    return sorted(e.qid for e in examples if e.qid not in excluded)


# --------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------


def write_splits(
    examples: Sequence[Example],
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    source: Path = DEFAULT_DATASET,
    seed: int = SPLIT_SEED,
    force: bool = False,
) -> tuple[SplitFile, SplitFile]:
    """Generate and write both split files.

    Refuses to overwrite existing splits without `force`: once results have been
    reported against a frozen TEST set, silently regenerating it would invalidate them.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    test_path = splits_dir / TEST_FILENAME
    smoke_path = splits_dir / SMOKE_FILENAME

    existing = [p for p in (test_path, smoke_path) if p.exists()]
    if existing and not force:
        raise SplitError(
            "refusing to overwrite frozen splits: "
            + ", ".join(str(p) for p in existing)
            + " (pass --force if you really mean to re-freeze them)"
        )

    test = SplitFile(
        name="test",
        seed=seed,
        source=str(source),
        description=(
            f"Frozen TEST split: {TEST_SIZE} ids, stratified "
            + ", ".join(f"{n} {t.value}" for t, n in sorted(TEST_PER_TYPE.items(), key=lambda kv: kv[0].value))
            + ". Never tune against this split."
        ),
        ids=tuple(select_test_ids(examples, seed=seed)),
    )
    smoke = SplitFile(
        name="smoke",
        seed=seed + 1,
        source=str(source),
        description=(
            f"SMOKE subset: {SMOKE_SIZE} ids drawn from DEV (disjoint from TEST), "
            "proportional to DEV's type composition. For fast iteration only."
        ),
        ids=tuple(select_smoke_ids(examples, test.ids, seed=seed)),
    )

    for path, split in ((test_path, test), (smoke_path, smoke)):
        path.write_text(json.dumps(split.to_json(), indent=2) + "\n", encoding="utf-8")
    return test, smoke


def read_split(path: Path) -> SplitFile:
    return SplitFile.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def load_split_ids(name: str, *, splits_dir: Path = DEFAULT_SPLITS_DIR) -> frozenset[str]:
    """Ids of a named split ('test' or 'smoke'), for filtering examples at load time."""
    filename = {"test": TEST_FILENAME, "smoke": SMOKE_FILENAME}[name]
    return frozenset(read_split(splits_dir / filename).ids)


# --------------------------------------------------------------------------------
# Acceptance check
# --------------------------------------------------------------------------------


@dataclass
class VerificationResult:
    checks: list[tuple[str, bool, str]]

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def render(self) -> str:
        lines = []
        for label, passed, detail in self.checks:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        lines.append(f"\n  {'ALL CHECKS PASSED' if self.ok else 'VERIFICATION FAILED'}")
        return "\n".join(lines)


def verify_splits(
    examples: Sequence[Example],
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    seed: int = SPLIT_SEED,
) -> VerificationResult:
    """Check the written splits against every property the plan requires."""
    test = read_split(splits_dir / TEST_FILENAME)
    smoke = read_split(splits_dir / SMOKE_FILENAME)
    test_ids, smoke_ids = set(test.ids), set(smoke.ids)
    all_ids = {e.qid for e in examples}
    by_id = {e.qid: e for e in examples}

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        ("TEST size", len(test.ids) == TEST_SIZE, f"{len(test.ids)} ids (expected {TEST_SIZE})")
    )
    checks.append(
        ("SMOKE size", len(smoke.ids) == SMOKE_SIZE, f"{len(smoke.ids)} ids (expected {SMOKE_SIZE})")
    )
    checks.append(
        ("TEST ids unique", len(test_ids) == len(test.ids), f"{len(test_ids)} distinct")
    )
    checks.append(
        ("SMOKE ids unique", len(smoke_ids) == len(smoke.ids), f"{len(smoke_ids)} distinct")
    )

    overlap = test_ids & smoke_ids
    checks.append(("TEST/SMOKE disjoint", not overlap, f"{len(overlap)} shared ids"))

    strat = Counter(by_id[i].question_type.value for i in test.ids if i in by_id)
    expected_strat = {t.value: n for t, n in TEST_PER_TYPE.items()}
    checks.append(
        (
            "TEST stratification",
            dict(strat) == expected_strat,
            f"{dict(sorted(strat.items()))} (expected {dict(sorted(expected_strat.items()))})",
        )
    )

    unknown = (test_ids | smoke_ids) - all_ids
    checks.append(("All ids exist in dataset", not unknown, f"{len(unknown)} unknown ids"))

    dev = all_ids - test_ids
    checks.append(
        (
            "SMOKE drawn from DEV",
            smoke_ids <= dev,
            f"{len(smoke_ids - dev)} smoke ids outside DEV",
        )
    )
    checks.append(
        (
            "DEV partitions the dataset",
            len(dev) + len(test_ids) == len(all_ids),
            f"DEV {len(dev)} + TEST {len(test_ids)} = {len(dev) + len(test_ids)} of {len(all_ids)}",
        )
    )

    # Determinism: re-run selection from scratch and require identical output.
    regenerated_test = select_test_ids(examples, seed=seed)
    regenerated_smoke = select_smoke_ids(examples, regenerated_test, seed=seed)
    checks.append(
        (
            "TEST selection deterministic",
            regenerated_test == list(test.ids),
            "regenerated selection is identical",
        )
    )
    checks.append(
        (
            "SMOKE selection deterministic",
            regenerated_smoke == list(smoke.ids),
            "regenerated selection is identical",
        )
    )

    return VerificationResult(checks)


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["generate", "verify"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite frozen splits")
    args = parser.parse_args(argv)

    examples = load_hotpotqa(args.dataset)

    if args.command == "generate":
        test, smoke = write_splits(
            examples, splits_dir=args.splits_dir, source=args.dataset, force=args.force
        )
        print(f"seed: {SPLIT_SEED}")
        print(f"wrote {args.splits_dir / TEST_FILENAME}  ({len(test.ids)} ids)")
        print(f"wrote {args.splits_dir / SMOKE_FILENAME} ({len(smoke.ids)} ids)")

    result = verify_splits(examples, splits_dir=args.splits_dir)
    print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
