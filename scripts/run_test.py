"""Run the frozen TEST evaluation end to end.

    .venv/bin/python scripts/run_test.py

Executes `amrag.pipeline.run` — the real pipeline, not experiment logic — over the
150 frozen TEST questions, scores them with the existing harness, and writes a
machine-readable artifact to data/processed/test_results.json.

Resume-safe: the pipeline appends every verdict and answer to its caches as it goes,
so a stopped run can simply be restarted. Completed questions are served from cache
and cost no API call. Nothing is ever deleted or overwritten.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from amrag import correct, dense, multihop, pipeline
from amrag import answer as answer_mod
from amrag import verify
from amrag.data_loader import load_hotpotqa
from amrag.harness import aggregate, evaluate
from amrag.splits import load_split_ids

DATASET = Path("data/raw/hotpot_dev_distractor_v1.json")
RESULTS = Path("data/processed/test_results.json")
QUOTA_MARKERS = ("PerDay", "RESOURCE_EXHAUSTED", "429")

SECONDS_PER_CALL = 4.2
"""Provider limit is 15 requests/minute; 4.2s per call paces us to ~14.3. Same value
the Phase 7 and Phase 9 runners used. Sleeping is charged per *call actually made*,
so a cache-served question waits not at all and a resumed run stays fast."""

KEY = os.environ.get("GEMINI_API_KEY", "")
scrub = lambda s: s.replace(KEY, "<REDACTED>") if KEY else s
rows = lambda path: sum(1 for _ in path.open()) if path.exists() else 0


def main() -> int:
    examples = load_hotpotqa(DATASET)
    by_id = {e.qid: e for e in examples}
    test_ids = sorted(load_split_ids("test"))          # the frozen artifact, as-is
    subset = [by_id[qid] for qid in test_ids]

    dense.load_cache()
    pipeline.load_caches()

    print(f"FROZEN TEST RUN — {len(subset)} questions")
    print(f"  verifier  {verify.MODEL}   generator {answer_mod.MODEL}")
    print(f"  ROUTING_THRESHOLD {multihop.ROUTING_THRESHOLD}   "
          f"BASELINE_K {correct.BASELINE_K}   CORRECTION_K {correct.CORRECTION_K}\n", flush=True)

    calls_before = rows(verify.CACHE_PATH) + rows(answer_mod.CACHE_PATH)
    predictions, failures, latencies = {}, [], {}
    stopped = None

    for i, example in enumerate(subset, 1):
        start = time.perf_counter()
        calls_at_start = rows(verify.CACHE_PATH) + rows(answer_mod.CACHE_PATH)
        try:
            predictions[example.qid] = pipeline.run(example)
            latencies[example.qid] = time.perf_counter() - start
        except Exception as exc:                       # failures are recorded, not answered
            message = scrub(f"{type(exc).__name__}: {exc}")[:400]
            failures.append({"qid": example.qid, "error": message})
            print(f"  [{i}] FAIL {message[:110]}", flush=True)
            if any(m in message for m in QUOTA_MARKERS):
                stopped = "quota exhausted"
                break
        if i % 25 == 0:
            print(f"  ... {i}/{len(subset)}  ok={len(predictions)} fail={len(failures)}", flush=True)
        made = rows(verify.CACHE_PATH) + rows(answer_mod.CACHE_PATH) - calls_at_start
        time.sleep(max(0.0, made * SECONDS_PER_CALL - (time.perf_counter() - start)))

    api_calls = rows(verify.CACHE_PATH) + rows(answer_mod.CACHE_PATH) - calls_before
    attempted = len(predictions) + len(failures)
    skipped = len(subset) - attempted
    if stopped:
        print(f"\n!! {stopped} — stopping safely; {skipped} questions not attempted", flush=True)

    # Score with the existing harness by replaying the predictions we already have.
    scored = [e for e in subset if e.qid in predictions]
    results = evaluate(lambda e: predictions[e.qid], scored)
    summary = aggregate(results)

    lat = sorted(latencies.values())
    artifact = {
        "split": "test",
        "n_test": len(subset),
        "successful": len(predictions),
        "failed": len(failures),
        "skipped": skipped,
        "stopped": stopped,
        "config": {
            "verifier_model": verify.MODEL,
            "generator_model": answer_mod.MODEL,
            "verifier_prompt_version": verify.PROMPT_VERSION,
            "generator_prompt_version": answer_mod.PROMPT_VERSION,
            "routing_threshold": multihop.ROUTING_THRESHOLD,
            "baseline_k": correct.BASELINE_K,
            "correction_k": correct.CORRECTION_K,
        },
        "calls": {
            "logical_per_question": pipeline.LLM_CALLS,
            "new_api_calls": api_calls,
            "cache_served": max(0, pipeline.LLM_CALLS * len(predictions) - api_calls),
        },
        "latency": {
            "mean": sum(lat) / len(lat) if lat else None,
            "median": lat[len(lat) // 2] if lat else None,
        },
        "aggregate": summary,
        "failures": failures,
        "per_question": [
            {"qid": r.qid, "predicted": r.predicted_answer, "gold": r.gold_answer,
             "em": r.answer_em, "f1": r.answer_f1, "sp_precision": r.sp_precision,
             "sp_recall": r.sp_recall, "sp_f1": r.sp_f1, "llm_calls": r.llm_calls}
            for r in results
        ],
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== TEST RESULTS ({len(predictions)}/{len(subset)} scored) ===")
    for k in ("answer_em", "answer_f1", "sp_precision", "sp_recall", "sp_f1"):
        print(f"  {k:<14} {summary.get(k, float('nan')):.3f}")
    print(f"\n  successful {len(predictions)}   failed {len(failures)}   skipped {skipped}")
    print(f"  new api calls {api_calls}   cache-served {artifact['calls']['cache_served']}"
          f"   logical calls/question {pipeline.LLM_CALLS}")
    if lat:
        print(f"  latency mean {artifact['latency']['mean']:.2f}s  "
              f"median {artifact['latency']['median']:.2f}s")
    print(f"  artifact -> {RESULTS}")
    return 1 if (failures or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
