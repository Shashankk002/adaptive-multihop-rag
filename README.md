# Adaptive Multi-Hop RAG with Evidence Verification

A retrieval pipeline for HotpotQA that checks whether the evidence it retrieved is
sufficient to answer the question, and widens the evidence set when it is not.

Research codebase. The deliverable is the measured behaviour — including the approaches
that were evaluated and rejected — not a product.

## Problem

HotpotQA questions need facts from two paragraphs. Retrieval reliably finds one and
often misses the other, and the reader then answers from incomplete evidence without
signalling a problem. On this project's development data, dense retrieval placed *both*
required paragraphs in the top 2 for only 55% of questions.

The pipeline adds a step that detects incomplete evidence and responds to it.

## Pipeline

```
HotpotQA per-question candidate set (~10 paragraphs)
        │
        ▼
  dense retrieval                    bge-small-en-v1.5, cosine over the candidates
        │
        ▼
  adaptive multi-hop routing         second hop only when the ranking looks unsure
        │
        ▼
  top-2 evidence
        │
        ▼
  sufficiency verification           LLM: SUFFICIENT / INSUFFICIENT
        │
   ┌────┴──────────────────┐
   │ sufficient            │ insufficient
   ▼                       ▼
  keep top-2          expand to top-5 of the same ranking
   └────┬──────────────────┘
        ▼
  answer generation                  LLM, restricted to the selected evidence
        │
        ▼
  answer, or the literal string INSUFFICIENT
```

Two LLM calls per question: one verification, one generation.

Scope note: the corpus is the **HotpotQA per-question candidate set**, not a document
store. Each question ships ~10 candidate paragraphs and both gold paragraphs are always
among them, so there is nothing to fetch from elsewhere — "correction" can only mean
looking deeper into the ranking already computed.

## Modules

| module | role |
|---|---|
| `data_loader.py`, `schema.py` | HotpotQA parsing into dataset-agnostic records |
| `retrieval.py` | BM25, hand-written |
| `dense.py` | embeddings, cosine similarity, on-disk vector cache |
| `multihop.py` | two-hop retrieval and the routing signal |
| `verify.py` | LLM evidence-sufficiency verifier |
| `correct.py` | selective evidence expansion |
| `answer.py` | grounded answer generation |
| `pipeline.py` | composition of the above |
| `harness.py`, `metrics.py` | evaluation; HotpotQA-standard EM / F1 / supporting-fact PRF |
| `splits.py` | frozen, seeded evaluation splits and their verification |
| `hybrid.py`, `rerank.py` | **evaluated and rejected**; kept so the negative results stay reproducible |

Imports flow one way: retrieval → multi-hop → verification → expansion → answer.

## Key decisions

**Routing signal.** A second retrieval hop helps bridge questions and harms comparison
questions, so it is applied selectively. The signal is `m23`, the gap between the 2nd
and 3rd dense scores: a wide gap means the ranking has cleanly separated a pair and a
second hop has nothing to add. When `m23 < 0.020` the top-1 paragraph's text is appended
to the question and the candidates are re-ranked. `m23` predicts evidence sufficiency
(AUC 0.776) better than it predicts question type (AUC 0.717), so it is a
retrieval-confidence signal rather than a question classifier. The rank-1/rank-2
margin, the obvious first choice, is *anti*-predictive (AUC 0.456).

**Verification.** One call; the question and the top-2 paragraphs are judged
**jointly**, because every complete-evidence question needs facts from both — a
per-paragraph judge would call each one insufficient on its own. The verdict is binary:
HotpotQA distractors are irrelevant rather than contradictory, so a REFUTED class has no
population. The model also returns a confidence, which is recorded but **unused**: 93.8%
of verdicts report 1.00, and mean confidence is 0.996 when correct versus 0.985 when
wrong.

**Expansion.** On INSUFFICIENT the evidence widens from the top 2 to the top 5 of the
ranking already held. One round, no extra retrieval or LLM call. This is not query
reformulation and not autonomous correction: on flagged questions the missing paragraph
was inside the top 10 in 100% of cases, so looking deeper is the only useful move, and a
second round could reach nothing a larger `k` does not.

**Generation.** Plain text with a 64-token cap — HotpotQA answers are two words at the
median, so a JSON schema would only add a parse-failure mode. The prompt forbids outside
knowledge and asks for the literal string `INSUFFICIENT` when the evidence does not
contain the answer.

**Failures are never coerced.** A failed verifier call is not an INSUFFICIENT verdict
and a failed generation is not an answer; both raise and are recorded as failures.

## Evaluation setup

HotpotQA dev (distractor), 7,405 questions, partitioned into frozen, seeded splits
(`data/splits/*.json`, committed):

| split | n | purpose |
|---|---|---|
| TEST | 150 (75 bridge / 75 comparison) | held out; read **once**, at the end |
| DEV / DEV-EVAL | 7,255 / 6,255 | retrieval development (DEV-EVAL = DEV − TUNE) |
| TUNE | 1,000 | threshold selection |
| `verify_tune` | 500 | verifier evaluation (subset of TUNE) |
| `answer_dev` | 200 | evidence-policy development (subset of `verify_tune`) |
| SMOKE | 100 | wiring checks only |

Every threshold and constant was frozen on a development split before evaluation. No
TEST question id appeared in either LLM cache before the TEST evaluation began.
`python -m amrag.splits verify` regenerates TEST, SMOKE, TUNE and `verify_tune` from the
seed and checks them against the committed files. `answer_dev` cannot be regenerated
from the seed alone because its candidate pool was "verify_tune questions with a usable
verdict"; it is checked for size, containment and disjointness instead.

Retrieval is scored by **gold-pair recall** (`both@2`: both gold paragraphs in the top
2), since a multi-hop question is unanswerable from one of them. Answers use standard
HotpotQA EM and token F1, and supporting facts use the official set-based P/R/F1; the
implementation was checked against `hotpot_evaluate_v1.py` on 3,600 answer pairs and
5,000 supporting-fact cases.

## Held-out TEST result

150 questions, read once, nothing changed afterward. Split file SHA-256
`05d02a27…f643` (`shasum -a 256 data/splits/test_ids.json`).

| metric | value |
|---|---|
| **answer EM** | **0.620** |
| **answer F1** | **0.732** |
| bridge EM / F1 | 0.520 / 0.622 |
| comparison EM / F1 | 0.720 / 0.843 |
| supporting-fact P / R / F1 | 0.317 / 0.930 / 0.442 |
| abstentions (`INSUFFICIENT`) | 21 / 150 (14.0%) |

150/150 completed, 0 failures, 2 LLM calls per question, ~3.7 s/question. The run was
interrupted twice by daily quota and resumed from cache; 282 of the 300 calls were made
fresh and 18 were served from those earlier attempts of the same run.

Supporting-fact precision is low **by construction**: the pipeline emits every sentence
of the selected paragraphs because sentence-level selection was never built. Recall
(0.930) is the meaningful half.

This is not a competitive HotpotQA result — published systems exceed 0.70 EM. It is an
end-to-end measurement of this pipeline on a genuinely untouched sample.

## Development results

Everything in this section was measured on development splits, not held out.

**Retrieval** (`both@2`, DEV, n = 7,255 unless noted):

| retriever | `both@2` |
|---|---|
| BM25 | 0.320 |
| dense | 0.554 |
| **dense + adaptive routing** (adopted; DEV-EVAL, n = 6,255) | **0.586** |

The routing gain is +0.031 (McNemar p = 2e-13), sending 32.1% of questions to a second
hop. TUNE and DEV-EVAL agreed on the gain to three decimals.

**Rejected approaches:**

- **Hybrid RRF fusion.** Unweighted fusion *lost* to dense alone at every fusion
  constant (best 0.507 vs 0.554). Equal weighting lets a retriever scoring 0.280 on
  comparison questions outvote one scoring 0.885. A weighted variant recovered +0.017 —
  real, but not worth two tuned hyperparameters.
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`). Made retrieval *worse* at
  every depth (0.554 → 0.468 at N = 5) even though the model itself works and improves
  recall@1. It is trained on single-hop relevance, so it promotes the paragraph that
  matches the question and buries the bridge paragraph, which is relevant only via the
  first. Roughly 2× as many questions broken as fixed, at ~1000× the latency.
- **Always-on two-hop expansion.** Netted to zero: bridge +0.047, comparison −0.364.
  This is what motivated routing.
- **Keyword routing** ("does the question start with *Were*…"). Rejected for `m23`,
  which scored higher (0.586 vs 0.577 / 0.583) with no linguistic assumptions.

**Verification** (`verify_tune`, 486 usable verdicts of 500; 1 excluded for a
truncated JSON response, 13 lost to daily quota): accuracy 0.809, INSUFFICIENT
P / R / F1 0.766 / 0.770 / 0.768, AUC 0.804 — against 0.729 for the free `m23` signal
on the same questions.

**Expansion** (same 486): gold-pair recall 0.588 → 0.796 at `k = 5`, mean evidence
2.00 → 3.24 paragraphs, 0 regressions (structural: expansion returns a superset). 41.4%
of questions were expanded; 23.4% of those expansions were unnecessary, which is the
verifier's false-alarm rate expressed as cost.

**Answer quality** (`answer_dev`, 200 questions, four evidence policies):

| evidence policy | EM | F1 | mean paragraphs | abstention |
|---|---|---|---|---|
| top-2 | 0.415 | 0.498 | 2.00 | 0.380 |
| expand to 3 | 0.505 | 0.597 | 2.48 | 0.275 |
| expand to 4 | 0.510 | 0.619 | 2.95 | 0.230 |
| **expand to 5** | **0.535** | **0.659** | 3.42 | 0.180 |

Almost all of the gain is reduced abstention rather than corrected answers: across the
whole range only 1–2 questions moved from a wrong answer to a right one. `k = 5` was
chosen over `k = 3` on F1 (+0.061, 95% CI [+0.021, +0.103]); EM alone could not separate
them. Holding the question set fixed, more context did not reduce reading accuracy.

Full tables, per-phase analysis and the decision log are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Configuration

| | |
|---|---|
| retriever | `BAAI/bge-small-en-v1.5` @ `5c38ec7c`, 384-dim, cosine |
| routing | second hop when `m23 < 0.020` |
| evidence sizes | `BASELINE_K = 2`, `CORRECTION_K = 5` |
| verifier | `gemini-3.5-flash-lite`, prompt v1, JSON schema, 256 output tokens |
| generator | `gemini-3.1-flash-lite`, prompt v1, plain text, 64 output tokens |
| decoding | temperature 0, seed 20260908 |

## Repository

```
src/amrag/            library code
tests/                pytest suite — offline, no API key, no network, no dataset
scripts/run_test.py   the frozen TEST runner (the only script)
data/splits/          frozen split id files (committed)
data/raw/             HotpotQA dev distractor file (gitignored)
data/processed/       embedding cache, LLM caches, test_results.json (gitignored)
PROJECT_PLAN.md       full experimental record and decision log
```

## Setup

Python 3.12, as pinned in `pyproject.toml` (3.14 lacked wheels for the ML stack when the
project started).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest          # 333 tests, ~2 s, fully offline
```

Reported results were produced with `sentence-transformers` 6.0.1, `torch` 2.14.0,
`numpy` 2.5.3, `google-genai` 2.22.0 on macOS (Apple MPS; falls back to CPU elsewhere,
with identical rankings).

To reproduce the evaluation, place `hotpot_dev_distractor_v1.json` (HotpotQA dev,
distractor setting) at `data/raw/`, export `GEMINI_API_KEY`, and run from the repository
root:

```bash
.venv/bin/python -m amrag.splits verify      # regenerate and check the frozen splits
.venv/bin/python scripts/run_test.py         # ~300 API calls at free-tier pacing, ~20 min
```

The runner is resume-safe: verdicts and answers are appended to on-disk caches as they
are produced, so an interrupted run continues without repeating successful calls. The
embedding model (~130 MB) downloads on first use; the embedding cache is optional.

## Limitations

- **TEST is one 150-question sample.** Standard error on EM is roughly ±0.04.
- **Development numbers are not held out.** The routing threshold, `CORRECTION_K` and
  the evidence policy were selected on development splits.
- **Supporting-fact precision is low by design.** No sentence-level selection exists.
- **The sufficiency proxy is imperfect.** "Both labelled gold paragraphs retrieved" is
  used as ground truth, but HotpotQA labels one sufficient evidence path, not every one,
  so some verifier "false positives" are genuinely answerable and the measured error
  rate is an upper bound.
- **HotpotQA is public and probably in the LLMs' pretraining data.** Answers produced
  without the gold answer present in the evidence were measured at 0.5–2.0% on
  development data.
- **Two different Gemini models** are used for verification and generation, forced by
  free-tier daily quotas rather than chosen.
- **Gemini model ids are moving aliases** and cannot be pinned to a revision, so exact
  reproduction of the LLM calls depends on the (gitignored) caches rather than on the
  model. The embedding model is pinned to a commit and its results are exactly
  reproducible.
- **HotpotQA only.** The routing threshold is calibrated to this retriever's score
  scale; generalisation to other multi-hop datasets was not attempted.

---

*Earlier working name: "Self-Correcting Multi-Hop RAG". Renamed because the system does
not perform reasoning-level self-correction — it verifies evidence sufficiency and
expands the evidence set.*
