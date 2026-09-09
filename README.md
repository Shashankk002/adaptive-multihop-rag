# Adaptive Multi-Hop RAG with Evidence Verification

An adaptive multi-hop retrieval pipeline that verifies evidence sufficiency and
selectively expands retrieved evidence before answer generation.

Built as a research codebase on HotpotQA. The deliverable is measured behaviour —
including the approaches that were evaluated and rejected — rather than a product.

## The problem

HotpotQA questions require facts from two different paragraphs. Retrieval reliably finds
one of them and often misses the other, so the reader is asked to answer from incomplete
evidence and does so without signalling a problem. On this pipeline's own development
data, dense retrieval put *both* required paragraphs in the top 2 for only 55% of
questions.

This project adds a step that detects incomplete evidence and responds to it.

## Architecture

```
HotpotQA per-question candidate set (~10 paragraphs)
        │
        ▼
  dense retrieval                       bge-small-en-v1.5, cosine over the candidates
        │
        ▼
  adaptive multi-hop routing            second hop only when the ranking looks unsure
        │
        ▼
  top-2 evidence
        │
        ▼
  evidence sufficiency verification     LLM: SUFFICIENT / INSUFFICIENT
        │
   ┌────┴─────────────────────┐
   │ sufficient               │ insufficient
   ▼                          ▼
  keep top-2            expand to top-5          selective evidence expansion
   └────┬─────────────────────┘
        ▼
  answer generation                     LLM, constrained to the selected evidence
        │
        ▼
  answer, or the literal string INSUFFICIENT
```

Two LLM calls per question: one verification, one generation.

Important scoping note: the corpus is the **HotpotQA per-question candidate set**, not a
document store. Each question ships ~10 candidate paragraphs and both gold paragraphs
are always among them. There is nothing to fetch from elsewhere.

## Modules

| module | role |
|---|---|
| `data_loader.py`, `schema.py` | HotpotQA parsing → dataset-agnostic records |
| `retrieval.py` | BM25, hand-written (~40 lines) |
| `dense.py` | embeddings, cosine similarity, on-disk vector cache |
| `multihop.py` | two-hop retrieval and the routing signal |
| `verify.py` | LLM evidence-sufficiency verifier |
| `correct.py` | selective evidence expansion |
| `answer.py` | grounded answer generation |
| `pipeline.py` | composition of the above |
| `harness.py`, `metrics.py` | evaluation, HotpotQA-standard EM/F1 |
| `splits.py` | frozen, seeded, hashed evaluation splits |
| `hybrid.py`, `rerank.py` | **evaluated and rejected** — retained so the negative results stay reproducible |

Imports flow one direction: retrieval → multi-hop → verification → expansion → answer.

## Adaptive multi-hop routing

A second retrieval hop helps bridge questions and actively harms comparison questions,
so it must be applied selectively. Routing uses **`m23`**: the gap between the 2nd and
3rd dense retrieval scores. A wide gap means the ranking has cleanly separated a pair
and a second hop has nothing to add.

When `m23 < 0.020`, the pipeline runs a second hop — appending the top-1 paragraph's
text to the question and re-retrieving over the same candidates.

This is a retrieval-confidence signal, not a question classifier: its AUC for predicting
evidence sufficiency (0.776) is higher than its AUC for separating question types
(0.717). The rank-1/rank-2 margin, which is what most people reach for first, is
*anti*-predictive (AUC 0.456).

## Evidence verification

One LLM call. The question and the top-2 paragraphs go in together; a binary
SUFFICIENT/INSUFFICIENT verdict comes out, with a short reason and a confidence score.

The paragraphs are judged **jointly**, because every complete-evidence question requires
facts from both — a per-paragraph verifier would call each one individually
insufficient. The verdict is binary rather than three-way because HotpotQA distractors
are irrelevant, not contradictory, so a REFUTED class has no population.

The confidence score is recorded but **not used**: 93.8% of verdicts report 1.00, and
mean confidence is 0.996 when correct versus 0.985 when wrong.

## Selective evidence expansion

On INSUFFICIENT, the evidence set widens from the top 2 to the top 5 **of the ranking
already computed**. One round. No additional LLM call, no additional retrieval.

Stated precisely, because the distinction matters: this is **not** query reformulation,
**not** autonomous correction, and **not** fetching new documents. The candidate set is
fixed per question, and on flagged questions the missing paragraph is already inside the
top 10 in 100% of cases. The only useful move is to look deeper into the existing
ranking. Termination is structural — with a fixed candidate set, a second round reaches
nothing a larger `k` does not.

## Answer generation

One LLM call. Plain text, not JSON — HotpotQA answers are two words at the median, so a
schema would only add a parse-failure mode. The prompt forbids outside knowledge and
instructs the model to return the literal string `INSUFFICIENT` when the evidence does
not contain the answer. That abstention is what distinguishes an answer grounded in the
retrieved evidence from one recalled from pretraining.

## Evaluation methodology

HotpotQA dev (distractor), 7,405 questions, partitioned into frozen, seeded,
SHA-256-hashed splits:

| split | n | purpose |
|---|---|---|
| TEST | 150 | held out; read **once**, at the end |
| DEV / DEV-EVAL | 7,255 / 6,255 | retrieval development |
| TUNE | 1,000 | threshold selection |
| `verify_tune` | 500 | verifier evaluation |
| `answer_dev` | 200 | evidence-policy development |
| SMOKE | 100 | wiring checks only |

Every threshold and constant was frozen on a development split before evaluation. TEST
was never read during development — verified by confirming no TEST question id appeared
in either LLM cache before the final run.

Retrieval is scored by **gold-pair recall** (`both@2`: are *both* gold paragraphs in the
top 2), since a multi-hop question is unanswerable from one of them. Answers use
standard HotpotQA EM and token-F1.

## Held-out TEST result

150 questions (75 bridge / 75 comparison), read once, nothing changed afterward.
Split hash `05d02a27…f643`.

| metric | value |
|---|---|
| **Answer EM** | **0.620** |
| **Answer F1** | **0.732** |
| bridge EM / F1 | 0.520 / 0.622 |
| comparison EM / F1 | 0.720 / 0.843 |
| supporting-fact precision | 0.317 |
| supporting-fact recall | 0.930 |
| supporting-fact F1 | 0.442 |
| abstentions (`INSUFFICIENT`) | 21/150 (14.0%) |

150/150 completed, 0 failures, 2 LLM calls per question, ~3.7 s/question.

Supporting-fact precision is low **by construction**: the pipeline emits every sentence
of the selected paragraphs, because sentence-level selection was never built. Recall
(0.930) is the meaningful half of that pair.

This is not a competitive HotpotQA result — published systems exceed 0.70 EM. It is an
honest end-to-end measurement of this pipeline.

## Retrieval methods evaluated

All figures below are development results, not held out.

| retriever | `both@2` (DEV, n=7,255) |
|---|---|
| BM25 | 0.320 |
| Dense | 0.554 |
| **Dense + adaptive routing** (adopted) | **0.586** |

**Three approaches were evaluated and rejected**, which is much of what this project has
to say:

- **Hybrid RRF fusion.** Unweighted fusion *lost* to dense alone (0.507 vs 0.554) at
  every fusion constant. Equal weighting let a retriever scoring 0.280 on comparison
  questions outvote one scoring 0.885. A weighted variant recovered +0.017 — real, but
  not worth two development-tuned hyperparameters.
- **Cross-encoder reranking.** Made retrieval *worse* at every depth (0.554 → 0.468 at
  N=5). The model works correctly and even improves recall@1; it is trained on MS MARCO
  single-hop relevance, so it confidently promotes the paragraph matching the question
  and buries the bridge paragraph, which is relevant only via the first. It broke about
  twice as many questions as it fixed, at roughly 1000× the latency.
- **Always-on multi-hop.** Netted to zero: query expansion helped bridge questions
  (+0.047) and destroyed comparison questions (−0.364).

That last result is what motivated routing. A **hand-tuned keyword rule** ("does the
question start with *Were*…") was also rejected in favour of the score-based `m23`
signal, which performed better (0.586 vs 0.577/0.583) with no linguistic assumptions.

**Verification** (`verify_tune`, 486 usable verdicts): accuracy 0.809, INSUFFICIENT F1
0.768, AUC 0.804 — against 0.729 for the free `m23` signal on the same questions.

**Evidence expansion** (`verify_tune`, 486): gold-pair recall 0.588 → 0.796 at `k=5`,
mean evidence 2.00 → 3.24 paragraphs, zero regressions (structural — expansion returns a
superset).

**Answer quality** (`answer_dev`, 200, four evidence policies, 485 generations): EM
0.415 → 0.535 and F1 0.498 → 0.659 from top-2 to `k=5`. Nearly all of the gain is
reduced abstention rather than corrected answers. `k=5` was confirmed over `k=3` on F1
(+0.061, 95% CI [+0.021, +0.103]); EM alone could not separate them.

## Repository

```
src/amrag/            library code
tests/                pytest suite (offline; no API key or network needed)
scripts/run_test.py   the single frozen TEST runner
data/splits/          frozen split id files (committed)
data/raw/             dataset (gitignored)
data/processed/       caches and results (gitignored)
PROJECT_PLAN.md       full experimental record and decision log
```

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest -q          # 333 tests, fully offline
```

The test suite needs no API key, no network and no dataset: LLM and embedding calls sit
behind seams that tests replace with fakes.

To reproduce the evaluation you also need the HotpotQA dev distractor file at
`data/raw/hotpot_dev_distractor_v1.json` and `GEMINI_API_KEY` exported from a file every
shell reads (`~/.zshenv`, not `~/.zshrc` — non-interactive shells do not read the latter):

```bash
./.venv/bin/python -m amrag.splits verify     # check the frozen splits
./.venv/bin/python scripts/run_test.py        # ~300 API calls, ~20 min
```

The runner is resume-safe: verdicts and answers are cached on disk as they are produced,
so an interrupted run continues without repeating successful calls.

## Configuration

| | |
|---|---|
| retriever | `bge-small-en-v1.5`, 384-dim, cosine |
| routing threshold | `m23 < 0.020` |
| evidence sizes | `BASELINE_K` 2, `CORRECTION_K` 5 |
| verifier | `gemini-3.5-flash-lite`, prompt v1 |
| generator | `gemini-3.1-flash-lite`, prompt v1, max 64 output tokens |
| decoding | temperature 0, seed 20260908 |

## Limitations

- **TEST is a single 150-question sample.** The standard error on EM is roughly ±0.04.
- **Development numbers are not held out.** The routing threshold, `CORRECTION_K` and
  the evidence policy were all selected on development splits.
- **Supporting-fact precision is low by design** — no sentence-level selection exists.
- **The evidence-sufficiency proxy is imperfect.** It treats "both labelled gold
  paragraphs retrieved" as ground truth, but HotpotQA labels *one* sufficient evidence
  path, not every one, so some verifier "false positives" are genuinely answerable and
  the measured error rate is an upper bound.
- **HotpotQA is public and probably in the LLMs' pretraining data.** Answers produced
  without the evidence present were measured at 0.5–2.0% on development data.
- **Two different Gemini models** are used (verifier and generator). This was forced by
  free-tier daily quotas, not chosen.
- **Gemini model ids are moving aliases** and cannot be pinned to a revision, so exact
  reproduction depends on the cached outputs rather than on the model.
- **HotpotQA only.** Generalisation to 2WikiMultihopQA or MuSiQue was not attempted, and
  the routing threshold is calibrated to this retriever's score scale.

---

*Earlier working name: "Self-Correcting Multi-Hop RAG". Renamed because the system does
not perform reasoning-level self-correction — it verifies evidence sufficiency and
expands the evidence set.*
