# PROJECT_PLAN.md

Roadmap and decision record for Self-Correcting Multi-Hop RAG.

Status: **Phase 0 complete.** Nothing implemented.

## Goal

Build a multi-hop RAG pipeline that detects when its own retrieved evidence fails to
support an answer, and corrects itself by re-querying — rather than answering from
insufficient evidence.

The research question: *does explicit evidence verification plus a correction loop beat
a strong single-pass retrieval baseline, and by how much, at what cost in latency and
LLM calls?*

## Success criteria

Measured on HotpotQA distractor, DEV split:

| Axis | Metric | Bar |
|---|---|---|
| Answer quality | EM / F1 | Beat the BM25 + reader baseline established in Phase 2 |
| Evidence quality | Supporting-fact F1 | Beat the same baseline |
| Correction value | Δ metrics with loop on vs. off | Positive and larger than run-to-run noise |
| Cost | LLM calls & wall-clock per question | Tracked every phase; a win that costs 10× is reported as such |

The Phase 2 BM25 baseline is the number every later phase is judged against. There is
no point improving a pipeline whose baseline we never measured.

## Dataset

**HotpotQA, distractor setting.** Chosen for the initial benchmark because each question
ships with its own 10-paragraph candidate set (2 gold + 8 distractors), which makes the
retrieval problem well-posed and self-contained — no full-Wikipedia index needed to get
a real multi-hop signal. It also provides sentence-level supporting facts, which is
exactly the supervision the verification stage needs.

2WikiMultihopQA and MuSiQue are added **after** the pipeline works end to end, as
generalization checks. MuSiQue in particular is adversarially constructed against
shortcut reasoning and will be the honest test of whether correction helps.

**Not downloaded yet.** Requires approval.

### Splits

- **DEV** — all development, tuning, prompt iteration, error analysis.
- **TEST** — frozen. Read at most once per milestone. Never tuned against.
- **SMOKE** — a small fixed DEV subset (~100 questions) for fast iteration. Seeded and
  committed as ids only, never as data.

## Phases

Each phase has one deliverable and an acceptance check. A phase is not done until its
check passes.

### Phase 0 — Scaffold ✅
Repo structure, Python 3.12 venv, CLAUDE.md, PROJECT_PLAN.md.
**Acceptance:** repo committed; `scrag` importable once installed.

### Phase 1 — Data loading + evaluation harness
HotpotQA loader, split definitions, and the metrics (EM, F1, supporting-fact F1) that
every later phase reports.
**Acceptance:** loader reproduces official dataset counts; metrics reproduce known
scores on a hand-built fixture with a hand-computed answer.

*Eval comes before retrieval on purpose. Building the scoreboard first means every
subsequent phase is measured rather than assumed to have helped.*

### Phase 2 — BM25 retrieval (baseline) ✅
Sparse retrieval over the distractor candidate set.
**Acceptance:** recorded recall@k on SMOKE and DEV. This becomes the baseline row in
the results table.

**Results (query = the full question, DEV n=7255):**

| k | recall@k | both@k |
|---|---|---|
| 1 | 0.413 | 0.000 |
| 2 | 0.624 | **0.320** |
| 3 | 0.726 | 0.486 |
| 5 | 0.837 | 0.685 |

`both@2` = **0.320** is the Phase 2 baseline. SMOKE (n=100) gives 0.300, within noise
of DEV. By type: bridge 0.329, comparison 0.280. Retrieval costs 0.19 ms/question.

`recall@10` and `both@10` are not reported: every question has exactly 2 gold
paragraphs and both are always among the ≤10 candidates, so any metric at k=10 is
identically 1.0 and measures the dataset, not the retriever. `both@1` is likewise
degenerate at 0.0 — two gold paragraphs cannot fit in one slot.

### Phase 3 — Dense retrieval ✅
Embedding-based retrieval behind the same interface as BM25.
**Acceptance:** recall@k reported next to BM25; the two are swappable without changes
to calling code.

**Setup:** `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`,
384-dim, device `mps`, query prefix applied, same `title + body` text as BM25.
Resolved versions: sentence-transformers 6.0.1, torch 2.14.0, numpy 2.5.3,
transformers 5.16.1, Python 3.12.4.

**Results (query = the full question, DEV n=7255):**

| retriever | recall@1 | recall@2 | recall@3 | recall@5 | both@2 | ms/q |
|---|---|---|---|---|---|---|
| BM25 | 0.413 | 0.624 | 0.726 | 0.837 | 0.320 | 0.22 |
| **Dense** | **0.454** | **0.762** | **0.855** | **0.926** | **0.554** | 0.07 |

`both@2` improves **+0.234 (0.320 → 0.554)**. By type: bridge 0.329 → 0.474,
comparison 0.280 → **0.885**. Nearly all of the comparison gap closes, because both
entities are named in the question and semantic matching finds them; bridge questions
remain the hard case, exactly as predicted. Questions missing a gold paragraph at k=2
fall from 68.0% to 44.6%.

Timing is cache-served (0.07 ms/q); the one-off embedding pass over 74,040 unique
texts took 6.6 min on MPS.

**BM25 ↔ dense overlap (motivates Phase 4):**

| | share of DEV |
|---|---|
| both retrievers succeed at both@2 | 21.3% |
| BM25 only | 10.7% |
| dense only | 34.2% |
| neither | 33.9% |

Oracle fusion ceiling at k=2 is **0.661** against the best single retriever's 0.554 —
**+0.107 of headroom**. The 10.7% BM25-only column is the important one: sparse
retrieval still finds questions dense misses, so hybrid is motivated by evidence
rather than by assumption.

### Phase 4 — Hybrid retrieval ⚠️ largely negative
Reciprocal Rank Fusion of the complete BM25 and dense rankings.
**Acceptance:** beats the better of its two components on recall@k, or the negative
result is recorded and hybrid is dropped.

**Unweighted RRF loses to dense at every rrf_k.** DEV n=7255, selection metric both@2:

| rrf_k | recall@1 | recall@2 | recall@3 | recall@5 | both@2 | vs dense |
|---|---|---|---|---|---|---|
| 0 | 0.457 | 0.736 | 0.856 | 0.933 | 0.494 | −0.060 |
| 1 | 0.458 | 0.741 | 0.853 | 0.932 | 0.502 | −0.052 |
| 2 | 0.458 | 0.741 | 0.849 | 0.931 | 0.504 | −0.050 |
| 3 | 0.457 | 0.742 | 0.847 | 0.931 | **0.507** | −0.047 |
| 5 | 0.458 | 0.737 | 0.843 | 0.929 | 0.501 | −0.053 |
| 10 | 0.457 | 0.733 | 0.835 | 0.926 | 0.493 | −0.061 |
| 20 | 0.456 | 0.730 | 0.830 | 0.923 | 0.489 | −0.065 |
| 60 | 0.456 | 0.730 | 0.827 | 0.918 | 0.488 | −0.066 |

The curve is shallow and peaks at rrf_k=3, confirming that the published default of 60
is mis-scaled for 10-candidate lists — but no constant rescues the method. At rrf_k=3
hybrid recovers 66.5% of the BM25-only slice yet keeps only 54.7% of the dense-only
slice: 777 recoveries against **1123 regressions**, net −346 questions. Comparison
questions are hit hardest (dense 0.885 → hybrid 0.676) because equal weighting lets a
retriever scoring 0.280 on that slice outvote one scoring 0.885.

**Weighted RRF (dense weight w) recovers a small real gain**, run only because the
unweighted result triggered the pre-agreed condition:

| w | rrf_k=1 | rrf_k=2 | rrf_k=3 | rrf_k=5 |
|---|---|---|---|---|
| 0.5 | 0.502 | 0.504 | 0.507 | 0.501 |
| 0.6 | 0.550 | 0.553 | 0.548 | 0.542 |
| 0.7 | **0.571** | 0.567 | 0.565 | 0.557 |
| 0.8 | 0.564 | 0.563 | 0.563 | 0.560 |
| 0.9 | 0.554 | 0.554 | 0.554 | 0.554 |

Best: **w=0.7, rrf_k=1 → both@2 = 0.571 (+0.017 over dense)**. Paired analysis: 317
recoveries vs 197 regressions, net +120 questions, McNemar z=5.25 (p<0.0001) — real,
not noise. recall@1 0.456 / @2 0.774 / @3 0.872 / @5 0.938. Bridge improves
0.474 → 0.500; comparison *degrades* 0.885 → 0.863. Captures 15.7% of the oracle
headroom (oracle 0.661).

**Decision: keep dense as the default retriever; do not adopt hybrid.** +0.017 is only
marginally above the ~+0.015 line agreed in advance, and buying it costs a second
retriever, two DEV-tuned hyperparameters, and a regression on comparison questions.
`hybrid_retrieve` stays in the codebase for reuse in later phases, where fusion over
better inputs (reranked or multi-hop candidates) may pay off more.

### Phase 5 — Reranking
Cross-encoder reranking over the fused candidates.
**Acceptance:** precision@k improves at fixed recall; added latency measured and
reported.

### Phase 6 — Multi-hop retrieval
Query decomposition and iterative hop-by-hop retrieval.
**Acceptance:** on questions whose second gold paragraph is unreachable from the
original query, gold-pair recall improves over single-pass.

### Phase 7 — Evidence verification
Judge whether retrieved evidence actually supports the claim it is retrieved for.
**Acceptance:** verifier agreement against HotpotQA supporting-fact labels, reported as
precision/recall — including its false-confidence rate, which is the number that
matters for the loop.

### Phase 8 — Self-correction loop
On verification failure, reformulate and re-retrieve, under a hop/iteration budget.
**Acceptance:** measurable gain over Phase 6 on the same questions; terminates within
budget on every question; no infinite loops.

### Phase 9 — Grounded answer generation + final evaluation
Answers constrained to verified evidence, with citations.
**Acceptance:** full DEV results table across all phases; one TEST run; every answer
traceable to the evidence that produced it.

### Phase 10 — Generalization
2WikiMultihopQA and MuSiQue.
**Acceptance:** results reported without re-tuning on the new datasets.

## Evaluation protocol

- Fixed seeds; runs reproducible from config + seed.
- Every phase appends a row to a results table: metrics, LLM calls, wall-clock, config.
- Ablations are run with a single variable changed. A gain reported without an ablation
  is not a result.
- Negative results are recorded, not deleted. A stage that does not help gets dropped,
  and the plan says why.

## Constraints

- No LangChain, LangGraph, or agent frameworks.
- No graph database, no Streamlit, no serving infrastructure.
- Dependencies added only with approval, one phase at a time.
- Simplest implementation that solves the problem.

## Non-goals

Serving layer, API, UI, production hardening, multi-user concerns, latency
optimization beyond honest measurement.

## Open questions

- Which LLM backs verification and generation, and whether one model does both —
  deferred to Phase 7.
- Whether hop count is fixed or dynamically decided — deferred to Phase 6.
- Correction budget (max iterations) — deferred to Phase 8.

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-08 | Python 3.12, venv recreated | 3.14 lacks wheels for the ML stack |
| 2026-09-08 | HotpotQA distractor first | Self-contained candidate sets + sentence-level supporting facts |
| 2026-09-08 | No `configs/` or `scripts/` yet | Added when a real need appears, not speculatively |
| 2026-09-08 | Eval harness before retrieval (Phase 1 before 2) | Every later phase must be measurable against a baseline |
| 2026-09-08 | No frameworks (LangChain et al.) | The orchestration is the research contribution; a framework hides it |
| 2026-09-08 | Hand-written BM25, no `rank_bm25` | Corpus is 10 paragraphs per question, so no library performance argument applies; keeps the IDF choice explicit |
| 2026-09-08 | IDF = `log(1 + (N-df+0.5)/(df+0.5))` | The textbook Okapi form goes negative for any term in >half of a 10-document corpus, penalising a paragraph for containing a query word |
| 2026-09-08 | Index `title + body` | 65.3% of questions name a gold title verbatim; body-only is kept as a clean future ablation |
| 2026-09-08 | `both@2` is the headline retrieval metric | A multi-hop question is unanswerable from one of its two gold paragraphs |
| 2026-09-08 | Paragraph-level retrieval metrics kept separate from the harness | The harness scores sentences; conflating the two levels would blur both |
| 2026-09-08 | Dense model `BAAI/bge-small-en-v1.5` | 512-token context (only 0.15% of paragraphs truncated vs 4.0% for MiniLM's 256), asymmetric-retrieval training, 33M params runs comfortably on an 8 GB M3 |
| 2026-09-08 | NumPy dot product, no FAISS | ~10 candidates per question; an ANN index would be pure ceremony |
| 2026-09-08 | Device `mps` | Measured 2.5x faster than CPU (202 vs 81 texts/sec) with byte-identical top-1/2/3 rankings on 100 real questions |
| 2026-09-08 | SHA1-keyed `.npz` embedding cache | A DEV sweep costs ~15 min uncached; identity is the exact indexed text, so ablations cannot collide. 120 MB, gitignored, rebuilt not committed |
| 2026-09-08 | Embeddings L2-normalised at encode time | Makes cosine a dot product, and bounds scores in [-1,1] for Phase 4 fusion |
| 2026-09-08 | RRF over complete rankings, not truncated lists | Both retrievers already score all ~10 candidates; truncation adds a hyperparameter and creates a missing-document case that otherwise cannot arise |
| 2026-09-08 | `rrf_k` swept, not fixed at the published 60 | Over 10 candidates, k=60 spans weights of only 1/61–1/70 and flattens RRF into "average rank"; DEV curve peaks at 3 |
| 2026-09-08 | Hybrid NOT adopted as default; dense retained | Unweighted RRF loses 0.554 → 0.507; best weighted variant gains only +0.017 for two tuned hyperparameters and a comparison-question regression |
| 2026-09-08 | Batch composition perturbs embeddings at ~1e-7 | Same input in batches of 32 vs 10 differs slightly (padding changes float reduction order). Cannot reorder top-k, but reproducibility holds to ~1e-7, not bitwise |
