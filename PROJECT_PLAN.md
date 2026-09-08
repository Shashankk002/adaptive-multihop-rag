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

### Phase 3 — Dense retrieval
Embedding-based retrieval behind the same interface as BM25.
**Acceptance:** recall@k reported next to BM25; the two are swappable without changes
to calling code.

### Phase 4 — Hybrid retrieval
Score fusion of sparse and dense.
**Acceptance:** beats the better of its two components on recall@k, or the negative
result is recorded and hybrid is dropped.

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

- Embedding model for Phase 3 — deferred to Phase 3.
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
