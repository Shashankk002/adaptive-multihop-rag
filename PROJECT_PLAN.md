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

### Phase 5 — Reranking ❌ negative result; reranker not adopted
Cross-encoder reranking of the dense shortlist.
**Acceptance:** precision@k improves at fixed recall; added latency measured and
reported.

**Setup:** `cross-encoder/ms-marco-MiniLM-L-6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a`,
22.7M params, 512 context, device `mps`, raw question (no bge prefix) paired with the
same `title + body` text. Reranked top-N placed above the untouched dense tail, so
recall@5 stays defined at N=3.

**A cross-encoder makes multi-hop retrieval worse at every depth (DEV n=7255):**

| retriever | recall@1 | recall@2 | recall@3 | recall@5 | both@2 | vs dense |
|---|---|---|---|---|---|---|
| Dense | 0.454 | 0.762 | 0.855 | 0.926 | **0.554** | — |
| Rerank N=3 | 0.460 | 0.729 | 0.855 | 0.926 | 0.489 | −0.065 |
| Rerank N=5 | 0.459 | 0.719 | 0.803 | 0.926 | 0.468 | −0.087 |
| Rerank N=10 | 0.459 | 0.713 | 0.792 | 0.866 | 0.457 | −0.098 |

**The reranker is working correctly** — mean gold score +2.814 vs distractor −1.799,
and the top-scoring paragraph is gold in 43/50 sampled questions. It also improves
**recall@1** (0.454 → 0.460): it is *better* than dense at identifying the single most
relevant paragraph. It is worse at everything that requires both.

**Why: MS MARCO trains single-hop relevance.** The cross-encoder confidently promotes
the paragraph that directly matches the question, then ranks the second (bridge)
paragraph below topically-similar distractors, because the bridge paragraph is not
relevant to the question text — it is relevant only via the entity the first paragraph
supplies. Deeper shortlists give it more distractors to mis-promote, so the damage
grows monotonically with N.

Case A / Case B, and recovery vs regression:

| N | Case B | conditional recovery | Case A | recoveries | regressions | reg/rec | net |
|---|---|---|---|---|---|---|---|
| 3 | 1232 | 36.6% | 2002 | 451 | 925 | 2.05 | −474 |
| 5 | 2190 | 25.8% | 1044 | 564 | 1193 | 2.12 | −629 |
| 10 | 3234 | 18.0% | 0 | 583 | 1291 | 2.21 | −708 |

Reranking genuinely recovers 18–37% of reachable failures, but breaks roughly twice as
many questions it was already winning (McNemar z = 12.8–16.3, all p ≪ 0.001). It
changes the dense top-2 on 32–46% of questions and moves gold paragraphs **down** more
often than up (N=5: 2266 up, 3263 down). Comparison questions suffer most
(0.885 → 0.739 at N=5), mirroring the Phase 4 failure mode.

**Cost:** 72,208 pairs in 8.8 min on MPS (136 pairs/sec) — roughly 73 ms/question live,
against dense's 0.07 ms cache-served. About **1000x the latency for a negative result.**

**Decision: do not adopt the cross-encoder.** `rerank.py` is retained; a reranker may
help in Phase 6 once queries are decomposed into single-hop sub-questions, which is
the shape this model was actually trained for.

### Phase 6 — Multi-hop retrieval ✅ adopted
Two-hop retrieval with evidence-based routing.
**Acceptance:** on questions whose second gold paragraph is unreachable from the
original query, gold-pair recall improves over single-pass.

**Hypothesis.** Dense retrieval finds *a* gold paragraph for 90.9% of bridge questions
at rank 1, while bridge `both@2` is only 0.474 — so hop 1 is solved and the entire
deficit is hop 2. Adding hop-1 text to the query should supply the context that makes
the second paragraph reachable. Structural analysis of DEV supports this: 47.2% of
bridge questions name the second paragraph inside the first's supporting sentences,
73.9% have a one-directional link, and only 3.0% are order-ambiguous. Every question
has exactly 2 gold paragraphs, so **two hops suffice by construction** — no loop or
termination rule is needed.

**Unrouted 2-hop (DEV n=7255) nets to nothing — because it helps and harms equally:**

| retriever | recall@1 | recall@2 | recall@3 | recall@5 | both@2 | vs dense |
|---|---|---|---|---|---|---|
| Dense | 0.454 | 0.762 | 0.855 | 0.926 | 0.554 | — |
| 2-hop (title) | 0.454 | 0.760 | 0.849 | 0.924 | 0.555 | +0.001 |
| 2-hop (paragraph) | 0.454 | 0.737 | 0.828 | 0.908 | 0.521 | −0.033 |
| 2-hop (sentence) | 0.454 | 0.747 | 0.835 | 0.911 | 0.540 | −0.014 |

The aggregate hides two opposing effects. With paragraph expansion, **bridge improves
0.474 → 0.521 (+0.047) while comparison collapses 0.885 → 0.521 (−0.364)**. Comparison
questions name both entities outright (99.9% of them) and have no sequential structure;
stuffing hop-1 text into the query drowns the second entity.

**Routing signal: `m23` = dense score(rank 2) − score(rank 3).** A wide gap means dense
has cleanly separated a pair and hop 2 has nothing to add. AUC 0.776 for predicting
"dense top-2 already holds both gold" (0.891 on comparison). Notably the **rank-1/rank-2
margin is useless (AUC 0.456) — mildly anti-predictive**: a large gap there means rank 2
is weak relative to rank 1. `m23` beat both hand-tuned keyword rules (0.586 vs 0.577 and
0.583) while using no linguistic knowledge, and it is not merely a type detector — its
AUC for separating comparison from bridge (0.717) is *lower* than for predicting dense
sufficiency.

**Threshold selection.** SMOKE (n=100) **failed** to select a threshold: its whole curve
spanned 0.8 standard errors, the "peak" was one question, the apparent plateau was
non-contiguous, and it disagreed in sign with DEV at t=0.020 and t=0.040. A **TUNE split**
(1000 ids, seed 20260910, stratified 805 bridge / 195 comparison) was carved from DEV for
selection, leaving **DEV-EVAL** (6255) for reporting. TUNE gave a smooth curve with a
contiguous stable region of 0.010–0.030; **t = 0.020 was frozen as its midpoint**, not
the arithmetic peak at 0.025 (which was better by 0.002 — one-eighth of a standard error).

**DEV-EVAL result (n=6255), threshold frozen, not retuned:**

| system | recall@1 | recall@2 | recall@3 | recall@5 | both@2 | vs dense | ms/q |
|---|---|---|---|---|---|---|---|
| Dense | 0.455 | 0.762 | 0.855 | 0.925 | 0.554 | — | 0.04 |
| 2-hop everywhere | 0.455 | 0.737 | 0.828 | 0.909 | 0.521 | −0.033 | 11.05 |
| **Routed (m23<0.020)** | 0.455 | **0.773** | 0.851 | 0.916 | **0.586** | **+0.031** | 3.66 |

By type: bridge 0.474 → **0.527**, comparison 0.885 → 0.831. 454 recoveries vs 257
regressions (reg/rec 0.57), net +197 questions, **McNemar z = 7.35, p = 2e-13**.
Routes 32.1% of questions to hop 2 — 36.5% of bridge, 13.6% of comparison — without ever
seeing a question type. TUNE and DEV-EVAL agree closely (+0.031 vs +0.031; 32.5% vs
32.1% routed), which is the main evidence the threshold is not overfitted.

**Limitations.**
- **DEV-EVAL is not a pristine held-out estimate for the full model-selection process.**
  An earlier exploratory analysis swept `m23` on all of DEV, so the choice to pursue this
  signal was informed by data that includes DEV-EVAL. The TUNE/DEV-EVAL split cleanly
  isolates the *threshold* choice only. TEST remains untouched and is the clean estimate.
- The threshold 0.020 is calibrated to `bge-small` cosine scores over ~10 candidates. The
  *signal* should transfer to 2Wiki/MuSiQue; the *constant* will not and needs
  recalibration per dataset.
- `m23` assumes exactly 2 gold paragraphs. MuSiQue's 3–4 hop questions need a cliff after
  rank *h*, and choosing *h* is itself unsolved. This is a 2-hop-specific instance of a
  general idea.
- Routing still costs comparison questions 0.054 (0.885 → 0.831); the router is right
  about type roughly 86% of the time, not always.
- Per-question oracle routing would reach 0.678, so 0.586 captures about 26% of the
  available headroom.

**Decision: adopt routed 2-hop retrieval as the default retriever.** First gain since
Phase 3, and the first phase whose mechanism was predicted in advance and confirmed.

### Phase 7 — Evidence verification ✅ adopted
Judge whether the retrieved evidence is sufficient to answer the question.
**Acceptance:** verifier agreement against HotpotQA supporting-fact labels, reported as
precision/recall — including its false-confidence rate, which is the number that
matters for the loop.

**Formulation.** Binary SUFFICIENT / INSUFFICIENT over the question plus the top-2
paragraphs from routed retrieval, judged jointly in one LLM call. Not three-way:
HotpotQA distractors are irrelevant, not contradictory, so REFUTED has no population.
Not answer-verification either — that needs an answer generator, which is Phase 9.
Paragraphs are judged together because all 3664 complete-evidence questions in DEV-EVAL
need facts from *both* gold paragraphs; a per-paragraph verifier would call each one
individually insufficient every time.

**Setup.** Gemini `gemini-3.5-flash-lite` via `google-genai`, temperature 0, seed
20260908, `max_output_tokens=256`, no thinking configuration, structured JSON output
against a fixed schema (`reason` first, so a short justification precedes the verdict).
Prompt version `v1`, frozen. Verdicts cached append-only in
`data/processed/verify_cache.jsonl`, keyed by `sha1(model + prompt_version + prompt)` —
the model is in the key, so verdicts can never be reused across models.

**Evaluation split.** A frozen 500-question subset of TUNE
(`verify_tune_ids.json`, sha256 `dd0a1dd8…`, seed 20260911), stratified 403 bridge /
97 comparison to preserve TUNE's ratio. Selection is independent of any verifier output.
**Coverage: 486/500 (97.2%)** — 1 excluded (JSON truncation, permanent at temperature 0)
and 13 unevaluated (daily quota). The run was stopped at the quota rather than resumed.

**Results (n=486):**

| | gold INSUFFICIENT | gold SUFFICIENT |
|---|---|---|
| verifier INSUFFICIENT | 154 | 47 |
| verifier SUFFICIENT | 46 | 239 |

**Accuracy 0.809.** INSUFFICIENT precision 0.766 / recall 0.770 / **F1 0.768**.
False SUFFICIENT (missed insufficiency) 46/200 = **0.230**; false INSUFFICIENT
47/286 = 0.164. Errors are near-symmetric, and the verifier's SUFFICIENT rate (58.6%)
tracks gold (58.8%).

**Against the free `m23` baseline, recomputed on the same 486 questions:**

| | verifier | m23 |
|---|---|---|
| AUC | **0.804** | 0.729 |
| precision | **0.766** | 0.632 |
| recall | **0.770** | 0.490 |
| F1 | **0.768** | 0.552 |

The recall gap is the substantive one: a score margin misses over half of insufficient
cases; the verifier catches 77%.

**Confidence is not usable for routing.** 93.8% of verdicts report 1.00 and 96.1% are
≥0.95; mean confidence is 0.996 when correct and 0.985 when wrong. The *ranking* still
carries signal (AUC 0.804), almost all of it from the ~4% below 1.00 — but the values
cannot be thresholded. Phase 8 must use the binary verdict.

**Cost:** 1.004 calls/question, mean latency **1.42 s** — roughly 390x routed
retrieval's 3.66 ms, and now the dominant cost in the pipeline. 2 failures in 475 calls
(0.4%): one JSON truncation, one daily-quota stop.

**By type:** bridge n=390 accuracy 0.828, false-SUFFICIENT rate 0.238; comparison n=96
accuracy 0.729, false-SUFFICIENT rate 0.091. Bridge questions are over-approved,
comparison questions over-flagged.

**Known limitation — the gold proxy is a lower bound on the verifier.** Sufficiency is
proxied by "both HotpotQA-labelled gold paragraphs were retrieved", but HotpotQA labels
*one* sufficient evidence path, not every one. Several scored false-SUFFICIENT cases are
genuinely answerable from the retrieved text (e.g. Sissy Spacek + Thomas Rickman both
naming "Coal Miner's Daughter"). **The measured 0.230 false-positive rate is therefore
an upper bound on true error, and 0.809 a lower bound on true accuracy.** Two different
models produced the same pattern independently, which points to a labelling artifact
rather than a model quirk.

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

- Which LLM backs answer generation — deferred to Phase 9. Verification uses Gemini
  Flash Lite; the free tier's 500 requests/day is the binding constraint on scale.
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
| 2026-09-09 | Verification is binary SUFFICIENT/INSUFFICIENT | Distractors are irrelevant, not contradictory, so REFUTED has no population; answer-verification would require Phase 9's generator |
| 2026-09-09 | Gemini Flash Lite, not Anthropic or a local model | Zero budget. `gemini-2.5-flash` is closed to new keys and `gemini-3.6-flash` allows only 20 requests/day; Flash Lite gives 15 RPM / 500 RPD |
| 2026-09-09 | Verdict cache is the reproducibility mechanism | Gemini model ids are moving aliases and cannot be pinned to a revision the way `bge-small` was |
| 2026-09-09 | Frozen 500-question verifier split, stratified | A prefix of TUNE would have shifted the bridge/comparison ratio to 77.8%; the stratified draw holds 80.6% |
| 2026-09-09 | Confidence NOT used for routing | 93.8% of verdicts report 1.00; correct-vs-wrong confidence differs by 0.011 |
| 2026-09-09 | Verifier adopted over m23 for Phase 8 | +0.216 F1 and +0.280 recall on the same 486 questions, at 1.42 s/question |
| 2026-09-08 | Routed 2-hop retrieval adopted as default | +0.031 both@2 on DEV-EVAL (0.554 → 0.586), McNemar p=2e-13, at 3.66 ms/q |
| 2026-09-08 | Routing on `m23`, not keyword rules | Beats hand-tuned keyword rules (0.586 vs 0.577/0.583) with no linguistic assumptions, and should transfer to datasets without a bridge/comparison dichotomy |
| 2026-09-08 | TUNE split (1000 ids) carved from DEV | SMOKE's 100 questions cannot resolve a 0.03 effect (SE 0.05); its curve disagreed in sign with DEV. DEV keeps its 7255 definition; TUNE/DEV-EVAL are Phase-6 sub-splits |
| 2026-09-08 | Threshold frozen at the stable-region midpoint | 0.020 over the peak 0.025, which was better by 0.002 — one-eighth of a standard error |
| 2026-09-08 | Exactly 2 hops, no loop or termination rule | Every question has exactly 2 gold paragraphs; an n-hop loop would solve a problem this dataset does not contain |
| 2026-09-08 | Cross-encoder reranking NOT adopted | MS MARCO single-hop relevance training actively hurts multi-hop both@2 (0.554 → 0.468 at N=5), breaking ~2x more questions than it fixes, at ~1000x the latency |
| 2026-09-08 | Reranker device `mps` | Measured 2.38x faster than CPU (317 vs 133 pairs/sec), exact repeatability, identical top-1/2/3 on 100 real questions |
| 2026-09-08 | No cross-encoder score cache | Cache key is the (query, passage) pair; every question has a distinct query, so there is no cross-question reuse to exploit |
| 2026-09-08 | Hybrid NOT adopted as default; dense retained | Unweighted RRF loses 0.554 → 0.507; best weighted variant gains only +0.017 for two tuned hyperparameters and a comparison-question regression |
| 2026-09-08 | Batch composition perturbs embeddings at ~1e-7 | Same input in batches of 32 vs 10 differs slightly (padding changes float reduction order). Cannot reorder top-k, but reproducibility holds to ~1e-7, not bitwise |
