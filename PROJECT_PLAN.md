# PROJECT_PLAN.md — final project record

**Adaptive Multi-Hop RAG with Evidence Verification.** This was a development plan; it
is now the record of what was built, what was measured, and what was rejected. Every
phase is closed.

*(The project's earlier working name was "Self-Correcting Multi-Hop RAG". It was renamed
once Phase 8's mechanism was measured: the system verifies evidence sufficiency and
expands the evidence set, and does not perform reasoning-level self-correction. Phase
headings below keep their original numbering.)*

## Final architecture

```
HotpotQA per-question candidate set (~10 paragraphs)
      ↓
dense retrieval
      ↓
adaptive multi-hop routing          second hop when m23 < 0.020
      ↓
top-2 evidence
      ↓
evidence sufficiency verification   LLM: SUFFICIENT / INSUFFICIENT
      ↓
if sufficient ──────────────────┐
                                ↓
if insufficient → expand to top-5   (selective evidence expansion)
                                ↓
                         answer generation
```

The corpus is the HotpotQA per-question candidate set, not a document store.

## Outcome

A retrieval pipeline for HotpotQA that verifies whether its retrieved evidence is
sufficient and widens the evidence set when it is not. Two LLM calls per question.

**Held-out TEST — 150 questions, read once, nothing changed afterward:**

| metric | value |
|---|---|
| Answer EM | **0.620** |
| Answer F1 | **0.732** |
| bridge EM / F1 | 0.520 / 0.622 |
| comparison EM / F1 | 0.720 / 0.843 |
| supporting-fact precision / recall / F1 | 0.317 / 0.930 / 0.442 |
| abstentions | 21/150 (14.0%) |

150/150 completed, 0 failures, 282 new API calls + 18 cache-served, ~3.7 s/question.
TEST split hash `05d02a274c9f13912d5c47cd1690e486384e3db5d95dc3052d5f3d5d9c7ef643`.

**Frozen configuration:** verifier `gemini-3.5-flash-lite` (prompt v1),
generator `gemini-3.1-flash-lite` (prompt v1, max_output_tokens 64), temperature 0,
seed 20260908, `ROUTING_THRESHOLD` 0.020, `BASELINE_K` 2, `CORRECTION_K` 5.

Supporting-fact precision is low by construction: the pipeline emits every sentence of
the selected paragraphs, because sentence-level selection was never built.

## Dataset and splits

HotpotQA dev (distractor), 7,405 questions. Chosen because each question ships its own
10-paragraph candidate set (2 gold, 8 distractors) and sentence-level supporting facts,
so the retrieval problem is well-posed without a full-Wikipedia index.

Two structural facts shaped every later decision: every question has **exactly 2 gold
paragraphs**, and **both are always among the candidates**. So `recall@10` is
identically 1.0 and useless, `both@2` (are both gold in the top 2) is the meaningful
retrieval metric, and "re-retrieval" is impossible — only re-ranking or looking deeper.

| split | n | seed | purpose |
|---|---|---|---|
| TEST | 150 | 20260908 | held out, read once at the end |
| DEV | 7,255 | — | by exclusion from TEST |
| TUNE | 1,000 | 20260910 | threshold selection (Phase 6 on) |
| DEV-EVAL | 6,255 | — | DEV minus TUNE |
| `verify_tune` | 500 | 20260911 | verifier evaluation |
| `answer_dev` | 200 | 20260912 | answer-policy development |
| SMOKE | 100 | 20260909 | wiring checks only |

All are stratified, deterministic, SHA-256 hashed, and regenerable via
`python -m amrag.splits`. TEST was never read during development: no TEST question id
appeared in either LLM cache before the final run.

## Phases

| phase | outcome |
|---|---|
| 0 Scaffold | ✅ |
| 1 Data loading + eval harness | ✅ loader reproduces official counts; metrics match `hotpot_evaluate_v1.py` on 3,600 pairs + 5,000 supporting-fact cases |
| 2 BM25 baseline | ✅ `both@2` 0.320 |
| 3 Dense retrieval | ✅ `both@2` 0.554 (adopted) |
| 4 Hybrid RRF | ❌ rejected |
| 5 Cross-encoder reranking | ❌ rejected |
| 6 Multi-hop + routing | ✅ `both@2` 0.586 on DEV-EVAL (adopted) |
| 7 Evidence verification | ✅ accuracy 0.809, INSUFFICIENT F1 0.768 (adopted) |
| 8A Selective evidence expansion | ✅ gold-pair 0.588 → 0.796 (adopted, `k=5`) |
| 8B LLM re-ranker | not built — see below |
| 9 Answer generation + TEST | ✅ EM 0.415 → 0.535 on dev; TEST EM 0.620 |
| 10 Generalization (2Wiki/MuSiQue) | not attempted |

### Phase 2–3 — retrieval

BM25 hand-written (~40 lines) rather than `rank_bm25`: the corpus is 10 paragraphs per
question, so no library performance argument applies, and the IDF choice stays visible.
The textbook Okapi IDF goes negative for any term in more than half of a 10-document
corpus, so the always-positive `log(1 + (N-df+0.5)/(df+0.5))` variant was used.
Title + body is indexed — 65.3% of questions name a gold title verbatim.

Dense retrieval (`bge-small-en-v1.5`, 384-dim, cosine over normalised vectors, no ANN
index) raised `both@2` 0.320 → 0.554. Comparison questions improved most
(0.280 → 0.885); bridge questions remained the hard case (0.329 → 0.474).

### Phase 4 — hybrid RRF ❌

Unweighted Reciprocal Rank Fusion **lost** to dense alone at every constant
(best 0.507 vs 0.554). At `rrf_k=3` it recovered 66.5% of the BM25-only slice but kept
only 54.7% of the larger dense-only slice: 777 recoveries against 1,123 regressions.
Comparison questions fell 0.885 → 0.676, because equal weighting lets a retriever
scoring 0.280 on that slice outvote one scoring 0.885. A weighted variant (dense 0.7)
reached 0.571, +0.017 over dense — real (McNemar p<0.0001) but not worth two DEV-tuned
hyperparameters. **Not adopted.**

The published RRF constant of 60 was also shown to be mis-scaled here: over 10
candidates it spans weights of only 1/61–1/70, flattening RRF into "average rank".

### Phase 5 — cross-encoder reranking ❌

`ms-marco-MiniLM-L-6-v2` made retrieval **worse at every depth** (0.554 → 0.489/0.468/
0.457 at N=3/5/10). The reranker was verified to be working — mean gold score +2.814 vs
distractor −1.799, top-scoring paragraph gold in 43/50 sampled questions — and it
*improved* recall@1 (0.454 → 0.460).

The mechanism: MS MARCO trains single-hop relevance. The model confidently promotes the
paragraph that directly matches the question, then ranks the bridge paragraph below
topically-similar distractors, because the bridge paragraph is relevant only via the
entity the first supplies. Damage grew monotonically with depth. It recovered 18–37% of
reachable failures but broke ~2× as many questions as it fixed (McNemar z 12.8–16.3),
at ~1000× the latency. **Not adopted.**

### Phase 6 — multi-hop retrieval ✅

Structural analysis first: dense retrieval already puts a gold paragraph at rank 1 for
90.9% of bridge questions while bridge `both@2` is only 0.474 — hop 1 is solved, the
entire deficit is hop 2.

**Always-on two-hop expansion netted to zero**: bridge improved 0.474 → 0.521 (+0.047)
while comparison collapsed 0.885 → 0.521 (−0.364). Comparison questions name both
entities outright (99.9% of them) and have no sequential structure; appending hop-1 text
drowns the second entity.

That motivated routing. **A hand-tuned keyword rule was rejected** in favour of `m23`,
the gap between the rank-2 and rank-3 dense scores. `m23` beat the keyword rules
(0.586 vs 0.577 and 0.583) with no linguistic assumptions, and it is not merely a
type detector: its AUC for predicting evidence sufficiency (0.776) exceeds its AUC for
separating comparison from bridge (0.717). The rank-1/rank-2 margin is *anti*-predictive
(AUC 0.456) — a large gap there means rank 2 is weak.

SMOKE (n=100) **failed** to select the threshold: its curve spanned 0.8 standard errors
and disagreed in sign with DEV. A 1,000-question TUNE split was carved out for
selection, and `t = 0.020` was frozen as the midpoint of the stable region 0.010–0.030
rather than the arithmetic peak. TUNE and DEV-EVAL then agreed to three decimals
(+0.031 both), which is the main evidence the threshold is not overfitted.

**DEV-EVAL: 0.554 → 0.586 (+0.031), McNemar p=2e-13**, routing 32.1% of questions to a
second hop — 36.5% of bridge, 13.6% of comparison, without ever seeing a question type.

### Phase 7 — evidence verification ✅

Binary SUFFICIENT / INSUFFICIENT over the question and the top-2 paragraphs, judged
jointly in one LLM call. Not three-way: HotpotQA distractors are irrelevant, not
contradictory. Judged jointly because all complete-evidence questions require facts from
*both* paragraphs — a per-paragraph verifier would call each individually insufficient.

**486/500 usable verdicts** (1 excluded for JSON truncation, 13 lost to daily quota):
accuracy **0.809**, INSUFFICIENT precision/recall/F1 **0.766 / 0.770 / 0.768**, AUC
**0.804** against **0.729** for the free `m23` signal on the same questions. The recall
gap is the substantive one: a score margin misses over half of insufficient cases.

**Confidence is not usable for routing** — 93.8% of verdicts report 1.00, and mean
confidence is 0.996 when correct versus 0.985 when wrong.

Known limitation: sufficiency is proxied by "both labelled gold paragraphs retrieved",
but HotpotQA labels one sufficient evidence path, not every one. Several scored false
positives are genuinely answerable, so 0.809 is a lower bound on true accuracy.

### Phase 8A — selective evidence expansion ✅

On INSUFFICIENT, widen the evidence from top-2 to top-5 of the ranking already held.
**This is evidence-level expansion, not query reformulation, not reasoning correction,
and not new-document retrieval** — the
candidate set is fixed and the missing paragraph is within the top 10 in 100% of flagged
cases. One round; termination is structural.

`verify_tune` (486): gold-pair recall **0.588 → 0.796 (+0.208)**, mean evidence
2.00 → 3.24 paragraphs, 101 recoveries, **0 regressions** (structural — expansion is a
superset). Expansion rate 41.4%; 23.4% of expansions were unnecessary, which is the
verifier's false-alarm rate expressed as cost. Bridge 0.515 → 0.756, comparison
0.885 → 0.958. Zero additional LLM calls.

### Phase 8B — LLM re-ranker, not built

Deferred pending Phase 9, then dropped. Its premise was that extra context would hurt
the reader, so a smarter selection would beat a bigger one. Phase 9's fixed-denominator
analysis showed extra context did **not** degrade reading accuracy, removing the
motivation. Not building it is the recorded decision, not an omission.

### Phase 9 — grounded answer generation ✅

Plain-text generation (HotpotQA answers are two words at the median, so JSON would only
add a parse-failure mode), constrained to the supplied evidence, returning the literal
string `INSUFFICIENT` when the evidence does not contain the answer.

Four evidence policies over the frozen 200-question `answer_dev` split, 485 unique
generations, 0 failures. Content-keyed caching meant policies sharing evidence shared a
generation: 485 calls instead of 800.

| policy | EM | F1 | mean evidence | abstention |
|---|---|---|---|---|
| baseline top-2 | 0.415 | 0.498 | 2.00 | 0.380 |
| k=3 | 0.505 | 0.597 | 2.48 | 0.275 |
| k=4 | 0.510 | 0.619 | 2.95 | 0.230 |
| **k=5** | **0.535** | **0.659** | 3.42 | 0.180 |

Paired analysis: **almost all of the gain is reduced abstention, not corrected answers**
— across the whole range only 1–2 questions moved from a wrong answer to a right one.
`baseline vs k=5` is significant (McNemar p<0.0001); no step *between* expansion levels
is individually significant on EM.

`CORRECTION_K = 5` was confirmed over `k=3` on **F1** (+0.061, 95% CI [+0.021, +0.103],
p=0.0033); EM could not separate them (+0.030, p=0.13), partly because EM scores
"Carbon County" for "Carbon" as a miss. A fixed-denominator analysis — the 146 questions
whose answer was already in the baseline top-2 — showed reading accuracy flat to
slightly *rising* with more context (0.541 → 0.568) and abstention falling, so the
apparent grounded-EM peak at k=3 was a denominator-composition artifact, not distraction.

Answers produced without the gold answer being present in the evidence: 0.5–2.0%, a
small but non-zero measure of pretraining leakage.

## Methodological notes

- Every threshold and constant was frozen on a development split before evaluation.
- Negative results are kept, not deleted; `hybrid.py` and `rerank.py` remain so the
  rejected approaches stay reproducible.
- LLM calls sit behind seams with fake implementations; the full test suite runs
  offline with no API key.
- Verdicts and answers are cached on disk keyed by model + prompt version + exact
  input, so runs are resumable and results can never mix across models. Gemini model
  ids are moving aliases and cannot be pinned to a revision — the cache is the
  reproducibility mechanism.

## Limitations

- TEST is one 150-question sample (SE on EM ≈ ±0.04).
- Development results are not held out; `CORRECTION_K`, the routing threshold and the
  evidence policy were all selected on development splits.
- Supporting-fact precision is low by design; no sentence-level selection exists.
- The sufficiency proxy understates the verifier (see Phase 7).
- Two Gemini models were used, forced by free-tier daily quotas rather than chosen.
- HotpotQA only; the routing threshold is calibrated to this retriever's score scale.

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
| 2026-09-09 | Correction is evidence expansion, not re-retrieval | The candidate set is fixed per question and always contains both gold paragraphs; on flagged questions the missing paragraph is within the top 10 in 100% of cases |
| 2026-09-09 | One correction round, no loop | With a fixed candidate set, a second round reaches nothing a larger k does not; termination is structural |
| 2026-09-09 | CORRECTION_K = 5, frozen on TUNE | +0.208 gold-pair over baseline at 3.24 mean paragraphs; re-confirmed in Phase 9 against answer F1 (+0.061 over k=3) |
| 2026-09-09 | LLM re-ranker (Phase 8B) not built | Its premise was that extra context hurts the reader; Phase 9's fixed-denominator analysis showed it does not |
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
