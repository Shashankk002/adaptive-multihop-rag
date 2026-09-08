# Self-Correcting Multi-Hop RAG

An experimental multi-hop retrieval-augmented generation pipeline that verifies its own
evidence and re-queries when that evidence is insufficient, instead of answering anyway.

Pipeline stages:

```
data → retrieval (BM25 / dense / hybrid) → reranking → multi-hop
     → evidence verification → self-correction → grounded answer
```

Initial benchmark: **HotpotQA (distractor)**.

## Status

Phase 0 — scaffold only. No components implemented yet.

## Setup

```bash
.venv/bin/python -m pytest
```

Python 3.12. Dependencies are added phase by phase; none are installed yet.

## Documents

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — phases, acceptance checks, decisions log
- [CLAUDE.md](CLAUDE.md) — working rules for this repository
