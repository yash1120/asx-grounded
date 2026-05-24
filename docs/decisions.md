# Architecture Decision Records

Short-form ADRs. Each captures a decision, alternatives considered, and the trade-off accepted.

---

## ADR-0001 — Use Qdrant (managed free tier) over pgvector

**Status:** accepted

**Context.** We need a vector store that supports metadata filtering, cosine similarity, and is cheap-to-free at the scale of ASX 50 × 12 months (~50k chunks).

**Decision.** Use Qdrant Cloud (free tier).

**Alternatives.** pgvector on the same Supabase Postgres; Pinecone (cost); Chroma (operational complexity in production).

**Trade-offs.** Two services to manage (Qdrant + Postgres) vs. one. Worth it for Qdrant's filter performance and payload size limits, which `retrieval.hybrid.py` relies on for `released_at` range filters.

---

## ADR-0002 — Hybrid BM25 + vector with RRF, plus a cross-encoder reranker

**Status:** accepted

**Context.** ASX announcement language includes regulatory phrases ("substantial holder", "Appendix 3X") that a bi-encoder embedding under-weights, and proper nouns (ASX codes) that BM25 captures exactly. Either alone leaves recall on the table.

**Decision.** Run both in parallel, fuse with Reciprocal Rank Fusion (k=60), rerank the top-50 with `bge-reranker-large` to top-8.

**Alternatives.** Vector-only (cheaper, lower recall on regulatory queries); weighted sum (parameter to tune, more brittle); ColBERT (operationally heavier).

**Trade-offs.** ~150 ms extra latency per query for the reranker. Acceptable given the recall+precision lift; measured in `docs/eval-methodology.md`.

---

## ADR-0003 — Claude Sonnet 4.6 for serving, Opus 4.7 only for eval-as-judge

**Status:** accepted

**Context.** We want strong citation-following at serving time and rigorous judging at eval time. The serving path runs on every user query; the judge runs ~80 times per eval run.

**Decision.** `claude-sonnet-4-6` for `agent/generate.py`; `claude-opus-4-7` for `eval/judge.py`.

**Alternatives.** All-Opus (5× more expensive per query, marginal lift on citation tasks); all-Sonnet for grading (the judge would share the same blind spots as the generator).

**Trade-offs.** Two model dependencies in the stack. Worth it for the cost ratio and the methodological cleanliness of a different model grading.

---

## ADR-0004 — Hard refusal, no graceful fallback to training-data answers

**Status:** accepted

**Context.** The differentiator of this project is grounding. If the model silently fills gaps from training data, we lose the property the entire pitch hinges on.

**Decision.** The system prompt requires either (a) every claim cited from the provided context, or (b) `REFUSE: <reason>`. The eval explicitly grades refusal calibration on a stratified subset including out-of-corpus and adversarial questions.

**Alternatives.** Soft fallback ("I don't have specific information on X, but generally..."); confidence-thresholded answers.

**Trade-offs.** Higher refusal rate on broad questions. We accept this — refusing well is the senior engineering behaviour. The eval distinguishes correct refusals from over-refusals.

---

## ADR-0005 — Citation verification is regex-first, LLM-second (opt-in)

**Status:** accepted

**Context.** Citations need two things checked: (a) the cited id exists in the retrieved set, and (b) the cited text actually supports the claim. The first is cheap and deterministic; the second costs an LLM call per claim.

**Decision.** Always run the regex extractor + retrieved-set membership check. Run the per-claim LLM entailment check on `verify_with_llm=True` only — used by the eval, not by every user query.

**Alternatives.** Always-on LLM verification (cost); never verify (defeats the purpose).

**Trade-offs.** Production answers can contain claims that *cite a real chunk* but mis-summarise it. The eval catches these and the failure-cases section of the blog post will surface them.

---

## ADR-0006 — Pre-bake embedding models into the Docker image

**Status:** accepted

**Context.** Fly.io spins machines to zero between requests. A cold start that downloads `bge-large-en-v1.5` (~1.3 GB) is unacceptable demo UX.

**Decision.** `Dockerfile` runs `SentenceTransformer(...)` and `CrossEncoder(...)` at build time to bake the weights into the image layer.

**Alternatives.** Mount a persistent volume; switch to a smaller embedding model (recall drop).

**Trade-offs.** Image is ~3 GB. Fly's free tier handles it; first-deploy push is slow.
