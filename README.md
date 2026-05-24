# asx-grounded

A grounded Q&A agent over ASX continuous-disclosure announcements with hard citation enforcement and a public hallucination evaluation scoreboard.

> Most RAG demos show one cherry-picked answer. This one ships with numbers.

---

## What this is

A small, opinionated retrieval-augmented Q&A system over a real-world Australian corpus — ASX continuous-disclosure announcements from the top-50 listed companies. You ask a natural-language question, the system returns an answer where **every factual claim links to a specific paragraph in a specific ASX announcement**. If the corpus doesn't support an answer, it refuses instead of guessing.

Alongside the running demo, the repo publishes a continuously-updated scoreboard showing how often the system is factually correct, how often it hallucinates, and how well it refuses unanswerable questions — all graded by an independent stronger model on a stratified, hand-labelled test set.

## The idea

Most RAG demos are vibes. You see a slick chatbot, a confident answer, no way to tell if it's true. The hard parts — citation grounding, refusal calibration, hallucination measurement — get hand-waved. Hiring managers and customers in regulated industries (banking, insurance, super funds, gov) all want the same thing: *evidence* that the LLM isn't making things up. Almost nobody ships that evidence.

`asx-grounded` is built around the opposite default. Every design decision optimises for two questions:

1. **Can the system prove its answers?** Citations are mechanical, machine-checkable, and refused-by-default.
2. **Can we measure how often it's wrong?** A separate eval harness with a stronger judge model produces real numbers, published openly — including the failure cases.

The corpus is ASX announcements specifically because (a) they're public, (b) they're messy and realistic, and (c) they're the kind of data every Big-4 bank, fintech, super fund, and short-selling shop in Australia spends real money trying to index. The technique generalises to any high-stakes corpus; the choice of dataset is the AUS-market anchor.

## The approach

Five design choices distinguish this from a generic RAG demo.

### 1. Hybrid retrieval, not vector-only

ASX announcement language is half regulatory boilerplate ("Appendix 4D", "substantial holder notice") and half free-form business prose. A pure embedding-based retriever under-weights the regulatory phrases; a pure BM25 retriever misses paraphrases.

`asx-grounded` runs both in parallel, fuses with **Reciprocal Rank Fusion (k=60)**, then reranks the top-50 candidates with a **BGE cross-encoder** to keep the top-8. The reranker is a one-time ~150ms cost that materially improves precision; the impact is measured in the eval.

### 2. Citation enforcement, not citation suggestion

The system prompt requires `[chunk_id]` after every factual claim. A regex-based verifier then strips any citation referring to a `chunk_id` the model didn't actually receive — i.e. fabricated citations are mechanically caught before the answer is returned.

For the eval (and optionally at serving time), a second-pass LLM check verifies that each cited chunk *actually supports* the claim it's attached to, not just that the id exists.

### 3. Hard refusal, not graceful fallback

Most LLMs gracefully fall back to training-data knowledge when retrieved context is thin. That's a feature for chatbots and a bug for grounded QA. The system prompt forbids it explicitly:

> If the provided context does NOT contain enough information to answer confidently, respond with EXACTLY: `REFUSE: <one short sentence>`. Do not guess. Do not use general knowledge.

Refusal calibration is then a measured metric — both *correct refusal* on unanswerable questions and *false refusal* on answerable ones are tracked, so the system can't game the score by refusing everything.

### 4. Stratified eval with a stronger judge

The test set is stratified across five categories — `answerable`, `unanswerable`, `time_bounded`, `comparative`, `adversarial` — with hand-labelled expected behaviour for each. **Claude Sonnet 4.6 serves; Claude Opus 4.7 judges.** Running a stronger model as the grader reduces the risk that the judge shares the generator's blind spots, and the eval set is small enough (~80 questions) that the cost stays under a couple of dollars per run.

The judge returns six structured fields per question: factual correctness, citation accuracy, citation recall, refusal correctness, format compliance, and a free-text reasoning trace. These are aggregated into the public scoreboard at `/eval`.

### 5. Eval-on-PR as a regression gate

Every pull request labelled `run-eval` triggers a full eval run in CI. If the hallucination rate regresses by more than 2 percentage points against the baseline, the PR is blocked. This is the engineering discipline that separates "ran an eval once" from "operates a measured system."

## Status

- [ ] W1 — Ingest the ASX 50 corpus
- [ ] W2 — Hybrid retrieval + grounded generation
- [ ] W3 — Eval harness + scoreboard
- [ ] W4 — UI, deploy, broadcast

See [plan.md](plan.md) for the 4-weekend build plan and [BACKLOG.md](BACKLOG.md) for what's deliberately out of v1 scope.

## Quickstart

```bash
# 1. Install
uv sync   # or: pip install -e ".[dev]"

# 2. Configure
cp .env.example .env   # fill in ANTHROPIC_API_KEY, QDRANT_URL, etc.

# 3. Ingest a small sample (3 companies, last 30 days)
python -m asx_grounded.ingestion.fetch_asx --codes CBA,BHP,WBC --days 30
python -m asx_grounded.ingestion.embed

# 4. Run the API
uvicorn asx_grounded.api.main:app --reload --port 8000

# 5. Run the eval
python -m asx_grounded.eval.run_eval

# 6. (Optional) UI
cd web && npm install && npm run dev
```

Tests run without network access: `pytest -q`.

## Architecture

```
ASX announcements → ingestion → Qdrant + Postgres
                                    │
user query → metadata filter (regex + LLM fallback)
           → hybrid retrieval (BM25 + vector + RRF)
           → cross-encoder rerank
                                    │
                          Claude Sonnet 4.6 (citation-forced)
                                    │
                          citation verifier (drop fabricated, optionally entail-check)
                                    │
                              FastAPI ── Next.js UI
                                    │
                         Eval harness (Opus 4.7 as judge)
                                    │
                              public scoreboard (/eval)
```

| Layer | Module | Notes |
|---|---|---|
| Data | [`ingestion/`](src/asx_grounded/ingestion) | httpx + tenacity for fetch; pdfplumber for parse; tiktoken-aware chunker |
| Retrieval | [`retrieval/`](src/asx_grounded/retrieval) | BM25 (`rank_bm25`) + Qdrant + BGE reranker; RRF fusion |
| Agent | [`agent/`](src/asx_grounded/agent) | Citation-forced prompt; regex + optional LLM citation verifier |
| API | [`api/main.py`](src/asx_grounded/api/main.py) | FastAPI; loads retriever + reranker at startup |
| Eval | [`eval/`](src/asx_grounded/eval) | Opus 4.7 judge; stratified scoreboard JSON |
| UI | [`web/`](web) | Next.js 14 + Tailwind; renders citation pills + scoreboard |

See [docs/architecture.md](docs/architecture.md) for the full module contracts and [docs/decisions.md](docs/decisions.md) for stack ADRs.

## Eval methodology

The differentiator. Six metrics on a stratified, hand-labelled test set, graded by Claude Opus 4.7. The seed test set ships in [src/asx_grounded/eval/testset.jsonl](src/asx_grounded/eval/testset.jsonl); launch target is 80 questions across five categories. CI runs the eval on labelled PRs and blocks regressions.

Full methodology, including known limitations and failure-mode analysis, in [docs/eval-methodology.md](docs/eval-methodology.md) — this doubles as the launch blog post.

## Why this project exists (the honest version)

I'm a Sydney-based ML engineer applying for full-time roles at AUS banks, fintechs, and consultancies. Most candidates show RAG demos with no measurement story — a chatbot wrapped around their PDFs, no eval, no failure cases, no refusal behaviour. The two things I keep hearing from hiring managers in regulated industries are: *"how do you know it doesn't hallucinate"* and *"can you prove your answers."* This project is my answer.

If you're hiring for AI/ML engineering in Australia and want to see the engineering reasoning behind any specific design choice, the ADRs in [docs/decisions.md](docs/decisions.md) are written for you.

## Disclaimer

ASX announcements are public. This project stores only embeddings and short excerpts; every cited answer deep-links to the official ASX page for the source announcement. Not affiliated with ASX Limited. **Not financial advice.**

## License

MIT — see [LICENSE](LICENSE).
