# ASX-Grounded — Project Plan

**Codename:** `asx-grounded` (good repo name, good domain candidate)
**Mission:** A grounded Q&A agent over ASX continuous-disclosure announcements with hard citation enforcement and a public hallucination evaluation scoreboard.
**Time:** ~4 weekends of focused work + 2 weekday evenings/week.
**Budget:** ~$80–100 in API and hosting costs total. Comfortably student-affordable.

---

## 1. Definition of done

You don't ship this project until **all six** are true:

1. Live URL anyone can hit without signing in.
2. At least **50 ASX-listed companies** indexed, **12 months** of announcements.
3. Every answer cites specific documents; clicking a citation opens the actual ASX page.
4. A public **eval scoreboard** showing hallucination rate, citation accuracy, refusal calibration, p50/p95 latency, cost/query — with real numbers, not "TBD."
5. A **90-second demo video** pinned on the landing page.
6. An **engineering write-up** (blog post) explaining the eval methodology and three things that broke.

Anything less and it's a portfolio project. The above is a hireable artifact.

---

## 2. Architecture (one-glance mental model)

```
   ASX Announcements feed ─┐
                           │
                           ▼
                  ┌─────────────────┐
                  │  Ingestion      │   nightly cron
                  │  (PDF → text →  │
                  │   chunk → embed)│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Vector store   │   Qdrant
                  │  + metadata DB  │   + Postgres
                  └────────┬────────┘
                           │
        user query ───────►│
                           ▼
                  ┌─────────────────┐
                  │  Retriever      │   BM25 + vector + reranker
                  │  (hybrid +      │
                  │   metadata)     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Generator      │   Claude Sonnet 4.6
                  │  (citation-     │   strict citation prompt
                  │   forced)       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Citation       │   verify cited chunks
                  │  verifier       │   contain claims
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐              ┌──────────────┐
        │  Web UI  │              │ Eval harness │
        │ (Next.js)│              │ (offline)    │
        └──────────┘              └──────┬───────┘
                                         ▼
                                  ┌──────────────┐
                                  │  Scoreboard  │
                                  │  (public)    │
                                  └──────────────┘
```

---

## 3. Tech stack — decisions locked

Picking the stack now so you don't waste a week on FOMO.

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Your stack. |
| Ingestion | `httpx` + `pdfplumber` | PDF parsing for ASX docs. |
| Chunking | `langchain-text-splitters` recursive + sentence-aware | Boring, works. |
| Embeddings | **`BAAI/bge-large-en-v1.5`** (open) | Free, strong, runs on CPU for this scale. |
| Vector DB | **Qdrant Cloud free tier** | Generous limits, metadata filtering. |
| Metadata DB | **Postgres** (Supabase free tier) | For company, date, announcement-type filters. |
| Retriever | Hybrid: BM25 (`rank_bm25`) + vector + **BGE reranker** | Reranker is the cheap quality win. |
| LLM | **Claude Sonnet 4.6** (`claude-sonnet-4-6`) | Strong citation following, cheaper than Opus, you should mention this on CV. |
| LLM-as-judge (eval only) | **Claude Opus 4.7** (`claude-opus-4-7`) | Use the stronger model only for grading. |
| API server | **FastAPI** | You already use it. |
| UI | **Next.js + Tailwind + shadcn/ui** | Looks professional in 60 seconds. |
| Hosting (API) | **Fly.io** | Cheap, fast cold start. |
| Hosting (UI) | **Vercel** | Free, instant. |
| Auth | None on the demo, rate-limit by IP | Don't add friction. |
| Observability | **Logfire** or **Langfuse Cloud free tier** | Trace every query, makes the engineering post real. |
| Domain | `asx-grounded.app` or `.au` (~$15–30/yr) | Worth it. |

**Cost estimate:** Embeddings free (open model, CPU), Qdrant free tier, Supabase free, Vercel free, Fly ~$5/mo, Claude API ~$30 dev + ~$20 eval + ~$0.01/query in production. Domain $20. **Under $80 total to ship.**

---

## 4. Repo structure

```
asx-grounded/
├── README.md                  ← Hook + demo gif + live link first
├── docs/
│   ├── architecture.md
│   ├── eval-methodology.md    ← The blog post lives here too
│   └── decisions.md           ← ADRs for stack choices
├── ingestion/
│   ├── fetch_asx.py
│   ├── parse_pdf.py
│   ├── chunk.py
│   └── embed.py
├── retrieval/
│   ├── hybrid.py
│   ├── rerank.py
│   └── filters.py
├── agent/
│   ├── prompts.py
│   ├── generate.py
│   └── verify_citations.py
├── eval/
│   ├── testset.jsonl          ← Your 80 golden questions
│   ├── judge.py
│   ├── run_eval.py
│   └── scoreboard_data.json   ← Published to UI
├── api/
│   └── main.py                ← FastAPI app
├── web/                       ← Next.js
│   ├── app/
│   │   ├── page.tsx           ← Query interface
│   │   └── eval/page.tsx      ← Scoreboard
│   └── ...
├── tests/
├── pyproject.toml
├── Dockerfile
├── fly.toml
└── .github/workflows/
    └── ci.yml                 ← Lint + tests + eval-on-PR
```

---

## 5. Week-by-week plan

### Week 1 — Ingest the corpus *(weekend + 2 evenings)*

| Task | Deliverable |
|---|---|
| Identify scope: **ASX 50** companies, **last 12 months** of announcements. Start small. | `companies.csv` with ASX codes. |
| Build `fetch_asx.py` — pull announcement listing + PDFs. Respect robots/rate-limit. | ~5,000 PDFs on local disk + JSONL metadata. |
| Build `parse_pdf.py` — extract text, handle multi-column, drop boilerplate (cover pages, ASX template). | Clean text per announcement. |
| Build `chunk.py` — recursive splitter, 600-token chunks, 100-token overlap, **preserve announcement-id and page-num in metadata**. | `chunks.jsonl` |
| Embed with bge-large-en-v1.5 (batched, CPU is fine), push to Qdrant with metadata payloads. | Indexed vector DB. |
| Push announcement metadata (company, date, type, materiality) into Postgres. | Queryable metadata layer. |
| Add a one-liner `make ingest` and a CI test that runs ingestion against a 3-doc fixture. | Reproducibility. |

**End of week 1:** You can query the vector DB by company × date range and get back relevant chunks.

---

### Week 2 — Retrieval + grounded generation *(weekend + 2 evenings)*

| Task | Deliverable |
|---|---|
| Hybrid retriever: BM25 + vector, fuse with Reciprocal Rank Fusion, top-50. | `retrieval/hybrid.py` |
| Rerank top-50 → top-8 with BGE reranker. | Better precision. |
| Metadata filter parser — if query mentions a company or date, restrict accordingly (use Claude Sonnet for query parsing). | "What did CBA announce in March?" hits only CBA-March chunks. |
| Generation prompt: **forces citations in `[ann_id:chunk_id]` format, refuses if confidence low**. Use Claude Sonnet 4.6. | `agent/prompts.py` |
| Citation verifier: post-generation, extract citations, confirm each is in retrieved set, run a quick LLM check that the cited chunk supports the claim. Retry once or downgrade to refusal. | `agent/verify_citations.py` |
| Add a `/query` FastAPI endpoint that returns `{answer, citations[], retrieval_debug{}}` | Working backend. |
| Trace everything with Langfuse. | You can show traces on demo. |

**End of week 2:** You can ask questions via curl and get cited answers.

---

### Week 3 — The eval harness (the differentiator) *(weekend + 3 evenings)*

This is the week that turns the project from "yet another RAG" into a hireable artifact. Spend the most time here.

| Task | Deliverable |
|---|---|
| Write **80 golden questions** across 5 categories: factual-answerable (30), unanswerable / out-of-corpus (15), time-bounded (10), comparative (10), adversarial / hallucination-bait (15). Hand-label expected citations. | `eval/testset.jsonl` |
| LLM-as-judge using Claude Opus 4.7 with structured reasoning. Grade five axes per query: factual correctness, citation accuracy, citation recall, refusal calibration, format compliance. | `eval/judge.py` |
| Run the eval. Capture latency p50/p95 and $/query from Langfuse traces. | First scoreboard numbers. |
| **Iterate on the agent** (better prompts, better reranker thresholds, better refusal logic) until: hallucination rate < 5%, citation accuracy > 90%, refusal-on-unanswerable > 80%. **Don't fake the numbers.** | Real, defensible metrics. |
| Write `docs/eval-methodology.md` — this doubles as your blog post. Include the failure cases, not just the wins. | Draft post. |
| Publish `scoreboard_data.json` for the UI. | Public scoreboard. |
| Add an eval-on-PR GitHub Action that runs the harness on every commit and blocks regressions > 2%. | "Eval-driven development" — strong CV bullet. |

**End of week 3:** You have *numbers*. Senior engineers respect this more than any feature.

---

### Week 4 — Polish, ship, broadcast *(weekend + 3 evenings)*

| Task | Deliverable |
|---|---|
| Next.js UI: query box, streamed answer with inline citation pills, click → opens ASX page in new tab. Retrieval-debug accordion. | `/` page. |
| Scoreboard page: bar charts of metrics, table of the 80 test queries with model output + judge reasoning. | `/eval` page. |
| Landing-page hero: 1-sentence pitch, embedded 90-second Loom, "Try it" CTA, link to GitHub + scoreboard. | Conversion-grade landing page. |
| Deploy backend on Fly, frontend on Vercel, point domain. | Live URL. |
| Record the 90s Loom: "Here's the question. Here's the cited answer. Here's the citation linking to the ASX page. Here's the eval scoreboard with real numbers." Don't script it stiffly. | Loom in landing page. |
| Polish README — demo gif at top, badges, architecture diagram, "Why this exists" section, eval results. | Repo gets a star. |
| Publish blog post (your own site, Medium, or dev.to) and the engineering write-up. | One canonical URL to share. |
| Update LinkedIn headline + featured section. Update CV's Key Projects to lead with this. | Visibility. |
| **Outreach week starts the day this ships.** | First 5 DMs. |

**End of week 4:** Live, demoable, with numbers. You're shippable.

---

## 6. The eval methodology — what makes this senior

Most candidates show *one* number ("90% accuracy"). You're going to show *five*, on a *labelled, stratified test set*, with *judge transparency*. That alone outranks 80% of portfolios.

Publish on the scoreboard:

| Metric | What it measures | How |
|---|---|---|
| **Citation accuracy** | Of cited chunks, what % actually support the answer? | Opus judges each citation→claim pair. |
| **Citation recall** | Of supporting chunks in the corpus, what % did the agent cite? | Compared to your hand-labelled golden citations. |
| **Hallucination rate** | What % of answers contain at least one unsupported claim? | Opus judges each claim against cited chunks only. |
| **Refusal calibration** | On unanswerable questions, what % did the agent refuse? On answerable questions, what % did it incorrectly refuse? | Two confusion-matrix numbers. |
| **p50 / p95 latency** | End-to-end wall clock | Langfuse traces. |
| **Cost per query** | USD | API spend / queries. |

Publish failures too. Pick three: a hallucination it makes, a refusal it gets wrong, and a citation it fakes. Showing your own failures is the senior move — it tells hiring managers you understand the system rather than just running it.

---

## 7. The blog post (outline — write this in Week 3 alongside building)

Working title: **"Grounded RAG over ASX filings — what I learned from building an evaluation harness first"**

1. **Hook (2 paragraphs):** "Most RAG demos show one cherry-picked answer. I wanted numbers. Here's what happened when I held mine to a real eval."
2. **The corpus** — ASX announcements, why they're hard (PDFs, boilerplate, scanned tables, mixed-quality OCR).
3. **The retrieval stack** — hybrid + rerank, with the ablation numbers (BM25 alone, vector alone, hybrid, hybrid+rerank).
4. **Citation enforcement** — the prompt, the verifier, what happens when the verifier rejects.
5. **The eval set** — how I wrote 80 questions across 5 categories, including the unanswerable ones.
6. **Results** — the scoreboard, plus three specific failure cases I haven't fixed yet.
7. **What I'd build next.**
8. **Try it / fork it.**

Aim: 1,200–1,800 words, two embedded charts, three code snippets. Cross-post to LinkedIn (native, not link), HN (Show HN), `r/MachineLearning`, AusAI Slack, Sydney ML Meetup channel.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ASX republishing terms | Store only embeddings + short excerpts; deep-link every answer to the official ASX page; add a disclaimer; never claim data ownership. |
| PDF parsing is messier than expected | Restrict scope to text-based PDFs in Week 1; mark image-only ones as "skipped" in metadata; this becomes a known limitation in the blog. |
| Eval scores embarrass you | They're supposed to be honest. Publish them as-is and iterate. A 78% citation accuracy with rigor beats a fake 99% with vibes. |
| Scope creep (you'll want to add multi-hop, summaries, alerts…) | Each item lives in a `BACKLOG.md`. Nothing new enters Weeks 1–4. Add them after launch as a v2 post. |
| LLM costs spiral | Cache aggressively (prompt + answer + citations keyed by query hash). Use Sonnet not Opus for serving. Hard-cap with a query budget. |
| You ship late | The minimum hireable artifact = ingestion + retrieval + grounded generation + 30 eval questions + live URL. Cut UI polish before cutting eval depth. |

---

## 9. Day 0 — what to do today

In order. Each is 5–30 minutes.

1. Create the GitHub repo `asx-grounded`. Public. MIT license. Add the README skeleton from §4 above.
2. Buy `asx-grounded.app` (or similar). Worth the $20 commitment device.
3. Sign up: Qdrant Cloud, Supabase, Fly.io, Vercel, Langfuse Cloud, Anthropic API. Free tiers across the board.
4. Pin a one-line milestone tracker in the README: `[ ] W1 ingest [ ] W2 grounded gen [ ] W3 eval [ ] W4 ship`.
5. Write the **first 10 eval questions** before writing any retrieval code. This is the "eval-driven development" mindset and forces you to think product-first.
6. Block four Saturdays on your calendar. Non-negotiable.
7. Tell one person you're doing this — a friend, your SAS manager, a Sydney ML person. Public commitment makes it real.
