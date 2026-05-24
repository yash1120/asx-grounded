# Sprints — asx-grounded

> Scrum cadence adapted for a solo developer balancing a full-time SAS internship + teaching tutor work. ~10 hours/week capacity (2 weekday evenings × 2hrs + 1 weekend day × 6hrs).

**Today:** 2026-05-24 (Sun)
**Ship target:** 2026-06-21 (live URL + real scoreboard numbers + blog post + first 25 outreach touches)
**Hire target:** 2026-08 to 2026-09 (8–14 weeks after ship, per plan.md)

---

## Roles (you wear all of them)

| Role | When | What |
|---|---|---|
| **Product Owner** | Friday evenings | Re-prioritise backlog, write/update stories, set next sprint goal |
| **Scrum Master** | Sunday + Wednesday 8pm | Run check-ins, unblock self, update sprint board |
| **Dev team** | Tue/Thu evenings + Sat | Build the code |
| **Stakeholder** | End of each sprint | You-as-hiring-manager critiques the demo |

## Ceremonies (solo-adapted)

| Ceremony | Cadence | Format |
|---|---|---|
| **Sprint planning** | Sunday 7pm, 60min | Re-read this file, pull stories into current sprint, refine ACs |
| **Daily standup** | Skip. Use the Wednesday mid-sprint check-in instead. |
| **Mid-sprint check-in** | Wednesday 8pm, 15min | What's done, what's blocked, do I cut scope? |
| **Sprint review** | Sunday 6pm, 30min | Demo to a friend, your SAS manager, or record yourself. No "almost done" — only shipped counts. |
| **Sprint retro** | Sunday 6:30pm, 15min | Three lines: kept, dropped, changed. Append below. |

## Story-point reference (Fibonacci)

| Points | Means | Example |
|---|---|---|
| 1 | <1hr, no thinking | Update a config value |
| 2 | 1–2hrs, well-understood | Write one tested module |
| 3 | Half a day, one risk to navigate | New endpoint with tests |
| 5 | Full day, multiple unknowns | First end-to-end integration with a third-party service |
| 8 | 2 days, significant unknowns | Whole new subsystem (e.g. eval harness from scratch) |
| 13 | Too big. Split. |

**Capacity per sprint:** ~13 points (10hrs × ~75% efficiency for solo work).

---

## Product backlog — Epics

Prioritised top-to-bottom. Each epic links to the sprint where it lands.

| # | Epic | Sprint | Status |
|---|---|---|---|
| E0 | Project scaffold (code, tests, lint, docs) | S0 | ✅ done |
| E1 | Real ASX ingestion proven end-to-end | S1 | ⏳ |
| E2 | Grounded generation working live | S2 | ⏳ |
| E3 | First eval run with real numbers | S2 | ⏳ |
| E4 | Deployed live URL (backend + frontend) | S3 | ⏳ |
| E5 | Expanded eval set (10 → 80) + published scoreboard | S3 | ⏳ |
| E6 | Blog post + broadcast + outreach kickoff | S4 | ⏳ |
| E7 | Job applications wave 1 (15 targeted) | S4 | ⏳ |
| E8 | Stretch v2 features (OCR, multi-hop, streaming) | post-launch | 🅿️ parked |

---

## Sprint 0 — Foundation **[DONE]**

**Dates:** 2026-05-17 → 2026-05-24
**Goal:** Stand up a complete, lint-clean scaffold with tests and docs.

| Story | Pts | Status |
|---|---|---|
| Project meta + config + models | 2 | ✅ |
| Ingestion pipeline modules | 3 | ✅ |
| Retrieval (hybrid + rerank + filters) | 3 | ✅ |
| Agent (prompts + generate + verify) | 3 | ✅ |
| Eval harness scaffold + 10 seed questions | 3 | ✅ |
| FastAPI app | 2 | ✅ |
| Next.js skeleton | 2 | ✅ |
| Dockerfile + fly.toml + CI | 2 | ✅ |
| Architecture + ADRs + eval methodology docs | 2 | ✅ |
| Tests for pure modules | 2 | ✅ |
| Lint clean (ruff check + format) | 1 | ✅ |

**Velocity:** 25 pts in ~1 week (one-off intensive — won't repeat at that rate).

**Retro:**
- **Kept:** Eval-first thinking (test set written before retrieval).
- **Dropped:** Mypy strictness (CI has `continue-on-error`); revisit in S4.
- **Changed:** Bumped line-length to 120; added per-file E501 ignores for prompt files.

---

## Sprint 1 — Real ASX ingestion **[NEXT]**

**Dates:** 2026-05-25 (Mon) → 2026-05-31 (Sun)
**Goal:** Prove the ingestion pipeline works against the live ASX with at least 5 ASX-50 companies indexed, 30 days back, embedded, queryable.
**Demo at end:** A curl against the locally-running `/query` returns cited chunks for "What did CBA announce in the last month?"

### Backlog

| ID | Story | Pts | AC |
|---|---|---|---|
| S1-01 | First-commit + GitHub repo public | 1 | `git init`, MIT licence in repo, first commit pushed to `github.com/<you>/asx-grounded`, README renders correctly |
| S1-02 | Validate ASX endpoint shape | 3 | Hit live ASX listing API for CBA; if response shape differs from `_parse_item`, patch the parser and add 1 unit test against a saved fixture |
| S1-03 | Sign up Qdrant Cloud + Supabase + Anthropic API + Langfuse | 1 | Keys in `.env`; `python -c "from qdrant_client import QdrantClient; QdrantClient(...).get_collections()"` succeeds |
| S1-04 | Ingest CBA + BHP + WBC + CSL + WES, 30 days | 3 | `make ingest` runs cleanly; `data/raw/announcements.jsonl` has ≥50 rows; PDFs on disk |
| S1-05 | Embed corpus + write BM25 snapshot | 2 | `make embed` completes; Qdrant collection has ≥500 vectors; `bm25_corpus.jsonl` written |
| S1-06 | Smoke test retrieval locally | 2 | Python REPL: `HybridRetriever(...).retrieve("CBA dividend")` returns ≥5 non-empty `RetrievedChunk`s with valid metadata |
| S1-07 | Handle 3 real-world parse failures | 2 | Identify ≥3 PDFs that fail or parse poorly; log them; add 1 fix or document as known-limitation in `docs/eval-methodology.md` |

**Total: 14 pts.** Slightly over capacity — scope-cut candidate is S1-07 (defer to S2).

### Risks

| Risk | Mitigation |
|---|---|
| ASX endpoint shape has drifted | S1-02 is first; if it's broken, swap in a fallback (manually-downloaded PDF set) and document |
| Qdrant free tier doesn't fit | Switch to local Qdrant in Docker for dev; cloud only on deploy |
| Embedding model download is slow | One-time cost; bake into Docker for prod (already in Dockerfile) |

### Definition of Done

- [ ] GitHub repo public with clean first commit
- [ ] At least 5 companies × 30 days ingested end-to-end
- [ ] Qdrant collection populated; BM25 snapshot on disk
- [ ] `make serve` runs and `/healthz` returns 200
- [ ] One smoke-test query returns ≥5 cited chunks via curl
- [ ] All tests still pass; lint still clean

---

## Sprint 2 — Grounded generation + first eval

**Dates:** 2026-06-01 (Mon) → 2026-06-07 (Sun)
**Goal:** End-to-end `/query` returns cited answers with verifiable citations, and the first real eval run produces a `scoreboard.json` with honest numbers (even if they're embarrassing).
**Demo at end:** Live curl to `/query` returning a cited answer; `cat data/processed/scoreboard.json | jq .summary` shows real metrics.

### Backlog

| ID | Story | Pts | AC |
|---|---|---|---|
| S2-01 | First end-to-end `/query` against real corpus | 3 | curl POST `/query` with "What is CBA's most recent dividend?" returns a JSON `QueryResponse` with ≥1 valid citation and `refused=false` |
| S2-02 | Confirm citation verifier strips fabricated ids | 2 | Inject a fake `[FAKE:0]` into a test answer; verifier must drop it (already tested but verify with real LLM output) |
| S2-03 | Run eval on the 10 seed questions | 2 | `make eval` produces `scoreboard.json` with `n_questions=10`; record costs |
| S2-04 | Expand test set to 30 questions | 3 | `testset.jsonl` has 30 entries across all 5 categories; each has hand-labelled `expected_refusal` + `notes` |
| S2-05 | Re-run eval; record failure cases | 2 | New `scoreboard.json`; pick 3 worst failures and document in `docs/eval-methodology.md` §"Three things that broke" |
| S2-06 | Wire Langfuse traces | 1 | `LANGFUSE_*` env vars set; every `/query` and eval run produces a trace |
| S2-07 | Tighten the citation-forcing prompt if hallucination > 10% | 2 | Iterate on `prompts.py`; commit improvements with before/after eval numbers in commit message |

**Total: 15 pts.** Cut S2-06 (Langfuse) to S3 if needed — it's a nice-to-have, not on the critical path.

### Risks

| Risk | Mitigation |
|---|---|
| First eval has hallucination > 20% | That's the point of eval — iterate. Don't fake the numbers. Document the iteration. |
| Anthropic API quota / rate limits | 30 questions × ~$0.03 = ~$1/run. Bounded. |
| Real LLM emits citations in slight format variations | `verify_citations.py` regex is permissive; if format drift breaks it, fix the regex AND add a unit test |

### Definition of Done

- [ ] `/query` works end-to-end with real ASX data
- [ ] Test set ≥30 stratified questions
- [ ] First scoreboard.json with real numbers (hallucination, citation accuracy, refusal calibration, latency, cost)
- [ ] Three documented failure cases
- [ ] Lint + tests green

---

## Sprint 3 — Ship the live URL

**Dates:** 2026-06-08 (Mon) → 2026-06-14 (Sun)
**Goal:** A stranger on the internet can hit a public URL, ask a question, get a cited answer, click the citation, land on the real ASX page. Public scoreboard shows real numbers.
**Demo at end:** Send a link to a friend. They use it without instructions. Record a 90-second Loom.

### Backlog

| ID | Story | Pts | AC |
|---|---|---|---|
| S3-01 | Frontend smoke test locally | 2 | `cd web && npm install && npm run dev`; query box works, citation pills render, links open ASX page |
| S3-02 | Buy domain (`asx-grounded.app` or similar) | 1 | Domain purchased; DNS configured |
| S3-03 | Deploy backend to Fly.io (Syd region) | 3 | `fly deploy`; `https://asx-grounded.fly.dev/healthz` returns 200; secrets set via `fly secrets set` |
| S3-04 | Deploy frontend to Vercel | 2 | `vercel --prod`; `NEXT_PUBLIC_API_BASE` points at Fly app; live URL works |
| S3-05 | Expand test set 30 → 80 | 3 | `testset.jsonl` has 80 entries; stratification maintained |
| S3-06 | Re-run eval on 80 questions; publish scoreboard | 2 | `/eval` page renders updated scoreboard with all metrics |
| S3-07 | Record 90s Loom demo | 1 | Loom URL pinned in README + landing page hero |
| S3-08 | Polish landing page hero (1-sentence pitch + CTA) | 1 | Anyone landing cold understands the value in <30s |

**Total: 15 pts.**

### Risks

| Risk | Mitigation |
|---|---|
| Fly machines too small for embedding model | Bake weights into image (already done); use 2gb VM (already configured); fall back to remote inference if needed |
| Eval cost spike on 80 questions | 80 × ~$0.03 = ~$2.40 per run. Bounded. |
| First public visitors hit empty/broken endpoint | Add a `/query` rate limiter and an `OPTIONS` preflight handler before public DNS goes live |

### Definition of Done

- [ ] Live URL accessible without auth
- [ ] Working query → cited answer → ASX page round-trip
- [ ] Public scoreboard with ≥80 graded questions
- [ ] Loom demo recorded
- [ ] CI still green
- [ ] At least one third party (friend, SAS colleague) has tested it

---

## Sprint 4 — Broadcast + first outreach wave

**Dates:** 2026-06-15 (Mon) → 2026-06-21 (Sun)
**Goal:** The project exists publicly. Hiring managers in Sydney see it. First applications fly with the demo URL in the cover letter.
**Demo at end:** 25 outreach touches sent, 1 published blog post, 1 LinkedIn featured-section update, ≥5 applications submitted with the demo link.

### Backlog

| ID | Story | Pts | AC |
|---|---|---|---|
| S4-01 | Write engineering blog post | 3 | 1200–1800 words, 2 charts, 3 code snippets; based on `docs/eval-methodology.md`; published to your blog/Medium/dev.to |
| S4-02 | Cross-post to LinkedIn (native, not link) | 1 | Native LinkedIn article; includes scoreboard screenshot + demo link |
| S4-03 | Cross-post to HN (Show HN), r/MachineLearning | 1 | Posts submitted with the demo URL up top |
| S4-04 | Post in 2 Sydney AI Slack groups | 1 | At least 2 communities pinged |
| S4-05 | Update CV: lead "Key Projects" with this one | 1 | New version of `Yash_goyal_cv.pdf` pinned to drive + LinkedIn |
| S4-06 | Update LinkedIn headline + Featured section | 1 | Demo URL in Featured; one-line pitch in headline |
| S4-07 | Target list: 15 hiring managers across 15 Sydney employers | 2 | Spreadsheet with name, role, company, LinkedIn URL, status |
| S4-08 | First 15 outreach DMs sent | 2 | Each DM 3 sentences max; includes demo URL; tracked in spreadsheet |
| S4-09 | First 5 targeted job applications | 2 | Cover letters reference the demo URL specifically |
| S4-10 | Submit talk proposal: Sydney ML Meetup | 1 | CFP submitted; record submission even if rejected |

**Total: 15 pts.**

### Risks

| Risk | Mitigation |
|---|---|
| Live URL goes down under HN-front-page traffic | Fly auto-scaling configured; rate limiter in place; worst case the homepage stays up |
| Outreach DMs feel spammy | Each is hand-personalised; mention something specific about the recipient/company; cap at 5/day |
| Embarrassing eval numbers picked apart in public | They're already documented honestly. Lean in. "Here's where it fails" is the senior signal. |

### Definition of Done

- [ ] Blog post published, ≥1 cross-post live
- [ ] CV + LinkedIn updated
- [ ] ≥15 outreach DMs sent
- [ ] ≥5 applications submitted
- [ ] 1 talk proposal submitted
- [ ] Demo URL still up

---

## Post-launch — Continuous mode

After Sprint 4, switch from project-build mode to **job-hunt operating mode**. Weekly cadence:

| Day | Action |
|---|---|
| Mon | Triage any inbound replies; respond within 24hrs |
| Wed | 5 new outreach touches + 3 new applications |
| Fri | Update tracking spreadsheet; pull one feature from BACKLOG.md if time permits |
| Sat | Interview prep + portfolio polish; eval re-run if any change shipped |

### Stretch backlog (parked)

These don't ship before the first job offer. Listed so they don't tempt scope creep mid-sprint.

| # | Item | When to revisit |
|---|---|---|
| B1 | OCR pass for image-only PDFs | When 2+ employers ask about it |
| B2 | Streaming `/query` responses (SSE) | If demo feels sluggish in interviews |
| B3 | Multi-hop reasoning planner | v2 blog post candidate |
| B4 | Per-company timeline view | UX upgrade for round-2 interview demos |
| B5 | Push alerts on new price-sensitive announcements | Productisation if a real customer asks |
| B6 | Tighten `mypy --strict` | When CI feedback loop matters to a team review |

---

## Velocity & burndown

| Sprint | Planned | Delivered | Notes |
|---|---|---|---|
| S0 | 25 | 25 | Intensive scaffold week, not sustainable cadence |
| S1 | 14 | — | |
| S2 | 15 | — | |
| S3 | 15 | — | |
| S4 | 15 | — | |

Update these numbers in Sunday retro. If delivered < planned for two sprints in a row, **cut scope** — don't extend the timeline. The whole project loses value if it ships in September instead of June.

---

## Retrospective log (append after each sprint)

### S0 — 2026-05-24
- **Kept:** Eval-first; small focused modules; per-file ignores for prompt strings.
- **Dropped:** Strict mypy gate; will revisit in S4.
- **Changed:** Picked Annotated[] typer pattern over `=typer.Option(...)` to placate B008.

### S1 — TBD
- ...

### S2 — TBD
- ...

### S3 — TBD
- ...

### S4 — TBD
- ...
