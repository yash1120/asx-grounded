# Grounded RAG over ASX filings — what I learned from building the eval harness first

## Most demos show one good answer. I wanted the distribution.

Every RAG demo looks the same. A slick chat box, one confident answer, a citation or two, and a vibe that says *trust me*. Nobody shows you the answer they cherry-picked away from. Nobody tells you how often the thing makes something up, or what it does when the corpus simply doesn't contain the answer. For a toy, fine. For anything a bank, insurer, or super fund would put in front of a regulator, "trust me" is not a deliverable.

So I built the eval harness first. Before I had a retriever, before I had a single embedding in a vector store, I wrote down questions the system would have to get right — and the ones it had to refuse — and decided how I'd grade it. The corpus is ASX continuous-disclosure announcements from the top-50 listed companies: public, messy, and exactly the kind of data Australian financial firms spend real money trying to index. This is the story of building that system back-to-front, and what writing the test set first changed about everything downstream.

## The corpus, and why it fought back

ASX announcements are a deceptively horrible corpus. They arrive as PDFs — dividend notices, Appendix 4D half-year results, substantial-holder filings, price-sensitive trading updates — wrapped in a standard ASX template with a cover page, page-numbered footers, copyright lines, and the occasional multi-column table that PDF text extraction shreds. Some are scanned images with no text layer at all. The boilerplate alone is enough to poison retrieval if you don't strip it, so the parser scrubs it explicitly:

```python
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*ASX (?:Market )?Announcement\b.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*page \d+ of \d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # bare page numbers
    re.compile(r"^\s*©.*all rights reserved.*$", re.IGNORECASE | re.MULTILINE),
]
```

Image-only PDFs get flagged at parse time and dropped from the index rather than silently embedded as empty strings — a known gap I count in the metrics rather than pretend away. OCR is on the backlog, not in v1.

But the real war story was getting the data at all. Every tutorial and half the Stack Overflow answers point at `https://www.asx.com.au/asx/1/company/{code}/announcements`. I wired it up, ran it, and got a wall of `404`s. The endpoint hadn't been deprecated with a notice — it had silently moved. ASX's live announcement data is now served by a **Markit Digital** backend at `asx.api.markitdigital.com`, with a completely different envelope: camelCase keys (`documentKey`, `isPriceSensitive`, `announcementType`), a `{"data": {"items": [...]}}` wrapper, and — the part that cost me an afternoon — a `url` field on each item that comes back *empty*. You have to construct both the PDF link and the human-readable viewer link yourself from the `documentKey`:

```python
ASX_LISTING_URL = "https://asx.api.markitdigital.com/asx-research/1.0/companies/{code}/announcements"
ASX_PDF_URL_TEMPLATE = "https://asx.api.markitdigital.com/asx-research/1.0/file/{key}"
ASX_PAGE_URL_TEMPLATE = "https://www.asx.com.au/asx/statistics/displayAnnouncement.do?display=pdf&idsId={key}"
```

The lesson I keep relearning: the integration layer is where real corpora hurt you, not the model. I quarantined all of it behind a single `AsxClient` so that when ASX inevitably moves the goalposts again, exactly one file changes.

## Hybrid retrieval, and why each piece pays rent

ASX language is bimodal. Half of it is rigid regulatory vocabulary — "substantial holder", "Appendix 3X", "Appendix 4D" — and exact tickers like CBA or FMG. The other half is free-form business prose that paraphrases wildly. A pure embedding retriever under-weights the regulatory phrases and fumbles exact codes; a pure BM25 retriever misses every paraphrase. Running one alone leaves obvious recall on the floor.

So I run both in parallel and fuse them with Reciprocal Rank Fusion. RRF is the piece I'd defend hardest, because it's almost insultingly simple and it just works — no weights to tune, no per-query calibration, parameter-free except for the constant `k=60`:

```python
@staticmethod
def _rrf(rankings: Iterable[list[tuple[str, float]]]) -> list[tuple[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            totals[chunk_id] += 1.0 / (RRF_K + rank)
    return sorted(totals.items(), key=lambda t: t[1], reverse=True)
```

It only uses rank position, never raw scores, which is exactly why it's robust: BM25 scores and cosine similarities live on incomparable scales, and any weighted-sum fusion forces you to reconcile them with a knob you'll be forever tuning. RRF sidesteps that entirely.

Then a `bge-reranker-large` cross-encoder re-scores the top-50 fused candidates down to the top-8. The bi-encoder embeddings used for the vector leg are fast but lossy — they compress query and document into separate vectors and hope the dot product captures relevance. The cross-encoder reads the query and the candidate *together*, which is far more precise and far too slow to run over the whole corpus. Top-50 is the sweet spot: wide enough that the right chunk is almost always in the candidate set, narrow enough that the ~150 ms reranker cost is bounded. Each piece earns its place: BM25 for exact terms, vectors for paraphrase, RRF to fuse without tuning, the cross-encoder for final precision.

## Citation enforcement, not citation suggestion

A model that's *asked* to cite will cite — including citing chunk IDs it never received. That's not grounding, that's a confident hallucination with a footnote. So enforcement is mechanical. The system prompt demands a `[chunk_id]` after every factual claim, and then a regex verifier strips any citation referencing an ID that wasn't in the retrieved set:

```python
retrieved_ids = {r.chunk.chunk_id: r for r in retrieved}
fabricated: list[str] = []
for _, cid in cited_pairs:
    if cid not in retrieved_ids:
        fabricated.append(cid)
        continue
    # ... keep only citations whose id we actually sent the model
```

Fabricated IDs are stripped from the answer text rather than left as dangling brackets, and logged so I can see how often it happens. This stage is cheap, deterministic, and always on. There's a second, opt-in stage — an LLM entailment check that asks whether the cited chunk *actually supports* the claim, not just that the ID exists — but that one costs a model call per sentence, so it runs in the eval rather than on every user query. The split is deliberate: cheap-and-always for fabrication, expensive-and-occasional for misattribution.

## Hard refusal: no quiet fallback to training data

Here's the design decision I'm most opinionated about. When retrieved context is thin, most LLMs will gracefully fill the gap from training data — "I don't have the specifics, but generally..." That's a feature for a chatbot and a bug for grounded QA, because it destroys the exact property the whole system promises. So the prompt forbids it outright:

```
2. If the provided context does NOT contain enough information to answer confidently, respond with EXACTLY:
   REFUSE: <one short sentence explaining what is missing>
   Do not guess. Do not use general knowledge.

5. When the question is unanswerable from the corpus, refuse — even if you could answer from training data.
```

Refusing well is senior behaviour; bluffing is junior behaviour. But a system can game a hallucination score by refusing *everything*, so refusal can't be a free pass — it has to be calibrated. I measure two numbers: the correct-refusal rate on genuinely unanswerable questions, and the false-refusal rate on answerable ones. You only get credit for refusing when refusing was the right call.

## The eval: a stratified set, and a stronger model holding the pen

The test set is 80 hand-labelled questions, stratified across five categories: `answerable` (the corpus can answer it), `unanswerable` (other markets, PII, future dates — must refuse), `time_bounded` (needs a date filter applied correctly), `comparative` (multi-company, partial refusal allowed), and `adversarial` (leading questions and hallucination bait like "confirm BHP's merger with Rio Tinto"). Each carries an `expected_refusal` flag and free-text notes telling the grader what good looks like.

The grader is **Claude Opus 4.7**, while the system being graded serves on **Claude Sonnet 4.6**. That asymmetry is the point. A judge that shares the generator's blind spots will happily rubber-stamp the generator's mistakes; giving the grader strictly more capability than the artifact it's grading reduces that risk. The set is small enough — roughly 80 questions at a few cents each — that a full run costs a couple of dollars, which means I can afford to run it on every pull request labelled `run-eval` and block any merge that regresses hallucination rate by more than two percentage points. The judge returns six metrics per question: factual correctness, citation accuracy, citation recall, refusal calibration, format compliance, and hallucination rate, plus latency and cost.

## Results

> **[CHART: headline scoreboard — six metric cards (factual correctness, citation accuracy, citation recall, refusal calibration, format compliance, hallucination rate), plus a by-category bar chart. Pull from `data/processed/scoreboard.json`.]**

These numbers do not exist yet — the corpus ingest and first real eval run are the next step, and I refuse to invent them in a post about not making things up. Placeholders below get filled from the live scoreboard:

- Hallucination rate: `[INSERT hallucination_rate]`
- Factual correctness rate: `[INSERT factual_correctness_rate]`
- Citation accuracy: `[INSERT citation_accuracy]` / citation recall: `[INSERT citation_recall]`
- Refusal calibration: `[INSERT correct_refusal_rate]` correct-refusal, `[INSERT false_refusal_rate]` false-refusal
- Format compliance: `[INSERT format_compliance]`
- Latency p50/p95: `[INSERT latency_p50]` / `[INSERT latency_p95]`; cost per query: `[INSERT cost_per_query]`

And the part most write-ups skip — the failures:

> **[INSERT 3 honest failure cases from the first real eval run: the question, what the system did, and why it was wrong — e.g. a citation to a real chunk that mis-summarises it, an over-refusal on an answerable question, or a numeric drift inside a multi-figure sentence.]**

I already know roughly where the cracks will be: claims that cite a genuinely retrieved chunk but subtly mis-state it (the regex check can't catch that — only the entailment pass can), and sentences packing several figures where one numeral drifts while the entailment check still passes the sentence overall.

## What's next, and how to kick the tyres

The honest limitations are documented, not buried: no multi-hop reasoning across announcements, no isolation test for numeric precision, no recency-tiebreak when two filings contradict, and no formal test for prompt injection from PDF contents. Each is a line item, not a surprise. Next up is OCR for the image-only PDFs, a query cache to cut repeat-query cost, and broadening the test set beyond 80 once the first run tells me which categories are weakest.

The whole thing is open. Clone it, point it at your own ASX codes, and run the eval against your own corpus:

```bash
python -m asx_grounded.ingestion.fetch_asx --codes CBA,BHP,WBC --days 30
python -m asx_grounded.ingestion.embed
python -m asx_grounded.eval.run_eval   # writes data/processed/scoreboard.json
```

If you're hiring for ML engineering in Australia and you want to interrogate any single design choice, the ADRs in `docs/decisions.md` are written for exactly that conversation. The scoreboard is the pitch: not "trust me", but "here are the numbers, including the ones that make me look bad."
