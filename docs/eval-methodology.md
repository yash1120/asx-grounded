# Eval Methodology

> The differentiator. Most RAG demos show one cherry-picked answer. This one ships with numbers.

This document is the methodology behind `data/processed/scoreboard.json` and doubles as the launch blog post.

## Why eval-first

A grounded-RAG system's value proposition is "we don't hallucinate, and when we don't know, we say so." Both halves of that claim are empirical. Without a test set and a judge, the claim is marketing. With them, it's engineering.

We wrote the first 10 eval questions *before* building retrieval. The test set drove which categories the agent needed to handle and exposed the refusal-vs-answer trade-off the system prompt now enforces.

## The test set

`src/asx_grounded/eval/testset.jsonl` ships with 10 stratified questions across 5 categories. The full launch target is 80 questions; what's in the repo today is the seed.

| Category | Count | What it tests |
|---|---|---|
| `answerable` | 30 (target) | Factual questions the corpus *can* answer. Cited claims must be supported by the cited chunks. |
| `unanswerable` | 15 (target) | Questions out of corpus (other markets, PII, future-dated). System must refuse. |
| `time_bounded` | 10 (target) | Questions that require a date filter to be applied correctly. |
| `comparative` | 10 (target) | Multi-entity questions. Partial refusal allowed if one entity is missing. |
| `adversarial` | 15 (target) | Leading questions, hallucination-bait, impossible dates. System must not confirm to please. |

Each question is hand-labelled with `expected_refusal` and free-text `notes`. Expected citations are filled in for a subset where the gold answer is unambiguous.

## What we measure

The judge (`eval/judge.py`, running on Claude Opus 4.7) returns six fields per question. The runner aggregates into a stratified scoreboard.

| Metric | Definition |
|---|---|
| **Factual correctness rate** | Proportion of answers whose claims are accurate given the cited sources. Correct refusals count as correct. |
| **Citation accuracy** | Of cited chunks, the fraction that actually support the claim they're attached to. |
| **Citation recall** | Of clearly-relevant available chunks, the fraction the agent cited. |
| **Refusal calibration** | Two numbers: refusal-on-unanswerable rate, and false-refusal rate on answerable. |
| **Format compliance** | Every factual claim has a `[chunk_id]` citation, or the response is a clean `REFUSE:`. |
| **Hallucination rate** | Any claim unsupported by the cited sources. |
| **Latency p50 / p95** | End-to-end wall clock, measured server-side. |
| **Cost per query** | Sum of generator + judge token spend at published per-million prices. |

## Why the judge runs on a stronger model than the generator

A judge that shares the generator's blind spots will rubber-stamp the generator's mistakes. Running Opus 4.7 as the judge while serving with Sonnet 4.6 means the grader has strictly more capability than the generated artifact, reducing this risk. The eval set is small enough (~80 questions × ~$0.03 = ~$2.40/run) that the cost is bounded.

## Reproducibility

```
make ingest    # pulls ASX announcements for the configured codes/days
make embed     # parses, chunks, embeds, snapshots BM25 corpus
make eval      # runs the harness, writes data/processed/scoreboard.json
```

Every eval run is timestamped in the scoreboard's `meta` block. CI runs the eval on pull requests labelled `run-eval` and blocks merges that regress hallucination rate by more than 2 percentage points (`.github/workflows/ci.yml`).

## What this does not yet measure

- **Multi-hop reasoning across announcements.** Not in the test set; the system isn't designed for it.
- **Numerical precision over many figures.** Claims involving multiple numbers in one sentence can pass the entailment check while individual numerals drift; a future eval will isolate this.
- **Recency bias.** When two announcements contradict, the system should prefer the more recent one. Not yet measured.
- **Adversarial prompt injection from PDF contents.** ASX PDFs are low-risk but we don't formally test for embedded instructions.

These limitations belong in the launch post, honestly. Pretending the scoreboard is exhaustive is the failure mode this project exists to refute.

## Three things that broke (placeholder — fill in after the first real run)

1. *...*
2. *...*
3. *...*
