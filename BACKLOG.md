# BACKLOG

Ideas that explicitly do NOT belong in v1. Each was named, considered, and deferred so the v1 cut stays sharp.

## Stretch — post-launch v2 candidates

- OCR pass for image-only PDFs (Tesseract or a small VLM)
- Streaming responses end-to-end (Claude streams → SSE → UI typewriter)
- Multi-hop reasoning across announcements with a planner
- Numerical-claim isolation eval (one figure per claim)
- Recency tie-breaking when sources contradict
- Per-company timeline view in the UI
- Push alerts on new price-sensitive announcements for a watchlist
- Postgres-backed query cache keyed by (question_hash, corpus_hash)
- Per-query rate limiter via Upper-Limit header on the API

## Explicit non-goals

- User accounts, billing, multi-tenancy
- Mobile-native UI
- Anything that requires private ASX subscription feeds
- Financial-advice features (regulatory hazard, out of scope)
