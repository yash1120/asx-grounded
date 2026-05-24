# Architecture

`asx-grounded` is a grounded RAG system over ASX continuous-disclosure announcements with hard citation enforcement and a published eval scoreboard. This document describes the runtime, the data flow, and the contracts between modules.

## Runtime view

```
                        +---------------------+
                        |   ASX listing API   |
                        +----------+----------+
                                   |
                                   v
            +------------------------------------------+
            |  ingestion/fetch_asx.py                  |
            |   rate-limited httpx + tenacity retry    |
            +-----+----------------+-------------------+
                  |                |
        announcements.jsonl     PDFs on disk
                  |                |
                  +----+      +----+
                       |      |
                       v      v
                 +----------------+
                 |  parse_pdf +   |
                 |    chunk       |
                 +-------+--------+
                         |
                 +-------+--------+
                 |    embed.py    |
                 |  bge-large +   |
                 |  Qdrant upsert |
                 +-------+--------+
                         |
                         v
                +-------------------+
                |  Qdrant +         |
                |  bm25_corpus.jsonl |
                +---------+----------+
                          |
   user query ----------->+
                          v
              +------------------------+
              |  retrieval/filters.py  |  regex + LLM metadata extraction
              +-----------+------------+
                          v
              +------------------------+
              |  retrieval/hybrid.py   |  BM25 + vector + RRF
              +-----------+------------+
                          v
              +------------------------+
              |  retrieval/rerank.py   |  BGE cross-encoder
              +-----------+------------+
                          v
              +------------------------+
              |  agent/generate.py     |  Claude Sonnet 4.6 + strict citation prompt
              +-----------+------------+
                          v
              +------------------------+
              |  verify_citations.py   |  drop fabricated ids + optional LLM check
              +-----------+------------+
                          v
                    QueryResponse
                          |
              +-----------+------------+
              |   FastAPI /query       |
              +-----------+------------+
                          |
                          v
                +-------------------+
                |   Next.js UI      |
                +-------------------+

eval (offline): same path, scored by Claude Opus 4.7 → scoreboard.json
```

## Module contracts

| Module | Owns | Reads | Writes |
|---|---|---|---|
| `ingestion.fetch_asx` | ASX HTTP integration | ASX listing API | `data/raw/pdfs/`, `data/raw/announcements.jsonl` |
| `ingestion.parse_pdf` | PDF → text per page | PDF bytes | `ParsedPdf` (in-memory) |
| `ingestion.chunk` | Token-aware chunking | `ParsedPdf` | `Chunk` (in-memory) |
| `ingestion.embed` | Embeddings + Qdrant + BM25 snapshot | PDFs, manifest | Qdrant collection, `data/processed/bm25_corpus.jsonl` |
| `retrieval.filters` | Query → metadata filter | query string | `MetadataFilter` |
| `retrieval.hybrid` | BM25 + vector + RRF | Qdrant, BM25 corpus | `list[RetrievedChunk]` |
| `retrieval.rerank` | Cross-encoder rerank | retrieved chunks | reranked + score-thresholded list |
| `agent.prompts` | System prompt + context render | retrieved chunks | prompt strings |
| `agent.generate` | Claude call | prompt + chunks | `GenerationResult` |
| `agent.verify_citations` | Citation extraction + verification | answer text + chunks | `VerifiedAnswer` |
| `api.main` | HTTP surface | all of the above | `QueryResponse` JSON |
| `eval.run_eval` | Eval orchestration | testset.jsonl | `data/processed/scoreboard.json` |
| `eval.judge` | LLM-as-judge | Opus 4.7 | `JudgeVerdict` |

## Failure modes & defences

| Failure mode | Defence |
|---|---|
| ASX API returns 5xx / changes shape | tenacity retry; per-code failures don't poison the run |
| PDF is scanned (image-only) | flagged at parse time, skipped from embedding; counted in metrics |
| Model fabricates a citation | regex extractor drops any chunk_id not in the retrieved set |
| Model answers from training data | system prompt forbids it; refusal is graded by the eval |
| Empty retrieval | API short-circuits with a refusal — no LLM call |
| Retrieval too narrow due to bad filter | filter extraction returns a permissive default when ambiguous |
| Cost runaway | reranker + rerank-top-k cap reduce input tokens; cache by query hash (future) |

## Non-goals (v1)

- OCR for image-only PDFs (tracked in BACKLOG.md).
- Streaming responses (Sonnet streams; UI consumes the final response for now).
- User accounts, rate limits beyond the demo IP-based throttle.
- Multi-hop reasoning across announcements (the test set explicitly avoids this).
