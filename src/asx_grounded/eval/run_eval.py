"""Orchestrate the full eval loop.

For each question:
  1. Retrieve + rerank (same path as production /query).
  2. Generate + verify citations (same path).
  3. Judge with Opus 4.7.
  4. Record latency, $/query, verdict.

At the end: aggregate into a scoreboard JSON the UI consumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, quantiles

import structlog
import typer

from asx_grounded.agent.generate import generate
from asx_grounded.agent.verify_citations import verify_citations
from asx_grounded.config import get_settings
from asx_grounded.eval.judge import judge
from asx_grounded.models import EvalQuestion, QueryResponse
from asx_grounded.retrieval.filters import extract_filter
from asx_grounded.retrieval.hybrid import HybridRetriever
from asx_grounded.retrieval.rerank import CrossEncoderReranker

log = structlog.get_logger()
app = typer.Typer(help="Run the eval harness and write the scoreboard.")


# Conservative per-query cost estimate for Claude Sonnet 4.6 with ~3k input + ~400 output.
# Refine with real Langfuse-reported token counts during run.
_SONNET_INPUT_USD_PER_MTOK = 3.00
_SONNET_OUTPUT_USD_PER_MTOK = 15.00
_OPUS_INPUT_USD_PER_MTOK = 15.00
_OPUS_OUTPUT_USD_PER_MTOK = 75.00


def _load_testset(path: Path) -> list[EvalQuestion]:
    out: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(EvalQuestion.model_validate_json(line))
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    qs = quantiles(values, n=100)
    idx = min(int(p) - 1, len(qs) - 1)
    return qs[idx]


def _run_one(
    q: EvalQuestion,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> tuple[QueryResponse, list, int, int]:
    mfilter = extract_filter(q.question)
    candidates = retriever.retrieve(q.question, mfilter=mfilter)
    top = reranker.rerank(q.question, candidates)
    started = time.perf_counter()
    gen = generate(q.question, top)
    verified = verify_citations(gen.text, top, verify_with_llm=False)
    for c in verified.citations:
        c.asx_page_url = retriever.page_url(c.chunk_id)
    latency_ms = int((time.perf_counter() - started) * 1000)
    resp = QueryResponse(
        query=q.question,
        answer=verified.answer_text,
        citations=verified.citations,
        refused=gen.refused,
        refusal_reason=gen.refusal_reason,
        retrieval_debug={"candidates": len(candidates), "reranked": len(top)},
        latency_ms=latency_ms,
        model=gen.model,
    )
    return resp, top, gen.input_tokens, gen.output_tokens


def _aggregate(
    verdicts: list[dict],
    latencies: list[int],
    costs: list[float],
) -> dict:
    if not verdicts:
        return {"status": "no_results"}
    n = len(verdicts)
    hallucination_rate = sum(1 for v in verdicts if v["hallucination"]) / n
    citation_acc = mean(v["citation_accuracy"] for v in verdicts)
    citation_rec = mean(v["citation_recall"] for v in verdicts)
    refusal_correct_rate = sum(1 for v in verdicts if v["refusal_correct"]) / n
    format_rate = sum(1 for v in verdicts if v["format_compliant"]) / n
    factual_rate = sum(1 for v in verdicts if v["factually_correct"]) / n

    by_category: dict[str, dict[str, float]] = {}
    cats = {v["category"] for v in verdicts}
    for cat in cats:
        rows = [v for v in verdicts if v["category"] == cat]
        if not rows:
            continue
        by_category[cat] = {
            "n": len(rows),
            "factually_correct": sum(1 for v in rows if v["factually_correct"]) / len(rows),
            "hallucination_rate": sum(1 for v in rows if v["hallucination"]) / len(rows),
            "refusal_correct": sum(1 for v in rows if v["refusal_correct"]) / len(rows),
        }

    return {
        "summary": {
            "n_questions": n,
            "factually_correct_rate": round(factual_rate, 3),
            "citation_accuracy": round(citation_acc, 3),
            "citation_recall": round(citation_rec, 3),
            "refusal_calibration": round(refusal_correct_rate, 3),
            "format_compliance": round(format_rate, 3),
            "hallucination_rate": round(hallucination_rate, 3),
            "latency_p50_ms": int(_percentile(latencies, 50)),
            "latency_p95_ms": int(_percentile(latencies, 95)),
            "cost_usd_total": round(sum(costs), 4),
            "cost_usd_per_query": round(mean(costs), 4) if costs else 0.0,
        },
        "by_category": by_category,
        "verdicts": verdicts,
    }


def run_eval(
    testset_path: Path,
    bm25_corpus: Path,
    scoreboard_out: Path,
) -> dict:
    questions = _load_testset(testset_path)
    log.info("eval.start", n=len(questions))

    retriever = HybridRetriever(bm25_corpus)
    reranker = CrossEncoderReranker()

    verdicts: list[dict] = []
    latencies: list[int] = []
    costs: list[float] = []

    for q in questions:
        try:
            resp, sources, in_tok, out_tok = _run_one(q, retriever, reranker)
        except Exception as exc:  # pragma: no cover — surface any per-question crash
            log.error("eval.run_failed", qid=q.qid, error=str(exc))
            continue
        verdict = judge(q, resp, sources)
        cost = (
            in_tok / 1_000_000 * _SONNET_INPUT_USD_PER_MTOK
            + out_tok / 1_000_000 * _SONNET_OUTPUT_USD_PER_MTOK
            # Opus judge: assume ~2k input, ~200 output per verdict.
            + 2000 / 1_000_000 * _OPUS_INPUT_USD_PER_MTOK
            + 200 / 1_000_000 * _OPUS_OUTPUT_USD_PER_MTOK
        )
        verdicts.append({
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "expected_refusal": q.expected_refusal,
            "answer": resp.answer,
            "refused": resp.refused,
            "citations": [c.model_dump() for c in resp.citations],
            "factually_correct": verdict.factually_correct,
            "citation_accuracy": verdict.citation_accuracy,
            "citation_recall": verdict.citation_recall,
            "refusal_correct": verdict.refusal_correct,
            "format_compliant": verdict.format_compliant,
            "hallucination": verdict.hallucination,
            "reasoning": verdict.reasoning,
            "latency_ms": resp.latency_ms,
            "cost_usd": round(cost, 4),
        })
        latencies.append(resp.latency_ms)
        costs.append(cost)
        log.info("eval.judged", qid=q.qid, hallucination=verdict.hallucination)

    scoreboard = _aggregate(verdicts, latencies, costs)
    scoreboard["meta"] = {
        "generator_model": get_settings().generator_model,
        "judge_model": get_settings().judge_model,
        "testset": str(testset_path),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    scoreboard_out.parent.mkdir(parents=True, exist_ok=True)
    scoreboard_out.write_text(json.dumps(scoreboard, indent=2), encoding="utf-8")
    log.info("eval.done", out=str(scoreboard_out))
    return scoreboard


@app.command()
def cli(
    testset: Path = typer.Option(Path("src/asx_grounded/eval/testset.jsonl"), "--testset"),
    bm25: Path = typer.Option(Path("data/processed/bm25_corpus.jsonl"), "--bm25"),
    out: Path = typer.Option(Path("data/processed/scoreboard.json"), "--out"),
) -> None:
    scoreboard = run_eval(testset, bm25, out)
    summary = scoreboard.get("summary", {})
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
