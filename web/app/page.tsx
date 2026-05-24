"use client";

import { useState } from "react";

type Citation = {
  chunk_id: string;
  ann_id: string;
  asx_page_url: string;
  verified: boolean;
};

type QueryResponse = {
  query: string;
  answer: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string;
  retrieval_debug: Record<string, unknown>;
  latency_ms: number;
  model: string;
};

function renderAnswerWithCitations(answer: string, citations: Citation[]) {
  const byId = new Map(citations.map((c) => [c.chunk_id, c]));
  // Replace [chunk_id] (and comma-separated groups) with anchor pills.
  const parts = answer.split(/(\[[^\[\]]+\])/g);
  return parts.map((part, i) => {
    const m = /^\[([^\[\]]+)\]$/.exec(part);
    if (!m) return <span key={i}>{part}</span>;
    const ids = m[1].split(",").map((s) => s.trim());
    return (
      <span key={i} className="inline-flex items-center gap-1 align-baseline">
        {ids.map((id, j) => {
          const c = byId.get(id);
          const cls =
            "ml-1 inline-block rounded px-1.5 py-0.5 text-[10px] font-mono " +
            (c ? "bg-emerald-900/40 text-emerald-200 hover:bg-emerald-800/60" : "bg-red-900/40 text-red-300");
          return c?.asx_page_url ? (
            <a key={j} href={c.asx_page_url} target="_blank" rel="noreferrer" className={cls} title={id}>
              {id}
            </a>
          ) : (
            <span key={j} className={cls} title={id}>
              {id}
            </span>
          );
        })}
      </span>
    );
  });
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [resp, setResp] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function ask() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setErr(null);
    setResp(null);
    try {
      const r = await fetch("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!r.ok) throw new Error(`API ${r.status}`);
      setResp(await r.json());
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Ask a question about an ASX-listed company</h1>
        <p className="mt-2 text-sm text-zinc-400">
          Every claim is cited. Click a citation to open the original ASX announcement page.
          When the corpus does not support an answer, the system refuses rather than guessing.
        </p>
      </section>

      <section className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="e.g. What was CBA's most recent dividend?"
          className="flex-1 rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm outline-none focus:border-zinc-500"
        />
        <button
          onClick={ask}
          disabled={loading || !question.trim()}
          className="rounded-md bg-emerald-700 px-4 py-2 text-sm font-medium hover:bg-emerald-600 disabled:opacity-50"
        >
          {loading ? "Thinking…" : "Ask"}
        </button>
      </section>

      {err && <div className="rounded-md border border-red-800 bg-red-950/40 p-3 text-sm text-red-200">{err}</div>}

      {resp && (
        <section className="space-y-4">
          <div className="rounded-md border border-zinc-800 bg-zinc-950 p-4 text-sm leading-relaxed">
            {resp.refused ? (
              <div className="text-amber-300">Refused: {resp.refusal_reason || "context insufficient"}</div>
            ) : (
              <div className="whitespace-pre-wrap">{renderAnswerWithCitations(resp.answer, resp.citations)}</div>
            )}
          </div>
          <details className="text-xs text-zinc-500">
            <summary className="cursor-pointer">Retrieval debug · {resp.latency_ms} ms · {resp.model}</summary>
            <pre className="mt-2 overflow-x-auto rounded bg-zinc-950 p-3 text-[11px]">
              {JSON.stringify(resp.retrieval_debug, null, 2)}
            </pre>
          </details>
        </section>
      )}
    </div>
  );
}
