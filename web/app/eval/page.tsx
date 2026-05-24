"use client";

import { useEffect, useState } from "react";

type Summary = {
  n_questions: number;
  factually_correct_rate: number;
  citation_accuracy: number;
  citation_recall: number;
  refusal_calibration: number;
  format_compliance: number;
  hallucination_rate: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  cost_usd_per_query: number;
};

type Verdict = {
  qid: string;
  category: string;
  question: string;
  hallucination: boolean;
  factually_correct: boolean;
  refusal_correct: boolean;
  citation_accuracy: number;
  reasoning: string;
};

type Scoreboard = {
  summary: Summary;
  by_category: Record<string, Record<string, number>>;
  verdicts: Verdict[];
  meta: { generator_model: string; judge_model: string; ts_utc: string };
};

function pct(x: number) {
  return `${(x * 100).toFixed(1)}%`;
}

export default function EvalPage() {
  const [data, setData] = useState<Scoreboard | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/eval/scoreboard")
      .then((r) => (r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="text-red-300">Failed to load: {err}</div>;
  if (!data || !data.summary) {
    return <div className="text-zinc-400">No eval runs yet. Run <code className="font-mono text-zinc-200">make eval</code>.</div>;
  }
  const s = data.summary;

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Eval scoreboard</h1>
        <p className="mt-1 text-sm text-zinc-400">
          {s.n_questions} questions · generator <span className="font-mono">{data.meta.generator_model}</span> · judge{" "}
          <span className="font-mono">{data.meta.judge_model}</span> · {data.meta.ts_utc}
        </p>
      </section>

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {([
          ["Factually correct", pct(s.factually_correct_rate)],
          ["Hallucination rate", pct(s.hallucination_rate)],
          ["Citation accuracy", pct(s.citation_accuracy)],
          ["Citation recall", pct(s.citation_recall)],
          ["Refusal calibration", pct(s.refusal_calibration)],
          ["Format compliance", pct(s.format_compliance)],
          ["p50 latency", `${s.latency_p50_ms} ms`],
          ["p95 latency", `${s.latency_p95_ms} ms`],
          ["$/query", `$${s.cost_usd_per_query.toFixed(4)}`],
        ] as const).map(([label, value]) => (
          <div key={label} className="rounded-md border border-zinc-800 bg-zinc-950 p-3">
            <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
            <div className="mt-1 font-mono text-lg">{value}</div>
          </div>
        ))}
      </section>

      <section>
        <h2 className="text-lg font-semibold">By category</h2>
        <table className="mt-2 w-full text-sm">
          <thead className="text-left text-zinc-400">
            <tr>
              <th className="py-1">Category</th>
              <th>n</th>
              <th>Correct</th>
              <th>Halluc.</th>
              <th>Refusal</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {Object.entries(data.by_category).map(([cat, row]) => (
              <tr key={cat} className="border-t border-zinc-800">
                <td className="py-1">{cat}</td>
                <td>{row.n}</td>
                <td>{pct(row.factually_correct)}</td>
                <td>{pct(row.hallucination_rate)}</td>
                <td>{pct(row.refusal_correct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="text-lg font-semibold">Per-question verdicts</h2>
        <div className="mt-2 space-y-2">
          {data.verdicts.map((v) => (
            <details key={v.qid} className="rounded-md border border-zinc-800 bg-zinc-950 p-3 text-sm">
              <summary className="cursor-pointer">
                <span className="font-mono text-xs text-zinc-500">{v.qid}</span>{" "}
                <span className={v.hallucination ? "text-red-300" : "text-emerald-300"}>
                  {v.hallucination ? "HALLUCINATION" : "ok"}
                </span>{" "}
                — {v.question}
              </summary>
              <div className="mt-2 text-zinc-300">{v.reasoning}</div>
            </details>
          ))}
        </div>
      </section>
    </div>
  );
}
