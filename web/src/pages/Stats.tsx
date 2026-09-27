import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { KvSpectrum, TensorStats, WeightStatsReport } from "../api/types";
import { DotChart } from "../components/charts";
import { Card, ErrorBox, Notes, useCall, useCapped } from "../components/ui";
import { pct } from "../lib/format";

type Col = { key: keyof TensorStats; label: string; fmt: (v: TensorStats) => string; num?: boolean };
const f3 = (v: number) => (Number.isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(3)) : "–");
const COLS: Col[] = [
  { key: "name", label: "tensor", fmt: (t) => t.name },
  { key: "dtype", label: "dtype", fmt: (t) => t.dtype.toUpperCase() },
  { key: "rms", label: "rms", fmt: (t) => f3(t.rms), num: true },
  { key: "max_abs", label: "max |w|", fmt: (t) => f3(t.max_abs), num: true },
  { key: "kurtosis", label: "kurtosis", fmt: (t) => f3(t.kurtosis), num: true },
  {
    key: "channel_outlier_ratio",
    label: "outlier ch.",
    fmt: (t) => (t.channel_outlier_ratio === null ? "–" : f3(t.channel_outlier_ratio)),
    num: true,
  },
  { key: "zero_fraction", label: "zeros", fmt: (t) => pct(t.zero_fraction), num: true },
  { key: "ternary_group_fraction", label: "ternary groups", fmt: (t) => pct(t.ternary_group_fraction), num: true },
];

function TensorTable({ rows }: { rows: TensorStats[] }) {
  const [sort, setSort] = useState<{ key: keyof TensorStats; desc: boolean }>({ key: "kurtosis", desc: true });
  const sorted = useMemo(() => {
    const v = [...rows];
    v.sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
      return sort.desc ? -c : c;
    });
    return v;
  }, [rows, sort]);
  const [page, more] = useCapped(sorted, 40);
  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  className={`sortable${c.num ? " num" : ""}`}
                  onClick={() => setSort({ key: c.key, desc: sort.key === c.key ? !sort.desc : true })}
                  aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : undefined}
                >
                  {c.label}
                  {sort.key === c.key ? (sort.desc ? " ↓" : " ↑") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.map((t) => (
              <tr key={t.name}>
                {COLS.map((c) => (
                  <td key={c.key} className={c.num ? "num" : c.key === "name" ? "mono" : undefined}>
                    {c.fmt(t)}
                    {c.key === "name" && t.unrotated && (
                      <span className="muted" title="moments and channel statistics computed after undoing the folded rotation">
                        {" "}
                        ↺
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more}
    </>
  );
}

function Spectra({ spectra }: { spectra: KvSpectrum[] }) {
  const [level, setLevel] = useState<"energy_rank_90" | "energy_rank_95" | "energy_rank_99">("energy_rank_90");
  const frac = (s: KvSpectrum, which: "k" | "v" | "kv") => s[which][level] / s[which].full_rank;
  return (
    <Card
      title="K/V spectra"
      extra={
        <select value={level} onChange={(e) => setLevel(e.target.value as typeof level)} aria-label="energy threshold">
          <option value="energy_rank_90">90% energy</option>
          <option value="energy_rank_95">95% energy</option>
          <option value="energy_rank_99">99% energy</option>
        </select>
      }
    >
      <p className="secondary" style={{ marginTop: 0 }}>
        Share of each projection's rank needed to keep that much spectral energy. Lower means K/V compress better (MLA or low-rank KV).
      </p>
      <DotChart
        title="rank needed per attention layer"
        categories={spectra.map((s) => String(s.layer))}
        xLabel="layer"
        yDomain={[0, 1]}
        yFormat={(v) => pct(v, 0)}
        series={[
          { name: "K", color: "var(--series-1)", values: spectra.map((s) => frac(s, "k")) },
          { name: "V", color: "var(--series-2)", values: spectra.map((s) => frac(s, "v")) },
          { name: "[K;V]", color: "var(--series-3)", values: spectra.map((s) => frac(s, "kv")) },
        ]}
        describe={(i) => {
          const s = spectra[i]!;
          return (
            <div className="muted">
              [K;V] {s.kv[level]} of {s.kv.full_rank} dims · eff. rank {s.kv.effective_rank.toFixed(0)}
            </div>
          );
        }}
      />
      <details>
        <summary>Table</summary>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>layer</th>
                <th className="num">K rank</th>
                <th className="num">V rank</th>
                <th className="num">[K;V] rank</th>
                <th className="num">[K;V] full</th>
                <th className="num">[K;V] effective</th>
              </tr>
            </thead>
            <tbody>
              {spectra.map((s) => (
                <tr key={s.layer}>
                  <td>{s.layer}</td>
                  <td className="num">{s.k[level]}</td>
                  <td className="num">{s.v[level]}</td>
                  <td className="num">{s.kv[level]}</td>
                  <td className="num">{s.kv.full_rank}</td>
                  <td className="num">{s.kv.effective_rank.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Card>
  );
}

export function StatsView({ data }: { data: WeightStatsReport }) {
  return (
    <div className="stack">
      {data.kv_spectra.length > 0 && <Spectra spectra={data.kv_spectra} />}
      <Card title={`Tensors (${data.tensors.length})`}>
        <TensorTable rows={data.tensors} />
      </Card>
      {(data.notes.length > 0 || data.skipped.length > 0) && (
        <Card title="Notes">
          <Notes items={[...data.notes, ...data.skipped.map(([n, why]) => `skipped ${n}: ${why}`)]} />
        </Card>
      )}
    </div>
  );
}

export function StatsPage({ model }: { model: string }) {
  const { data, error, loading, run } = useCall(api.stats);
  const [only, setOnly] = useState("");
  const [spectra, setSpectra] = useState(true);
  if (!model) return <div className="empty">Choose a model above.</div>;
  const go = () =>
    void run({
      path: model,
      only: only
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      kv_spectra: spectra,
    });
  return (
    <div className="stack">
      <Card>
        <div className="row">
          <input
            type="text"
            placeholder="only tensors containing… (comma-separated, e.g. attn_k,attn_v)"
            value={only}
            onChange={(e) => setOnly(e.target.value)}
            style={{ flex: 1, minWidth: 240 }}
          />
          <label className="check">
            <input type="checkbox" checked={spectra} onChange={(e) => setSpectra(e.target.checked)} /> K/V spectra
          </label>
          <button className="btn primary" onClick={go} disabled={loading}>
            {loading ? "Streaming weights…" : "Compute statistics"}
          </button>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          Streams every selected tensor from disk one at a time. On a 27B model this takes minutes; filter to the projections you care about
          to go faster.
        </p>
      </Card>
      <ErrorBox error={error} />
      {data && <StatsView data={data} />}
    </div>
  );
}
