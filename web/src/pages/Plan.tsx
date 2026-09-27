import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Catalog, ComputeEstimate, FeaturePlan, Fit, Plan } from "../api/types";
import { RangeBar } from "../components/charts";
import { Card, ErrorBox, KV, Notes, Status, useCall, useStored, type Tone } from "../components/ui";
import { bytes, count, range } from "../lib/format";

const FIT: Record<Fit, [Tone, string]> = {
  yes: ["good", "fits"],
  with_packed_trunk: ["warning", "fits with packed trunk"],
  no: ["critical", "doesn't fit"],
  not_applicable: ["neutral", "n/a"],
};
const RISK: Record<string, Tone> = { low: "good", medium: "warning", high: "critical" };

function Compute({ rows }: { rows: ComputeEstimate[] }) {
  if (rows.length === 0) return null;
  const lows = rows.map((r) => r.gpu_hours.low).filter((v) => v > 0);
  const min = Math.min(...lows, ...rows.map((r) => r.gpu_hours.high)) || 1e-3;
  const max = Math.max(...rows.map((r) => r.gpu_hours.high), min * 10);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>hardware</th>
            <th className="num">GPU-hours</th>
            <th style={{ minWidth: 140 }}>
              <span className="muted">log scale</span>
            </th>
            <th className="num">wall-clock</th>
            <th className="num">peak / GPU</th>
            <th>fits</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const [tone, label] = FIT[c.fits];
            return (
              <tr key={c.profile}>
                <td>{c.profile}</td>
                <td className="num">{range(c.gpu_hours.low, c.gpu_hours.high, "h")}</td>
                <td>
                  <RangeBar
                    low={Math.max(c.gpu_hours.low, min)}
                    high={c.gpu_hours.high}
                    min={min}
                    max={max}
                    label={`${c.profile}: ${range(c.gpu_hours.low, c.gpu_hours.high, "GPU-hours")}`}
                  />
                </td>
                <td className="num">{range(c.wall_hours.low, c.wall_hours.high, "h")}</td>
                <td className="num" title={`packed trunk: ${c.peak_gib_per_gpu_packed_trunk.toFixed(1)} GiB`}>
                  {c.peak_gib_per_gpu.toFixed(1)} GiB
                </td>
                <td>
                  <Status tone={tone}>{label}</Status>
                  {c.notes.length > 0 && (
                    <div className="muted" style={{ fontSize: 12 }}>
                      {c.notes.join(" ")}
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FeatureCard({ f }: { f: FeaturePlan }) {
  const blocked = f.compat.blockers.length > 0;
  const det = f.detection;
  const e = f.estimate;
  return (
    <Card
      title={
        <>
          {f.title} <span className="muted mono">{f.id}</span>
        </>
      }
      extra={
        <div className="row">
          {det.state !== "absent" && <Status tone="neutral">{det.state === "present" ? "already present" : "partially present"}</Status>}
          {blocked ? (
            <Status tone="critical">blocked</Status>
          ) : f.compat.warnings.length ? (
            <Status tone="warning">compatible, with warnings</Status>
          ) : (
            <Status tone="good">compatible</Status>
          )}
          {e && <Status tone={RISK[e.risk.level] ?? "neutral"}>{e.risk.level} risk</Status>}
          {e && <span className="badge">confidence: {e.confidence}</span>}
        </div>
      }
    >
      <div className="stack">
        <p className="secondary" style={{ margin: 0 }}>
          {f.summary}
          {det.state !== "absent" && ` — ${det.detail}`}
        </p>
        {Object.keys(f.params).length > 0 && (
          <div className="mono secondary">
            {Object.entries(f.params)
              .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
              .join("  ")}
          </div>
        )}
        {blocked && <Notes items={f.compat.blockers} />}
        {f.compat.warnings.length > 0 && (
          <div>
            <h3>Warnings</h3>
            <Notes items={f.compat.warnings} />
          </div>
        )}
        {e && (
          <>
            {e.effects.length > 0 && (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>effect</th>
                      <th className="num">before</th>
                      <th className="num">after</th>
                    </tr>
                  </thead>
                  <tbody>
                    {e.effects.map((x) => (
                      <tr key={x.metric}>
                        <td>{x.metric}</td>
                        <td className="num">{effect(x.before, x.unit)}</td>
                        <td className="num">{effect(x.after, x.unit)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <div>
              <h3>Training</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>stage</th>
                      <th className="num">trainable</th>
                      <th className="num">tokens</th>
                      <th>loss</th>
                      <th>data</th>
                    </tr>
                  </thead>
                  <tbody>
                    {e.stages.map((s) => (
                      <tr key={s.name}>
                        <td>
                          <b>{s.name}</b>
                          <div className="secondary">{s.what}</div>
                        </td>
                        <td className="num">{count(s.trainable_params)}</td>
                        <td className="num">
                          {count(s.tokens.low)}–{count(s.tokens.high)}
                        </td>
                        <td>{s.loss}</td>
                        <td className="secondary">{s.data}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <Compute rows={f.compute} />
            <KV
              rows={[
                ["expected", e.risk.expected],
                ["recovery", e.risk.recovery],
              ]}
            />
          </>
        )}
        {(f.surgery.length > 0 || f.export_notes.length > 0 || e) && (
          <details>
            <summary>Surgery, export, assumptions and references</summary>
            <div className="grid">
              {f.surgery.length > 0 && (
                <div>
                  <h3>Surgery</h3>
                  <Notes items={f.surgery} />
                </div>
              )}
              {f.export_notes.length > 0 && (
                <div>
                  <h3>Export</h3>
                  <Notes items={f.export_notes} />
                </div>
              )}
              {e && (
                <div>
                  <h3>Assumptions</h3>
                  <Notes items={e.assumptions} />
                </div>
              )}
              {e && e.references.length > 0 && (
                <div>
                  <h3>References</h3>
                  <Notes items={e.references} />
                </div>
              )}
            </div>
          </details>
        )}
      </div>
    </Card>
  );
}

function effect(v: number, unit: string): string {
  if (unit === "bytes") return bytes(v);
  if (unit.startsWith("params")) return `${count(v)}${unit === "params" ? "" : ` ${unit.slice(7)}`}`;
  return `${Number.isInteger(v) ? v : v.toFixed(2)}${unit ? ` ${unit}` : ""}`;
}

export function PlanView({ plan }: { plan: Plan }) {
  return (
    <div className="stack">
      {plan.features.map((f) => (
        <FeatureCard key={`${f.id}-${JSON.stringify(f.params)}`} f={f} />
      ))}
      <Card title="Cost model">
        <Notes items={plan.cost_model_assumptions} />
      </Card>
    </div>
  );
}

export function PlanPage({ model }: { model: string }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [picked, setPicked] = useStored<Record<string, string>>("mb.plan.features", {});
  const [hardware, setHardware] = useStored<string[]>("mb.plan.hardware", []);
  const [mode, setMode] = useStored<"form" | "recipe">("mb.plan.mode", "form");
  const [recipe, setRecipe] = useStored(
    "mb.plan.recipe",
    '[source]\npath = "models/model.gguf"\n\n[hardware]\nprofiles = ["1x24GB"]\n\n[[feature]]\nid = "kv-share"\ngroup = 2\n',
  );
  const { data, error, loading, run } = useCall(api.plan);
  const [catErr, setCatErr] = useState<unknown>(null);
  useEffect(() => {
    api.catalog().then(setCatalog, setCatErr);
  }, []);

  const go = () =>
    void run(
      mode === "recipe"
        ? { path: model || null, recipe, features: [], hardware: [] }
        : {
            path: model,
            recipe: null,
            features: Object.entries(picked).map(([id, params]) => (params.trim() ? `${id}:${params.trim()}` : id)),
            hardware,
          },
    );

  return (
    <div className="stack">
      <Card
        title="What would it take?"
        extra={
          <div className="tabs" role="tablist">
            <button className="tab" role="tab" aria-selected={mode === "form"} onClick={() => setMode("form")}>
              Pick features
            </button>
            <button className="tab" role="tab" aria-selected={mode === "recipe"} onClick={() => setMode("recipe")}>
              Recipe (TOML)
            </button>
          </div>
        }
      >
        <ErrorBox error={catErr} />
        {mode === "form" && catalog && (
          <div className="stack">
            <div className="table-wrap">
              <table>
                <tbody>
                  {catalog.features.map((f) => {
                    const on = f.id in picked;
                    return (
                      <tr key={f.id}>
                        <td style={{ width: 28 }}>
                          <input
                            type="checkbox"
                            aria-label={f.title}
                            checked={on}
                            onChange={(e) => {
                              const next = { ...picked };
                              if (e.target.checked) next[f.id] = "";
                              else delete next[f.id];
                              setPicked(next);
                            }}
                          />
                        </td>
                        <td>
                          <b>{f.title}</b> <span className="muted mono">{f.id}</span>
                          <div className="secondary">{f.summary}</div>
                        </td>
                        <td style={{ width: "32%" }}>
                          {on && (
                            <input
                              type="text"
                              style={{ width: "100%" }}
                              placeholder="parameters, e.g. group=2 or from=/path/to/reference"
                              value={picked[f.id]}
                              onChange={(e) => setPicked({ ...picked, [f.id]: e.target.value })}
                            />
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="row">
              <span className="secondary">Hardware:</span>
              {catalog.hardware.map((h) => (
                <label key={h.id} className="check" title={h.description}>
                  <input
                    type="checkbox"
                    checked={hardware.includes(h.id)}
                    onChange={(e) => setHardware(e.target.checked ? [...hardware, h.id] : hardware.filter((x) => x !== h.id))}
                  />
                  {h.id}
                </label>
              ))}
              <span className="muted">(none checked: all)</span>
            </div>
            <p className="muted" style={{ margin: 0 }}>
              No features checked evaluates the whole catalog with default parameters.
            </p>
          </div>
        )}
        {mode === "recipe" && (
          <div className="stack">
            <textarea rows={12} value={recipe} onChange={(e) => setRecipe(e.target.value)} spellCheck={false} aria-label="recipe" />
            <p className="muted" style={{ margin: 0 }}>
              Same format as <code>modelbuilder plan --recipe</code>. The model chosen above, if any, overrides <code>[source]</code>.
            </p>
          </div>
        )}
        <div className="row" style={{ marginTop: 12 }}>
          <button className="btn primary" onClick={go} disabled={loading || (mode === "form" && !model)}>
            {loading ? "Planning…" : "Plan"}
          </button>
          {mode === "form" && !model && <span className="muted">Choose a model above first.</span>}
        </div>
      </Card>
      <ErrorBox error={error} />
      {data && <PlanView plan={data} />}
    </div>
  );
}
