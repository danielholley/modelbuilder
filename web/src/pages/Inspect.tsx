import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { InspectResponse, LayerPattern, LayerRow, ParamBreakdown, TensorRow } from "../api/types";
import { Bars } from "../components/charts";
import { Card, ErrorBox, KV, Notes, Status, Tile, useCall, useCapped } from "../components/ui";
import { bytes, count, dtype } from "../lib/format";

export function patternText(p: LayerPattern): string {
  const list = (v: [number, string][]) => v.map(([n, l]) => `${n} × ${l}`).join(", ");
  const head = p.prefix.length ? `[${list(p.prefix)}] then ` : "";
  const only = p.block.length === 1 && p.block[0]![0] === 1 ? p.block[0] : undefined;
  return head + (only ? `${p.repeats} × ${only[1]}` : `${p.repeats} × [${list(p.block)}]`);
}

const PARAM_PARTS: [keyof ParamBreakdown, string][] = [
  ["embedding", "embedding"],
  ["lm_head", "LM head"],
  ["attention", "attention"],
  ["linear_attention", "linear attention"],
  ["ffn_dense", "dense FFN"],
  ["moe_routed_experts", "routed experts"],
  ["moe_shared_experts", "shared experts"],
  ["router", "router"],
  ["norms", "norms"],
  ["mtp", "MTP"],
  ["multimodal", "multimodal"],
  ["other", "other"],
];

const MIXER_COLOR: Record<string, string> = {
  attention: "var(--series-1)",
  linear_attention: "var(--series-2)",
  unknown: "var(--muted)",
};
const MIXER_NAME: Record<string, string> = { attention: "full attention", linear_attention: "linear attention", unknown: "unknown" };

function LayerStrip({ layers }: { layers: LayerRow[] }) {
  const kinds = [...new Set(layers.map((l) => l.layer.mixer.type))];
  return (
    <div>
      <div className="legend">
        {kinds.map((k) => (
          <span key={k}>
            <span className="swatch dot" style={{ background: MIXER_COLOR[k] }} />
            {MIXER_NAME[k]} ({layers.filter((l) => l.layer.mixer.type === k).length})
          </span>
        ))}
      </div>
      <div className="layers">
        {layers.map((l) => (
          <div
            key={l.layer.index}
            className="layer-cell"
            style={{ background: MIXER_COLOR[l.layer.mixer.type] }}
            title={`layer ${l.layer.index}: ${l.label} · ${count(l.layer.params)} params`}
          />
        ))}
      </div>
    </div>
  );
}

function Tensors({ rows }: { rows: TensorRow[] }) {
  const [filter, setFilter] = useState("");
  const shown = useMemo(() => {
    const f = filter.toLowerCase();
    return rows.filter(
      (r) => !f || r.info.name.toLowerCase().includes(f) || r.role.kind.includes(f) || dtype(r.info.dtype).toLowerCase().includes(f),
    );
  }, [rows, filter]);
  const [page, more] = useCapped(shown, 40);
  return (
    <Card
      title={`Tensors (${rows.length})`}
      extra={<input type="text" placeholder="filter by name, kind or dtype" value={filter} onChange={(e) => setFilter(e.target.value)} />}
    >
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>name</th>
              <th>dtype</th>
              <th>shape</th>
              <th className="num">bytes</th>
              <th>role</th>
            </tr>
          </thead>
          <tbody>
            {page.map((r) => (
              <tr key={r.info.name}>
                <td className="mono">{r.info.name}</td>
                <td>{dtype(r.info.dtype)}</td>
                <td className="mono">[{r.info.shape.join(", ")}]</td>
                <td className="num">
                  {bytes(r.info.n_bytes)}
                  {!r.info.bytes_exact && <span className="muted"> ≈</span>}
                </td>
                <td className="secondary">
                  {r.role.component}
                  {r.role.layer !== null ? ` ${r.role.layer}` : ""} · {r.role.kind}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more}
    </Card>
  );
}

export function InspectView({ data }: { data: InspectResponse }) {
  const { report: r } = data;
  const a = r.architecture;
  const q = r.quantization;
  const kv = r.kv_cache;
  const p = r.provenance;
  const parts = PARAM_PARTS.map(([k, label]) => ({ label, value: r.params[k] as number })).filter((x) => x.value > 0);
  return (
    <div className="stack">
      <div className="tiles">
        <Tile
          label="parameters"
          value={count(r.params.total)}
          sub={r.params.active_per_token !== r.params.total ? `${count(r.params.active_per_token)} active / token` : undefined}
        />
        <Tile label="size on disk" value={bytes(q.total_bytes)} sub={`${q.bits_per_param.toFixed(2)} bits / param`} />
        <Tile label="layers" value={a.num_layers} sub={a.mtp_modules ? `+ ${a.mtp_modules} MTP` : undefined} />
        <Tile label="hidden · vocab" value={count(a.hidden_size)} sub={`vocab ${count(a.vocab_size)}`} />
        <Tile
          label="context"
          value={count(a.max_positions)}
          sub={p.original_context ? `trained ${count(p.original_context)}` : undefined}
        />
        <Tile label="KV cache / token" value={bytes(kv.precisions[0]?.bytes_per_token)} sub={kv.precisions[0]?.name} />
      </div>
      {r.warnings.length > 0 && (
        <Card title={<Status tone="warning">Warnings</Status>}>
          <Notes items={r.warnings} />
        </Card>
      )}
      <div className="grid">
        <Card title="Architecture">
          <KV
            rows={[
              ["family", a.family ?? "–"],
              ["architectures", a.architectures.join(", ") || "–"],
              ["pattern", <code key="p">{patternText(a.layer_pattern)}</code>],
              ["RoPE θ", a.rope_theta ?? "–"],
              ["RoPE scaling", a.rope_scaling ? <code key="rs">{JSON.stringify(a.rope_scaling)}</code> : "none"],
              ["tied embeddings", a.tie_word_embeddings === null ? "–" : a.tie_word_embeddings ? "yes" : "no"],
              ["multimodal", a.multimodal ? "yes" : "no"],
              [
                "source",
                <span key="s" className="mono">{`${r.source.format} · ${r.source.files} file(s) · ${r.source.tensors} tensors`}</span>,
              ],
            ]}
          />
        </Card>
        <Card title="Parameters">
          <Bars rows={parts} format={count} />
        </Card>
      </div>
      <Card title="Layers">
        <LayerStrip layers={data.layers} />
      </Card>
      <div className="grid">
        <Card title="Quantization">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>dtype</th>
                  <th className="num">tensors</th>
                  <th className="num">params</th>
                  <th className="num">bytes</th>
                  <th className="num">bits/param</th>
                </tr>
              </thead>
              <tbody>
                {q.by_dtype.map((d) => (
                  <tr key={d.dtype}>
                    <td>{d.dtype}</td>
                    <td className="num">{d.tensors}</td>
                    <td className="num">{count(d.params)}</td>
                    <td className="num">
                      {bytes(d.bytes)}
                      {!d.bytes_exact && <span className="muted"> ≈</span>}
                    </td>
                    <td className="num">{d.bits_per_param.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="secondary">Trunk: {q.trunk_bits_per_param.toFixed(2)} bits/param.</p>
          {q.rotation && (
            <KV
              rows={[
                [
                  "rotation",
                  `${q.rotation.scheme}${q.rotation.version !== null ? ` v${q.rotation.version}` : ""}, block ${q.rotation.block_size ?? "–"}`,
                ],
                ["rotated tensors", `${q.rotation.rotated_tensors} (+${q.rotation.inverse_tensors} inverse)`],
                ["signs", q.rotation.sign_mode],
              ]}
            />
          )}
          {q.notes.length > 0 && <Notes items={q.notes} />}
        </Card>
        <Card title="KV cache">
          <p className="secondary" style={{ marginTop: 0 }}>
            {kv.global_layers} global, {kv.windowed_layers} windowed, {kv.linear_layers} linear layers · {count(kv.elements_per_token)}{" "}
            elements/token{kv.context ? ` · at ${count(kv.context)} tokens` : ""}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>precision</th>
                  <th className="num">per token</th>
                  <th className="num">at context</th>
                </tr>
              </thead>
              <tbody>
                {kv.precisions.map((k) => (
                  <tr key={k.name}>
                    <td>{k.name}</td>
                    <td className="num">{bytes(k.bytes_per_token)}</td>
                    <td className="num">{bytes(k.bytes_at_context)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {kv.linear_state_bytes !== null && (
            <p className="secondary">Linear-attention state: {bytes(kv.linear_state_bytes)} per sequence.</p>
          )}
          <details>
            <summary>Assumptions</summary>
            <Notes items={kv.assumptions} />
          </details>
        </Card>
        <Card title="Provenance">
          <KV
            rows={[
              ["name", p.name ?? "–"],
              ["license", p.license ?? "–"],
              ["lineage", p.lineage_hint],
              ["base models", p.base_models.join(", ") || "–"],
              ["chat template", p.has_chat_template ? "yes" : "no"],
              ["dtype", p.dtype ?? "–"],
              ["tags", p.tags.join(", ") || "–"],
            ]}
          />
        </Card>
      </div>
      {data.tensors && <Tensors rows={data.tensors} />}
    </div>
  );
}

export function InspectPage({ model }: { model: string }) {
  const { data, error, loading, run } = useCall(api.inspect);
  const [context, setContext] = useState("");
  useEffect(() => {
    if (model) void run({ path: model, context: null, tensors: true });
  }, [model, run]);
  if (!model) return <div className="empty">Choose a model above: a .gguf file or a Hugging Face directory.</div>;
  return (
    <div className="stack">
      <div className="row">
        <label className="secondary">KV context</label>
        <input
          type="number"
          min={1}
          placeholder="model max"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          style={{ width: 140 }}
        />
        <button
          className="btn"
          onClick={() => void run({ path: model, context: context ? Number(context) : null, tensors: true })}
          disabled={loading}
        >
          {loading ? "Reading…" : "Refresh"}
        </button>
      </div>
      <ErrorBox error={error} />
      {data && <InspectView data={data} />}
    </div>
  );
}
