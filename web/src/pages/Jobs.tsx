import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamJob } from "../api/client";
import type { EventRecord, JobStatus, JobSummary } from "../api/types";
import { LineChart } from "../components/charts";
import { PathPicker } from "../components/PathPicker";
import { Card, ErrorBox, KV, Status, Tile, useStored, type Tone } from "../components/ui";
import { count, duration, pct } from "../lib/format";
import { applyUpdate, curves, START, type JobView } from "../lib/jobs";

const STATUS: Record<JobStatus, Tone> = { running: "neutral", succeeded: "good", failed: "critical", cancelled: "warning" };

export function JobStatusBadge({ status }: { status: JobStatus }) {
  return <Status tone={STATUS[status]}>{status}</Status>;
}

function sourceLabel(j: JobSummary): string {
  return j.source.kind === "spec" ? j.source.spec_path : j.source.events_path;
}

function eventLine(e: EventRecord): string {
  switch (e.event) {
    case "started":
      return `started ${e.job_id} on ${e.device}: ${count(e.trainable_params)} trainable, ${count(e.train_tokens)} train / ${count(e.eval_tokens)} eval tokens`;
    case "progress":
      return `step ${e.step}/${e.steps}  loss ${e.loss.toFixed(4)}  acc ${pct(e.accuracy)}  lr ${e.lr?.toExponential(2) ?? "–"}  ${e.tokens_per_s?.toFixed(0) ?? "–"} tok/s`;
    case "eval":
      return `eval @ ${e.step}  loss ${e.loss.toFixed(4)}  top-1 ${pct(e.accuracy)}`;
    case "checkpoint":
      return `checkpoint @ ${e.step}: ${e.path}`;
    case "finished":
      return `finished: ${e.status}`;
    case "error":
      return `error: ${e.message}`;
  }
}

function JobDetail({ view, onCancel }: { view: JobView; onCancel: () => void }) {
  const s = view.summary;
  const c = curves(view.events);
  const p = s.latest_progress?.event === "progress" ? s.latest_progress : null;
  const ev = s.latest_eval?.event === "eval" ? s.latest_eval : null;
  const started = view.events.find((e) => e.event === "started");
  const end = s.finished_at ?? Date.now() / 1000;
  const eta = p && p.tokens_per_s && p.step > 0 ? ((end - s.started_at) / p.step) * (p.steps - p.step) : null;
  return (
    <div className="stack">
      <Card
        title={
          <>
            {s.job_id ?? `job ${s.id}`} <JobStatusBadge status={s.status} />
          </>
        }
        extra={
          s.status === "running" && (
            <button className="btn" onClick={onCancel}>
              {s.source.kind === "spec" ? "Cancel job" : "Stop following"}
            </button>
          )
        }
      >
        <div className="tiles">
          <Tile label="step" value={p ? `${p.step} / ${p.steps}` : "–"} sub={p ? pct(p.step / p.steps, 0) : undefined} />
          <Tile
            label="train loss"
            value={p ? p.loss.toFixed(4) : "–"}
            sub={p?.accuracy !== undefined && p?.accuracy !== null ? `acc ${pct(p.accuracy)}` : undefined}
          />
          <Tile label="eval loss" value={ev ? ev.loss.toFixed(4) : "–"} sub={ev ? `top-1 ${pct(ev.accuracy)} @ ${ev.step}` : undefined} />
          <Tile
            label="throughput"
            value={p?.tokens_per_s ? `${count(p.tokens_per_s)} tok/s` : "–"}
            sub={started?.event === "started" ? started.device : undefined}
          />
          <Tile
            label="elapsed"
            value={duration(end - s.started_at)}
            sub={s.status === "running" && eta !== null ? `~${duration(eta)} left` : undefined}
          />
        </div>
        <div style={{ marginTop: 12 }}>
          <KV
            rows={[
              [
                s.source.kind === "spec" ? "spec" : "events",
                <span key="s" className="mono">
                  {sourceLabel(s)}
                </span>,
              ],
              ...Object.entries(s.outputs).map(([k, v]): [string, React.ReactNode] => [
                k,
                <span key={k} className="mono">
                  {v}
                </span>,
              ]),
              ...(s.error
                ? ([
                    [
                      "error",
                      <span key="e" style={{ color: "var(--critical)" }}>
                        {s.error}
                      </span>,
                    ],
                  ] as [string, React.ReactNode][])
                : []),
            ]}
          />
        </div>
      </Card>
      <div className="grid">
        <Card title="Loss">
          <LineChart
            title="loss by step"
            xLabel="step"
            yFormat={(v) => v.toFixed(2)}
            series={[
              { name: "train", color: "var(--series-1)", points: c.trainLoss },
              { name: "eval", color: "var(--series-2)", points: c.evalLoss },
            ].filter((x) => x.points.length > 0)}
          />
        </Card>
        <Card title="Top-1 accuracy (t+2)">
          <LineChart
            title="accuracy by step"
            xLabel="step"
            yDomain={[0, 1]}
            yFormat={(v) => pct(v, 0)}
            series={[
              { name: "train", color: "var(--series-1)", points: c.trainAcc },
              { name: "eval", color: "var(--series-2)", points: c.evalAcc },
            ].filter((x) => x.points.length > 0)}
          />
        </Card>
      </div>
      <div className="grid">
        <Card title={`Events (${view.events.length})`}>
          <pre className="log">{view.events.map(eventLine).join("\n") || "none yet"}</pre>
        </Card>
        <Card title={`Log (${view.log.length})`}>
          <pre className="log">{view.log.slice(-400).join("\n") || "empty"}</pre>
        </Card>
      </div>
    </div>
  );
}

export function JobsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [views, setViews] = useState<Record<number, JobView>>({});
  const [kind, setKind] = useStored<"spec" | "events">("mb.jobs.kind", "spec");
  const [path, setPath] = useStored("mb.jobs.path", "");
  const [python, setPython] = useStored("mb.jobs.python", "");
  const [error, setError] = useState<unknown>(null);
  const last = useRef(START);

  const refresh = useCallback(() => {
    api.jobs().then((j) => {
      setJobs(j);
      setSelected((s) => s ?? j[j.length - 1]?.id ?? null);
    }, setError);
  }, []);
  useEffect(refresh, [refresh]);

  useEffect(() => {
    if (selected === null) return;
    const have = views[selected];
    last.current = have?.cursor ?? START;
    if (have && have.summary.status !== "running") return;
    return streamJob(selected, last.current, (u) => {
      const since = last.current;
      last.current = u.cursor;
      setViews((v) => ({ ...v, [selected]: applyUpdate(v[selected], u, since) }));
      setJobs((js) => js.map((j) => (j.id === u.summary.id ? u.summary : j)));
    });
    // Re-subscribe only when the selection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  const start = async () => {
    setError(null);
    try {
      const j = await api.startJob(kind === "spec" ? { kind, spec_path: path, python: python || null } : { kind, events_path: path });
      setJobs((js) => [...js, j]);
      setSelected(j.id);
    } catch (e) {
      setError(e);
    }
  };

  const view = selected !== null ? views[selected] : undefined;
  return (
    <div className="stack">
      <Card title="Start or follow a job">
        <div className="stack">
          <div className="row">
            <label className="check">
              <input type="radio" checked={kind === "spec"} onChange={() => setKind("spec")} /> Run a job spec here
            </label>
            <label className="check">
              <input type="radio" checked={kind === "events"} onChange={() => setKind("events")} /> Follow an events file
            </label>
          </div>
          <div className="row">
            <PathPicker
              value={path}
              onChange={setPath}
              accept={kind === "spec" ? ["job_spec", "other"] : ["events", "other"]}
              placeholder={
                kind === "spec"
                  ? "runs/<name>/job.json (from `modelbuilder job mtp-align --emit-only`)"
                  : "events.jsonl (from `modelbuilder-train run job.json --events FILE`)"
              }
            />
            {kind === "spec" && (
              <input
                type="text"
                placeholder="python (default: server's)"
                value={python}
                onChange={(e) => setPython(e.target.value)}
                style={{ width: 220 }}
              />
            )}
            <button className="btn primary" onClick={() => void start()} disabled={!path}>
              {kind === "spec" ? "Start" : "Follow"}
            </button>
          </div>
        </div>
      </Card>
      <ErrorBox error={error} />
      {jobs.length > 0 && (
        <Card
          title="Jobs"
          extra={
            <button className="btn" onClick={refresh}>
              Refresh
            </button>
          }
        >
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>job</th>
                  <th>status</th>
                  <th>source</th>
                  <th className="num">events</th>
                  <th>started</th>
                </tr>
              </thead>
              <tbody>
                {[...jobs].reverse().map((j) => (
                  <tr
                    key={j.id}
                    onClick={() => setSelected(j.id)}
                    style={{ cursor: "pointer", fontWeight: j.id === selected ? 600 : undefined }}
                  >
                    <td>{j.id}</td>
                    <td>{j.job_id ?? "–"}</td>
                    <td>
                      <JobStatusBadge status={j.status} />
                    </td>
                    <td className="mono">{sourceLabel(j)}</td>
                    <td className="num">{j.events}</td>
                    <td>{new Date(j.started_at * 1000).toLocaleTimeString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {view && (
        <JobDetail
          view={view}
          onCancel={() => api.cancelJob(view.summary.id).then((s) => setJobs((js) => js.map((j) => (j.id === s.id ? s : j))), setError)}
        />
      )}
      {jobs.length === 0 && (
        <div className="empty">No jobs yet. Jobs run on the machine the server runs on, and last until the server stops.</div>
      )}
    </div>
  );
}
