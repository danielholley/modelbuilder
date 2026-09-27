import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

export function Card({ title, extra, children }: { title?: ReactNode; extra?: ReactNode; children: ReactNode }) {
  return (
    <section className="card">
      {(title || extra) && (
        <div className="card-head">
          {title && <h2>{title}</h2>}
          <span className="spacer" />
          {extra}
        </div>
      )}
      {children}
    </section>
  );
}

export function Tile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <dt>{k}</dt>
          <dd>{v ?? <span className="muted">–</span>}</dd>
        </div>
      ))}
    </dl>
  );
}

export type Tone = "good" | "warning" | "serious" | "critical" | "neutral";
const ICON: Record<Tone, string> = { good: "✓", warning: "!", serious: "!", critical: "✕", neutral: "•" };

/** Status never rides on color alone: an icon and a label go with it. */
export function Status({ tone, children }: { tone: Tone; children: ReactNode }) {
  const color = tone === "neutral" ? "var(--muted)" : `var(--${tone})`;
  return (
    <span className="badge">
      <span aria-hidden style={{ color, fontWeight: 700 }}>
        {ICON[tone]}
      </span>
      {children}
    </span>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="error">{msg}</div>;
}

export function Notes({ items }: { items: string[] }) {
  if (items.length === 0) return <span className="muted">none</span>;
  return (
    <ul className="notes">
      {items.map((n, i) => (
        <li key={i}>{n}</li>
      ))}
    </ul>
  );
}

/** Runs an async call on demand; keeps the latest result and error, aborts superseded calls. */
export function useCall<A extends unknown[], T>(fn: (...args: [...A, AbortSignal]) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const ctl = useRef<AbortController | null>(null);
  const run = useCallback(
    async (...args: A) => {
      ctl.current?.abort();
      const c = new AbortController();
      ctl.current = c;
      setLoading(true);
      setError(null);
      try {
        const r = await fn(...args, c.signal);
        if (!c.signal.aborted) setData(r);
      } catch (e) {
        if (!c.signal.aborted) setError(e);
      } finally {
        if (!c.signal.aborted) setLoading(false);
      }
    },
    [fn],
  );
  useEffect(() => () => ctl.current?.abort(), []);
  return { data, error, loading, run, setData };
}

/** localStorage-backed state; falls back to memory when storage is unavailable. */
export function useStored<T>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => {
    try {
      const s = localStorage.getItem(key);
      return s === null ? initial : (JSON.parse(s) as T);
    } catch {
      return initial;
    }
  });
  const set = useCallback(
    (nv: T) => {
      setV(nv);
      try {
        localStorage.setItem(key, JSON.stringify(nv));
      } catch {
        // private mode or storage blocked: keep it in memory only
      }
    },
    [key],
  );
  return [v, set];
}

/** Shows the first `initial` rows of a long list, with a button for the rest. */
export function useCapped<T>(rows: T[], initial = 50): [T[], ReactNode] {
  const [all, setAll] = useState(false);
  const shown = all ? rows : rows.slice(0, initial);
  const more =
    rows.length > initial ? (
      <div className="row" style={{ marginTop: 8 }}>
        <span className="muted">{all ? `All ${rows.length} rows` : `${initial} of ${rows.length} rows`}</span>
        <button className="btn" onClick={() => setAll(!all)}>
          {all ? `Show first ${initial}` : `Show all ${rows.length}`}
        </button>
      </div>
    ) : null;
  return [shown, more];
}
