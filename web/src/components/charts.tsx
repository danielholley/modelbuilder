// Small SVG charts. Series colors come from the categorical slots in fixed
// order (--series-1..3); text stays in ink tokens; grid and axes are recessive.
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Point } from "../lib/jobs";
import { extent, linear, niceTicks, tickLabel } from "../lib/scale";

export interface Series {
  name: string;
  /** A CSS color, normally `var(--series-N)`. */
  color: string;
  points: Point[];
}

const M = { top: 12, right: 12, bottom: 28, left: 44 };

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => e && setW(Math.max(240, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

function Legend({ series, dot }: { series: { name: string; color: string }[]; dot?: boolean }) {
  if (series.length < 2) return null;
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.name}>
          <span className={dot ? "swatch dot" : "swatch"} style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}

/** Direct labels at each series' last point, pushed apart so they never overlap. */
function endLabels(items: { name: string; p: Point | undefined }[], x: (v: number) => number, y: (v: number) => number, bottom: number) {
  const gap = 13;
  const placed = items
    .filter((i): i is { name: string; p: Point } => !!i.p)
    .map((i) => ({ name: i.name, x: x(i.p.x), y: y(i.p.y) }))
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < placed.length; k++) {
    const prev = placed[k - 1]!;
    const cur = placed[k]!;
    if (cur.y - prev.y < gap) cur.y = prev.y + gap;
  }
  // Keep them inside the plot: if the lowest ran past the bottom, push the stack back up.
  for (let k = placed.length - 1; k >= 0; k--) {
    const limit = k === placed.length - 1 ? bottom - 4 : placed[k + 1]!.y - gap;
    if (placed[k]!.y > limit) placed[k]!.y = limit;
  }
  return placed;
}

function nearest(points: Point[], x: number): Point | undefined {
  let best: Point | undefined;
  for (const p of points) if (!best || Math.abs(p.x - x) < Math.abs(best.x - x)) best = p;
  return best;
}

/** Line chart over a shared numeric x (e.g. training step), one y scale. */
export function LineChart({
  series,
  height = 220,
  xLabel,
  yFormat = tickLabel,
  yDomain,
  title,
}: {
  series: Series[];
  height?: number;
  xLabel?: string;
  yFormat?: (v: number) => string;
  yDomain?: [number, number];
  title?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.points);
  if (all.length === 0) return <div className="empty">No data yet.</div>;

  const labelRoom = series.length <= 4 && series.length > 1 ? 72 : 0;
  const iw = width - M.left - M.right - labelRoom;
  const ih = height - M.top - M.bottom;
  const [x0, x1] = extent(all.map((p) => p.x));
  const yExt = yDomain ?? extent(all.map((p) => p.y));
  const yTicks = niceTicks(yExt[0], yExt[1], 4);
  const xTicks = niceTicks(x0, x1, Math.max(2, Math.floor(iw / 90)));
  const ys: [number, number] = yDomain ?? [yTicks[0]!, yTicks[yTicks.length - 1]!];
  const x = linear([x0, x1 === x0 ? x0 + 1 : x1], [0, iw]);
  const y = linear(ys, [ih, 0]);
  const path = (pts: Point[]) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
  const hx = hover === null ? null : nearest(all, hover)?.x;

  return (
    <div className="chart" ref={ref} aria-label={title}>
      <Legend series={series} />
      <svg width={width} height={height} role="img" aria-label={title}>
        <g transform={`translate(${M.left},${M.top})`}>
          {yTicks
            .filter((t) => t >= ys[0] && t <= ys[1])
            .map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x2={iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end">
                  {yFormat(t)}
                </text>
              </g>
            ))}
          <line y1={ih} y2={ih} x2={iw} stroke="var(--axis)" />
          {xTicks
            .filter((t) => t >= x0 && t <= x1)
            .map((t) => (
              <text key={t} x={x(t)} y={ih + 16} textAnchor="middle">
                {tickLabel(t)}
              </text>
            ))}
          {xLabel && (
            <text x={iw} y={ih + 16} dx={labelRoom} textAnchor="end">
              {xLabel}
            </text>
          )}
          {series.map((s) => (
            <g key={s.name}>
              <path d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {s.points.length <= 40 &&
                s.points.map((p) => (
                  <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                ))}
            </g>
          ))}
          {labelRoom > 0 &&
            endLabels(
              series.map((s) => ({ name: s.name, p: s.points[s.points.length - 1] })),
              x,
              y,
              ih,
            ).map((l) => (
              <text key={l.name} x={l.x + 8} y={l.y} dy="0.32em" style={{ fill: "var(--text-2)" }}>
                {l.name}
              </text>
            ))}
          {hx !== null && hx !== undefined && (
            <g>
              <line x1={x(hx)} x2={x(hx)} y2={ih} stroke="var(--axis)" />
              {series.map((s) => {
                const p = nearest(s.points, hx);
                return p && p.x === hx ? (
                  <circle key={s.name} cx={x(p.x)} cy={y(p.y)} r={5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                ) : null;
              })}
            </g>
          )}
          <rect
            width={iw}
            height={ih}
            fill="transparent"
            onMouseMove={(e) => {
              const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
              setHover(x0 + ((e.clientX - r.left) / iw) * (x1 - x0));
            }}
            onMouseLeave={() => setHover(null)}
          />
        </g>
      </svg>
      {hx !== null && hx !== undefined && (
        <Tooltip left={M.left + x(hx)} width={width}>
          <div className="t-head">
            {xLabel ?? "x"} {hx}
          </div>
          {series.map((s) => {
            const p = nearest(s.points, hx);
            return p && p.x === hx ? (
              <div className="t-row" key={s.name}>
                <span className="swatch" style={{ background: s.color }} />
                {s.name} <b>{yFormat(p.y)}</b>
              </div>
            ) : null;
          })}
        </Tooltip>
      )}
    </div>
  );
}

function Tooltip({ left, width, children }: { left: number; width: number; children: ReactNode }) {
  const flip = left > width * 0.6;
  return (
    <div className="tooltip" style={{ top: 28, ...(flip ? { right: width - left + 12 } : { left: left + 12 }) }}>
      {children}
    </div>
  );
}

export interface DotSeries {
  name: string;
  color: string;
  /** One value per category (null: none). */
  values: (number | null)[];
}

/** Dots over discrete categories (e.g. attention layers), one y scale, per-dot hover. */
export function DotChart({
  categories,
  series,
  height = 220,
  yFormat = tickLabel,
  yDomain,
  xLabel,
  describe,
  title,
}: {
  categories: string[];
  series: DotSeries[];
  height?: number;
  yFormat?: (v: number) => string;
  yDomain?: [number, number];
  xLabel?: string;
  /** Extra tooltip lines for category i. */
  describe?: (i: number) => ReactNode;
  title?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  if (categories.length === 0) return <div className="empty">No data.</div>;
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const vals = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const yExt = yDomain ?? extent(vals);
  const yTicks = niceTicks(yExt[0], yExt[1], 4);
  const ys: [number, number] = yDomain ?? [yTicks[0]!, yTicks[yTicks.length - 1]!];
  const y = linear(ys, [ih, 0]);
  const band = iw / categories.length;
  const cx = (i: number) => band * (i + 0.5);
  const every = Math.ceil(categories.length / Math.max(1, Math.floor(iw / 36)));

  return (
    <div className="chart" ref={ref}>
      <Legend series={series} dot />
      <svg width={width} height={height} role="img" aria-label={title}>
        <g transform={`translate(${M.left},${M.top})`}>
          {yTicks
            .filter((t) => t >= ys[0] && t <= ys[1])
            .map((t) => (
              <g key={t} transform={`translate(0,${y(t)})`}>
                <line x2={iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end">
                  {yFormat(t)}
                </text>
              </g>
            ))}
          <line y1={ih} y2={ih} x2={iw} stroke="var(--axis)" />
          {categories.map((c, i) =>
            i % every === 0 ? (
              <text key={c} x={cx(i)} y={ih + 16} textAnchor="middle">
                {c}
              </text>
            ) : null,
          )}
          {xLabel && (
            <text x={iw} y={ih + 28} textAnchor="end">
              {xLabel}
            </text>
          )}
          {hover !== null && <rect x={band * hover} width={band} height={ih} fill="var(--surface-2)" />}
          {series.map((s, si) =>
            s.values.map((v, i) =>
              v === null ? null : (
                <circle
                  key={`${si}-${i}`}
                  cx={cx(i) + (si - (series.length - 1) / 2) * Math.min(6, band / 4)}
                  cy={y(v)}
                  r={4}
                  fill={s.color}
                  stroke="var(--surface)"
                  strokeWidth={2}
                />
              ),
            ),
          )}
          {categories.map((c, i) => (
            <rect
              key={c}
              x={band * i}
              width={band}
              height={ih}
              fill="transparent"
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </g>
      </svg>
      {hover !== null && (
        <Tooltip left={M.left + cx(hover)} width={width}>
          <div className="t-head">
            {xLabel ?? ""} {categories[hover]}
          </div>
          {series.map((s) => {
            const v = s.values[hover];
            return v === null || v === undefined ? null : (
              <div className="t-row" key={s.name}>
                <span className="swatch" style={{ background: s.color }} />
                {s.name} <b>{yFormat(v)}</b>
              </div>
            );
          })}
          {describe?.(hover)}
        </Tooltip>
      )}
    </div>
  );
}

/** Horizontal magnitude bars (one hue), labeled; for part-of-whole lists like a parameter breakdown. */
export function Bars({ rows, format }: { rows: { label: string; value: number }[]; format: (v: number) => string }) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <div className="bars">
      {rows.map((r) => (
        <div key={r.label} style={{ display: "contents" }} title={`${r.label}: ${format(r.value)}`}>
          <span className="secondary">{r.label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${(100 * r.value) / max}%` }} />
          </div>
          <span className="mono">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** A low–high range on a log scale shared across rows (e.g. GPU-hours per profile). */
export function RangeBar({ low, high, min, max, label }: { low: number; high: number; min: number; max: number; label: string }) {
  const lg = (v: number) => Math.log10(Math.max(v, 1e-9));
  const span = lg(max) - lg(min) || 1;
  const a = ((lg(low) - lg(min)) / span) * 100;
  const b = ((lg(high) - lg(min)) / span) * 100;
  return (
    <div className="range-bar" title={label} aria-label={label}>
      <span style={{ left: `${a}%`, width: `${Math.max(b - a, 1.5)}%` }} />
    </div>
  );
}
