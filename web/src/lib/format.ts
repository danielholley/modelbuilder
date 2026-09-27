// Number formatting shared by every page. Mirrors the CLI's renderer
// (crates/modelbuilder/src/render.rs) so both UIs print the same figures.

const UNITS = ["", "K", "M", "B", "T"];

/** 1234567 → "1.23M". */
export function count(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  let v = Math.abs(n);
  let i = 0;
  while (v >= 1000 && i < UNITS.length - 1) {
    v /= 1000;
    i++;
  }
  const s = i === 0 ? String(Math.round(v)) : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
  return (n < 0 ? "-" : "") + s + UNITS[i];
}

const BYTES = ["B", "KiB", "MiB", "GiB", "TiB"];

/** Binary units: 1536 → "1.50 KiB". */
export function bytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "–";
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1024 && i < BYTES.length - 1) {
    v /= 1024;
    i++;
  }
  return i === 0 ? `${Math.round(v)} B` : `${v.toFixed(2)} ${BYTES[i]}`;
}

export function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined || !Number.isFinite(x) ? "–" : `${(100 * x).toFixed(digits)}%`;
}

/** A low–high range, e.g. GPU-hours: "3.2–12 h". Collapses equal ends. */
export function range(low: number, high: number, unit = ""): string {
  const f = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v >= 0.1 ? v.toFixed(2) : v.toPrecision(2));
  const u = unit ? ` ${unit}` : "";
  return low === high ? `${f(low)}${u}` : `${f(low)}–${f(high)}${u}`;
}

export function dtype(d: string | { ggml: number }): string {
  return typeof d === "string" ? d.toUpperCase() : `GGML(${d.ggml})`;
}

export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}
