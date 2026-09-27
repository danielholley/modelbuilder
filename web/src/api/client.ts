// Thin typed wrappers over the mb-server JSON API. Types come from the Rust
// side (./types.ts is generated; see crates/mb-server/src/types.rs).
import type {
  Catalog,
  Cursor,
  DirListing,
  ErrorBody,
  Health,
  InspectRequest,
  InspectResponse,
  JobSource,
  JobSummary,
  JobUpdate,
  ListRequest,
  Plan,
  PlanRequest,
  StatsRequest,
  WeightStatsReport,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? {} : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  const text = await res.text();
  if (!res.ok) {
    let message = text || res.statusText;
    try {
      message = (JSON.parse(text) as ErrorBody).error;
    } catch {
      // not JSON (e.g. axum's own rejection text): keep the raw body
    }
    throw new ApiError(res.status, message);
  }
  return JSON.parse(text) as T;
}

export const api = {
  health: () => request<Health>("GET", "/health"),
  catalog: () => request<Catalog>("GET", "/catalog"),
  inspect: (r: InspectRequest, signal?: AbortSignal) => request<InspectResponse>("POST", "/inspect", r, signal),
  stats: (r: StatsRequest, signal?: AbortSignal) => request<WeightStatsReport>("POST", "/stats", r, signal),
  plan: (r: PlanRequest, signal?: AbortSignal) => request<Plan>("POST", "/plan", r, signal),
  list: (r: ListRequest) => request<DirListing>("POST", "/fs/list", r),
  jobs: () => request<JobSummary[]>("GET", "/jobs"),
  startJob: (s: JobSource) => request<JobSummary>("POST", "/jobs", s),
  cancelJob: (id: number) => request<JobSummary>("POST", `/jobs/${id}/cancel`),
};

/** Subscribes to a job's server-sent updates. Returns a function that closes the stream. */
export function streamJob(id: number, since: Cursor, onUpdate: (u: JobUpdate) => void): () => void {
  const q = new URLSearchParams({ events: String(since.events), log: String(since.log), version: String(since.version) });
  const es = new EventSource(`/api/jobs/${id}/stream?${q}`);
  es.addEventListener("update", (e) => onUpdate(JSON.parse((e as MessageEvent<string>).data) as JobUpdate));
  // The server ends the stream when the job ends; don't let EventSource reconnect forever.
  es.onerror = () => es.close();
  return () => es.close();
}
