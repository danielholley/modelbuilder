import { describe, expect, it } from "vitest";
import type { EventRecord, JobSummary, JobUpdate } from "../api/types";
import { applyUpdate, curves, START } from "./jobs";

const summary: JobSummary = {
  id: 1,
  source: { kind: "events", events_path: "e.jsonl" },
  status: "running",
  job_id: null,
  started_at: 0,
  finished_at: null,
  events: 0,
  latest_progress: null,
  latest_eval: null,
  outputs: {},
  error: null,
};
const progress = (step: number, loss: number): EventRecord => ({
  schema_version: 1,
  time: step,
  stage: "s",
  event: "progress",
  step,
  steps: 10,
  tokens: null,
  loss,
  accuracy: 0.5,
  lr: null,
  tokens_per_s: null,
});
const update = (events: EventRecord[], cursor: JobUpdate["cursor"]): JobUpdate => ({ summary, events, log: [], cursor });

describe("applyUpdate", () => {
  it("appends updates that line up and ignores replays", () => {
    const a = applyUpdate(undefined, update([progress(1, 2)], { events: 1, log: 0, version: 1 }), START);
    const b = applyUpdate(a, update([progress(2, 1.5)], { events: 2, log: 0, version: 2 }), a.cursor);
    expect(b.events.map((e) => (e.event === "progress" ? e.step : -1))).toEqual([1, 2]);
    // The same update again, asked from an old cursor: not appended twice.
    const c = applyUpdate(b, update([progress(2, 1.5)], { events: 2, log: 0, version: 2 }), a.cursor);
    expect(c.events).toHaveLength(2);
  });
  it("extracts train and eval curves", () => {
    const ev: EventRecord = { schema_version: 1, time: 3, stage: "s", event: "eval", step: 2, loss: 1.2, accuracy: 0.7, tokens: null };
    const c = curves([progress(1, 2), progress(2, 1.5), ev]);
    expect(c.trainLoss).toEqual([
      { x: 1, y: 2 },
      { x: 2, y: 1.5 },
    ]);
    expect(c.evalAcc).toEqual([{ x: 2, y: 0.7 }]);
  });
});
