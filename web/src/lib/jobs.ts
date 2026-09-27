import type { Cursor, EventRecord, JobSummary, JobUpdate } from "../api/types";

/** Everything the dashboard keeps for one job, rebuilt from incremental updates. */
export interface JobView {
  summary: JobSummary;
  events: EventRecord[];
  log: string[];
  cursor: Cursor;
}

export const START: Cursor = { events: 0, log: 0, version: 0 };

/** Folds an update in. Updates carry records from the cursor they were asked with,
 *  so an update that doesn't line up with what we hold (a replay) is ignored. */
export function applyUpdate(view: JobView | undefined, u: JobUpdate, since: Cursor): JobView {
  const base = view ?? { summary: u.summary, events: [], log: [], cursor: START };
  if (since.events !== base.events.length || since.log !== base.log.length) {
    return u.cursor.version > base.cursor.version ? { ...base, summary: u.summary } : base;
  }
  return {
    summary: u.summary,
    events: base.events.concat(u.events),
    log: base.log.concat(u.log),
    cursor: u.cursor,
  };
}

export interface Point {
  x: number;
  y: number;
}

/** Train loss/accuracy from progress events and eval loss/accuracy, keyed by step. */
export function curves(events: EventRecord[]) {
  const trainLoss: Point[] = [];
  const trainAcc: Point[] = [];
  const evalLoss: Point[] = [];
  const evalAcc: Point[] = [];
  for (const e of events) {
    if (e.event === "progress") {
      trainLoss.push({ x: e.step, y: e.loss });
      if (e.accuracy !== null) trainAcc.push({ x: e.step, y: e.accuracy });
    } else if (e.event === "eval") {
      evalLoss.push({ x: e.step, y: e.loss });
      evalAcc.push({ x: e.step, y: e.accuracy });
    }
  }
  return { trainLoss, trainAcc, evalLoss, evalAcc };
}
