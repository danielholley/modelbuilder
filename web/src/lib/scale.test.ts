import { describe, expect, it } from "vitest";
import { extent, linear, niceTicks, tickLabel } from "./scale";

describe("scale", () => {
  it("makes nice ticks that cover the domain", () => {
    expect(niceTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks(0.13, 0.87, 4)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    const t = niceTicks(1.07, 5.58, 5);
    expect(t[0]).toBeLessThanOrEqual(1.07);
    expect(t[t.length - 1]).toBeGreaterThanOrEqual(5.58);
    expect(niceTicks(3, 3).length).toBeGreaterThan(1);
  });
  it("maps linearly and labels ticks", () => {
    const s = linear([0, 10], [100, 0]);
    expect(s(5)).toBe(50);
    expect(extent([3, NaN, -1, 8])).toEqual([-1, 8]);
    expect(tickLabel(12000)).toBe("12K");
    expect(tickLabel(0.25)).toBe("0.25");
  });
});
