import { describe, expect, it } from "vitest";
import { bytes, count, duration, pct, range } from "./format";

describe("format", () => {
  it("counts with SI suffixes", () => {
    expect(count(0)).toBe("0");
    expect(count(999)).toBe("999");
    expect(count(1234)).toBe("1.23K");
    expect(count(27_300_000_000)).toBe("27.3B");
    expect(count(424_697_856)).toBe("425M");
    expect(count(null)).toBe("–");
  });
  it("formats bytes in binary units", () => {
    expect(bytes(512)).toBe("512 B");
    expect(bytes(1536)).toBe("1.50 KiB");
    expect(bytes(7.2 * 1024 ** 3)).toBe("7.20 GiB");
  });
  it("formats percentages, ranges and durations", () => {
    expect(pct(0.844)).toBe("84.4%");
    expect(range(3.2, 12, "h")).toBe("3.20–12.0 h");
    expect(range(5, 5)).toBe("5.00");
    expect(duration(3725)).toBe("1h 2m");
  });
});
