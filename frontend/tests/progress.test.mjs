import { test } from "node:test";
import assert from "node:assert";
import { clampPct, formatDuration, computeEta } from "../lib/progress.ts";

test("clampPct rounds to a 0-100 integer and clamps out-of-range input", () => {
  assert.strictEqual(clampPct(0), 0);
  assert.strictEqual(clampPct(1), 100);
  assert.strictEqual(clampPct(0.421), 42);
  assert.strictEqual(clampPct(-0.5), 0);
  assert.strictEqual(clampPct(1.5), 100);
});

test("formatDuration formats seconds and minutes in Russian", () => {
  assert.strictEqual(formatDuration(14), "14 с");
  assert.strictEqual(formatDuration(59), "59 с");
  assert.strictEqual(formatDuration(60), "1 мин 0 с");
  assert.strictEqual(formatDuration(92), "1 мин 32 с");
});

test("computeEta is null below the 8% noise floor and extrapolates linearly above it", () => {
  assert.strictEqual(computeEta(5, 0.05), null);
  assert.strictEqual(computeEta(0, 0), null);
  assert.strictEqual(computeEta(10, 0.5), 10);
  assert.strictEqual(computeEta(20, 1), 0);
});
