import { test } from "node:test";
import assert from "node:assert";
import { percentileThreshold, darkSegmentsMask } from "../lib/mask/darkpercent.ts";

test("percentileThreshold finds the value covering the target fraction of matrix pixels", () => {
  const dark = Uint8Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
  const phaseMap = Uint8Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // all matrix
  const t = percentileThreshold(dark, phaseMap, 0.5);
  assert.strictEqual(t, 50); // darkest 50% => values <=50 covers 5/10
});

test("percentileThreshold ignores non-matrix pixels", () => {
  const dark = Uint8Array.from([10, 200, 20, 200, 30]);
  const phaseMap = Uint8Array.from([0, 2, 0, 1, 0]); // only idx0,2,4 are matrix (10,20,30)
  const t = percentileThreshold(dark, phaseMap, 1 / 3); // darkest third of 3 matrix px => value 10
  assert.strictEqual(t, 10);
});

test("percentileThreshold returns -1 when there is no matrix", () => {
  const dark = Uint8Array.from([10, 20]);
  const phaseMap = Uint8Array.from([1, 2]);
  assert.strictEqual(percentileThreshold(dark, phaseMap, 0.5), -1);
});

test("darkSegmentsMask marks darkest matrix pixels not already talc", () => {
  const dark = Uint8Array.from([10, 20, 30, 40]);
  const phaseMap = Uint8Array.from([0, 0, 0, 0]);
  const talc = Uint8Array.from([0, 1, 0, 0]);
  const mask = darkSegmentsMask(dark, phaseMap, talc, 0.5); // threshold=20; idx1 excluded (already talc)
  assert.deepStrictEqual([...mask], [1, 0, 0, 0]);
});
