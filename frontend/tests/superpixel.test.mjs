import { test } from "node:test";
import assert from "node:assert";
import { cellIndices, computeBoundaries } from "../lib/mask/superpixel.ts";

test("computeBoundaries marks pixels differing from left/top neighbor", () => {
  // 3x2 label grid:
  // 0 0 1
  // 0 0 1
  const labels = Uint16Array.from([0, 0, 1, 0, 0, 1]);
  const b = computeBoundaries(labels, 3, 2);
  assert.deepStrictEqual([...b], [0, 0, 1, 0, 0, 1]);
});

test("computeBoundaries is all-zero for a single uniform segment", () => {
  const labels = new Uint16Array(9).fill(5);
  const b = computeBoundaries(labels, 3, 3);
  assert.deepStrictEqual([...b], new Array(9).fill(0));
});

test("cellIndices still finds every pixel sharing the seed's segment id", () => {
  const labels = Uint16Array.from([0, 0, 1, 0, 0, 1]);
  assert.deepStrictEqual(cellIndices(labels, 0), [0, 1, 3, 4]);
  assert.deepStrictEqual(cellIndices(labels, 2), [2, 5]);
});
