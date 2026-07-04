import { test } from "node:test";
import assert from "node:assert";
import { labelMapToBytes } from "../lib/mask/encode.ts";

test("labelMapToBytes preserves class ids", () => {
  const out = labelMapToBytes(Uint8Array.from([0, 1, 2, 0]));
  assert.deepStrictEqual([...out], [0, 1, 2, 0]);
});
