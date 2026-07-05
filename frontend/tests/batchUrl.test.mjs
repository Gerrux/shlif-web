import { test } from "node:test";
import assert from "node:assert";
import { parseBatchParams, buildBatchQuery } from "../lib/batchUrl.ts";

test("parseBatchParams reads batch", () => {
  assert.deepStrictEqual(parseBatchParams(new URLSearchParams("batch=b1")), { batchId: "b1" });
});

test("parseBatchParams is empty without a batch", () => {
  assert.deepStrictEqual(parseBatchParams(new URLSearchParams("job=abc")), { batchId: null });
});

test("buildBatchQuery is empty without a batchId", () => {
  assert.strictEqual(buildBatchQuery(null, null), "");
});

test("buildBatchQuery encodes batch alone", () => {
  assert.strictEqual(buildBatchQuery("b1", null), "batch=b1");
});

test("buildBatchQuery encodes batch and job together", () => {
  assert.strictEqual(buildBatchQuery("b1", "j1"), "batch=b1&job=j1");
});
