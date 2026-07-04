import { test } from "node:test";
import assert from "node:assert";
import { parseJobParams, buildJobQuery } from "../lib/jobUrl.ts";

test("buildJobQuery is empty without a job", () => {
  assert.strictEqual(buildJobQuery(null, null), "");
});

test("buildJobQuery encodes job and startedAt", () => {
  assert.strictEqual(buildJobQuery("abc", 123), "job=abc&started=123");
});

test("buildJobQuery omits started when unknown", () => {
  assert.strictEqual(buildJobQuery("abc", null), "job=abc");
});

test("parseJobParams reads job and started", () => {
  assert.deepStrictEqual(parseJobParams(new URLSearchParams("job=abc&started=123")), { jobId: "abc", startedAt: 123 });
});

test("parseJobParams is empty without a job", () => {
  assert.deepStrictEqual(parseJobParams(new URLSearchParams("started=123")), { jobId: null, startedAt: null });
});

test("parseJobParams treats a missing/invalid started as unknown", () => {
  assert.deepStrictEqual(parseJobParams(new URLSearchParams("job=abc")), { jobId: "abc", startedAt: null });
  assert.deepStrictEqual(parseJobParams(new URLSearchParams("job=abc&started=nope")), { jobId: "abc", startedAt: null });
});
