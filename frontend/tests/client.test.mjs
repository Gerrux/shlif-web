import { test } from "node:test";
import assert from "node:assert";
import { maskUrl, mapUrl, imageUrl, reportUrl } from "../lib/api/client.ts";

test("url builders", () => {
  assert.strictEqual(maskUrl("abc", "phases"), "/api/masks/abc/phases.png");
  assert.strictEqual(mapUrl("abc", "darkness"), "/api/maps/abc/darkness.png");
  assert.strictEqual(mapUrl("abc", "confidence"), "/api/maps/abc/confidence.png");
  assert.strictEqual(imageUrl("abc"), "/api/images/abc.jpg");
  assert.strictEqual(reportUrl("abc"), "/api/report/abc.pdf");
});
