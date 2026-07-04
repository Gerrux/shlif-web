import { test } from "node:test";
import assert from "node:assert";
import { applyBrightness, applyClahe } from "../lib/mask/enhance.ts";

test("applyBrightness scales RGB and clamps, alpha untouched", () => {
  const rgba = Uint8ClampedArray.from([100, 50, 200, 255]);
  const out = applyBrightness(rgba, 2);
  assert.deepStrictEqual([...out], [200, 100, 255, 255]); // 200*2=400 clamps to 255
});

test("applyBrightness at 1.0 is identity", () => {
  const rgba = Uint8ClampedArray.from([10, 20, 30, 255, 40, 50, 60, 255]);
  const out = applyBrightness(rgba, 1);
  assert.deepStrictEqual([...out], [...rgba]);
});

test("applyClahe preserves buffer size/alpha and stays in range on a tiny image", () => {
  const w = 4, h = 4;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const v = (i * 17) % 256;
    rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
  }
  const out = applyClahe(rgba, w, h);
  assert.strictEqual(out.length, rgba.length);
  for (let i = 0; i < w * h; i++) {
    assert.strictEqual(out[i * 4 + 3], 255);
    for (const c of [0, 1, 2]) assert.ok(out[i * 4 + c] >= 0 && out[i * 4 + c] <= 255);
  }
});
