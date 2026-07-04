import { test } from "node:test";
import assert from "node:assert";
import { initState, applyPhase, applyTalc, undo, redo, layerToClass } from "../components/corrector/reducer.ts";

const st0 = () => initState(new Uint8Array(4), new Uint8Array(4), 2, 2);

test("layerToClass maps phases", () => {
  assert.strictEqual(layerToClass("matrix"), 0);
  assert.strictEqual(layerToClass("magnetite"), 1);
  assert.strictEqual(layerToClass("sulfide"), 2);
});

test("applyPhase sets class ids and is undoable", () => {
  let s = st0();
  s = applyPhase(s, [0, 1], "sulfide");
  assert.deepStrictEqual([...s.phaseMap], [2, 2, 0, 0]);
  s = undo(s);
  assert.deepStrictEqual([...s.phaseMap], [0, 0, 0, 0]);
  s = redo(s);
  assert.deepStrictEqual([...s.phaseMap], [2, 2, 0, 0]);
});

test("applyTalc toggles the talc overlay independently", () => {
  let s = st0();
  s = applyTalc(s, [3], true);
  assert.deepStrictEqual([...s.talc], [0, 0, 0, 1]);
  assert.deepStrictEqual([...s.phaseMap], [0, 0, 0, 0]);
});
