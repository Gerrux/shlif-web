export type Tool = "superpixel" | "brush" | "eraser" | "threshold" | "autofill";
export type Layer = "matrix" | "magnetite" | "sulfide" | "talc";
interface Snapshot { phaseMap: Uint8Array; talc: Uint8Array; }
export interface CorrectorState {
  phaseMap: Uint8Array; talc: Uint8Array; w: number; h: number;
  tool: Tool; layer: Layer; brush: number; undoStack: Snapshot[]; redoStack: Snapshot[];
}
export function layerToClass(l: Layer): 0 | 1 | 2 {
  return l === "sulfide" ? 2 : l === "magnetite" ? 1 : 0;
}
export function initState(phaseMap: Uint8Array, talc: Uint8Array, w: number, h: number): CorrectorState {
  return { phaseMap, talc, w, h, tool: "brush", layer: "sulfide", brush: 12, undoStack: [], redoStack: [] };
}
function snap(s: CorrectorState): Snapshot {
  return { phaseMap: Uint8Array.from(s.phaseMap), talc: Uint8Array.from(s.talc) };
}
export function applyPhase(s: CorrectorState, idxs: number[], layer: Layer): CorrectorState {
  const cls = layerToClass(layer);
  const phaseMap = Uint8Array.from(s.phaseMap);
  for (const i of idxs) phaseMap[i] = cls;
  return { ...s, phaseMap, undoStack: [...s.undoStack, snap(s)], redoStack: [] };
}
export function applyTalc(s: CorrectorState, idxs: number[], value: boolean): CorrectorState {
  const talc = Uint8Array.from(s.talc);
  for (const i of idxs) talc[i] = value ? 1 : 0;
  return { ...s, talc, undoStack: [...s.undoStack, snap(s)], redoStack: [] };
}
export function undo(s: CorrectorState): CorrectorState {
  if (!s.undoStack.length) return s;
  const prev = s.undoStack[s.undoStack.length - 1];
  return { ...s, phaseMap: Uint8Array.from(prev.phaseMap), talc: Uint8Array.from(prev.talc),
    undoStack: s.undoStack.slice(0, -1), redoStack: [...s.redoStack, snap(s)] };
}
export function redo(s: CorrectorState): CorrectorState {
  if (!s.redoStack.length) return s;
  const next = s.redoStack[s.redoStack.length - 1];
  return { ...s, phaseMap: Uint8Array.from(next.phaseMap), talc: Uint8Array.from(next.talc),
    redoStack: s.redoStack.slice(0, -1), undoStack: [...s.undoStack, snap(s)] };
}
