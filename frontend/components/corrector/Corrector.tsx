"use client";
import { useEffect, useRef, useState } from "react";
import { initState, applyPhase, applyTalc, undo, redo, type CorrectorState, type Tool, type Layer } from "./reducer";
import { imageUrl, maskUrl, mapUrl, saveMasks } from "@/lib/api/client";
import { maskToPngBlob, rawMaskToPngBlob } from "@/lib/mask/encode";
import { loadSuperpixels, cellIndices } from "@/lib/mask/superpixel";
import type { Verdict } from "@/lib/api/types";

const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const TOOLS: [Tool, string][] = [["superpixel", "Суперпиксель"], ["brush", "Кисть"], ["eraser", "Ластик"], ["threshold", "Тёмные области"], ["autofill", "Авто-заполнение"]];
const LAYERS: [Layer, string][] = [["sulfide", "сульфид"], ["magnetite", "магнетит"], ["matrix", "матрица"], ["talc", "тальк"]];

// Decode a grayscale PNG (phases 0/1/2 or darkness 0-255) into a flat byte array.
// R === G === B for these grayscale-encoded maps, so reading R is exact.
async function pngToArray(url: string, w: number, h: number): Promise<Uint8Array> {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = d[i * 4]; // grayscale in R
  return out;
}

export function Corrector({ jobId, size, onVerdict }: { jobId: string; size: [number, number]; onVerdict: (v: Verdict) => void }) {
  const [w, h] = size;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseRef = useRef<ImageBitmap | null>(null);
  const spRef = useRef<Uint16Array | null>(null);
  const darkRef = useRef<Uint8Array | null>(null);
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      baseRef.current = await createImageBitmap(await (await fetch(imageUrl(jobId))).blob());
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h); // raw 0/1/2 label map
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h); // 0/255
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
      setState(initState(Uint8Array.from(phasesGray), Uint8Array.from(talc.map((v) => (v > 127 ? 1 : 0))), w, h));
    })();
  }, [jobId, w, h]);

  useEffect(() => { if (state) draw(); });

  function draw() {
    const cv = canvasRef.current, s = state;
    if (!cv || !s || !baseRef.current) return;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(baseRef.current, 0, 0, w, h);
    const overlay = ctx.getImageData(0, 0, w, h);
    for (let i = 0; i < w * h; i++) {
      const cls = s.phaseMap[i];
      const rgb = cls ? PHASE_RGB[cls] : null;
      if (rgb) {
        overlay.data[i * 4] = 0.45 * overlay.data[i * 4] + 0.55 * rgb[0];
        overlay.data[i * 4 + 1] = 0.45 * overlay.data[i * 4 + 1] + 0.55 * rgb[1];
        overlay.data[i * 4 + 2] = 0.45 * overlay.data[i * 4 + 2] + 0.55 * rgb[2];
      }
      if (s.talc[i]) {
        overlay.data[i * 4] = 0.4 * overlay.data[i * 4] + 0.6 * TALC_RGB[0];
        overlay.data[i * 4 + 1] = 0.4 * overlay.data[i * 4 + 1] + 0.6 * TALC_RGB[1];
        overlay.data[i * 4 + 2] = 0.4 * overlay.data[i * 4 + 2] + 0.6 * TALC_RGB[2];
      }
    }
    ctx.putImageData(overlay, 0, 0);
  }

  function paintAt(cx: number, cy: number) {
    if (!state) return;
    const idx = cy * w + cx;
    const isTalc = state.layer === "talc";
    if (state.tool === "superpixel" && spRef.current) {
      const idxs = cellIndices(spRef.current, idx);
      setState(isTalc ? applyTalc(state, idxs, true) : applyPhase(state, idxs, state.layer));
    } else if (state.tool === "brush" || state.tool === "eraser") {
      const r = state.brush, idxs: number[] = [];
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x >= 0 && x < w && y >= 0 && y < h && dx * dx + dy * dy <= r * r) idxs.push(y * w + x);
      }
      const erase = state.tool === "eraser";
      setState(isTalc ? applyTalc(state, idxs, !erase) : applyPhase(state, idxs, erase ? "matrix" : state.layer));
    } else if (state.tool === "threshold" && darkRef.current) {
      const idxs: number[] = [];
      for (let i = 0; i < w * h; i++) if (darkRef.current[i] <= thr && state.phaseMap[i] === 0) idxs.push(i);
      setState(applyTalc(state, idxs, true)); // dark-area → talc within matrix
    } else if (state.tool === "autofill") {
      // seed: re-load pipeline talc as-is (already in state); no-op placeholder for future detectors
    }
  }

  // Single click paints once. (Deliberately NOT also bound to onMouseDown: binding
  // both fires paintAt twice per click — once on mousedown, once on the following
  // click — which double-pushes the undo stack for what looks like one edit.)
  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * w);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * h);
    paintAt(cx, cy);
  }

  async function save() {
    if (!state) return;
    setSaving(true);
    try {
      // Phase map is a raw 0/1/2 label map — must NOT go through the 0/255 threshold
      // encoder (that would collapse magnetite(1)/sulfide(2) into 255).
      const phaseBlob = await rawMaskToPngBlob(state.phaseMap, w, h);
      const talcBlob = await maskToPngBlob(state.talc, w, h);
      const v = await saveMasks(jobId, phaseBlob, talcBlob);
      onVerdict(v);
    } finally { setSaving(false); }
  }

  if (!state) return <div className="stage" style={{ padding: 40 }}>Загрузка редактора…</div>;
  return (
    <div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {TOOLS.map(([t, ru]) => <button key={t} onClick={() => setState({ ...state, tool: t })}
          style={{ fontWeight: state.tool === t ? 700 : 400 }}>{ru}</button>)}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {LAYERS.map(([l, ru]) => <button key={l} onClick={() => setState({ ...state, layer: l })}
          style={{ fontWeight: state.layer === l ? 700 : 400 }}>{ru}</button>)}
        <label>кисть <input type="range" min={2} max={40} value={state.brush} onChange={(e) => setState({ ...state, brush: +e.target.value })} /></label>
        {state.tool === "threshold" && <label>порог <input type="range" min={5} max={200} value={thr} onChange={(e) => setThr(+e.target.value)} /></label>}
        <button onClick={() => setState(undo(state))}>↶</button>
        <button onClick={() => setState(redo(state))}>↷</button>
      </div>
      <div className="stage"><canvas ref={canvasRef} width={w} height={h} onClick={onClick} style={{ width: "100%", cursor: "crosshair" }} /></div>
      <button onClick={save} disabled={saving} style={{ marginTop: 8 }}>{saving ? "Сохранение…" : "💾 Сохранить и пересчитать вердикт"}</button>
    </div>
  );
}
