"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  initState, applyPhase, applyTalc, undo, redo, snapshot, commitStroke, layerToClass,
  type CorrectorState, type Tool, type Layer, type Snapshot,
} from "./reducer";
import { imageUrl, maskUrl, mapUrl, saveMasks } from "@/lib/api/client";
import { maskToPngBlob, rawMaskToPngBlob } from "@/lib/mask/encode";
import { loadSuperpixels, cellIndices } from "@/lib/mask/superpixel";
import type { Verdict } from "@/lib/api/types";
import { useZoomPan } from "@/lib/useZoomPan";
import {
  IconSave, IconUndo, IconRedo, IconZoomIn, IconZoomOut, IconReset, IconHand,
  IconBrush, IconEye, IconEyeOff,
} from "@/components/icons";

const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const TOOLS: [Tool, string][] = [
  ["brush", "Кисть"], ["eraser", "Ластик"], ["superpixel", "Суперпиксель"], ["threshold", "Тёмные области"], ["pan", "Рука"],
];
// Слои: цвет-образец совпадает с наложением на холсте. matrix — база без наложения (нет «глаза»).
type LayerDef = { key: Layer; ru: string; sw: string; overlay: "sulfide" | "magnetite" | "talc" | null };
const LAYERS: LayerDef[] = [
  { key: "sulfide", ru: "сульфид", sw: "rgb(201,180,95)", overlay: "sulfide" },
  { key: "magnetite", ru: "магнетит", sw: "rgb(150,160,182)", overlay: "magnetite" },
  { key: "talc", ru: "тальк", sw: "rgb(79,143,240)", overlay: "talc" },
  { key: "matrix", ru: "матрица", sw: "var(--surface-3)", overlay: null },
];
type Vis = { sulfide: boolean; magnetite: boolean; talc: boolean };

async function pngToArray(url: string, w: number, h: number): Promise<Uint8Array> {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = d[i * 4];
  return out;
}

export function Corrector({
  jobId, size, info, onVerdict,
}: {
  jobId: string; size: [number, number]; info?: ReactNode; onVerdict: (v: Verdict) => void;
}) {
  const [w, h] = size;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseRef = useRef<ImageBitmap | null>(null);
  const spRef = useRef<Uint16Array | null>(null);
  const darkRef = useRef<Uint8Array | null>(null);
  const strokeRef = useRef<{ pre: Snapshot; pm: Uint8Array; tc: Uint8Array; lx: number; ly: number } | null>(null);
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
  const [vis, setVis] = useState<Vis>({ sulfide: true, magnetite: true, talc: true });
  const [cursor, setCursor] = useState<{ x: number; y: number; d: number; on: boolean }>({ x: 0, y: 0, d: 0, on: false });
  const [grabbing, setGrabbing] = useState(false);
  const zp = useZoomPan();

  useEffect(() => {
    (async () => {
      baseRef.current = await createImageBitmap(await (await fetch(imageUrl(jobId))).blob());
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h);
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h);
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
      setState(initState(Uint8Array.from(phasesGray), Uint8Array.from(talc.map((v) => (v > 127 ? 1 : 0))), w, h));
    })();
  }, [jobId, w, h]);

  // Перерисовка из зафиксированного состояния. Во время активного мазка пропускаем —
  // холст в этот момент рисуется вручную из рабочих массивов мазка (strokeRef).
  useEffect(() => { if (state && !strokeRef.current) draw(state.phaseMap, state.talc); });

  function draw(pm: Uint8Array, tc: Uint8Array) {
    const cv = canvasRef.current;
    if (!cv || !baseRef.current) return;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(baseRef.current, 0, 0, w, h);
    const od = ctx.getImageData(0, 0, w, h); const d = od.data;
    for (let i = 0; i < w * h; i++) {
      const cls = pm[i];
      const showPhase = (cls === 2 && vis.sulfide) || (cls === 1 && vis.magnetite);
      if (showPhase) {
        const rgb = PHASE_RGB[cls];
        d[i * 4] = 0.45 * d[i * 4] + 0.55 * rgb[0];
        d[i * 4 + 1] = 0.45 * d[i * 4 + 1] + 0.55 * rgb[1];
        d[i * 4 + 2] = 0.45 * d[i * 4 + 2] + 0.55 * rgb[2];
      }
      if (tc[i] && vis.talc) {
        d[i * 4] = 0.4 * d[i * 4] + 0.6 * TALC_RGB[0];
        d[i * 4 + 1] = 0.4 * d[i * 4 + 1] + 0.6 * TALC_RGB[1];
        d[i * 4 + 2] = 0.4 * d[i * 4 + 2] + 0.6 * TALC_RGB[2];
      }
    }
    ctx.putImageData(od, 0, 0);
  }

  function toCanvas(e: React.PointerEvent) {
    const cv = canvasRef.current!; const r = cv.getBoundingClientRect();
    return {
      cx: Math.floor(((e.clientX - r.left) / r.width) * w),
      cy: Math.floor(((e.clientY - r.top) / r.height) * h),
    };
  }
  function stamp(cx: number, cy: number, r: number, into: number[]) {
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const x = cx + dx, y = cy + dy;
      if (x >= 0 && x < w && y >= 0 && y < h && dx * dx + dy * dy <= r * r) into.push(y * w + x);
    }
  }
  function strokeLine(x0: number, y0: number, x1: number, y1: number, r: number, into: number[]) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const n = Math.max(1, Math.ceil(dist / Math.max(1, r / 2)));
    for (let k = 0; k <= n; k++) stamp(Math.round(x0 + ((x1 - x0) * k) / n), Math.round(y0 + ((y1 - y0) * k) / n), r, into);
  }
  function applyStamp(pm: Uint8Array, tc: Uint8Array, idxs: number[], s: CorrectorState) {
    if (s.layer === "talc") { const v = s.tool === "eraser" ? 0 : 1; for (const i of idxs) tc[i] = v; }
    else { const cls = s.tool === "eraser" ? 0 : layerToClass(s.layer); for (const i of idxs) pm[i] = cls; }
  }

  function onPointerDown(e: React.PointerEvent) {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (!state) return;
    if (e.button === 1 || state.tool === "pan") { setGrabbing(true); zp.startPan(e); return; }
    const { cx, cy } = toCanvas(e);
    if (state.tool === "brush" || state.tool === "eraser") {
      const pre = snapshot(state);
      const pm = Uint8Array.from(state.phaseMap), tc = Uint8Array.from(state.talc);
      const idxs: number[] = []; stamp(cx, cy, state.brush, idxs); applyStamp(pm, tc, idxs, state);
      strokeRef.current = { pre, pm, tc, lx: cx, ly: cy };
      draw(pm, tc);
    } else if (state.tool === "superpixel" && spRef.current) {
      const idxs = cellIndices(spRef.current, cy * w + cx);
      setState(state.layer === "talc" ? applyTalc(state, idxs, true) : applyPhase(state, idxs, state.layer));
    } else if (state.tool === "threshold" && darkRef.current) {
      const idxs: number[] = [];
      for (let i = 0; i < w * h; i++) if (darkRef.current[i] <= thr && state.phaseMap[i] === 0) idxs.push(i);
      setState(applyTalc(state, idxs, true));
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    updateCursor(e);
    if (strokeRef.current && state && (state.tool === "brush" || state.tool === "eraser")) {
      const { cx, cy } = toCanvas(e);
      const st = strokeRef.current;
      const idxs: number[] = []; strokeLine(st.lx, st.ly, cx, cy, state.brush, idxs);
      applyStamp(st.pm, st.tc, idxs, state);
      st.lx = cx; st.ly = cy;
      draw(st.pm, st.tc);
    } else if (zp.isPanning()) {
      zp.movePan(e);
    }
  }

  function endStroke() {
    if (strokeRef.current) {
      const { pre, pm, tc } = strokeRef.current;
      strokeRef.current = null;
      setState((s) => (s ? commitStroke(s, pre, pm, tc) : s));
    }
    zp.endPan(); setGrabbing(false);
  }

  function updateCursor(e: React.PointerEvent) {
    const vp = zp.vpRef.current, cv = canvasRef.current;
    if (!vp || !cv || !state || (state.tool !== "brush" && state.tool !== "eraser")) {
      setCursor((c) => (c.on ? { ...c, on: false } : c));
      return;
    }
    const vr = vp.getBoundingClientRect(), cr = cv.getBoundingClientRect();
    setCursor({ x: e.clientX - vr.left, y: e.clientY - vr.top, d: state.brush * 2 * (cr.width / w), on: true });
  }

  async function save() {
    if (!state) return;
    setSaving(true);
    try {
      const phaseBlob = await rawMaskToPngBlob(state.phaseMap, w, h);
      const talcBlob = await maskToPngBlob(state.talc, w, h);
      const v = await saveMasks(jobId, phaseBlob, talcBlob);
      onVerdict(v);
    } finally { setSaving(false); }
  }

  const vpClass = state?.tool === "pan" ? (grabbing ? "grabbing" : "grab")
    : (state?.tool === "brush" || state?.tool === "eraser") ? "paint" : "";

  return (
    <div className="workspace">
      <aside className="ws-side">
        {info}
        <div className="card">
          <div className="side-h">Редактор масок<span className="ann">{state ? `${state.brush}px` : "…"}</span></div>
          <div className="side-b">
            {!state ? (
              <div className="stage-empty" style={{ minHeight: 120 }}><div className="hint">Загрузка масок…</div></div>
            ) : (
              <>
                <div className="tool-group">
                  <span className="toolbar-label">Инструмент</span>
                  <div className="seg" role="group" aria-label="Инструмент">
                    {TOOLS.map(([t, ru]) => (
                      <button key={t} type="button" className={state.tool === t ? "active" : ""}
                        aria-pressed={state.tool === t} onClick={() => setState({ ...state, tool: t })}>
                        {t === "brush" ? <IconBrush className="ico-sm" /> : t === "pan" ? <IconHand className="ico-sm" /> : null}{ru}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="tool-group">
                  <span className="toolbar-label">Кисть</span>
                  <label className="ctl">размер
                    <input className="slider" type="range" min={2} max={60} value={state.brush}
                      onChange={(e) => setState({ ...state, brush: +e.target.value })} />
                    <span className="slider-val">{state.brush}px</span>
                  </label>
                  {state.tool === "threshold" && (
                    <label className="ctl">порог
                      <input className="slider" type="range" min={5} max={200} value={thr}
                        onChange={(e) => setThr(+e.target.value)} />
                      <span className="slider-val">{thr}</span>
                    </label>
                  )}
                </div>
                <div className="tool-group">
                  <span className="toolbar-label">Слои</span>
                  <div className="layers">
                    {LAYERS.map((l) => {
                      const off = l.overlay ? !vis[l.overlay] : false;
                      return (
                        <div key={l.key} className={`layer-row${state.layer === l.key ? " active" : ""}${off ? " off" : ""}`}
                          role="button" tabIndex={0} onClick={() => setState({ ...state, layer: l.key })}>
                          <span className="sw" style={{ background: l.sw }} />
                          <span className="nm">{l.ru}</span>
                          {l.overlay ? (
                            <button type="button" className="eye" aria-label={off ? "Показать слой" : "Скрыть слой"}
                              onClick={(e) => { e.stopPropagation(); setVis((v) => ({ ...v, [l.overlay!]: !v[l.overlay!] })); }}>
                              {off ? <IconEyeOff className="ico-sm" /> : <IconEye className="ico-sm" />}
                            </button>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="tool-group" style={{ display: "flex", gap: 8, gridTemplateColumns: "unset" }}>
                  <button type="button" className="btn ghost sm icon" title="Отменить" aria-label="Отменить"
                    onClick={() => setState(undo(state))}><IconUndo /></button>
                  <button type="button" className="btn ghost sm icon" title="Повторить" aria-label="Повторить"
                    onClick={() => setState(redo(state))}><IconRedo /></button>
                  <button type="button" className="btn primary" style={{ flex: 1 }} onClick={save} disabled={saving}>
                    <IconSave /> {saving ? "Сохранение…" : "Сохранить"}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </aside>

      <div className="ws-view">
        <div ref={zp.vpRef} className={`zoom-vp ${vpClass}`}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endStroke}
          onPointerLeave={() => { setCursor((c) => ({ ...c, on: false })); }}>
          <div className="zoom-content" style={{ transform: zp.transform }}>
            {state ? <canvas ref={canvasRef} width={w} height={h} /> : null}
          </div>
          {!state ? <div className="stage-empty"><div className="hint">Загрузка редактора…</div></div> : null}
          {cursor.on ? <div className="brush-cursor" style={{ left: cursor.x, top: cursor.y, width: cursor.d, height: cursor.d }} /> : null}
          <div className="zoom-hint">колесо — зум · «Рука»/средняя кнопка — сдвиг</div>
          <div className="zoom-level">{Math.round(zp.view.zoom * 100)}%</div>
          <div className="zoom-controls">
            <button type="button" className="btn dark sm icon" title="Отдалить" onClick={zp.zoomOut}><IconZoomOut /></button>
            <button type="button" className="btn dark sm icon" title="Сбросить" onClick={zp.reset}><IconReset /></button>
            <button type="button" className="btn dark sm icon" title="Приблизить" onClick={zp.zoomIn}><IconZoomIn /></button>
          </div>
        </div>
      </div>
    </div>
  );
}
