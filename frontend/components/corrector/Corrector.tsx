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
import { applyBrightness, applyClahe } from "@/lib/mask/enhance";

const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const TOOLS: [Tool, string][] = [
  ["brush", "Кисть"], ["eraser", "Ластик"], ["superpixel", "Суперпиксель"], ["threshold", "Тёмные области"], ["pan", "Рука"],
];
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
  const cursorRef = useRef<HTMLDivElement>(null);
  const spRef = useRef<Uint16Array | null>(null);
  const darkRef = useRef<Uint8Array | null>(null);
  // Пиксельный конвейер: базовый снимок RGBA кэшируется один раз; композит наложения
  // считается инкрементально (только изменённые пиксели) и блитится раз в кадр (rAF).
  const baseRGBA = useRef<Uint8ClampedArray | null>(null);
  const enhancedRGBA = useRef<Uint8ClampedArray | null>(null);
  const outRef = useRef<ImageData | null>(null);
  const srcRef = useRef<{ pm: Uint8Array; tc: Uint8Array } | null>(null);
  const strokeRef = useRef<{ pre: Snapshot; pm: Uint8Array; tc: Uint8Array; lx: number; ly: number } | null>(null);
  const rafRef = useRef(0);
  const pendingFull = useRef(false);
  const pendingIdx = useRef<number[]>([]);
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
  const [vis, setVis] = useState<Vis>({ sulfide: true, magnetite: true, talc: true });
  const visRef = useRef(vis); visRef.current = vis;
  const [maskAlpha, setMaskAlpha] = useState(0.5);
  const maskAlphaRef = useRef(maskAlpha); maskAlphaRef.current = maskAlpha;
  const [brightness, setBrightness] = useState(1.0);
  const [clahe, setClahe] = useState(false);
  const [grabbing, setGrabbing] = useState(false);
  const [sideTab, setSideTab] = useState<"edit" | "report">("edit");
  const zp = useZoomPan();

  useEffect(() => {
    (async () => {
      const bmp = await createImageBitmap(await (await fetch(imageUrl(jobId))).blob());
      const off = document.createElement("canvas"); off.width = w; off.height = h;
      const octx = off.getContext("2d")!; octx.drawImage(bmp, 0, 0, w, h);
      baseRGBA.current = octx.getImageData(0, 0, w, h).data;
      outRef.current = new ImageData(new Uint8ClampedArray(baseRGBA.current), w, h);
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h);
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h);
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
      const st = initState(Uint8Array.from(phasesGray), Uint8Array.from(talc.map((v) => (v > 127 ? 1 : 0))), w, h);
      srcRef.current = { pm: st.phaseMap, tc: st.talc };
      setState(st);
    })();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [jobId, w, h]);

  // Полная перерисовка только когда изменились сами маски или видимость слоёв —
  // не на смену инструмента/размера кисти (те не трогают холст).
  useEffect(() => {
    if (state && baseRGBA.current && !strokeRef.current) {
      srcRef.current = { pm: state.phaseMap, tc: state.talc };
      requestDraw();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phaseMap, state?.talc, vis, maskAlpha]);

  // Яркость/CLAHE — приводим базовый снимок один раз в закэшированный буфер;
  // composePixel всегда читает его вместо необработанного baseRGBA.
  useEffect(() => {
    if (!baseRGBA.current) return;
    let buf = applyBrightness(baseRGBA.current, brightness);
    if (clahe) buf = applyClahe(buf, w, h);
    enhancedRGBA.current = buf;
    if (state && !strokeRef.current) { srcRef.current = { pm: state.phaseMap, tc: state.talc }; requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brightness, clahe]);

  function composePixel(pm: Uint8Array, tc: Uint8Array, i: number) {
    const b = (enhancedRGBA.current ?? baseRGBA.current)!, o = outRef.current!.data; const j = i * 4;
    let r = b[j], g = b[j + 1], bl = b[j + 2];
    const cls = pm[i], v = visRef.current, a = maskAlphaRef.current;
    if ((cls === 2 && v.sulfide) || (cls === 1 && v.magnetite)) {
      const c = PHASE_RGB[cls];
      r = (1 - a) * r + a * c[0]; g = (1 - a) * g + a * c[1]; bl = (1 - a) * bl + a * c[2];
    }
    if (tc[i] && v.talc) {
      r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
  function flush() {
    rafRef.current = 0;
    const src = srcRef.current, out = outRef.current, cv = canvasRef.current;
    if (src && out && cv) {
      if (pendingFull.current) for (let i = 0; i < w * h; i++) composePixel(src.pm, src.tc, i);
      else { const ids = pendingIdx.current; for (let k = 0; k < ids.length; k++) composePixel(src.pm, src.tc, ids[k]); }
      cv.getContext("2d")!.putImageData(out, 0, 0);
    }
    pendingFull.current = false; pendingIdx.current = [];
  }
  function requestDraw(idxs?: number[]) {
    if (idxs) { if (!pendingFull.current) for (let k = 0; k < idxs.length; k++) pendingIdx.current.push(idxs[k]); }
    else pendingFull.current = true;
    if (!rafRef.current) rafRef.current = requestAnimationFrame(flush);
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

  function moveCursor(e: React.PointerEvent) {
    const cd = cursorRef.current, vp = zp.vpRef.current, cv = canvasRef.current;
    if (!cd) return;
    if (!vp || !cv || !state || sideTab === "report" || (state.tool !== "brush" && state.tool !== "eraser")) { cd.style.display = "none"; return; }
    const vr = vp.getBoundingClientRect(), cr = cv.getBoundingClientRect();
    const d = state.brush * 2 * (cr.width / w);
    cd.style.display = "block";
    cd.style.left = `${e.clientX - vr.left}px`; cd.style.top = `${e.clientY - vr.top}px`;
    cd.style.width = `${d}px`; cd.style.height = `${d}px`;
  }

  function onPointerDown(e: React.PointerEvent) {
    if ((e.target as Element).closest(".zoom-controls")) return; // клики по кнопкам зума не рисуют
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (!state) return;
    if (sideTab === "report" || e.button === 1 || state.tool === "pan") { setGrabbing(true); zp.startPan(e); return; }
    const { cx, cy } = toCanvas(e);
    if (state.tool === "brush" || state.tool === "eraser") {
      const pre = snapshot(state);
      const pm = Uint8Array.from(state.phaseMap), tc = Uint8Array.from(state.talc);
      strokeRef.current = { pre, pm, tc, lx: cx, ly: cy };
      srcRef.current = { pm, tc };
      const idxs: number[] = []; stamp(cx, cy, state.brush, idxs); applyStamp(pm, tc, idxs, state);
      requestDraw(idxs);
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
    moveCursor(e);
    const st = strokeRef.current;
    if (st && state && (state.tool === "brush" || state.tool === "eraser")) {
      const { cx, cy } = toCanvas(e);
      const idxs: number[] = []; strokeLine(st.lx, st.ly, cx, cy, state.brush, idxs);
      applyStamp(st.pm, st.tc, idxs, state);
      st.lx = cx; st.ly = cy;
      requestDraw(idxs);
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

  const vpClass = sideTab === "report" || state?.tool === "pan" ? (grabbing ? "grabbing" : "grab")
    : (state?.tool === "brush" || state?.tool === "eraser") ? "paint" : "";

  return (
    <div className="workspace">
      <aside className="ws-side">
        <div className="seg" role="group" aria-label="Раздел сайдбара">
          <button type="button" className={sideTab === "edit" ? "active" : ""} aria-pressed={sideTab === "edit"}
            onClick={() => setSideTab("edit")}>Редактирование</button>
          <button type="button" className={sideTab === "report" ? "active" : ""} aria-pressed={sideTab === "report"}
            onClick={() => setSideTab("report")}>Отчёт</button>
        </div>
        {sideTab === "report" ? info : (
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
                              onClick={(e) => { e.stopPropagation(); setVis((vv) => ({ ...vv, [l.overlay!]: !vv[l.overlay!] })); }}>
                              {off ? <IconEyeOff className="ico-sm" /> : <IconEye className="ico-sm" />}
                            </button>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="tool-group">
                  <span className="toolbar-label">Вид (не меняет маску)</span>
                  <label className="ctl">прозрачность
                    <input className="slider" type="range" min={0.10} max={0.90} step={0.05} value={maskAlpha}
                      onChange={(e) => setMaskAlpha(+e.target.value)} />
                    <span className="slider-val">{maskAlpha.toFixed(2)}</span>
                  </label>
                  <label className="ctl">яркость
                    <input className="slider" type="range" min={0.40} max={2.60} step={0.1} value={brightness}
                      onChange={(e) => setBrightness(+e.target.value)} />
                    <span className="slider-val">{brightness.toFixed(1)}</span>
                  </label>
                  <label className="switch-row">
                    <span>CLAHE (контраст)</span>
                    <button type="button" className={`switch${clahe ? " on" : ""}`} role="switch" aria-checked={clahe}
                      onClick={() => setClahe((v) => !v)}><span className="knob" /></button>
                  </label>
                </div>
                <div className="tool-group" style={{ display: "flex", gap: 8 }}>
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
        )}
      </aside>

      <div className="ws-view">
        <div ref={zp.vpRef} className={`zoom-vp ${vpClass}`}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={endStroke}
          onPointerLeave={() => { if (cursorRef.current) cursorRef.current.style.display = "none"; }}>
          <div className="zoom-content" style={{ transform: zp.transform }}>
            {state ? <canvas ref={canvasRef} width={w} height={h} /> : null}
          </div>
          {!state ? <div className="stage-empty"><div className="hint">Загрузка редактора…</div></div> : null}
          <div ref={cursorRef} className="brush-cursor" style={{ display: "none" }} />
          <div className="zoom-hint">колесо — зум · «Рука»/средняя кнопка — сдвиг</div>
          <div className="zoom-level">{Math.round(zp.view.zoom * 100)}%</div>
          <div className="zoom-controls" onPointerDown={(e) => e.stopPropagation()}>
            <button type="button" className="btn dark sm icon" title="Отдалить" onClick={zp.zoomOut}><IconZoomOut /></button>
            <button type="button" className="btn dark sm icon" title="Сбросить" onClick={zp.reset}><IconReset /></button>
            <button type="button" className="btn dark sm icon" title="Приблизить" onClick={zp.zoomIn}><IconZoomIn /></button>
          </div>
        </div>
      </div>
    </div>
  );
}
