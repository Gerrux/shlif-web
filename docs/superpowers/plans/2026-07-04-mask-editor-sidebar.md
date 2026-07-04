# Редактор масок — таб-сайдбар + вид/слои-подсказки — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the mask editor to the top of the closeup sidebar via a Редактирование/Отчёт tab split, and port the "Вид" (opacity/brightness/CLAHE) and "Слои-подсказки" (dark-segments preview→apply) controls from `hakaton_nornikel`'s Streamlit annotator into the React corrector, with tool-specific settings shown per mode.

**Architecture:** All changes are frontend-only, in `frontend/`. Two new pure TS modules (`lib/mask/enhance.ts`, `lib/mask/darkpercent.ts`) hold the new math (brightness/CLAHE, percentile-threshold dark-segment masking) and are unit-tested the same way `lib/mask/encode.ts` already is (`node:test` importing `.ts` directly, no DOM). `components/corrector/Corrector.tsx` is extended incrementally: tab split first (pure layout), then the view controls wired into the existing rAF pixel-compose pipeline, then the hint-layer preview/apply, then the per-mode tool settings. `components/corrector/reducer.ts` needs exactly one change (drop the now-unused `"threshold"` member from the `Tool` union) — no other reducer changes; every new mask edit reuses the existing exported `applyPhase`/`applyTalc`.

**Tech Stack:** Next.js 15 / React 19 / TypeScript, `node:test` + `tsx` for unit tests, plain Canvas 2D (no new npm dependencies).

## Global Constraints

- Defaults must not change the current visual output: brightness starts at `1.0` (neutral), CLAHE starts off, and the new single `maskAlpha` (`0.5`) replaces the previously-hardcoded per-layer blend weights (`0.45/0.55` phases, `0.4/0.6` talc) as the closest single equivalent.
- Default active sidebar tab is **«Редактирование»** — this is what satisfies "move the editor to the top".
- No new npm dependencies. CLAHE is implemented in plain TypeScript.
- `frontend/tests/reducer.test.mjs` must keep passing unmodified — no reducer behavior change beyond the `Tool` union edit.
- Exact Russian UI strings (copied from the approved spec): «Редактирование», «Отчёт», «Вид (не меняет маску)», «Слои-подсказки (превью → применить в маску)», «Тёмные сегменты, % матрицы», «Показать превью», «+ Применить к тальку», «Ставить», «Убирать», «Клик по суперпикселю».
- No "Размер на экране, px" control — explicitly out of scope (zoom/pan already covers it).
- Spec: `docs/superpowers/specs/2026-07-04-mask-editor-sidebar-design.md`.

---

## Task 1: `lib/mask/enhance.ts` — brightness + CLAHE-on-luma

**Files:**
- Create: `frontend/lib/mask/enhance.ts`
- Test: `frontend/tests/enhance.test.mjs`

**Interfaces:**
- Produces: `applyBrightness(rgba: Uint8ClampedArray, brightness: number): Uint8ClampedArray`
- Produces: `applyClahe(rgba: Uint8ClampedArray, w: number, h: number, tilesX?: number, tilesY?: number, clipLimit?: number): Uint8ClampedArray`
- Consumed by: Task 4 (`Corrector.tsx`).

- [ ] **Step 1: Write the failing test**

Create `frontend/tests/enhance.test.mjs`:

```js
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --test-name-pattern=applyBrightness`
Expected: FAIL — `Cannot find module '../lib/mask/enhance.ts'`

- [ ] **Step 3: Write minimal implementation**

Create `frontend/lib/mask/enhance.ts`:

```ts
// Вид (не меняет маску): яркость + CLAHE-по-luma для канваса корректора.
// Реконструируем RGB через коэффициент Y'/Y — без перехода в Lab, но с тем же
// эффектом локального контраста, что и cv2 CLAHE на L-канале в hakaton_nornikel.
export function applyBrightness(rgba: Uint8ClampedArray, brightness: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba.length);
  for (let i = 0; i < rgba.length; i += 4) {
    out[i] = rgba[i] * brightness;
    out[i + 1] = rgba[i + 1] * brightness;
    out[i + 2] = rgba[i + 2] * brightness;
    out[i + 3] = rgba[i + 3];
  }
  return out;
}

function luma(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// Тайловое CLAHE (clip-limited histogram equalization) с билинейной интерполяцией
// между центрами соседних тайлов — устраняет швы на границах тайлов. Значения по
// умолчанию (8x8, clip 2.5) — те же, что и cv2.createCLAHE(clipLimit=2.5,
// tileGridSize=(8,8)) в hakaton_nornikel/annotate_talc.py.
export function applyClahe(
  rgba: Uint8ClampedArray, w: number, h: number, tilesX = 8, tilesY = 8, clipLimit = 2.5,
): Uint8ClampedArray {
  const n = w * h;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    y[i] = luma(rgba[j], rgba[j + 1], rgba[j + 2]);
  }
  const tileW = Math.max(1, Math.ceil(w / tilesX));
  const tileH = Math.max(1, Math.ceil(h / tilesY));
  const cdfs: Uint8ClampedArray[][] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    const row: Uint8ClampedArray[] = [];
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * tileW, x1 = Math.min(w, x0 + tileW);
      const y0 = ty * tileH, y1 = Math.min(h, y0 + tileH);
      const hist = new Uint32Array(256);
      let count = 0;
      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          hist[Math.round(y[py * w + px])]++;
          count++;
        }
      }
      const limit = Math.max(1, Math.round((clipLimit * count) / 256));
      let excess = 0;
      for (let v = 0; v < 256; v++) {
        if (hist[v] > limit) { excess += hist[v] - limit; hist[v] = limit; }
      }
      const bump = Math.floor(excess / 256);
      const rem = excess - bump * 256;
      for (let v = 0; v < 256; v++) hist[v] += bump + (v < rem ? 1 : 0);
      const cdf = new Uint8ClampedArray(256);
      let cum = 0;
      const denom = Math.max(1, count);
      for (let v = 0; v < 256; v++) {
        cum += hist[v];
        cdf[v] = Math.round((cum / denom) * 255);
      }
      row.push(cdf);
    }
    cdfs.push(row);
  }
  const out = new Uint8ClampedArray(rgba.length);
  for (let py = 0; py < h; py++) {
    const fy = (py + 0.5) / tileH - 0.5;
    let ty0 = Math.floor(fy), wy = fy - ty0, ty1 = ty0 + 1;
    if (ty0 < 0) { ty0 = 0; wy = 0; }
    if (ty1 < 0) ty1 = 0;
    if (ty0 > tilesY - 1) ty0 = tilesY - 1;
    if (ty1 > tilesY - 1) { ty1 = tilesY - 1; wy = 0; }
    for (let px = 0; px < w; px++) {
      const fx = (px + 0.5) / tileW - 0.5;
      let tx0 = Math.floor(fx), wx = fx - tx0, tx1 = tx0 + 1;
      if (tx0 < 0) { tx0 = 0; wx = 0; }
      if (tx1 < 0) tx1 = 0;
      if (tx0 > tilesX - 1) tx0 = tilesX - 1;
      if (tx1 > tilesX - 1) { tx1 = tilesX - 1; wx = 0; }
      const i = py * w + px;
      const v = Math.round(y[i]);
      const m00 = cdfs[ty0][tx0][v], m01 = cdfs[ty0][tx1][v];
      const m10 = cdfs[ty1][tx0][v], m11 = cdfs[ty1][tx1][v];
      const top = m00 + (m01 - m00) * wx;
      const bot = m10 + (m11 - m10) * wx;
      const mapped = top + (bot - top) * wy;
      const ratio = mapped / Math.max(1, y[i]);
      const j = i * 4;
      out[j] = rgba[j] * ratio; out[j + 1] = rgba[j + 1] * ratio; out[j + 2] = rgba[j + 2] * ratio; out[j + 3] = rgba[j + 3];
    }
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: all tests PASS, including the three new ones in `enhance.test.mjs`.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/mask/enhance.ts frontend/tests/enhance.test.mjs
git commit -m "feat(corrector): add brightness+CLAHE display-enhance helpers"
```

---

## Task 2: `lib/mask/darkpercent.ts` — percentile dark-segments hint

**Files:**
- Create: `frontend/lib/mask/darkpercent.ts`
- Test: `frontend/tests/darkpercent.test.mjs`

**Interfaces:**
- Produces: `percentileThreshold(dark: Uint8Array, phaseMap: Uint8Array, frac: number): number`
- Produces: `darkSegmentsMask(dark: Uint8Array, phaseMap: Uint8Array, talc: Uint8Array, frac: number): Uint8Array`
- Consumed by: Task 5 (`Corrector.tsx`).

- [ ] **Step 1: Write the failing test**

Create `frontend/tests/darkpercent.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert";
import { percentileThreshold, darkSegmentsMask } from "../lib/mask/darkpercent.ts";

test("percentileThreshold finds the value covering the target fraction of matrix pixels", () => {
  const dark = Uint8Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
  const phaseMap = Uint8Array.from([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // all matrix
  const t = percentileThreshold(dark, phaseMap, 0.5);
  assert.strictEqual(t, 50); // darkest 50% => values <=50 covers 5/10
});

test("percentileThreshold ignores non-matrix pixels", () => {
  const dark = Uint8Array.from([10, 200, 20, 200, 30]);
  const phaseMap = Uint8Array.from([0, 2, 0, 1, 0]); // only idx0,2,4 are matrix (10,20,30)
  const t = percentileThreshold(dark, phaseMap, 1 / 3); // darkest third of 3 matrix px => value 10
  assert.strictEqual(t, 10);
});

test("percentileThreshold returns -1 when there is no matrix", () => {
  const dark = Uint8Array.from([10, 20]);
  const phaseMap = Uint8Array.from([1, 2]);
  assert.strictEqual(percentileThreshold(dark, phaseMap, 0.5), -1);
});

test("darkSegmentsMask marks darkest matrix pixels not already talc", () => {
  const dark = Uint8Array.from([10, 20, 30, 40]);
  const phaseMap = Uint8Array.from([0, 0, 0, 0]);
  const talc = Uint8Array.from([0, 1, 0, 0]);
  const mask = darkSegmentsMask(dark, phaseMap, talc, 0.5); // threshold=20; idx1 excluded (already talc)
  assert.deepStrictEqual([...mask], [1, 0, 0, 0]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --test-name-pattern=percentileThreshold`
Expected: FAIL — `Cannot find module '../lib/mask/darkpercent.ts'`

- [ ] **Step 3: Write minimal implementation**

Create `frontend/lib/mask/darkpercent.ts`:

```ts
// Слой-подсказка «тёмные сегменты»: по карте темноты (0=чёрный..255=белый) и текущей
// фазовой карте находим самую тёмную долю `frac` пикселей матрицы — кандидат зоны
// талька (см. darkest_segments_mask в hakaton_nornikel, там — ранжирование целых
// суперпикселей; здесь — попиксельно, проще и не менее точно для превью-подсказки).
export function percentileThreshold(dark: Uint8Array, phaseMap: Uint8Array, frac: number): number {
  const hist = new Uint32Array(256);
  let total = 0;
  for (let i = 0; i < dark.length; i++) {
    if (phaseMap[i] === 0) { hist[dark[i]]++; total++; }
  }
  if (total === 0) return -1;
  const target = frac * total;
  let cum = 0;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= target) return v;
  }
  return 255;
}

export function darkSegmentsMask(dark: Uint8Array, phaseMap: Uint8Array, talc: Uint8Array, frac: number): Uint8Array {
  const t = percentileThreshold(dark, phaseMap, frac);
  const out = new Uint8Array(dark.length);
  if (t < 0) return out;
  for (let i = 0; i < dark.length; i++) {
    if (phaseMap[i] === 0 && talc[i] === 0 && dark[i] <= t) out[i] = 1;
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: all tests PASS, including the four new ones in `darkpercent.test.mjs`.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/mask/darkpercent.ts frontend/tests/darkpercent.test.mjs
git commit -m "feat(corrector): add percentile-threshold dark-segments hint helper"
```

---

## Task 3: Corrector.tsx — tab split (Редактирование / Отчёт)

**Files:**
- Modify: `frontend/components/corrector/Corrector.tsx`

**Interfaces:**
- Consumes: nothing new (pure layout change).
- Produces: `sideTab` state (`"edit" | "report"`), read by Tasks 4–6's JSX placement (all new controls go inside the `sideTab === "edit"` branch, which already exists as the mask-editor `<div className="card">`).

- [ ] **Step 1: Add `sideTab` state**

In `frontend/components/corrector/Corrector.tsx`, find:

```tsx
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
  const [vis, setVis] = useState<Vis>({ sulfide: true, magnetite: true, talc: true });
  const visRef = useRef(vis); visRef.current = vis;
  const [grabbing, setGrabbing] = useState(false);
  const zp = useZoomPan();
```

Replace with:

```tsx
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
  const [vis, setVis] = useState<Vis>({ sulfide: true, magnetite: true, talc: true });
  const visRef = useRef(vis); visRef.current = vis;
  const [grabbing, setGrabbing] = useState(false);
  const [sideTab, setSideTab] = useState<"edit" | "report">("edit");
  const zp = useZoomPan();
```

- [ ] **Step 2: Make the report tab pan-only (no accidental edits while reading the report)**

Find:

```tsx
  function moveCursor(e: React.PointerEvent) {
    const cd = cursorRef.current, vp = zp.vpRef.current, cv = canvasRef.current;
    if (!cd) return;
    if (!vp || !cv || !state || (state.tool !== "brush" && state.tool !== "eraser")) { cd.style.display = "none"; return; }
```

Replace with:

```tsx
  function moveCursor(e: React.PointerEvent) {
    const cd = cursorRef.current, vp = zp.vpRef.current, cv = canvasRef.current;
    if (!cd) return;
    if (!vp || !cv || !state || sideTab === "report" || (state.tool !== "brush" && state.tool !== "eraser")) { cd.style.display = "none"; return; }
```

Find:

```tsx
  function onPointerDown(e: React.PointerEvent) {
    if ((e.target as Element).closest(".zoom-controls")) return; // клики по кнопкам зума не рисуют
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (!state) return;
    if (e.button === 1 || state.tool === "pan") { setGrabbing(true); zp.startPan(e); return; }
```

Replace with:

```tsx
  function onPointerDown(e: React.PointerEvent) {
    if ((e.target as Element).closest(".zoom-controls")) return; // клики по кнопкам зума не рисуют
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (!state) return;
    if (sideTab === "report" || e.button === 1 || state.tool === "pan") { setGrabbing(true); zp.startPan(e); return; }
```

Find:

```tsx
  const vpClass = state?.tool === "pan" ? (grabbing ? "grabbing" : "grab")
    : (state?.tool === "brush" || state?.tool === "eraser") ? "paint" : "";
```

Replace with:

```tsx
  const vpClass = sideTab === "report" || state?.tool === "pan" ? (grabbing ? "grabbing" : "grab")
    : (state?.tool === "brush" || state?.tool === "eraser") ? "paint" : "";
```

- [ ] **Step 3: Split the sidebar into two tabs**

Find:

```tsx
      <aside className="ws-side">
        {info}
        <div className="card">
          <div className="side-h">Редактор масок<span className="ann">{state ? `${state.brush}px` : "…"}</span></div>
          <div className="side-b">
```

Replace with:

```tsx
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
```

Then find the end of that same card (the closing of `side-b`/`card`/`aside`):

```tsx
          </div>
        </div>
      </aside>

      <div className="ws-view">
```

Replace with:

```tsx
          </div>
        </div>
        )}
      </aside>

      <div className="ws-view">
```

- [ ] **Step 4: Verify it builds**

Run: `cd frontend && npm run build`
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 5: Manual verification**

Run: `cd frontend && npm run dev`, open the app, upload a closeup image (or use an existing job).
Check:
- The sidebar shows a «Редактирование | Отчёт» switcher at the top; «Редактирование» is active by default and shows the mask-editor card immediately (nothing to scroll past).
- Clicking «Отчёт» shows the Образец/Вердикт/download card instead.
- While «Отчёт» is active, dragging on the image pans instead of painting, regardless of which tool was last selected.
- Switching back to «Редактирование» restores normal tool behavior.

- [ ] **Step 6: Commit**

```bash
git add frontend/components/corrector/Corrector.tsx
git commit -m "feat(corrector): split sidebar into Редактирование/Отчёт tabs"
```

---

## Task 4: Corrector.tsx — «Вид» section (opacity / brightness / CLAHE)

**Files:**
- Modify: `frontend/components/corrector/Corrector.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `applyBrightness`, `applyClahe` from Task 1.
- Produces: `maskAlpha`/`maskAlphaRef` state, `enhancedRGBA` ref — read by Task 5's `composePixel` extension.

- [ ] **Step 1: Import the enhance helpers**

Find:

```tsx
import {
  IconSave, IconUndo, IconRedo, IconZoomIn, IconZoomOut, IconReset, IconHand,
  IconBrush, IconEye, IconEyeOff,
} from "@/components/icons";
```

Replace with:

```tsx
import {
  IconSave, IconUndo, IconRedo, IconZoomIn, IconZoomOut, IconReset, IconHand,
  IconBrush, IconEye, IconEyeOff,
} from "@/components/icons";
import { applyBrightness, applyClahe } from "@/lib/mask/enhance";
```

- [ ] **Step 2: Add the enhanced-buffer ref**

Find:

```tsx
  const baseRGBA = useRef<Uint8ClampedArray | null>(null);
  const outRef = useRef<ImageData | null>(null);
```

Replace with:

```tsx
  const baseRGBA = useRef<Uint8ClampedArray | null>(null);
  const enhancedRGBA = useRef<Uint8ClampedArray | null>(null);
  const outRef = useRef<ImageData | null>(null);
```

- [ ] **Step 3: Add view-control state**

Find:

```tsx
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
  const [vis, setVis] = useState<Vis>({ sulfide: true, magnetite: true, talc: true });
  const visRef = useRef(vis); visRef.current = vis;
  const [grabbing, setGrabbing] = useState(false);
  const [sideTab, setSideTab] = useState<"edit" | "report">("edit");
  const zp = useZoomPan();
```

Replace with:

```tsx
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
```

- [ ] **Step 4: Wire `maskAlpha` and the enhanced buffer into `composePixel`**

Find:

```tsx
  function composePixel(pm: Uint8Array, tc: Uint8Array, i: number) {
    const b = baseRGBA.current!, o = outRef.current!.data; const j = i * 4;
    let r = b[j], g = b[j + 1], bl = b[j + 2];
    const cls = pm[i], v = visRef.current;
    if ((cls === 2 && v.sulfide) || (cls === 1 && v.magnetite)) {
      const c = PHASE_RGB[cls];
      r = 0.45 * r + 0.55 * c[0]; g = 0.45 * g + 0.55 * c[1]; bl = 0.45 * bl + 0.55 * c[2];
    }
    if (tc[i] && v.talc) {
      r = 0.4 * r + 0.6 * TALC_RGB[0]; g = 0.4 * g + 0.6 * TALC_RGB[1]; bl = 0.4 * bl + 0.6 * TALC_RGB[2];
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
```

Replace with:

```tsx
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
```

- [ ] **Step 5: Redraw on `maskAlpha` change; recompute the enhanced buffer on brightness/CLAHE change**

Find:

```tsx
  useEffect(() => {
    if (state && baseRGBA.current && !strokeRef.current) {
      srcRef.current = { pm: state.phaseMap, tc: state.talc };
      requestDraw();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phaseMap, state?.talc, vis]);
```

Replace with:

```tsx
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
```

- [ ] **Step 6: Add the «Вид» control group to the sidebar JSX**

Find:

```tsx
                <div className="tool-group" style={{ display: "flex", gap: 8 }}>
                  <button type="button" className="btn ghost sm icon" title="Отменить" aria-label="Отменить"
                    onClick={() => setState(undo(state))}><IconUndo /></button>
```

Replace with:

```tsx
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
```

- [ ] **Step 7: Add the `.switch` toggle CSS**

In `frontend/app/globals.css`, find:

```css
.tool-group { display: grid; gap: 9px; }
.tool-group + .tool-group { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
```

Replace with:

```css
.tool-group { display: grid; gap: 9px; }
.tool-group + .tool-group { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }

/* ---- Тумблер (вид/превью, не меняет маску) ---- */
.switch-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 12.5px; color: var(--text-2); }
.switch { position: relative; width: 36px; height: 21px; border-radius: 999px; background: var(--surface-3); border: 1px solid var(--border); padding: 0; cursor: pointer; flex-shrink: 0; transition: background .15s, border-color .15s; }
.switch .knob { position: absolute; top: 1px; left: 1px; width: 17px; height: 17px; border-radius: 50%; background: var(--surface); box-shadow: var(--shadow-1); transition: transform .15s; }
.switch.on { background: var(--brand); border-color: var(--brand); }
.switch.on .knob { transform: translateX(15px); }
```

- [ ] **Step 8: Verify it builds**

Run: `cd frontend && npm run build`
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 9: Manual verification**

Run: `cd frontend && npm run dev`, open a closeup job in the corrector.
Check:
- Under «Вид (не меняет маску)» in the «Редактирование» tab: moving «прозрачность» changes overlay strength live; moving «яркость» changes image brightness live; toggling CLAHE visibly increases local contrast/texture.
- None of these controls change the saved mask: paint something, adjust all three view controls, click «Сохранить» — the resulting mask (visible after reload) matches what was painted, unaffected by the view settings.

- [ ] **Step 10: Commit**

```bash
git add frontend/components/corrector/Corrector.tsx frontend/app/globals.css
git commit -m "feat(corrector): add Вид section (opacity/brightness/CLAHE)"
```

---

## Task 5: Corrector.tsx — «Слои-подсказки» (dark-segments preview → apply)

**Files:**
- Modify: `frontend/components/corrector/Corrector.tsx`

**Interfaces:**
- Consumes: `darkSegmentsMask` from Task 2; the `composePixel`/effect structure from Task 4.
- Produces: `darkMaskRef` (read by nothing outside this task, but keeps the same naming for Task 6 to leave untouched).

- [ ] **Step 1: Import the dark-segments helper**

Find:

```tsx
import { applyBrightness, applyClahe } from "@/lib/mask/enhance";
```

Replace with:

```tsx
import { applyBrightness, applyClahe } from "@/lib/mask/enhance";
import { darkSegmentsMask } from "@/lib/mask/darkpercent";
```

- [ ] **Step 2: Add the preview color constant**

Find:

```tsx
const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
```

Replace with:

```tsx
const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const DARK_RGB: [number, number, number] = [200, 60, 220];
```

- [ ] **Step 3: Add hint-layer state**

Find:

```tsx
  const [maskAlpha, setMaskAlpha] = useState(0.5);
  const maskAlphaRef = useRef(maskAlpha); maskAlphaRef.current = maskAlpha;
  const [brightness, setBrightness] = useState(1.0);
  const [clahe, setClahe] = useState(false);
```

Replace with:

```tsx
  const [maskAlpha, setMaskAlpha] = useState(0.5);
  const maskAlphaRef = useRef(maskAlpha); maskAlphaRef.current = maskAlpha;
  const [brightness, setBrightness] = useState(1.0);
  const [clahe, setClahe] = useState(false);
  const [darkFrac, setDarkFrac] = useState(45);
  const [showDarkPreview, setShowDarkPreview] = useState(false);
  const darkMaskRef = useRef<Uint8Array | null>(null);
```

- [ ] **Step 4: Blend the preview overlay into `composePixel`**

Find:

```tsx
    if (tc[i] && v.talc) {
      r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
```

Replace with:

```tsx
    if (tc[i] && v.talc) {
      r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
    }
    const dm = darkMaskRef.current;
    if (dm && dm[i] && !tc[i]) {
      r = 0.5 * r + 0.5 * DARK_RGB[0]; g = 0.5 * g + 0.5 * DARK_RGB[1]; bl = 0.5 * bl + 0.5 * DARK_RGB[2];
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
```

- [ ] **Step 5: Recompute the preview mask on demand, and add the apply function**

Find:

```tsx
    if (state && !strokeRef.current) { srcRef.current = { pm: state.phaseMap, tc: state.talc }; requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brightness, clahe]);
```

Replace with:

```tsx
    if (state && !strokeRef.current) { srcRef.current = { pm: state.phaseMap, tc: state.talc }; requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brightness, clahe]);

  // Слой-подсказка «тёмные сегменты»: превью пересчитывается только пока включено;
  // применение в маску — отдельная явная кнопка (applyDarkSegments), не автоматически.
  useEffect(() => {
    if (!state || !darkRef.current || !showDarkPreview) { darkMaskRef.current = null; requestDraw(); return; }
    darkMaskRef.current = darkSegmentsMask(darkRef.current, state.phaseMap, state.talc, darkFrac / 100);
    if (!strokeRef.current) { srcRef.current = { pm: state.phaseMap, tc: state.talc }; requestDraw(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [darkFrac, showDarkPreview, state?.phaseMap, state?.talc]);

  function applyDarkSegments() {
    if (!state || !darkRef.current) return;
    const mask = darkSegmentsMask(darkRef.current, state.phaseMap, state.talc, darkFrac / 100);
    const idxs: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) idxs.push(i);
    setState(applyTalc(state, idxs, true));
  }
```

- [ ] **Step 6: Add the «Слои-подсказки» control group to the sidebar JSX**

Find:

```tsx
                  <label className="switch-row">
                    <span>CLAHE (контраст)</span>
                    <button type="button" className={`switch${clahe ? " on" : ""}`} role="switch" aria-checked={clahe}
                      onClick={() => setClahe((v) => !v)}><span className="knob" /></button>
                  </label>
                </div>
                <div className="tool-group" style={{ display: "flex", gap: 8 }}>
                  <button type="button" className="btn ghost sm icon" title="Отменить" aria-label="Отменить"
                    onClick={() => setState(undo(state))}><IconUndo /></button>
```

Replace with:

```tsx
                  <label className="switch-row">
                    <span>CLAHE (контраст)</span>
                    <button type="button" className={`switch${clahe ? " on" : ""}`} role="switch" aria-checked={clahe}
                      onClick={() => setClahe((v) => !v)}><span className="knob" /></button>
                  </label>
                </div>
                <div className="tool-group">
                  <span className="toolbar-label">Слои-подсказки (превью → применить в маску)</span>
                  <label className="ctl">тёмные сегменты, % матрицы
                    <input className="slider" type="range" min={10} max={70} step={5} value={darkFrac}
                      onChange={(e) => setDarkFrac(+e.target.value)} />
                    <span className="slider-val">{darkFrac}%</span>
                  </label>
                  <label className="switch-row">
                    <span>Показать превью</span>
                    <button type="button" className={`switch${showDarkPreview ? " on" : ""}`} role="switch" aria-checked={showDarkPreview}
                      onClick={() => setShowDarkPreview((v) => !v)}><span className="knob" /></button>
                  </label>
                  <button type="button" className="btn ghost sm" onClick={applyDarkSegments}>+ Применить к тальку</button>
                </div>
                <div className="tool-group" style={{ display: "flex", gap: 8 }}>
                  <button type="button" className="btn ghost sm icon" title="Отменить" aria-label="Отменить"
                    onClick={() => setState(undo(state))}><IconUndo /></button>
```

- [ ] **Step 7: Verify it builds**

Run: `cd frontend && npm run build`
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 8: Manual verification**

Run: `cd frontend && npm run dev`, open a closeup job in the corrector.
Check:
- Moving the «тёмные сегменты, % матрицы» slider with «Показать превью» on updates a purple highlight live, over matrix pixels only, excluding pixels already marked talc.
- Clicking «+ Применить к тальку» adds the previewed pixels into the talc mask (blue), and an «Отменить» (undo) reverts it.
- Turning «Показать превью» off removes the purple highlight without touching the mask.

- [ ] **Step 9: Commit**

```bash
git add frontend/components/corrector/Corrector.tsx
git commit -m "feat(corrector): add Слои-подсказки dark-segments preview/apply"
```

---

## Task 6: Corrector.tsx — per-mode Инструмент settings

**Files:**
- Modify: `frontend/components/corrector/Corrector.tsx`
- Modify: `frontend/components/corrector/reducer.ts`

**Interfaces:**
- Consumes: `applyPhase`, `applyTalc` (existing, unchanged signatures) from `reducer.ts`.
- Produces: nothing consumed by later tasks (this is the last task).

- [ ] **Step 1: Drop the now-unused `"threshold"` tool from the `Tool` union**

In `frontend/components/corrector/reducer.ts`, find:

```ts
export type Tool = "superpixel" | "brush" | "eraser" | "threshold" | "pan";
```

Replace with:

```ts
export type Tool = "superpixel" | "brush" | "eraser" | "pan";
```

- [ ] **Step 2: Remove "Тёмные области" from the tool list**

In `frontend/components/corrector/Corrector.tsx`, find:

```tsx
const TOOLS: [Tool, string][] = [
  ["brush", "Кисть"], ["eraser", "Ластик"], ["superpixel", "Суперпиксель"], ["threshold", "Тёмные области"], ["pan", "Рука"],
];
```

Replace with:

```tsx
const TOOLS: [Tool, string][] = [
  ["brush", "Кисть"], ["eraser", "Ластик"], ["superpixel", "Суперпиксель"], ["pan", "Рука"],
];
```

- [ ] **Step 3: Remove the now-unused `thr` state**

Find:

```tsx
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);
```

Replace with:

```tsx
  const [state, setState] = useState<CorrectorState | null>(null);
  const [saving, setSaving] = useState(false);
```

- [ ] **Step 4: Add the superpixel Ставить/Убирать action state**

Find:

```tsx
  const [darkFrac, setDarkFrac] = useState(45);
  const [showDarkPreview, setShowDarkPreview] = useState(false);
  const darkMaskRef = useRef<Uint8Array | null>(null);
```

Replace with:

```tsx
  const [darkFrac, setDarkFrac] = useState(45);
  const [showDarkPreview, setShowDarkPreview] = useState(false);
  const darkMaskRef = useRef<Uint8Array | null>(null);
  const [spAction, setSpAction] = useState<"Ставить" | "Убирать">("Ставить");
```

- [ ] **Step 5: Remove the threshold click-tool branch; respect `spAction` for superpixel clicks**

Find:

```tsx
    } else if (state.tool === "superpixel" && spRef.current) {
      const idxs = cellIndices(spRef.current, cy * w + cx);
      setState(state.layer === "talc" ? applyTalc(state, idxs, true) : applyPhase(state, idxs, state.layer));
    } else if (state.tool === "threshold" && darkRef.current) {
      const idxs: number[] = [];
      for (let i = 0; i < w * h; i++) if (darkRef.current[i] <= thr && state.phaseMap[i] === 0) idxs.push(i);
      setState(applyTalc(state, idxs, true));
    }
```

Replace with:

```tsx
    } else if (state.tool === "superpixel" && spRef.current) {
      const idxs = cellIndices(spRef.current, cy * w + cx);
      const setting = spAction === "Ставить";
      setState(state.layer === "talc" ? applyTalc(state, idxs, setting) : applyPhase(state, idxs, setting ? state.layer : "matrix"));
    }
```

- [ ] **Step 6: Gate the brush-size slider to Кисть/Ластик; add the superpixel action toggle**

Find:

```tsx
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
```

Replace with:

```tsx
                {(state.tool === "brush" || state.tool === "eraser") && (
                  <div className="tool-group">
                    <span className="toolbar-label">Кисть</span>
                    <label className="ctl">размер
                      <input className="slider" type="range" min={2} max={60} value={state.brush}
                        onChange={(e) => setState({ ...state, brush: +e.target.value })} />
                      <span className="slider-val">{state.brush}px</span>
                    </label>
                  </div>
                )}
                {state.tool === "superpixel" && (
                  <div className="tool-group">
                    <span className="toolbar-label">Клик по суперпикселю</span>
                    <div className="seg" role="group" aria-label="Клик по суперпикселю">
                      {(["Ставить", "Убирать"] as const).map((a) => (
                        <button key={a} type="button" className={spAction === a ? "active" : ""} aria-pressed={spAction === a}
                          onClick={() => setSpAction(a)}>{a}</button>
                      ))}
                    </div>
                  </div>
                )}
```

- [ ] **Step 7: Run the unit tests**

Run: `cd frontend && npm test`
Expected: all tests PASS (`reducer`, `client`, `encode`, `enhance`, `darkpercent` — `reducer.test.mjs` is unaffected since `layerToClass`/`applyPhase`/`applyTalc` signatures didn't change).

- [ ] **Step 8: Verify it builds**

Run: `cd frontend && npm run build`
Expected: build succeeds, no TypeScript errors, no unused-variable warnings for `thr`/`"threshold"`.

- [ ] **Step 9: Manual verification**

Run: `cd frontend && npm run dev`, open a closeup job in the corrector.
Check:
- Tool list no longer shows «Тёмные области».
- Selecting Кисть or Ластик shows the «Кисть» size slider; selecting Суперпиксель or Рука hides it.
- Selecting Суперпиксель shows «Клик по суперпикселю: Ставить/Убирать»; with «Убирать» active, clicking a superpixel on a phase layer resets it to matrix, and on the talc layer clears the talc bit — with «Ставить» active, behavior matches the old (pre-change) click-to-add behavior.

- [ ] **Step 10: Commit**

```bash
git add frontend/components/corrector/Corrector.tsx frontend/components/corrector/reducer.ts
git commit -m "feat(corrector): per-mode Инструмент settings — gate brush size, add superpixel Ставить/Убирать"
```
