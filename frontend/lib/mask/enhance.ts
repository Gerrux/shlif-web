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
