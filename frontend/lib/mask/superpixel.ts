// Decode the SLIC label map PNG into a Uint16Array of per-pixel segment ids.
// The backend packs each uint16 id into an 8-bit RGB PNG (R=id>>8, G=id&255) —
// see `encode_png_label_rgb` in backend/app/pipeline/masks.py. Canvas
// getImageData is 8-bit/channel, so a genuine 16-bit single-channel PNG would
// lose its low byte; the R/G byte-pair survives canvas losslessly instead.
export async function loadSuperpixels(url: string, w: number, h: number): Promise<Uint16Array> {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint16Array(w * h);
  // Reconstruct the id from the red (high byte) + green (low byte) pair.
  for (let i = 0; i < w * h; i++) out[i] = (data[i * 4] << 8) | data[i * 4 + 1];
  return out;
}
export function cellIndices(labels: Uint16Array, seedIdx: number): number[] {
  const id = labels[seedIdx];
  const out: number[] = [];
  for (let i = 0; i < labels.length; i++) if (labels[i] === id) out.push(i);
  return out;
}
