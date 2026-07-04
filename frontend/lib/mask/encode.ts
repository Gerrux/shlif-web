// Convert a 0/1 mask to a 0/255 grayscale PNG blob using an offscreen canvas.
export async function maskToPngBlob(mask: Uint8Array, w: number, h: number): Promise<Blob> {
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 255 : 0;
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob((b) => res(b as Blob), "image/png"));
}
// Pure helper (unit-testable without a DOM): pack a class label map to bytes.
export function labelMapToBytes(map: Uint8Array): Uint8Array {
  return Uint8Array.from(map); // already 0/1/2 per pixel
}
// Convert a raw label map (e.g. phase ids 0/1/2) to a grayscale PNG blob whose
// pixel value IS the label (not thresholded to 0/255 like maskToPngBlob).
// The backend reads phases.png back as a raw uint8 label map, so this must
// preserve values exactly.
export async function rawMaskToPngBlob(map: Uint8Array, w: number, h: number): Promise<Blob> {
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < map.length; i++) {
    const v = map[i];
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob((b) => res(b as Blob), "image/png"));
}
