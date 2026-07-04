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
