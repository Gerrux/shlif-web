export function clampPct(progress: number): number {
  const clamped = Math.min(1, Math.max(0, progress));
  return Math.round(clamped * 100);
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} с`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m} мин ${rem} с`;
}

export function computeEta(elapsedSec: number, progress: number): number | null {
  if (progress < 0.08) return null;
  const total = elapsedSec / progress;
  return Math.max(0, total - elapsedSec);
}
