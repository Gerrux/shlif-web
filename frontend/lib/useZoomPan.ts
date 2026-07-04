"use client";
// Зум/панорамирование области просмотра. Контент трансформируется через CSS
// (translate+scale); координаты рисования берутся из canvas.getBoundingClientRect(),
// который уже учитывает трансформацию, поэтому маппинг мыши в пиксели маски не меняется.
import { useCallback, useEffect, useRef, useState } from "react";

export interface View { zoom: number; x: number; y: number }
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

export function useZoomPan(min = 1, max = 12) {
  const [view, setView] = useState<View>({ zoom: 1, x: 0, y: 0 });
  const vpRef = useRef<HTMLDivElement>(null);
  const panRef = useRef<{ sx: number; sy: number; x: number; y: number } | null>(null);

  const zoomAt = useCallback((clientX: number, clientY: number, factor: number) => {
    const vp = vpRef.current;
    if (!vp) return;
    const rect = vp.getBoundingClientRect();
    const mx = clientX - rect.left, my = clientY - rect.top;
    setView((v) => {
      const nz = clamp(v.zoom * factor, min, max);
      if (nz === 1) return { zoom: 1, x: 0, y: 0 };
      const cx = (mx - v.x) / v.zoom, cy = (my - v.y) / v.zoom;
      return { zoom: nz, x: mx - cx * nz, y: my - cy * nz };
    });
  }, [min, max]);

  // Нативный wheel-листенер с passive:false — иначе preventDefault не сработает.
  useEffect(() => {
    const vp = vpRef.current;
    if (!vp) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.18 : 1 / 1.18);
    };
    vp.addEventListener("wheel", onWheel, { passive: false });
    return () => vp.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  const startPan = useCallback((e: { clientX: number; clientY: number }) => {
    setView((v) => { panRef.current = { sx: e.clientX, sy: e.clientY, x: v.x, y: v.y }; return v; });
  }, []);
  const movePan = useCallback((e: { clientX: number; clientY: number }) => {
    const p = panRef.current;
    if (!p) return;
    setView((v) => ({ ...v, x: p.x + (e.clientX - p.sx), y: p.y + (e.clientY - p.sy) }));
  }, []);
  const endPan = useCallback(() => { panRef.current = null; }, []);
  const isPanning = useCallback(() => panRef.current != null, []);

  const zoomCenter = useCallback((factor: number) => {
    const vp = vpRef.current;
    if (!vp) return;
    const rect = vp.getBoundingClientRect();
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, factor);
  }, [zoomAt]);
  const reset = useCallback(() => setView({ zoom: 1, x: 0, y: 0 }), []);

  const transform = `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`;
  return { view, vpRef, transform, startPan, movePan, endPan, isPanning, reset,
    zoomIn: () => zoomCenter(1.4), zoomOut: () => zoomCenter(1 / 1.4) };
}
