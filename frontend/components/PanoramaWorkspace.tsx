"use client";
// Просмотрщик панорамы: зумируемое/панорамируемое изображение оверлея + сайдбар с
// подробной информацией. Правка масок панорам требует поддержки бэкенда — здесь только просмотр.
import { useState, type ReactNode } from "react";
import { useZoomPan } from "@/lib/useZoomPan";
import { IconZoomIn, IconZoomOut, IconReset } from "@/components/icons";

export function PanoramaWorkspace({ src, info }: { src: string; info?: ReactNode }) {
  const zp = useZoomPan();
  const [grabbing, setGrabbing] = useState(false);
  return (
    <div className="workspace">
      <aside className="ws-side">{info}</aside>
      <div className="ws-view">
        <div ref={zp.vpRef} className={`zoom-vp ${grabbing ? "grabbing" : "grab"}`}
          onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setGrabbing(true); zp.startPan(e); }}
          onPointerMove={(e) => { if (zp.isPanning()) zp.movePan(e); }}
          onPointerUp={() => { zp.endPan(); setGrabbing(false); }}
          onPointerLeave={() => { zp.endPan(); setGrabbing(false); }}>
          <div className="zoom-content" style={{ transform: zp.transform }}>
            <img src={src} alt="панорама шлифа" draggable={false} />
          </div>
          <div className="zoom-hint">колесо — зум · перетаскивание — сдвиг</div>
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
