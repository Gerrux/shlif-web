"use client";
import { useEffect, useRef } from "react";
import OpenSeadragon from "openseadragon";
import { tileUrl } from "@/lib/api/client";

export interface TileManifest { width: number; height: number; tileSize: number; maxLevel: number }

export function DeepZoomViewer({ jobId, manifest }: { jobId: string; manifest: TileManifest }) {
  const elRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!elRef.current) return;
    const viewer = OpenSeadragon({
      element: elRef.current,
      showNavigationControl: false,
      tileSources: {
        width: manifest.width,
        height: manifest.height,
        tileSize: manifest.tileSize,
        minLevel: 0,
        maxLevel: manifest.maxLevel,
        getTileUrl: (level: number, x: number, y: number) => tileUrl(jobId, level, x, y),
      },
    });
    return () => viewer.destroy();
  }, [jobId, manifest]);

  return <div ref={elRef} className="osd-container" />;
}
