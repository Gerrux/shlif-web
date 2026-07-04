"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { tileManifestUrl } from "@/lib/api/client";
import { IconZoomIn } from "@/components/icons";
import type { TileManifest } from "@/components/DeepZoomViewer";

const DeepZoomViewer = dynamic(
  () => import("@/components/DeepZoomViewer").then((m) => m.DeepZoomViewer),
  { ssr: false }
);

export function PanoramaZoomModal({ jobId }: { jobId: string }) {
  const [manifest, setManifest] = useState<TileManifest | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(tileManifestUrl(jobId))
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => { if (!cancelled) setManifest(m); })
      .catch(() => { if (!cancelled) setManifest(null); });
    return () => { cancelled = true; };
  }, [jobId]);

  if (!manifest) return null;

  return (
    <>
      <button type="button" className="btn ghost" onClick={() => setOpen(true)}>
        <IconZoomIn /> Открыть в максимальном разрешении
      </button>
      {open ? (
        <div className="deepzoom-modal">
          <button type="button" className="btn dark sm icon deepzoom-close" aria-label="Закрыть" onClick={() => setOpen(false)}>×</button>
          <DeepZoomViewer jobId={jobId} manifest={manifest} />
        </div>
      ) : null}
    </>
  );
}
