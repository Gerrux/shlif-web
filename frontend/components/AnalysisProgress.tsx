"use client";
import { useEffect, useState } from "react";
import type { Job } from "@/lib/api/types";
import { clampPct, computeEta, formatDuration } from "@/lib/progress";

export function AnalysisProgress({ job, startedAt, fallback }: { job?: Job; startedAt: number; fallback: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);

  const progress = job?.progress ?? 0;
  const pct = clampPct(progress);
  const elapsedSec = Math.max(0, (now - startedAt) / 1000);
  const etaSec = computeEta(elapsedSec, progress);

  return (
    <div className="stage-empty">
      <div className="hint">Анализ снимка…</div>
      <div className="sub">{job?.message || fallback}</div>
      <div className="progress-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-meta">
        <span>{pct}%</span>
        <span>{formatDuration(elapsedSec)}{etaSec != null ? ` · осталось ≈ ${formatDuration(etaSec)}` : ""}</span>
      </div>
    </div>
  );
}
