"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { imageUrl } from "@/lib/api/client";
import type { Mode } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";

export default function Home() {
  const [mode, setMode] = useState<Mode>("closeup");
  const [jobId, setJobId] = useState<string | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) analyze.mutate({ file: f, mode }, { onSuccess: (r) => setJobId(r.job_id) });
  }
  const result = job.data?.status === "done" ? job.data.result : null;
  const busy = analyze.isPending || (job.data && ["queued", "running"].includes(job.data.status));

  return (
    <main style={{ maxWidth: 1280, margin: "0 auto", padding: "22px 20px" }}>
      <div className="topbar"><div className="logo">◈</div>
        <div><div className="crumb">Шлиф · классификация руд</div><h1 style={{ margin: 0 }}>Скажи мне кто твой шлиф</h1></div></div>
      <div style={{ display: "flex", gap: 12, margin: "14px 0" }}>
        <label><input type="radio" checked={mode === "closeup"} onChange={() => setMode("closeup")} /> Крупный план</label>
        <label><input type="radio" checked={mode === "panorama"} onChange={() => setMode("panorama")} /> Панорама</label>
        <input type="file" accept="image/*" onChange={onFile} />
      </div>
      <div className="grid2">
        <div className="stage">
          {jobId && result ? <img src={result.overlay_url ?? imageUrl(jobId)} alt="шлиф" /> :
            <div style={{ padding: 40, color: "var(--muted)" }}>{busy ? "Анализ…" : "Загрузите снимок шлифа"}</div>}
        </div>
        <div>{result ? <VerdictPanel result={result} /> :
          job.data?.status === "error" ? <div className="note">Ошибка: {job.data.message}</div> : null}</div>
      </div>
    </main>
  );
}
