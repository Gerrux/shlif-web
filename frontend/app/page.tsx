"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { imageUrl } from "@/lib/api/client";
import type { Mode, Verdict } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";

export default function Home() {
  const [mode, setMode] = useState<Mode>("closeup");
  const [jobId, setJobId] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setEditing(false);
      setVOverride(null);
      analyze.mutate({ file: f, mode }, { onSuccess: (r) => setJobId(r.job_id) });
    }
  }
  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
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
          {editing && result?.size ? (
            <Corrector jobId={jobId!} size={result.size} onVerdict={(v) => setVOverride(v)} />
          ) : jobId && shown ? <img src={shown.overlay_url ?? imageUrl(jobId)} alt="шлиф" /> :
            <div style={{ padding: 40, color: "var(--muted)" }}>{busy ? "Анализ…" : "Загрузите снимок шлифа"}</div>}
        </div>
        <div>
          {shown ? <VerdictPanel result={shown} /> :
            job.data?.status === "error" ? <div className="note">Ошибка: {job.data.message}</div> : null}
          {shown && mode === "closeup" && !editing ? <button onClick={() => setEditing(true)} style={{ marginTop: 12 }}>✎ Доработать маски</button> : null}
        </div>
      </div>
    </main>
  );
}
