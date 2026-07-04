"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { imageUrl, reportUrl } from "@/lib/api/client";
import type { Mode, Verdict } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { Welcome } from "@/components/Welcome";
import { ThemeToggle } from "@/components/ThemeToggle";
import { IconHex, IconUpload, IconEdit, IconScan, IconAlert } from "@/components/icons";

const MODES: [Mode, string][] = [["closeup", "Крупный план"], ["panorama", "Панорама"]];
const STATUS: Record<string, [string, string]> = {
  queued: ["queued", "в очереди"],
  running: ["running", "анализ"],
  done: ["done", "готово"],
  error: ["error", "ошибка"],
};

export default function Home() {
  const [mode, setMode] = useState<Mode>("closeup");
  const [jobId, setJobId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) {
      setEditing(false);
      setVOverride(null);
      setFileName(f.name);
      analyze.mutate({ file: f, mode }, { onSuccess: (r) => setJobId(r.job_id) });
    }
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const busy = analyze.isPending || (!!job.data && ["queued", "running"].includes(job.data.status));
  const badgeKey = analyze.isPending ? "running" : job.data?.status;

  return (
    <main className="app-main">
      <header className="topbar">
        <div className="logo"><IconHex className="ico-md" /></div>
        <div className="grow">
          <div className="crumb">DATA FORCE · классификация руд</div>
          <h1>Скажи мне кто твой шлиф</h1>
        </div>
        <ThemeToggle />
      </header>

      {!started ? (
        <Welcome mode={mode} onMode={setMode} onFile={onFile} />
      ) : (
        <>
          <div className="mode-bar">
            <div className="seg" role="group" aria-label="Режим анализа">
              {MODES.map(([m, label]) => (
                <button key={m} type="button" className={mode === m ? "active" : ""}
                  aria-pressed={mode === m} onClick={() => setMode(m)}>{label}</button>
              ))}
            </div>
            <label className="btn primary">
              <IconUpload /> Новый шлиф
              <input type="file" accept="image/*" onChange={onFile} style={{ display: "none" }} />
            </label>
            {fileName ? <span className="chip muted">{fileName}</span> : null}
            {badgeKey && STATUS[badgeKey] ? (
              <span className={`status-badge ${STATUS[badgeKey][0]}`}>
                <span className="bd" />{STATUS[badgeKey][1]}
              </span>
            ) : null}
          </div>

          <div className="grid2">
            <div>
              {editing && result?.size ? (
                <Corrector jobId={jobId!} size={result.size} onVerdict={(v) => setVOverride(v)} />
              ) : (
                <div className="stage">
                  {jobId && shown ? (
                    <img src={shown.overlay_url ?? imageUrl(jobId)} alt="шлиф" />
                  ) : (
                    <div className="stage-empty">
                      <IconScan className="ico-lg" />
                      <div className="hint">{busy ? "Анализ снимка…" : "Загрузите снимок шлифа"}</div>
                      {!busy ? <div className="sub">OM · отражённый свет · JPG / PNG</div> : null}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="side">
              {shown ? (
                <VerdictPanel result={shown} />
              ) : job.data?.status === "error" ? (
                <div className="note danger">
                  <span className="ico"><IconAlert className="ico-md" /></span>
                  <span>Ошибка: {job.data.message}</span>
                </div>
              ) : null}
              {shown && shown.mode === "closeup" && !editing ? (
                <button className="btn ghost" onClick={() => setEditing(true)}>
                  <IconEdit /> Доработать маски
                </button>
              ) : null}
              {shown && jobId ? (
                <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
                  ⬇ Скачать протокол (PDF)
                </a>
              ) : null}
            </div>
          </div>
        </>
      )}
    </main>
  );
}
