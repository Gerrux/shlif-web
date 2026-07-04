"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { imageUrl, reportUrl } from "@/lib/api/client";
import type { Mode, Verdict } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { PanoramaWorkspace } from "@/components/PanoramaWorkspace";
import { Welcome } from "@/components/Welcome";
import { ThemeToggle } from "@/components/ThemeToggle";
import { IconHex, IconAlert, IconDownload } from "@/components/icons";

const MODES: [Mode, string][] = [["closeup", "Крупный план"], ["panorama", "Панорама"]];
const STATUS: Record<string, [string, string]> = {
  queued: ["queued", "в очереди"], running: ["running", "анализ"],
  done: ["done", "готово"], error: ["error", "ошибка"],
};

export default function Home() {
  const [mode, setMode] = useState<Mode>("closeup");
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function runAnalyze(f: File, m: Mode) {
    setFile(f);
    setVOverride(null);
    setJobId(null);
    analyze.mutate({ file: f, mode: m }, { onSuccess: (r) => setJobId(r.job_id) });
  }
  function onMode(m: Mode) {
    if (m === mode) return;
    setMode(m);
    if (file) runAnalyze(file, m);
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const badgeKey = analyze.isPending ? "running" : job.data?.status;

  const infoNode = (
    <>
      <div className="card">
        <div className="side-h">Образец<span className="ann">{mode === "closeup" ? "крупный план" : "панорама"}</span></div>
        <div className="side-b"><div className="meta-rows">
          <div className="kv"><span className="k">Файл</span><span className="v">{file?.name ?? "—"}</span></div>
          {shown?.size ? <div className="kv"><span className="k">Размер</span><span className="v">{shown.size[0]}×{shown.size[1]}</span></div> : null}
        </div></div>
      </div>
      {shown ? <VerdictPanel result={shown} /> : null}
      {shown && jobId ? (
        <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
          <IconDownload /> Скачать протокол (PDF)
        </a>
      ) : null}
    </>
  );

  if (!started) {
    return (
      <>
        <Welcome onFile={(f) => runAnalyze(f, mode)} />
        <div className="theme-float"><ThemeToggle /></div>
      </>
    );
  }

  return (
    <main className="app-main">
      <header className="topbar" style={{ flexWrap: "wrap" }}>
        <div className="logo"><IconHex className="ico-md" /></div>
        <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
        <div className="grow" />
        <div className="seg" role="group" aria-label="Режим анализа">
          {MODES.map(([m, label]) => (
            <button key={m} type="button" className={mode === m ? "active" : ""}
              aria-pressed={mode === m} onClick={() => onMode(m)}>{label}</button>
          ))}
        </div>
        {badgeKey && STATUS[badgeKey] ? (
          <span className={`status-badge ${STATUS[badgeKey][0]}`}><span className="bd" />{STATUS[badgeKey][1]}</span>
        ) : null}
        <ThemeToggle />
      </header>

      {result && mode === "closeup" && result.size ? (
        <Corrector jobId={jobId!} size={result.size} info={infoNode} onVerdict={setVOverride} />
      ) : result && mode === "panorama" ? (
        <PanoramaWorkspace src={shown!.overlay_url ?? imageUrl(jobId!)} info={infoNode} />
      ) : (
        <div className="workspace">
          <aside className="ws-side">{infoNode}</aside>
          <div className="ws-view">
            <div className="zoom-vp">
              {job.data?.status === "error" ? (
                <div className="stage-empty">
                  <IconAlert className="ico-lg" />
                  <div className="hint">Ошибка анализа</div>
                  <div className="sub">{job.data.message ?? "неизвестная ошибка"}</div>
                </div>
              ) : (
                <div className="stage-empty">
                  <div className="hint">Анализ снимка…</div>
                  <div className="sub">{mode === "panorama" ? "панорама · сегментация тайлов" : "крупный план · сегментация фаз"}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
