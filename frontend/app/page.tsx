"use client";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { reportUrl } from "@/lib/api/client";
import type { Verdict } from "@/lib/api/types";
import { buildJobQuery, parseJobParams } from "@/lib/jobUrl";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { Welcome } from "@/components/Welcome";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { IconHex, IconAlert, IconDownload } from "@/components/icons";

const STATUS: Record<string, [string, string]> = {
  queued: ["queued", "в очереди"], running: ["running", "анализ"],
  done: ["done", "готово"], error: ["error", "ошибка"],
};

function Home() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [restored] = useState(() => parseJobParams(searchParams));

  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => restored.jobId);
  const [startedAt, setStartedAt] = useState<number | null>(() => restored.startedAt ?? (restored.jobId ? Date.now() : null));
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  // Держим ссылку в актуальном состоянии: перезагрузка страницы должна вернуть к тому же
  // анализу, а не к пустому экрану загрузки.
  useEffect(() => {
    const qs = buildJobQuery(jobId, startedAt);
    if (qs === searchParams.toString()) return;
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [jobId, startedAt, pathname, router, searchParams]);

  // Job, восстановленный из ссылки, мог устареть или быть удалён на сервере — откатываемся
  // на экран загрузки вместо того, чтобы зависнуть на бесконечном "Анализ снимка…".
  useEffect(() => {
    if (job.isError && jobId) {
      setJobId(null);
      setStartedAt(null);
      setVOverride(null);
    }
  }, [job.isError, jobId]);

  function runAnalyze(f: File) {
    setFile(f);
    setVOverride(null);
    setJobId(null);
    setStartedAt(Date.now());
    analyze.mutate({ file: f }, { onSuccess: (r) => setJobId(r.job_id) });
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const badgeKey = analyze.isPending ? "running" : job.data?.status;

  const infoNode = (
    <>
      <div className="card">
        <div className="side-h">Образец<span className="ann">{shown?.mode === "panorama" ? "панорама" : "крупный план"}</span></div>
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
        <Welcome onFile={runAnalyze} />
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
        {badgeKey && STATUS[badgeKey] ? (
          <span className={`status-badge ${STATUS[badgeKey][0]}`}><span className="bd" />{STATUS[badgeKey][1]}</span>
        ) : null}
        <ThemeToggle />
      </header>

      {result && result.size ? (
        <Corrector jobId={jobId!} size={result.size} info={infoNode} onVerdict={setVOverride} />
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
                <AnalysisProgress
                  job={job.data}
                  startedAt={startedAt ?? Date.now()}
                  fallback={jobId ? "сегментация фаз" : "загрузка файла на сервер"}
                />
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={null}>
      <Home />
    </Suspense>
  );
}
