"use client";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { analyze as analyzeApi } from "@/lib/api/client";
import { useAnalyze, useJob, useBatchJobs } from "@/lib/api/hooks";
import { reportUrl } from "@/lib/api/client";
import type { Verdict } from "@/lib/api/types";
import { buildJobQuery, parseJobParams } from "@/lib/jobUrl";
import { parseBatchParams, buildBatchQuery } from "@/lib/batchUrl";
import { jobsToCsv, downloadCsv } from "@/lib/csv";
import { STATUS_LABELS } from "@/lib/statusLabels";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { Welcome } from "@/components/Welcome";
import { BatchGallery } from "@/components/batch/BatchGallery";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { IconAlert, IconDownload, IconUpload } from "@/components/icons";
import { PanoramaZoomModal } from "@/components/PanoramaZoomModal";

function Home() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [restoredJob] = useState(() => parseJobParams(searchParams));
  const [restoredBatch] = useState(() => parseBatchParams(searchParams));

  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => restoredJob.jobId);
  const [startedAt, setStartedAt] = useState<number | null>(() => restoredJob.startedAt ?? (restoredJob.jobId ? Date.now() : null));
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const [batchId, setBatchId] = useState<string | null>(() => restoredBatch.batchId);
  const [uploadFailures, setUploadFailures] = useState<{ filename: string; error: string }[]>([]);
  const analyze = useAnalyze();
  const job = useJob(jobId);
  const batchJobs = useBatchJobs(batchId);

  // Держим ссылку в актуальном состоянии: перезагрузка страницы должна вернуть к тому же
  // анализу (или к той же партии), а не к пустому экрану загрузки.
  useEffect(() => {
    const qs = batchId ? buildBatchQuery(batchId, jobId) : buildJobQuery(jobId, startedAt);
    if (qs === searchParams.toString()) return;
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [jobId, startedAt, batchId, pathname, router, searchParams]);

  // Job, восстановленный из ссылки, мог устареть или быть удалён на сервере — откатываемся
  // на партию (если она есть) или на экран загрузки, а не зависаем на "Анализ снимка…".
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

  function startBatch(files: File[]) {
    const id = crypto.randomUUID();
    setFile(null);
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
    setUploadFailures([]);
    setBatchId(id);
    Promise.allSettled(files.map((f) => analyzeApi(f, id))).then((results) => {
      const failed = results
        .map((res, i) => (res.status === "rejected" ? { filename: files[i].name, error: String(res.reason) } : null))
        .filter((x): x is { filename: string; error: string } => x !== null);
      if (failed.length) setUploadFailures((prev) => [...prev, ...failed]);
    });
  }

  function handleFiles(files: File[]) {
    if (files.length <= 1) {
      if (files[0]) runAnalyze(files[0]);
      return;
    }
    startBatch(files);
  }

  function openBatchItem(id: string) {
    setJobId(id);
    setStartedAt(Date.now());
    setVOverride(null);
  }

  function backToBatch() {
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
  }

  function resetToUpload() {
    analyze.reset();
    setFile(null);
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
    setBatchId(null);
    setUploadFailures([]);
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const badgeKey = analyze.isPending ? "running" : job.data?.status;
  const activeBatchJob = batchId && jobId ? batchJobs.data?.find((j) => j.id === jobId) ?? null : null;
  const activeFileName = file?.name ?? activeBatchJob?.filename ?? null;
  const inGallery = !!batchId && !jobId;

  const infoNode = (
    <>
      <div className="card">
        <div className="side-h">Образец<span className="ann">{shown?.mode === "panorama" ? "панорама" : "крупный план"}</span></div>
        <div className="side-b"><div className="meta-rows">
          <div className="kv"><span className="k">Файл</span><span className="v">{activeFileName ?? "—"}</span></div>
          {shown?.size ? <div className="kv"><span className="k">Размер</span><span className="v">{shown.size[0]}×{shown.size[1]}</span></div> : null}
        </div></div>
      </div>
      {shown ? <VerdictPanel result={shown} /> : null}
      {shown && jobId ? (
        <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
          <IconDownload /> Скачать протокол (PDF)
        </a>
      ) : null}
      {shown && jobId ? (
        <button
          type="button"
          className="btn ghost"
          onClick={() => downloadCsv(`shlif-${jobId}.csv`, jobsToCsv([job.data!]))}
        >
          <IconDownload /> Скачать CSV
        </button>
      ) : null}
      {shown?.mode === "panorama" && jobId ? <PanoramaZoomModal jobId={jobId} /> : null}
    </>
  );

  if (!batchId && !started) {
    return (
      <>
        <Welcome onFiles={handleFiles} />
        <div className="theme-float"><ThemeToggle /></div>
      </>
    );
  }

  if (inGallery) {
    return (
      <main className="app-main">
        <header className="topbar" style={{ flexWrap: "wrap" }}>
          <div className="logo" aria-hidden="true">🚀</div>
          <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
          <div className="grow" />
          <ThemeToggle />
        </header>
        <BatchGallery
          batchId={batchId!}
          jobs={batchJobs.data ?? []}
          onOpen={openBatchItem}
          onNewAnalysis={resetToUpload}
        />
        {uploadFailures.length ? (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="card-b">
              {uploadFailures.map((f, i) => (
                <div key={i} className="kv"><span className="k">{f.filename}</span><span className="v">не загружен: {f.error}</span></div>
              ))}
            </div>
          </div>
        ) : null}
      </main>
    );
  }

  return (
    <main className="app-main">
      <header className="topbar" style={{ flexWrap: "wrap" }}>
        <div className="logo" aria-hidden="true">🚀</div>
        <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
        <div className="grow" />
        {badgeKey && STATUS_LABELS[badgeKey] ? (
          <span className={`status-badge ${STATUS_LABELS[badgeKey][0]}`}><span className="bd" />{STATUS_LABELS[badgeKey][1]}</span>
        ) : null}
        {batchId ? (
          <button type="button" className="btn ghost sm" onClick={backToBatch}>← к партии</button>
        ) : null}
        <button type="button" className="btn ghost sm" onClick={resetToUpload}>
          <IconUpload className="ico-sm" /> Новый анализ
        </button>
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
