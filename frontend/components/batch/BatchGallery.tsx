"use client";
import type { Job } from "@/lib/api/types";
import { jobsToCsv, downloadCsv } from "@/lib/csv";
import { STATUS_LABELS } from "@/lib/statusLabels";
import { imageUrl } from "@/lib/api/client";
import { IconDownload, IconUpload } from "@/components/icons";

export function BatchGallery({
  batchId, jobs, onOpen, onNewAnalysis,
}: {
  batchId: string;
  jobs: Job[];
  onOpen: (jobId: string) => void;
  onNewAnalysis: () => void;
}) {
  const doneCount = jobs.filter((j) => j.status === "done").length;

  return (
    <div className="batch-gallery">
      <div className="batch-head">
        <div>
          <div className="side-h">Партия снимков</div>
          <div className="sub">{doneCount} / {jobs.length} готово</div>
        </div>
        <div className="grow" />
        <button
          type="button"
          className="btn ghost sm"
          disabled={doneCount === 0}
          onClick={() => downloadCsv(`shlif-batch-${batchId}.csv`, jobsToCsv(jobs))}
        >
          <IconDownload className="ico-sm" /> Скачать CSV (партия)
        </button>
        <button type="button" className="btn ghost sm" onClick={onNewAnalysis}>
          <IconUpload className="ico-sm" /> Новый анализ
        </button>
      </div>
      {jobs.length === 0 ? (
        <div className="stage-empty"><div className="hint">Загрузка файлов…</div></div>
      ) : (
        <div className="batch-grid">
          {jobs.map((job) => {
            const badge = STATUS_LABELS[job.status];
            return (
              <button key={job.id} type="button" className="batch-card" onClick={() => onOpen(job.id)}>
                {job.status === "done" ? (
                  <img className="batch-card-thumb" src={imageUrl(job.id)} alt={job.filename ?? job.id} />
                ) : (
                  <div className="batch-card-thumb placeholder" aria-hidden="true" />
                )}
                <span className="batch-card-name">{job.filename ?? job.id}</span>
                {badge ? (
                  <span className={`status-badge ${badge[0]}`}><span className="bd" />{badge[1]}</span>
                ) : null}
                {job.status === "error" ? <span className="batch-card-err">{job.message}</span> : null}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
