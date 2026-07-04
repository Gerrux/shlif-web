// Кодирование/декодирование состояния текущего анализа (job id + время старта) в query-параметры
// адресной строки, чтобы перезагрузка страницы возвращала на тот же анализ, а не на экран загрузки.

export interface JobParams {
  jobId: string | null;
  startedAt: number | null;
}

export function parseJobParams(sp: URLSearchParams): JobParams {
  const jobId = sp.get("job");
  if (!jobId) return { jobId: null, startedAt: null };
  const startedRaw = sp.get("started");
  const started = startedRaw ? Number(startedRaw) : NaN;
  return { jobId, startedAt: Number.isFinite(started) ? started : null };
}

export function buildJobQuery(jobId: string | null, startedAt: number | null): string {
  if (!jobId) return "";
  const params = new URLSearchParams();
  params.set("job", jobId);
  if (startedAt != null) params.set("started", String(startedAt));
  return params.toString();
}
