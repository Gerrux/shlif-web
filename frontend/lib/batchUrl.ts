// Кодирование/декодирование batch_id в query-параметр адресной строки — по аналогии с
// jobUrl.ts, чтобы перезагрузка страницы во время партийной обработки возвращала в галерею
// партии (а не на пустой экран загрузки), а не только к одиночному job.

export interface BatchParams {
  batchId: string | null;
}

export function parseBatchParams(sp: URLSearchParams): BatchParams {
  return { batchId: sp.get("batch") };
}

export function buildBatchQuery(batchId: string | null, jobId: string | null): string {
  if (!batchId) return "";
  const params = new URLSearchParams();
  params.set("batch", batchId);
  if (jobId) params.set("job", jobId);
  return params.toString();
}
