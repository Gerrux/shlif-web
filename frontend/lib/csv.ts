import type { Job } from "./api/types";

const METRIC_KEYS = [
  "sulfide_frac", "magnetite_frac", "matrix_frac", "talc_frac",
  "talc_share_est", "fine_share", "confidence", "undetermined_fraction",
] as const;

const HEADERS = ["job_id", "filename", "mode", "status", "ore_class", ...METRIC_KEYS, "created_at"];

function csvEscape(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function jobRow(job: Job): string {
  const metrics = job.result?.verdict?.metrics ?? {};
  const cells = [
    job.id,
    job.filename ?? "",
    job.mode,
    job.status,
    job.result?.verdict?.ore_class ?? "",
    ...METRIC_KEYS.map((k) => (metrics[k] != null ? String(metrics[k]) : "")),
    job.created_at ?? "",
  ];
  return cells.map(csvEscape).join(",");
}

export function jobsToCsv(jobs: Job[]): string {
  return [HEADERS.join(","), ...jobs.map(jobRow)].join("\r\n");
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
