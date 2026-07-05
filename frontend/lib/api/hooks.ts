import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob, listJobsByBatch } from "./client";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File }) => analyze(v.file) });
}
export function useJob(id: string | null) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => getJob(id as string),
    enabled: !!id,
    retry: 1,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      if (s === "done" || s === "error") return false;
      return q.state.status === "error" ? false : 800;
    },
  });
}
export function useBatchJobs(batchId: string | null, uploadsSettled: boolean) {
  return useQuery({
    queryKey: ["batch", batchId],
    queryFn: () => listJobsByBatch(batchId as string),
    enabled: !!batchId,
    refetchInterval: (q) => {
      const jobs = q.state.data;
      if (!jobs) return 800;
      if (jobs.length === 0) {
        // Empty could mean "still uploading" (keep polling) or "every upload in
        // this batch already failed, no job will ever land" (stop) — only the
        // caller knows which, since a failed POST never reaches the job store.
        return uploadsSettled ? false : 800;
      }
      const pending = jobs.some((j) => j.status === "queued" || j.status === "running");
      return pending ? 800 : false;
    },
  });
}
