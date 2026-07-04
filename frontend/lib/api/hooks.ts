import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob } from "./client";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File }) => analyze(v.file) });
}
export function useJob(id: string | null) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => getJob(id as string),
    enabled: !!id,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "done" || s === "error" ? false : 800;
    },
  });
}
