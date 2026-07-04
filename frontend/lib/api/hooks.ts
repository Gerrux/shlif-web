import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob } from "./client";
import type { Mode } from "./types";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File; mode: Mode }) => analyze(v.file, v.mode) });
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
