export type Mode = "closeup" | "panorama";
export type OreClass = "ordinary" | "hard" | "talcose" | "review";

export interface Verdict {
  ore_class: OreClass;
  text: string;
  metrics: Record<string, number> & { talc_frac?: number; fine_share?: number; confidence?: number };
}
export interface SortCard { classes: Record<string, number>; top: string; }
export interface AnalyzeResult {
  mode: Mode; verdict: Verdict; sort: SortCard | null; text?: string;
  size?: [number, number]; overlay_url?: string; n_ore?: number; n_tiles?: number;
}
export interface Job {
  id: string; mode: string; status: "queued" | "running" | "done" | "error";
  progress: number; message: string | null; result: AnalyzeResult | null;
}
