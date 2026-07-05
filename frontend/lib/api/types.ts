export type Mode = "closeup" | "panorama";
export type OreClass = "ordinary" | "hard" | "talcose" | "review";

export interface Verdict {
  ore_class: OreClass;
  text: string;
  metrics: Record<string, number> & {
    talc_frac?: number; talc_share_est?: number; fine_share?: number;
    confidence?: number; undetermined_fraction?: number;
  };
}
export interface SortCard { classes: Record<string, number>; top: string; }
export interface LowConfZone { area: number; phase_a: string; phase_b: string; bbox: number[]; }
export interface AnalyzeResult {
  mode: Mode; verdict: Verdict; sort: SortCard | null; text?: string;
  size?: [number, number]; n_ore?: number; n_tiles?: number;
  low_conf_zones?: LowConfZone[];
}
export interface Job {
  id: string; mode: string; status: "queued" | "running" | "done" | "error";
  progress: number; message: string | null; result: AnalyzeResult | null;
  batch_id: string | null; filename: string | null; created_at: string | null;
}
