import type { Job, Verdict } from "./types";

const base = "";

export async function analyze(file: File): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("image", file);
  const r = await fetch(`${base}/api/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
  return r.json();
}
export async function getJob(id: string): Promise<Job> {
  const r = await fetch(`${base}/api/jobs/${id}`);
  if (!r.ok) throw new Error(`job failed: ${r.status}`);
  return r.json();
}
export const maskUrl = (id: string, layer: "phases" | "talc" | "intergrowth") => `${base}/api/masks/${id}/${layer}.png`;
export const mapUrl = (id: string, name: "superpixels" | "darkness" | "confidence") => `${base}/api/maps/${id}/${name}.png`;
export const imageUrl = (id: string) => `${base}/api/images/${id}.jpg`;
export const reportUrl = (id: string) => `${base}/api/report/${id}.pdf`;

export async function saveMasks(id: string, phases: Blob, talc: Blob): Promise<Verdict> {
  const fd = new FormData();
  fd.append("phases", phases, "phases.png");
  fd.append("talc", talc, "talc.png");
  const r = await fetch(`${base}/api/masks/${id}`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`save failed: ${r.status}`);
  return r.json();
}
