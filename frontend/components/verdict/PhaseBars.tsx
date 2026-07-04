import type { Verdict } from "@/lib/api/types";
import { oreRu } from "@/lib/ore";
const ROWS: [string, string, string][] = [
  ["Доля талька", "talc_frac", "var(--phase-talc-ink)"],
  ["Тонкие срастания", "fine_share", "var(--phase-fine-ink)"],
  ["Обычные срастания", "normal_share", "var(--phase-normal-ink)"],
  ["Доля сульфидов", "sulfide_frac", "var(--text)"],
];
export function PhaseBars({ verdict }: { verdict: Verdict }) {
  const m = verdict.metrics;
  return (
    <div className="verdict">
      <div className="vh"><div className="eye">Фазовый состав · правило</div>
        <div style={{ marginTop: 8 }}><span className={`oreclass ${verdict.ore_class}`}>{verdict.text ? "" : ""}{oreRu(verdict.ore_class)}</span></div></div>
      <div className="vb">
        {ROWS.map(([label, key, col]) => (
          <div className="kv" key={key}><span className="k">{label}</span>
            <span className="v" style={{ color: col }}>{((m[key] ?? 0) * 100).toFixed(1)}%</span></div>
        ))}
      </div>
      <div className="vf"><span>уверенность {(m.confidence ?? 0).toFixed(2)}</span><span>seg+rule</span></div>
    </div>
  );
}
