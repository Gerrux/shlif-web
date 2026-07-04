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
  const conf = m.confidence ?? 0;
  const normal = (m.normal_share ?? 0) * 100;
  const fine = (m.fine_share ?? 0) * 100;
  return (
    <div className="verdict">
      <div className="vh">
        <div className="eye">Фазовый состав · правило</div>
        <div className="cls"><span className={`oreclass ${verdict.ore_class}`}>{oreRu(verdict.ore_class)}</span></div>
      </div>
      <div className="vb">
        {ROWS.map(([label, key, col]) => (
          <div className="kv" key={key}><span className="k">{label}</span>
            <span className="v" style={{ color: col }}>{((m[key] ?? 0) * 100).toFixed(1)}%</span></div>
        ))}
        {normal + fine > 0 ? (
          <div className="stackbar" title="обычные / тонкие срастания">
            <span style={{ width: `${normal}%`, background: "var(--phase-normal)" }} />
            <span style={{ width: `${fine}%`, background: "var(--phase-fine)" }} />
          </div>
        ) : null}
      </div>
      <div className="vf">
        <span className="conf-line">
          <span className="dot" style={{ background: conf >= 0.85 ? "var(--success)" : "var(--warn)" }} />
          уверенность {conf.toFixed(2)}
        </span>
        <span>seg+rule</span>
      </div>
    </div>
  );
}
