import type { AnalyzeResult } from "@/lib/api/types";
import { SortCard } from "./SortCard";
import { PhaseBars } from "./PhaseBars";
import { oreRu } from "@/lib/ore";
export function VerdictPanel({ result }: { result: AnalyzeResult }) {
  return (
    <div>
      {result.mode === "panorama" ? (
        <div className="verdict">
          <div className="vh">
            <div className="eye">Секционный вердикт · на проверку</div>
            <div className="cls">
              <span className={`oreclass ${["ordinary", "hard", "talcose"].includes(result.verdict.ore_class) ? result.verdict.ore_class : "review"}`}>
                {oreRu(result.verdict.ore_class)}
              </span>
            </div>
          </div>
          <div className="vb">
            <div className="kv"><span className="k">Тальк-кандидаты</span><span className="v">{((result.verdict.metrics.talc_frac ?? 0) * 100).toFixed(1)}%</span></div>
            <div className="kv"><span className="k">Рудных тайлов</span><span className="v">{result.n_ore} / {result.n_tiles}</span></div>
          </div>
        </div>
      ) : (
        <>
          <SortCard sort={result.sort} />
          <PhaseBars verdict={result.verdict} />
        </>
      )}
      {result.text ? <div className="note" style={{ marginTop: 12 }}>{result.text}</div> : null}
    </div>
  );
}
