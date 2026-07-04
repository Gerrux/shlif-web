import type { AnalyzeResult } from "@/lib/api/types";
import { SortCard } from "./SortCard";
import { PhaseBars } from "./PhaseBars";
export function VerdictPanel({ result }: { result: AnalyzeResult }) {
  return (
    <div>
      <div className="ptag-row">
        <span className="ptag normal"><i className="d" />Обычные срастания</span>
        <span className="ptag fine"><i className="d" />Тонкие срастания</span>
        <span className="ptag talc"><i className="d" />Тальк</span>
      </div>
      <SortCard sort={result.sort} />
      <PhaseBars verdict={result.verdict} />
      {result.text ? <div className="note" style={{ marginTop: 12 }}>{result.text}</div> : null}
    </div>
  );
}
