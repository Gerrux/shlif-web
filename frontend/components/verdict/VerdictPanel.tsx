import type { AnalyzeResult } from "@/lib/api/types";
import { SortCard } from "./SortCard";
import { PhaseBars } from "./PhaseBars";
export function VerdictPanel({ result }: { result: AnalyzeResult }) {
  return (<div><SortCard sort={result.sort} /><PhaseBars verdict={result.verdict} />
    {result.text ? <div className="note" style={{ marginTop: 12 }}>{result.text}</div> : null}</div>);
}
