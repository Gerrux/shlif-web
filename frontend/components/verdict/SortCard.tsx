import type { SortCard as Sort } from "@/lib/api/types";
const RU: Record<string, string> = { ordinary: "рядовая руда", hard: "труднообогатимая руда", talcose: "оталькованная руда" };
const BAR: Record<string, string> = { ordinary: "rgb(80,190,120)", hard: "rgb(225,85,80)", talcose: "rgb(95,140,235)" };
export function SortCard({ sort }: { sort: Sort | null }) {
  if (!sort) return <div className="note">Классификатор сорта недоступен (нет models/classifier.pkl).</div>;
  const top = sort.top;
  return (
    <div className="verdict" style={{ marginBottom: 14 }}>
      <div className="vh"><div className="eye">Сорт руды · классификатор (RF · F1 0.84)</div>
        <div style={{ marginTop: 8 }}><span className={`oreclass ${top}`}>{RU[top]}</span></div></div>
      <div className="vb">
        {Object.entries(sort.classes).map(([k, v]) => (
          <div className="mrow" key={k}>
            <div className="top"><span>{RU[k]}</span><span className="pct">{Math.round(v * 100)}%</span></div>
            <div className="mbar"><i style={{ width: `${Math.min(v * 100, 100)}%`, background: BAR[k] }} /></div>
          </div>
        ))}
      </div>
    </div>
  );
}
