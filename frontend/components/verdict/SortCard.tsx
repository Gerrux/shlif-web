import type { SortCard as Sort } from "@/lib/api/types";
const RU: Record<string, string> = { ordinary: "рядовая руда", hard: "труднообогатимая руда", talcose: "оталькованная руда" };
// Цвет полосы = ведущая фаза класса руды (принцип ДС): рядовая→зелёный, труднообогатимая→красный, оталькованная→синий.
const BAR: Record<string, string> = { ordinary: "var(--phase-normal)", hard: "var(--phase-fine)", talcose: "var(--phase-talc)" };
export function SortCard({ sort }: { sort: Sort | null }) {
  if (!sort) return <div className="note">Классификатор сорта недоступен (нет models/classifier.pkl).</div>;
  const top = sort.top;
  return (
    <div className="verdict">
      <div className="vh">
        <div className="eye">Сорт руды · классификатор (RF · F1 0.84)</div>
        <div className="cls"><span className={`oreclass ${top}`}>{RU[top] ?? top}</span></div>
      </div>
      <div className="vb">
        {Object.entries(sort.classes).map(([k, v]) => (
          <div className="mrow" key={k}>
            <div className="mbar-top"><span>{RU[k] ?? k}</span><span className="pct">{Math.round(v * 100)}%</span></div>
            <div className="mbar"><i style={{ width: `${Math.min(v * 100, 100)}%`, background: BAR[k] ?? "var(--muted)" }} /></div>
          </div>
        ))}
      </div>
    </div>
  );
}
