export function oreRu(c: string): string {
  return { ordinary: "рядовая руда", hard: "труднообогатимая руда", talcose: "оталькованная руда", review: "на проверку" }[c] ?? c;
}
