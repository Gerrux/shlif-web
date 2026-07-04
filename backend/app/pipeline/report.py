"""PDF protocol export — a passport-style report for a finished analysis.

Cyrillic is rendered with the vendored DejaVuSans TTF (``app/assets/fonts``), so
the report renders identically in any container without system fonts. The layout
follows the «Шлиф» framing: an automatic result the expert then confirms — every
report is stamped «на проверку экспертом».
"""

from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

_FONT = "DejaVuSans"
_FONT_B = "DejaVuSans-Bold"
_FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

ORE_CLASS_RU = {"ordinary": "рядовая", "hard": "труднообогатимая",
                "talcose": "оталькованная", "review": "на проверку"}

# metric key -> (RU label, formatter). Only keys present in the result are shown.
_METRIC_ROWS = [
    ("sulfide_frac", "Доля сульфидов", "pct"),
    ("magnetite_frac", "Доля магнетита", "pct"),
    ("matrix_frac", "Доля матрицы", "pct"),
    ("talc_frac", "Тальк (сегментатор)", "pct"),
    ("talc_share_est", "Тальк (оценка доли)", "pct"),
    ("fine_share", "Тонкие срастания", "pct"),
    ("confidence", "Уверенность", "num"),
    ("undetermined_fraction", "Неопределённость", "pct"),
]


def _ensure_fonts() -> None:
    registered = set(pdfmetrics.getRegisteredFontNames())
    for name, fname in ((_FONT, "DejaVuSans.ttf"), (_FONT_B, "DejaVuSans-Bold.ttf")):
        if name in registered:
            continue
        candidates = [_FONT_DIR / fname, Path("/usr/share/fonts/truetype/dejavu") / fname]
        path = next((p for p in candidates if p.exists()), None)
        if path is not None:
            pdfmetrics.registerFont(TTFont(name, str(path)))


def _fmt(kind: str, v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{100 * v:.1f} %" if kind == "pct" else f"{v:.2f}"


def build_report_pdf(jid: str, mode: str, result: dict, image_path: Path | None) -> bytes:
    """Render a one-page PDF protocol for a finished job → raw PDF bytes."""
    _ensure_fonts()
    have_bold = _FONT_B in set(pdfmetrics.getRegisteredFontNames())
    bold = _FONT_B if have_bold else _FONT

    verdict = result.get("verdict", {}) or {}
    metrics = verdict.get("metrics", {}) or {}
    ore = verdict.get("ore_class", "review")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontName=bold, fontSize=17)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold, fontSize=12)
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=_FONT, fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.grey)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Протокол {jid}",
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    flow = [
        Paragraph("Протокол исследования аншлифа", h1),
        Paragraph(f"Режим: {'панорама' if mode == 'panorama' else 'крупный план'} · "
                  f"идентификатор: {jid}", small),
        Spacer(1, 6 * mm),
    ]

    if image_path is not None and Path(image_path).exists():
        try:
            img = RLImage(str(image_path))
            max_w = doc.width
            scale = min(1.0, max_w / float(img.imageWidth))
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            flow += [img, Spacer(1, 6 * mm)]
        except Exception:
            pass

    flow += [
        Paragraph(f"Сорт руды: <b>{ORE_CLASS_RU.get(ore, ore)}</b>", h2),
    ]
    if verdict.get("text"):
        flow.append(Paragraph(str(verdict["text"]), body))
    flow.append(Spacer(1, 4 * mm))

    rows = [["Показатель", "Значение"]]
    for key, label, kind in _METRIC_ROWS:
        if key in metrics:
            rows.append([label, _fmt(kind, metrics[key])])
    if len(rows) > 1:
        table = Table(rows, colWidths=[doc.width * 0.62, doc.width * 0.38])
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), _FONT),
            ("FONTNAME", (0, 0), (-1, 0), bold),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2f36")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f1ec")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfcabb")),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow += [table, Spacer(1, 5 * mm)]

    zones = result.get("low_conf_zones") or []
    if zones:
        flow.append(Paragraph("Спорные зоны (проверить вручную)", h2))
        for z in zones[:8]:
            flow.append(Paragraph(
                f"• область {z.get('area', '?')} px — спор фаз "
                f"{z.get('phase_a', '?')} / {z.get('phase_b', '?')}", body))
        flow.append(Spacer(1, 4 * mm))

    flow.append(Paragraph(
        "Автоматический результат. Требует подтверждения экспертом (на проверку). "
        "Тальк по оптической микроскопии не квантуется точно — доля талька приведена "
        "как оценка.", small))

    doc.build(flow)
    return buf.getvalue()
