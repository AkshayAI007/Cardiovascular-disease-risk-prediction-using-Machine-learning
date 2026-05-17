"""
Publication-quality PDF from TECHNICAL_DOCUMENTATION.md using ReportLab Platypus.
Run: python generate_pdf.py
"""
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
pt = 1
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import HexColor

# ── Colour palette ────────────────────────────────────────────────────────────
INK       = HexColor("#1a1714")
CRIMSON   = HexColor("#B8302A")
CREAM     = HexColor("#F4EFE6")
CREAM2    = HexColor("#EDE6D9")
RULE      = HexColor("#D8D0BF")
MUTED     = HexColor("#8A8278")
CODE_BG   = HexColor("#F0EDE6")
ROW_ALT   = HexColor("#F7F4EF")
WHITE     = colors.white

PDF_FILE  = "Cardiovascular_Risk_Prediction_Technical_Report.pdf"
MD_FILE   = "TECHNICAL_DOCUMENTATION.md"

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

STYLES = {
    "h1": S("h1",
        fontName="Times-Bold", fontSize=18, leading=22,
        textColor=INK, spaceBefore=24, spaceAfter=10,
        borderPadding=(0,0,4,0),
    ),
    "h2": S("h2",
        fontName="Times-Bold", fontSize=14, leading=18,
        textColor=INK, spaceBefore=18, spaceAfter=8,
    ),
    "h3": S("h3",
        fontName="Helvetica-Bold", fontSize=10, leading=14,
        textColor=CRIMSON, spaceBefore=14, spaceAfter=6,
        textTransform="uppercase",
    ),
    "h4": S("h4",
        fontName="Helvetica-Bold", fontSize=10, leading=13,
        textColor=INK, spaceBefore=10, spaceAfter=5,
    ),
    "body": S("body",
        fontName="Times-Roman", fontSize=10, leading=15,
        textColor=INK, spaceBefore=0, spaceAfter=7,
        alignment=TA_JUSTIFY,
    ),
    "bullet": S("bullet",
        fontName="Times-Roman", fontSize=10, leading=14,
        textColor=INK, spaceBefore=2, spaceAfter=2,
        leftIndent=16, firstLineIndent=0,
        bulletIndent=4,
    ),
    "code": S("code",
        fontName="Courier", fontSize=8, leading=11,
        textColor=INK, spaceBefore=6, spaceAfter=6,
        backColor=CODE_BG,
        leftIndent=10, rightIndent=10,
        borderPadding=6,
    ),
    "caption": S("caption",
        fontName="Helvetica", fontSize=8, leading=11,
        textColor=MUTED, alignment=TA_CENTER,
        spaceBefore=2, spaceAfter=8,
    ),
    "cover_kicker": S("cover_kicker",
        fontName="Helvetica", fontSize=8, leading=12,
        textColor=CRIMSON, alignment=TA_CENTER,
        spaceBefore=0, spaceAfter=6,
    ),
    "cover_title": S("cover_title",
        fontName="Times-Bold", fontSize=30, leading=36,
        textColor=INK, alignment=TA_CENTER,
        spaceBefore=0, spaceAfter=6,
    ),
    "cover_sub": S("cover_sub",
        fontName="Times-Italic", fontSize=14, leading=18,
        textColor=HexColor("#444444"), alignment=TA_CENTER,
        spaceBefore=0, spaceAfter=30,
    ),
    "cover_disclaimer": S("cover_disclaimer",
        fontName="Helvetica", fontSize=8, leading=11,
        textColor=CRIMSON, alignment=TA_CENTER,
        spaceBefore=30, spaceAfter=0,
    ),
    "toc_title": S("toc_title",
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=CRIMSON, spaceBefore=0, spaceAfter=8,
    ),
    "toc_item": S("toc_item",
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=INK, spaceBefore=1, spaceAfter=1,
        leftIndent=0,
    ),
    "toc_sub": S("toc_sub",
        fontName="Helvetica", fontSize=8.5, leading=12,
        textColor=HexColor("#555"), spaceBefore=1, spaceAfter=1,
        leftIndent=14,
    ),
    "footer_note": S("footer_note",
        fontName="Times-Italic", fontSize=9, leading=13,
        textColor=MUTED, alignment=TA_CENTER,
        spaceBefore=10, spaceAfter=0,
    ),
}

# ── Header / footer canvas callbacks ─────────────────────────────────────────
def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page == 1:
        canvas.restoreState()
        return

    # header rule + title
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(2.4*cm, h - 1.8*cm, w - 2.4*cm, h - 1.8*cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.4*cm, h - 1.6*cm,
        "Cardiovascular Disease Risk Prediction — Technical Report")

    # footer rule + page number
    canvas.line(2.4*cm, 1.8*cm, w - 2.4*cm, 1.8*cm)
    canvas.drawString(2.4*cm, 1.3*cm,
        "Research Instrument · Not for Clinical Use")
    canvas.drawRightString(w - 2.4*cm, 1.3*cm,
        f"Page {doc.page}")

    canvas.restoreState()


# ── Cover page builder ────────────────────────────────────────────────────────
def build_cover():
    story = []
    story.append(Spacer(1, 3.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=CRIMSON))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("TECHNICAL REPORT · RESEARCH PUBLICATION", STYLES["cover_kicker"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Cardiovascular Disease Risk<br/>Prediction Using Machine Learning",
        STYLES["cover_title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "10-Year Coronary Heart Disease Risk Estimation via<br/>"
        "Calibrated Machine Learning on the Framingham Cohort",
        STYLES["cover_sub"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="60%", thickness=0.5, color=RULE, hAlign="CENTER"))
    story.append(Spacer(1, 0.5*cm))

    meta = [
        ["Author",       "Akshay Bawaliwale"],
        ["Version",      "1.0"],
        ["Date",         "May 2026"],
        ["Dataset",      "Framingham Heart Study · 3,390 patients"],
        ["Best Model",   "Logistic Regression (Platt-calibrated)"],
        ["ROC-AUC",      "0.714  ·  Sensitivity 72.6 %  ·  Specificity 58.0 %"],
        ["Repository",   "github.com/AkshayAI007/Cardiovascular-disease-risk-\nprediction-using-Machine-learning"],
    ]
    meta_style = TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (0,0), (0,-1), MUTED),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Oblique"),
        ("FONTSIZE",    (0,0), (0,-1), 8),
        ("TEXTCOLOR",   (1,0), (1,-1), INK),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        ("LINEBELOW",   (0,0), (-1,-2), 0.3, RULE),
        ("ALIGN",       (0,0), (0,-1), "RIGHT"),
    ])
    t = Table(meta, colWidths=[3.5*cm, 10.5*cm])
    t.setStyle(meta_style)
    story.append(t)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="60%", thickness=0.5, color=RULE, hAlign="CENTER"))
    story.append(Paragraph(
        "Research Instrument Only — Not a Substitute for Clinical Judgement",
        STYLES["cover_disclaimer"]))
    story.append(PageBreak())
    return story


# ── Markdown parser → Platypus flowables ─────────────────────────────────────
def escape(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text

def inline(text):
    """Convert inline markdown (bold, italic, code, links) to ReportLab XML."""
    text = escape(text)
    # bold-italic ***
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    # bold **
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # italic *
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # inline code `
    text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="8.5" color="#333333">\1</font>', text)
    # strip markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text

def build_table_flowable(rows):
    if not rows:
        return []
    header = rows[0]
    body   = rows[1:]
    if len(body) > 0 and all(set(r) == {'-','|',' '} for r in [''.join(r) for r in body[:1]]):
        body = body[1:]

    col_count = len(header)
    col_w = (W - 4.8*cm) / col_count

    tdata = []
    # header row
    tdata.append([Paragraph(f"<b>{inline(c)}</b>", ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE,
        leading=12, alignment=TA_LEFT)) for c in header])
    # body rows
    for i, row in enumerate(body):
        while len(row) < col_count:
            row.append("")
        tdata.append([Paragraph(inline(c), ParagraphStyle(
            "td", fontName="Times-Roman", fontSize=9, textColor=INK,
            leading=13, alignment=TA_LEFT)) for c in row])

    ts = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), INK),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",(0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0),(-1,-1), 8),
        ("LINEBELOW",  (0,0), (-1,-1), 0.4, RULE),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ])
    for i in range(1, len(tdata)):
        if i % 2 == 0:
            ts.add("BACKGROUND", (0,i), (-1,i), ROW_ALT)

    t = Table(tdata, colWidths=[col_w]*col_count, repeatRows=1)
    t.setStyle(ts)
    return [Spacer(1, 6), t, Spacer(1, 8)]


def parse_table_row(line):
    line = line.strip().strip("|")
    return [c.strip() for c in line.split("|")]

def is_separator_row(row):
    return all(re.match(r'^[-: ]+$', c) for c in row if c)

def md_to_story(md_text):
    story = []
    lines = md_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # Skip the very first H1 (title — already on cover)
        if line.startswith("# ") and i < 5:
            i += 1
            continue

        # H1
        if line.startswith("# "):
            text = line[2:].strip()
            story.append(Spacer(1, 0.3*cm))
            story.append(HRFlowable(width="100%", thickness=1.2, color=INK))
            story.append(Paragraph(inline(text), STYLES["h1"]))
            story.append(HRFlowable(width="100%", thickness=0.4, color=RULE))
            story.append(Spacer(1, 0.2*cm))
            i += 1
            continue

        # H2
        if line.startswith("## "):
            text = line[3:].strip()
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph(inline(text), STYLES["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.4, color=RULE))
            i += 1
            continue

        # H3
        if line.startswith("### "):
            text = line[4:].strip()
            story.append(Paragraph(inline(text), STYLES["h3"]))
            i += 1
            continue

        # H4
        if line.startswith("#### "):
            text = line[5:].strip()
            story.append(Paragraph(inline(text), STYLES["h4"]))
            i += 1
            continue

        # HR ---
        if re.match(r'^---+$', line.strip()):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Fenced code block ```
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # closing ```
            code_text = "\n".join(code_lines)
            # wrap in a box
            story.append(Spacer(1, 4))
            box_data = [[Paragraph(
                f'<font face="Courier" size="8">{escape(code_text)}</font>',
                ParagraphStyle("codeinner", fontName="Courier", fontSize=8,
                    leading=11, textColor=INK, leftIndent=0))]]
            box = Table(box_data, colWidths=[W - 4.8*cm])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), CODE_BG),
                ("LEFTPADDING",(0,0),(-1,-1), 10),
                ("RIGHTPADDING",(0,0),(-1,-1), 10),
                ("TOPPADDING", (0,0),(-1,-1), 8),
                ("BOTTOMPADDING",(0,0),(-1,-1), 8),
                ("LINEAFTER",  (0,0), (0,-1), 2.5, CRIMSON),
            ]))
            story.append(box)
            story.append(Spacer(1, 6))
            continue

        # Table
        if line.startswith("|"):
            table_rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = parse_table_row(lines[i])
                if not is_separator_row(row):
                    table_rows.append(row)
                i += 1
            story.extend(build_table_flowable(table_rows))
            continue

        # Bullet / unordered list
        if re.match(r'^[\-\*] ', line):
            bullets = []
            while i < len(lines) and re.match(r'^[\-\*] ', lines[i]):
                text = re.sub(r'^[\-\*] ', '', lines[i]).strip()
                bullets.append(Paragraph(
                    f"•&nbsp;&nbsp;{inline(text)}", STYLES["bullet"]))
                i += 1
            story.extend(bullets)
            story.append(Spacer(1, 4))
            continue

        # Numbered list
        if re.match(r'^\d+\. ', line):
            items = []
            while i < len(lines) and re.match(r'^\d+\. ', lines[i]):
                m = re.match(r'^(\d+)\. (.*)', lines[i])
                num, text = m.group(1), m.group(2).strip()
                items.append(Paragraph(
                    f"<b>{num}.</b>&nbsp;&nbsp;{inline(text)}", STYLES["bullet"]))
                i += 1
            story.extend(items)
            story.append(Spacer(1, 4))
            continue

        # Blank line
        if not line.strip():
            story.append(Spacer(1, 5))
            i += 1
            continue

        # Normal paragraph
        story.append(Paragraph(inline(line.strip()), STYLES["body"]))
        i += 1

    return story


# ── TOC (static) ─────────────────────────────────────────────────────────────
TOC_ENTRIES = [
    ("1. Project Overview", False),
    ("2. Dataset", False),
    ("  2.1 Feature Inventory", True),
    ("  2.2 Missing Values", True),
    ("3. System Architecture", False),
    ("4. Methodology", False),
    ("  4.1 Exploratory Data Analysis", True),
    ("  4.2 Data Preprocessing Pipeline", True),
    ("  4.3 Feature Engineering", True),
    ("  4.4 Class Imbalance Handling", True),
    ("  4.5 Model Training & Hyperparameter Tuning", True),
    ("  4.6 Stacking Ensemble", True),
    ("  4.7 Decision Threshold Optimisation", True),
    ("  4.8 Probability Calibration", True),
    ("5. Model Evaluation", False),
    ("6. Inference Pipeline", False),
    ("7. API Reference", False),
    ("8. Web Interface", False),
    ("9. Deployment", False),
    ("10. Project Structure", False),
    ("11. Reproducibility", False),
    ("12. Limitations & Future Work", False),
    ("13. References", False),
]

def build_toc():
    story = []
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("TABLE OF CONTENTS", STYLES["toc_title"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(Spacer(1, 0.2*cm))
    for label, is_sub in TOC_ENTRIES:
        style = STYLES["toc_sub"] if is_sub else STYLES["toc_item"]
        story.append(Paragraph(label.strip(), style))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(PageBreak())
    return story


# ── Build & save PDF ──────────────────────────────────────────────────────────
def main():
    with open(MD_FILE, encoding="utf-8") as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        PDF_FILE,
        pagesize=A4,
        leftMargin=2.4*cm, rightMargin=2.4*cm,
        topMargin=2.4*cm, bottomMargin=2.6*cm,
        title="Cardiovascular Disease Risk Prediction — Technical Report",
        author="Akshay Bawaliwale",
        subject="10-Year CHD Risk Prediction via Machine Learning",
        creator="Python ReportLab",
    )

    story = []
    story.extend(build_cover())
    story.extend(build_toc())
    story.extend(md_to_story(md_text))

    # Final disclaimer
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(Paragraph(
        "This instrument is intended for research and educational purposes only. "
        "Its outputs do not constitute medical advice, diagnosis, or treatment.",
        STYLES["footer_note"]))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    import os
    kb = os.path.getsize(PDF_FILE) // 1024
    print(f"Done — {PDF_FILE}  ({kb} KB)")


if __name__ == "__main__":
    main()
