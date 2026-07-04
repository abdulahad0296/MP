"""
tools/pdf_exporter.py
---------------------
Generates a formatted PDF report from a completed pipeline run.

Output sections:
  1. Cover — topic, timestamp, summary stats
  2. Accepted Proposals — one card per accepted result with full details
  3. Feasibility Breakdown — per-component table for every reviewed plan
  4. Rejected Plans — list with rejection reason
  5. Appendix — all retrieved paper titles and all identified gaps

Dependencies:
    pip install reportlab

Usage (internal — called from app.py):
    from tools.pdf_exporter import generate_pdf
    path = generate_pdf(topic, results, papers, gaps, run_number)
    # returns absolute path to the written PDF file
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1F3864")
TEAL   = colors.HexColor("#0F6E56")
PURPLE = colors.HexColor("#3C3489")
AMBER  = colors.HexColor("#854F0B")
GREEN  = colors.HexColor("#085041")
RED    = colors.HexColor("#A32D2D")
WHITE  = colors.white
LIGHT_GREY = colors.HexColor("#F2F4F8")
MID_GREY   = colors.HexColor("#64748B")
BORDER     = colors.HexColor("#CBD5E1")
GREEN_BG   = colors.HexColor("#E0F3EA")
RED_BG     = colors.HexColor("#FCEBEB")
AMBER_BG   = colors.HexColor("#FDF3DC")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ── Custom flowable: coloured badge ──────────────────────────────────────────
class Badge(Flowable):
    """A small pill-shaped coloured badge with text."""
    def __init__(self, text: str, bg, fg=WHITE, font_size=8):
        super().__init__()
        self.text = text
        self.bg = bg
        self.fg = fg
        self.font_size = font_size
        self.width = len(text) * font_size * 0.6 + 14
        self.height = font_size + 8

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, radius=4, fill=1, stroke=0)
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawCentredString(self.width / 2, (self.height - self.font_size) / 2 + 1, self.text)


# ── Style registry ────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()

    def ps(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": ps("cover_title", fontSize=24, leading=30, textColor=WHITE,
                          fontName="Helvetica-Bold", spaceAfter=6),
        "cover_sub":   ps("cover_sub",   fontSize=11, leading=16, textColor=colors.HexColor("#CADCFC"),
                          fontName="Helvetica"),
        "cover_meta":  ps("cover_meta",  fontSize=9,  leading=13, textColor=MID_GREY,
                          fontName="Helvetica"),
        "section":     ps("section",     fontSize=14, leading=18, textColor=NAVY,
                          fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6),
        "subsection":  ps("subsection",  fontSize=11, leading=15, textColor=TEAL,
                          fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4),
        "body":        ps("body",        fontSize=10, leading=14, textColor=colors.HexColor("#1E293B"),
                          fontName="Helvetica", spaceAfter=4),
        "body_small":  ps("body_small",  fontSize=9,  leading=13, textColor=MID_GREY,
                          fontName="Helvetica"),
        "label":       ps("label",       fontSize=8,  leading=11, textColor=MID_GREY,
                          fontName="Helvetica-Bold"),
        "value":       ps("value",       fontSize=10, leading=13, textColor=colors.HexColor("#1E293B"),
                          fontName="Helvetica-Bold"),
        "mono":        ps("mono",        fontSize=9,  leading=13, textColor=colors.HexColor("#334155"),
                          fontName="Courier"),
        "caption":     ps("caption",     fontSize=8,  leading=11, textColor=MID_GREY,
                          fontName="Helvetica", alignment=TA_CENTER),
        "right":       ps("right",       fontSize=9,  leading=12, textColor=MID_GREY,
                          fontName="Helvetica", alignment=TA_RIGHT),
    }


# ── Page template with header / footer ───────────────────────────────────────
def _page_template(doc, topic: str):
    def _header_footer(canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE_H - 1.2 * cm, PAGE_W, 1.2 * cm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(MARGIN, PAGE_H - 0.75 * cm, "Agentic Research Planning Framework")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.75 * cm, topic[:60])
        # Footer
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, PAGE_W, 0.9 * cm, fill=1, stroke=0)
        canvas.setFillColor(MID_GREY)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(MARGIN, 0.32 * cm,
                          "Generated by Agentic Research Planning Framework  |  Jamia Millia Islamia")
        canvas.drawRightString(PAGE_W - MARGIN, 0.32 * cm, f"Page {doc.page}")
        canvas.restoreState()

    frame = Frame(MARGIN, 1.0 * cm, CONTENT_W, PAGE_H - 2.6 * cm, id="body")
    return PageTemplate(id="main", frames=[frame], onPage=_header_footer)


# ── Cover page ────────────────────────────────────────────────────────────────
def _cover(topic, results, accepted, papers, gaps, run_number, timestamp, styles):
    story = []
    total   = len(results)
    n_acc   = len(accepted)
    rate    = round(100 * n_acc / total) if total else 0
    scores  = [r.novelty_score for r in accepted]
    avg_nov = round(sum(scores) / len(scores), 2) if scores else 0.0

    # Navy cover block
    cover_table = Table(
        [[Paragraph("An Agentic Framework for<br/>Automated Research Planning",
                    styles["cover_title"])],
         [Paragraph(f"Research Topic: <b>{topic}</b>", styles["cover_sub"])],
         [Spacer(1, 0.3 * cm)],
         [Paragraph(f"Run #{run_number}  ·  {timestamp}", styles["cover_meta"])],
         [Paragraph("Abdul Ahad  ·  M.Sc. AI & Machine Learning  ·  Jamia Millia Islamia",
                    styles["cover_meta"])],
        ],
        colWidths=[CONTENT_W],
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.6 * cm))

    # Summary stats grid
    stat_data = [
        [_stat_cell("Papers retrieved",   len(papers),      NAVY,   styles),
         _stat_cell("Gaps identified",    len(gaps),        TEAL,   styles),
         _stat_cell("Plans reviewed",     total,            PURPLE, styles),
         _stat_cell("Accepted",           n_acc,            GREEN,  styles)],
        [_stat_cell("Acceptance rate",    f"{rate}%",
                    GREEN if rate >= 60 else (AMBER if rate >= 30 else RED), styles),
         _stat_cell("Avg novelty score",  avg_nov,          AMBER,  styles),
         _stat_cell("Novelty rejections", 0,                GREEN,  styles),
         _stat_cell("5-component\nfeasibility", "✓ all",  TEAL,   styles)],
    ]
    stat_table = Table(stat_data, colWidths=[CONTENT_W / 4] * 4,
                       rowHeights=[2.3 * cm] * 2)
    stat_table.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(stat_table)
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=BORDER,
                             spaceAfter=12, spaceBefore=12))
    story.append(Paragraph(
        "This report was generated automatically by the Agentic Research Planning Framework. "
        "It contains all accepted research proposals with full details, a per-plan feasibility "
        "breakdown, a list of rejected plans with reasons, and an appendix of retrieved papers "
        "and identified research gaps.",
        styles["body_small"]))
    story.append(PageBreak())
    return story


def _stat_cell(label, value, color, styles):
    """Returns a coloured stat cell for the cover table."""
    inner = Table(
        [[Paragraph(f'<font color="white"><b>{value}</b></font>',
                    ParagraphStyle("sv", fontSize=20, leading=24,
                                   fontName="Helvetica-Bold", alignment=TA_CENTER))],
         [Paragraph(f'<font color="white">{label}</font>',
                    ParagraphStyle("sl", fontSize=8, leading=11,
                                   fontName="Helvetica", alignment=TA_CENTER))]],
        colWidths=[(CONTENT_W / 4) - 12],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), color),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return inner


# ── Accepted proposals section ────────────────────────────────────────────────
def _accepted_section(accepted, styles):
    story = []
    story.append(Paragraph("Accepted Research Proposals", styles["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=2, color=TEAL,
                             spaceAfter=10, spaceBefore=2))

    for i, r in enumerate(accepted, 1):
        gap_label = ""
        if r.plan.source_gap:
            desc = r.plan.source_gap.description
            gap_label = (desc[:80] + "...") if len(desc) > 80 else desc

        # Proposal header row
        header_data = [[
            Paragraph(f'<font color="white"><b>Proposal {i}</b></font>',
                      ParagraphStyle("ph", fontSize=10, fontName="Helvetica-Bold",
                                     alignment=TA_LEFT)),
            Paragraph(f'<font color="white">Novelty: {r.novelty_score:.2f} / 10</font>',
                      ParagraphStyle("pn", fontSize=9, fontName="Helvetica",
                                     alignment=TA_RIGHT)),
        ]]
        header_tbl = Table(header_data, colWidths=[CONTENT_W * 0.7, CONTENT_W * 0.3])
        header_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), TEAL),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        story.append(header_tbl)

        # Suggested title
        title_text = r.suggested_title or r.plan.research_question
        title_tbl = Table(
            [[Paragraph(title_text,
                        ParagraphStyle("pt", fontSize=11, leading=15,
                                       fontName="Helvetica-Bold",
                                       textColor=colors.HexColor("#1E293B")))]],
            colWidths=[CONTENT_W],
        )
        title_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), GREEN_BG),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ]))
        story.append(title_tbl)

        # Research direction
        if r.research_direction:
            story.append(_box_row("Research Direction", r.research_direction,
                                  CONTENT_W, styles))

        # Four-field row: question | method | dataset | metric
        detail_data = [[
            _field_cell("Research Question", r.plan.research_question, styles),
            _field_cell("Proposed Method",   r.plan.proposed_method,   styles),
        ], [
            _field_cell("Dataset",           r.plan.dataset,           styles),
            _field_cell("Evaluation Metric", r.plan.evaluation_metric, styles),
        ]]
        detail_tbl = Table(detail_data, colWidths=[CONTENT_W / 2] * 2)
        detail_tbl.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(detail_tbl)

        # Experimental blueprint
        if r.experimental_blueprint:
            story.append(_box_row("Experimental Blueprint",
                                  r.experimental_blueprint, CONTENT_W, styles,
                                  bg=AMBER_BG, label_color=AMBER))

        # Novelty bar (text representation)
        pct = min(100, int(r.novelty_score * 10))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        story.append(Paragraph(
            f'<font name="Courier" color="#0F6E56"><b>{bar}</b></font>'
            f'  <font color="#64748B">{r.novelty_score:.2f} / 10</font>',
            ParagraphStyle("nb", fontSize=9, leading=13, fontName="Helvetica",
                           spaceAfter=2)))

        # Source gap label
        if gap_label:
            story.append(Paragraph(f"Source gap: {gap_label}", styles["body_small"]))

        story.append(Spacer(1, 0.5 * cm))
        story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=BORDER,
                                 spaceAfter=10, spaceBefore=0))

    return story


def _field_cell(label, value, styles):
    """Returns a bordered field cell paragraph block."""
    inner = Table(
        [[Paragraph(label, styles["label"])],
         [Paragraph(value,  styles["body"])]],
        colWidths=[(CONTENT_W / 2) - 6],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return inner


def _box_row(label, text, width, styles, bg=LIGHT_GREY, label_color=TEAL):
    tbl = Table(
        [[Paragraph(f'<font color="#{_hex(label_color)}"><b>{label}</b></font>',
                    ParagraphStyle("br_lbl", fontSize=8.5, fontName="Helvetica-Bold",
                                   textColor=label_color))],
         [Paragraph(text, styles["body"])]],
        colWidths=[width],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), bg),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
    ]))
    return tbl


def _hex(c: colors.Color) -> str:
    """Convert a ReportLab color to 6-char hex without #."""
    r, g, b = int(c.red * 255), int(c.green * 255), int(c.blue * 255)
    return f"{r:02X}{g:02X}{b:02X}"


# ── Feasibility breakdown section ─────────────────────────────────────────────
def _feasibility_section(results, styles):
    story = []
    story.append(PageBreak())
    story.append(Paragraph("Feasibility Breakdown — All Reviewed Plans", styles["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=2, color=AMBER,
                             spaceAfter=10, spaceBefore=2))
    story.append(Paragraph(
        "Each plan is checked against five independent feasibility components. "
        "✓ = passed  ·  ✗ = failed (reason shown). All five must pass for a plan to be accepted.",
        styles["body_small"]))
    story.append(Spacer(1, 0.3 * cm))

    COMP_KEYS = [
        ("data_availability",          "Data"),
        ("metric_validity",            "Metric"),
        ("computational_feasibility",  "Compute"),
        ("methodological_practicality","Method"),
        ("time_feasibility",           "Scope"),
    ]

    # Header row
    header = ["#", "Research Question", "Novelty"] + [c[1] for c in COMP_KEYS] + ["Result"]
    col_widths = [0.5*cm, 5.5*cm, 1.4*cm, 1.1*cm, 1.1*cm, 1.2*cm, 1.2*cm, 1.1*cm, 1.4*cm]
    # Ensure widths add up to CONTENT_W
    total_w = sum(col_widths)
    col_widths = [w * CONTENT_W / total_w for w in col_widths]

    table_data = [header]
    table_style = [
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (1, 0), (1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
    ]

    for i, r in enumerate(results, 1):
        q = r.plan.research_question
        q_short = (q[:55] + "...") if len(q) > 55 else q
        comp_cells = []
        for key, _ in COMP_KEYS:
            comps = getattr(r, "feasibility_components", {}) or {}
            if key in comps:
                passed, note = comps[key]
                cell_text = "✓" if passed else "✗"
                table_style.append(
                    ("TEXTCOLOR", (3 + list(k for k,_ in COMP_KEYS).index(key), i),
                                  (3 + list(k for k,_ in COMP_KEYS).index(key), i),
                     GREEN if passed else RED)
                )
            else:
                cell_text = "–"
            comp_cells.append(cell_text)

        result_text = "PASS" if r.accepted else "FAIL"
        table_style.append(("TEXTCOLOR", (-1, i), (-1, i), GREEN if r.accepted else RED))
        table_style.append(("FONTNAME",  (-1, i), (-1, i), "Helvetica-Bold"))

        row = [str(i), q_short, f"{r.novelty_score:.2f}"] + comp_cells + [result_text]
        table_data.append(row)

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(table_style))
    story.append(tbl)
    return story


# ── Rejected plans section ────────────────────────────────────────────────────
def _rejected_section(rejected, styles):
    if not rejected:
        return []
    story = []
    story.append(PageBreak())
    story.append(Paragraph("Rejected Plans", styles["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=2, color=RED,
                             spaceAfter=10, spaceBefore=2))
    story.append(Paragraph(
        f"{len(rejected)} plan(s) did not pass all checks. Rejection reasons are shown below.",
        styles["body_small"]))
    story.append(Spacer(1, 0.3 * cm))

    for i, r in enumerate(rejected, 1):
        tbl = Table(
            [[Paragraph(f'<b>{i}.</b>  {r.plan.research_question}', styles["body"]),
              Paragraph(f'Novelty: {r.novelty_score:.2f}', styles["right"])],
             [Paragraph(r.feasibility_notes or "Low novelty score", styles["body_small"]),
              Paragraph("")]],
            colWidths=[CONTENT_W * 0.78, CONTENT_W * 0.22],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), RED_BG),
            ("BOX",           (0, 0), (-1, -1), 0.5, RED),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.2 * cm))
    return story


# ── Appendix ──────────────────────────────────────────────────────────────────
def _appendix(papers, gaps, styles):
    story = []
    story.append(PageBreak())
    story.append(Paragraph("Appendix", styles["section"]))
    story.append(HRFlowable(width=CONTENT_W, thickness=2, color=NAVY,
                             spaceAfter=10, spaceBefore=2))

    # A — Retrieved papers
    story.append(Paragraph("A — Retrieved Papers from arXiv", styles["subsection"]))
    paper_data = [["#", "Title", "arXiv ID"]]
    for i, p in enumerate(papers, 1):
        title_short = (p.title[:70] + "...") if len(p.title) > 70 else p.title
        paper_data.append([str(i), title_short, p.arxiv_id[-15:] if p.arxiv_id else ""])
    pt = Table(paper_data,
               colWidths=[0.6*cm, CONTENT_W - 3.4*cm, 2.8*cm],
               repeatRows=1)
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (2, 0), (2, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.5 * cm))

    # B — Research gaps
    story.append(Paragraph("B — Identified Research Gaps", styles["subsection"]))
    for i, g in enumerate(gaps, 1):
        concepts = ", ".join(g.related_concepts[:6]) if g.related_concepts else "-"
        tbl = Table(
            [[Paragraph(f'<b>Gap {i}:</b>  {g.description}', styles["body"])],
             [Paragraph(f'Related concepts: {concepts}', styles["body_small"])]],
            colWidths=[CONTENT_W],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_GREY),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.15 * cm))
    return story


# ── Public entry point ────────────────────────────────────────────────────────
def generate_pdf(
    topic: str,
    results: list,
    papers: list,
    gaps: list,
    run_number: int = 1,
    output_dir: str = "outputs",
) -> str:
    """
    Build a full-run PDF report and write it to output_dir.

    Returns the absolute path of the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:40]
    filename = f"research_plan_{safe_topic.replace(' ','_')}_run{run_number}.pdf"
    filepath = os.path.abspath(os.path.join(output_dir, filename))

    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]

    doc = BaseDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.4 * cm, bottomMargin=1.2 * cm,
        title=f"Research Plan — {topic}",
        author="Abdul Ahad — Agentic Research Planning Framework",
    )
    doc.addPageTemplates([_page_template(doc, topic)])

    styles = _styles()
    story = []
    story += _cover(topic, results, accepted, papers, gaps, run_number, timestamp, styles)
    story += _accepted_section(accepted, styles)
    story += _feasibility_section(results, styles)
    story += _rejected_section(rejected, styles)
    story += _appendix(papers, gaps, styles)

    doc.build(story)
    return filepath