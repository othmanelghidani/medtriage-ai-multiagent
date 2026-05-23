import os
import hashlib
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ── Palette — clean clinical whites + one strong accent ──────────────
INK        = colors.HexColor("#0F1923")   # near-black for text
SLATE      = colors.HexColor("#1E293B")   # dark slate for headers
ACCENT     = colors.HexColor("#0369A1")   # strong medical blue
ACCENT_LT  = colors.HexColor("#E0F2FE")   # light blue tint
RULE       = colors.HexColor("#0EA5E9")   # sky blue rule lines
GREEN      = colors.HexColor("#15803D")
GREEN_LT   = colors.HexColor("#DCFCE7")
RED        = colors.HexColor("#B91C1C")
RED_LT     = colors.HexColor("#FEE2E2")
ORANGE     = colors.HexColor("#C2410C")
ORANGE_LT  = colors.HexColor("#FFEDD5")
MUTED      = colors.HexColor("#64748B")
BORDER     = colors.HexColor("#CBD5E1")
SURFACE    = colors.HexColor("#F8FAFC")
STRIPE     = colors.HexColor("#F1F5F9")
WHITE      = colors.white

PAGE_W, PAGE_H = A4
ML = 20 * mm
MR = 20 * mm
MT = 22 * mm
MB = 20 * mm
CW = PAGE_W - ML - MR   # usable content width


# ── Helpers ───────────────────────────────────────────────────────────
def normalize_confidence(value):
    try:
        v = float(str(value).replace("%", "").strip())
        return v if v <= 1 else v / 100
    except Exception:
        return 0.0


def short_risk_label(risk):
    r = str(risk).lower()
    if "high"      in r: return "High Risk"
    if "medium"    in r: return "Moderate Risk"
    if "uncertain" in r: return "Uncertain"
    if "low"       in r: return "Low Risk"
    return str(risk) if risk else "Not specified"


def risk_color(risk):
    r = str(risk).lower()
    if "high"           in r: return RED
    if "medium"         in r: return ORANGE
    if "uncertain"      in r: return ORANGE
    return GREEN


def risk_bg(risk):
    r = str(risk).lower()
    if "high"           in r: return RED_LT
    if "medium"         in r: return ORANGE_LT
    if "uncertain"      in r: return ORANGE_LT
    return GREEN_LT


def prediction_color(pred):
    return RED if "pneumonia" in str(pred).lower() else GREEN


def prediction_bg(pred):
    return RED_LT if "pneumonia" in str(pred).lower() else GREEN_LT


def make_report_id(final_report):
    raw = str(final_report) + datetime.now().isoformat()
    return "MTR-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10].upper()


# ── Custom canvas — draws header + footer on every page ───────────────
class MedicalCanvas(canvas.Canvas):
    def __init__(self, *args, report_id="", patient_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.report_id  = report_id
        self.patient_id = patient_id
        self._saved     = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_page(n)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_page(self, total):
        self.saveState()

        # ── HEADER band ──────────────────────────────────────────────
        H_TOP = PAGE_H - 18 * mm
        BAND  = 16 * mm

        # White base + bottom border
        self.setFillColor(WHITE)
        self.rect(0, H_TOP, PAGE_W, BAND + 2 * mm, fill=1, stroke=0)

        # Left colour block
        self.setFillColor(SLATE)
        self.rect(0, H_TOP, 56 * mm, BAND, fill=1, stroke=0)

        # Thin accent stripe along the left block right edge
        self.setFillColor(RULE)
        self.rect(56 * mm, H_TOP, 0.8, BAND, fill=1, stroke=0)

        # Bottom rule across full width
        self.setFillColor(RULE)
        self.rect(0, H_TOP, PAGE_W, 0.6, fill=1, stroke=0)

        # Wordmark inside dark block
        self.setFont("Helvetica-Bold", 10)
        self.setFillColor(WHITE)
        self.drawString(ML, H_TOP + 9.5 * mm, "MEDTRIAGE AI")
        self.setFont("Helvetica", 6.5)
        self.setFillColor(colors.HexColor("#94A3B8"))
        self.drawString(ML, H_TOP + 5 * mm, "AI-ASSISTED RADIOLOGY")

        # Right side — report metadata
        self.setFont("Helvetica", 7)
        self.setFillColor(MUTED)
        self.drawRightString(PAGE_W - MR, H_TOP + 10.5 * mm,
                             f"Report ID:  {self.report_id}")
        self.drawRightString(PAGE_W - MR, H_TOP + 5.8 * mm,
                             f"Patient:  {self.patient_id}")

        # ── FOOTER band ───────────────────────────────────────────────
        self.setFillColor(SURFACE)
        self.rect(0, 0, PAGE_W, 14 * mm, fill=1, stroke=0)
        self.setFillColor(RULE)
        self.rect(0, 14 * mm, PAGE_W, 0.5, fill=1, stroke=0)

        self.setFont("Helvetica", 6.2)
        self.setFillColor(MUTED)
        self.drawString(ML, 5.5 * mm,
            "CONFIDENTIAL — AI-generated preliminary triage only. "
            "Must be reviewed by a qualified medical professional before clinical use.")
        self.setFont("Helvetica-Bold", 6.8)
        self.setFillColor(ACCENT)
        self.drawRightString(PAGE_W - MR, 5.5 * mm,
                             f"Page {self._pageNumber} of {total}")

        # ── Subtle watermark ─────────────────────────────────────────
        self.saveState()
        self.setFillColor(colors.Color(0.85, 0.90, 0.95, alpha=0.07))
        self.setFont("Helvetica-Bold", 44)
        self.translate(PAGE_W / 2, PAGE_H / 2)
        self.rotate(38)
        self.drawCentredString(0, 0, "AI-ASSISTED  PRELIMINARY")
        self.restoreState()

        self.restoreState()


# ── PDF builder ───────────────────────────────────────────────────────
def generate_pdf_report(final_report: dict, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)

    report_id  = make_report_id(final_report)
    filename   = f"medical_report_{report_id}.pdf"
    pdf_path   = os.path.join(output_dir, filename)
    patient_id = str(final_report.get("patient_id", "N/A"))

    def canvas_maker(*a, **kw):
        return MedicalCanvas(*a, report_id=report_id, patient_id=patient_id, **kw)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT + 18 * mm,
        bottomMargin=MB + 14 * mm,
        canvasmaker=canvas_maker,
    )

    # ── Shared style factory ─────────────────────────────────────────
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9, leading=13,
                    textColor=INK, alignment=TA_LEFT)
        base.update(kw)
        return ParagraphStyle(name, **base)

    sTitle    = S("sTitle",  fontName="Helvetica-Bold", fontSize=20, leading=24,
                             textColor=SLATE, spaceAfter=2)
    sDate     = S("sDate",   fontSize=8, textColor=MUTED, spaceAfter=0)
    sSec      = S("sSec",    fontName="Helvetica-Bold", fontSize=9, leading=11,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=4,
                             letterSpacing=0.6)
    sLbl      = S("sLbl",    fontName="Helvetica-Bold", fontSize=6.8, leading=9,
                             textColor=MUTED)
    sVal      = S("sVal",    fontSize=9, leading=13, textColor=INK)
    sValC     = S("sValC",   fontName="Helvetica-Bold", fontSize=9.5, leading=12,
                             textColor=INK, alignment=TA_CENTER)
    sBody     = S("sBody",   fontSize=8.8, leading=14, textColor=INK)
    sDisc     = S("sDisc",   fontSize=7.8, leading=12,
                             textColor=colors.HexColor("#7C2D12"))

    story = []

    # ── Derived data ─────────────────────────────────────────────────
    prediction     = str(final_report.get("prediction",              "Not available"))
    confidence_raw = final_report.get("confidence",                  "N/A")
    confidence     = normalize_confidence(confidence_raw)
    conf_label     = f"{confidence * 100:.1f}%"
    risk           = short_risk_label(final_report.get("risk_level", "Not specified"))
    validation     = str(final_report.get("human_validation_status", "Pending"))
    reviewer       = str(final_report.get("reviewer_name",           "Not provided"))
    review_time    = str(final_report.get("review_timestamp",        "Not provided"))
    analysis_type  = str(final_report.get("analysis_type",           "Chest X-Ray Analysis"))
    gen_mode       = str(final_report.get("generation_mode",         "AI-assisted"))
    recommendation = str(final_report.get("recommendation",          "No recommendation provided."))
    final_status   = str(final_report.get("final_status",            "Pending"))
    disclaimer_txt = str(final_report.get(
        "medical_disclaimer",
        "This AI-generated report is intended for preliminary triage purposes only and must not "
        "be used as the sole basis for clinical decision-making. All findings must be reviewed "
        "and confirmed by a qualified radiologist or medical professional."))

    pred_col = prediction_color(prediction)
    pred_bg  = prediction_bg(prediction)
    risk_col = risk_color(risk)
    risk_bg_ = risk_bg(risk)

    # ══════════════════════════════════════════════════════════════════
    # 1. TITLE BLOCK
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("Medical Triage Report", sTitle))
    story.append(Spacer(1, 1 * mm))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')} at "
        f"{datetime.now().strftime('%H:%M')} &nbsp;·&nbsp; "
        f"Report ID: <b>{report_id}</b>",
        sDate))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width=CW, thickness=0.5, color=BORDER, spaceAfter=5 * mm))

    # ══════════════════════════════════════════════════════════════════
    # 2. IDENTITY ROW
    # ══════════════════════════════════════════════════════════════════
    def label_val(lbl, val):
        return [Paragraph(lbl, sLbl), Paragraph(str(val), sVal)]

    id_data = [
        [Paragraph("PATIENT / CASE ID", sLbl),
         Paragraph("ANALYSIS TYPE",     sLbl),
         Paragraph("GENERATION MODE",   sLbl),
         Paragraph("REPORT DATE",       sLbl)],
        [Paragraph(patient_id,          sVal),
         Paragraph(analysis_type,       sVal),
         Paragraph(gen_mode,            sVal),
         Paragraph(datetime.now().strftime("%d %b %Y"), sVal)],
    ]
    id_tbl = Table(id_data, colWidths=[CW / 4] * 4)
    id_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), SLATE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.HexColor("#94A3B8")),
        ("BACKGROUND",   (0, 1), (-1, 1), WHITE),
        ("GRID",         (0, 0), (-1, -1), 0.4, BORDER),
        ("LINEBELOW",    (0, 0), (-1, 0), 0, WHITE),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(id_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════════════════════════════
    # 3. KEY FINDINGS — 4 metric tiles
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("KEY FINDINGS", sSec))

    def metric_cell(label, value, fg, bg):
        return Table(
            [[Paragraph(label, sLbl)],
             [Paragraph(f'<font color="{fg.hexval()}"><b>{value}</b></font>',
                        S("mc", fontName="Helvetica-Bold", fontSize=11,
                          leading=14, textColor=fg, alignment=TA_CENTER))]],
            colWidths=[CW / 4 - 3 * mm],
            rowHeights=[8 * mm, 14 * mm],
        )

    tile_pred = Table(
        [[Paragraph("AI PREDICTION", sLbl)],
         [Paragraph(prediction, S("tp", fontName="Helvetica-Bold", fontSize=10.5,
                                  leading=13, textColor=pred_col, alignment=TA_CENTER))]],
        colWidths=[CW / 4 - 2 * mm],
    )
    tile_risk = Table(
        [[Paragraph("RISK LEVEL", sLbl)],
         [Paragraph(risk, S("tr", fontName="Helvetica-Bold", fontSize=10.5,
                             leading=13, textColor=risk_col, alignment=TA_CENTER))]],
        colWidths=[CW / 4 - 2 * mm],
    )
    tile_conf = Table(
        [[Paragraph("AI CONFIDENCE", sLbl)],
         [Paragraph(conf_label, S("tc", fontName="Helvetica-Bold", fontSize=14,
                                   leading=16, textColor=ACCENT, alignment=TA_CENTER))]],
        colWidths=[CW / 4 - 2 * mm],
    )
    val_color = GREEN if "approv" in validation.lower() else (RED if "reject" in validation.lower() else ORANGE)
    tile_val = Table(
        [[Paragraph("VALIDATION", sLbl)],
         [Paragraph(validation, S("tv", fontName="Helvetica-Bold", fontSize=10.5,
                                   leading=13, textColor=val_color, alignment=TA_CENTER))]],
        colWidths=[CW / 4 - 2 * mm],
    )

    for t, bg, fg in [(tile_pred, pred_bg, pred_col),
                      (tile_risk, risk_bg_, risk_col),
                      (tile_conf, ACCENT_LT, ACCENT),
                      (tile_val,  GREEN_LT if "approv" in validation.lower()
                                  else (RED_LT if "reject" in validation.lower()
                                        else ORANGE_LT), val_color)]:
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), bg),
            ("BOX",          (0, 0), (-1, -1), 0.5, BORDER),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
            ("LEFTPADDING",  (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ]))

    tiles_row = Table(
        [[tile_pred, tile_risk, tile_conf, tile_val]],
        colWidths=[CW / 4] * 4,
        spaceBefore=0,
    )
    tiles_row.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",(0, 0), (-1, -1), 2),
    ]))
    story.append(tiles_row)
    story.append(Spacer(1, 3 * mm))

    # ── Confidence bar ────────────────────────────────────────────────
    filled = max(3 * mm, CW * confidence)
    empty  = CW - filled
    
    

    bar = Table([["", ""]], colWidths=[filled, empty], rowHeights=[4.5 * mm])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), ACCENT),
        ("BACKGROUND", (1,0),(1,0), colors.HexColor("#E2E8F0")),
        ("BOX",        (0,0),(-1,-1), 0.3, BORDER),
    ]))
    story.append(bar)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════════════════════════════
    # 4. CLINICAL SECTIONS — helper that draws a left-accented block
    # ══════════════════════════════════════════════════════════════════
    def add_section(heading, text, accent_col=ACCENT, bg_col=WHITE):
        if not text or not str(text).strip():
            return
        story.append(Paragraph(heading.upper(), sSec))
        tbl = Table([[Paragraph(str(text), sBody)]], colWidths=[CW])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), bg_col),
            ("BOX",          (0,0),(-1,-1), 0.4, BORDER),
            ("LINEBEFORE",   (0,0),(0,-1),  3, accent_col),
            ("TOPPADDING",   (0,0),(-1,-1), 9),
            ("BOTTOMPADDING",(0,0),(-1,-1), 9),
            ("LEFTPADDING",  (0,0),(-1,-1), 12),
            ("RIGHTPADDING", (0,0),(-1,-1), 11),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 3 * mm))

    add_section(
        "Radiological Impression",
        f"Preliminary AI finding: <b>{prediction}</b>. Assessed risk level: <b>{risk}</b>.",
        risk_col, risk_bg_,
    )
    add_section("Clinical Recommendation", recommendation, ACCENT, ACCENT_LT)
    add_section(
        "Final Status",
        final_status,
        GREEN if "approv" in final_status.lower() else ORANGE,
        GREEN_LT if "approv" in final_status.lower() else ORANGE_LT,
    )

    # ══════════════════════════════════════════════════════════════════
    # 5. HUMAN REVIEW TABLE
    # ══════════════════════════════════════════════════════════════════
    story.append(Paragraph("HUMAN REVIEW & VALIDATION", sSec))

    rev_data = [
        [Paragraph("REVIEWER NAME",       sLbl),
         Paragraph("REVIEW TIMESTAMP",    sLbl),
         Paragraph("VALIDATION STATUS",   sLbl)],
        [Paragraph(reviewer,              sVal),
         Paragraph(review_time,           sVal),
         Paragraph(f'<font color="{val_color.hexval()}"><b>{validation}</b></font>',
                   S("vs", fontName="Helvetica-Bold", fontSize=9,
                     textColor=val_color))],
    ]
    rev_tbl = Table(rev_data, colWidths=[CW * 0.36, CW * 0.36, CW * 0.28])
    rev_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), SLATE),
        ("TEXTCOLOR",    (0,0),(-1,0), colors.HexColor("#94A3B8")),
        ("BACKGROUND",   (0,1),(-1,1), WHITE),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, STRIPE]),
        ("GRID",         (0,0),(-1,-1), 0.4, BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(rev_tbl)
    story.append(Spacer(1, 3 * mm))

    # Signature strip
    sig_data = [
        [Paragraph("AUTHORISED BY", sLbl),
         Paragraph("DIGITAL VALIDATION REFERENCE", sLbl)],
        [Paragraph(f"<b>{reviewer}</b>", sVal),
         Paragraph(
             f"Validated through MedTriage AI Human-in-the-Loop Workflow &nbsp;·&nbsp; "
             f"Report ID: <b>{report_id}</b>",
             S("sv", fontSize=8.5, textColor=MUTED))],
    ]
    sig_tbl = Table(sig_data, colWidths=[CW * 0.35, CW * 0.65])
    sig_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), ACCENT_LT),
        ("BACKGROUND",   (0,1),(-1,1), WHITE),
        ("GRID",         (0,0),(-1,-1), 0.4, BORDER),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",  (0,0),(-1,-1), 9),
        ("RIGHTPADDING", (0,0),(-1,-1), 9),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LINEABOVE",    (0,1),(0,1),   1.5, ACCENT),
    ]))
    story.append(sig_tbl)
    story.append(Spacer(1, 5 * mm))

    # ══════════════════════════════════════════════════════════════════
    # 6. DISCLAIMER
    # ══════════════════════════════════════════════════════════════════
    
    

    doc.build(story)
    return pdf_path