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
)
from reportlab.pdfgen import canvas


# Clinical palette: simple, medical, printable
INK = colors.HexColor("#111827")
SLATE = colors.HexColor("#1F2937")
BLUE = colors.HexColor("#1D4ED8")
LIGHT_BLUE = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#CBD5E1")
SURFACE = colors.HexColor("#F8FAFC")
MUTED = colors.HexColor("#64748B")
WHITE = colors.white

GREEN = colors.HexColor("#166534")
RED = colors.HexColor("#991B1B")
ORANGE = colors.HexColor("#9A3412")

PAGE_W, PAGE_H = A4
ML = 20 * mm
MR = 20 * mm
MT = 22 * mm
MB = 20 * mm
CW = PAGE_W - ML - MR


def normalize_confidence(value):
    try:
        v = float(str(value).replace("%", "").strip())
        return v if v <= 1 else v / 100
    except Exception:
        return 0.0


def short_risk_label(risk):
    r = str(risk).lower()
    if "high" in r:
        return "High Risk"
    if "medium" in r:
        return "Moderate Risk"
    if "uncertain" in r:
        return "Uncertain"
    if "low" in r:
        return "Low Risk"
    return str(risk) if risk else "Not specified"


def risk_color(risk):
    r = str(risk).lower()
    if "high" in r:
        return RED
    if "medium" in r or "uncertain" in r:
        return ORANGE
    return GREEN


def prediction_color(prediction):
    return RED if "pneumonia" in str(prediction).lower() else GREEN


def make_report_id(final_report):
    raw = str(final_report) + datetime.now().isoformat()
    return "MTR-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10].upper()


class MedicalCanvas(canvas.Canvas):
    def __init__(self, *args, report_id="", patient_id="", **kwargs):
        super().__init__(*args, **kwargs)
        self.report_id = report_id
        self.patient_id = patient_id
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_header_footer(total_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_header_footer(self, total_pages):
        self.saveState()

        # Header
        self.setFillColor(WHITE)
        self.rect(0, PAGE_H - 25 * mm, PAGE_W, 25 * mm, fill=1, stroke=0)

        self.setStrokeColor(BORDER)
        self.setLineWidth(0.5)
        self.line(ML, PAGE_H - 25 * mm, PAGE_W - MR, PAGE_H - 25 * mm)

        self.setFont("Helvetica-Bold", 12)
        self.setFillColor(SLATE)
        self.drawString(ML, PAGE_H - 14 * mm, "MEDTRIAGE AI")

        self.setFont("Helvetica", 7)
        self.setFillColor(MUTED)
        self.drawString(ML, PAGE_H - 18 * mm, "AI-Assisted Preliminary Radiology Report")

        self.setFont("Helvetica", 7)
        self.drawRightString(PAGE_W - MR, PAGE_H - 14 * mm, f"Report ID: {self.report_id}")
        self.drawRightString(PAGE_W - MR, PAGE_H - 18 * mm, f"Patient ID: {self.patient_id}")

        # Footer
        self.setStrokeColor(BORDER)
        self.line(ML, 15 * mm, PAGE_W - MR, 15 * mm)

        self.setFont("Helvetica", 6.5)
        self.setFillColor(MUTED)
        self.drawString(
            ML,
            9 * mm,
            "Confidential medical document — AI-assisted preliminary triage only. Human validation required."
        )

        self.drawRightString(PAGE_W - MR, 9 * mm, f"Page {self._pageNumber} of {total_pages}")

        # Light watermark
        self.saveState()
        self.setFillColor(colors.Color(0.75, 0.80, 0.85, alpha=0.06))
        self.setFont("Helvetica-Bold", 42)
        self.translate(PAGE_W / 2, PAGE_H / 2)
        self.rotate(35)
        self.drawCentredString(0, 0, "PRELIMINARY AI REPORT")
        self.restoreState()

        self.restoreState()


def generate_pdf_report(final_report: dict, output_dir: str = "reports") -> str:
    os.makedirs(output_dir, exist_ok=True)

    report_id = make_report_id(final_report)
    filename = f"medical_report_{report_id}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    patient_id = str(final_report.get("patient_id", "N/A"))

    def canvas_maker(*args, **kwargs):
        return MedicalCanvas(*args, report_id=report_id, patient_id=patient_id, **kwargs)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=ML,
        rightMargin=MR,
        topMargin=35 * mm,
        bottomMargin=25 * mm,
        canvasmaker=canvas_maker,
    )

    def style(name, **kwargs):
        base = dict(
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=INK,
            alignment=TA_LEFT,
        )
        base.update(kwargs)
        return ParagraphStyle(name, **base)

    title_style = style(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=SLATE,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = style(
        "Subtitle",
        fontSize=9,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    section_style = style(
        "Section",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=SLATE,
        spaceBefore=10,
        spaceAfter=5,
    )

    label_style = style(
        "Label",
        fontName="Helvetica-Bold",
        fontSize=7,
        textColor=MUTED,
    )

    value_style = style(
        "Value",
        fontSize=9,
        textColor=INK,
    )

    body_style = style(
        "Body",
        fontSize=9,
        leading=14,
        textColor=INK,
    )

    small_style = style(
        "Small",
        fontSize=7.5,
        leading=11,
        textColor=MUTED,
    )

    story = []

    prediction = str(final_report.get("prediction", "Not available"))
    confidence_raw = final_report.get("confidence", "N/A")
    confidence = normalize_confidence(confidence_raw)
    confidence_label = f"{confidence * 100:.1f}%"

    risk = short_risk_label(final_report.get("risk_level", "Not specified"))
    validation = str(final_report.get("human_validation_status", "Pending"))
    doctor_name = str(final_report.get("reviewer_name", "Medical Reviewer"))
    review_time = str(final_report.get("review_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    analysis_type = str(final_report.get("analysis_type", "AI-assisted preliminary chest X-ray analysis"))
    generation_mode = str(final_report.get("generation_mode", "AI-assisted"))
    recommendation = str(final_report.get("recommendation", "No recommendation provided."))
    final_status = str(final_report.get("final_status", "Pending human validation."))
    disclaimer = str(final_report.get(
        "medical_disclaimer",
        "This report is generated by an AI system for preliminary triage assistance only. "
        "It does not replace a medical diagnosis. Final validation must be performed by a qualified healthcare professional."
    ))

    issue_date = datetime.now().strftime("%B %d, %Y")
    issue_time = datetime.now().strftime("%H:%M")

    story.append(Paragraph("CHEST X-RAY PRELIMINARY MEDICAL REPORT", title_style))
    story.append(Paragraph("AI-assisted triage report validated through Human-in-the-Loop review", subtitle_style))
    story.append(HRFlowable(width=CW, thickness=0.6, color=BORDER, spaceAfter=8 * mm))

    # Report information
    info_data = [
        [
            Paragraph("Report ID", label_style),
            Paragraph(report_id, value_style),
            Paragraph("Report Date", label_style),
            Paragraph(f"{issue_date} — {issue_time}", value_style),
        ],
        [
            Paragraph("Patient / Case ID", label_style),
            Paragraph(patient_id, value_style),
            Paragraph("Report Type", label_style),
            Paragraph(analysis_type, value_style),
        ],
        [
            Paragraph("Generated By", label_style),
            Paragraph("MedTriage AI System", value_style),
            Paragraph("LLM Mode", label_style),
            Paragraph(generation_mode, value_style),
        ],
    ]

    info_table = Table(info_data, colWidths=[32 * mm, 55 * mm, 32 * mm, CW - 119 * mm])
    info_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 7 * mm))

    # Key results
    story.append(Paragraph("1. AI ANALYSIS SUMMARY", section_style))

    result_data = [
        [
            Paragraph("AI Prediction", label_style),
            Paragraph(f'<font color="{prediction_color(prediction).hexval()}"><b>{prediction}</b></font>', value_style),
            Paragraph("Confidence", label_style),
            Paragraph(f"<b>{confidence_label}</b>", value_style),
        ],
        [
            Paragraph("Risk Level", label_style),
            Paragraph(f'<font color="{risk_color(risk).hexval()}"><b>{risk}</b></font>', value_style),
            Paragraph("Validation Status", label_style),
            Paragraph(f"<b>{validation}</b>", value_style),
        ],
    ]

    result_table = Table(result_data, colWidths=[32 * mm, 55 * mm, 32 * mm, CW - 119 * mm])
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 5 * mm))

    # Confidence bar, simple
    filled = max(2 * mm, CW * confidence)
    empty = CW - filled
    bar = Table([["", ""]], colWidths=[filled, empty], rowHeights=[3 * mm])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), BLUE),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E5E7EB")),
        ("BOX", (0, 0), (-1, -1), 0.3, BORDER),
    ]))
    story.append(bar)
    story.append(Spacer(1, 7 * mm))

    # Medical sections
    def add_section(title, text):
        story.append(Paragraph(title, section_style))
        table = Table([[Paragraph(str(text), body_style)]], colWidths=[CW])
        table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), WHITE),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(table)
        story.append(Spacer(1, 4 * mm))

    add_section(
        "2. Radiological Impression",
        f"Preliminary AI-assisted finding: {prediction}. Estimated triage risk level: {risk}."
    )

    add_section(
        "3. Clinical Recommendation",
        recommendation
    )

    add_section(
        "4. Final Human Validation Status",
        final_status
    )

    add_section(
        "5. Medical Disclaimer",
        disclaimer
    )

    # Doctor validation
    story.append(Paragraph("6. Doctor Review and Signature", section_style))

    doctor_data = [
        [
            Paragraph("Doctor / Reviewer Name", label_style),
            Paragraph(doctor_name, value_style),
        ],
        [
            Paragraph("Review Date and Time", label_style),
            Paragraph(review_time, value_style),
        ],
        [
            Paragraph("Validation Decision", label_style),
            Paragraph(validation, value_style),
        ],
    ]

    doctor_table = Table(doctor_data, colWidths=[55 * mm, CW - 55 * mm])
    doctor_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (0, -1), SURFACE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(doctor_table)
    story.append(Spacer(1, 10 * mm))

    # Signature area
    signature_table = Table(
        [
            [
                Paragraph("Doctor Signature", label_style),
                Paragraph("Medical Stamp / Seal", label_style),
            ],
            [
                Paragraph("<br/><br/><br/>Signature: ________________________________", value_style),
                Paragraph("<br/><br/><br/>Stamp: ________________________________", value_style),
            ],
        ],
        colWidths=[CW / 2, CW / 2],
        rowHeights=[8 * mm, 28 * mm],
    )

    signature_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(signature_table)
    story.append(Spacer(1, 7 * mm))

    story.append(Paragraph(
        f"Document finalized on {issue_date} at {issue_time}. "
        f"This report was approved by {doctor_name} through the MedTriage AI Human-in-the-Loop workflow.",
        small_style
    ))

    doc.build(story)
    return pdf_path