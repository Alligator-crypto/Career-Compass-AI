"""
report_generator.py
----------------------
WHY: Users often want to save or share their analysis results (e.g. email
to a mentor, keep for later). This generates a clean, single-page PDF
summary of an analysis session using reportlab, so the whole product isn't
locked inside a browser tab that resets on refresh.
"""
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib import colors


def build_report_pdf(analysis: dict) -> bytes:
    """Render an analysis dict (matching app/streamlit_app.py's session_state.analysis
    structure) into a PDF and return the raw bytes for a Streamlit download button."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.65*inch, rightMargin=0.65*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=4)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
                                    spaceAfter=4, textColor=colors.HexColor("#1F3864"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15)
    score_style = ParagraphStyle("Score", parent=styles["Normal"], fontSize=26, textColor=colors.HexColor("#1F6FEB"),
                                  spaceAfter=6)

    def rule():
        return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#1F3864"), spaceAfter=6)

    story = [Paragraph("CareerCompass AI — Resume Analysis Report", title_style), rule()]

    story.append(Paragraph("ATS Compatibility Score", section_style))
    story.append(Paragraph(f"{analysis['ats_score']} / 100", score_style))
    for label, points in analysis["breakdown"].items():
        story.append(Paragraph(f"&bull; {label}: {points}", body_style))

    story.append(Paragraph("Predicted Best-Fit Role", section_style))
    story.append(Paragraph(analysis["pred_role"], body_style))

    story.append(Paragraph("Detected Skills", section_style))
    story.append(Paragraph(", ".join(analysis["found_skills"]) or "None detected", body_style))

    story.append(Paragraph(f"Missing Skills (target role: {analysis['target_role']})", section_style))
    story.append(Paragraph(", ".join(analysis["missing"]) or "None — strong coverage", body_style))

    story.append(Paragraph("Strengths", section_style))
    for s in analysis["strengths"]:
        story.append(Paragraph(f"&bull; {s}", body_style))

    story.append(Paragraph("Weaknesses", section_style))
    for w in analysis["weaknesses"]:
        story.append(Paragraph(f"&bull; {w}", body_style))

    story.append(Paragraph("Resume Improvement Suggestions", section_style))
    for s in analysis["suggestions"]:
        story.append(Paragraph(f"&bull; {s}", body_style))

    if analysis.get("jd_analysis"):
        jd = analysis["jd_analysis"]
        story.append(Paragraph("Job Description Match", section_style))
        story.append(Paragraph(f"Semantic match: {jd['semantic_match_pct']}% &nbsp;|&nbsp; "
                                f"Skill coverage: {jd['skill_coverage_pct']}%", body_style))
        story.append(Paragraph(f"Missing skills for this posting: {', '.join(jd['missing']) or 'None'}", body_style))

    doc.build(story)
    return buf.getvalue()
