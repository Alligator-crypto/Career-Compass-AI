"""
resume_parser.py
------------------
WHY: Users upload resumes in different formats. This module normalizes
any of PDF / DOCX / TXT into plain text so the same NLP pipeline can
process it regardless of source format.

UPGRADE (v2): added SECTION-AWARE parsing. v1 only produced one flat text
blob, so scoring/suggestions had to guess about the whole resume at once
("does this resume mention 'project' anywhere?"). Real ATS systems and
recruiters look at *specific sections* — is the Experience section
substantive, is there a real Projects section, etc. `extract_sections()`
splits the resume into labeled sections so the rest of the app can reason
about them individually (see ats_scorer.py and recommender.py v2).
"""
import io
import re
import pdfplumber
import docx

SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "objective", "profile"],
    "education": ["education", "academic background", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies"],
    "experience": ["experience", "work experience", "professional experience", "employment history"],
    "projects": ["projects", "personal projects", "academic projects", "portfolio"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
    "additional": ["additional information", "extracurricular", "activities", "interests"],
}
_ALL_HEADER_PHRASES = {phrase: canon for canon, phrases in SECTION_HEADERS.items() for phrase in phrases}


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = filename.lower().split(".")[-1]
    try:
        if ext == "pdf":
            return _extract_pdf(file_bytes)
        elif ext == "docx":
            return _extract_docx(file_bytes)
        elif ext == "txt":
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: .{ext}. Use PDF, DOCX or TXT.")
    except Exception as e:
        raise RuntimeError(f"Failed to parse '{filename}': {e}")


def _extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def _extract_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def _looks_like_header(line: str) -> str | None:
    """Return the canonical section name if `line` looks like a section header, else None."""
    clean = line.strip().strip(":").lower()
    if not clean or len(clean.split()) > 5:
        return None
    for phrase, canon in _ALL_HEADER_PHRASES.items():
        if clean == phrase or clean.startswith(phrase):
            return canon
    return None


def extract_sections(text: str) -> dict:
    """Split resume text into labeled sections by detecting header lines.
    Returns a dict {section_name: section_text}; anything before the first
    detected header is kept under 'header' (name/contact info block)."""
    lines = text.split("\n")
    sections = {"header": []}
    current = "header"
    for line in lines:
        header = _looks_like_header(line)
        if header:
            current = header
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}
