"""
jd_matcher.py
---------------
WHY: A fixed skill list per role is a good starting point, but real job
postings vary — one "Data Scientist" posting wants Spark experience,
another wants A/B testing and SQL. This module lets a user paste an
*actual* job description and get a similarity score + skill-gap analysis
against that specific posting, using the same sentence-embedding model as
career prediction (see preprocessing/text_preprocessing.py).

ALGORITHM: cosine similarity between the resume's embedding and the job
description's embedding. Embeddings are already unit-normalized
(normalize_embeddings=True in build_embeddings), so cosine similarity is
just a dot product — fast and numerically stable.
"""
import sys
import numpy as np
sys.path.append("/home/claude/careercompass-ai")
from preprocessing.text_preprocessing import build_embeddings
from utils.skill_extractor import extract_skills


def match_score(resume_text: str, jd_text: str) -> float:
    """Cosine similarity (0-100 scale) between resume and job description embeddings."""
    embeds = build_embeddings([resume_text, jd_text])
    similarity = float(np.dot(embeds[0], embeds[1]))
    return round(max(min(similarity, 1.0), 0.0) * 100, 1)


def skill_gap_vs_jd(resume_text: str, jd_text: str) -> dict:
    """Extract skills from both texts and diff them, so recommendations are specific
    to *this* job posting rather than a generic role template."""
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    return {
        "matched": sorted(resume_skills & jd_skills),
        "missing": sorted(jd_skills - resume_skills),
        "extra": sorted(resume_skills - jd_skills),
        "jd_skill_count": len(jd_skills),
    }


def analyze_against_jd(resume_text: str, jd_text: str) -> dict:
    """Full JD-match analysis combining semantic similarity + explicit skill gap."""
    score = match_score(resume_text, jd_text)
    gaps = skill_gap_vs_jd(resume_text, jd_text)
    coverage = (len(gaps["matched"]) / gaps["jd_skill_count"] * 100) if gaps["jd_skill_count"] else 0.0
    return {
        "semantic_match_pct": score,
        "skill_coverage_pct": round(coverage, 1),
        **gaps,
    }
