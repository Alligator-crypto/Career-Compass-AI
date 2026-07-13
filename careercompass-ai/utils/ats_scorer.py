"""
ats_scorer.py
---------------
WHY: Recruiters' Applicant Tracking Systems screen resumes before a human
ever sees them. We approximate that gate with a transparent, rule-based
0-100 score (not a black-box model) so users can see *exactly* why they
got their score and what to fix — trust matters more than marginal
accuracy for this particular feature.

UPGRADE (v2): now section-aware. v1 guessed "does has_projects=True" from
a single keyword search on the whole resume blob. v2 takes the actual
parsed `sections` dict (from utils.resume_parser.extract_sections) so it
can check real section presence/substance (e.g. "Experience section exists
AND has more than a couple lines") instead of one global keyword flag.
"""


def compute_ats_score(resume_text: str, found_skills: list, target_role_skills: list,
                       sections: dict, num_sections: int = None):
    breakdown = {}
    sections = sections or {}
    num_sections = num_sections if num_sections is not None else len(sections)

    coverage = len(set(found_skills) & set(target_role_skills)) / max(len(target_role_skills), 1)
    breakdown["Skill match to target role (55 pts max)"] = round(coverage * 55, 1)

    word_count = len(resume_text.split())
    length_score = min(word_count / 300, 1) * 10
    breakdown["Resume length/detail (10 pts max)"] = round(length_score, 1)

    projects_text = sections.get("projects", "")
    has_substantive_projects = len(projects_text.split()) > 15
    breakdown["Has substantive projects section (12 pts)"] = 12.0 if has_substantive_projects else 0.0

    has_certifications = bool(sections.get("certifications", "").strip())
    breakdown["Has certifications (8 pts)"] = 8.0 if has_certifications else 0.0

    structure_score = min(num_sections / 8, 1) * 10
    breakdown["Structure/sections completeness (10 pts max)"] = round(structure_score, 1)

    action_verbs = ["led", "built", "designed", "improved", "automated", "reduced", "launched", "optimized"]
    experience_text = sections.get("experience", "") + " " + projects_text
    verb_score = 5.0 if any(v in experience_text.lower() for v in action_verbs) else 0.0
    breakdown["Uses strong action verbs (5 pts)"] = verb_score

    total = round(sum(breakdown.values()), 1)
    total = min(total, 100.0)
    return total, breakdown
