"""
generate_dataset.py
--------------------
Builds the training dataset used by CareerCompass AI.

WHY SYNTHETIC DATA: This sandbox has no internet access to Kaggle, so we
programmatically generate a resume-skills dataset that mirrors the
structure of Kaggle's "Resume Dataset" (skills/category) and "Job
Description Dataset" (role -> required skills). To use a *real* Kaggle
dataset instead, download e.g.:
  - https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
  - https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset
and map their columns to: ['resume_text', 'role'] then re-run preprocessing.

UPGRADE (v2): resumes are now generated as realistic MULTI-SECTION text
(SUMMARY / EDUCATION / SKILLS / EXPERIENCE / PROJECTS / CERTIFICATIONS
headers, matching utils/resume_parser.py's section detector) instead of a
single flat paragraph. Sections are randomly omitted (not everyone lists
certifications), skill coverage is noisier, and phrasing varies per row —
this makes classification meaningfully harder than a toy 100%-separable
problem, and lets section-aware ATS scoring/suggestions be tested
properly against realistic input structure.
"""
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

ROLE_SKILLS = {
    "AI Engineer": ["python", "tensorflow", "pytorch", "machine learning", "deep learning",
                     "nlp", "computer vision", "sql", "git", "docker", "keras", "opencv"],
    "Data Scientist": ["python", "pandas", "numpy", "sql", "machine learning", "statistics",
                        "tableau", "power bi", "scikit-learn", "r", "excel", "matplotlib"],
    "Software Engineer": ["java", "c++", "python", "data structures", "algorithms", "git",
                           "sql", "oop", "system design", "linux", "rest api", "junit"],
    "Frontend Developer": ["html", "css", "javascript", "react", "typescript", "redux",
                            "figma", "webpack", "git", "responsive design", "next.js"],
    "Backend Developer": ["python", "java", "node.js", "sql", "mongodb", "django", "flask",
                           "rest api", "docker", "microservices", "redis", "git"],
    "Cloud Engineer": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux",
                        "ci/cd", "python", "networking", "cloudformation"],
    "DevOps Engineer": ["docker", "kubernetes", "jenkins", "ci/cd", "linux", "ansible",
                         "terraform", "aws", "git", "bash", "monitoring", "prometheus"],
    "Cybersecurity Analyst": ["network security", "penetration testing", "siem", "python",
                               "firewalls", "linux", "cryptography", "incident response",
                               "vulnerability assessment", "wireshark"],
    "Database Administrator": ["sql", "mysql", "postgresql", "oracle", "mongodb", "backup",
                                "indexing", "query optimization", "linux", "etl"],
    "Business Analyst": ["excel", "sql", "power bi", "tableau", "requirements gathering",
                          "stakeholder management", "agile", "jira", "data analysis"],
    "UI/UX Designer": ["figma", "adobe xd", "sketch", "wireframing", "prototyping",
                        "user research", "html", "css", "design systems", "usability testing"],
}
ALL_SKILLS = sorted({s for skills in ROLE_SKILLS.values() for s in skills})

SUMMARY_HEADERS = ["SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "PROFILE"]
EDUCATION_HEADERS = ["EDUCATION", "ACADEMIC BACKGROUND"]
SKILLS_HEADERS = ["SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES"]
EXPERIENCE_HEADERS = ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE"]
PROJECTS_HEADERS = ["PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS"]
CERT_HEADERS = ["CERTIFICATIONS", "CERTIFICATES"]

SUMMARY_TEMPLATES = [
    "Motivated {role} with hands-on experience in {s1}, {s2} and {s3}. Built projects applying {s4} to solve real-world problems.",
    "Detail-oriented professional skilled in {s1} and {s2}, seeking a {role} role. Completed coursework and internships involving {s3} and {s4}.",
    "{role} candidate with a strong foundation in {s1}, {s2}, {s3}. Passionate about {s4} and continuous learning.",
    "Results-driven {role} with {yrs} years of practical exposure to {s1} and {s2}. Enjoys solving problems with {s3}.",
]
EXPERIENCE_TEMPLATES = [
    "{title}, {company} ({yrs} yrs)\n- Built and maintained systems using {s1} and {s2}.\n- Collaborated with a small team to ship features on schedule.",
    "{title}, {company} ({yrs} yrs)\n- Automated workflows involving {s1}, reducing manual effort.\n- Used {s2} daily to support ongoing projects.",
    "{title} Intern, {company} ({yrs} yrs)\n- Assisted senior engineers with tasks involving {s1} and {s3}.\n- Learned {s2} on the job through mentorship.",
]
PROJECT_TEMPLATES = [
    "{proj_name}\nBuilt using {s1} and {s2}. Focused on solving a practical problem and deploying a working demo.",
    "{proj_name}\nApplied {s1} to analyze data and {s2} to build the final solution, presenting results to peers.",
]
COMPANIES = ["Acme Solutions", "BrightPath Labs", "NovaTech", "Skyline Systems", "Quantum Analytics",
             "Vertex Digital", "Northwind Software", "Cascade Consulting"]
PROJ_NAMES = ["Personal Portfolio System", "Data Insights Dashboard", "Automation Toolkit",
              "Recommendation Prototype", "Campus Utility App", "Mini Analytics Engine"]
DEGREES = ["Bachelors", "Masters", "Bachelors", "PhD", "Diploma"]


def _fmt(template, **kw):
    return template.format(**kw)


def make_resume_row(role: str):
    role_skill_pool = ROLE_SKILLS[role]
    n_role_skills = random.randint(4, len(role_skill_pool))  # noisier: sometimes low coverage
    picked = random.sample(role_skill_pool, n_role_skills)
    n_noise = random.randint(0, 4)
    noise = random.sample(ALL_SKILLS, min(n_noise, len(ALL_SKILLS)))
    skills = list(dict.fromkeys(picked + noise))
    random.shuffle(skills)

    filler = (picked + ["problem solving", "teamwork", "communication"])
    s1, s2, s3, s4 = (filler + filler)[:4]
    years_exp = round(np.random.exponential(2.0), 1)
    education = random.choice(DEGREES)

    sections = []

    # SUMMARY - present ~85% of the time
    has_summary = random.random() < 0.85
    if has_summary:
        header = random.choice(SUMMARY_HEADERS)
        body = random.choice(SUMMARY_TEMPLATES).format(role=role, s1=s1, s2=s2, s3=s3, s4=s4, yrs=years_exp)
        sections.append(f"{header}\n{body}")

    # EDUCATION - always present
    edu_header = random.choice(EDUCATION_HEADERS)
    sections.append(f"{edu_header}\n{education} degree, relevant coursework in {random.choice(role_skill_pool)}.")

    # SKILLS - always present
    skills_header = random.choice(SKILLS_HEADERS)
    sections.append(f"{skills_header}\n{', '.join(skills)}")

    # EXPERIENCE - present ~70% of the time
    has_experience = random.random() < 0.70
    if has_experience:
        exp_header = random.choice(EXPERIENCE_HEADERS)
        body = random.choice(EXPERIENCE_TEMPLATES).format(
            title=role, company=random.choice(COMPANIES), yrs=max(years_exp, 0.5), s1=s1, s2=s2, s3=s3)
        sections.append(f"{exp_header}\n{body}")

    # PROJECTS - present ~75% of the time
    has_projects = random.random() < 0.75
    if has_projects:
        proj_header = random.choice(PROJECTS_HEADERS)
        n_proj = random.randint(1, 2)
        proj_bodies = []
        for _ in range(n_proj):
            proj_bodies.append(random.choice(PROJECT_TEMPLATES).format(
                proj_name=random.choice(PROJ_NAMES), s1=s1, s2=s2))
        sections.append(f"{proj_header}\n" + "\n\n".join(proj_bodies))

    # CERTIFICATIONS - present ~40% of the time
    has_certifications = random.random() < 0.40
    if has_certifications:
        cert_header = random.choice(CERT_HEADERS)
        sections.append(f"{cert_header}\nRelevant certification related to {random.choice(role_skill_pool)}.")

    resume_text = "\n\n".join(sections)
    num_sections = len(sections)

    coverage = len(set(picked)) / len(role_skill_pool)
    score = (
        coverage * 55
        + min(years_exp / 5, 1) * 15
        + int(has_projects) * 12
        + int(has_certifications) * 8
        + min(num_sections / 6, 1) * 10
    )
    score = round(min(max(score + np.random.normal(0, 5), 5), 100), 1)

    return {
        "resume_text": resume_text,
        "skills": ", ".join(skills),
        "num_skills": len(skills),
        "years_experience": years_exp,
        "education": education,
        "has_projects": int(has_projects),
        "has_certifications": int(has_certifications),
        "num_sections": num_sections,
        "resume_score": score,
        "role": role,
    }


def build_dataset(n_per_role: int = 120) -> pd.DataFrame:
    rows = [make_resume_row(role) for role in ROLE_SKILLS for _ in range(n_per_role)]
    df = pd.DataFrame(rows)
    dup = df.sample(15, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    for col in ["years_experience", "education"]:
        idx = df.sample(frac=0.02, random_state=2).index
        df.loc[idx, col] = np.nan
    return df.sample(frac=1, random_state=3).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("datasets/resumes_dataset.csv", index=False)
    print(f"Saved {len(df)} rows -> datasets/resumes_dataset.csv")
    print(df["role"].value_counts())
    print("\nSample resume:\n", df.iloc[0]["resume_text"])
