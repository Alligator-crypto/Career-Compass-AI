"""
recommender.py
-----------------
WHY: Once we know a user's detected skills, missing skills, and predicted
role, this module turns that into actionable guidance — the part of the
product that actually helps a student improve, not just get scored.
This is deliberately rule-based (curated mappings) rather than ML-based:
recommendation quality here depends on curated, trustworthy content
(real course names, real certs) which a classifier cannot invent reliably.

UPGRADE (v2): resume_suggestions() is now section-aware (uses the actual
parsed sections dict instead of guessing from whole-text keyword search),
giving more specific, actionable feedback (e.g. "Your Experience section
is only 8 words — add real detail" instead of a generic "add more detail").
"""
import sys
sys.path.append("/home/claude/careercompass-ai")
from datasets.generate_dataset import ROLE_SKILLS

COURSES = {
    "python": [("Python for Everybody — Coursera", "coursera.org"), ("CS50P — Harvard", "cs50.harvard.edu")],
    "machine learning": [("Machine Learning Specialization — Coursera (Andrew Ng)", "coursera.org"),
                          ("Kaggle Learn: Intro to ML", "kaggle.com/learn")],
    "deep learning": [("Deep Learning Specialization — Coursera", "coursera.org")],
    "tensorflow": [("TensorFlow Developer Certificate Prep — Coursera", "coursera.org")],
    "sql": [("SQL for Data Science — Coursera", "coursera.org"), ("Kaggle Learn: SQL", "kaggle.com/learn")],
    "react": [("Meta Front-End Developer — Coursera", "coursera.org"), ("freeCodeCamp: React", "freecodecamp.org")],
    "aws": [("AWS Cloud Practitioner Essentials", "aws.amazon.com/training"), ("AWS on Coursera", "coursera.org")],
    "docker": [("Docker & Kubernetes: The Practical Guide — Udemy", "udemy.com")],
    "kubernetes": [("Kubernetes for the Absolute Beginners — Udemy", "udemy.com")],
    "network security": [("Google Cybersecurity Certificate — Coursera", "coursera.org")],
    "power bi": [("Microsoft Power BI — Microsoft Learn", "learn.microsoft.com")],
    "excel": [("Excel Skills for Business — Coursera", "coursera.org")],
    "figma": [("Google UX Design Certificate — Coursera", "coursera.org")],
    "javascript": [("The Odin Project", "theodinproject.com"), ("freeCodeCamp: JS Algorithms", "freecodecamp.org")],
    "spark": [("Big Data Analysis with Spark — Coursera", "coursera.org")],
    "airflow": [("Data Engineering with Airflow — Udemy", "udemy.com")],
    "a/b testing": [("A/B Testing — Udacity/Coursera", "coursera.org")],
}
DEFAULT_COURSES = [("Google Career Certificates", "grow.google"), ("freeCodeCamp", "freecodecamp.org")]

CERTIFICATIONS = {
    "AI Engineer": ["TensorFlow Developer Certificate", "AWS Certified Machine Learning – Specialty"],
    "Data Scientist": ["Google Data Analytics Certificate", "Microsoft Certified: Azure Data Scientist Associate"],
    "Software Engineer": ["Oracle Certified Professional: Java SE", "AWS Certified Developer"],
    "Frontend Developer": ["Meta Front-End Developer Certificate"],
    "Backend Developer": ["Meta Back-End Developer Certificate"],
    "Cloud Engineer": ["AWS Certified Solutions Architect", "Microsoft Azure Fundamentals (AZ-900)"],
    "DevOps Engineer": ["Certified Kubernetes Administrator (CKA)", "AWS Certified DevOps Engineer"],
    "Cybersecurity Analyst": ["CompTIA Security+", "Google Cybersecurity Certificate"],
    "Database Administrator": ["Oracle Database SQL Certified Associate", "Microsoft Azure Database Administrator"],
    "Business Analyst": ["Google Data Analytics Certificate", "CBAP (Certified Business Analysis Professional)"],
    "UI/UX Designer": ["Google UX Design Certificate", "NN/g UX Certification"],
}

PROJECTS = {
    "AI Engineer": ["Image classifier with a CNN (PyTorch/TensorFlow)", "Chatbot using an NLP transformer model"],
    "Data Scientist": ["End-to-end EDA + prediction project on a Kaggle dataset", "Interactive dashboard (Tableau/Power BI) from real data"],
    "Software Engineer": ["REST API with authentication (Flask/Django)", "CLI tool solving a real workflow problem"],
    "Frontend Developer": ["Responsive portfolio site (React)", "Clone of a popular app UI (e.g. Spotify) with React"],
    "Backend Developer": ["Microservice with Docker + database", "E-commerce backend with payment integration (sandbox)"],
    "Cloud Engineer": ["Deploy a 3-tier app on AWS with Terraform", "CI/CD pipeline with auto-scaling"],
    "DevOps Engineer": ["Kubernetes cluster with Jenkins CI/CD pipeline", "Infrastructure-as-Code project (Terraform + Ansible)"],
    "Cybersecurity Analyst": ["Home-lab penetration testing writeup (legal targets)", "SIEM dashboard for log-based threat detection"],
    "Database Administrator": ["Design & optimize a normalized schema for a real dataset", "Backup/recovery automation script"],
    "Business Analyst": ["Business requirements doc + dashboard for a case study", "Process automation using SQL + Power BI"],
    "UI/UX Designer": ["Full case study: research -> wireframes -> hi-fi prototype (Figma)", "Redesign of a well-known app with rationale"],
}

INTERVIEW_TIPS = {
    "AI Engineer": ["Be ready to explain bias-variance tradeoff and overfitting.",
                    "Practice explaining a project's model choice and evaluation metrics.",
                    "Review gradient descent, backpropagation, and common loss functions."],
    "Data Scientist": ["Practice SQL joins/window functions and pandas manipulation live.",
                        "Be ready to explain a p-value and A/B testing in plain English.",
                        "Prepare a story: how you turned a messy dataset into an insight."],
    "Software Engineer": ["Practice data structures & algorithms (arrays, trees, graphs).",
                           "Review time/space complexity (Big-O) for your solutions.",
                           "Be ready to discuss system design at a beginner level."],
    "DEFAULT": ["Research the company and role thoroughly beforehand.",
                "Prepare 2-3 STAR-format stories about real projects.",
                "Have thoughtful questions ready for the interviewer."],
}


def recommend_courses(missing_skills: list, limit=6):
    recs = []
    for skill in missing_skills:
        for course in COURSES.get(skill, []):
            if course not in recs:
                recs.append(course)
    if not recs:
        recs = DEFAULT_COURSES
    return recs[:limit]


def recommend_certifications(role: str):
    return CERTIFICATIONS.get(role, ["Google Career Certificates (relevant track)"])


def recommend_projects(role: str):
    return PROJECTS.get(role, ["Build & document 2-3 portfolio projects relevant to your target role."])


def interview_tips(role: str):
    return INTERVIEW_TIPS.get(role, INTERVIEW_TIPS["DEFAULT"])


def career_roadmap(role: str, missing_skills: list):
    steps = [
        f"Month 1: Learn foundational tools for {role} — start with: {', '.join(missing_skills[:2]) or 'polish existing core skills'}.",
        f"Month 2: Build 1 portfolio project using: {', '.join(missing_skills[2:4]) or 'your strongest current skills'}.",
        f"Month 3: Earn a relevant certification: {recommend_certifications(role)[0]}.",
        "Month 4: Contribute to an open-source repo or publish your project on GitHub.",
        "Month 5: Mock interviews + tailor resume/LinkedIn to target job postings.",
        f"Month 6: Apply actively to {role} roles, iterate resume based on feedback/ATS results.",
    ]
    return steps


def resume_suggestions(sections: dict, num_skills: int):
    """Section-aware suggestions: reasons about the actual parsed sections
    instead of guessing from whole-resume keyword search."""
    suggestions = []
    full_text = " ".join(sections.values())
    word_count = len(full_text.split())

    if word_count < 150:
        suggestions.append("Resume looks short overall — add more detail on responsibilities and measurable outcomes.")
    if num_skills < 6:
        suggestions.append("List more relevant technical skills explicitly in a dedicated 'Skills' section.")

    projects_text = sections.get("projects", "")
    if not projects_text:
        suggestions.append("Add a 'Projects' section — even 2-3 concrete projects boost credibility strongly.")
    elif len(projects_text.split()) < 15:
        suggestions.append("Your Projects section is thin — add 1-2 sentences per project on what you built and its impact.")

    experience_text = sections.get("experience", "")
    if experience_text and len(experience_text.split()) < 15:
        suggestions.append("Your Experience section is quite short — expand on responsibilities and results.")

    if not sections.get("certifications"):
        suggestions.append("Consider adding relevant certifications to strengthen ATS keyword matches.")

    action_verbs = ["led", "built", "designed", "improved", "automated", "reduced", "launched", "optimized"]
    if not any(v in full_text.lower() for v in action_verbs):
        suggestions.append("Use strong action verbs (Built, Led, Automated, Improved) instead of passive phrasing.")
    if not any(ch.isdigit() for ch in full_text):
        suggestions.append("Quantify achievements with numbers (e.g. 'reduced load time by 30%').")
    if not sections.get("summary"):
        suggestions.append("Add a short professional summary at the top (2-3 sentences).")

    if not suggestions:
        suggestions.append("Resume structure looks solid — focus on tailoring keywords per job description.")
    return suggestions


def strengths_and_weaknesses(found_skills: list, missing_skills: list, resume_score: float):
    strengths = [f"Demonstrated proficiency in {s}" for s in found_skills[:5]] or ["Clear structure present"]
    weaknesses = [f"No evidence of {s} — a commonly required skill" for s in missing_skills[:5]]
    if resume_score < 60:
        weaknesses.append("Overall ATS compatibility is below average for competitive roles.")
    if not weaknesses:
        weaknesses = ["No major gaps detected — focus on continuous skill deepening."]
    return strengths, weaknesses
