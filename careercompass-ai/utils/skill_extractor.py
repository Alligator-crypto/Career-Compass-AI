"""
skill_extractor.py
--------------------
WHY: Career prediction and missing-skill analysis both depend on knowing
exactly which skills appear in a resume. We use a curated skill taxonomy
+ matching (rather than pure ML) because skill names are a known, finite
vocabulary — precision matters more than recall, and dictionary matching
is 100% explainable to the end user ("why did you say I have Python?").

UPGRADE (v2): added FUZZY matching (rapidfuzz) alongside exact matching so
real-world variants like "ReactJS", "React.js", "Node JS", "Sklearn" still
correctly map to the canonical taxonomy entries ("react", "node.js",
"scikit-learn"), instead of being silently missed by strict word-boundary
regex the way v1 was.
"""
import re
from rapidfuzz import fuzz

SKILL_TAXONOMY = [
    "python", "java", "c++", "c#", "javascript", "typescript", "sql", "r", "go", "rust",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "keras", "scikit-learn", "opencv", "pandas", "numpy",
    "html", "css", "react", "angular", "vue", "next.js", "redux", "node.js", "django",
    "flask", "fastapi", "rest api", "graphql", "mongodb", "mysql", "postgresql", "oracle",
    "redis", "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "jenkins", "ci/cd", "git", "github", "linux", "bash", "microservices",
    "tableau", "power bi", "excel", "statistics", "data analysis", "etl",
    "figma", "adobe xd", "sketch", "wireframing", "prototyping", "user research",
    "network security", "penetration testing", "siem", "firewalls", "cryptography",
    "agile", "scrum", "jira", "junit", "system design", "data structures", "algorithms",
    "oop", "xgboost", "lightgbm", "spark", "hadoop", "airflow", "kafka", "snowflake",
    "a/b testing", "hypothesis testing", "data visualization", "seaborn", "matplotlib",
]

# common real-world spelling/spacing variants -> canonical taxonomy name
SKILL_ALIASES = {
    "reactjs": "react", "react.js": "react", "react js": "react",
    "nodejs": "node.js", "node js": "node.js",
    "nextjs": "next.js", "next js": "next.js",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "tf": "tensorflow", "pytorch lightning": "pytorch",
    "postgres": "postgresql", "mongo": "mongodb",
    "k8s": "kubernetes", "vuejs": "vue", "vue.js": "vue",
    "cicd": "ci/cd", "ci cd": "ci/cd",
    "js": "javascript", "ts": "typescript",
    "ml": "machine learning", "dl": "deep learning", "cv": "computer vision",
    "aws cloud": "aws", "google cloud": "gcp", "microsoft azure": "azure",
}

FUZZY_THRESHOLD = 88  # rapidfuzz token_sort_ratio, 0-100; conservative to avoid false positives


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_skills(text: str) -> list:
    """Return the sorted list of taxonomy skills found in `text`.
    Combines exact word-boundary matching, alias lookup, and fuzzy matching
    on short candidate phrases (1-2 words) so real resume phrasing variance
    doesn't cause false negatives."""
    text_norm = _normalize(text)
    found = set()

    # 1. exact word-boundary match against taxonomy
    for skill in SKILL_TAXONOMY:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, text_norm):
            found.add(skill)

    # 2. alias match
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, text_norm):
            found.add(canonical)

    # 3. fuzzy match on candidate 1-2 word phrases (catches typos/odd spacing)
    words = re.findall(r"[a-z0-9\+\#\.]+", text_norm)
    candidates = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
    for cand in candidates:
        if len(cand) < 3:
            continue
        for skill in SKILL_TAXONOMY:
            if skill in found:
                continue
            if fuzz.ratio(cand, skill) >= FUZZY_THRESHOLD:
                found.add(skill)
                break

    return sorted(found)


def missing_skills_for_role(found_skills: list, role: str, role_skills_map: dict) -> list:
    """Diff resume skills against the required skill set for a target role."""
    required = set(role_skills_map.get(role, []))
    return sorted(required - set(found_skills))
