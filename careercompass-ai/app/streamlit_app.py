"""
streamlit_app.py
-------------------
WHY STREAMLIT: For an ML-heavy showcase product, Streamlit lets us ship a
polished interactive dashboard directly in Python (no separate frontend
build step) while still supporting custom CSS for a modern dark theme,
Plotly charts, and file uploads.

UPGRADE (v2) additions wired in here:
  - Section-aware parsing feeds smarter ATS scoring + suggestions
  - Fuzzy skill matching (rapidfuzz) catches real-world skill phrasing
  - Job Description Match page: paste a real posting, get a direct score
  - Downloadable PDF report of any analysis
  - Local history (SQLite) tracking ATS score over repeated uploads this session
"""
import sys
import os
import uuid
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.resume_parser import extract_text, extract_sections
from utils.skill_extractor import extract_skills, missing_skills_for_role
from utils.recommender import (recommend_courses, recommend_certifications, recommend_projects,
                                interview_tips, career_roadmap, resume_suggestions, strengths_and_weaknesses)
from utils.ats_scorer import compute_ats_score
from utils.jd_matcher import analyze_against_jd
from utils.report_generator import build_report_pdf
from utils.storage import save_analysis, get_history
from preprocessing.text_preprocessing import clean_text, build_embeddings
from datasets.generate_dataset import ROLE_SKILLS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED = os.path.join(BASE, "saved_models")

st.set_page_config(page_title="CareerCompass AI", page_icon="🧭", layout="wide",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #E6E6E6; }
.cc-card {
    background: linear-gradient(145deg, #161B22, #1C2230);
    border: 1px solid #2A3140; border-radius: 14px;
    padding: 18px 22px; margin-bottom: 14px;
}
.cc-metric-big { font-size: 42px; font-weight: 700; color: #4C9AFF; }
.cc-badge {
    display:inline-block; padding: 4px 12px; border-radius: 20px;
    background: #1F6FEB33; color:#58A6FF; font-size: 13px; margin: 3px 4px 3px 0;
    border: 1px solid #1F6FEB55;
}
.cc-badge-missing {
    background:#F8514933; color:#FF7B72; border:1px solid #F8514955;
}
h1, h2, h3 { color: #F0F3F8 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{SAVED}/best_model.pkl")
    scaler = joblib.load(f"{SAVED}/numeric_scaler.pkl")
    label_encoder = joblib.load(f"{SAVED}/label_encoder.pkl")
    return model, scaler, label_encoder


def predict_role(text: str, numeric_feats: list, model, scaler, label_encoder):
    embed = build_embeddings([text])
    numeric_scaled = scaler.transform([numeric_feats])
    X = np.hstack([embed, numeric_scaled])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return label_encoder.inverse_transform([pred])[0], proba, label_encoder.classes_


def skill_radar_chart(found_skills, role_skills):
    categories = role_skills[:8]
    values = [1 if s in found_skills else 0 for s in categories]
    values += values[:1]; categories = categories + [categories[0]] if categories else categories
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself',
                                   line_color="#4C9AFF", fillcolor="rgba(76,154,255,0.35)"))
    fig.update_layout(polar=dict(bgcolor="#161B22", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                       showlegend=False, paper_bgcolor="#0E1117", font_color="#E6E6E6", height=380,
                       margin=dict(l=40, r=40, t=30, b=30))
    return fig


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# ---------------------------------------------------------------- sidebar nav
st.sidebar.markdown("## 🧭 CareerCompass AI")
page = st.sidebar.radio("Navigate", ["Home", "Upload Resume", "Results", "Career Prediction",
                                      "Skill Analysis", "Job Description Match", "Course Recommendation",
                                      "History", "About", "Contact"])

# ---------------------------------------------------------------- HOME
if page == "Home":
    st.title("🧭 CareerCompass AI")
    st.subheader("Intelligent Resume Analyzer & Career Recommendation Platform")
    st.write("Upload your resume and get an AI-powered breakdown: ATS score, detected vs. missing skills, "
             "a predicted best-fit career role, a direct match score against a real job posting, course/"
             "certification recommendations, and a 6-month roadmap.")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip([c1, c2, c3, c4],
                                ["Roles Modeled", "Skills Tracked", "ML Models Compared", "Resumes Trained On"],
                                ["11", "70+", "7", "1,300+"]):
        with col:
            st.markdown(f'<div class="cc-card"><div class="cc-metric-big">{val}</div>{label}</div>',
                        unsafe_allow_html=True)
    st.info("👉 Go to **Upload Resume** in the sidebar to get started.")

# ---------------------------------------------------------------- UPLOAD
elif page == "Upload Resume":
    st.title("📤 Upload Resume")
    uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    target_role = st.selectbox("Target job role (for skill-gap analysis)", list(ROLE_SKILLS.keys()))

    if uploaded and st.button("Analyze Resume 🚀", type="primary"):
        with st.spinner("Extracting text and running ML pipeline..."):
            try:
                text = extract_text(uploaded.read(), uploaded.name)
                if len(text.strip()) < 30:
                    st.error("Couldn't extract meaningful text from this file. Try another file.")
                else:
                    model, scaler, label_encoder = load_artifacts()
                    sections = extract_sections(text)
                    found_skills = extract_skills(text)
                    missing = missing_skills_for_role(found_skills, target_role, ROLE_SKILLS)

                    has_projects = bool(sections.get("projects", "").strip())
                    has_certifications = bool(sections.get("certifications", "").strip())
                    num_sections = len(sections)
                    years_exp_guess = 2.0

                    numeric_feats = [len(found_skills), years_exp_guess, int(has_projects),
                                      int(has_certifications), num_sections, 1]
                    pred_role, proba, classes = predict_role(text, numeric_feats, model, scaler, label_encoder)

                    ats_score, breakdown = compute_ats_score(
                        text, found_skills, ROLE_SKILLS[target_role], sections, num_sections)

                    strengths, weaknesses = strengths_and_weaknesses(found_skills, missing, ats_score)
                    suggestions = resume_suggestions(sections, len(found_skills))

                    st.session_state.analysis = dict(
                        text=text, sections=sections, target_role=target_role, found_skills=found_skills,
                        missing=missing, pred_role=pred_role, proba=proba, classes=classes, ats_score=ats_score,
                        breakdown=breakdown, strengths=strengths, weaknesses=weaknesses, suggestions=suggestions,
                        jd_analysis=None)
                    save_analysis(st.session_state.session_id, st.session_state.analysis)
                    st.success("Analysis complete! Open the **Results** tab.")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------------------- RESULTS
elif page == "Results":
    st.title("📊 Results")
    a = st.session_state.analysis
    if not a:
        st.warning("Upload and analyze a resume first (see **Upload Resume**).")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f'<div class="cc-card"><h3>ATS Score</h3>'
                        f'<div class="cc-metric-big">{a["ats_score"]}/100</div></div>', unsafe_allow_html=True)
            st.progress(min(int(a["ats_score"]), 100))
            pdf_bytes = build_report_pdf(a)
            st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                file_name="careercompass_report.pdf", mime="application/pdf")
        with c2:
            st.markdown("#### Score Breakdown")
            bdf = pd.DataFrame(list(a["breakdown"].items()), columns=["Component", "Points"])
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Strengths")
            for s in a["strengths"]:
                st.markdown(f"- {s}")
        with col2:
            st.markdown("#### ⚠️ Weaknesses")
            for w in a["weaknesses"]:
                st.markdown(f"- {w}")

        st.markdown("#### 📝 Resume Improvement Suggestions")
        for s in a["suggestions"]:
            st.markdown(f"- {s}")

# ---------------------------------------------------------------- CAREER PREDICTION
elif page == "Career Prediction":
    st.title("🎯 Career Prediction")
    a = st.session_state.analysis
    if not a:
        st.warning("Upload and analyze a resume first.")
    else:
        st.markdown(f'<div class="cc-card"><h3>Best-Fit Role</h3>'
                    f'<div class="cc-metric-big">{a["pred_role"]}</div></div>', unsafe_allow_html=True)
        if a["proba"] is not None:
            prob_df = pd.DataFrame({"Role": a["classes"], "Confidence": a["proba"]}).sort_values(
                "Confidence", ascending=True)
            fig = px.bar(prob_df, x="Confidence", y="Role", orientation="h", color="Confidence",
                         color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="#E6E6E6", height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🗺️ Personalized 6-Month Career Roadmap")
        for step in career_roadmap(a["target_role"], a["missing"]):
            st.markdown(f"- {step}")

        st.markdown("#### 🎤 Interview Preparation Tips")
        for tip in interview_tips(a["target_role"]):
            st.markdown(f"- {tip}")

# ---------------------------------------------------------------- SKILL ANALYSIS
elif page == "Skill Analysis":
    st.title("🧩 Skill Analysis")
    a = st.session_state.analysis
    if not a:
        st.warning("Upload and analyze a resume first.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Detected Skills")
            st.markdown("".join(f'<span class="cc-badge">{s}</span>' for s in a["found_skills"]) or "None found",
                        unsafe_allow_html=True)
            st.markdown("#### Missing Skills for target role")
            st.markdown("".join(f'<span class="cc-badge cc-badge-missing">{s}</span>' for s in a["missing"]) or "None — great coverage!",
                        unsafe_allow_html=True)
        with col2:
            st.plotly_chart(skill_radar_chart(a["found_skills"], ROLE_SKILLS[a["target_role"]]),
                            use_container_width=True)

        st.markdown("#### 💡 Suggested Portfolio Projects")
        for p in recommend_projects(a["pred_role"]):
            st.markdown(f"- {p}")
        st.markdown("#### 🏅 Recommended Certifications")
        for c in recommend_certifications(a["pred_role"]):
            st.markdown(f"- {c}")

# ---------------------------------------------------------------- JOB DESCRIPTION MATCH
elif page == "Job Description Match":
    st.title("📋 Job Description Match")
    a = st.session_state.analysis
    if not a:
        st.warning("Upload and analyze a resume first (see **Upload Resume**).")
    else:
        st.write("Paste a real job posting below to see exactly how your resume matches *this specific job* — "
                 "semantic similarity plus a skill-by-skill gap, instead of a generic role template.")
        jd_text = st.text_area("Paste job description here", height=220,
                                placeholder="e.g. paste the full text of a LinkedIn/Indeed job posting...")
        if st.button("Match Against This Job 🔍", type="primary") and jd_text.strip():
            with st.spinner("Comparing resume against job description..."):
                jd_result = analyze_against_jd(a["text"], jd_text)
                a["jd_analysis"] = jd_result
                st.session_state.analysis = a

        if a.get("jd_analysis"):
            jd = a["jd_analysis"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="cc-card"><h3>Semantic Match</h3>'
                            f'<div class="cc-metric-big">{jd["semantic_match_pct"]}%</div>'
                            f'How closely your resume\'s overall meaning matches this posting</div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="cc-card"><h3>Skill Coverage</h3>'
                            f'<div class="cc-metric-big">{jd["skill_coverage_pct"]}%</div>'
                            f'Of the explicit skills this posting mentions</div>', unsafe_allow_html=True)

            st.markdown("#### ✅ Matched Skills")
            st.markdown("".join(f'<span class="cc-badge">{s}</span>' for s in jd["matched"]) or "None found",
                        unsafe_allow_html=True)
            st.markdown("#### ❌ Missing Skills for This Posting")
            st.markdown("".join(f'<span class="cc-badge cc-badge-missing">{s}</span>' for s in jd["missing"]) or "None — full coverage!",
                        unsafe_allow_html=True)
            if jd["extra"]:
                st.markdown("#### ➕ Skills You Have That the Posting Doesn't Mention")
                st.markdown("".join(f'<span class="cc-badge">{s}</span>' for s in jd["extra"]), unsafe_allow_html=True)

# ---------------------------------------------------------------- COURSE RECOMMENDATION
elif page == "Course Recommendation":
    st.title("🎓 Course Recommendation")
    a = st.session_state.analysis
    if not a:
        st.warning("Upload and analyze a resume first.")
    else:
        courses = recommend_courses(a["missing"])
        for name, link in courses:
            st.markdown(f'<div class="cc-card"><b>{name}</b><br><span style="color:#8B949E">{link}</span></div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- HISTORY
elif page == "History":
    st.title("📈 Analysis History")
    st.caption("Tracked for this browser session. Upload multiple resumes (or re-upload after edits) to see your ATS score trend.")
    history = get_history(st.session_state.session_id)
    if not history:
        st.info("No analyses yet this session — upload a resume to get started.")
    else:
        hdf = pd.DataFrame(history)[["timestamp", "target_role", "predicted_role", "ats_score", "num_skills", "num_missing"]]
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        if len(hdf) > 1:
            fig = px.line(hdf.iloc[::-1], x="timestamp", y="ats_score", markers=True, title="ATS Score Over Time")
            fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", font_color="#E6E6E6")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- ABOUT
elif page == "About":
    st.title("ℹ️ About CareerCompass AI")
    st.write("""
    CareerCompass AI is an end-to-end machine learning platform that analyzes resumes and
    recommends careers. It combines NLP preprocessing (sentence embeddings), supervised
    learning (Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM),
    unsupervised learning (K-Means + PCA), direct job-description matching, and a rule-based
    recommendation engine into one deployable Streamlit application.
    """)
    st.markdown("Built as a demonstration of a full ML product lifecycle: data generation, "
                "preprocessing, model training/evaluation, ensemble comparison, and deployment.")

# ---------------------------------------------------------------- CONTACT
elif page == "Contact":
    st.title("📬 Contact")
    st.write("Have feedback or want to collaborate?")
    st.text_input("Your name")
    st.text_input("Your email")
    st.text_area("Message")
    if st.button("Send (demo only)"):
        st.success("Thanks! (This is a demo form — wire it to an email/DB backend for production.)")
