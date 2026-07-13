# Deployment & Sharing Guide

## Deploy (Streamlit Community Cloud — free)
1. Push this repo to GitHub (steps below).
2. Go to share.streamlit.io → "New app" → select your repo/branch.
3. Set main file path to `app/streamlit_app.py`.
4. Add `requirements.txt` (already included) — Streamlit Cloud installs it automatically.
5. Deploy. You'll get a public `*.streamlit.app` URL.

*(Docker/AWS/Render also work — containerize with a simple `Dockerfile` running `streamlit run app/streamlit_app.py --server.port $PORT`.)*

## Upload to GitHub
```bash
cd careercompass-ai
git init
git add .
git commit -m "Initial commit: CareerCompass AI"
git branch -M main
git remote add origin https://github.com/<your-username>/careercompass-ai.git
git push -u origin main
```
Add a `.gitignore` with `__pycache__/`, `*.pyc`, and optionally exclude large `saved_models/*.pkl` if you'd rather regenerate them via `python main.py`.

## LinkedIn Post
> 🧭 Excited to share my latest project: **CareerCompass AI** — an end-to-end ML platform that analyzes resumes and recommends careers.
>
> Upload a resume (PDF/DOCX/TXT) and instantly get: an ATS compatibility score, detected vs. missing skills, a predicted best-fit career role, a direct match score against a real job posting you paste in, personalized course/certification recommendations, a downloadable PDF report, and a 6-month roadmap.
>
> Built the full pipeline myself: NLP preprocessing with sentence embeddings, 7 supervised models compared (Logistic Regression → XGBoost/LightGBM), K-Means clustering + PCA visualization, full evaluation suite (ROC-AUC, confusion matrices, learning curves), fuzzy skill matching, section-aware resume parsing, and a Streamlit dashboard with session history tracking.
>
> #MachineLearning #NLP #DataScience #Python #CareerTech #Portfolio
>
> 🔗 GitHub: [link] | 🚀 Live demo: [link]
