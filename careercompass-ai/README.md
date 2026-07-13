# 🧭 CareerCompass AI — Intelligent Resume Analyzer & Career Recommendation Platform

An end-to-end machine learning platform that analyzes a resume (PDF/DOCX/TXT) and returns an ATS score, detected/missing skills, a predicted best-fit career role, a direct match score against a real job posting, course & certification recommendations, suggested projects, interview tips, a personalized 6-month roadmap, a downloadable PDF report, and a session history of your score over time — all through an interactive Streamlit dashboard.

## Features
- Resume upload & parsing (PDF, DOCX, TXT) with **section-aware detection** (Summary, Education, Skills, Experience, Projects, Certifications)
- ATS-style Resume Score (0–100) with an explainable point breakdown, scored against actual section content
- Automatic technical skill extraction (70+ skill taxonomy) with **fuzzy + alias matching** (catches "ReactJS", "Sklearn", "Node JS", typos, etc.)
- Missing-skill detection vs. a chosen target role
- **Job Description Match**: paste a real job posting and get a direct semantic similarity score + skill-gap analysis against that specific posting, not just a generic role template
- Career / best-fit job role prediction (11 roles) via semantic sentence embeddings
- Course, certification, and portfolio-project recommendations
- Resume improvement suggestions (section-specific), strengths & weaknesses
- Interview preparation tips + personalized career roadmap
- **Downloadable PDF report** of any analysis
- **Session history**: SQLite-backed tracking of ATS score across repeated uploads, with a trend chart
- Skill radar chart, confidence bar chart, cluster/PCA visualizations

## Dataset
No internet access was available to pull Kaggle datasets directly in this build environment, so `datasets/generate_dataset.py` programmatically builds a **1,300+ row synthetic resume dataset** structured like realistic multi-section resumes (Summary/Education/Skills/Experience/Projects/Certifications headers, sections randomly omitted, noisy skill coverage) with a role → required-skills mapping.

**To use real data instead:** download a dataset such as
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- [Job Description Dataset (Kaggle)](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)

and map the columns to `resume_text` / `role`, then re-run `main.py`.

## ML Workflow
1. **Preprocessing** (`preprocessing/text_preprocessing.py`): dedupe, missing-value imputation, feature engineering, plus **sentence-transformer embeddings** (`all-MiniLM-L6-v2`) as the primary text representation.
2. **Supervised Learning** (`models/train_models.py`): Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, LightGBM — trained on embeddings + scaled engineered numeric features, 70/15/15 train/val/test split.
3. **Model Evaluation**: 5-fold cross-validation, Accuracy/Precision/Recall/F1, confusion matrix, ROC-AUC, feature-block importance, learning curves.
4. **Ensemble Learning**: Random Forest vs. XGBoost vs. LightGBM compared on accuracy, speed, and feature importance.
5. **Unsupervised Learning** (`models/clustering.py`): K-Means (k chosen via elbow + silhouette score) + PCA for 2D cluster visualization, also on embeddings.
6. **Job Description Matching** (`utils/jd_matcher.py`): cosine similarity between resume and pasted-JD embeddings, plus an explicit skill-set diff — reuses the same embedding model as career prediction.

> **Why embeddings over TF-IDF:** TF-IDF only matches exact overlapping words, so "developed ML systems" and "built machine learning models" look unrelated to it. Sentence embeddings place semantically similar text near each other in vector space, which improves both career-role prediction and JD matching.

## Results
Run `python main.py` to generate `reports/` with current numbers — this README doesn't hardcode results since they depend on the embedding model, which downloads on first run (see Installation).

## Project Structure
```
careercompass-ai/
├── datasets/              # dataset generator + generated CSVs
├── models/                # training + clustering scripts
├── saved_models/          # trained model, scaler, encoder, history.db (.pkl / .db)
├── preprocessing/         # NLP cleaning + embedding pipeline
├── utils/
│   ├── resume_parser.py     # PDF/DOCX/TXT extraction + section splitting
│   ├── skill_extractor.py   # taxonomy + fuzzy/alias skill matching
│   ├── ats_scorer.py        # section-aware 0-100 ATS score
│   ├── recommender.py       # courses, certs, projects, roadmap, suggestions
│   ├── jd_matcher.py        # resume vs. pasted job description matching
│   ├── report_generator.py  # PDF report export
│   └── storage.py           # SQLite session history
├── app/                   # Streamlit dashboard
├── reports/                # generated charts + metrics
├── requirements.txt
├── main.py                # runs the full pipeline
└── README.md
```

## Installation & Usage
```bash
git clone <your-repo-url>
cd careercompass-ai
pip install -r requirements.txt

# 1. Run the full ML pipeline (generates data, trains, evaluates, clusters)
#    First run downloads the sentence-transformers model (~80MB, one-time, needs internet)
python main.py

# 2. Launch the web app
streamlit run app/streamlit_app.py
```

## Honest Limitations (read before you present this)
- **Synthetic training data.** No Kaggle access in the build sandbox — swap in real data per the Dataset section above for production-grade numbers.
- **History is local/session-based.** `saved_models/history.db` is a local SQLite file; on ephemeral hosting (some free Streamlit Cloud tiers) it can reset on redeploy. Fine for a demo/portfolio; swap for a hosted DB for a real multi-user product.
- **No authentication.** History is keyed by a random per-browser-session id, not a real login.
- **Embedding model wasn't verified in the build sandbox** — it can reach PyPI but not Hugging Face, so the pipeline logic was pressure-tested with stand-in vectors instead of the real model. It downloads and works automatically the first time you run `main.py` with normal internet access.

## Future Improvements
- Swap in real Kaggle resume/job datasets for production-grade accuracy
- Real user accounts + hosted database instead of local SQLite
- Fine-tune ATS scoring weights against real recruiter feedback
- Deploy via Streamlit Community Cloud / Docker + a cloud provider

## License
MIT License — free to use and modify for educational/portfolio purposes.

## Author
Built as a capstone-style ML engineering project demonstrating the full lifecycle: data generation, NLP preprocessing, supervised + unsupervised learning, model evaluation, ensemble methods, and deployment.
