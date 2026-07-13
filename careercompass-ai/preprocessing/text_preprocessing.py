"""
text_preprocessing.py
----------------------
WHY: Resumes are free-form text. Before any ML model can use them we must
turn text into clean, comparable numerical features.

UPGRADE (v2): We now use SENTENCE EMBEDDINGS (sentence-transformers,
model: all-MiniLM-L6-v2) instead of TF-IDF as the primary feature
extractor. TF-IDF only matches exact overlapping words, so "developed ML
systems" and "built machine learning models" look unrelated to it.
Embeddings are trained to place semantically similar sentences near each
other in vector space, so the model now understands meaning, not just
vocabulary overlap. This directly improves career-role prediction quality
and (later) resume-vs-job-description matching.

TF-IDF is kept available below (build_tfidf) since it's still useful as a
fast, fully-interpretable fallback and for feature-importance charts.
"""
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words("english"))

# ---------------------------------------------------------------- embeddings
_embedding_model = None
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, 384-dim, great quality/speed tradeoff


def get_embedding_model():
    """Lazily load and cache the sentence-transformer model (downloads once, ~80MB)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def build_embeddings(texts) -> np.ndarray:
    """Encode a list/Series of resume texts into semantic embedding vectors."""
    model = get_embedding_model()
    texts = [str(t) for t in texts]
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ---------------------------------------------------------------- classic NLP cleaning (still used for
# TF-IDF fallback, skill-keyword matching, and resume-quality heuristics like action-verb detection)
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/numbers, tokenize, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stopwords and len(t) > 2]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Full preprocessing: dedupe, impute missing values, feature engineer, clean text."""
    df = pd.read_csv(csv_path)

    before = len(df)
    df = df.drop_duplicates(subset=["resume_text"]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    df["years_experience"] = df["years_experience"].fillna(df["years_experience"].median())
    df["education"] = df["education"].fillna("Bachelors")

    df["clean_text"] = df["resume_text"].apply(clean_text)

    edu_order = {"Diploma": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    df["education_encoded"] = df["education"].map(edu_order)

    return df


def build_tfidf(train_texts, max_features=300):
    """Kept as an optional fast/interpretable fallback (see README for when to use which)."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(train_texts)
    return X, vectorizer


def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


if __name__ == "__main__":
    df = load_and_clean("./datasets/resumes_dataset.csv")
    df.to_csv("./datasets/resumes_cleaned.csv", index=False)
    print(df[["resume_text", "clean_text"]].head(3))
    print(f"Final dataset shape: {df.shape}")
