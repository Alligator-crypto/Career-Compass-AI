"""
train_models.py
-----------------
WHY: Core ML workflow (Experiments 4, 5, 7, 8). We train several classifiers
to predict the best-fit career `role` from a resume's SEMANTIC EMBEDDING
(sentence-transformers) + scaled engineered numeric features, evaluate
rigorously, then pick a winner via ensemble comparison.

UPGRADE (v2): Features are now sentence embeddings instead of TF-IDF (see
preprocessing/text_preprocessing.py for why). Numeric engineered features
(num_skills, years_experience, etc.) are now StandardScaler-normalized
before concatenation, since embeddings are already unit-normalized and raw
numeric features on a different scale would otherwise dominate distance-
and margin-based models (KNN, SVM).

ALGORITHMS & WHY EACH IS INCLUDED:
- Logistic Regression: fast, interpretable linear baseline.
- Decision Tree: captures non-linear feature interactions, easy to explain.
- Random Forest (ensemble/bagging): reduces overfitting vs a single tree.
- KNN: distance-based baseline; embeddings make this far more meaningful
  than it was on sparse TF-IDF vectors.
- SVM (RBF kernel): strong on dense, moderate-dimensional embedding space.
- XGBoost (gradient boosting): usually top accuracy on structured data.
- LightGBM (histogram-based boosting): faster training at similar accuracy.
"""
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_curve, auc, classification_report)
import xgboost as xgb
import lightgbm as lgb
import joblib
import sys
sys.path.append("/home/claude/careercompass-ai")
from preprocessing.text_preprocessing import load_and_clean, build_embeddings, encode_labels

REPORTS = "./reports"
SAVED = "./saved_models"

NUMERIC_COLS = ["num_skills", "years_experience", "has_projects",
                 "has_certifications", "num_sections", "education_encoded"]


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, eval_metric="mlogloss",
                                      random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1),
    }


def build_features(df):
    print("Encoding resumes into semantic embeddings (first run downloads the model, ~80MB)...")
    X_embed = build_embeddings(df["resume_text"])  # use raw text; the transformer handles casing/stopwords itself

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[NUMERIC_COLS].values)

    X = np.hstack([X_embed, X_numeric])
    y, label_encoder = encode_labels(df["role"])
    return X, y, scaler, label_encoder, X_embed.shape[1]


def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_val)

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "f1": round(f1_score(y_val, y_pred, average="macro", zero_division=0), 4),
        "train_time_sec": round(train_time, 3),
    }
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    metrics["cv_mean_accuracy"] = round(cv_scores.mean(), 4)
    metrics["cv_std"] = round(cv_scores.std(), 4)
    return metrics, model, y_pred


def plot_confusion_matrix(y_val, y_pred, label_encoder, model_name, path):
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_roc_curves(model, X_val, y_val, label_encoder, path):
    n_classes = len(label_encoder.classes_)
    y_val_bin = label_binarize(y_val, classes=range(n_classes))
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)
    else:
        y_score = model.decision_function(X_val)

    plt.figure(figsize=(9, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - One-vs-Rest (Best Model)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_feature_importance(model, embed_dim, path, top_n=20):
    """NOTE: with embeddings, individual dimensions aren't human-readable words like TF-IDF
    was. This chart now shows which *feature blocks* (semantic embedding vs. engineered
    numeric features like num_skills/years_experience) matter most, which is more honest
    and more useful than labeling raw embedding dimensions."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return
    embed_importance = importances[:embed_dim].sum()
    numeric_importance = importances[embed_dim:]

    labels = ["Semantic embedding\n(overall resume meaning)"] + NUMERIC_COLS
    values = [embed_importance] + list(numeric_importance)

    plt.figure(figsize=(8, 6))
    plt.barh(labels, values, color="#4C9AFF")
    plt.title("Feature Block Importance (embedding vs. engineered features)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_learning_curve(model, X_train, y_train, path):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=3, train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy")
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation score")
    plt.xlabel("Training examples"); plt.ylabel("Accuracy")
    plt.title("Learning Curve (Best Model)")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def plot_model_comparison(results_df, path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.2
    for i, metric in enumerate(["accuracy", "precision", "recall", "f1"]):
        plt.bar(x + i * width, results_df[metric], width, label=metric)
    plt.xticks(x + 1.5 * width, results_df["model"], rotation=30, ha="right")
    plt.ylabel("Score"); plt.title("Model Comparison")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    df = load_and_clean("./datasets/resumes_dataset.csv")
    X, y, scaler, label_encoder, embed_dim = build_features(df)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]} | Embedding dim: {embed_dim}")

    results = []
    trained_models = {}
    for name, model in get_models().items():
        print(f"Training {name}...")
        metrics, fitted, y_pred = evaluate_model(name, model, X_train, X_val, y_train, y_val)
        results.append(metrics)
        trained_models[name] = fitted
        print(f"  {metrics}")

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(f"{REPORTS}/model_comparison.csv", index=False)
    plot_model_comparison(results_df, f"{REPORTS}/model_comparison.png")

    ensemble_names = ["Random Forest", "XGBoost", "LightGBM"]
    ensemble_df = results_df[results_df["model"].isin(ensemble_names)]
    print("\nEnsemble comparison:\n", ensemble_df)

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nBEST MODEL: {best_name}")

    y_test_pred = best_model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    plot_confusion_matrix(y_test, y_test_pred, label_encoder, best_name, f"{REPORTS}/confusion_matrix.png")
    plot_roc_curves(best_model, X_val, y_val, label_encoder, f"{REPORTS}/roc_curve.png")
    plot_feature_importance(best_model, embed_dim, f"{REPORTS}/feature_importance.png")
    plot_learning_curve(get_models()[best_name], X_train, y_train, f"{REPORTS}/learning_curve.png")

    joblib.dump(best_model, f"{SAVED}/best_model.pkl")
    joblib.dump(scaler, f"{SAVED}/numeric_scaler.pkl")
    joblib.dump(label_encoder, f"{SAVED}/label_encoder.pkl")

    with open(f"{REPORTS}/best_model_info.json", "w") as f:
        json.dump({"best_model": best_name,
                    "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
                    "test_f1_macro": round(f1_score(y_test, y_test_pred, average="macro"), 4),
                    "feature_type": "sentence-transformer embeddings + scaled numeric features"}, f, indent=2)

    print("\nAll models & reports saved.")
    return results_df


if __name__ == "__main__":
    main()
