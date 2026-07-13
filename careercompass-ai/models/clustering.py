"""
clustering.py
--------------
WHY: Beyond predicting an explicit role label, it's useful to discover
natural groupings in resumes without labels (Experiment 6). We use K-Means
to cluster resumes by semantic profile and PCA to compress the embedding
space down to 2D for visualization.

UPGRADE (v2): Clustering now runs on sentence embeddings instead of TF-IDF,
so resumes cluster by actual meaning (e.g. "builds ML pipelines" resumes
group together even if they never share exact keywords) rather than raw
word overlap.

ALGORITHM CHOICE: K-Means is chosen because clusters are expected to be
roughly well-separated in embedding space once resume content differs by
domain, and it's fast + simple to explain to a reviewer. The optimal K is
chosen via the Elbow Method (inertia) and confirmed with Silhouette Score.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append("/home/claude/careercompass-ai")
from preprocessing.text_preprocessing import load_and_clean, build_embeddings

REPORTS = "./reports"
SAVED = "./saved_models"
NUMERIC_COLS = ["num_skills", "years_experience", "has_projects", "has_certifications", "num_sections"]


def find_best_k(X, k_range=range(2, 12)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_, sample_size=500, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(k_range), inertias, "o-")
    axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[1].plot(list(k_range), silhouettes, "o-", color="green")
    axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("k")
    plt.tight_layout(); plt.savefig(f"{REPORTS}/kmeans_elbow_silhouette.png", dpi=110); plt.close()

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    return best_k


def run_clustering():
    df = load_and_clean("./datasets/resumes_dataset.csv")
    X_embed = build_embeddings(df["resume_text"])

    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[NUMERIC_COLS].values)
    X = np.hstack([X_embed, X_numeric])

    best_k = find_best_k(X)
    print(f"Best k (by silhouette): {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    df["cluster"] = kmeans.labels_

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=kmeans.labels_, cmap="tab10", s=18, alpha=0.75)
    plt.title(f"Resume Clusters (K-Means, k={best_k}) — PCA 2D Projection of Embeddings")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout(); plt.savefig(f"{REPORTS}/pca_clusters.png", dpi=110); plt.close()

    print("\nDominant role per cluster:")
    print(df.groupby("cluster")["role"].agg(lambda s: s.value_counts().idxmax()))

    joblib.dump(kmeans, f"{SAVED}/kmeans_model.pkl")
    joblib.dump(pca, f"{SAVED}/pca_model.pkl")
    df.to_csv("./datasets/resumes_clustered.csv", index=False)
    return df, kmeans, pca


if __name__ == "__main__":
    run_clustering()
