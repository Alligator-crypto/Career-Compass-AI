"""
main.py
--------
Entry point that runs the full CareerCompass AI ML pipeline end-to-end:
  1. Generate dataset
  2. Preprocess (clean text, engineer features)
  3. Train + evaluate supervised models, pick the best, save it
  4. Run K-Means clustering + PCA visualization

After running this once, launch the web app with:
    streamlit run app/streamlit_app.py
"""
from datasets.generate_dataset import build_dataset
from preprocessing.text_preprocessing import load_and_clean
from models.train_models import main as train_main
from models.clustering import run_clustering

DATA_PATH = "datasets/resumes_dataset.csv"


def run_pipeline():
    print("STEP 1/4: Generating dataset...")
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"  -> {len(df)} rows saved to {DATA_PATH}")

    print("\nSTEP 2/4: Preprocessing...")
    cleaned = load_and_clean(DATA_PATH)
    print(f"  -> cleaned shape: {cleaned.shape}")

    print("\nSTEP 3/4: Training & evaluating supervised + ensemble models...")
    train_main()

    print("\nSTEP 4/4: Unsupervised clustering (K-Means + PCA)...")
    run_clustering()

    print("\nPipeline complete. Launch the app with:\n  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    run_pipeline()
