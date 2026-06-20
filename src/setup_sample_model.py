from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "Modified_SQL_Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "sample_logistic_char_ngram_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "model" / "sample_logistic_char_ngram_vectorizer.pkl"


def create_sample_model():
    if not DATASET_PATH.exists():
        print("Dataset not found:", DATASET_PATH)
        return

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["Query", "Label"]) if not df.empty else df
    X = df["Query"].astype(str)
    y = df["Label"].astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=5000,
    )
    X_vec = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_vec, y)

    joblib.dump(model, str(MODEL_PATH))
    joblib.dump(vectorizer, str(VECTORIZER_PATH))
    print("Saved model and vectorizer to:", MODEL_PATH, VECTORIZER_PATH)


if __name__ == "__main__":
    create_sample_model()
