from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from project_utils import load_labeled_dataset, package_versions, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "Modified_SQL_Dataset.csv"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, dataset_info = load_labeled_dataset(DATASET_PATH)

    print("Dataset loaded successfully!")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("Duplicate queries removed:", dataset_info["duplicate_rows_removed"])
    print(df.head())

    X = df["Query"].astype(str)
    y = df["Label"].astype(int)

    print("\nLabel distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=5000,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\nVectorization completed!")
    print("Vector shape:", X_train_vec.shape)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    start_train = time.perf_counter()
    model.fit(X_train_vec, y_train)
    training_time = time.perf_counter() - start_train

    print(f"\nTraining completed in {training_time:.4f} seconds")

    start_pred = time.perf_counter()
    y_pred = model.predict(X_test_vec)
    prediction_time = time.perf_counter() - start_pred

    avg_latency_ms = (prediction_time / len(X_test)) * 1000 if len(X_test) > 0 else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n===== Evaluation Results =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if avg_latency_ms is not None:
        print(f"Average prediction latency: {avg_latency_ms:.4f} ms/sample")
    else:
        print("Average prediction latency: N/A (no test samples)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Benign", "SQL Injection"],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    model_path = MODEL_DIR / "xgboost_char_ngram_model.pkl"
    vectorizer_path = MODEL_DIR / "char_ngram_vectorizer.pkl"
    metadata_path = MODEL_DIR / "xgboost_char_ngram_metadata.json"
    results_path = RESULTS_DIR / "xgboost_char_ngram_results.csv"

    joblib.dump(model, str(model_path))
    joblib.dump(vectorizer, str(vectorizer_path))

    results = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Feature_Method": "Character-level n-gram",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_score": f1,
                "Training_Time_Seconds": training_time,
                "Average_Prediction_Latency_ms": avg_latency_ms,
            }
        ]
    )
    results.to_csv(str(results_path), index=False)

    metadata = {
        "model": "XGBoost",
        "feature_method": "Character-level n-gram",
        "dataset": dataset_info,
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "vectorizer": {
            "class": "TfidfVectorizer",
            "analyzer": "char",
            "ngram_range": [3, 5],
            "lowercase": True,
            "max_features": 5000,
        },
        "model_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time_seconds": training_time,
            "average_prediction_latency_ms": avg_latency_ms,
        },
        "artifacts": {
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
            "results_path": str(results_path),
        },
        "package_versions": package_versions(
            ["pandas", "scikit-learn", "xgboost", "joblib"]
        ),
    }
    write_json(metadata_path, metadata)

    print(f"\nModel saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
