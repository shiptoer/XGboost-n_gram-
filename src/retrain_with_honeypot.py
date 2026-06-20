from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from project_utils import load_labeled_dataset, package_versions, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DATASET_PATH = PROJECT_ROOT / "dataset" / "Modified_SQL_Dataset.csv"
HONEYPOT_TEST_PATH = PROJECT_ROOT / "results" / "honeypot_test_responses.csv"

OLD_MODEL_PATH = PROJECT_ROOT / "model" / "xgboost_char_ngram_model.pkl"
OLD_VECTORIZER_PATH = PROJECT_ROOT / "model" / "char_ngram_vectorizer.pkl"

UPDATED_DATASET_PATH = PROJECT_ROOT / "dataset" / "updated_dataset_with_honeypot.csv"

NEW_MODEL_PATH = PROJECT_ROOT / "model" / "xgboost_honeypot_updated_model.pkl"
NEW_VECTORIZER_PATH = PROJECT_ROOT / "model" / "char_ngram_vectorizer_honeypot_updated.pkl"
NEW_METADATA_PATH = PROJECT_ROOT / "model" / "xgboost_honeypot_updated_metadata.json"

RESULTS_PATH = PROJECT_ROOT / "results" / "honeypot_adaptation_comparison.csv"


def evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEW_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    original_df, dataset_info = load_labeled_dataset(ORIGINAL_DATASET_PATH)

    print("Original dataset loaded:", original_df.shape)
    print("Duplicate queries removed:", dataset_info["duplicate_rows_removed"])

    honeypot_df = pd.read_csv(HONEYPOT_TEST_PATH)
    honeypot_df = honeypot_df.dropna(subset=["payload"])

    honeypot_training_df = pd.DataFrame(
        {
            "Query": honeypot_df["payload"].astype(str),
            "Label": 1,
        }
    )

    if honeypot_training_df.empty:
        raise ValueError(f"No honeypot payloads found in {HONEYPOT_TEST_PATH}")

    print("Honeypot payloads loaded:", honeypot_training_df.shape)

    old_model = joblib.load(str(OLD_MODEL_PATH))
    old_vectorizer = joblib.load(str(OLD_VECTORIZER_PATH))

    X_honeypot = honeypot_training_df["Query"].astype(str)
    y_honeypot = honeypot_training_df["Label"].astype(int)

    X_honeypot_old_vec = old_vectorizer.transform(X_honeypot)

    start_old = time.perf_counter()
    old_predictions = old_model.predict(X_honeypot_old_vec)
    old_latency_ms = ((time.perf_counter() - start_old) / len(X_honeypot)) * 1000

    old_metrics = evaluate_predictions(y_honeypot, old_predictions)
    old_detected = int(sum(old_predictions))
    old_total = len(y_honeypot)

    print("\nBefore Honeypot Update:")
    print("Detected:", old_detected, "/", old_total)
    print("Accuracy:", old_metrics["Accuracy"])
    print("Recall:", old_metrics["Recall"])
    print("F1-score:", old_metrics["F1_score"])

    updated_df = pd.concat([honeypot_training_df, original_df], ignore_index=True)
    before_update_dedupe = len(updated_df)
    updated_df = updated_df.drop_duplicates(subset=["Query"], keep="first").reset_index(drop=True)
    updated_duplicates_removed = before_update_dedupe - len(updated_df)
    updated_df.to_csv(str(UPDATED_DATASET_PATH), index=False)

    print("\nUpdated dataset saved:", UPDATED_DATASET_PATH)
    print("Updated dataset shape:", updated_df.shape)
    print("Updated duplicate queries removed:", updated_duplicates_removed)
    print("Updated label distribution:")
    print(updated_df["Label"].value_counts())

    X_train = updated_df["Query"].astype(str)
    y_train = updated_df["Label"].astype(int)

    new_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=5000,
    )
    X_train_vec = new_vectorizer.fit_transform(X_train)

    new_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    start_train = time.perf_counter()
    new_model.fit(X_train_vec, y_train)
    training_time = time.perf_counter() - start_train

    print("\nRetraining completed.")
    print("Training time:", training_time, "seconds")

    X_honeypot_new_vec = new_vectorizer.transform(X_honeypot)

    start_new = time.perf_counter()
    new_predictions = new_model.predict(X_honeypot_new_vec)
    new_latency_ms = ((time.perf_counter() - start_new) / len(X_honeypot)) * 1000

    new_metrics = evaluate_predictions(y_honeypot, new_predictions)
    new_detected = int(sum(new_predictions))

    print("\nAfter Honeypot Update:")
    print("Detected:", new_detected, "/", old_total)
    print("Accuracy:", new_metrics["Accuracy"])
    print("Recall:", new_metrics["Recall"])
    print("F1-score:", new_metrics["F1_score"])

    joblib.dump(new_model, str(NEW_MODEL_PATH))
    joblib.dump(new_vectorizer, str(NEW_VECTORIZER_PATH))

    results_df = pd.DataFrame(
        [
            {
                "Stage": "Before Honeypot Update",
                "Test_Set": "Honeypot SQL Injection Payloads",
                "Detected_Attacks": old_detected,
                "Total_Attacks": old_total,
                "Detection_Rate": old_detected / old_total,
                "Accuracy": old_metrics["Accuracy"],
                "Precision": old_metrics["Precision"],
                "Recall": old_metrics["Recall"],
                "F1_score": old_metrics["F1_score"],
                "Average_Prediction_Latency_ms": old_latency_ms,
            },
            {
                "Stage": "After Honeypot Update",
                "Test_Set": "Honeypot SQL Injection Payloads",
                "Detected_Attacks": new_detected,
                "Total_Attacks": old_total,
                "Detection_Rate": new_detected / old_total,
                "Accuracy": new_metrics["Accuracy"],
                "Precision": new_metrics["Precision"],
                "Recall": new_metrics["Recall"],
                "F1_score": new_metrics["F1_score"],
                "Average_Prediction_Latency_ms": new_latency_ms,
            },
        ]
    )
    results_df.to_csv(str(RESULTS_PATH), index=False)

    metadata = {
        "model": "XGBoost",
        "feature_method": "Character-level n-gram",
        "original_dataset": dataset_info,
        "honeypot_payloads": {
            "path": str(HONEYPOT_TEST_PATH),
            "rows": len(honeypot_training_df),
        },
        "updated_dataset": {
            "path": str(UPDATED_DATASET_PATH),
            "rows": len(updated_df),
            "duplicate_queries_removed_after_merge": updated_duplicates_removed,
            "honeypot_rows_preferred_on_duplicate_query": True,
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
        "before_update_metrics": {
            **old_metrics,
            "detected_attacks": old_detected,
            "total_attacks": old_total,
            "detection_rate": old_detected / old_total,
            "average_prediction_latency_ms": old_latency_ms,
        },
        "after_update_metrics": {
            **new_metrics,
            "detected_attacks": new_detected,
            "total_attacks": old_total,
            "detection_rate": new_detected / old_total,
            "average_prediction_latency_ms": new_latency_ms,
            "training_time_seconds": training_time,
        },
        "artifacts": {
            "model_path": str(NEW_MODEL_PATH),
            "vectorizer_path": str(NEW_VECTORIZER_PATH),
            "results_path": str(RESULTS_PATH),
        },
        "package_versions": package_versions(
            ["pandas", "scikit-learn", "xgboost", "joblib"]
        ),
    }
    write_json(NEW_METADATA_PATH, metadata)

    print("\nUpdated model saved to:", NEW_MODEL_PATH)
    print("Updated vectorizer saved to:", NEW_VECTORIZER_PATH)
    print("Updated metadata saved to:", NEW_METADATA_PATH)
    print("Comparison results saved to:", RESULTS_PATH)
    print("\n===== Honeypot Adaptation Comparison =====")
    print(results_df)


if __name__ == "__main__":
    main()
