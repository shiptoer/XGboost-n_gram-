from pathlib import Path
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from project_utils import load_labeled_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "Modified_SQL_Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, dataset_info = load_labeled_dataset(DATASET_PATH)

    X = df["Query"].astype(str)
    y = df["Label"].astype(int)

    print("Dataset loaded:", df.shape)
    print("Duplicate queries removed:", dataset_info["duplicate_rows_removed"])
    print("Label distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_methods = {
        "Word-level n-gram": TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            lowercase=True,
            max_features=5000,
        ),
        "Character-level n-gram": TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=True,
            max_features=5000,
        ),
    }

    results = []

    for feature_name, vectorizer in feature_methods.items():
        print(f"\nRunning feature method: {feature_name}")

        start_vec = time.perf_counter()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        vectorization_time = time.perf_counter() - start_vec

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

        start_pred = time.perf_counter()
        y_pred = model.predict(X_test_vec)
        prediction_time = time.perf_counter() - start_pred

        avg_latency_ms = (prediction_time / len(X_test)) * 1000 if len(X_test) > 0 else None
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "Feature_Method": feature_name,
                "Model": "XGBoost",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_score": f1,
                "Vectorization_Time_Seconds": vectorization_time,
                "Training_Time_Seconds": training_time,
                "Average_Prediction_Latency_ms": avg_latency_ms,
            }
        )

        print(f"{feature_name} Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"Vectorization time: {vectorization_time:.4f} seconds")
        print(f"Training time: {training_time:.4f} seconds")
        if avg_latency_ms is not None:
            print(f"Average prediction latency: {avg_latency_ms:.4f} ms/sample")
        else:
            print("Average prediction latency: N/A (no test samples)")

    output_path = RESULTS_DIR / "ngram_comparison_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(str(output_path), index=False)

    print("\n===== Final n-gram Comparison =====")
    print(results_df)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
