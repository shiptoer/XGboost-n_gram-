from pathlib import Path
import time

import pandas as pd
import requests

from project_utils import get_api_url, sanitize_record_for_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
NUM_ROUNDS = 100

test_payloads = [
    "' OR 1=1--",
    "' UNION SELECT username, password FROM users--",
    "admin' --",
    "' OR 'a'='a",
    "SELECT * FROM products WHERE id = 5",
    "SELECT name, price FROM products",
    "UPDATE users SET name = 'John' WHERE id = 1",
    "INSERT INTO orders VALUES (1, 2, 50000)",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    api_url = get_api_url()
    records = []

    print("Starting API latency test...")
    print("API URL:", api_url)

    for i in range(NUM_ROUNDS):
        for payload in test_payloads:
            start_time = time.perf_counter()
            try:
                response = requests.get(api_url, params={"payload": payload}, timeout=5)
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                response.raise_for_status()
                data = response.json()
                records.append(
                    sanitize_record_for_csv(
                        {
                        "round": i + 1,
                        "payload": payload,
                        "prediction": data.get("prediction"),
                        "result": data.get("result"),
                        "confidence": data.get("confidence"),
                        "internal_model_latency_ms": data.get("latency_ms"),
                        "total_api_latency_ms": total_latency_ms,
                        }
                    )
                )
            except requests.RequestException as e:
                total_latency_ms = (time.perf_counter() - start_time) * 1000
                records.append(
                    sanitize_record_for_csv(
                        {
                        "round": i + 1,
                        "payload": payload,
                        "prediction": None,
                        "result": "ERROR",
                        "confidence": None,
                        "internal_model_latency_ms": None,
                        "total_api_latency_ms": total_latency_ms,
                        "error": str(e),
                        }
                    )
                )

    df = pd.DataFrame(records)
    output_path = RESULTS_DIR / "api_latency_results.csv"
    df.to_csv(str(output_path), index=False)

    print("\n===== API Latency Summary =====")
    print("Total requests:", len(df))
    print("\nInternal model latency:")
    print("Average:", df["internal_model_latency_ms"].mean(), "ms")
    print("Minimum:", df["internal_model_latency_ms"].min(), "ms")
    print("Maximum:", df["internal_model_latency_ms"].max(), "ms")
    print("\nTotal API latency:")
    print("Average:", df["total_api_latency_ms"].mean(), "ms")
    print("Minimum:", df["total_api_latency_ms"].min(), "ms")
    print("Maximum:", df["total_api_latency_ms"].max(), "ms")
    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
