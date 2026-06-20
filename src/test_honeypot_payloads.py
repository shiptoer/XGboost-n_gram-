from pathlib import Path
import time

import pandas as pd
import requests

from project_utils import get_api_url, sanitize_record_for_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

attack_payloads = [
    "' OR 1=1--",
    "' OR 'a'='a",
    "admin' --",
    "' UNION SELECT username, password FROM users--",
    "' UNION SELECT NULL, NULL--",
    "' AND 1=1--",
    "' AND 1=2--",
    "' OR 1=1#",
    "\" OR \"1\"=\"1",
    "' OR sleep(5)--",
    "'; DROP TABLE users;--",
    "'; DELETE FROM users;--",
    "' OR EXISTS(SELECT * FROM users)--",
    "' UNION SELECT database(), user()--",
    "' UNION SELECT table_name, column_name FROM information_schema.columns--",
    "1 OR 1=1",
    "1 UNION SELECT username,password FROM users",
    "1; DROP TABLE logs",
    "' OR 1=1 LIMIT 1--",
    "' OR 'x'='x'--",
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    api_url = get_api_url()
    records = []

    print("Sending SQL Injection payloads to API...")
    print("API URL:", api_url)

    for payload in attack_payloads:
        try:
            start_time = time.perf_counter()
            response = requests.get(api_url, params={"payload": payload}, timeout=5)
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            response.raise_for_status()
            data = response.json()
            records.append(
                sanitize_record_for_csv(
                    {
                        "payload": payload,
                        "prediction": data.get("prediction"),
                        "result": data.get("result"),
                        "confidence": data.get("confidence"),
                        "latency_ms": data.get("latency_ms"),
                        "action": data.get("action"),
                        "total_latency_ms": total_latency_ms,
                    }
                )
            )
            print(payload, "=>", data.get("result"), "|", data.get("action"))
        except requests.RequestException as e:
            records.append(
                sanitize_record_for_csv(
                    {
                        "payload": payload,
                        "prediction": None,
                        "result": "ERROR",
                        "confidence": None,
                        "latency_ms": None,
                        "action": "ERROR",
                        "error": str(e),
                    }
                )
            )
            print(payload, "=> ERROR:", e)

    df = pd.DataFrame(records)
    output_path = RESULTS_DIR / "honeypot_test_responses.csv"
    df.to_csv(str(output_path), index=False)

    print("\nTest completed.")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
