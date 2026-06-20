from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://127.0.0.1:8000/detect"
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def get_api_url():
    return os.getenv("API_URL", DEFAULT_API_URL)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labeled_dataset(path, dedupe_query=True):
    df = pd.read_csv(path)
    raw_rows = len(df)

    df = df.dropna(subset=["Query", "Label"]).copy()
    df["Query"] = df["Query"].astype(str)
    df["Label"] = df["Label"].astype(int)
    clean_rows = len(df)

    duplicate_rows_removed = 0
    if dedupe_query:
        before_dedupe = len(df)
        df = df.drop_duplicates(subset=["Query"], keep="first").reset_index(drop=True)
        duplicate_rows_removed = before_dedupe - len(df)

    dataset_info = {
        "path": str(Path(path)),
        "sha256": file_sha256(path),
        "raw_rows": raw_rows,
        "clean_rows": clean_rows,
        "dedupe_query": dedupe_query,
        "duplicate_rows_removed": duplicate_rows_removed,
        "final_rows": len(df),
    }
    return df, dataset_info


def package_versions(package_names):
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = None
    return versions


def write_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def sanitize_csv_cell(value):
    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_record_for_csv(record):
    return {key: sanitize_csv_cell(value) for key, value in record.items()}
