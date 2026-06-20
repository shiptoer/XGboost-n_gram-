# SQLi XGBoost Honeypot

Prototype for SQL Injection detection using XGBoost, character-level n-gram TF-IDF features, and honeypot-style logging.

## Features

- FastAPI detection endpoint for SQL Injection payloads.
- XGBoost classifier with character-level n-gram vectorization.
- Honeypot logging for detected malicious payloads.
- Model comparison scripts for Logistic Regression, Random Forest, and XGBoost.
- n-gram comparison scripts for word-level and character-level features.
- Retraining workflow using honeypot payload responses.
- Model metadata files for reproducibility.
- Smoke tests and GitHub Actions CI.

## Project Structure

- `src/api.py`: FastAPI detection API.
- `src/train_xgboost.py`: trains the main XGBoost model.
- `src/compare_models.py`: compares Logistic Regression, Random Forest, and XGBoost.
- `src/compare_ngram.py`: compares word-level and character-level n-grams.
- `src/test_model_smoke.py`: checks model loading, basic predictions, payload validation, and CSV sanitization.
- `src/test_honeypot_payloads.py`: sends sample SQLi payloads to the API and saves responses.
- `src/test_api_latency.py`: measures API latency.
- `src/retrain_with_honeypot.py`: retrains the model with honeypot payload responses.
- `dataset/`: training datasets.
- `model/`: model, vectorizer, and metadata artifacts.
- `results/`: evaluation results and generated response CSV files.

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project pins `scikit-learn==1.8.0` because the committed model/vectorizer pickle artifacts were generated with that version.

## Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --app-dir src --reload
```

Quick checks:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/"
Invoke-RestMethod "http://127.0.0.1:8000/detect?payload=1%20OR%201%3D1"
```

## Run Tests

Tests that do not require the API server:

```powershell
.\.venv\Scripts\python.exe src\run_checks.py
.\.venv\Scripts\python.exe src\test_model_smoke.py
```

Tests that require the API server:

```powershell
.\.venv\Scripts\python.exe src\test_honeypot_payloads.py
.\.venv\Scripts\python.exe src\test_api_latency.py
```

If the API runs on another port or host, set `API_URL`:

```powershell
$env:API_URL = "http://127.0.0.1:8010/detect"
.\.venv\Scripts\python.exe src\test_honeypot_payloads.py
```

## Train and Retrain

Train the main model:

```powershell
.\.venv\Scripts\python.exe src\train_xgboost.py
```

Main outputs:

- `model/xgboost_char_ngram_model.pkl`
- `model/char_ngram_vectorizer.pkl`
- `model/xgboost_char_ngram_metadata.json`
- `results/xgboost_char_ngram_results.csv`

Retrain with honeypot payload responses:

```powershell
.\.venv\Scripts\python.exe src\retrain_with_honeypot.py
```

Main outputs:

- `model/xgboost_honeypot_updated_model.pkl`
- `model/char_ngram_vectorizer_honeypot_updated.pkl`
- `model/xgboost_honeypot_updated_metadata.json`
- `dataset/updated_dataset_with_honeypot.csv`
- `results/honeypot_adaptation_comparison.csv`

## Docker

Build:

```powershell
docker build -t sqli-xgboost-honeypot .
```

Run:

```powershell
docker run --rm -p 8000:8000 sqli-xgboost-honeypot
```

Then open:

```text
http://127.0.0.1:8000/
```

## Dataset Sources

The committed training artifacts are based on local CSV files in `dataset/`, especially:

- `dataset/Modified_SQL_Dataset.csv`
- `dataset/SQLI_Dataset.csv`
- `dataset/updated_dataset_with_honeypot.csv`

The local workspace also contained third-party payload corpora such as PayloadsAllTheThings and payload-box. Those corpora are not committed because they are external references and are not required to run the current code. If they are used in future experiments, cite their upstream repositories and licenses in the report.

## Current Results

Latest main XGBoost training run:

- Accuracy: `0.9940`
- Precision: `0.9978`
- Recall: `0.9859`
- F1-score: `0.9918`

Latest honeypot adaptation run:

- Before update: `11/20` detected
- After update: `19/20` detected

See:

- `model/xgboost_char_ngram_metadata.json`
- `model/xgboost_honeypot_updated_metadata.json`
- `results/xgboost_char_ngram_results.csv`
- `results/honeypot_adaptation_comparison.csv`

## Security and Limitations

- This is a prototype, not a production WAF replacement.
- Pickle/joblib model files should only be loaded from trusted sources.
- SQLi payloads can be obfuscated; real-world bypass testing is still needed.
- Dataset duplicates are removed by `Query` before train/test split to reduce leakage, but external evaluation is still recommended.
- Payloads written to CSV are sanitized to reduce CSV formula injection risk when opened in spreadsheet tools.
- API payload length is limited to `2048` characters.

## License

Code is released under the MIT License. Dataset and third-party payload sources may have their own licenses.
