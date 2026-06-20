# SQLi XGBoost Honeypot

Prototype phát hiện SQL Injection bằng XGBoost, character-level n-gram và honeypot logging.

## Cấu trúc chính

- `src/api.py`: FastAPI detection API.
- `src/train_xgboost.py`: train model XGBoost chính.
- `src/compare_models.py`: so sánh Logistic Regression, Random Forest và XGBoost.
- `src/compare_ngram.py`: so sánh word-level và character-level n-gram.
- `src/test_honeypot_payloads.py`: gửi payload SQLi mẫu đến API và lưu kết quả.
- `src/test_api_latency.py`: đo latency API.
- `src/retrain_with_honeypot.py`: retrain model bằng payload thu được từ honeypot test.
- `dataset/`: dữ liệu huấn luyện và payload tham khảo.
- `model/`: model, vectorizer và metadata.
- `results/`: kết quả đánh giá và log.

## Cài đặt

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu chưa có `.venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --app-dir src --reload
```

Kiểm tra nhanh:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/"
Invoke-RestMethod "http://127.0.0.1:8000/detect?payload=1%20OR%201%3D1"
```

## Chạy kiểm tra

Không cần server:

```powershell
.\.venv\Scripts\python.exe src\test_model_smoke.py
.\.venv\Scripts\python.exe src\run_checks.py
```

Cần API đang chạy:

```powershell
.\.venv\Scripts\python.exe src\test_honeypot_payloads.py
.\.venv\Scripts\python.exe src\test_api_latency.py
```

Nếu API chạy port khác, set `API_URL`:

```powershell
$env:API_URL = "http://127.0.0.1:8010/detect"
.\.venv\Scripts\python.exe src\test_honeypot_payloads.py
```

## Train và retrain

Train model chính:

```powershell
.\.venv\Scripts\python.exe src\train_xgboost.py
```

Output chính:

- `model/xgboost_char_ngram_model.pkl`
- `model/char_ngram_vectorizer.pkl`
- `model/xgboost_char_ngram_metadata.json`
- `results/xgboost_char_ngram_results.csv`

Retrain bằng payload honeypot:

```powershell
.\.venv\Scripts\python.exe src\retrain_with_honeypot.py
```

Output chính:

- `model/xgboost_honeypot_updated_model.pkl`
- `model/char_ngram_vectorizer_honeypot_updated.pkl`
- `model/xgboost_honeypot_updated_metadata.json`
- `dataset/updated_dataset_with_honeypot.csv`
- `results/honeypot_adaptation_comparison.csv`

## Ghi chú kỹ thuật

- Dataset được `dropna` và `drop_duplicates(subset=["Query"])` trước khi split để giảm data leakage.
- `requirements.txt` pin `scikit-learn==1.8.0` để khớp với model/vectorizer pickle hiện có.
- Payload ghi ra CSV log được sanitize để giảm rủi ro CSV injection khi mở bằng Excel.
- API giới hạn payload tối đa `2048` ký tự.
