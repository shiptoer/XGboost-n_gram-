import csv
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from project_utils import sanitize_csv_cell


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "xgboost_char_ngram_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "model" / "char_ngram_vectorizer.pkl"
HONEYPOT_LOG_PATH = PROJECT_ROOT / "results" / "honeypot_logs.csv"
MAX_PAYLOAD_LENGTH = 2048


HONEYPOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model():
    """Load the trained model and vectorizer from the project-level model directory."""
    try:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            logger.error("Model or vectorizer file not found: %s, %s", MODEL_PATH, VECTORIZER_PATH)
            return None, None

        model_obj = joblib.load(str(MODEL_PATH))
        vectorizer_obj = joblib.load(str(VECTORIZER_PATH))
        logger.info("Loaded model from %s and vectorizer from %s", MODEL_PATH, VECTORIZER_PATH)
        return model_obj, vectorizer_obj
    except Exception:
        logger.exception("Error loading model/vectorizer")
        return None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_obj, vectorizer_obj = load_model()
    app.state.model = model_obj
    app.state.vectorizer = vectorizer_obj
    app.state.model_loaded = model_obj is not None and vectorizer_obj is not None
    if not app.state.model_loaded:
        logger.warning("Model not loaded; /detect will return 503 until a real model is available.")
    yield


app = FastAPI(
    title="SQL Injection Detection API",
    description="XGBoost + Character-level n-gram + Honeypot Logging Prototype",
    version="1.1.0",
    lifespan=lifespan,
)


class PayloadRequest(BaseModel):
    payload: str = Field(..., min_length=1, max_length=MAX_PAYLOAD_LENGTH)


def log_to_honeypot(payload, prediction, result, confidence, latency_ms):
    HONEYPOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        max_bytes = 5 * 1024 * 1024
        if HONEYPOT_LOG_PATH.exists() and HONEYPOT_LOG_PATH.stat().st_size > max_bytes:
            rotated = HONEYPOT_LOG_PATH.with_name(
                HONEYPOT_LOG_PATH.stem
                + "_"
                + datetime.now().strftime("%Y%m%d%H%M%S")
                + HONEYPOT_LOG_PATH.suffix
            )
            HONEYPOT_LOG_PATH.rename(rotated)
            logger.info("Rotated honeypot log to %s", rotated)
    except Exception:
        logger.exception("Failed to rotate honeypot log")

    lock = FileLock(str(HONEYPOT_LOG_PATH) + ".lock", timeout=5)
    try:
        with lock:
            file_exists = HONEYPOT_LOG_PATH.exists()
            with HONEYPOT_LOG_PATH.open(mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "payload",
                            "prediction",
                            "result",
                            "confidence",
                            "latency_ms",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        sanitize_csv_cell(payload),
                        prediction,
                        result,
                        confidence,
                        latency_ms,
                    ]
                )
    except Timeout:
        logger.warning("Timeout acquiring honeypot log lock; skipping log.")
    except Exception:
        logger.exception("Error writing honeypot log")


def get_loaded_model():
    model = getattr(app.state, "model", None)
    vectorizer = getattr(app.state, "vectorizer", None)
    if model is not None and vectorizer is not None:
        return model, vectorizer

    model, vectorizer = load_model()
    app.state.model = model
    app.state.vectorizer = vectorizer
    app.state.model_loaded = model is not None and vectorizer is not None
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not available")
    return model, vectorizer


def detect_sql_injection(payload: str):
    model, vectorizer = get_loaded_model()
    start_time = time.perf_counter()

    try:
        payload_vector = vectorizer.transform([payload])
        prediction_raw = model.predict(payload_vector)
        try:
            prediction = int(prediction_raw[0])
        except Exception:
            prediction = int(prediction_raw)

        try:
            probability = model.predict_proba(payload_vector)[0]
            confidence = float(max(probability))
        except Exception:
            confidence = 0.0
    except Exception as exc:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    latency_ms = (time.perf_counter() - start_time) * 1000
    result = "SQL Injection" if prediction == 1 else "Benign"
    action = "BLOCK_AND_LOG" if prediction == 1 else "ALLOW"

    if prediction == 1:
        log_to_honeypot(
            payload=payload,
            prediction=prediction,
            result=result,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    return {
        "payload": payload,
        "prediction": prediction,
        "result": result,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "action": action,
    }


@app.get("/")
def home():
    return {
        "message": "SQL Injection Detection API is running",
        "model": "XGBoost",
        "feature_extraction": "Character-level n-gram",
        "honeypot_logging": "Enabled",
        "model_loaded": getattr(app.state, "model_loaded", False),
    }


@app.get("/detect")
def detect_get(payload: str = Query(..., min_length=1, max_length=MAX_PAYLOAD_LENGTH)):
    return detect_sql_injection(payload)


@app.post("/detect")
def detect_post(request: PayloadRequest):
    return detect_sql_injection(request.payload)
