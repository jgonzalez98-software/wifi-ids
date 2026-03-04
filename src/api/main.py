from __future__ import annotations

import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.inference.detect import predict_df

app = FastAPI(title="Wi-Fi IDS API")

UI_INDEX = Path("ui/index.html")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    if not UI_INDEX.exists():
        return "<h2>Dashboard not found</h2><p>Create <code>ui/index.html</code>.</p>"
    return UI_INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/model_info")
def model_info() -> Dict[str, Any]:
    metrics_path = Path("results/latest/metrics.json")
    if not metrics_path.exists():
        return {"available": False}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = data.get("classification_report", {})
    accuracy = report.get("accuracy")
    per_class_f1 = {
        k: round(v["f1-score"], 4)
        for k, v in report.items()
        if k.isdigit()
    }
    return {
        "available": True,
        "model_tag": "Logistic Regression (baseline)",
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "n_train": data.get("n_train"),
        "n_test": data.get("n_test"),
        "per_class_f1": per_class_f1,
    }


@app.post("/predict_summary")
async def predict_summary(
    file: UploadFile = File(...),
    top_k: int = 50,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    # runs the uploaded CSV through the model and returns everything the dashboard needs
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    t0 = time.perf_counter()
    result = predict_df(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    preds = result.get("preds", [])
    scores: Optional[list] = result.get("scores")

    counts = Counter(preds)

    response: Dict[str, Any] = {
        "n_rows": result.get("n_rows", len(preds)),
        "prediction_counts": dict(counts),
        "model_tag": "latest.joblib (baseline logreg pipeline)",  # swap this out when RF/IF models are added
        "elapsed_ms": round(elapsed_ms, 2),
        "rows_per_sec": round((len(preds) / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0, 2),
        "min_score_used": float(min_score),
    }

    if scores is not None:
        s = np.asarray(scores, dtype=float)
        response.update(
            {
                "avg_score": float(s.mean()) if len(s) else None,
                "min_score": float(s.min()) if len(s) else None,
                "max_score": float(s.max()) if len(s) else None,
            }
        )

        if len(s):
            mask = s >= float(min_score)
            response["alert_count_above_threshold"] = int(mask.sum())

            idx = np.where(mask)[0]
            if idx.size:
                idx_sorted = idx[np.argsort(s[idx])[::-1]][:top_k]
                response["top_alerts"] = [
                    {"row_index": int(i), "pred": preds[int(i)], "score": float(s[int(i)])}
                    for i in idx_sorted
                ]
            else:
                response["top_alerts"] = []
        else:
            response["alert_count_above_threshold"] = 0
            response["top_alerts"] = []
    else:
        response["alert_count_above_threshold"] = None
        response["top_alerts"] = []

    return response
