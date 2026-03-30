from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.inference.detect import predict_df, load_pipeline, load_feature_columns, align_features

app = FastAPI(title="Wi-Fi IDS API")

live_sessions: Dict[str, Any] = {}

UI_INDEX = Path("ui/index.html")

MODELS = {
    "lr": {
        "path": Path("artifacts/models/latest.joblib"),
        "features": Path("artifacts/models/feature_columns.joblib"),
        "metrics": Path("results/latest/metrics.json"),
        "tag": "Logistic Regression (baseline)",
    },
    "rf": {
        "path": Path("artifacts/models/rf_pipeline.joblib"),
        "features": Path("artifacts/models/feature_columns.joblib"),
        "metrics": Path("results/rf/metrics.json"),
        "tag": "Random Forest",
    },
}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    if not UI_INDEX.exists():
        return "<h2>Dashboard not found</h2><p>Create <code>ui/index.html</code>.</p>"
    return UI_INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/model_info")
def model_info(model: str = "lr") -> Dict[str, Any]:
    cfg = MODELS.get(model)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose: {list(MODELS)}")
    metrics_path = cfg["metrics"]
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
        "model_tag": cfg["tag"],
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
    model: str = "lr",
) -> Dict[str, Any]:
    # runs the uploaded CSV through the selected model and returns everything the dashboard needs
    cfg = MODELS.get(model)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose: {list(MODELS)}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    t0 = time.perf_counter()
    result = predict_df(df, model_path=cfg["path"], features_path=cfg["features"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    preds = result.get("preds", [])
    scores: Optional[list] = result.get("scores")

    counts = Counter(preds)

    response: Dict[str, Any] = {
        "n_rows": result.get("n_rows", len(preds)),
        "prediction_counts": dict(counts),
        "model_tag": cfg["tag"],
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


@app.post("/start_live")
async def start_live(
    file: UploadFile = File(...),
    model: str = "lr",
) -> Dict[str, Any]:
    cfg = MODELS.get(model)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose: {list(MODELS)}")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    result = predict_df(df, model_path=cfg["path"], features_path=cfg["features"])

    session_id = uuid.uuid4().hex[:8]
    live_sessions[session_id] = {
        "preds": result["preds"],
        "scores": result.get("scores"),
        "n_rows": result["n_rows"],
        "model_tag": cfg["tag"],
    }
    return {"session_id": session_id, "n_rows": result["n_rows"]}


@app.websocket("/ws/live/{session_id}")
async def live_websocket(
    websocket: WebSocket,
    session_id: str,
    batch_size: int = 50,
    delay_ms: int = 200,
):
    await websocket.accept()
    session = live_sessions.pop(session_id, None)
    if not session:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return

    preds     = session["preds"]
    scores    = session["scores"]
    n_rows    = session["n_rows"]
    model_tag = session["model_tag"]

    try:
        for i in range(0, n_rows, batch_size):
            bp = preds[i : i + batch_size]
            bs = scores[i : i + batch_size] if scores else [None] * len(bp)
            rows = [
                {"row_index": i + j, "pred": int(p), "score": float(s) if s is not None else None}
                for j, (p, s) in enumerate(zip(bp, bs))
            ]
            await websocket.send_json({
                "batch_start": i,
                "batch_end": i + len(bp),
                "total": n_rows,
                "model_tag": model_tag,
                "rows": rows,
            })
            await asyncio.sleep(delay_ms / 1000.0)
        await websocket.send_json({"done": True, "total": n_rows})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await websocket.close()
