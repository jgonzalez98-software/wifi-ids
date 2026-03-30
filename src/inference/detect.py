from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("artifacts/models/latest.joblib")
FEATURES_PATH = Path("artifacts/models/feature_columns.joblib")


def load_pipeline(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return joblib.load(model_path)


def load_feature_columns(features_path: Path = FEATURES_PATH) -> Optional[list[str]]:
    if features_path.exists():
        return joblib.load(features_path)
    return None


def align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_cols]  # drop extras + enforce order
    return df


def predict_df(
    df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
) -> Dict[str, Any]:
    pipe = load_pipeline(model_path)
    feature_cols = load_feature_columns(features_path)

    # match training behavior: numeric only
    X = df.select_dtypes(include=[np.number]).copy()

    if feature_cols is not None:
        X = align_features(X, feature_cols)

    preds = pipe.predict(X)

    scores = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        scores = np.max(proba, axis=1).tolist()
    elif hasattr(pipe, "decision_function"):
        dec = pipe.decision_function(X)
        if getattr(dec, "ndim", 1) == 1:
            scores = dec.tolist()
        else:
            scores = np.max(dec, axis=1).tolist()

    return {
        "n_rows": int(len(X)),
        "preds": preds.tolist(),
        "scores": scores
    }
