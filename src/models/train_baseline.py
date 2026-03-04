import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


DATA = "data/processed/train_sample.csv"
LABEL_COL = "attack_id"

RESULTS_DIR = Path("results/latest")
ARTIFACTS_DIR = Path("artifacts/models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "latest.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.joblib"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)

    y = df[LABEL_COL].astype(int)
    X = df.select_dtypes(include=[np.number]).copy()

    if LABEL_COL in X.columns:
        X = X.drop(columns=[LABEL_COL])

    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Total NaNs in X:", int(X.isna().sum().sum()))

    # save column order before splitting so inference uses the same schema
    joblib.dump(list(X.columns), FEATURES_PATH)
    print(f"Saved feature columns -> {FEATURES_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # fill missing → scale → logistic regression
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, solver="saga", n_jobs=-1))
    ])

    print("Training...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipe.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, digits=4))

    report_dict = classification_report(y_test, preds, digits=4, output_dict=True, zero_division=0)

    metrics_out = {
        "label_col": LABEL_COL,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "classification_report": report_dict
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"Saved metrics -> {metrics_path}")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, preds, xticks_rotation=45, colorbar=False, values_format="d"
    )

    fig = disp.figure_
    fig.set_size_inches(10, 8)
    fig.tight_layout()

    cm_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    print(f"Saved confusion matrix -> {cm_path}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH}")

    print("\nDone")


if __name__ == "__main__":
    main()
