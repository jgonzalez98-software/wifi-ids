import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


DATA = "data/processed/train_sample.csv"
LABEL_COL = "attack_id"

RESULTS_DIR = Path("results/rf")
ARTIFACTS_DIR = Path("artifacts/models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "rf_pipeline.joblib"
BASELINE_METRICS = Path("results/latest/metrics.json")

LABELS = {
    0: "Normal",
    1: "Deauth", 2: "Disas", 3: "Re/Assoc", 4: "Rogue AP",
    5: "Krack", 6: "Kr00k", 7: "SSH", 8: "Botnet", 9: "Malware",
    10: "SQL Injection", 11: "SSDP", 12: "Evil Twin", 13: "Website Spoofing",
}


def print_comparison(rf_report: dict):
    if not BASELINE_METRICS.exists():
        print("No baseline metrics found, skipping comparison.")
        return

    baseline = json.loads(BASELINE_METRICS.read_text())
    base_report = baseline.get("classification_report", {})

    print("\n--- Model Comparison (macro F1) ---")
    print(f"{'Class':<18} {'LR F1':>8} {'RF F1':>8} {'Diff':>8}")
    print("-" * 46)

    classes = sorted([k for k in rf_report if k.isdigit()], key=int)
    for k in classes:
        name = LABELS.get(int(k), f"Class {k}")
        rf_f1 = rf_report[k]["f1-score"]
        lr_f1 = base_report.get(k, {}).get("f1-score", None)
        if lr_f1 is not None:
            diff = rf_f1 - lr_f1
            sign = "+" if diff >= 0 else ""
            print(f"{name:<18} {lr_f1:>8.4f} {rf_f1:>8.4f} {sign}{diff:>7.4f}")
        else:
            print(f"{name:<18} {'N/A':>8} {rf_f1:>8.4f}")

    print("-" * 46)
    rf_acc = rf_report.get("accuracy", 0)
    lr_acc = base_report.get("accuracy", 0)
    diff = rf_acc - lr_acc
    sign = "+" if diff >= 0 else ""
    print(f"{'Overall accuracy':<18} {lr_acc:>8.4f} {rf_acc:>8.4f} {sign}{diff:>7.4f}")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)

    y = df[LABEL_COL].astype(int)
    X = df.select_dtypes(include=[np.number]).copy()

    if LABEL_COL in X.columns:
        X = X.drop(columns=[LABEL_COL])

    print("X shape:", X.shape, "| y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RF doesn't need scaling, just impute missing values
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])

    print("Training Random Forest (this takes a few minutes)...")
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

    # confusion matrix
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

    # feature importance — top 20
    rf = pipe.named_steps["model"]
    feature_names = list(X.columns)
    importances = rf.feature_importances_

    top_idx = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_vals[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances (Random Forest)")
    fig.tight_layout()
    fi_path = RESULTS_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=200)
    plt.close(fig)
    print(f"Saved feature importance -> {fi_path}")

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH}")

    print_comparison(report_dict)

    print("\nDone")


if __name__ == "__main__":
    main()
