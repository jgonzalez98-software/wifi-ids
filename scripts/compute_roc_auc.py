import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc


DATA = "data/processed/train_sample.csv"
LABEL_COL = "attack_id"

MODELS = {
    "Logistic Regression": {
        "model_path": "artifacts/models/latest.joblib",
        "results_dir": Path("results/latest"),
    },
    "Random Forest": {
        "model_path": "artifacts/models/rf_pipeline.joblib",
        "results_dir": Path("results/rf"),
    },
}

LABELS = {
    1: "Deauth", 2: "Disas", 3: "Re/Assoc", 4: "Rogue AP",
    5: "Krack", 6: "Kr00k", 7: "SSH", 8: "Botnet", 9: "Malware",
    10: "SQL Injection", 11: "SSDP", 12: "Evil Twin", 13: "Website Spoofing",
}


def compute_and_save(name, model_path, results_dir, X_test, y_test, classes):
    print(f"\n--- {name} ---")
    pipe = joblib.load(model_path)
    y_prob = pipe.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=classes)

    roc_auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    roc_auc_weighted = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")

    print(f"Macro ROC-AUC:    {roc_auc_macro:.6f}")
    print(f"Weighted ROC-AUC: {roc_auc_weighted:.6f}")

    # save to metrics.json
    metrics_path = results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["roc_auc_macro"] = roc_auc_macro
    metrics["roc_auc_weighted"] = roc_auc_weighted
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Updated metrics -> {metrics_path}")

    # plot per-class ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        cls_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label=f"{LABELS.get(cls, cls)} (AUC={cls_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {name}")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()

    roc_path = results_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)
    print(f"Saved ROC curves -> {roc_path}")


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)

    y = df[LABEL_COL].astype(int)
    X = df.select_dtypes(include=[np.number]).copy()
    if LABEL_COL in X.columns:
        X = X.drop(columns=[LABEL_COL])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = sorted(y.unique())

    for name, cfg in MODELS.items():
        compute_and_save(name, cfg["model_path"], cfg["results_dir"], X_test, y_test, classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
