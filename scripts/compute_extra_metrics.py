import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score


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


def plot_pr_curves(name, pipe, results_dir, X_test, y_test, classes, y_bin):
    y_prob = pipe.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, lw=1, label=f"{LABELS.get(cls, cls)} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves — {name}")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()

    path = results_dir / "pr_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved PR curves -> {path}")


def plot_comparison(lr_metrics, rf_metrics, classes):
    lr_report = lr_metrics["classification_report"]
    rf_report = rf_metrics["classification_report"]

    class_names = [LABELS.get(c, str(c)) for c in classes]
    lr_f1 = [lr_report[str(c)]["f1-score"] for c in classes]
    rf_f1 = [rf_report[str(c)]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, lr_f1, width, label="Logistic Regression")
    ax.bar(x + width / 2, rf_f1, width, label="Random Forest")

    ax.set_xlabel("Attack Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score per Attack Class — LR vs RF")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.95, 1.005)
    ax.legend()
    fig.tight_layout()

    path = Path("results/model_comparison.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved model comparison -> {path}")


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
    y_bin = label_binarize(y_test, classes=classes)

    pipes = {}
    for name, cfg in MODELS.items():
        print(f"\n--- {name} ---")
        pipe = joblib.load(cfg["model_path"])
        pipes[name] = pipe
        plot_pr_curves(name, pipe, cfg["results_dir"], X_test, y_test, classes, y_bin)

    print("\n--- Model Comparison Chart ---")
    lr_metrics = json.loads(Path("results/latest/metrics.json").read_text())
    rf_metrics = json.loads(Path("results/rf/metrics.json").read_text())
    plot_comparison(lr_metrics, rf_metrics, classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
