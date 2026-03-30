# Wi-Fi IDS

A machine learning-based wireless intrusion detection system trained on the [AWID3](https://icsdweb.aegean.gr/awid/awid3) dataset. Detects 13 types of Wi-Fi attacks from pre-processed network traffic features, served through a FastAPI backend and dark-themed web dashboard.

---

## What it detects

| ID | Attack Type      | ID | Attack Type     |
|----|------------------|----|-----------------|
| 1  | Deauth           | 8  | Botnet          |
| 2  | Disas            | 9  | Malware         |
| 3  | Re/Assoc         | 10 | SQL Injection   |
| 4  | Rogue AP         | 11 | SSDP            |
| 5  | Krack            | 12 | Evil Twin       |
| 6  | Kr00k            | 13 | Website Spoofing|
| 7  | SSH              |    |                 |

---

## Project structure

```
wifi-ids/
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI app — serves dashboard + /predict_summary + /model_info
│   ├── inference/
│   │   └── detect.py         # Loads model, aligns features, runs predictions
│   └── models/
│       ├── train_baseline.py # Trains logistic regression pipeline, saves artifacts
│       └── train_rf.py       # Trains Random Forest, saves artifacts + feature importance
├── scripts/
│   ├── process_one_class.py  # Preprocesses raw AWID3 CSV files for one attack class
│   ├── train_sample.py       # Builds a balanced training CSV from processed class files
│   ├── capture_live.sh       # Puts a WiFi adapter in monitor mode and captures traffic
│   ├── extract_features.py   # Extracts AWID3-compatible features from a .pcap file
│   ├── compute_roc_auc.py    # Computes ROC AUC scores and saves ROC curve plots
│   └── compute_extra_metrics.py  # Computes PR curves and other supplementary metrics
├── artifacts/
│   └── models/
│       ├── latest.joblib           # Logistic regression pipeline
│       ├── rf_pipeline.joblib      # Random Forest pipeline
│       └── feature_columns.joblib  # Feature column order used during training
├── results/
│   ├── latest/                 # Logistic regression results
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   ├── roc_curves.png
│   │   └── pr_curves.png
│   ├── rf/                     # Random Forest results
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── roc_curves.png
│   │   └── pr_curves.png
│   └── model_comparison.png    # Side-by-side LR vs RF metrics
├── ui/
│   └── index.html            # Web dashboard (dark theme, no build step)
├── data/                     # Private — not included in repo (see below)
└── .gitignore
```

---

## Requirements

- Python 3.10+
- The `.venv` virtual environment (included) **or** install dependencies manually

Dependencies used:
```
fastapi
uvicorn[standard]
scikit-learn
pandas
numpy
joblib
matplotlib
```

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd wifi-ids
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" scikit-learn pandas numpy joblib matplotlib
```

### 4. Add the dataset (private)

The AWID3 dataset is not included in this repo. Place the raw CSV files under:

```
data/raw/CSV/<attack_folder>/*.csv
```

Where each folder is named like `1.Deauth`, `2.Disas`, etc.

> If you already have processed data, you can skip straight to **Training**.

---

## Data preprocessing

These steps only need to be run once to prepare the dataset.

### Process raw CSVs for each attack class

Edit `CLASS_FOLDER` in `scripts/process_one_class.py` to the folder you want to process, then run:

```bash
python scripts/process_one_class.py
```

Repeat for each of the 13 attack classes. This writes a `processed.csv` to `data/processed/CSV/<class>/`.

### Build the training sample

Once all 13 classes are processed, create a balanced training file (20,000 rows per class):

```bash
python scripts/train_sample.py
```

Output: `data/processed/train_sample.csv`

---

## Training

### Logistic Regression (baseline)

```bash
python -m src.models.train_baseline
```

Saves:
- `artifacts/models/latest.joblib`
- `artifacts/models/feature_columns.joblib`
- `results/latest/metrics.json`
- `results/latest/confusion_matrix.png`

### Random Forest

```bash
python -m src.models.train_rf
```

Saves:
- `artifacts/models/rf_pipeline.joblib`
- `results/rf/metrics.json`
- `results/rf/confusion_matrix.png`
- `results/rf/feature_importance.png` — top 20 most important features

After training, the script automatically prints a side-by-side comparison against the baseline.

### Model comparison

| Attack Type      | LR F1  | RF F1  | Diff    |
|------------------|--------|--------|---------|
| Deauth           | 0.9987 | 1.0000 | +0.0013 |
| Disas            | 0.9957 | 1.0000 | +0.0043 |
| Re/Assoc         | 0.9995 | 1.0000 | +0.0005 |
| Rogue AP         | 1.0000 | 1.0000 | +0.0000 |
| Krack            | 1.0000 | 1.0000 | +0.0000 |
| Kr00k            | 0.9839 | 1.0000 | +0.0161 |
| SSH              | 0.9569 | 0.9999 | +0.0429 |
| Botnet           | 1.0000 | 1.0000 | +0.0000 |
| Malware          | 0.9813 | 1.0000 | +0.0187 |
| SQL Injection    | 0.9835 | 1.0000 | +0.0165 |
| SSDP             | 0.9674 | 1.0000 | +0.0326 |
| Evil Twin        | 0.9758 | 1.0000 | +0.0242 |
| Website Spoofing | 0.9726 | 0.9999 | +0.0273 |
| **Overall**      | **0.9858** | **1.0000** | **+0.0142** |

RF outperforms logistic regression across all 13 classes. The biggest improvement is SSH (+4.3%), which was the weakest class under the baseline.

---

## Live capture (optional)

To run the model against real network traffic rather than a pre-processed CSV:

### 1. Capture packets

```bash
sudo bash scripts/capture_live.sh [interface] [duration_seconds]
# e.g. sudo bash scripts/capture_live.sh wlan0 60
```

Requires `tshark` (`sudo apt install tshark`). Saves a `.pcap` to `data/captures/`.

### 2. Extract features

```bash
python scripts/extract_features.py --pcap data/captures/capture_YYYYMMDD_HHMMSS.pcap \
                                    --out  data/captures/features.csv
```

Outputs a CSV with the 107 numeric features the model expects. Upload it to the dashboard or pass it to `/predict_summary`.

To label the capture as normal traffic (class 0) for retraining:

```bash
python scripts/extract_features.py --pcap <file> --out <file> --label normal
```

---

## Running the dashboard

Start the API server:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open your browser at **http://127.0.0.1:8000**

The dashboard will:
- Load model info and per-class F1 scores automatically on page open
- Accept a processed CSV file upload
- Show flagged vs clean traffic split, attack distribution chart, and top alerts table

### Quick test

To generate a small test file from your training data:

```bash
head -n 5000 data/processed/train_sample.csv > data/processed/small_test.csv
```

Upload `small_test.csv` in the dashboard and click **Run**.

---

## API endpoints

| Method | Path               | Description                                      |
|--------|--------------------|--------------------------------------------------|
| GET    | `/`                | Serves the web dashboard                         |
| GET    | `/health`          | Health check                                     |
| GET    | `/model_info`      | Returns model accuracy and per-class F1 scores   |
| POST   | `/predict_summary` | Accepts a CSV upload, returns detection summary  |
| GET    | `/docs`            | Auto-generated Swagger UI                        |

### `/predict_summary` parameters

| Parameter   | Type  | Default | Description                                 |
|-------------|-------|---------|---------------------------------------------|
| `top_k`     | int   | 50      | Max number of alerts to return              |
| `min_score` | float | 0.0     | Confidence threshold for flagging a row     |

---

## Notes

- The model is trained on **pre-extracted features** from the AWID3 dataset, not raw packet captures. Input CSVs must contain the same numeric features used during training.
- All 13 classes in this dataset are attack types — there is no "normal" traffic class in the current training set. Use `extract_features.py --label normal` to capture live normal traffic and retrain with a class 0.
- Two models are trained: logistic regression (baseline) and Random Forest. ROC and PR curves for both are in `results/`.
- The RF model scores near-perfect on this dataset (ROC AUC macro: 1.000). Real-world performance will depend on traffic diversity and the presence of a normal class.
