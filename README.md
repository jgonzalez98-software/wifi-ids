# Wi-Fi IDS

A machine learning-based wireless intrusion detection system trained on the [AWID3](https://icsdweb.aegean.gr/awid/awid3) dataset. Detects 13 types of Wi-Fi attacks from pre-processed network traffic features using a logistic regression classifier, served through a FastAPI backend and dark-themed web dashboard.

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
│       └── train_baseline.py # Trains logistic regression pipeline, saves artifacts
├── scripts/
│   ├── process_one_class.py  # Preprocesses raw AWID3 CSV files for one attack class
│   └── train_sample.py       # Builds a balanced training CSV from processed class files
├── artifacts/
│   └── models/
│       ├── latest.joblib           # Trained sklearn pipeline (imputer + scaler + model)
│       └── feature_columns.joblib  # Feature column order used during training
├── results/
│   └── latest/
│       ├── metrics.json        # Classification report + accuracy per class
│       └── confusion_matrix.png
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

Train the baseline logistic regression pipeline:

```bash
python -m src.models.train_baseline
```

This saves:
- `artifacts/models/latest.joblib` — trained pipeline
- `artifacts/models/feature_columns.joblib` — feature schema
- `results/latest/metrics.json` — per-class precision, recall, F1
- `results/latest/confusion_matrix.png` — confusion matrix

### Current baseline results

| Metric          | Score   |
|-----------------|---------|
| Overall accuracy | 98.58% |
| Macro F1        | 98.58%  |
| Weakest class   | SSH (95.69% F1) |

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
- All 13 classes in this dataset are attack types — there is no "normal" traffic class in the current training set.
- The model is a **baseline** logistic regression. Random Forest and anomaly detection models are planned for future iterations.
