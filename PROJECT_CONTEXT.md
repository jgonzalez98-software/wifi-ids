# Wi-Fi IDS — Project Context & Progress

This document tracks the current state of the project, decisions made, and the
remaining work needed. Update it as work progresses.

---

## Goal

Build a real-world Wi-Fi Intrusion Detection System that:
1. Is trained on a mix of **normal** and **attack** traffic — not just attacks
2. Can **classify live captured packets** (not just the AWID3 test set)
3. Has a working end-to-end pipeline: capture → featurize → predict → dashboard

---

## Architecture

```
Live WiFi capture (monitor mode)
        │
        ▼ scripts/capture_live.sh (tshark)
   .pcap file
        │
        ▼ scripts/extract_features.py
   features.csv  (107 numeric columns matching AWID3 schema)
        │
   ┌────┴─────────────────────┐
   │ Retrain path             │ Inference path
   ▼                          ▼
label as 0.Normal         src/inference/detect.py
drop into                       │
data/processed/CSV/0.Normal/    ▼
        │                 src/api/main.py → dashboard
        ▼
  scripts/train_sample.py
        │
        ▼
  src/models/train_rf.py
```

---

## Current State (updated 2026-03-27)

### What works
- RF model trained on 13 attack classes (AWID3), near-perfect F1 on held-out test data
- FastAPI backend serving the dashboard + `/predict_summary`
- Dashboard accepts CSV upload and shows flagged traffic, attack distribution, top alerts
- TP-Link RTL8188EU USB adapter passed through to VM and recognized by Linux
- `iw`, `tshark`, `aircrack-ng` all installed
- Monitor mode confirmed working on interface `wlxbc071d4820f7`

### Blocked on
- **RTL8188EU driver bug**: The mainline `rtl8xxxu` driver enters monitor mode but captures
  0 packets (known issue). The patched aircrack-ng driver (`rtl8188eus`) was cloned and
  `make dkms_install` was run, but the module did not appear under
  `/lib/modules/6.17.0-14-generic` — likely a build failure against this very new kernel.
  Need to confirm dkms status and fix the build before capture can proceed.

### The problem being solved
**All training data is attacks-only.** The model has no concept of "normal" traffic.
Feeding it real-world mixed traffic causes every packet to be classified as one of the
13 attack types — there is no class 0 to "escape" to.

**AWID3 does not include a normal traffic class in the folders on this machine.**
(Confirmed: only `1.Deauth` through `13.Website_spoofing` are present under `data/raw/CSV/`.)

---

## Steps Taken

### Phase 1 — AWID3 baseline (complete)
- [x] Processed all 13 AWID3 attack class CSVs with `scripts/process_one_class.py`
- [x] Built balanced training sample (20 k rows × 13 classes) with `scripts/train_sample.py`
- [x] Trained Logistic Regression baseline (`src/models/train_baseline.py`)
- [x] Trained Random Forest — overall F1 = 1.0000 on AWID3 test set (`src/models/train_rf.py`)
- [x] Dashboard working end-to-end

### Phase 2 — Real-world capture pipeline (in progress)
- [x] `scripts/capture_live.sh` — puts adapter in monitor mode, runs tshark capture
- [x] `scripts/extract_features.py` — converts .pcap → model-ready CSV (exactly 107 features)
- [x] Added class 0 ("Normal") to `LABELS` in `train_rf.py` and the dashboard color/label maps
- [x] TP-Link RTL8188EU USB adapter connected and passed through to VM
- [x] Installed: `iw`, `tshark`, `aircrack-ng`
- [x] Added user to `wireshark` group (`sudo usermod -aG wireshark $USER`)
- [x] Monitor mode enabled on `wlxbc071d4820f7`
- [ ] **FIX RTL8188EU DRIVER** — see troubleshooting section below
- [ ] Capture actual normal traffic → `data/captures/normal.pcap`
- [ ] Extract features → `data/processed/CSV/0.Normal/processed.csv`
- [ ] Retrain RF with normal class included
- [ ] Evaluate: confusion matrix should show class 0 separated from attacks
- [ ] Run live test: capture mixed traffic, upload to dashboard, verify normal != flagged

### Phase 3 — Anomaly detection (planned, not started)
- [ ] Train Isolation Forest / One-Class SVM on normal traffic only
- [ ] Use as a binary pre-filter before the 13-class classifier
- [ ] Useful if the normal-class retraining approach still bleeds false positives

---

## How to Capture Normal Traffic and Add It to Training

### Step 1 — Install tshark (if not installed)
```bash
sudo apt install tshark
```

### Step 2 — Capture normal WiFi traffic
```bash
sudo bash scripts/capture_live.sh wlan0 120 data/captures/normal_capture.pcap
# Replace wlan0 with your interface name (check: ip link show)
# 120 = capture for 2 minutes while browsing normally
```

If monitor mode won't enable on `wlan0` try:
```bash
sudo airmon-ng start wlan0
# Then use wlan0mon as the interface name
sudo bash scripts/capture_live.sh wlan0mon 120 data/captures/normal_capture.pcap
```

### Step 3 — Extract features
```bash
python scripts/extract_features.py \
  --pcap data/captures/normal_capture.pcap \
  --out  data/captures/normal_features.csv \
  --label normal
```

### Step 4 — Add to training data
```bash
mkdir -p data/processed/CSV/0.Normal
cp data/captures/normal_features.csv data/processed/CSV/0.Normal/processed.csv
```

### Step 5 — Rebuild training sample and retrain
```bash
python scripts/train_sample.py
python -m src.models.train_rf
```

### Step 6 — Test with mixed traffic
Capture again (normally + trigger some known attack like a deauth frame), then:
```bash
python scripts/extract_features.py \
  --pcap data/captures/mixed_capture.pcap \
  --out  data/captures/mixed_features.csv
# Upload mixed_features.csv in the dashboard
```

---

## Feature Schema

The model uses **107 features** — all standard tshark/Wireshark field names:
- `frame.*` — frame-level metadata (always present)
- `radiotap.*` / `wlan_radio.*` — 802.11 radio layer (present in monitor mode captures)
- `wlan.*` — 802.11 MAC header fields
- `eapol.*` / `wlan_rsna_eapol.*` — WPA handshake fields (present during auth)
- `arp.*`, `icmpv6.*`, `tcp.*` — network layer (present for matching traffic)
- `smb.*`, `smb2.*`, `dhcp.*`, `dns.*`, `http.*` — application layer (sparse)
- `ssh.*`, `tls.*` — SSH and TLS fields (sparse, very useful for SSH/SSDP attacks)

Most fields will be NaN for most packets. The `SimpleImputer(strategy="median")`
in the pipeline handles this during training and inference.

---

## Model Artifacts

| File | Description |
|---|---|
| `artifacts/models/rf_pipeline.joblib` | RF pipeline (imputer + RandomForestClassifier) |
| `artifacts/models/latest.joblib` | LR baseline pipeline |
| `artifacts/models/feature_columns.joblib` | Ordered list of 107 feature column names |
| `results/rf/metrics.json` | RF classification report |
| `results/latest/metrics.json` | LR classification report |

---

## Class Label Map

| attack_id | Label |
|---|---|
| 0 | Normal (to be added) |
| 1 | Deauth |
| 2 | Disas |
| 3 | Re/Assoc |
| 4 | Rogue AP |
| 5 | Krack |
| 6 | Kr00k |
| 7 | SSH |
| 8 | Botnet |
| 9 | Malware |
| 10 | SQL Injection |
| 11 | SSDP |
| 12 | Evil Twin |
| 13 | Website Spoofing |

---

## Troubleshooting: RTL8188EU Driver (BLOCKER)

### Problem
The mainline `rtl8xxxu` driver supports monitor mode in name only — it reports
"entered promiscuous mode" in dmesg but tshark captures 0 packets. This is a known
upstream bug with this driver + RTL8188EU chip.

### What was tried
1. Manually set monitor mode via `iw` → confirmed `type monitor` in `iw dev info`
2. Ran tshark with `sg wireshark` (group workaround) → 0 packets
3. Channel hopped channels 1-11 while capturing → still 0 packets
4. Cloned `https://github.com/aircrack-ng/rtl8188eus` and ran `sudo make dkms_install`
   → module did NOT appear in `/lib/modules/6.17.0-14-generic/`
5. `dkms status` → failed because `dkms` package itself was not installed

### Next steps to fix
```bash
# 1. Install dkms and kernel headers (required to build the patched driver)
sudo apt install -y dkms linux-headers-$(uname -r)

# 2. Check if the rtl8188eus source is already cloned
ls ~/rtl8188eus   # if missing: git clone https://github.com/aircrack-ng/rtl8188eus

# 3. Build and install
cd ~/rtl8188eus
sudo make dkms_install

# 4. Verify the module was built
sudo dkms status
# Should show: 8188eu/1.0, 6.17.0-14-generic, x86_64: installed

# 5. If dkms_install fails on kernel 6.17, check build log:
sudo dkms build 8188eu/1.0 2>&1 | tail -30

# 6. Load the new driver
sudo rmmod rtl8xxxu 2>/dev/null || true
sudo modprobe 8188eu

# 7. Re-enable monitor mode
sudo ip link set wlxbc071d4820f7 down
sudo iw dev wlxbc071d4820f7 set type monitor
sudo ip link set wlxbc071d4820f7 up
```

### If kernel 6.17 is too new for the patched driver
The rtl8188eus repo may not yet have patches for kernel 6.17. Options:
- Check open issues/PRs on https://github.com/aircrack-ng/rtl8188eus
- Try the `v5.3.9` branch: `git checkout v5.3.9` before building
- Downgrade to an older kernel temporarily (not recommended)
- Use a different USB adapter (e.g. Alfa AWUS036ACH uses mt76 driver — much better support)

---

## Known Issues / Decisions

- **Perfect F1 on AWID3 is suspicious** — likely overfitting to dataset artifacts or the
  balanced sampling producing too-easy separation. Real-world F1 will be lower. This is
  expected and normal; the AWID3 test shows the features are meaningful.

- **No normal class yet** — every packet currently gets a 1–13 label. Until Phase 2 is
  complete, the model cannot be trusted on real-world traffic.

- **tshark not installed** — needs `sudo apt install tshark` before capture scripts work.

- **Monitor mode** — requires root + a compatible WiFi adapter. Some adapters (especially
  built-in Intel cards) have limited monitor mode support. A USB adapter like Alfa AWUS036ACH
  works reliably.
