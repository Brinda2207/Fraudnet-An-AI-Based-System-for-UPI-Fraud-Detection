# 🛡️ FraudNet: An AI-Based System for UPI Fraud Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Supervised-FF6600?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Autoencoder-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A real-time hybrid fraud detection system combining supervised learning, anomaly detection, graph analysis, and explainable AI — deployed on the web.**

[🚀 Live Demo](#deployment) • [📊 Model Results](#model-results) • [⚙️ Setup](#setup--installation) • [📁 Project Structure](#project-structure)

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Models & Methodology](#models--methodology)
- [Model Results](#model-results)
- [Explainable AI (SHAP)](#explainable-ai-shap)
- [Graph-Based Fraud Detection](#graph-based-fraud-detection)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Deployment](#deployment)
- [Team](#team)

---

## 🔍 Overview

**FraudNet** is an end-to-end AI system built to detect fraudulent UPI (Unified Payments Interface) transactions in real time. The system combines:

- ✅ **Supervised Learning** — XGBoost ensemble trained on 98,000 labelled transactions
- ✅ **Unsupervised Anomaly Detection** — Autoencoder + LOF + K-Means
- ✅ **Graph-Based Detection** — Mule account and circular transaction identification
- ✅ **Explainable AI** — SHAP waterfall charts per prediction
- ✅ **Web Deployment** — Flask backend hosted on Render with an interactive HTML/CSS frontend

The final prediction uses a **hybrid ensemble score**:

```
Hybrid Score = 0.70 × XGBoost_prob + 0.30 × Autoencoder_MSE_norm + Amount_Risk_Boost
```

A transaction is flagged as fraud if `Hybrid Score ≥ 0.40`.

---

## 🏗️ System Architecture

```
User Input (Web Form)
        │
        ▼
┌─────────────────────────────────────────────────┐
│                  Flask Backend                  │
│                                                 │
│  ┌─────────────┐    ┌──────────────────────┐   │
│  │  Scaler +   │───▶│  XGBoost Classifier  │   │
│  │ Preprocessor│    │  (Supervised, 70%)   │   │
│  └─────────────┘    └──────────────────────┘   │
│         │                      │                │
│         ▼                      │                │
│  ┌─────────────┐               │                │
│  │ Autoencoder │               │                │
│  │ (NumPy fwd) │               │                │
│  │ Unsupervised│               │                │
│  │   (30%)     │               │                │
│  └─────────────┘               │                │
│         │           ┌──────────┘                │
│         ▼           ▼                           │
│     ┌───────────────────┐                       │
│     │  Hybrid Score     │◀── Amount Risk Boost  │
│     │  + Threshold 0.40 │                       │
│     └───────────────────┘                       │
│              │                                  │
│     ┌────────┴────────┐                         │
│     ▼                 ▼                         │
│  SHAP Chart     Graph Network                   │
│  (RF Explainer) (NetworkX)                      │
└─────────────────────────────────────────────────┘
        │
        ▼
  Result Page (Fraud / Legitimate + Risk Score + Explanation)
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **Source** | [PaySim — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Size** | 6.3 million transactions (744 hours simulated) |
| **Fraud Rate** | ~1.3% raw; resampled to ~8% for training |
| **Training Set** | 90,000 legitimate + 8,000 fraud transactions |
| **Type** | Mobile money / UPI-style financial transactions |

> **Note:** UPI-specific features (device type, location, payment method, device change flag) were synthetically generated using domain-realistic distributions and appended to the PaySim dataset to align with the Indian UPI context.

---

## 🧩 Features Used

| Feature | Description |
|---------|-------------|
| `Transaction_Amount` | Amount in ₹ (clipped at ₹2,00,000) |
| `Transaction_Type` | TRANSFER=0, PAYMENT=1, WITHDRAWAL=2 |
| `Device_Used` | Mobile=0, Desktop=1, Tablet=2 |
| `Location` | Local=0, Domestic=1, International=2 |
| `Previous_Fraudulent_Transactions` | Count of prior frauds by user |
| `Account_Age` | Days since account creation |
| `Number_of_Transactions_Last_24H` | Recent transaction frequency |
| `Payment_Method` | UPI=0, Net Banking=1, Card=2 |
| `Hour` | Hour of day (0–23) |
| `night_txn` | 1 if transaction between 10 PM – 6 AM |
| `device_change` | 1 if new/unknown device used |
| `User_ID` | Anonymised user identifier |

---

## 🤖 Models & Methodology

### Role 1 — Supervised Learning (Brinda)

#### Preprocessing
- Missing value imputation (median/mode)
- Duplicate removal
- Label encoding for categorical features
- StandardScaler normalisation
- Class imbalance handled via **undersampling** (90k legit) + **oversampling** (8k fraud with replacement)
- Stratified 80/20 train-test split

#### Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline |
| Naive Bayes | Baseline |
| Decision Tree | Depth-unlimited |
| Random Forest | 100 estimators, n_jobs=-1 |
| XGBoost | colsample_bylevel=0.7 to reduce feature dominance |
| LightGBM | max_depth=10, learning_rate=0.1 |
| Soft Voting Ensemble | RF + XGBoost + LightGBM |
| Weighted Ensemble | XGBoost weight×2 |

> Best model auto-selected by F1 score → **XGBoost** saved as `final_fraud_model.pkl`

#### Threshold Tuning
- Optimal XGBoost-only threshold: **0.25** (saved in `optimal_threshold.pkl`)
- Searched over `np.arange(0.1, 0.95, 0.05)` for best F1

---

### Role 2 — Anomaly Detection & Explainable AI (Rajeshwari)

#### Autoencoder
- Architecture: `12 → 8 (ReLU) → 12 (Linear)`
- Trained on **20,000 normal-only transactions** (unsupervised)
- Fraud detection via reconstruction error (MSE) with 95th percentile threshold
- Inference runs as **pure NumPy** (no TensorFlow dependency at runtime)

#### Local Outlier Factor (LOF)
- `n_neighbors=20`, `contamination=0.1`
- Density-based anomaly scoring on 10,000-sample subset

#### K-Means Clustering
- 2 clusters, distance-to-centroid as anomaly score
- 90th percentile distance threshold

#### Hybrid Ensemble (4-Model Research Version)
| Model | Weight |
|-------|--------|
| XGBoost (supervised) | 50% |
| Autoencoder (unsupervised) | 25% |
| LOF (unsupervised) | 15% |
| K-Means (unsupervised) | 10% |

> Production app uses simplified 2-model formula (XGBoost 70% + Autoencoder 30%) calibrated for real-time inference speed.

#### Graph-Based Fraud Detection
- Transaction–user graph built with **NetworkX**
- **Mule account detection**: users with >10 transactions/24h AND prior fraud history
- **Community detection**: Greedy modularity communities
- **Degree centrality** analysis for hub identification
- Visualised with colour-coded fraud/mule/normal nodes

#### SHAP Explainability
- `TreeExplainer` on Random Forest
- **Global**: Bar plot + Beeswarm plot
- **Local**: Waterfall chart per transaction
- Natural language explanation generated per prediction

---

## 📊 Model Results

### Supervised Model Comparison

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | — | — | — | — |
| Naive Bayes | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** ⭐ | — | — | — | — |
| LightGBM | — | — | — | — |
| Soft Voting Ensemble | — | — | — | — |
| Weighted Ensemble | — | — | — | — |

> ⭐ Fill in actual values from your notebook output before final submission

### Anomaly Detection Comparison

| Method | Type | Approach |
|--------|------|---------|
| Autoencoder | Unsupervised | Reconstruction error |
| LOF | Unsupervised | Density-based |
| K-Means | Unsupervised | Distance-based |
| Random Forest | Supervised | Label-based |

### End-to-End Prediction Verification

| Test Case | XGBoost | AE Score | Hybrid Score | Result |
|-----------|---------|----------|--------------|--------|
| Legit (₹500, day, local) | 0.001 | 0.000 | 0.001 | ✅ LEGIT |
| Fraud (₹5L, night, intl, prev fraud) | 0.956 | 1.000 | 1.000 | 🚨 FRAUD |
| Moderate Legit (₹5000, afternoon) | 0.000 | 0.000 | 0.000 | ✅ LEGIT |
| High-Risk (₹80k, night, new device) | 0.590 | 0.638 | 0.755 | 🚨 FRAUD |

---

## 🔎 Explainable AI (SHAP)

Each prediction generates a **SHAP waterfall chart** showing:
- Which features pushed the score **toward fraud** (red bars)
- Which features pushed the score **toward legitimate** (blue bars)
- A natural-language explanation of the top contributing factors

Example output:
> *"This transaction was flagged as FRAUD mainly because **Prev Frauds** increased the fraud risk, along with **Night Txn?** and **Device Change?**."*

---

## 🕸️ Graph-Based Fraud Detection

The app builds a real-time **transaction network graph** for each prediction:
- 🟡 **Yellow** — Current transaction's user
- 🔴 **Red** — Known fraud accounts in same location
- 🟠 **Orange** — High-risk / mule accounts
- 🔵 **Blue** — Normal accounts

The graph explains network-level risk: *"This user shares a location with 2 fraud accounts and 1 high-risk account out of 14 nearby users — raising network-level suspicion."*

---

## 📁 Project Structure

```
Fraudnet-An-AI-Based-System-for-UPI-Fraud-Detection/
│
├── app.py                          # Flask backend — prediction, SHAP, graph
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render deployment config
├── .python-version                 # Python version pin
│
├── models/
│   ├── final_fraud_model.pkl       # XGBoost (best supervised model)
│   ├── scaler.pkl                  # StandardScaler (fitted on training data)
│   ├── feature_list.pkl            # Ordered feature names (12 features)
│   ├── optimal_threshold.pkl       # XGBoost-only threshold (0.25)
│   ├── hybrid_threshold.pkl        # Production hybrid threshold (calibrated)
│   ├── hybrid_threshold_4model.pkl # Research 4-model threshold
│   ├── mse_stats.pkl               # Autoencoder MSE p50/p95 for normalisation
│   ├── rf_model.pkl                # Random Forest (for SHAP explanations)
│   └── autoencoder_weights.weights.h5  # Autoencoder weights (NumPy inference)
│
├── templates/
│   └── home.html                   # Frontend UI (transaction form + results)
│
└── notebooks/
    ├── 01_eda.ipynb                 # Data loading, preprocessing, EDA, visualisations
    ├── 02_modeling.ipynb            # Supervised models, ensemble, threshold tuning
    └── role_2_updated.ipynb        # Autoencoder, LOF, K-Means, graph, SHAP
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Brinda2207/Fraudnet-An-AI-Based-System-for-UPI-Fraud-Detection.git
cd Fraudnet-An-AI-Based-System-for-UPI-Fraud-Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

### Dependencies

```
flask
numpy
pandas
scikit-learn
xgboost
lightgbm
joblib
h5py
shap
networkx
matplotlib
```

---

## 🚀 Deployment

The app is deployed on **Render** using the `render.yaml` configuration.

**Live URL:** `https://your-render-url.onrender.com` *(replace with your actual URL)*

### How the hybrid inference works on Render (no GPU needed):
- XGBoost runs normally via `joblib`
- Autoencoder weights are loaded via `h5py` and inference runs as **pure NumPy matrix operations** — no TensorFlow installed on the server

---

## 👩‍💻 Team

| Role | Name | Responsibilities |
|------|------|-----------------|
| **ML Core, Models & Ensemble** | Brinda | Dataset, preprocessing, 5 supervised models, ensemble learning, evaluation, threshold tuning |
| **Advanced Detection & Explainable AI** | Rajeshwari | Autoencoder, LOF, K-Means, hybrid ensemble, graph-based detection, SHAP |
| **System Design, Backend & Frontend** | Nisarga | Flask API, HTML/CSS UI, model integration, Render deployment |

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ as part of our final year project — FraudNet
</div>
