# 🫀 CardioX Pro: Clinical Cardiovascular Disease Risk Prediction & Longitudinal Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Scikit-Learn & XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)

An enterprise-grade Medical Decision Support System (CDSS) for real-time cardiovascular disease risk assessment, multi-algorithm consensus validation, and longitudinal vital trajectory tracking.

---

## 🌟 Key Capabilities

1. **⚡ Real-Time Clinical Inference:**
   - Evaluates patient demographics, vitals, laboratory tests, and lifestyle habits with sub-millisecond latency.
   - Built-in patient case presets for instant clinical simulation.
2. **🤖 Multi-Algorithm Consensus Engine:**
   - Concurrently executes **5 trained supervised classifiers** (XGBoost, Random Forest, Gradient Boosting, Logistic Regression, Decision Tree) side-by-side to guarantee clinical agreement.
3. **📈 Longitudinal Health Tracking:**
   - Tracks multi-visit velocity changes ($\Delta\text{Systolic BP}$, $\Delta\text{Cholesterol}$, $\Delta\text{BMI}$) across consecutive patient checkups.
   - Trajectory classification (Escalating Risk 🚨 vs. Positive Recovery 🟢) to detect arterial degradation before acute events occur.
4. **📋 Dynamic Evidence-Based Clinical Recommendations Engine:**
   - Generates tailored clinical action plans mapped directly to:
     - **2017 ACC/AHA High Blood Pressure Clinical Practice Guidelines**
     - **2018 AHA/ACC Multi-Society Guideline on Blood Cholesterol**
     - **CDC & ACC Preventative Tobacco Protocols**
     - **AHA/ACSM Exercise Prescription Standards**
     - **ADA Standards of Medical Care in Diabetes**

---

## 📊 Clinical Benchmark & Model Performance

Trained on **55,442 SMOTE-balanced patient records** and strictly evaluated on **13,717 held-out validation patients** (zero data leakage):

| Model | Accuracy | F1-Score | Recall (Sensitivity) | Specificity | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **XGBoost** | **73.05%** | **71.58%** | **68.61% (4,656 CVD cases)** | **77.39%** | **79.41%** | ⭐ **Champion Model** |
| 🥈 **Gradient Boosting** | 73.03% | 71.43% | 68.16% | 77.81% | **79.55%** | Top ROC-AUC |
| 🥉 **Random Forest** | 73.03% | 71.46% | 68.24% | 77.71% | 79.48% | Top Bagging Ensemble |
| 4 **Decision Tree** | 72.43% | 71.24% | 69.01% | 75.78% | 77.67% | Tree Baseline |
| 5 **Logistic Regression** | 72.40% | 70.67% | 67.21% | 77.48% | 78.59% | Linear Baseline |

---

## 🏗️ Repository Architecture

```
CardioX_Pro/
├── app_streamlit.py                 # Primary CardioX Pro Streamlit Application
├── streamlit_app.py                 # Streamlit Cloud Deployment Entrypoint
├── main.py                          # 8-Step Preprocessing & Feature Engineering Pipeline
├── train_models.py                  # Model Training & Benchmark Evaluation Script
├── dataset_70000_patients.csv       # Sourced 70,000-Patient Clinical Cohort
├── requirements.txt                 # Optimized Deployment Dependencies
├── PROJECT_DEFENSE_GUIDE.md         # Complete Project Presentation & Viva Guide
├── models/
│   ├── best_cardio_model.joblib     # Serialized Champion Model (XGBoost)
│   ├── xgboost_model.joblib         # XGBoost Classifier
│   ├── random_forest_model.joblib   # Random Forest Classifier
│   ├── gradient_boosting_model.joblib # Gradient Boosting Classifier
│   ├── decision_tree_model.joblib   # Decision Tree Classifier
│   ├── logistic_regression_model.joblib # Logistic Regression Classifier
│   ├── feature_scaler.joblib        # Fitted StandardScaler (Zero Leakage)
│   └── model_comparison.json       # Quantitative Evaluation Benchmark Metrics
├── charts/
│   ├── correlation_heatmap.png      # Feature Correlation Matrix
│   ├── feature_importance_ranking.png # Multi-Criterion Consensus Rankings
│   └── smote_class_distribution.png # Class Balancing Verification Plot
└── processed_data/
    ├── train_70k_smote_balanced.csv # Balanced Training Partition (54,867 rows)
    ├── test_70k_held_out.csv        # Independent Validation Split (13,717 rows)
    └── feature_importance_rankings.csv # Consensus Feature Weights
```

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nakul-06/CardioX_Pro.git
   cd CardioX_Pro
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Web Application:**
   ```bash
   python -m streamlit run app_streamlit.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to your GitHub account (`Nakul-06/CardioX_Pro`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select:
   - **Repository:** `Nakul-06/CardioX_Pro`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py` (or `app_streamlit.py`)
4. Click **"Deploy!"** — the application will automatically build and launch in under 1 minute.

---

## 📜 Clinical Disclaimer
*CardioX Pro is developed as an academic and clinical research decision support tool. It is intended to assist medical professionals with quantitative risk stratification and should not replace individualized clinical judgement or diagnostic procedures.*
