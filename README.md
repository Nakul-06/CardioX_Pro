# 🫀 CardioX Pro: Clinical Cardiovascular Disease Risk Prediction & Longitudinal Monitoring System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Scikit-Learn & XGBoost](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)

An enterprise-grade Clinical Decision Support System (CDSS) for real-time cardiovascular disease risk assessment, longitudinal vital trajectory tracking, adaptive streaming incremental learning, and multi-cohort cross-dataset validation.

---

## 🌟 Key Capabilities

1. **⚡ Real-Time Clinical Inference:**
   - Evaluates patient demographics, vitals, laboratory tests, and lifestyle habits with sub-millisecond latency.
   - Built-in patient case presets (Low Risk, Moderate Risk, High Risk) for instant clinical simulation.
2. **🤖 Multi-Algorithm Consensus Engine:**
   - Concurrently executes **5 trained supervised classifiers** (XGBoost, Random Forest, Gradient Boosting, Logistic Regression, Decision Tree) side-by-side to guarantee clinical agreement.
3. **📈 Longitudinal Health Tracking:**
   - Tracks multi-visit velocity changes ($\Delta\text{Systolic BP}$, $\Delta\text{Cholesterol}$, $\Delta\text{BMI}$) across consecutive patient checkups.
   - Trajectory classification (Escalating Risk 🚨 vs. Positive Recovery 🟢) to detect arterial degradation before acute events occur.
4. **🔄 Adaptive Incremental Learning (Phase 5):**
   - Continuously adapts to incoming hospital EHR patient streams (500 patients/batch) using online mini-batch `partial_fit` (SGDClassifier with log loss).
   - Monitors distributional shifts in real time using the **Wasserstein distance concept drift metric** (+7.0% peak adaptation gain).
5. **🌐 Cross-Dataset Generalization (Phase 6):**
   - Harmonizes 5 universal clinical dimensions across the **70,000 Russian Cohort**, **UCI Cleveland Clinic (303 records)**, and **Statlog (270 records)**.
   - Generates a 3x3 Cross-Domain Transfer Matrix (**98.2% ROC-AUC** transfer from Cleveland to Statlog), confirming clinical domain invariance.
6. **📋 Evidence-Based Clinical Recommendations Engine:**
   - Tailored clinical action plans mapped directly to:
     - **2017 ACC/AHA High Blood Pressure Clinical Practice Guidelines**
     - **2018 AHA/ACC Multi-Society Guideline on Blood Cholesterol**
     - **CDC Preventative Tobacco Protocols**
     - **AHA/ACSM Exercise Standards**
     - **ADA Standards of Medical Care in Diabetes**

---

## 📊 Clinical Benchmark & Model Performance

In fulfillment of project specification Objective 2, SMOTE synthetic oversampling was performed across the cleaned cohort prior to partitioning to guarantee balanced 50/50 prior distributions across both splits:
- **Balanced Training Partition (80%):** 55,443 patient records (50.0% healthy / 50.0% CVD)
- **Balanced Validation Partition (20%):** 13,861 patient records (50.0% healthy / 50.0% CVD)

| Model | Accuracy | F1-Score | Recall (Sensitivity) | Specificity | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Gradient Boosting** | **73.22%** | **71.94%** | **68.67%** | **77.77%** | **79.69%** | ⭐ **Champion F1** |
| 🥈 **XGBoost** | 73.13% | **71.94%** | 68.89% | 77.38% | 79.57% | Top Recall |
| 🥉 **Random Forest** | **73.30%** | 71.92% | 68.38% | **78.21%** | **79.70%** | Top Accuracy & Specificity |
| 4 **Logistic Regression** | 72.64% | 71.26% | 67.85% | 77.42% | 78.74% | Linear Reference |
| 5 **Decision Tree** | 72.51% | 71.69% | **69.62%** | 75.39% | 77.71% | Interpretable Tree |

---

## 🏗️ Repository Architecture

```
CardioX_Pro/
├── app_streamlit.py                 # Primary CardioX Pro Streamlit Application (6 Modules)
├── streamlit_app.py                 # Synchronized Streamlit Cloud Deployment Entrypoint
├── main.py                          # Phase 1: 8-Step Preprocessing & Pre-Split SMOTE Pipeline
├── train_models.py                  # Phases 2 & 3: Supervised Model Training & Evaluation
├── incremental_learning.py          # Phase 5: Streaming Online Learning & Drift Tracking
├── cross_dataset_validation.py      # Phase 6: Multi-Cohort Harmonization & Transfer Matrix
├── dataset_70000_patients.csv       # Primary 70,000-Patient Clinical Screening Cohort
├── cleveland_raw.csv                # UCI Cleveland Clinic Benchmark Dataset (303 Records)
├── statlog_raw.dat                  # Statlog Heart Disease Benchmark Dataset (270 Records)
├── requirements.txt                 # Optimized Deployment Dependencies
├── PROJECT_DEFENSE_GUIDE.md         # Comprehensive Project Defense & Viva Guide
├── models/
│   ├── best_cardio_model.joblib     # Serialized Champion Model
│   ├── xgboost_model.joblib         # XGBoost Classifier
│   ├── random_forest_model.joblib   # Random Forest Classifier
│   ├── gradient_boosting_model.joblib # Gradient Boosting Classifier
│   ├── decision_tree_model.joblib   # Decision Tree Classifier
│   ├── logistic_regression_model.joblib # Logistic Regression Classifier
│   ├── incremental_online_model.joblib # Online SGD Incremental Model
│   ├── feature_scaler.joblib        # Fitted StandardScaler (12 features)
│   ├── model_comparison.json       # Supervised Evaluation Benchmark Metrics
│   ├── incremental_learning_metrics.json # Streaming Batch Adaptation Metrics
│   └── cross_dataset_metrics.json   # 3x3 Cross-Domain Generalization Matrix
├── charts/
│   ├── correlation_heatmap.png      # Feature Correlation Matrix
│   ├── feature_importance_ranking.png # Multi-Criterion Consensus Feature Ranking
│   ├── smote_class_distribution.png # Pre-Split SMOTE Balancing Plot
│   ├── incremental_learning_curve.png # Online Streaming Learning Curves
│   └── cross_dataset_generalization_matrix.png # Multi-Cohort Transferability Heatmap
└── processed_data/
    ├── train_70k_smote_balanced.csv # Balanced Training Split (55,443 rows)
    ├── test_70k_held_out.csv        # Balanced Validation Split (13,861 rows)
    └── feature_importance_rankings.csv # 12-Feature Consensus Weights
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

3. **Run the Complete End-to-End Pipeline (Optional - Pre-Generated Artifacts Included):**
   ```bash
   python main.py                      # Data preparation & pre-split SMOTE
   python train_models.py              # Supervised model training
   python incremental_learning.py      # Adaptive online streaming simulation
   python cross_dataset_validation.py  # Multi-cohort benchmark validation
   ```

4. **Launch the Web Application:**
   ```bash
   python -m streamlit run app_streamlit.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This repository is pre-configured for instant zero-configuration deployment on **Streamlit Community Cloud**:
- **Main file path:** `streamlit_app.py`
- **Environment specification:** `.python-version` set to `3.11`
- **Native system dependencies:** `packages.txt` (`libgomp1` for OpenMP support)
