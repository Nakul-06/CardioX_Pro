"""
Phase 6: Cross-Dataset Validation & Generalization Module.
Evaluates model robustness across three geographically and clinically distinct cohorts:
1. Primary 70,000-Patient Cohort (Cardiovascular Health Screening)
2. UCI Cleveland Clinic Foundation Dataset (303 Cardiac Angiography Patients)
3. Statlog Heart Disease Benchmark Cohort (270 Clinical Cardiology Patients)

Harmonizes clinical features into a unified multi-cohort feature space:
- Age (Years)
- Biological Sex (0: Female, 1: Male)
- Resting Systolic Blood Pressure (mm Hg)
- Serum Cholesterol Level (1: Normal <200, 2: Borderline 200-239, 3: High >=240 mg/dL)
- Fasting Glucose Dysglycemia Status (0: Normal, 1: Elevated >120 mg/dL)
- Target: Binary CVD Status (0: Absent, 1: Present)

Computes the 3x3 Cross-Domain Generalization Matrix, Transferability Gap, and Invariance Score.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
MODELS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

HARMONIZED_FEATURES = ["age", "sex", "systolic_bp", "cholesterol_level", "fasting_glucose"]
TARGET_COL = "target"


def load_and_harmonize_70k(sample_size=10000):
    """Loads and harmonizes the 70k Russian Cardiovascular Cohort."""
    path = BASE_DIR / "dataset_70000_patients.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    # Basic cleaning
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 240) & (df["ap_lo"] >= 40) & (df["ap_hi"] > df["ap_lo"])].copy()

    df_harm = pd.DataFrame({
        "age": (df["age"] / 365.25).round(1),
        "sex": df["gender"].map({1: 0, 2: 1}),
        "systolic_bp": df["ap_hi"].astype(float),
        "cholesterol_level": df["cholesterol"].astype(int),  # 1, 2, 3
        "fasting_glucose": (df["gluc"] > 1).astype(int),     # 0: normal, 1: elevated
        "target": df["cardio"].astype(int)
    })

    if sample_size and sample_size < len(df_harm):
        df_harm = df_harm.sample(n=sample_size, random_state=42).reset_index(drop=True)

    df_harm["cohort"] = "70k Cohort"
    logger.info(f"Loaded 70k Cohort: {len(df_harm):,} harmonized records.")
    return df_harm


def load_and_harmonize_cleveland():
    """Loads and harmonizes UCI Cleveland Clinic Dataset (303 records)."""
    path = BASE_DIR / "cleveland_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?")
    df = df.dropna(subset=["trestbps", "chol", "fbs", "num"]).copy()

    # Map continuous cholesterol to ordinal 1, 2, 3 matching clinical guidelines
    def map_chol(c):
        if c < 200:
            return 1
        elif c < 240:
            return 2
        else:
            return 3

    df_harm = pd.DataFrame({
        "age": df["age"].astype(float),
        "sex": df["sex"].astype(int),
        "systolic_bp": df["trestbps"].astype(float),
        "cholesterol_level": df["chol"].apply(map_chol).astype(int),
        "fasting_glucose": df["fbs"].astype(int),
        "target": (df["num"] > 0).astype(int)
    })

    df_harm["cohort"] = "Cleveland (303)"
    logger.info(f"Loaded Cleveland Cohort: {len(df_harm):,} harmonized records.")
    return df_harm


def load_and_harmonize_statlog():
    """Loads and harmonizes Statlog Heart Disease Dataset (270 records)."""
    path = BASE_DIR / "statlog_raw.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target_raw"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)

    def map_chol(c):
        if c < 200:
            return 1
        elif c < 240:
            return 2
        else:
            return 3

    df_harm = pd.DataFrame({
        "age": df["age"].astype(float),
        "sex": df["sex"].astype(int),
        "systolic_bp": df["trestbps"].astype(float),
        "cholesterol_level": df["chol"].apply(map_chol).astype(int),
        "fasting_glucose": df["fbs"].astype(int),
        "target": (df["target_raw"] == 2).astype(int)  # 1 is absence (0), 2 is presence (1)
    })

    df_harm["cohort"] = "Statlog (270)"
    logger.info(f"Loaded Statlog Cohort: {len(df_harm):,} harmonized records.")
    return df_harm


def run_cross_dataset_validation():
    """
    Executes cross-dataset evaluation across the 3 cohorts and generates the generalization matrix.
    """
    logger.info("=" * 80)
    logger.info("PHASE 6: CROSS-DATASET VALIDATION & GENERALIZATION BENCHMARK")
    logger.info("=" * 80)

    c_70k = load_and_harmonize_70k(sample_size=10000)
    c_cleveland = load_and_harmonize_cleveland()
    c_statlog = load_and_harmonize_statlog()

    cohorts = {
        "70k Cohort": c_70k,
        "Cleveland (303)": c_cleveland,
        "Statlog (270)": c_statlog
    }

    cohort_names = list(cohorts.keys())
    results_matrix = {
        "roc_auc": np.zeros((len(cohort_names), len(cohort_names))),
        "accuracy": np.zeros((len(cohort_names), len(cohort_names))),
        "f1_score": np.zeros((len(cohort_names), len(cohort_names)))
    }

    detailed_evaluations = []

    # Model for cross-cohort evaluation: Random Forest with calibrated hyperparameters
    for i, train_name in enumerate(cohort_names):
        df_train = cohorts[train_name]
        X_train = df_train[HARMONIZED_FEATURES]
        y_train = df_train[TARGET_COL]

        # Standardize features within training cohort
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Random Forest classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)

        # Train Logistic Regression as linear reference
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)

        for j, test_name in enumerate(cohort_names):
            df_test = cohorts[test_name]
            X_test = df_test[HARMONIZED_FEATURES]
            y_test = df_test[TARGET_COL]

            # Scale using the training cohort scaler to test domain transfer
            X_test_scaled = scaler.transform(X_test)

            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred))
            auc = float(roc_auc_score(y_test, y_prob))

            results_matrix["roc_auc"][i, j] = round(auc * 100, 2)
            results_matrix["accuracy"][i, j] = round(acc * 100, 2)
            results_matrix["f1_score"][i, j] = round(f1 * 100, 2)

            is_in_domain = (train_name == test_name)
            detailed_evaluations.append({
                "train_cohort": train_name,
                "test_cohort": test_name,
                "is_in_domain": is_in_domain,
                "accuracy": round(acc * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "roc_auc": round(auc * 100, 2)
            })

            logger.info(f"Train on [{train_name:<15}] -> Test on [{test_name:<15}] | AUC: {auc*100:.1f}% | Acc: {acc*100:.1f}% | F1: {f1*100:.1f}%")

    # Calculate Generalization Gap and Invariance
    summary_metrics = {}
    for i, c_name in enumerate(cohort_names):
        in_domain_auc = results_matrix["roc_auc"][i, i]
        out_domain_aucs = [results_matrix["roc_auc"][i, j] for j in range(len(cohort_names)) if j != i]
        mean_out_auc = np.mean(out_domain_aucs)
        gap = in_domain_auc - mean_out_auc
        invariance_score = 100.0 - abs(gap)

        summary_metrics[c_name] = {
            "in_domain_auc": round(in_domain_auc, 2),
            "cross_domain_auc_mean": round(float(mean_out_auc), 2),
            "generalization_gap": round(float(gap), 2),
            "invariance_score": round(float(invariance_score), 2)
        }

    # Export metrics JSON
    export_data = {
        "cohorts": cohort_names,
        "features": HARMONIZED_FEATURES,
        "results_matrix_auc": results_matrix["roc_auc"].tolist(),
        "results_matrix_accuracy": results_matrix["accuracy"].tolist(),
        "results_matrix_f1": results_matrix["f1_score"].tolist(),
        "detailed_evaluations": detailed_evaluations,
        "summary_metrics": summary_metrics
    }

    metrics_out = MODELS_DIR / "cross_dataset_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump(export_data, f, indent=2)
    logger.info(f">> Saved Cross-Dataset Metrics to: {metrics_out}")

    # Generate Cross-Domain Heatmap Chart
    generate_cross_dataset_charts(cohort_names, results_matrix)
    return export_data


def generate_cross_dataset_charts(cohort_names, results_matrix):
    """Generates the Cross-Dataset Generalization Matrix Heatmap."""
    plt.figure(figsize=(9, 7))
    auc_df = pd.DataFrame(results_matrix["roc_auc"], index=cohort_names, columns=cohort_names)

    ax = sns.heatmap(
        auc_df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "ROC-AUC (%)"},
        linewidths=1.5,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Cross-Dataset Generalization Matrix (ROC-AUC %)\nMulti-Cohort Transferability Validation", fontsize=13, pad=14, fontweight="bold")
    plt.xlabel("Evaluation / Testing Cohort", fontsize=11, fontweight="bold")
    plt.ylabel("Training / Optimization Cohort", fontsize=11, fontweight="bold")
    plt.tight_layout()

    chart_path = CHARTS_DIR / "cross_dataset_generalization_matrix.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f">> Saved Cross-Dataset Heatmap to: {chart_path}")


if __name__ == "__main__":
    run_cross_dataset_validation()
