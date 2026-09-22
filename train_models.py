"""
Model Training & Evaluation Script (Phases 2 & 3).
Trains multiple supervised ML classification models on the balanced 70,000 dataset:
1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier
4. XGBoost Classifier
5. Gradient Boosting Classifier
Evaluates on the held-out test set (13,717 records) and saves models for instant prediction in Streamlit.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "age_years", "gender_binary", "height", "weight", "bmi",
    "systolic_bp", "diastolic_bp", "cholesterol", "gluc",
    "smoke", "alco", "active"
]
TARGET_COL = "cardio"


def train_and_evaluate_all_models():
    logger.info("Loading processed 70,000 datasets...")
    train_path = DATA_DIR / "train_70k_smote_balanced.csv"
    test_path = DATA_DIR / "test_70k_held_out.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets not found. Please run main.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    logger.info(f"Training partition: {len(X_train):,} samples | Testing partition: {len(X_test):,} samples")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
    }

    results = []
    best_f1 = 0.0
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))

        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = float(tn / (tn + fp))

        res_entry = {
            "model_name": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "specificity": round(spec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "roc_auc": round(auc * 100, 2),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }
        results.append(res_entry)

        # Save individual model
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{slug}_model.joblib")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    # Save overall comparison JSON
    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save primary best model
    joblib.dump(best_model_obj, MODELS_DIR / "best_cardio_model.joblib")
    logger.info(f"Best model: {best_model_name} with F1-score: {best_f1*100:.2f}%")

    print("\n" + "=" * 80)
    print(f"{'Model':<22} {'Accuracy':<10} {'Precision':<11} {'Recall':<9} {'Specificity':<13} {'F1':<8} {'ROC-AUC':<9}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_name']:<22} {r['accuracy']:<10.2f} {r['precision']:<11.2f} {r['recall']:<9.2f} {r['specificity']:<13.2f} {r['f1_score']:<8.2f} {r['roc_auc']:<9.2f}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    train_and_evaluate_all_models()
