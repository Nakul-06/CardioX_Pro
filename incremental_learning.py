"""
Phase 5: Adaptive Incremental Learning Module.
Simulates real-world hospital streaming data arrival where patient records stream in batches.
Updates model parameters dynamically in real time using partial_fit (SGDClassifier with log_loss)
without requiring full-dataset retraining.
Tracks:
- Streaming Batch Performance (Pre-update vs Post-update accuracy)
- Cumulative ROC-AUC and F1-Score
- Concept Drift Detection (Wasserstein metric / distribution shift)
- Weight update delta (||ΔW||)
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed_data"
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
MODELS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "age_years", "gender_binary", "height", "weight", "bmi",
    "systolic_bp", "diastolic_bp", "cholesterol", "gluc",
    "smoke", "alco", "active"
]
TARGET_COL = "cardio"


class AdaptiveIncrementalLearner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=self.random_state
        )
        self.classes = np.array([0, 1])
        self.history = []
        self.baseline_distribution = None
        self.previous_weights = None

    def initialize_with_baseline(self, X_init, y_init):
        """Warm-start / calibrate model on initial baseline batch."""
        self.model.partial_fit(X_init, y_init, classes=self.classes)
        self.baseline_distribution = X_init.mean(axis=0).values
        self.previous_weights = self.model.coef_.copy().flatten()
        logger.info(f"Initialized Incremental Learner on baseline batch ({len(X_init):,} samples).")

    def process_streaming_batch(self, batch_id, X_batch, y_batch):
        """
        Evaluate before update (testing generalization),
        then update weights online (partial_fit),
        and evaluate post-update gain.
        """
        # 1. Pre-update performance (how well previous weights predict incoming stream)
        y_prob_pre = self.model.predict_proba(X_batch)[:, 1]
        y_pred_pre = (y_prob_pre >= 0.5).astype(int)
        acc_pre = float(accuracy_score(y_batch, y_pred_pre))

        # 2. Concept drift calculation (Wasserstein distance across mean feature profile)
        batch_distribution = X_batch.mean(axis=0).values
        drift_score = float(wasserstein_distance(self.baseline_distribution, batch_distribution))

        # 3. Online dynamic weight update (partial_fit)
        self.model.partial_fit(X_batch, y_batch)

        # 4. Post-update performance
        y_prob_post = self.model.predict_proba(X_batch)[:, 1]
        y_pred_post = (y_prob_post >= 0.5).astype(int)
        acc_post = float(accuracy_score(y_batch, y_pred_post))
        f1_post = float(f1_score(y_batch, y_pred_post))
        auc_post = float(roc_auc_score(y_batch, y_prob_post))

        # 5. Weight shift magnitude
        current_weights = self.model.coef_.copy().flatten()
        weight_delta = float(np.linalg.norm(current_weights - self.previous_weights))
        self.previous_weights = current_weights

        metrics = {
            "batch_id": int(batch_id),
            "batch_size": int(len(X_batch)),
            "pre_update_accuracy": round(acc_pre * 100, 2),
            "post_update_accuracy": round(acc_post * 100, 2),
            "adaptation_gain": round((acc_post - acc_pre) * 100, 2),
            "f1_score": round(f1_post * 100, 2),
            "roc_auc": round(auc_post * 100, 2),
            "concept_drift_score": round(drift_score, 4),
            "weight_delta_norm": round(weight_delta, 4)
        }
        self.history.append(metrics)
        logger.info(
            f"Batch {batch_id:02d} | Pre-Acc: {metrics['pre_update_accuracy']:.1f}% -> "
            f"Post-Acc: {metrics['post_update_accuracy']:.1f}% (+{metrics['adaptation_gain']:.1f}%) | "
            f"AUC: {metrics['roc_auc']:.1f}% | Drift: {metrics['concept_drift_score']:.4f}"
        )
        return metrics


def run_incremental_learning_simulation(n_batches=10, batch_size=500):
    """
    Simulates streaming data arrival using held-out test data streams.
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: ADAPTIVE INCREMENTAL LEARNING STREAMING SIMULATION")
    logger.info("=" * 80)

    # Load test dataset for streaming simulation
    test_path = DATA_DIR / "test_70k_held_out.csv"
    train_path = DATA_DIR / "train_70k_smote_balanced.csv"

    if not test_path.exists() or not train_path.exists():
        raise FileNotFoundError("Processed datasets not found. Please run main.py first.")

    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)

    # Use first 2,000 training samples as initial baseline warm-up
    warmup_df = df_train.sample(n=2000, random_state=42)
    X_init = warmup_df[FEATURE_COLS]
    y_init = warmup_df[TARGET_COL]

    learner = AdaptiveIncrementalLearner(random_state=42)
    learner.initialize_with_baseline(X_init, y_init)

    # Shuffle test set to simulate streaming hospital arrivals
    stream_df = df_test.sample(frac=1.0, random_state=42).reset_index(drop=True)

    for b_idx in range(1, n_batches + 1):
        start_idx = (b_idx - 1) * batch_size
        end_idx = start_idx + batch_size
        batch_slice = stream_df.iloc[start_idx:end_idx]

        X_b = batch_slice[FEATURE_COLS]
        y_b = batch_slice[TARGET_COL]

        learner.process_streaming_batch(b_idx, X_b, y_b)

    # Save model and metrics
    metrics_path = MODELS_DIR / "incremental_learning_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(learner.history, f, indent=2)

    joblib.dump(learner.model, MODELS_DIR / "incremental_online_model.joblib")
    logger.info(f">> Saved incremental metrics to: {metrics_path}")
    logger.info(f">> Saved trained incremental model to: {MODELS_DIR / 'incremental_online_model.joblib'}")

    # Generate Learning Curve Visualizations
    generate_incremental_charts(learner.history)
    return learner.history


def generate_incremental_charts(history):
    df_h = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Pre vs Post Update Accuracy & Adaptation
    batches = df_h["batch_id"]
    ax1.plot(batches, df_h["pre_update_accuracy"], "o--", color="#E74C3C", label="Pre-Update Accuracy (Zero-Shot)", linewidth=1.8)
    ax1.plot(batches, df_h["post_update_accuracy"], "s-", color="#27AE60", label="Post-Update Accuracy (After Online Fit)", linewidth=2.2)
    ax1.fill_between(batches, df_h["pre_update_accuracy"], df_h["post_update_accuracy"], color="#27AE60", alpha=0.15, label="Adaptation Gain Area")
    ax1.set_xlabel("Streaming Patient Batch (500 records/batch)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Online Model Adaptation Across Streaming Batches", fontsize=12, fontweight="bold")
    ax1.set_xticks(batches)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Subplot 2: Cumulative ROC-AUC and Concept Drift
    ax2_twin = ax2.twinx()
    p1, = ax2.plot(batches, df_h["roc_auc"], "d-", color="#2980B9", label="Batch ROC-AUC (%)", linewidth=2.0)
    p2, = ax2_twin.plot(batches, df_h["concept_drift_score"], "^--", color="#8E44AD", label="Concept Drift (Wasserstein Dist)", linewidth=1.8)

    ax2.set_xlabel("Streaming Patient Batch (500 records/batch)", fontsize=11)
    ax2.set_ylabel("ROC-AUC (%)", color="#2980B9", fontsize=11)
    ax2_twin.set_ylabel("Concept Drift Score", color="#8E44AD", fontsize=11)
    ax2.set_title("Discriminative Stability & Concept Drift Tracking", fontsize=12, fontweight="bold")
    ax2.set_xticks(batches)
    ax2.grid(True, linestyle="--", alpha=0.5)

    lines = [p1, p2]
    ax2.legend(lines, [l.get_label() for l in lines], loc="lower right")

    plt.tight_layout()
    chart_path = CHARTS_DIR / "incremental_learning_curve.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f">> Saved Incremental Learning Chart to: {chart_path}")


if __name__ == "__main__":
    run_incremental_learning_simulation()
