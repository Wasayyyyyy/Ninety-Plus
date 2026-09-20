"""
Model training and probabilistic evaluation for Module 2: Match Outcome Predictor.
Implements:
- Majority-class and heuristic baselines
- RandomForestClassifier (class_weight='balanced')
- XGBClassifier (with compute_sample_weight('balanced', y_train))
- Multiclass Log Loss, Brier score, and per-class precision/recall
- Calibration curve and feature importance plots
- Reproducibility outputs: module2_metrics.json, module2_predictions.csv
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from src.logging_config import get_logger
from src.module2_match_outcome.features import (
    HOME_WIN,
    DRAW,
    AWAY_WIN,
    INT_TO_TARGET,
    TARGET_CLASSES,
    TARGET_TO_INT,
    build_pre_match_features,
    get_feature_columns,
)

logger = get_logger(__name__)

MODEL_SAVE_DIR = PROCESSED_DATA_DIR / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RF_MODEL_PATH = MODEL_SAVE_DIR / "module2_rf.joblib"
XGB_MODEL_PATH = MODEL_SAVE_DIR / "module2_xgb.joblib"
METRICS_PATH = PROCESSED_DATA_DIR / "module2_metrics.json"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "module2_predictions.csv"


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3) -> float:
    """
    Computes multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    """
    y_one_hot = np.zeros((len(y_true), num_classes))
    for i, val in enumerate(y_true):
        y_one_hot[i, val] = 1.0
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Computes comprehensive probabilistic and classification metrics."""
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_prob, labels=[0, 1, 2])
    brier = compute_multiclass_brier_score(y_true, y_prob, num_classes=3)
    c_report = classification_report(
        y_true,
        y_pred,
        target_names=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "model": model_name,
        "accuracy": float(acc),
        "log_loss": float(loss),
        "brier_score": float(brier),
        "per_class": {
            cls: {
                "precision": float(c_report[cls]["precision"]),
                "recall": float(c_report[cls]["recall"]),
                "f1": float(c_report[cls]["f1-score"]),
                "support": int(c_report[cls]["support"]),
            }
            for cls in TARGET_CLASSES
        },
        "confusion_matrix": conf_matrix,
    }


def compute_baseline_predictions(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (majority_class_probabilities, heuristic_probabilities).
    """
    # 1. Majority class baseline: predict class frequencies from training set
    class_counts = np.bincount(y_train, minlength=3)
    train_dist = class_counts / len(y_train)
    majority_prob = np.tile(train_dist, (len(X_test), 1))

    # 2. Heuristic baseline: points-differential & home advantage heuristic
    heuristic_prob = []
    for _, row in X_test.iterrows():
        diff_ppm = row["diff_season_ppm"]
        # Home advantage bias: 0.20 base boost
        score = diff_ppm + 0.20
        if score > 0.45:
            # Home win strongly favored
            p = [0.60, 0.25, 0.15]
        elif score < -0.25:
            # Away win strongly favored
            p = [0.20, 0.28, 0.52]
        else:
            # Closely matched / draw likely
            p = [0.38, 0.35, 0.27]
        heuristic_prob.append(p)

    return majority_prob, np.array(heuristic_prob)


def plot_calibration_curves(
    y_test: np.ndarray,
    rf_prob: np.ndarray,
    xgb_prob: np.ndarray,
    save_path: Path,
) -> None:
    """Plots calibration curve for HOME_WIN predictions."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # True binary indicator for Home Win
    y_test_binary = (y_test == 0).astype(int)

    rf_frac, rf_mean = calibration_curve(y_test_binary, rf_prob[:, 0], n_bins=8, strategy="uniform")
    xgb_frac, xgb_mean = calibration_curve(y_test_binary, xgb_prob[:, 0], n_bins=8, strategy="uniform")

    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")
    ax.plot(rf_mean, rf_frac, "s-", color="#1f77b4", lw=2, label="Random Forest")
    ax.plot(xgb_mean, xgb_frac, "o-", color="#ff7f0e", lw=2, label="XGBoost")

    ax.set_title("Calibration Curve — HOME_WIN Probability", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of True Home Wins", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved calibration curve to %s", save_path)


def plot_feature_importances(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier,
    feature_names: list,
    save_path: Path,
) -> None:
    """Plots top feature importances side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # RF importances
    rf_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    rf_top = rf_imp.tail(10)
    rf_top.plot(kind="barh", ax=ax1, color="#1f77b4", edgecolor="black")
    ax1.set_title("Random Forest — Top 10 Features", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gini Importance")

    # XGB importances
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    xgb_top = xgb_imp.tail(10)
    xgb_top.plot(kind="barh", ax=ax2, color="#ff7f0e", edgecolor="black")
    ax2.set_title("XGBoost — Top 10 Features", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Gain Importance")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved feature importances plot to %s", save_path)


def train_and_evaluate(test_season: int = 2024) -> Dict[str, Any]:
    """
    Main training pipeline for Module 2:
    - Builds features chronologically.
    - Temporal split: train on seasons < test_season, test on test_season.
    - Trains RF and XGBoost with balanced class weights.
    - Evaluates baselines + models.
    - Generates plots and saves reproducibility artifacts.
    """
    logger.info("Building pre-match feature dataset...")
    df = build_pre_match_features()

    feature_cols = get_feature_columns()

    # Temporal split by season
    available_seasons = sorted(df["season"].unique())
    logger.info("Available seasons in games dataset: %s", available_seasons)

    # If requested test_season is not in data, pick the latest season
    if test_season not in available_seasons:
        test_season = int(available_seasons[-1])
    logger.info("Evaluating on held-out season: %d (trained on seasons < %d)", test_season, test_season)

    train_mask = df["season"] < test_season
    test_mask = df["season"] == test_season

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_int"].to_numpy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["target_int"].to_numpy()

    logger.info("Train set: %d matches, Test set (%d): %d matches.", len(X_train), test_season, len(X_test))

    # 1. Baselines
    majority_prob, heuristic_prob = compute_baseline_predictions(X_train, y_train, X_test)
    metrics_majority = evaluate_predictions(y_test, majority_prob, "Majority Class Baseline")
    metrics_heuristic = evaluate_predictions(y_test, heuristic_prob, "Points Heuristic Baseline")

    # 2. Random Forest
    logger.info("Fitting RandomForestClassifier (class_weight='balanced')...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)
    metrics_rf = evaluate_predictions(y_test, rf_prob, "Random Forest")

    # 3. XGBoost
    logger.info("Fitting XGBClassifier with sample_weight='balanced'...")
    sample_weights = compute_sample_weight("balanced", y_train)
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_prob = xgb_model.predict_proba(X_test)
    metrics_xgb = evaluate_predictions(y_test, xgb_prob, "XGBoost")

    # Save models
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    logger.info("Saved models to %s and %s", RF_MODEL_PATH, XGB_MODEL_PATH)

    # Generate plots
    calibration_plot_path = FIGURES_DIR / "module2_calibration.png"
    feat_imp_plot_path = FIGURES_DIR / "module2_feature_importance.png"
    plot_calibration_curves(y_test, rf_prob, xgb_prob, calibration_plot_path)
    plot_feature_importances(rf_model, xgb_model, feature_cols, feat_imp_plot_path)

    # Save predictions CSV
    preds_df = test_df[["game_id", "season", "date", "home_club_name", "away_club_name", "target"]].copy()
    preds_df["rf_pred"] = [INT_TO_TARGET[p] for p in np.argmax(rf_prob, axis=1)]
    preds_df["xgb_pred"] = [INT_TO_TARGET[p] for p in np.argmax(xgb_prob, axis=1)]
    preds_df["rf_home_prob"] = np.round(rf_prob[:, 0], 3)
    preds_df["rf_draw_prob"] = np.round(rf_prob[:, 1], 3)
    preds_df["rf_away_prob"] = np.round(rf_prob[:, 2], 3)
    preds_df["xgb_home_prob"] = np.round(xgb_prob[:, 0], 3)
    preds_df["xgb_draw_prob"] = np.round(xgb_prob[:, 1], 3)
    preds_df["xgb_away_prob"] = np.round(xgb_prob[:, 2], 3)
    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved predictions to %s", PREDICTIONS_PATH)

    # Package all metrics
    all_metrics = {
        "test_season": test_season,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "majority_baseline": metrics_majority,
        "heuristic_baseline": metrics_heuristic,
        "random_forest": metrics_rf,
        "xgboost": metrics_xgb,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", METRICS_PATH)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"MODULE 2 EVALUATION SUMMARY (Held-out Season: {test_season})")
    print("=" * 70)
    fmt = "{:<28} | {:<9} | {:<9} | {:<11} | {:<10}"
    print(fmt.format("Model / Baseline", "Accuracy", "Log Loss", "Brier Score", "Draw F1"))
    print("-" * 70)
    for m in [metrics_majority, metrics_heuristic, metrics_rf, metrics_xgb]:
        draw_f1 = m["per_class"]["DRAW"]["f1"]
        print(fmt.format(
            m["model"],
            f"{m['accuracy']:.3f}",
            f"{m['log_loss']:.3f}",
            f"{m['brier_score']:.3f}",
            f"{draw_f1:.3f}"
        ))
    print("=" * 70)

    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()
