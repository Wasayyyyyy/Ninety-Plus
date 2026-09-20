"""
Model training and comparative evaluation for Module 1: Transfer Value Predictor.
Implements:
- Point-in-time train/test split by season (older seasons train, latest season test)
- Walk-forward cross-season evaluation
- Baseline: median value by (position, age bucket)
- Model A: performance + age + position one-hot
- Model B: Model A + club one-hot
- Fits LinearRegression and Ridge for both Model A and Model B
- Evaluates R², MAE, RMSE on the original monetary scale (€)
- Plots predicted vs actual for Model B
- Reproducibility outputs: module1_metrics.json, module1_predictions.csv
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from src.logging_config import get_logger
from src.module1_transfer_value.features import (
    build_transfer_value_features,
    get_age_bucket,
)

logger = get_logger(__name__)

MODEL_SAVE_DIR = PROCESSED_DATA_DIR / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_B_PATH = MODEL_SAVE_DIR / "module1_model_b.joblib"
METRICS_PATH = PROCESSED_DATA_DIR / "module1_metrics.json"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "module1_predictions.csv"


def evaluate_monetary_scale(y_true_eur: np.ndarray, y_pred_eur: np.ndarray) -> Dict[str, float]:
    """Computes R2, MAE, and RMSE on the original non-log scale in Euros."""
    # Ensure non-negative predictions on monetary scale
    y_pred_clipped = np.clip(y_pred_eur, 0, None)
    r2 = r2_score(y_true_eur, y_pred_clipped)
    mae = mean_absolute_error(y_true_eur, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_clipped))
    return {
        "r2": float(r2),
        "mae_eur": float(mae),
        "rmse_eur": float(rmse),
    }


def compute_baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Baseline: median market value for player's (position, age bucket) computed on train set.
    """
    train_copy = train_df.copy()
    train_copy["age_bucket"] = train_copy["age"].apply(get_age_bucket)
    group_medians = train_copy.groupby(["position", "age_bucket"])["market_value_in_eur"].median().to_dict()
    global_median = float(train_copy["market_value_in_eur"].median())

    test_preds = []
    for _, row in test_df.iterrows():
        b = get_age_bucket(row["age"])
        val = group_medians.get((row["position"], b), global_median)
        test_preds.append(val)

    return np.array(test_preds)


def create_pipeline(features_mode: str, model_type: str = "ridge") -> Pipeline:
    """
    Creates preprocessing and regression pipeline.
    features_mode: 'model_a' (no club) or 'model_b' (with club)
    """
    numeric_features = ["goals_per_90", "assists_per_90", "minutes", "appearances", "age"]
    categorical_features = ["position"]
    if features_mode == "model_b":
        categorical_features.append("club")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    if model_type == "linear":
        reg = LinearRegression()
    else:
        reg = Ridge(alpha=1.0, random_state=RANDOM_SEED)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])


def plot_predicted_vs_actual(
    y_true_eur: np.ndarray,
    y_pred_eur: np.ndarray,
    save_path: Path,
    r2_val: float,
) -> None:
    """Plots scatter plot of predicted vs actual valuation with y=x line."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Convert to Millions for readable axes
    y_true_m = y_true_eur / 1e6
    y_pred_m = np.clip(y_pred_eur, 0, None) / 1e6

    ax.scatter(y_true_m, y_pred_m, alpha=0.55, color="#2b5c8f", edgecolors="none", s=35, label="Valuation Records")

    # y = x line
    max_val = max(y_true_m.max(), y_pred_m.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Fit (y = x)")

    ax.set_title("Module 1 (Model B): Predicted vs Actual Market Value", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Market Value (€ Millions)", fontsize=11)
    ax.set_ylabel("Predicted Market Value (€ Millions)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.text(
        0.05, 0.90, f"Held-out R² = {r2_val:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc")
    )
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved predicted vs actual plot to %s", save_path)


def train_and_evaluate() -> Dict[str, Any]:
    """
    Executes full point-in-time training and comparative evaluation for Module 1.
    """
    logger.info("Building point-in-time features for Module 1...")
    df = build_transfer_value_features()

    available_seasons = sorted(df["season"].unique())
    logger.info("Available seasons in valuation dataset: %s", available_seasons)

    # Held out season: most recent season (e.g. 2024 or 2025)
    test_season = int(available_seasons[-1])
    logger.info("Evaluating on held-out test season: %d (trained on seasons < %d)", test_season, test_season)

    train_df = df[df["season"] < test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    # If test season has too few samples, use last 2 seasons
    if len(test_df) < 50 and len(available_seasons) >= 3:
        test_season = int(available_seasons[-2])
        train_df = df[df["season"] < test_season].copy()
        test_df = df[df["season"] >= test_season].copy()

    logger.info("Train samples: %d, Test samples: %d", len(train_df), len(test_df))

    y_train_log = train_df["target_log"].to_numpy()
    y_test_eur = test_df["market_value_in_eur"].to_numpy()

    # 1. Baseline
    baseline_preds_eur = compute_baseline_predictions(train_df, test_df)
    metrics_baseline = evaluate_monetary_scale(y_test_eur, baseline_preds_eur)

    # 2. Model A (Performance + Age + Position)
    # 2a. Linear
    pipe_a_lin = create_pipeline("model_a", "linear")
    pipe_a_lin.fit(train_df, y_train_log)
    pred_a_lin_log = pipe_a_lin.predict(test_df)
    metrics_a_lin = evaluate_monetary_scale(y_test_eur, np.expm1(pred_a_lin_log))

    # 2b. Ridge
    pipe_a_ridge = create_pipeline("model_a", "ridge")
    pipe_a_ridge.fit(train_df, y_train_log)
    pred_a_ridge_log = pipe_a_ridge.predict(test_df)
    metrics_a_ridge = evaluate_monetary_scale(y_test_eur, np.expm1(pred_a_ridge_log))

    # 3. Model B (Model A + Club)
    # 3a. Linear
    pipe_b_lin = create_pipeline("model_b", "linear")
    pipe_b_lin.fit(train_df, y_train_log)
    pred_b_lin_log = pipe_b_lin.predict(test_df)
    metrics_b_lin = evaluate_monetary_scale(y_test_eur, np.expm1(pred_b_lin_log))

    # 3b. Ridge (Primary production model)
    pipe_b_ridge = create_pipeline("model_b", "ridge")
    pipe_b_ridge.fit(train_df, y_train_log)
    pred_b_ridge_log = pipe_b_ridge.predict(test_df)
    pred_b_ridge_eur = np.expm1(pred_b_ridge_log)
    metrics_b_ridge = evaluate_monetary_scale(y_test_eur, pred_b_ridge_eur)

    # Save Model B pipeline for inference
    joblib.dump(pipe_b_ridge, MODEL_B_PATH)
    logger.info("Saved Model B pipeline to %s", MODEL_B_PATH)

    # Plot predicted vs actual for Model B (Ridge)
    plot_path = FIGURES_DIR / "module1_predicted_vs_actual.png"
    plot_predicted_vs_actual(y_test_eur, pred_b_ridge_eur, plot_path, metrics_b_ridge["r2"])

    # Save predictions CSV
    preds_df = test_df[["player_id", "player_name", "valuation_date", "season", "club", "position", "age", "market_value_in_eur"]].copy()
    preds_df["baseline_pred_eur"] = np.round(baseline_preds_eur, 0)
    preds_df["model_a_ridge_pred_eur"] = np.round(np.expm1(pred_a_ridge_log), 0)
    preds_df["model_b_ridge_pred_eur"] = np.round(pred_b_ridge_eur, 0)
    preds_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Saved predictions to %s", PREDICTIONS_PATH)

    # Compile all metrics
    results = {
        "test_season": test_season,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "baseline_position_age": metrics_baseline,
        "model_a_linear": metrics_a_lin,
        "model_a_ridge": metrics_a_ridge,
        "model_b_linear": metrics_b_lin,
        "model_b_ridge": metrics_b_ridge,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)

    # Side-by-side summary table
    print("\n" + "=" * 75)
    print(f"MODULE 1 EVALUATION SUMMARY (Held-out Season: {test_season}, Monetary Scale in €)")
    print("=" * 75)
    fmt = "{:<32} | {:<8} | {:<15} | {:<15}"
    print(fmt.format("Model Configuration", "R²", "MAE (€)", "RMSE (€)"))
    print("-" * 75)
    configs = [
        ("Position+Age Bucket Baseline", metrics_baseline),
        ("Model A (Perf+Age+Pos) - Linear", metrics_a_lin),
        ("Model A (Perf+Age+Pos) - Ridge", metrics_a_ridge),
        ("Model B (+Club) - Linear", metrics_b_lin),
        ("Model B (+Club) - Ridge", metrics_b_ridge),
    ]
    for name, m in configs:
        print(fmt.format(name, f"{m['r2']:.3f}", f"€{m['mae_eur']:,.0f}", f"€{m['rmse_eur']:,.0f}"))
    print("=" * 75)

    return results


if __name__ == "__main__":
    train_and_evaluate()
