"""
Clustering and weighted similarity engine for Module 3: Player Scouting Dashboard.
Keeps K-means (used for 2D segmentation/visualization) separate from
the user-weighted distance metric (used for true similarity ranking).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process, fuzz

from src.config import K_RANGE, PROCESSED_DATA_DIR, RANDOM_SEED
from src.logging_config import get_logger
from src.data_layer.metadata import get_metadata
from src.data_layer.squad_overrides import get_current_club
from src.module3_scouting.features import build_scouting_features

logger = get_logger(__name__)

CLUSTER_METRICS_PATH = PROCESSED_DATA_DIR / "module3_cluster_metrics.json"
SCOUTING_DATA_PATH = PROCESSED_DATA_DIR / "module3_scouting_data.joblib"


FEATURE_GROUPS = {
    "attacking": ["goals_per_90", "assists_per_90", "goal_involvements_per_90"],
    "defensive_durability": ["minutes_played", "appearances", "cards_per_90"],
}
ALL_FEATURES = FEATURE_GROUPS["attacking"] + FEATURE_GROUPS["defensive_durability"]


def evaluate_kmeans_k(X_scaled: np.ndarray) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    """
    Evaluates K-means over K_RANGE using Inertia (elbow) and Silhouette score.
    Returns: (optimal_k, inertia_scores, silhouette_scores)
    """
    inertias = {}
    silhouettes = {}

    best_k = 4
    best_sil = -1.0

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        inertias[int(k)] = float(km.inertia_)
        silhouettes[int(k)] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = int(k)

    return best_k, inertias, silhouettes


def fit_scouting_pipeline() -> pd.DataFrame:
    """
    Builds the feature matrix, evaluates optimal K, fits K-Means and 2D PCA,
    and caches the processed dataset for instantaneous Streamlit queries.
    """
    df = build_scouting_features()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ALL_FEATURES])

    # Evaluate K-means
    best_k, inertias, silhouettes = evaluate_kmeans_k(X_scaled)
    logger.info("Optimal K determined by Silhouette score: k=%d (score=%.3f)", best_k, silhouettes[best_k])

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 2D PCA for dashboard visualization
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = np.round(pca_coords[:, 0], 3)
    df["pca_y"] = np.round(pca_coords[:, 1], 3)

    # Save cluster metrics
    cluster_metrics = {
        "optimal_k": best_k,
        "silhouette_score": silhouettes[best_k],
        "k_range_evaluated": list(K_RANGE),
        "silhouette_by_k": silhouettes,
        "inertia_by_k": inertias,
        "pca_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    }
    with open(CLUSTER_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(cluster_metrics, f, indent=2)
    logger.info("Saved clustering metrics to %s", CLUSTER_METRICS_PATH)

    # Save processed scouting bundle
    bundle = {
        "data": df,
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "features": ALL_FEATURES,
    }
    joblib.dump(bundle, SCOUTING_DATA_PATH)
    logger.info("Saved scouting pipeline bundle to %s", SCOUTING_DATA_PATH)

    return df


def load_scouting_bundle() -> Dict[str, Any]:
    """Loads cached scouting pipeline bundle, fitting if not present."""
    if not SCOUTING_DATA_PATH.exists():
        fit_scouting_pipeline()
    return joblib.load(SCOUTING_DATA_PATH)


def compute_weighted_similarity(
    target_idx: int,
    df: pd.DataFrame,
    scaler: StandardScaler,
    attack_weight: float = 1.0,
    defense_weight: float = 1.0,
    position_filter: str = "All",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Computes true player similarity using distance in the user-weighted, standardized feature space.
    Clustering is NOT the similarity algorithm — distance in the feature space is.
    """
    X_scaled = scaler.transform(df[ALL_FEATURES])

    # Apply user weights to feature dimensions
    weights = np.ones(len(ALL_FEATURES))
    for i, col in enumerate(ALL_FEATURES):
        if col in FEATURE_GROUPS["attacking"]:
            weights[i] = max(attack_weight, 0.05)
        else:
            weights[i] = max(defense_weight, 0.05)

    # Normalize weight vector
    weights = weights / np.mean(weights)
    X_weighted = X_scaled * np.sqrt(weights)

    target_vector = X_weighted[target_idx]

    # Euclidean distance in weighted space
    distances = np.linalg.norm(X_weighted - target_vector, axis=1)

    # Convert distance to a 0-100% similarity score
    # sim = 1 / (1 + distance)
    similarities = np.round((1.0 / (1.0 + (distances / 2.0))) * 100, 1)

    results_df = df.copy()
    results_df["distance"] = np.round(distances, 3)
    results_df["similarity_pct"] = similarities

    # Exclude the target player-season itself
    results_df = results_df.drop(index=target_idx)

    # Apply position filter if requested
    if position_filter and position_filter != "All":
        results_df = results_df[results_df["position_group"] == position_filter]

    # Sort by closest distance (highest similarity)
    ranked = results_df.sort_values("distance", ascending=True).head(top_n)
    return ranked


def find_similar_players_cli(
    player_query: str,
    top_n: int = 5,
    attack_weight: float = 1.0,
    defense_weight: float = 1.0,
    position_filter: str = "All",
) -> None:
    """CLI tool to query similar players with freshness footer."""
    bundle = load_scouting_bundle()
    df = bundle["data"]
    scaler = bundle["scaler"]

    names = df["display_name"].tolist()
    match = process.extractOne(player_query, names, scorer=fuzz.WRatio)
    if not match or match[1] < 60:
        print(f"\nPlayer '{player_query}' not found in scouting dataset.")
        return

    target_name = match[0]
    target_idx = df[df["display_name"] == target_name].index[0]
    target_row = df.loc[target_idx]

    # Resolve club
    current_club, club_source = get_current_club(
        target_row["name"], fallback_club=target_row["current_club_name"]
    )

    ranked = compute_weighted_similarity(
        target_idx=target_idx,
        df=df,
        scaler=scaler,
        attack_weight=attack_weight,
        defense_weight=defense_weight,
        position_filter=position_filter,
        top_n=top_n,
    )

    print("\n" + "=" * 68)
    print(f"SCOUTING REPORT FOR: {target_name}")
    print("=" * 68)
    print(f"Group: {target_row['position_group']} ({target_row['sub_position']}) | Cluster #{target_row['cluster']}")
    print(f"Club:  {current_club}")
    print(f"Stats: {target_row['goals_per_90']:.2f} G/90, {target_row['assists_per_90']:.2f} A/90 across {target_row['minutes_played']:,.0f} mins")
    print("-" * 68)
    print(f"TOP {top_n} STATISTICALLY SIMILAR PLAYERS (Weights: Attacking={attack_weight}x, Def={defense_weight}x):")
    print("-" * 68)
    fmt = "{:<4} | {:<28} | {:<5} | {:<7} | {:<7} | {:<10}"
    print(fmt.format("Rank", "Player (Season)", "Pos", "G/90", "A/90", "Similarity"))
    print("-" * 68)
    for rank, (_, r) in enumerate(ranked.iterrows(), start=1):
        print(fmt.format(
            f"#{rank}",
            str(r["display_name"])[:28],
            str(r["position_group"]),
            f"{r['goals_per_90']:.2f}",
            f"{r['assists_per_90']:.2f}",
            f"{r['similarity_pct']}%"
        ))
    print("=" * 68)

    meta = get_metadata()
    footer = meta.format_freshness_footer(current_club_source=club_source)
    print(footer)
    print("=" * 68 + "\n")


if __name__ == "__main__":
    fit_scouting_pipeline()
