"""
CLI for Module 1: Transfer Value Predictor.
Given a player name, predicts market value from stats, compares with actual valuation,
flags over/under-valuation, and outputs the standardized data freshness footer.
Runnable as: python -m src.module1_transfer_value.predict --player "Bukayo Saka"
"""

import argparse
import sys
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from src.config import GB1, MIN_MINUTES_THRESHOLD
from src.logging_config import get_logger
from src.data_layer.metadata import get_metadata
from src.data_layer.squad_overrides import get_current_club
from src.data_layer.transfermarkt_dataset import (
    load_appearances,
    load_games,
    load_player_valuations,
    load_players,
)
from src.module1_transfer_value.train import MODEL_B_PATH, train_and_evaluate

logger = get_logger("module1_predict")


def find_player(query: str, players_df: pd.DataFrame, score_threshold: float = 85.0) -> Tuple[Optional[pd.Series], float]:
    """
    Finds player in the dataset using fuzzy name matching.
    Returns (player_row, match_score) or (None, score) if absent.
    """
    clean_names = players_df["name"].dropna().tolist()
    match = process.extractOne(query, clean_names, scorer=fuzz.token_sort_ratio)

    if not match or match[1] < score_threshold:
        return None, (match[1] if match else 0.0)

    matched_name = match[0]
    player_row = players_df[players_df["name"] == matched_name].iloc[0]
    return player_row, match[1]


def predict_player_value(player_query: str) -> None:
    """Predicts player market value and prints comparison report."""
    players_df = load_players()
    player_row, score = find_player(player_query, players_df)

    if player_row is None:
        print(f"\nPlayer '{player_query}' is not covered by this dataset (no Premier League record in frozen dataset).")
        print("Please check spelling or verify that the player competed in the covered seasons (up to 2025/26).\n")
        return

    player_id = int(player_row["player_id"])
    player_name = str(player_row["name"])
    raw_club = str(player_row.get("current_club_name") or "Unknown")

    # Load player appearances
    apps_df = load_appearances(competition_filter=GB1)
    p_apps = apps_df[apps_df["player_id"] == player_id].copy()

    if p_apps.empty:
        print(f"\nPlayer '{player_query}' is not covered by this dataset (no Premier League record in frozen dataset).\n")
        return

    # Resolve club via squad_overrides.py
    current_club, club_source = get_current_club(player_name, fallback_club=raw_club)

    # Map games to seasons to use most recent season's stats (matching training feature distribution)
    games_df = load_games(competition_filter=GB1)
    games_season = games_df.set_index("game_id")["season"].to_dict()
    p_apps["season"] = p_apps["game_id"].map(games_season)
    latest_season = p_apps["season"].dropna().max()
    if pd.notna(latest_season):
        season_apps = p_apps[p_apps["season"] == latest_season]
    else:
        season_apps = p_apps

    # Season-to-date stats
    total_minutes = float(season_apps["minutes_played"].sum())
    total_goals = float(season_apps["goals"].sum())
    total_assists = float(season_apps["assists"].sum())
    n_appearances = len(season_apps)

    minutes_calc = max(total_minutes, 90.0)
    goals_per_90 = (total_goals / minutes_calc) * 90.0
    assists_per_90 = (total_assists / minutes_calc) * 90.0

    dob = player_row.get("date_of_birth")
    if pd.notna(dob):
        age = (pd.Timestamp.now() - pd.to_datetime(dob)).days / 365.25
    else:
        age = 25.0

    position = str(player_row.get("position") or "Midfield")

    # Load model
    if not MODEL_B_PATH.exists():
        logger.info("Trained model not found. Running training pipeline...")
        train_and_evaluate()

    pipeline = joblib.load(MODEL_B_PATH)

    # Inference DataFrame
    sample = pd.DataFrame([{
        "goals_per_90": goals_per_90,
        "assists_per_90": assists_per_90,
        "minutes": total_minutes,
        "appearances": n_appearances,
        "age": age,
        "position": position,
        "club": current_club,
    }])

    pred_log = pipeline.predict(sample)[0]
    pred_eur = float(np.expm1(pred_log))

    # Fetch actual market value from valuations or players table
    vals_df = load_player_valuations()
    p_vals = vals_df[vals_df["player_id"] == player_id].sort_values("date")
    if not p_vals.empty:
        last_val = p_vals.iloc[-1]
        actual_eur = float(last_val["market_value_in_eur"])
        val_date_str = str(last_val["date"])
    else:
        actual_eur = float(player_row.get("market_value_in_eur") or 0)
        val_date_str = "latest known"

    # Valuation assessment
    diff = pred_eur - actual_eur
    diff_pct = (diff / actual_eur * 100) if actual_eur > 0 else 0

    if diff > actual_eur * 0.20:
        verdict = f"UNDERVALUED (Model estimates +{abs(diff_pct):.1f}% upside)"
    elif diff < -actual_eur * 0.20:
        verdict = f"OVERVALUED (Market valuation is {abs(diff_pct):.1f}% above model)"
    else:
        verdict = "FAIRLY VALUED (Within ±20% band)"

    print("\n" + "=" * 60)
    print(f"TRANSFER VALUE REPORT: {player_name}")
    print("=" * 60)
    print(f"Position:        {position}")
    print(f"Current Club:    {current_club}")
    print(f"PL Minutes:      {total_minutes:,.0f} mins across {n_appearances} matches")
    print(f"Stats (per 90):  {goals_per_90:.2f} goals, {assists_per_90:.2f} assists")
    print("-" * 60)
    print(f"Actual Value:    €{actual_eur:,.0f} (as of {val_date_str})")
    print(f"Predicted Value: €{pred_eur:,.0f}")
    print(f"Difference:      €{diff:+,.0f} ({diff_pct:+.1f}%)")
    print(f"Assessment:      {verdict}")
    print("-" * 60)

    # Freshness footer
    meta = get_metadata()
    footer = meta.format_freshness_footer(current_club_source=club_source)
    print(footer)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict player transfer market value.")
    parser.add_argument("--player", type=str, default="Bukayo Saka", help="Player name (e.g. Bukayo Saka)")
    args = parser.parse_args()

    predict_player_value(args.player)


if __name__ == "__main__":
    main()
