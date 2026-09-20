"""
CLI for Module 2: Match Outcome Predictor.
Given two team names, computes pre-match features, runs inference,
and outputs standardized win/draw/loss probabilities with the data freshness footer.
Runnable as: python -m src.module2_match_outcome.predict --home "Arsenal" --away "Chelsea"
"""

import argparse
import sys
from datetime import datetime
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from src.config import GB1, PROCESSED_DATA_DIR
from src.logging_config import get_logger
from src.data_layer.metadata import get_metadata
from src.data_layer.transfermarkt_dataset import load_games
from src.data_layer.football_data_client import FootballDataClient
from src.module2_match_outcome.features import (
    HOME_WIN,
    DRAW,
    AWAY_WIN,
    build_pre_match_features,
    get_feature_columns,
)
from src.module2_match_outcome.train import (
    RF_MODEL_PATH,
    XGB_MODEL_PATH,
    train_and_evaluate,
)

logger = get_logger("module2_predict")


def find_team_id_and_name(query: str, games_df: pd.DataFrame) -> Tuple[int, str]:
    """Finds the best matching Premier League club id and canonical name using rapidfuzz."""
    home_clubs = games_df[["home_club_id", "home_club_name"]].drop_duplicates()
    unique_names = home_clubs["home_club_name"].dropna().tolist()

    match = process.extractOne(query, unique_names, scorer=fuzz.WRatio)
    if not match or match[1] < 60:
        raise ValueError(f"Team '{query}' could not be matched to any Premier League club in the dataset.")

    canonical_name = match[0]
    team_id = int(home_clubs[home_clubs["home_club_name"] == canonical_name]["home_club_id"].iloc[0])
    return team_id, canonical_name


def predict_fixture(
    home_query: str,
    away_query: str,
    match_date_str: Optional[str] = None,
    model_choice: str = "xgb",
) -> None:
    """Computes features for a fixture and prints the probability forecast."""
    games_df = load_games(competition_filter=GB1)

    try:
        home_id, home_name = find_team_id_and_name(home_query, games_df)
        away_id, away_name = find_team_id_and_name(away_query, games_df)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Check for model existence, train if not yet saved
    model_path = XGB_MODEL_PATH if model_choice == "xgb" else RF_MODEL_PATH
    if not model_path.exists():
        logger.info("Trained model not found. Running training pipeline...")
        train_and_evaluate()

    model = joblib.load(model_path)

    # Determine date and season
    if match_date_str:
        match_date = pd.to_datetime(match_date_str)
    else:
        # Check upcoming fixtures from football-data client
        fb_client = FootballDataClient()
        upcoming = fb_client.get_upcoming_fixtures()
        match_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))

    current_season = int(games_df["season"].max())

    # Create dummy row for target match and append to games_df to build chronological features
    dummy_match = pd.DataFrame([{
        "game_id": 9999999,
        "competition_id": GB1,
        "season": current_season,
        "date": match_date.strftime("%Y-%m-%d"),
        "home_club_id": home_id,
        "away_club_id": away_id,
        "home_club_name": home_name,
        "away_club_name": away_name,
        "home_club_goals": 0,  # placeholder
        "away_club_goals": 0,  # placeholder
    }])

    all_games = pd.concat([games_df, dummy_match], ignore_index=True)
    all_features = build_pre_match_features(all_games)

    match_feat = all_features[all_features["game_id"] == 9999999]
    if match_feat.empty:
        print("Error constructing match features.")
        return

    feature_cols = get_feature_columns()
    X = match_feat[feature_cols]

    probs = model.predict_proba(X)[0]
    p_home = probs[0]
    p_draw = probs[1]
    p_away = probs[2]

    # Output according to brief specification
    print("\n" + "=" * 50)
    print(f"{home_name} vs {away_name}")
    print("=" * 50)
    print(f"Home win ({home_name}): {p_home:.2f}")
    print(f"Draw:               {p_draw:.2f}")
    print(f"Away win ({away_name}): {p_away:.2f}")
    print("-" * 50)

    # Shared data freshness footer
    meta = get_metadata()
    footer = meta.format_freshness_footer(
        current_club_source="frozen dataset - last known club (2025/26)"
    )
    print(footer)
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Premier League match outcome.")
    parser.add_argument("--home", type=str, default="Arsenal", help="Home team name (e.g. Arsenal)")
    parser.add_argument("--away", type=str, default="Chelsea", help="Away team name (e.g. Chelsea)")
    parser.add_argument("--date", type=str, default=None, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, choices=["rf", "xgb"], default="xgb", help="Model to use")
    args = parser.parse_args()

    predict_fixture(args.home, args.away, match_date_str=args.date, model_choice=args.model)


if __name__ == "__main__":
    main()
