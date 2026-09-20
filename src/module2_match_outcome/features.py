"""
Feature engineering for Module 2: Match Outcome Predictor.
Enforces strict chronological pre-match calculations with zero data leakage.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import GB1
from src.logging_config import get_logger
from src.data_layer.transfermarkt_dataset import load_games

logger = get_logger(__name__)

# Standard target labels
HOME_WIN = "HOME_WIN"
DRAW = "DRAW"
AWAY_WIN = "AWAY_WIN"
TARGET_CLASSES = [HOME_WIN, DRAW, AWAY_WIN]
TARGET_TO_INT = {HOME_WIN: 0, DRAW: 1, AWAY_WIN: 2}
INT_TO_TARGET = {0: HOME_WIN, 1: DRAW, 2: AWAY_WIN}


def compute_match_target(home_goals: float, away_goals: float) -> str:
    """Computes standardized target from scoreline."""
    if home_goals > away_goals:
        return HOME_WIN
    elif home_goals == away_goals:
        return DRAW
    else:
        return AWAY_WIN


def build_pre_match_features(
    games_df: Optional[pd.DataFrame] = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """
    Builds pre-match features chronologically.
    Guarantees:
    - Every row on match date D only uses results from matches strictly before D.
    - No match uses information from itself or any future match.
    """
    if games_df is None:
        games_df = load_games(competition_filter=GB1)

    # Filter out games without goals/scores
    df = games_df.dropna(subset=["home_club_goals", "away_club_goals", "date", "season"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    # Sort chronologically
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Historical state tracking structures
    # team_history: team_id -> list of past match dicts {date, season, points, goals_for, goals_against, is_home}
    team_history: Dict[int, List[Dict]] = {}
    # h2h_history: frozenset({home_id, away_id}) -> list of past matches {home_id, home_goals, away_goals}
    h2h_history: Dict[frozenset, List[Dict]] = {}

    feature_rows = []

    for idx, row in df.iterrows():
        game_id = row["game_id"]
        season = row["season"]
        match_date = row["date"]
        home_id = row["home_club_id"]
        away_id = row["away_club_id"]
        home_name = row["home_club_name"]
        away_name = row["away_club_name"]
        home_goals = row["home_club_goals"]
        away_goals = row["away_club_goals"]

        target_label = compute_match_target(home_goals, away_goals)

        # Helper to extract rolling & season-to-date stats for a team prior to this match
        def get_team_pre_match_stats(team_id: int) -> Dict[str, float]:
            past_matches = team_history.get(team_id, [])
            # Filter strictly prior to current match date (or past list)
            prior = [m for m in past_matches if m["date"] < match_date]

            # 1. Rolling form (last N matches)
            last_n = prior[-rolling_window:] if len(prior) >= rolling_window else prior
            if last_n:
                roll_pts = sum(m["points"] for m in last_n)
                roll_gf = sum(m["goals_for"] for m in last_n)
                roll_ga = sum(m["goals_against"] for m in last_n)
                roll_gd = roll_gf - roll_ga
                roll_win_rate = sum(1 for m in last_n if m["points"] == 3) / len(last_n)
                rest_days = (match_date - prior[-1]["date"]).days
            else:
                roll_pts = 0.0
                roll_gf = 0.0
                roll_ga = 0.0
                roll_gd = 0.0
                roll_win_rate = 0.0
                rest_days = 14.0  # default rest for season start

            # 2. Season-to-date stats (strictly current season prior matches)
            season_matches = [m for m in prior if m["season"] == season]
            season_played = len(season_matches)
            if season_played > 0:
                s_pts = sum(m["points"] for m in season_matches)
                s_gf = sum(m["goals_for"] for m in season_matches)
                s_ga = sum(m["goals_against"] for m in season_matches)
                s_gd = s_gf - s_ga
                s_ppm = s_pts / season_played
            else:
                s_pts = 0.0
                s_gf = 0.0
                s_ga = 0.0
                s_gd = 0.0
                s_ppm = 1.35  # PL league average points per match prior

            return {
                "roll_pts": float(roll_pts),
                "roll_gf": float(roll_gf),
                "roll_ga": float(roll_ga),
                "roll_gd": float(roll_gd),
                "roll_win_rate": float(roll_win_rate),
                "rest_days": min(float(rest_days), 30.0),
                "season_played": float(season_played),
                "season_pts": float(s_pts),
                "season_gf": float(s_gf),
                "season_ga": float(s_ga),
                "season_gd": float(s_gd),
                "season_ppm": float(s_ppm),
            }

        home_stats = get_team_pre_match_stats(home_id)
        away_stats = get_team_pre_match_stats(away_id)

        # 3. Head-to-Head prior to this match
        pair_key = frozenset([home_id, away_id])
        past_h2h = h2h_history.get(pair_key, [])
        h2h_count = len(past_h2h)
        h2h_home_wins = sum(1 for m in past_h2h if (m["home_id"] == home_id and m["home_goals"] > m["away_goals"]) or (m["away_id"] == home_id and m["away_goals"] > m["home_goals"]))
        h2h_away_wins = sum(1 for m in past_h2h if (m["home_id"] == away_id and m["home_goals"] > m["away_goals"]) or (m["away_id"] == away_id and m["away_goals"] > m["home_goals"]))
        h2h_draws = sum(1 for m in past_h2h if m["home_goals"] == m["away_goals"])

        # Construct feature dict
        feat = {
            "game_id": game_id,
            "season": season,
            "date": match_date,
            "home_club_id": home_id,
            "away_club_id": away_id,
            "home_club_name": home_name,
            "away_club_name": away_name,
            "target": target_label,
            "target_int": TARGET_TO_INT[target_label],
            # Differentials and form
            "diff_season_ppm": home_stats["season_ppm"] - away_stats["season_ppm"],
            "diff_season_gd": home_stats["season_gd"] - away_stats["season_gd"],
            "diff_roll_pts": home_stats["roll_pts"] - away_stats["roll_pts"],
            "diff_roll_gd": home_stats["roll_gd"] - away_stats["roll_gd"],
            "diff_rest_days": home_stats["rest_days"] - away_stats["rest_days"],
            # Home team features
            "home_roll_pts": home_stats["roll_pts"],
            "home_roll_gf": home_stats["roll_gf"],
            "home_roll_ga": home_stats["roll_ga"],
            "home_roll_win_rate": home_stats["roll_win_rate"],
            "home_season_played": home_stats["season_played"],
            "home_season_ppm": home_stats["season_ppm"],
            "home_season_gd": home_stats["season_gd"],
            "home_rest_days": home_stats["rest_days"],
            # Away team features
            "away_roll_pts": away_stats["roll_pts"],
            "away_roll_gf": away_stats["roll_gf"],
            "away_roll_ga": away_stats["roll_ga"],
            "away_roll_win_rate": away_stats["roll_win_rate"],
            "away_season_played": away_stats["season_played"],
            "away_season_ppm": away_stats["season_ppm"],
            "away_season_gd": away_stats["season_gd"],
            "away_rest_days": away_stats["rest_days"],
            # Head-to-Head features
            "h2h_matches": float(h2h_count),
            "h2h_home_win_rate": float(h2h_home_wins / h2h_count) if h2h_count > 0 else 0.40,
            "h2h_away_win_rate": float(h2h_away_wins / h2h_count) if h2h_count > 0 else 0.30,
            "h2h_draw_rate": float(h2h_draws / h2h_count) if h2h_count > 0 else 0.30,
        }
        feature_rows.append(feat)

        # AFTER recording features, update historical states with this match result
        # Points calculation
        if home_goals > away_goals:
            home_pts, away_pts = 3, 0
        elif home_goals == away_goals:
            home_pts, away_pts = 1, 1
        else:
            home_pts, away_pts = 0, 3

        team_history.setdefault(home_id, []).append({
            "date": match_date,
            "season": season,
            "points": home_pts,
            "goals_for": home_goals,
            "goals_against": away_goals,
            "is_home": True,
        })
        team_history.setdefault(away_id, []).append({
            "date": match_date,
            "season": season,
            "points": away_pts,
            "goals_for": away_goals,
            "goals_against": home_goals,
            "is_home": False,
        })
        h2h_history.setdefault(pair_key, []).append({
            "date": match_date,
            "home_id": home_id,
            "away_id": away_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
        })

    feature_df = pd.DataFrame(feature_rows)
    logger.info("Built pre-match feature dataset: %d matches processed.", len(feature_df))
    return feature_df


def get_feature_columns() -> List[str]:
    """Returns the ordered list of predictive feature column names."""
    return [
        "diff_season_ppm",
        "diff_season_gd",
        "diff_roll_pts",
        "diff_roll_gd",
        "diff_rest_days",
        "home_roll_pts",
        "home_roll_gf",
        "home_roll_ga",
        "home_roll_win_rate",
        "home_season_played",
        "home_season_ppm",
        "home_season_gd",
        "home_rest_days",
        "away_roll_pts",
        "away_roll_gf",
        "away_roll_ga",
        "away_roll_win_rate",
        "away_season_played",
        "away_season_ppm",
        "away_season_gd",
        "away_rest_days",
        "h2h_matches",
        "h2h_home_win_rate",
        "h2h_away_win_rate",
        "h2h_draw_rate",
    ]
