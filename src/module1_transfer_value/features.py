"""
Point-in-time feature engineering for Module 1: Transfer Value Predictor.
Guarantees strict point-in-time calculation:
- Only Premier League (GB1) appearances in the valuation's season
- Only appearances strictly before valuation date (date < valuation.date)
- Enforces minutes threshold (minutes >= 450)
- Exposes Model A (performance + age + position) and Model B (+ club) feature sets
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import GB1, MIN_MINUTES_THRESHOLD
from src.logging_config import get_logger
from src.data_layer.transfermarkt_dataset import (
    load_appearances,
    load_games,
    load_player_valuations,
    load_players,
)

logger = get_logger(__name__)


def compute_season_boundaries(games_df: pd.DataFrame) -> Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Computes (min_date, max_date) for each season from GB1 games,
    extending season end to June 30 of the following year to capture post-season summer valuations.
    """
    gb1_games = games_df[games_df["competition_id"] == GB1].copy()
    gb1_games["date"] = pd.to_datetime(gb1_games["date"])

    season_bounds = {}
    for season, group in gb1_games.groupby("season"):
        min_date = group["date"].min()
        max_game_date = group["date"].max()
        # Valuations often update in late May / June after the final matchday of season
        season_year = int(season)
        extended_end = pd.Timestamp(f"{season_year + 1}-06-30")
        season_bounds[season_year] = (min_date, max(max_game_date, extended_end))

    return season_bounds


def assign_valuation_season(
    val_date: pd.Timestamp,
    season_bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]],
) -> Optional[int]:
    """Determines which Premier League season a valuation date belongs to."""
    for season, (start, end) in season_bounds.items():
        if start <= val_date <= end:
            return season
    # Fallback to standard European football calendar (July 1 - June 30)
    year = val_date.year
    if val_date.month >= 7:
        return year
    else:
        return year - 1


def build_transfer_value_features(
    min_minutes: int = MIN_MINUTES_THRESHOLD,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Builds the point-in-time transfer value dataset.
    Aggregates season-to-date stats strictly before each valuation date.
    """
    logger.info("Loading datasets for transfer value point-in-time join...")
    players_df = load_players()
    valuations_df = load_player_valuations()
    appearances_df = load_appearances(competition_filter=GB1)
    games_df = load_games(competition_filter=GB1)

    # 1. Season map for games
    games_season = games_df.set_index("game_id")["season"].to_dict()
    season_bounds = compute_season_boundaries(games_df)

    # 2. Prepare appearances with season and datetime
    appearances = appearances_df[appearances_df["competition_id"] == GB1].copy()
    appearances["date"] = pd.to_datetime(appearances["date"])
    appearances["season"] = appearances["game_id"].map(games_season)
    # Filter appearances where season is known
    appearances = appearances.dropna(subset=["season", "player_id", "date"])
    appearances["season"] = appearances["season"].astype(int)

    # Group appearances by (player_id, season) sorted by date for fast slicing
    player_season_apps: Dict[Tuple[int, int], List[Dict]] = {}
    for _, app in appearances.iterrows():
        key = (int(app["player_id"]), int(app["season"]))
        player_season_apps.setdefault(key, []).append({
            "date": app["date"],
            "goals": float(app.get("goals", 0) or 0),
            "assists": float(app.get("assists", 0) or 0),
            "minutes_played": float(app.get("minutes_played", 0) or 0),
        })

    # 3. Prepare players info
    players_map = players_df.set_index("player_id")[
        ["name", "date_of_birth", "position", "sub_position", "current_club_name"]
    ].to_dict("index")

    # 4. Filter GB1 valuations
    # Valuations in GB1 or for players in GB1
    val_gb1 = valuations_df[valuations_df["player_club_domestic_competition_id"] == GB1].copy()
    val_gb1["date"] = pd.to_datetime(val_gb1["date"])
    val_gb1 = val_gb1.dropna(subset=["market_value_in_eur", "date", "player_id"])
    val_gb1 = val_gb1[val_gb1["market_value_in_eur"] > 0]

    logger.info("Evaluating %d GB1 valuation records against season appearances...", len(val_gb1))

    feature_rows = []
    for _, val_row in val_gb1.iterrows():
        p_id = int(val_row["player_id"])
        val_date = val_row["date"]
        market_val = float(val_row["market_value_in_eur"])
        val_club = str(val_row.get("current_club_name") or "")

        val_season = assign_valuation_season(val_date, season_bounds)
        if val_season is None:
            continue

        p_info = players_map.get(p_id)
        if not p_info:
            continue

        # Point-in-time appearance join:
        # competition_id == GB1 AND appearance.date < valuation.date AND appearance.season == val_season
        apps = player_season_apps.get((p_id, val_season), [])
        prior_apps = [a for a in apps if a["date"] < val_date]

        if not prior_apps:
            continue

        total_minutes = sum(a["minutes_played"] for a in prior_apps)
        if total_minutes < min_minutes:
            continue

        total_goals = sum(a["goals"] for a in prior_apps)
        total_assists = sum(a["assists"] for a in prior_apps)
        n_apps = len(prior_apps)

        # per-90 metrics
        goals_per_90 = (total_goals / total_minutes) * 90.0
        assists_per_90 = (total_assists / total_minutes) * 90.0

        # Player age at valuation date
        dob = p_info.get("date_of_birth")
        if pd.notna(dob):
            dob_dt = pd.to_datetime(dob)
            age = (val_date - dob_dt).days / 365.25
        else:
            age = 25.0

        pos = p_info.get("position") or "Midfield"
        club = val_club if val_club and val_club != "nan" else (p_info.get("current_club_name") or "Unknown")

        feature_rows.append({
            "player_id": p_id,
            "player_name": p_info.get("name") or "Unknown",
            "valuation_date": val_date,
            "season": val_season,
            "market_value_in_eur": market_val,
            "target_log": np.log1p(market_val),
            "minutes": total_minutes,
            "appearances": n_apps,
            "goals": total_goals,
            "assists": total_assists,
            "goals_per_90": goals_per_90,
            "assists_per_90": assists_per_90,
            "age": round(age, 1),
            "position": str(pos),
            "club": str(club),
        })

    df_feats = pd.DataFrame(feature_rows)
    logger.info(
        "Successfully constructed %d point-in-time valuation feature rows (min_minutes=%d).",
        len(df_feats),
        min_minutes,
    )
    return df_feats


def get_age_bucket(age: float) -> str:
    """Categorizes player age into standard demographic scouting buckets."""
    if age <= 21:
        return "<=21"
    elif age <= 25:
        return "22-25"
    elif age <= 29:
        return "26-29"
    else:
        return "30+"
