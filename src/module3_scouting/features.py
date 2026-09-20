"""
Player-season feature aggregation and position normalization for Module 3: Scouting Dashboard.
Maps granular Transfermarkt positions into broad tactical groups (GK, DEF, MID, ATT).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import GB1, MIN_MINUTES_THRESHOLD
from src.logging_config import get_logger
from src.data_layer.transfermarkt_dataset import (
    load_appearances,
    load_games,
    load_players,
)

logger = get_logger(__name__)

# Position taxonomy: granular Transfermarkt positions -> broad groups
POSITION_MAP: Dict[str, str] = {
    # Goalkeepers
    "Goalkeeper": "GK",
    # Defenders
    "Centre-Back": "DEF",
    "Left-Back": "DEF",
    "Right-Back": "DEF",
    "Defender": "DEF",
    # Midfielders
    "Defensive Midfield": "MID",
    "Central Midfield": "MID",
    "Attacking Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Midfield": "MID",
    # Attackers
    "Centre-Forward": "ATT",
    "Left Winger": "ATT",
    "Right Winger": "ATT",
    "Second Striker": "ATT",
    "Attack": "ATT",
}


def normalize_position(granular_pos: Optional[str], broad_pos: Optional[str] = None) -> str:
    """Normalizes granular or broad position into one of GK, DEF, MID, ATT."""
    if granular_pos and str(granular_pos) in POSITION_MAP:
        return POSITION_MAP[str(granular_pos)]
    if broad_pos and str(broad_pos) in POSITION_MAP:
        return POSITION_MAP[str(broad_pos)]

    pos_str = str(granular_pos or broad_pos or "").lower()
    if "goal" in pos_str or "keeper" in pos_str:
        return "GK"
    elif "back" in pos_str or "defend" in pos_str:
        return "DEF"
    elif "midfield" in pos_str:
        return "MID"
    elif "forward" in pos_str or "wing" in pos_str or "attack" in pos_str or "striker" in pos_str:
        return "ATT"
    return "MID"


def build_scouting_features(min_minutes: int = MIN_MINUTES_THRESHOLD) -> pd.DataFrame:
    """
    Builds per-player-season aggregated statistics normalized per 90 minutes.
    """
    logger.info("Aggregating player-season statistics for scouting module...")
    players_df = load_players()
    appearances_df = load_appearances(competition_filter=GB1)
    games_df = load_games(competition_filter=GB1)

    games_season = games_df.set_index("game_id")["season"].to_dict()

    apps = appearances_df.copy()
    apps["season"] = apps["game_id"].map(games_season)
    apps = apps.dropna(subset=["season", "player_id", "minutes_played"])
    apps["season"] = apps["season"].astype(int)

    # Aggregate per (player_id, season)
    agg = apps.groupby(["player_id", "season"]).agg({
        "minutes_played": "sum",
        "goals": "sum",
        "assists": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "appearance_id": "count",
    }).reset_index()

    agg.rename(columns={"appearance_id": "appearances"}, inplace=True)

    # Filter min minutes
    agg = agg[agg["minutes_played"] >= min_minutes].copy()

    # Join with player details
    players_subset = players_df[[
        "player_id", "name", "position", "sub_position", "date_of_birth",
        "current_club_name", "market_value_in_eur"
    ]].drop_duplicates(subset=["player_id"]).copy()

    merged = pd.merge(agg, players_subset, on="player_id", how="inner")

    # Compute normalized per-90 metrics
    mins = merged["minutes_played"]
    merged["goals_per_90"] = (merged["goals"] / mins) * 90.0
    merged["assists_per_90"] = (merged["assists"] / mins) * 90.0
    merged["goal_involvements_per_90"] = ((merged["goals"] + merged["assists"]) / mins) * 90.0
    merged["cards_per_90"] = ((merged["yellow_cards"] + merged["red_cards"]) / mins) * 90.0

    # Normalize position taxonomy
    merged["position_group"] = merged.apply(
        lambda r: normalize_position(r["sub_position"], r["position"]), axis=1
    )

    # Compute player age during that season
    if "date_of_birth" in merged.columns:
        dob = pd.to_datetime(merged["date_of_birth"], errors="coerce")
        # Approximate season date as mid-season (January 1 of season + 1)
        season_mid = pd.to_datetime(merged["season"].astype(str) + "-01-01")
        merged["age"] = ((season_mid - dob).dt.days / 365.25).round(1).fillna(25.0)
    else:
        merged["age"] = 25.0

    merged["display_name"] = merged["name"] + " (" + merged["season"].astype(str) + "/" + (merged["season"] + 1).astype(str).str[-2:] + ")"

    logger.info("Scouting dataset created: %d player-seasons across groups %s", len(merged), merged["position_group"].value_counts().to_dict())
    return merged
