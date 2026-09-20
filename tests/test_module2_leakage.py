"""
Required Leakage Test for Module 2 (Match Outcome Predictor).
Asserts that no match uses information from that match or any later match.
"""

import pandas as pd
import numpy as np
import pytest
from src.module2_match_outcome.features import build_pre_match_features


def test_no_future_information_in_match_features():
    """
    Creates a synthetic fixture schedule where team results change drastically,
    and asserts that modifying future match scores NEVER changes past match features.
    """
    # 4 synthetic matches between Team 1 and Team 2
    synthetic_games = pd.DataFrame([
        {
            "game_id": 101,
            "competition_id": "GB1",
            "season": 2024,
            "date": "2024-09-01",
            "home_club_id": 1,
            "away_club_id": 2,
            "home_club_name": "Team 1",
            "away_club_name": "Team 2",
            "home_club_goals": 1,
            "away_club_goals": 0,
        },
        {
            "game_id": 102,
            "competition_id": "GB1",
            "season": 2024,
            "date": "2024-09-08",
            "home_club_id": 2,
            "away_club_id": 1,
            "home_club_name": "Team 2",
            "away_club_name": "Team 1",
            "home_club_goals": 2,
            "away_club_goals": 2,
        },
        {
            "game_id": 103,
            "competition_id": "GB1",
            "season": 2024,
            "date": "2024-09-15",
            "home_club_id": 1,
            "away_club_id": 2,
            "home_club_name": "Team 1",
            "away_club_name": "Team 2",
            "home_club_goals": 0,
            "away_club_goals": 5,
        },
    ])

    features_orig = build_pre_match_features(synthetic_games)
    m1_orig = features_orig[features_orig["game_id"] == 101].iloc[0]
    m2_orig = features_orig[features_orig["game_id"] == 102].iloc[0]

    # In match 1 (the first match), both teams must have 0 prior season games and 0 prior points
    assert m1_orig["home_season_played"] == 0
    assert m1_orig["away_season_played"] == 0
    assert m1_orig["home_roll_pts"] == 0
    assert m1_orig["away_roll_pts"] == 0

    # In match 2 (second match), Team 1 won match 1, so Team 1 should have 3 pts; Team 2 lost match 1 so 0 pts
    assert m2_orig["away_season_played"] == 1  # Team 1 is away in match 2
    assert m2_orig["away_roll_pts"] == 3
    assert m2_orig["home_season_played"] == 1  # Team 2 is home in match 2
    assert m2_orig["home_roll_pts"] == 0

    # Now simulate a run where match 3 (future) has completely altered score (e.g. 10-0 instead of 0-5)
    altered_games = synthetic_games.copy()
    altered_games.loc[altered_games["game_id"] == 103, "home_club_goals"] = 10
    altered_games.loc[altered_games["game_id"] == 103, "away_club_goals"] = 0

    features_altered = build_pre_match_features(altered_games)
    m1_altered = features_altered[features_altered["game_id"] == 101].iloc[0]
    m2_altered = features_altered[features_altered["game_id"] == 102].iloc[0]

    # Future match 103 must NOT alter past match 101 or 102 features in any way
    for col in ["diff_season_ppm", "diff_roll_pts", "home_roll_pts", "away_roll_pts"]:
        assert m1_orig[col] == m1_altered[col], f"Leakage detected in match 101 col {col}!"
        assert m2_orig[col] == m2_altered[col], f"Leakage detected in match 102 col {col}!"


def test_match_does_not_leak_into_own_features():
    """
    Asserts that the goals scored in match M do NOT appear in match M's pre-match rolling goals.
    """
    single_game = pd.DataFrame([
        {
            "game_id": 999,
            "competition_id": "GB1",
            "season": 2024,
            "date": "2024-09-01",
            "home_club_id": 50,
            "away_club_id": 60,
            "home_club_name": "A",
            "away_club_name": "B",
            "home_club_goals": 7,
            "away_club_goals": 4,
        }
    ])
    feats = build_pre_match_features(single_game)
    row = feats.iloc[0]

    # Pre-match rolling goals for this game MUST be 0, not 7 or 4
    assert row["home_roll_gf"] == 0.0
    assert row["away_roll_gf"] == 0.0
    assert row["home_season_played"] == 0.0
