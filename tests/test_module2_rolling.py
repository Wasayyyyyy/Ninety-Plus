"""
Unit tests for rolling form calculations in Module 2.
"""

import pandas as pd
from src.module2_match_outcome.features import build_pre_match_features


def test_rolling_form_window_limit():
    """
    Tests that rolling form respects the rolling window (last 5 matches),
    ignoring older matches beyond the window.
    """
    # Create 7 matches for Team A:
    # Matches 1..2: loses (0 pts)
    # Matches 3..7: wins (3 pts each)
    # At match 8: rolling 5 pts should be 5 * 3 = 15, NOT including matches 1 and 2
    records = []
    for i in range(1, 8):
        records.append({
            "game_id": 200 + i,
            "competition_id": "GB1",
            "season": 2024,
            "date": f"2024-09-{i:02d}",
            "home_club_id": 10,
            "away_club_id": 20 + i,
            "home_club_name": "Team A",
            "away_club_name": f"Opponent {i}",
            "home_club_goals": 0 if i <= 2 else 2,
            "away_club_goals": 1 if i <= 2 else 0,
        })

    # Match 8
    records.append({
        "game_id": 208,
        "competition_id": "GB1",
        "season": 2024,
        "date": "2024-09-08",
        "home_club_id": 10,
        "away_club_id": 99,
        "home_club_name": "Team A",
        "away_club_name": "Opponent 8",
        "home_club_goals": 1,
        "away_club_goals": 1,
    })

    df = pd.DataFrame(records)
    feats = build_pre_match_features(df, rolling_window=5)

    m8 = feats[feats["game_id"] == 208].iloc[0]
    # Prior 5 matches (matches 3, 4, 5, 6, 7) were all wins -> 15 points
    assert m8["home_roll_pts"] == 15.0
    assert m8["home_season_played"] == 7.0
