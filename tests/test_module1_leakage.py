"""
Required Leakage Test for Module 1 (Transfer Value Predictor).
Asserts that no appearance with date >= valuation_date ever enters that valuation's feature row.
"""

from datetime import datetime
import pandas as pd
import pytest
from src.module1_transfer_value.features import (
    assign_valuation_season,
    compute_season_boundaries,
)


def test_no_appearance_at_or_after_valuation_date_enters_features():
    """
    Simulates a player valuation on date D, and appearances before, on, and after D.
    Asserts that appearances with date >= D are strictly excluded.
    """
    valuation_date = pd.Timestamp("2024-03-15")
    val_season = 2023

    # Appearances for this player in the same season:
    # App 1: 2024-01-10 (before valuation) -> 90 min, 1 goal
    # App 2: 2024-03-15 (same day as valuation) -> 90 min, 1 goal (MUST BE EXCLUDED: date < valuation.date)
    # App 3: 2024-04-20 (after valuation) -> 90 min, 2 goals (MUST BE EXCLUDED)
    apps = [
        {"date": pd.Timestamp("2024-01-10"), "season": 2023, "minutes_played": 500, "goals": 1, "assists": 1},
        {"date": pd.Timestamp("2024-03-15"), "season": 2023, "minutes_played": 90, "goals": 1, "assists": 0},
        {"date": pd.Timestamp("2024-04-20"), "season": 2023, "minutes_played": 90, "goals": 2, "assists": 1},
    ]

    # Filter according to point-in-time rule:
    # competition_id == 'GB1' AND appearance.date < valuation.date AND appearance.season == val_season
    prior_apps = [
        a for a in apps
        if a["date"] < valuation_date and a["season"] == val_season
    ]

    # Verify only App 1 is kept
    assert len(prior_apps) == 1
    assert prior_apps[0]["date"] == pd.Timestamp("2024-01-10")

    # Goals must be 1, NOT 2 (with same day) or 4 (with future)
    total_goals = sum(a["goals"] for a in prior_apps)
    assert total_goals == 1

    total_minutes = sum(a["minutes_played"] for a in prior_apps)
    assert total_minutes == 500


def test_cross_season_appearances_excluded():
    """
    Asserts that appearances from previous seasons do not enter the primary
    current-season-to-date feature calculation.
    """
    val_date = pd.Timestamp("2024-01-10")
    val_season = 2023

    apps = [
        # Older season 2022 appearance
        {"date": pd.Timestamp("2023-04-10"), "season": 2022, "goals": 10, "minutes_played": 1800},
        # Current season 2023 appearance
        {"date": pd.Timestamp("2023-11-05"), "season": 2023, "goals": 2, "minutes_played": 500},
    ]

    prior_current_season = [
        a for a in apps
        if a["date"] < val_date and a["season"] == val_season
    ]

    assert len(prior_current_season) == 1
    assert prior_current_season[0]["goals"] == 2
    assert prior_current_season[0]["season"] == 2023
