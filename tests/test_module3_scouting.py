"""
Unit tests for Module 3: Player Scouting Dashboard features and similarity logic.
"""

import numpy as np
import pandas as pd
import pytest
from src.module3_scouting.features import normalize_position
from src.module3_scouting.cluster import compute_weighted_similarity
from sklearn.preprocessing import StandardScaler


def test_position_normalization():
    assert normalize_position("Centre-Forward") == "ATT"
    assert normalize_position("Left Winger") == "ATT"
    assert normalize_position("Right-Back") == "DEF"
    assert normalize_position("Centre-Back") == "DEF"
    assert normalize_position("Central Midfield") == "MID"
    assert normalize_position("Defensive Midfield") == "MID"
    assert normalize_position("Goalkeeper") == "GK"


def test_clustering_and_similarity_are_separate():
    """
    Asserts that changing user weights changes the similarity ranking
    without altering cluster assignments, proving separation of clustering and similarity.
    """
    # Create 3 synthetic player records:
    # Player 0: balanced target
    # Player 1: high attack, low defense
    # Player 2: low attack, high durability/defense
    records = [
        {"name": "P0", "goals_per_90": 0.3, "assists_per_90": 0.2, "goal_involvements_per_90": 0.5, "minutes_played": 1500, "appearances": 20, "cards_per_90": 0.1, "position_group": "MID", "cluster": 0},
        {"name": "P1", "goals_per_90": 0.9, "assists_per_90": 0.5, "goal_involvements_per_90": 1.4, "minutes_played": 1000, "appearances": 15, "cards_per_90": 0.2, "position_group": "ATT", "cluster": 1},
        {"name": "P2", "goals_per_90": 0.05, "assists_per_90": 0.05, "goal_involvements_per_90": 0.1, "minutes_played": 2800, "appearances": 32, "cards_per_90": 0.1, "position_group": "DEF", "cluster": 2},
    ]
    df = pd.DataFrame(records)
    scaler = StandardScaler()
    scaler.fit(df[["goals_per_90", "assists_per_90", "goal_involvements_per_90", "minutes_played", "appearances", "cards_per_90"]])

    # Weight attacking heavily (5x) vs defense (0.1x)
    rank_attack = compute_weighted_similarity(target_idx=0, df=df, scaler=scaler, attack_weight=5.0, defense_weight=0.1)

    # Weight defense heavily (5x) vs attack (0.1x)
    rank_defense = compute_weighted_similarity(target_idx=0, df=df, scaler=scaler, attack_weight=0.1, defense_weight=5.0)

    # The rankings or distances MUST differ between attacking vs defensive weighting
    assert rank_attack.iloc[0]["name"] != rank_defense.iloc[0]["name"] or rank_attack.iloc[0]["distance"] != rank_defense.iloc[0]["distance"]
    # Cluster assignment remains unchanged
    assert df.loc[1, "cluster"] == 1
    assert df.loc[2, "cluster"] == 2
