"""
Squad overrides loader and current-club resolver.
Implements the hand-maintained transfer updates layer.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
from src.config import RAW_DATA_DIR
from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_OVERRIDES_PATH = RAW_DATA_DIR / "squad_overrides.csv"

_OVERRIDES_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def load_squad_overrides(csv_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Loads manual squad overrides from CSV into a normalized dictionary lookup.
    """
    global _OVERRIDES_CACHE
    path = csv_path or DEFAULT_OVERRIDES_PATH
    if not path.exists():
        logger.debug("Squad overrides file not found at %s. Proceeding with empty overrides.", path)
        return {}

    try:
        df = pd.read_csv(path)
        expected_cols = {"player_name", "new_club", "effective_date"}
        if not expected_cols.issubset(df.columns):
            logger.warning(
                "squad_overrides.csv missing required columns %s. Found: %s",
                expected_cols - set(df.columns),
                list(df.columns),
            )
            return {}

        overrides = {}
        for _, row in df.iterrows():
            p_name = str(row["player_name"]).strip().lower()
            overrides[p_name] = {
                "player_name": str(row["player_name"]).strip(),
                "new_club": str(row["new_club"]).strip(),
                "effective_date": str(row.get("effective_date", "")).strip(),
                "source_note": str(row.get("source_note", "")).strip(),
            }
        _OVERRIDES_CACHE = overrides
        return overrides
    except Exception as e:
        logger.warning("Error reading squad overrides: %s", e)
        return {}


def get_current_club(
    player_name: str,
    fallback_club: Optional[str] = None,
    csv_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """
    Resolves the current club for a player.
    Lookup order:
    1. squad_overrides.csv if present -> ('Arsenal FC', 'squad_overrides.csv, manually updated 2024-08-30')
    2. fallback_club from frozen dataset -> (fallback_club, 'frozen dataset — last known club (2025/26)')

    Returns:
        (resolved_club_name, source_label)
    """
    overrides = load_squad_overrides(csv_path)
    clean_name = player_name.strip().lower()

    if clean_name in overrides:
        entry = overrides[clean_name]
        club = entry["new_club"]
        date_str = entry["effective_date"] or "recently"
        source_label = f"squad_overrides.csv (manual override), updated {date_str}"
        return club, source_label

    club = fallback_club if fallback_club and pd.notna(fallback_club) else "Unknown Club"
    source_label = "frozen dataset - last known club (2025/26)"
    return club, source_label
