"""
Loader and schema validator for static transfermarkt-datasets tables.
Downloads tables from Cloudflare R2 on demand, validates schemas, and updates DatasetMetadata.
"""

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd

from src.config import RAW_DATA_DIR, TRANSFERMARKT_R2_BASE_URL, GB1
from src.logging_config import get_logger
from src.data_layer.metadata import update_metadata, get_metadata

logger = get_logger(__name__)


class DatasetSchemaError(Exception):
    """Raised when an expected dataset table or required column is missing."""
    pass


# Schema expectations: table -> minimum required columns
EXPECTED_SCHEMAS: Dict[str, Set[str]] = {
    "players": {
        "player_id",
        "name",
        "last_season",
        "current_club_id",
        "date_of_birth",
        "position",
        "sub_position",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "current_club_name",
    },
    "player_valuations": {
        "player_id",
        "date",
        "market_value_in_eur",
        "current_club_id",
    },
    "appearances": {
        "appearance_id",
        "game_id",
        "player_id",
        "player_club_id",
        "date",
        "player_name",
        "competition_id",
        "goals",
        "assists",
        "minutes_played",
    },
    "games": {
        "game_id",
        "competition_id",
        "season",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "home_club_name",
        "away_club_name",
    },
    "clubs": {
        "club_id",
        "name",
        "domestic_competition_id",
    },
    "club_games": {
        "game_id",
        "club_id",
        "own_goals",
        "opponent_id",
        "opponent_goals",
        "hosting",
        "is_win",
    },
    "transfers": {
        "player_id",
        "transfer_season",
        "transfer_date",
        "from_club_id",
        "to_club_id",
    },
}

_LOADED_CACHE: Dict[str, pd.DataFrame] = {}


def get_table_path(table_name: str) -> Path:
    """Returns local path for a table, prioritizing .csv.gz or .csv."""
    gz_path = RAW_DATA_DIR / f"{table_name}.csv.gz"
    csv_path = RAW_DATA_DIR / f"{table_name}.csv"
    if gz_path.exists():
        return gz_path
    if csv_path.exists():
        return csv_path
    return gz_path


def download_table(table_name: str, force: bool = False) -> Path:
    """
    Downloads a table from the community Cloudflare R2 mirror if not present locally.
    Uses browser User-Agent to avoid Cloudflare 403 Forbidden.
    """
    target_gz = RAW_DATA_DIR / f"{table_name}.csv.gz"
    target_csv = RAW_DATA_DIR / f"{table_name}.csv"

    if not force and (target_gz.exists() or target_csv.exists()):
        return get_table_path(table_name)

    url = f"{TRANSFERMARKT_R2_BASE_URL}{table_name}.csv.gz"
    logger.info("Downloading %s from %s ...", table_name, url)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(target_gz, "wb") as out_file:
            shutil.copyfileobj(resp, out_file)
        logger.info("Successfully downloaded %s (%.2f MB)", table_name, target_gz.stat().st_size / 1e6)
        return target_gz
    except Exception as e:
        if target_gz.exists():
            target_gz.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download table '{table_name}' from {url}: {e}") from e


def validate_schema(table_name: str, df: pd.DataFrame) -> List[str]:
    """
    Validates DataFrame columns against EXPECTED_SCHEMAS.
    Returns list of missing columns (empty if all present).
    """
    expected = EXPECTED_SCHEMAS.get(table_name, set())
    missing = [col for col in expected if col not in df.columns]
    return missing


def load_table(
    table_name: str,
    competition_filter: Optional[str] = None,
    auto_download: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Loads a transfermarkt-dataset table, checking local presence, validating schema,
    and populating metadata.
    """
    cache_key = f"{table_name}:{competition_filter}"
    if use_cache and cache_key in _LOADED_CACHE:
        return _LOADED_CACHE[cache_key]

    path = get_table_path(table_name)
    if not path.exists():
        if auto_download:
            path = download_table(table_name)
        else:
            raise DatasetSchemaError(f"Table file {path} does not exist and auto_download is disabled.")

    logger.info("Loading table '%s' from %s...", table_name, path.name)
    df = pd.read_csv(path, low_memory=False)

    # Schema check: fail loudly if columns are missing
    missing = validate_schema(table_name, df)
    if missing:
        raise DatasetSchemaError(
            f"Schema validation FAILED for '{table_name}'. Missing required columns: {missing}"
        )

    # Apply competition filter if requested
    if competition_filter is not None:
        if "competition_id" in df.columns:
            df = df[df["competition_id"] == competition_filter].copy()
        elif "domestic_competition_id" in df.columns:
            df = df[df["domestic_competition_id"] == competition_filter].copy()

    # Update metadata dates based on loaded data
    if table_name == "players" and "last_season" in df.columns:
        max_season = int(df["last_season"].dropna().max())
        update_metadata(latest_player_season=max_season)

    elif table_name == "appearances" and "date" in df.columns:
        max_date = str(df["date"].dropna().max())
        update_metadata(latest_appearance_date=max_date)

    elif table_name == "player_valuations" and "date" in df.columns:
        max_date = str(df["date"].dropna().max())
        update_metadata(latest_valuation_date=max_date)

    elif table_name == "games" and "date" in df.columns:
        max_date = str(df["date"].dropna().max())
        update_metadata(latest_game_date=max_date)

    if use_cache:
        _LOADED_CACHE[cache_key] = df

    return df


def load_players(competition_filter: Optional[str] = None, **kwargs) -> pd.DataFrame:
    return load_table("players", competition_filter=competition_filter, **kwargs)


def load_player_valuations(**kwargs) -> pd.DataFrame:
    return load_table("player_valuations", **kwargs)


def load_appearances(competition_filter: Optional[str] = None, **kwargs) -> pd.DataFrame:
    return load_table("appearances", competition_filter=competition_filter, **kwargs)


def load_games(competition_filter: Optional[str] = None, **kwargs) -> pd.DataFrame:
    return load_table("games", competition_filter=competition_filter, **kwargs)


def load_clubs(competition_filter: Optional[str] = None, **kwargs) -> pd.DataFrame:
    return load_table("clubs", competition_filter=competition_filter, **kwargs)


def load_club_games(**kwargs) -> pd.DataFrame:
    return load_table("club_games", **kwargs)


def load_transfers(**kwargs) -> pd.DataFrame:
    return load_table("transfers", **kwargs)
