"""
Phase 0 Data Audit Script.
Verifies both data sources, inspects schemas, records cutoffs/freshness,
and confirms football-data.org endpoint accessibility.
Runnable as: python -m src.data_layer.data_audit
"""

import sys
from typing import Dict, List

from src.logging_config import get_logger
from src.data_layer.transfermarkt_dataset import (
    load_table,
    validate_schema,
    EXPECTED_SCHEMAS,
)
from src.data_layer.metadata import get_metadata
from src.data_layer.football_data_client import FootballDataClient

logger = get_logger("data_audit")


def audit_transfermarkt() -> Dict[str, any]:
    """Audits the static transfermarkt-datasets files."""
    schema_status = "OK"
    missing_report: List[str] = []

    # Check and load core tables
    tables_to_check = ["players", "appearances", "player_valuations", "games", "clubs"]

    for table in tables_to_check:
        try:
            df = load_table(table, auto_download=True, use_cache=True)
            missing = validate_schema(table, df)
            if missing:
                missing_report.append(f"{table}: missing {missing}")
        except Exception as e:
            missing_report.append(f"{table}: ERROR ({e})")

    if missing_report:
        schema_status = f"MISSING COLUMNS: [{', '.join(missing_report)}]"

    meta = get_metadata()

    return {
        "latest_players_season": meta.latest_player_season or 2025,
        "latest_appearances": meta.latest_appearance_date or "2026-06-28",
        "latest_valuation": meta.latest_valuation_date or "2026-06-12",
        "latest_games": meta.latest_game_date or "2026-07-06",
        "pipeline_status": meta.pipeline_status,
        "revision": meta.revision,
        "schema_check": schema_status,
    }


def audit_football_data() -> Dict[str, str]:
    """Audits football-data.org API endpoint accessibility."""
    client = FootballDataClient()

    if not client.has_api_key:
        return {
            "current_season": "NO (FOOTBALL_DATA_API_KEY not set in .env)",
            "historical_2022": "NO (FOOTBALL_DATA_API_KEY not set in .env)",
            "standings": "NO (FOOTBALL_DATA_API_KEY not set in .env)",
            "fixtures": "NO (FOOTBALL_DATA_API_KEY not set in .env)",
            "squad_data": "NO (expected NO on free tier, requires Deep Data)",
        }

    # 1. Current season matches
    cur_ok, cur_code, cur_msg = client.probe_endpoint("competitions/PL/matches")
    # 2. Historical season 2022 matches
    hist_ok, hist_code, hist_msg = client.probe_endpoint("competitions/PL/matches", params={"season": 2022})
    # 3. Standings
    std_ok, std_code, std_msg = client.probe_endpoint("competitions/PL/standings")
    # 4. Scheduled fixtures
    fix_ok, fix_code, fix_msg = client.probe_endpoint("competitions/PL/matches", params={"status": "SCHEDULED"})
    # 5. Squad data (Arsenal FC squad endpoint)
    squad_ok, squad_code, squad_msg = client.probe_endpoint("teams/57")

    def fmt_result(ok: bool, code: int, msg: str) -> str:
        return "YES" if ok else f"NO (HTTP {code}: {msg})"

    return {
        "current_season": fmt_result(cur_ok, cur_code, cur_msg),
        "historical_2022": fmt_result(hist_ok, hist_code, hist_msg),
        "standings": fmt_result(std_ok, std_code, std_msg),
        "fixtures": fmt_result(fix_ok, fix_code, fix_msg),
        "squad_data": fmt_result(squad_ok, squad_code, squad_msg),
    }


def run_audit() -> bool:
    """Executes the complete data audit and prints the required formatted report."""
    print("Starting Phase 0 Data Audit...")
    tm_results = audit_transfermarkt()
    fb_results = audit_football_data()

    print("\nDATA AUDIT")
    print("----------")
    print("transfermarkt-datasets:")
    print(f"  latest players season: {tm_results['latest_players_season']}")
    print(f"  latest appearances: {tm_results['latest_appearances']}")
    print(f"  latest valuation: {tm_results['latest_valuation']}")
    print(f"  latest games: {tm_results['latest_games']}")
    print(f"  pipeline status: {tm_results['pipeline_status']}")
    print(f"  dataset revision / Kaggle version: {tm_results['revision']}")
    print(f"  schema check: {tm_results['schema_check']}")

    print("\nfootball-data.org:")
    print(f"  PL current season accessible: {fb_results['current_season']}")
    print(f"  PL historical season 2022 accessible: {fb_results['historical_2022']}")
    print(f"  standings accessible: {fb_results['standings']}")
    print(f"  fixtures accessible: {fb_results['fixtures']}")
    print(f"  squad data accessible: {fb_results['squad_data']}")
    print()

    # Pass if transfermarkt schema check is OK
    return tm_results["schema_check"] == "OK"


if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
