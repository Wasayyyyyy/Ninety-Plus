"""
DatasetMetadata dataclass and standardized freshness footer generation.
Single source of truth for tracking source versions, cutoffs, and live call timestamps.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class DatasetMetadata:
    source: str = "transfermarkt-datasets"
    revision: str = "frozen-2026-07-06"
    pipeline_status: str = "PAUSED"
    latest_player_season: Optional[int] = 2025
    latest_appearance_date: Optional[str] = "2026-06-28"
    latest_valuation_date: Optional[str] = "2026-06-12"
    latest_game_date: Optional[str] = "2026-07-06"
    retrieved_at: Optional[str] = None

    def format_freshness_footer(
        self,
        current_club_source: str = "squad_overrides.csv, manually updated 2024-09-01",
        fixture_retrieved_date: Optional[str] = None,
    ) -> str:
        """
        Formats the standardized 4-line data freshness footer required by the project brief.
        """
        retrieval = fixture_retrieved_date or self.retrieved_at or datetime.now().strftime("%Y-%m-%d")
        app_date = self.latest_appearance_date or "N/A"
        val_date = self.latest_valuation_date or "N/A"

        lines = [
            f"Fixture data:      football-data.org, retrieved {retrieval}",
            f"Performance data:  {self.source}, latest appearance {app_date}",
            f"Market-value data: {self.source}, latest valuation {val_date}",
            f"Current club:      {current_club_source}",
        ]
        return "\n".join(lines)


# Global singleton instance updated on data load
GLOBAL_METADATA = DatasetMetadata()


def get_metadata() -> DatasetMetadata:
    return GLOBAL_METADATA


def update_metadata(**kwargs) -> DatasetMetadata:
    global GLOBAL_METADATA
    for k, v in kwargs.items():
        if hasattr(GLOBAL_METADATA, k) and v is not None:
            setattr(GLOBAL_METADATA, k, v)
    return GLOBAL_METADATA
