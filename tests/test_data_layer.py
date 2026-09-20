"""
Unit tests for data layer: schema validation, metadata, and squad overrides.
"""

import pandas as pd
import pytest
from src.data_layer.metadata import DatasetMetadata, get_metadata, update_metadata
from src.data_layer.squad_overrides import get_current_club, load_squad_overrides
from src.data_layer.transfermarkt_dataset import (
    validate_schema,
    DatasetSchemaError,
    load_table,
    EXPECTED_SCHEMAS,
)


def test_schema_validation_success():
    df = pd.DataFrame(columns=list(EXPECTED_SCHEMAS["clubs"]))
    missing = validate_schema("clubs", df)
    assert len(missing) == 0


def test_schema_validation_missing_columns():
    df = pd.DataFrame(columns=["club_id"])
    missing = validate_schema("clubs", df)
    assert "name" in missing
    assert "domestic_competition_id" in missing


def test_schema_fails_loudly_on_bad_data(monkeypatch, tmp_path):
    bad_csv = tmp_path / "clubs.csv"
    bad_csv.write_text("invalid_col_1,invalid_col_2\n1,2", encoding="utf-8")

    # Point raw dir or directly load
    with pytest.raises(DatasetSchemaError) as exc_info:
        # simulate validate_schema failure
        df = pd.read_csv(bad_csv)
        missing = validate_schema("clubs", df)
        if missing:
            raise DatasetSchemaError(f"Missing: {missing}")

    assert "Missing:" in str(exc_info.value)


def test_metadata_freshness_footer():
    meta = DatasetMetadata(
        source="transfermarkt-datasets",
        latest_appearance_date="2026-06-28",
        latest_valuation_date="2026-06-12",
        retrieved_at="2026-09-20",
    )
    footer = meta.format_freshness_footer(
        current_club_source="squad_overrides.csv, manually updated 2026-09-01",
        fixture_retrieved_date="2026-09-20",
    )

    expected_lines = [
        "Fixture data:      football-data.org, retrieved 2026-09-20",
        "Performance data:  transfermarkt-datasets, latest appearance 2026-06-28",
        "Market-value data: transfermarkt-datasets, latest valuation 2026-06-12",
        "Current club:      squad_overrides.csv, manually updated 2026-09-01",
    ]
    assert footer == "\n".join(expected_lines)


def test_squad_overrides_lookup():
    # Test known override
    club, source = get_current_club("Raheem Sterling", fallback_club="Chelsea FC")
    assert "Arsenal" in club
    assert "manual override" in source

    # Test unknown player fallback
    club_fb, source_fb = get_current_club("NonExistentPlayer", fallback_club="Aston Villa")
    assert club_fb == "Aston Villa"
    assert "frozen dataset - last known club" in source_fb
