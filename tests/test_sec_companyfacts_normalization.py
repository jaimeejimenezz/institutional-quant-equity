"""Tests for SEC Company Facts normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.data import (
    SECNormalizationError,
    build_companyfacts_quality_summary,
    normalize_companyfacts_payload,
    validate_normalized_companyfacts,
)


def make_payload() -> dict:
    """Create a representative SEC Company Facts payload."""
    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": ("Total assets."),
                    "units": {
                        "USD": [
                            {
                                "end": "2023-09-30",
                                "val": 352583000000,
                                "accn": ("0000320193-23-000106"),
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-11-03",
                                "frame": "CY2023Q3I",
                            }
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "Net Income Loss",
                    "description": ("Net income."),
                    "units": {
                        "USD": [
                            {
                                "start": "2022-09-25",
                                "end": "2023-09-30",
                                "val": 96995000000,
                                "accn": ("0000320193-23-000106"),
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-11-03",
                                "frame": "CY2023",
                            }
                        ]
                    },
                },
            },
            "dei": {
                "EntityPublicFloat": {
                    "label": ("Entity Public Float"),
                    "description": ("Public float."),
                    "units": {
                        "USD": [
                            {
                                "end": "2023-03-31",
                                "val": 2500000000000,
                                "accn": ("0000320193-23-000077"),
                                "fy": 2023,
                                "fp": "Q2",
                                "form": "10-Q",
                                "filed": "2023-05-05",
                            }
                        ]
                    },
                }
            },
        },
    }


def test_companyfacts_are_flattened() -> None:
    """Every SEC observation should become one row."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    assert len(data) == 3

    assert set(data["concept"]) == {
        "Assets",
        "NetIncomeLoss",
        "EntityPublicFloat",
    }


def test_instant_fact_has_no_start_date() -> None:
    """Balance-sheet facts may be instantaneous."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    assets = data.loc[data["concept"].eq("Assets")].iloc[0]

    assert pd.isna(assets["start_date"])

    assert assets["end_date"] == pd.Timestamp("2023-09-30")


def test_duration_fact_preserves_dates() -> None:
    """Income-statement facts should preserve start/end."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    net_income = data.loc[data["concept"].eq("NetIncomeLoss")].iloc[0]

    assert net_income["start_date"] == pd.Timestamp("2022-09-25")

    assert net_income["end_date"] == pd.Timestamp("2023-09-30")


def test_filing_metadata_are_preserved() -> None:
    """Point-in-time metadata must survive normalization."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    assets = data.loc[data["concept"].eq("Assets")].iloc[0]

    assert assets["filed_date"] == pd.Timestamp("2023-11-03")

    assert assets["form"] == "10-K"

    assert assets["fiscal_year"] == 2023

    assert assets["fiscal_period"] == "FY"

    assert assets["accession_number"] == ("0000320193-23-000106")


def test_multiple_taxonomies_are_preserved() -> None:
    """US-GAAP and DEI facts should remain distinguishable."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    assert set(data["taxonomy"]) == {
        "us-gaap",
        "dei",
    }


def test_invalid_numeric_value_is_rejected() -> None:
    """Non-numeric XBRL values should fail normalization."""
    payload = make_payload()

    payload["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"] = "not-a-number"

    with pytest.raises(
        SECNormalizationError,
        match="non-numeric",
    ):
        normalize_companyfacts_payload(
            payload,
            ticker="AAPL",
            source_file="AAPL.json",
        )


def test_quality_summary_is_created() -> None:
    """Normalization should produce company diagnostics."""
    data = normalize_companyfacts_payload(
        make_payload(),
        ticker="AAPL",
        source_file="AAPL.json",
    )

    validate_normalized_companyfacts(data)

    summary = build_companyfacts_quality_summary(data)

    assert len(summary) == 1

    assert summary["observation_count"].iloc[0] == 3

    assert summary["concept_count"].iloc[0] == 3
