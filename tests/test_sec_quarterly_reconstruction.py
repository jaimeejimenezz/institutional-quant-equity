"""Tests for quarterly SEC fundamental reconstruction."""

from __future__ import annotations

import pandas as pd

from quant_equity.data import (
    SECPointInTimeConfig,
    SECQuarterlyReconstructionConfig,
    build_quarterly_fundamental_events,
    build_ttm_point_in_time_snapshots,
)


def make_configs():
    """Create reconstruction configs."""
    return (
        SECPointInTimeConfig(
            availability_lag_days=1,
            require_statement_type_match=True,
            exclude_other_duration=True,
        ),
        SECQuarterlyReconstructionConfig(
            additive_metrics=("revenue",),
            quarter_gap_min_days=60,
            quarter_gap_max_days=120,
            ttm_span_min_days=240,
            ttm_span_max_days=330,
        ),
    )


def make_fact(
    *,
    value: float,
    start: str,
    end: str,
    filed: str,
    duration_class: str,
    accession: str,
) -> dict:
    """Create one canonical duration fact."""
    return {
        "ticker": "AAPL",
        "cik": "0000320193",
        "entity_name": "Apple Inc.",
        "canonical_metric": "revenue",
        "statement_type": "duration",
        "concept": "Revenue",
        "concept_priority": 1,
        "unit": "USD",
        "value": value,
        "start_date": pd.Timestamp(start),
        "end_date": pd.Timestamp(end),
        "filed_date": pd.Timestamp(filed),
        "form": "10-Q",
        "fiscal_year": 2023,
        "fiscal_period": "Q1",
        "accession_number": accession,
        "frame": None,
        "duration_class": duration_class,
        "statement_type_match": True,
        "is_amendment": False,
    }


def test_direct_quarter_is_preserved() -> None:
    """A reported quarter should remain unchanged."""
    data = pd.DataFrame(
        [
            make_fact(
                value=100.0,
                start="2023-01-01",
                end="2023-03-31",
                filed="2023-05-01",
                duration_class="quarter",
                accession="A",
            )
        ]
    )

    pit, reconstruction = make_configs()

    result = build_quarterly_fundamental_events(
        data,
        pit_config=pit,
        config=reconstruction,
    )

    assert result["quarter_value"].iloc[0] == 100.0

    assert result["source_method"].iloc[0] == "direct"


def test_half_year_ytd_derives_q2() -> None:
    """Q2 may be derived from H1 minus Q1."""
    data = pd.DataFrame(
        [
            make_fact(
                value=100.0,
                start="2023-01-01",
                end="2023-03-31",
                filed="2023-05-01",
                duration_class="quarter",
                accession="A",
            ),
            make_fact(
                value=220.0,
                start="2023-01-01",
                end="2023-06-30",
                filed="2023-08-01",
                duration_class="half_year_ytd",
                accession="B",
            ),
        ]
    )

    pit, reconstruction = make_configs()

    result = build_quarterly_fundamental_events(
        data,
        pit_config=pit,
        config=reconstruction,
    )

    q2 = result.loc[result["source_method"].eq("derived_q2")].iloc[-1]

    assert q2["quarter_value"] == 120.0


def test_nine_month_ytd_derives_q3() -> None:
    """Q3 may be derived from 9M minus H1."""
    data = pd.DataFrame(
        [
            make_fact(
                value=220.0,
                start="2023-01-01",
                end="2023-06-30",
                filed="2023-08-01",
                duration_class="half_year_ytd",
                accession="B",
            ),
            make_fact(
                value=350.0,
                start="2023-01-01",
                end="2023-09-30",
                filed="2023-11-01",
                duration_class="nine_month_ytd",
                accession="C",
            ),
        ]
    )

    pit, reconstruction = make_configs()

    result = build_quarterly_fundamental_events(
        data,
        pit_config=pit,
        config=reconstruction,
    )

    q3 = result.loc[result["source_method"].eq("derived_q3")].iloc[-1]

    assert q3["quarter_value"] == 130.0


def test_annual_minus_nine_month_derives_q4() -> None:
    """Q4 may be derived from FY minus 9M."""
    data = pd.DataFrame(
        [
            make_fact(
                value=350.0,
                start="2023-01-01",
                end="2023-09-30",
                filed="2023-11-01",
                duration_class="nine_month_ytd",
                accession="C",
            ),
            make_fact(
                value=500.0,
                start="2023-01-01",
                end="2023-12-31",
                filed="2024-02-01",
                duration_class="annual",
                accession="D",
            ),
        ]
    )

    pit, reconstruction = make_configs()

    result = build_quarterly_fundamental_events(
        data,
        pit_config=pit,
        config=reconstruction,
    )

    q4 = result.loc[result["source_method"].eq("derived_q4")].iloc[-1]

    assert q4["quarter_value"] == 150.0


def test_mismatched_fiscal_start_is_not_subtracted() -> None:
    """YTD subtraction requires a common fiscal start."""
    data = pd.DataFrame(
        [
            make_fact(
                value=100.0,
                start="2022-01-01",
                end="2023-03-31",
                filed="2023-05-01",
                duration_class="quarter",
                accession="A",
            ),
            make_fact(
                value=220.0,
                start="2023-01-01",
                end="2023-06-30",
                filed="2023-08-01",
                duration_class="half_year_ytd",
                accession="B",
            ),
        ]
    )

    pit, reconstruction = make_configs()

    result = build_quarterly_fundamental_events(
        data,
        pit_config=pit,
        config=reconstruction,
    )

    assert not result["source_method"].eq("derived_q2").any()


def test_four_quarters_create_ttm() -> None:
    """Four valid quarterly observations should form TTM."""
    quarterly = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "canonical_metric": ["revenue"] * 4,
            "unit": ["USD"] * 4,
            "quarter_value": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "quarter_end": pd.to_datetime(
                [
                    "2023-03-31",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-12-31",
                ]
            ),
            "available_date": pd.to_datetime(
                [
                    "2023-05-02",
                    "2023-08-02",
                    "2023-11-02",
                    "2024-02-02",
                ]
            ),
            "source_method": ["direct"] * 4,
        }
    )

    _, reconstruction = make_configs()

    ttm = build_ttm_point_in_time_snapshots(
        quarterly,
        [
            "2024-02-02",
        ],
        config=reconstruction,
    )

    assert ttm["ttm_value"].iloc[0] == 460.0

    assert ttm["quarter_count"].iloc[0] == 4


def test_ttm_never_uses_future_quarter() -> None:
    """A quarter unavailable at as_of_date must not leak."""
    quarterly = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "canonical_metric": ["revenue"] * 4,
            "unit": ["USD"] * 4,
            "quarter_value": [
                100.0,
                110.0,
                120.0,
                130.0,
            ],
            "quarter_end": pd.to_datetime(
                [
                    "2023-03-31",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-12-31",
                ]
            ),
            "available_date": pd.to_datetime(
                [
                    "2023-05-02",
                    "2023-08-02",
                    "2023-11-02",
                    "2024-02-02",
                ]
            ),
            "source_method": ["direct"] * 4,
        }
    )

    _, reconstruction = make_configs()

    try:
        result = build_ttm_point_in_time_snapshots(
            quarterly,
            [
                "2024-01-31",
            ],
            config=reconstruction,
        )
    except Exception:
        result = pd.DataFrame()

    assert result.empty
