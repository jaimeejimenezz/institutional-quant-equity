"""Tests for SEC point-in-time reconstruction."""

from __future__ import annotations

import pandas as pd

from quant_equity.data import (
    SECPointInTimeConfig,
    build_point_in_time_snapshots,
    get_fundamentals_as_of,
)


def make_config() -> SECPointInTimeConfig:
    """Create conservative PIT configuration."""
    return SECPointInTimeConfig(
        availability_lag_days=1,
        require_statement_type_match=True,
        exclude_other_duration=True,
    )


def make_fact(
    *,
    metric: str,
    duration_class: str,
    value: float,
    end_date: str,
    filed_date: str,
    start_date: str | None = None,
    concept: str = "Concept",
    priority: int = 1,
    accession: str = "A",
    amendment: bool = False,
    statement_type_match: bool = True,
) -> dict:
    """Create one canonical SEC fact."""
    statement_type = "instant" if duration_class == "instant" else "duration"

    return {
        "ticker": "AAPL",
        "cik": "0000320193",
        "entity_name": "Apple Inc.",
        "canonical_metric": metric,
        "statement_type": statement_type,
        "concept": concept,
        "concept_priority": priority,
        "unit": "USD",
        "value": value,
        "start_date": (pd.Timestamp(start_date) if start_date else pd.NaT),
        "end_date": pd.Timestamp(end_date),
        "filed_date": pd.Timestamp(filed_date),
        "form": ("10-Q/A" if amendment else "10-Q"),
        "fiscal_year": 2023,
        "fiscal_period": "Q1",
        "accession_number": accession,
        "frame": None,
        "duration_class": duration_class,
        "statement_type_match": (statement_type_match),
        "is_amendment": amendment,
    }


def test_same_day_filing_is_not_available() -> None:
    """A filing should respect the configured availability lag."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="assets",
                duration_class="instant",
                value=300.0,
                end_date="2023-03-31",
                filed_date="2023-05-01",
            )
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-05-01",
            "2023-05-02",
        ],
        config=make_config(),
    )

    assert pd.Timestamp("2023-05-01") not in set(snapshots["as_of_date"])

    assert pd.Timestamp("2023-05-02") in set(snapshots["as_of_date"])


def test_future_period_does_not_leak() -> None:
    """A future filing must not enter an earlier snapshot."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="assets",
                duration_class="instant",
                value=300.0,
                end_date="2023-03-31",
                filed_date="2023-05-01",
                accession="A",
            ),
            make_fact(
                metric="assets",
                duration_class="instant",
                value=320.0,
                end_date="2023-06-30",
                filed_date="2023-08-01",
                accession="B",
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-07-31",
            "2023-08-02",
        ],
        config=make_config(),
    )

    july = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-07-31",
    )

    august = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-08-02",
    )

    assert july["value"].iloc[0] == 300.0

    assert august["value"].iloc[0] == 320.0


def test_amendment_replaces_original_after_filing() -> None:
    """A later amendment may update the same reporting period."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="assets",
                duration_class="instant",
                value=300.0,
                end_date="2023-03-31",
                filed_date="2023-05-01",
                accession="A",
            ),
            make_fact(
                metric="assets",
                duration_class="instant",
                value=305.0,
                end_date="2023-03-31",
                filed_date="2023-05-15",
                accession="B",
                amendment=True,
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-05-10",
            "2023-05-16",
        ],
        config=make_config(),
    )

    before = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-05-10",
    )

    after = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-05-16",
    )

    assert before["value"].iloc[0] == 300.0

    assert after["value"].iloc[0] == 305.0


def test_preferred_concept_wins_same_period() -> None:
    """Concept priority should resolve alternative tags."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="revenue",
                duration_class="quarter",
                value=95.0,
                start_date="2023-01-01",
                end_date="2023-03-31",
                filed_date="2023-05-01",
                concept="Revenues",
                priority=2,
                accession="A",
            ),
            make_fact(
                metric="revenue",
                duration_class="quarter",
                value=100.0,
                start_date="2023-01-01",
                end_date="2023-03-31",
                filed_date="2023-05-01",
                concept="PreferredRevenue",
                priority=1,
                accession="A",
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-05-02",
        ],
        config=make_config(),
    )

    assert snapshots["value"].iloc[0] == 100.0

    assert snapshots["concept_priority"].iloc[0] == 1


def test_duration_classes_remain_separate() -> None:
    """Quarter and YTD values must not overwrite each other."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="revenue",
                duration_class="quarter",
                value=100.0,
                start_date="2023-01-01",
                end_date="2023-03-31",
                filed_date="2023-05-01",
                accession="A",
            ),
            make_fact(
                metric="revenue",
                duration_class="half_year_ytd",
                value=210.0,
                start_date="2023-01-01",
                end_date="2023-06-30",
                filed_date="2023-08-01",
                accession="B",
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-08-02",
        ],
        config=make_config(),
    )

    assert set(snapshots["duration_class"]) == {
        "quarter",
        "half_year_ytd",
    }


def test_statement_type_mismatch_is_removed() -> None:
    """Known context mismatches should not enter PIT data."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="assets",
                duration_class="instant",
                value=300.0,
                end_date="2023-03-31",
                filed_date="2023-05-01",
                statement_type_match=False,
            ),
            make_fact(
                metric="assets",
                duration_class="instant",
                value=301.0,
                end_date="2023-06-30",
                filed_date="2023-08-01",
                accession="B",
                statement_type_match=True,
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-08-02",
        ],
        config=make_config(),
    )

    assert len(snapshots) == 1

    assert snapshots["value"].iloc[0] == 301.0


def test_snapshot_never_uses_future_information() -> None:
    """Every selected fact must be available by as_of_date."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="assets",
                duration_class="instant",
                value=300.0,
                end_date="2023-03-31",
                filed_date="2023-05-01",
            )
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-05-02",
            "2023-06-30",
        ],
        config=make_config(),
    )

    assert (snapshots["available_date"] <= snapshots["as_of_date"]).all()


def test_later_filing_can_replace_higher_priority_old_fact() -> None:
    """Later information should supersede an older concept mapping."""
    data = pd.DataFrame(
        [
            make_fact(
                metric="revenue",
                duration_class="quarter",
                value=100.0,
                start_date="2023-01-01",
                end_date="2023-03-31",
                filed_date="2023-05-01",
                concept="PreferredRevenue",
                priority=1,
                accession="A",
            ),
            make_fact(
                metric="revenue",
                duration_class="quarter",
                value=105.0,
                start_date="2023-01-01",
                end_date="2023-03-31",
                filed_date="2023-05-15",
                concept="AlternativeRevenue",
                priority=2,
                accession="B",
            ),
        ]
    )

    snapshots = build_point_in_time_snapshots(
        data,
        [
            "2023-05-10",
            "2023-05-16",
        ],
        config=make_config(),
    )

    before = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-05-10",
    )

    after = get_fundamentals_as_of(
        snapshots,
        ticker="AAPL",
        as_of_date="2023-05-16",
    )

    assert before["value"].iloc[0] == 100.0

    assert after["value"].iloc[0] == 105.0
