"""Tests for canonical SEC fundamental facts."""

from __future__ import annotations

import pandas as pd
import pytest

from quant_equity.data import (
    ConceptMapping,
    DurationBands,
    SECCanonicalizationError,
    canonicalize_sec_facts,
    classify_duration,
)
from quant_equity.data.sec_fundamental_canonicalization import (
    CanonicalMetricDefinition,
)


def make_mapping() -> ConceptMapping:
    """Create a small canonical mapping."""
    return ConceptMapping(
        accepted_taxonomies=("us-gaap",),
        accepted_forms=(
            "10-K",
            "10-Q",
            "10-K/A",
            "10-Q/A",
        ),
        metrics=(
            CanonicalMetricDefinition(
                name="revenue",
                statement_type="duration",
                units=("USD",),
                concepts=(
                    ("RevenueFromContractWithCustomerExcludingAssessedTax"),
                    "Revenues",
                ),
            ),
            CanonicalMetricDefinition(
                name="assets",
                statement_type="instant",
                units=("USD",),
                concepts=("Assets",),
            ),
        ),
    )


def make_data() -> pd.DataFrame:
    """Create normalized SEC observations."""
    return pd.DataFrame(
        {
            "ticker": [
                "AAPL",
                "AAPL",
                "AAPL",
            ],
            "cik": [
                "0000320193",
                "0000320193",
                "0000320193",
            ],
            "entity_name": [
                "Apple Inc.",
                "Apple Inc.",
                "Apple Inc.",
            ],
            "taxonomy": [
                "us-gaap",
                "us-gaap",
                "us-gaap",
            ],
            "concept": [
                ("RevenueFromContractWithCustomerExcludingAssessedTax"),
                "Revenues",
                "Assets",
            ],
            "unit": [
                "USD",
                "USD",
                "USD",
            ],
            "value": [
                100.0,
                95.0,
                300.0,
            ],
            "start_date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-01"),
                pd.NaT,
            ],
            "end_date": [
                pd.Timestamp("2023-03-31"),
                pd.Timestamp("2023-03-31"),
                pd.Timestamp("2023-03-31"),
            ],
            "filed_date": [
                pd.Timestamp("2023-05-01"),
                pd.Timestamp("2023-05-01"),
                pd.Timestamp("2023-05-01"),
            ],
            "form": [
                "10-Q",
                "10-Q",
                "10-Q",
            ],
            "fiscal_year": [
                2023,
                2023,
                2023,
            ],
            "fiscal_period": [
                "Q1",
                "Q1",
                "Q1",
            ],
            "accession_number": [
                "A",
                "A",
                "A",
            ],
            "frame": [
                None,
                None,
                None,
            ],
        }
    )


def test_quarter_duration_is_classified() -> None:
    """A roughly three-month fact should be a quarter."""
    days, label = classify_duration(
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-03-31"),
        bands=DurationBands(),
    )

    assert days == 90
    assert label == "quarter"


def test_instant_fact_is_classified() -> None:
    """Missing start date should mean instant."""
    days, label = classify_duration(
        pd.NaT,
        pd.Timestamp("2023-03-31"),
        bands=DurationBands(),
    )

    assert days is None
    assert label == "instant"


def test_annual_duration_is_classified() -> None:
    """A full fiscal year should be annual."""
    _, label = classify_duration(
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-12-31"),
        bands=DurationBands(),
    )

    assert label == "annual"


def test_concepts_are_mapped_to_metric() -> None:
    """Alternative XBRL concepts should map to one metric."""
    canonical = canonicalize_sec_facts(
        make_data(),
        mapping=make_mapping(),
        duration_bands=(DurationBands()),
    )

    assert set(canonical["canonical_metric"]) == {
        "revenue",
        "assets",
    }


def test_concept_priority_is_preserved() -> None:
    """Preferred concepts should receive lower priority numbers."""
    canonical = canonicalize_sec_facts(
        make_data(),
        mapping=make_mapping(),
        duration_bands=(DurationBands()),
    )

    revenue = canonical.loc[canonical["canonical_metric"].eq("revenue")]

    priorities = dict(
        zip(
            revenue["concept"],
            revenue["concept_priority"],
            strict=True,
        )
    )

    assert priorities[("RevenueFromContractWithCustomerExcludingAssessedTax")] == 1

    assert priorities["Revenues"] == 2


def test_statement_type_matches_are_detected() -> None:
    """Instant and duration metrics should match their contexts."""
    canonical = canonicalize_sec_facts(
        make_data(),
        mapping=make_mapping(),
        duration_bands=(DurationBands()),
    )

    assert canonical["statement_type_match"].all()


def test_unaccepted_forms_are_excluded() -> None:
    """Forms outside 10-K/10-Q should not enter canonical facts."""
    data = make_data()

    data["form"] = "8-K"

    with pytest.raises(
        SECCanonicalizationError,
        match="No SEC facts matched",
    ):
        canonicalize_sec_facts(
            data,
            mapping=make_mapping(),
            duration_bands=(DurationBands()),
        )
