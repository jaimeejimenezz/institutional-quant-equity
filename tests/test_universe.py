"""Tests for equity universe loading and validation."""

import pandas as pd
import pytest

from quant_equity.data import (
    REQUIRED_COLUMNS,
    VALID_SECTORS,
    UniverseValidationError,
    load_universe,
    validate_universe,
)


@pytest.fixture
def universe() -> pd.DataFrame:
    """Return the validated version-one universe."""
    return load_universe("v1")


def test_universe_can_be_loaded(
    universe: pd.DataFrame,
) -> None:
    """The configured universe should load successfully."""
    assert len(universe) == 50
    assert tuple(universe.columns) == REQUIRED_COLUMNS


def test_universe_contains_unique_identifiers(
    universe: pd.DataFrame,
) -> None:
    """Tickers, CIKs and company names should be unique."""
    assert universe["ticker"].is_unique
    assert universe["cik"].is_unique
    assert universe["company_name"].is_unique


def test_universe_contains_all_expected_sectors(
    universe: pd.DataFrame,
) -> None:
    """The initial universe should represent all sectors."""
    assert set(universe["sector"]) == VALID_SECTORS


def test_cik_identifiers_have_ten_digits(
    universe: pd.DataFrame,
) -> None:
    """Every CIK should preserve its leading zeroes."""
    valid_ciks = universe["cik"].str.fullmatch(r"\d{10}")

    assert valid_ciks.all()


def test_all_initial_companies_are_active(
    universe: pd.DataFrame,
) -> None:
    """Version one should contain active companies only."""
    assert universe["is_active"].all()
    assert universe["end_date"].isna().all()


def test_membership_dates_are_valid(
    universe: pd.DataFrame,
) -> None:
    """Every row should have a valid membership start date."""
    expected_start_date = pd.Timestamp("2014-01-01")

    assert universe["start_date"].notna().all()
    assert universe["start_date"].eq(expected_start_date).all()


def test_duplicate_ticker_is_rejected(
    universe: pd.DataFrame,
) -> None:
    """Duplicated tickers should fail validation."""
    invalid_universe = universe.copy()
    invalid_universe.loc[1, "ticker"] = invalid_universe.loc[0, "ticker"]

    with pytest.raises(
        UniverseValidationError,
        match="Duplicate tickers",
    ):
        validate_universe(
            invalid_universe,
            expected_count=50,
        )


def test_invalid_cik_is_rejected(
    universe: pd.DataFrame,
) -> None:
    """Malformed CIK values should fail validation."""
    invalid_universe = universe.copy()
    invalid_universe.loc[0, "cik"] = "123"

    with pytest.raises(
        UniverseValidationError,
        match="exactly 10 digits",
    ):
        validate_universe(
            invalid_universe,
            expected_count=50,
        )


def test_incorrect_universe_size_is_rejected(
    universe: pd.DataFrame,
) -> None:
    """A size inconsistent with configuration should fail."""
    with pytest.raises(
        UniverseValidationError,
        match="Unexpected universe size",
    ):
        validate_universe(
            universe,
            expected_count=51,
        )
