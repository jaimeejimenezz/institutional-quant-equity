"""Load and validate equity universe reference data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from quant_equity.config import PROJECT_ROOT, load_config

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "company_name",
    "sector",
    "industry",
    "cik",
    "start_date",
    "end_date",
    "is_active",
    "inclusion_source",
)

VALID_SECTORS = frozenset(
    {
        "Communication Services",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Financials",
        "Health Care",
        "Industrials",
        "Information Technology",
        "Materials",
        "Real Estate",
        "Utilities",
    }
)


class UniverseValidationError(ValueError):
    """Raised when an equity universe fails validation."""


def get_universe_path(version: str = "v1") -> Path:
    """Return the path associated with a universe version.

    Parameters
    ----------
    version:
        Universe version using a format such as ``v1`` or ``v2``.

    Returns
    -------
    Path
        Absolute path to the universe CSV file.
    """
    normalized_version = version.strip().lower()

    if re.fullmatch(r"v[1-9]\d*", normalized_version) is None:
        raise ValueError(
            "Universe version must use the format 'v1', 'v2', and so on."
        )

    return (
        PROJECT_ROOT
        / "data"
        / "reference"
        / f"universe_{normalized_version}.csv"
    )


def load_universe(
    version: str = "v1",
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load and optionally validate an equity universe.

    Parameters
    ----------
    version:
        Universe version to load.
    validate:
        Whether to execute all universe validation checks.

    Returns
    -------
    pandas.DataFrame
        Normalized universe sorted by ticker.

    Raises
    ------
    FileNotFoundError
        If the requested universe file does not exist.
    UniverseValidationError
        If the universe does not satisfy the validation rules.
    """
    universe_path = get_universe_path(version)

    if not universe_path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {universe_path}"
        )

    string_dtypes = {
        column: "string"
        for column in REQUIRED_COLUMNS
    }

    universe = pd.read_csv(
        universe_path,
        dtype=string_dtypes,
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in universe.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise UniverseValidationError(
            f"Missing required universe columns: {missing}"
        )

    text_columns = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "cik",
        "inclusion_source",
    ]

    for column in text_columns:
        universe[column] = universe[column].str.strip()

    universe["ticker"] = universe["ticker"].str.upper()

    universe["start_date"] = pd.to_datetime(
        universe["start_date"],
        errors="coerce",
    )

    universe["end_date"] = pd.to_datetime(
        universe["end_date"],
        errors="coerce",
    )

    active_values = (
        universe["is_active"]
        .str.strip()
        .str.lower()
    )

    universe["is_active"] = (
        active_values
        .map(
            {
                "true": True,
                "false": False,
            }
        )
        .astype("boolean")
    )

    if validate:
        config = load_config()

        try:
            expected_count = int(
                config["universe"]["expected_count"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise UniverseValidationError(
                "universe.expected_count must be a valid integer."
            ) from error

        validate_universe(
            universe,
            expected_count=expected_count,
        )

    return (
        universe
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def validate_universe(
    universe: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> None:
    """Validate the schema and contents of an equity universe.

    Parameters
    ----------
    universe:
        Universe table to validate.
    expected_count:
        Optional expected number of companies.

    Raises
    ------
    UniverseValidationError
        If one or more validation rules fail.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in universe.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise UniverseValidationError(
            f"Missing required universe columns: {missing}"
        )

    errors: list[str] = []

    if universe.empty:
        errors.append("The universe is empty.")

    if expected_count is not None and len(universe) != expected_count:
        errors.append(
            "Unexpected universe size: "
            f"expected {expected_count}, found {len(universe)}."
        )

    blank_required_columns = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "cik",
        "inclusion_source",
    ]

    for column in blank_required_columns:
        blank_mask = (
            universe[column].isna()
            | universe[column].str.strip().eq("")
        )

        if blank_mask.any():
            errors.append(
                f"Column '{column}' contains missing or blank values."
            )

    duplicate_tickers = (
        universe.loc[
            universe["ticker"].duplicated(keep=False),
            "ticker",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if duplicate_tickers:
        errors.append(
            "Duplicate tickers found: "
            + ", ".join(sorted(duplicate_tickers))
            + "."
        )

    duplicate_ciks = (
        universe.loc[
            universe["cik"].duplicated(keep=False),
            "cik",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if duplicate_ciks:
        errors.append(
            "Duplicate CIK identifiers found: "
            + ", ".join(sorted(duplicate_ciks))
            + "."
        )

    duplicate_companies = (
        universe.loc[
            universe["company_name"].duplicated(keep=False),
            "company_name",
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if duplicate_companies:
        errors.append(
            "Duplicate company names found: "
            + ", ".join(sorted(duplicate_companies))
            + "."
        )

    valid_ticker_mask = (
        universe["ticker"]
        .fillna("")
        .str.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}")
    )

    if not valid_ticker_mask.all():
        invalid_tickers = (
            universe.loc[~valid_ticker_mask, "ticker"]
            .astype("string")
            .tolist()
        )

        errors.append(
            "Invalid ticker format: "
            + ", ".join(invalid_tickers)
            + "."
        )

    valid_cik_mask = (
        universe["cik"]
        .fillna("")
        .str.fullmatch(r"\d{10}")
    )

    if not valid_cik_mask.all():
        invalid_ciks = (
            universe.loc[~valid_cik_mask, "cik"]
            .astype("string")
            .tolist()
        )

        errors.append(
            "CIK identifiers must contain exactly 10 digits: "
            + ", ".join(invalid_ciks)
            + "."
        )

    invalid_sectors = sorted(
        set(universe["sector"].dropna())
        .difference(VALID_SECTORS)
    )

    if invalid_sectors:
        errors.append(
            "Unknown sectors found: "
            + ", ".join(invalid_sectors)
            + "."
        )

    if universe["start_date"].isna().any():
        errors.append(
            "All companies must have a valid start_date."
        )

    invalid_date_order = (
        universe["end_date"].notna()
        & universe["start_date"].notna()
        & (
            universe["end_date"]
            < universe["start_date"]
        )
    )

    if invalid_date_order.any():
        invalid_tickers = (
            universe.loc[invalid_date_order, "ticker"]
            .tolist()
        )

        errors.append(
            "end_date is earlier than start_date for: "
            + ", ".join(invalid_tickers)
            + "."
        )

    if universe["is_active"].isna().any():
        errors.append(
            "is_active must contain only true or false."
        )

    active_with_end_date = (
        universe["is_active"]
        .eq(True)
        .fillna(False)
        & universe["end_date"].notna()
    )

    if active_with_end_date.any():
        invalid_tickers = (
            universe.loc[active_with_end_date, "ticker"]
            .tolist()
        )

        errors.append(
            "Active companies cannot have end_date: "
            + ", ".join(invalid_tickers)
            + "."
        )

    inactive_without_end_date = (
        universe["is_active"]
        .eq(False)
        .fillna(False)
        & universe["end_date"].isna()
    )

    if inactive_without_end_date.any():
        invalid_tickers = (
            universe.loc[
                inactive_without_end_date,
                "ticker",
            ]
            .tolist()
        )

        errors.append(
            "Inactive companies require end_date: "
            + ", ".join(invalid_tickers)
            + "."
        )

    if errors:
        formatted_errors = "\n- ".join(errors)

        raise UniverseValidationError(
            "Universe validation failed:\n"
            f"- {formatted_errors}"
        )