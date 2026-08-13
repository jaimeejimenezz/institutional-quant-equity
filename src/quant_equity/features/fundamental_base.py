"""Monthly point-in-time fundamental input panel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


class FundamentalBaseError(ValueError):
    """Raised when the fundamental base cannot be constructed."""


@dataclass(frozen=True)
class FundamentalBaseConfig:
    """Configuration for the monthly fundamental base."""

    ttm_metrics: tuple[str, ...]
    require_exact_market_date: bool = True

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> FundamentalBaseConfig:
        """Build configuration from existing project settings."""
        metrics = tuple(
            str(metric).strip()
            for metric in values.get(
                "additive_metrics",
                [],
            )
        )

        if not metrics:
            raise FundamentalBaseError("At least one TTM metric is required.")

        return cls(
            ttm_metrics=metrics,
            require_exact_market_date=True,
        )


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    """Require a set of columns."""
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise FundamentalBaseError(f"{dataset_name} is missing columns: {missing}")


def _normalize_tickers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize ticker symbols."""
    result = data.copy()

    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()

    return result


def _normalize_date(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Normalize one date column."""
    result = data.copy()

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
    ).dt.normalize()

    if result[column].isna().any():
        raise FundamentalBaseError(f"Invalid dates found in {column}.")

    return result


def _build_universe_grid(
    universe: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Create one row per active ticker and rebalance date."""
    _require_columns(
        universe,
        (
            "ticker",
            "company_name",
            "sector",
            "industry",
            "cik",
            "start_date",
            "end_date",
        ),
        dataset_name="Universe",
    )

    calendar_column = (
        "as_of_date" if "as_of_date" in rebalance_calendar.columns else "rebalance_date"
    )

    _require_columns(
        rebalance_calendar,
        (calendar_column,),
        dataset_name="Rebalance calendar",
    )

    universe_data = _normalize_tickers(universe)

    universe_data["start_date"] = pd.to_datetime(
        universe_data["start_date"],
        errors="coerce",
    ).dt.normalize()

    universe_data["end_date"] = pd.to_datetime(
        universe_data["end_date"],
        errors="coerce",
    ).dt.normalize()

    if universe_data["start_date"].isna().any():
        raise FundamentalBaseError("Universe contains invalid start dates.")

    dates = pd.DataFrame(
        {
            "as_of_date": (
                pd.to_datetime(
                    rebalance_calendar[calendar_column],
                    errors="coerce",
                )
                .dt.normalize()
                .dropna()
                .drop_duplicates()
                .sort_values()
            )
        }
    )

    if dates.empty:
        raise FundamentalBaseError("Rebalance calendar contains no dates.")

    grid = universe_data.merge(
        dates,
        how="cross",
    )

    active = grid["start_date"].le(grid["as_of_date"]) & (
        grid["end_date"].isna() | grid["end_date"].ge(grid["as_of_date"])
    )

    grid = grid.loc[
        active,
        [
            "as_of_date",
            "ticker",
            "company_name",
            "sector",
            "industry",
            "cik",
        ],
    ].copy()

    duplicates = grid.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalBaseError("Universe grid contains duplicate date-ticker rows.")

    return grid.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)


def _pivot_metric_values(
    data: pd.DataFrame,
    *,
    value_column: str,
    suffix: str,
) -> pd.DataFrame:
    """Pivot canonical metrics into wide columns."""
    if data.empty:
        return pd.DataFrame(
            columns=[
                "as_of_date",
                "ticker",
            ]
        )

    duplicates = data.duplicated(
        [
            "as_of_date",
            "ticker",
            "canonical_metric",
        ]
    )

    if duplicates.any():
        duplicate_count = int(duplicates.sum())

        raise FundamentalBaseError(
            f"Fundamental data contain {duplicate_count} duplicate date-ticker-metric rows."
        )

    wide = data.pivot(
        index=[
            "as_of_date",
            "ticker",
        ],
        columns="canonical_metric",
        values=value_column,
    )

    wide.columns = [f"{column}{suffix}" for column in wide.columns]

    wide.columns.name = None

    return wide.reset_index()


def _pivot_metric_dates(
    data: pd.DataFrame,
    *,
    date_column: str,
    suffix: str,
) -> pd.DataFrame:
    """Pivot metric availability dates."""
    wide = data.pivot(
        index=[
            "as_of_date",
            "ticker",
        ],
        columns="canonical_metric",
        values=date_column,
    )

    wide.columns = [f"{column}{suffix}" for column in wide.columns]

    wide.columns.name = None

    return wide.reset_index()


def _prepare_pit_inputs(
    snapshots: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Prepare balance-sheet point-in-time inputs."""
    _require_columns(
        snapshots,
        (
            "as_of_date",
            "ticker",
            "canonical_metric",
            "value",
            "statement_type",
        ),
        dataset_name="Point-in-time fundamentals",
    )

    data = _normalize_tickers(snapshots)

    data = _normalize_date(
        data,
        "as_of_date",
    )

    data = data.loc[data["statement_type"].eq("instant")].copy()

    if data.empty:
        raise FundamentalBaseError("No instant point-in-time fundamentals were found.")

    if "available_date" in data.columns:
        availability_column = "available_date"
    elif "filed_date" in data.columns:
        availability_column = "filed_date"
    else:
        raise FundamentalBaseError(
            "Point-in-time fundamentals do not contain an availability date."
        )

    data = _normalize_date(
        data,
        availability_column,
    )

    future_information = data[availability_column] > data["as_of_date"]

    if future_information.any():
        raise FundamentalBaseError("Point-in-time inputs contain future information.")

    values = _pivot_metric_values(
        data,
        value_column="value",
        suffix="",
    )

    dates = _pivot_metric_dates(
        data,
        date_column=availability_column,
        suffix="_available_date",
    )

    return values, dates


def _prepare_ttm_inputs(
    snapshots: pd.DataFrame,
    *,
    config: FundamentalBaseConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Prepare trailing-twelve-month inputs."""
    _require_columns(
        snapshots,
        (
            "as_of_date",
            "ticker",
            "canonical_metric",
            "ttm_value",
            "latest_component_available_date",
        ),
        dataset_name="TTM fundamentals",
    )

    data = _normalize_tickers(snapshots)

    data = _normalize_date(
        data,
        "as_of_date",
    )

    data = _normalize_date(
        data,
        "latest_component_available_date",
    )

    data = data.loc[data["canonical_metric"].isin(config.ttm_metrics)].copy()

    available_metrics = set(data["canonical_metric"].unique())

    missing_metrics = set(config.ttm_metrics) - available_metrics

    if missing_metrics:
        raise FundamentalBaseError(
            f"Configured TTM metrics are missing entirely: {sorted(missing_metrics)}"
        )

    future_information = data["latest_component_available_date"] > data["as_of_date"]

    if future_information.any():
        raise FundamentalBaseError("TTM inputs contain future information.")

    values = _pivot_metric_values(
        data,
        value_column="ttm_value",
        suffix="_ttm",
    )

    dates = _pivot_metric_dates(
        data,
        date_column=("latest_component_available_date"),
        suffix="_ttm_available_date",
    )

    return values, dates


def _prepare_market_prices(
    market_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare valid close prices for rebalance dates."""
    _require_columns(
        market_daily,
        (
            "date",
            "ticker",
            "close",
        ),
        dataset_name="Market data",
    )

    data = _normalize_tickers(market_daily)

    data = _normalize_date(
        data,
        "date",
    )

    duplicates = data.duplicated(
        [
            "date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalBaseError("Market data contain duplicate date-ticker rows.")

    data["close"] = pd.to_numeric(
        data["close"],
        errors="coerce",
    )

    valid_close = data["close"].notna() & data["close"].gt(0.0)

    data = data.loc[
        valid_close,
        [
            "date",
            "ticker",
            "close",
        ],
    ].copy()

    if data.empty:
        raise FundamentalBaseError("Market data contain no valid close prices.")

    return data.rename(
        columns={
            "date": "as_of_date",
            "close": "close_price",
        }
    ).reset_index(drop=True)


def build_monthly_fundamental_base(
    *,
    pit_snapshots: pd.DataFrame,
    ttm_snapshots: pd.DataFrame,
    market_daily: pd.DataFrame,
    universe: pd.DataFrame,
    rebalance_calendar: pd.DataFrame,
    config: FundamentalBaseConfig,
) -> pd.DataFrame:
    """Build the monthly point-in-time fundamental input panel."""
    grid = _build_universe_grid(
        universe,
        rebalance_calendar,
    )

    pit_values, pit_dates = _prepare_pit_inputs(pit_snapshots)

    ttm_values, ttm_dates = _prepare_ttm_inputs(
        ttm_snapshots,
        config=config,
    )

    prices = _prepare_market_prices(market_daily)

    result = grid.merge(
        prices,
        on=[
            "as_of_date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    if config.require_exact_market_date and result["close_price"].isna().any():
        missing_rows = int(result["close_price"].isna().sum())

        raise FundamentalBaseError(
            f"{missing_rows} rows do not have an exact close price on the rebalance date."
        )

    for frame in (
        pit_values,
        pit_dates,
        ttm_values,
        ttm_dates,
    ):
        result = result.merge(
            frame,
            on=[
                "as_of_date",
                "ticker",
            ],
            how="left",
            validate="one_to_one",
        )

    metadata_columns = {
        "as_of_date",
        "ticker",
        "company_name",
        "sector",
        "industry",
        "cik",
        "close_price",
    }

    trace_columns = {column for column in result.columns if column.endswith("_available_date")}

    input_columns = [
        column
        for column in result.columns
        if (column not in metadata_columns and column not in trace_columns)
    ]

    result["fundamental_input_count"] = result[input_columns].notna().sum(axis=1)

    result["fundamental_missing_count"] = result[input_columns].isna().sum(axis=1)

    future_violations = 0

    for column in sorted(trace_columns):
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        ).dt.normalize()

        future_violations += int(
            (result[column].notna() & (result[column] > result["as_of_date"])).sum()
        )

    if future_violations:
        raise FundamentalBaseError(
            f"Final fundamental base contains {future_violations} future-information violations."
        )

    duplicates = result.duplicated(
        [
            "as_of_date",
            "ticker",
        ]
    )

    if duplicates.any():
        raise FundamentalBaseError("Final fundamental base contains duplicate date-ticker rows.")

    return result.sort_values(
        [
            "as_of_date",
            "ticker",
        ]
    ).reset_index(drop=True)
