"""Quality validation and reporting for market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_equity.data.market import MARKET_DATA_COLUMNS

PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
)

CORE_DATA_COLUMNS: tuple[str, ...] = (
    *PRICE_COLUMNS,
    "volume",
)


class MarketDataQualityError(ValueError):
    """Raised when market data cannot be quality checked."""


@dataclass
class MarketDataQualityResult:
    """Complete market-data quality validation result."""

    is_valid: bool
    summary: dict[str, Any]
    ticker_summary: pd.DataFrame
    extreme_returns: pd.DataFrame
    adjustment_anomalies: pd.DataFrame
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_market_data(
    market_data: pd.DataFrame,
    *,
    expected_tickers: Iterable[str],
    start_date: str,
    end_date: str,
    minimum_observations: int,
    minimum_coverage_ratio: float,
    extreme_return_threshold: float,
    maximum_missing_ratio: float,
    adjustment_factor_jump_threshold: float,
) -> MarketDataQualityResult:
    """Validate a canonical long-format market dataset."""
    missing_columns = sorted(
        set(MARKET_DATA_COLUMNS)
        .difference(market_data.columns)
    )

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise MarketDataQualityError(
            f"Missing canonical market columns: {missing}"
        )

    if market_data.empty:
        raise MarketDataQualityError(
            "The market dataset is empty."
        )

    data = market_data.copy()

    data["ticker"] = (
        data["ticker"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    ).dt.normalize()

    expected_ticker_set = {
        ticker.strip().upper()
        for ticker in expected_tickers
    }

    observed_ticker_set = set(
        data["ticker"].dropna()
    )

    missing_tickers = sorted(
        expected_ticker_set
        .difference(observed_ticker_set)
    )

    unexpected_tickers = sorted(
        observed_ticker_set
        .difference(expected_ticker_set)
    )

    configured_start = pd.Timestamp(start_date)
    configured_end = pd.Timestamp(end_date)

    invalid_date_rows = int(
        data["date"].isna().sum()
    )

    duplicate_rows = int(
        data.duplicated(
            subset=["date", "ticker"],
            keep=False,
        ).sum()
    )

    invalid_price_mask = (
        data.loc[:, PRICE_COLUMNS]
        .le(0)
        .any(axis=1)
    )

    invalid_price_rows = int(
        invalid_price_mask.sum()
    )

    negative_volume_rows = int(
        data["volume"].lt(0).sum()
    )

    missing_core_values = int(
        data.loc[:, CORE_DATA_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    missing_denominator = (
        len(data) * len(CORE_DATA_COLUMNS)
    )

    missing_ratio = (
        missing_core_values / missing_denominator
        if missing_denominator
        else 0.0
    )

    out_of_range_mask = (
        data["date"].lt(configured_start)
        | data["date"].ge(configured_end)
    )

    out_of_range_rows = int(
        out_of_range_mask.fillna(False).sum()
    )

    high_reference = data[
        ["open", "low", "close"]
    ].max(
        axis=1,
        skipna=False,
    )

    low_reference = data[
        ["open", "high", "close"]
    ].min(
        axis=1,
        skipna=False,
    )

    ohlc_inconsistency_mask = (
        data["high"].lt(high_reference)
        | data["low"].gt(low_reference)
    )

    ohlc_inconsistency_rows = int(
        ohlc_inconsistency_mask
        .fillna(False)
        .sum()
    )

    valid_dates = (
        data["date"]
        .dropna()
        .drop_duplicates()
    )

    total_available_dates = len(valid_dates)

    ticker_rows: list[dict[str, Any]] = []
    extreme_return_frames: list[pd.DataFrame] = []
    adjustment_anomaly_frames: list[pd.DataFrame] = []

    for ticker in sorted(expected_ticker_set):
        ticker_data = (
            data.loc[
                data["ticker"].eq(ticker)
            ]
            .sort_values("date")
            .copy()
        )

        observations = len(ticker_data)

        coverage_ratio = (
            observations / total_available_dates
            if total_available_dates
            else 0.0
        )

        ticker_missing_values = int(
            ticker_data.loc[:, CORE_DATA_COLUMNS]
            .isna()
            .sum()
            .sum()
        )

        adjusted_returns = (
            ticker_data["adjusted_close"]
            .pct_change(fill_method=None)
        )

        extreme_mask = (
            adjusted_returns
            .abs()
            .gt(extreme_return_threshold)
        )

        if extreme_mask.any():
            extreme_frame = ticker_data.loc[
                extreme_mask,
                [
                    "date",
                    "ticker",
                    "adjusted_close",
                ],
            ].copy()

            extreme_frame["return"] = (
                adjusted_returns.loc[
                    extreme_mask
                ].to_numpy()
            )

            extreme_frames_columns = [
                "date",
                "ticker",
                "adjusted_close",
                "return",
            ]

            extreme_return_frames.append(
                extreme_frame.loc[
                    :,
                    extreme_frames_columns,
                ]
            )

        adjustment_factor = (
            ticker_data["adjusted_close"]
            / ticker_data["close"]
        )

        adjustment_jump = (
            adjustment_factor
            .pct_change(fill_method=None)
        )

        adjustment_mask = (
            adjustment_jump
            .abs()
            .gt(
                adjustment_factor_jump_threshold
            )
        )

        if adjustment_mask.any():
            adjustment_frame = ticker_data.loc[
                adjustment_mask,
                [
                    "date",
                    "ticker",
                    "close",
                    "adjusted_close",
                ],
            ].copy()

            adjustment_frame[
                "adjustment_factor"
            ] = adjustment_factor.loc[
                adjustment_mask
            ].to_numpy()

            adjustment_frame[
                "factor_change"
            ] = adjustment_jump.loc[
                adjustment_mask
            ].to_numpy()

            adjustment_anomaly_frames.append(
                adjustment_frame
            )

        ticker_rows.append(
            {
                "ticker": ticker,
                "observations": observations,
                "first_date": (
                    ticker_data["date"].min()
                    if observations
                    else pd.NaT
                ),
                "last_date": (
                    ticker_data["date"].max()
                    if observations
                    else pd.NaT
                ),
                "coverage_ratio": coverage_ratio,
                "missing_core_values": (
                    ticker_missing_values
                ),
                "extreme_return_count": int(
                    extreme_mask.sum()
                ),
                "adjustment_anomaly_count": int(
                    adjustment_mask.sum()
                ),
                "short_history": (
                    observations
                    < minimum_observations
                ),
                "low_coverage": (
                    coverage_ratio
                    < minimum_coverage_ratio
                ),
            }
        )

    ticker_summary = pd.DataFrame(
        ticker_rows
    )

    if extreme_return_frames:
        extreme_returns = (
            pd.concat(
                extreme_return_frames,
                ignore_index=True,
            )
            .sort_values(
                "return",
                key=lambda values: values.abs(),
                ascending=False,
            )
            .reset_index(drop=True)
        )
    else:
        extreme_returns = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "adjusted_close",
                "return",
            ]
        )

    if adjustment_anomaly_frames:
        adjustment_anomalies = (
            pd.concat(
                adjustment_anomaly_frames,
                ignore_index=True,
            )
            .sort_values(
                "factor_change",
                key=lambda values: values.abs(),
                ascending=False,
            )
            .reset_index(drop=True)
        )
    else:
        adjustment_anomalies = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "close",
                "adjusted_close",
                "adjustment_factor",
                "factor_change",
            ]
        )

    short_history_tickers = (
        ticker_summary.loc[
            ticker_summary["short_history"],
            "ticker",
        ]
        .tolist()
    )

    low_coverage_tickers = (
        ticker_summary.loc[
            ticker_summary["low_coverage"],
            "ticker",
        ]
        .tolist()
    )

    issues: list[str] = []
    warnings: list[str] = []

    if invalid_date_rows:
        issues.append(
            f"{invalid_date_rows} rows contain invalid dates."
        )

    if missing_tickers:
        issues.append(
            "Missing expected tickers: "
            + ", ".join(missing_tickers)
            + "."
        )

    if unexpected_tickers:
        issues.append(
            "Unexpected tickers: "
            + ", ".join(unexpected_tickers)
            + "."
        )

    if duplicate_rows:
        issues.append(
            f"{duplicate_rows} duplicated date-ticker rows found."
        )

    if invalid_price_rows:
        issues.append(
            f"{invalid_price_rows} rows contain non-positive prices."
        )

    if negative_volume_rows:
        issues.append(
            f"{negative_volume_rows} rows contain negative volume."
        )

    if missing_ratio > maximum_missing_ratio:
        issues.append(
            "The missing core-data ratio exceeds the "
            f"configured threshold: {missing_ratio:.6%}."
        )
    elif missing_core_values:
        warnings.append(
            f"{missing_core_values} missing core values were found."
        )

    if out_of_range_rows:
        issues.append(
            f"{out_of_range_rows} rows fall outside the requested range."
        )

    if ohlc_inconsistency_rows:
        issues.append(
            f"{ohlc_inconsistency_rows} rows contain inconsistent OHLC values."
        )

    if short_history_tickers:
        issues.append(
            "Tickers below the minimum observation count: "
            + ", ".join(short_history_tickers)
            + "."
        )

    if low_coverage_tickers:
        issues.append(
            "Tickers below the minimum coverage ratio: "
            + ", ".join(low_coverage_tickers)
            + "."
        )

    if not extreme_returns.empty:
        warnings.append(
            f"{len(extreme_returns)} extreme adjusted returns require review."
        )

    if not adjustment_anomalies.empty:
        warnings.append(
            f"{len(adjustment_anomalies)} adjustment-factor jumps require review."
        )

    summary: dict[str, Any] = {
        "rows": len(data),
        "tickers": data["ticker"].nunique(),
        "unique_dates": total_available_dates,
        "first_date": data["date"].min(),
        "last_date": data["date"].max(),
        "duplicate_rows": duplicate_rows,
        "invalid_date_rows": invalid_date_rows,
        "invalid_price_rows": invalid_price_rows,
        "negative_volume_rows": negative_volume_rows,
        "missing_core_values": missing_core_values,
        "missing_ratio": missing_ratio,
        "out_of_range_rows": out_of_range_rows,
        "ohlc_inconsistency_rows": (
            ohlc_inconsistency_rows
        ),
        "extreme_return_rows": len(
            extreme_returns
        ),
        "adjustment_anomaly_rows": len(
            adjustment_anomalies
        ),
    }

    return MarketDataQualityResult(
        is_valid=not issues,
        summary=summary,
        ticker_summary=ticker_summary,
        extreme_returns=extreme_returns,
        adjustment_anomalies=adjustment_anomalies,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _format_markdown_value(
    value: Any,
) -> str:
    """Format a scalar for a Markdown table."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _dataframe_to_markdown(
    frame: pd.DataFrame,
    *,
    maximum_rows: int | None = None,
) -> str:
    """Convert a small DataFrame to Markdown without extra dependencies."""
    displayed_frame = (
        frame.head(maximum_rows)
        if maximum_rows is not None
        else frame
    )

    if displayed_frame.empty:
        return "_No observations._"

    columns = [
        str(column)
        for column in displayed_frame.columns
    ]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| "
        + " | ".join("---" for _ in columns)
        + " |",
    ]

    for row in displayed_frame.itertuples(
        index=False,
        name=None,
    ):
        formatted_values = [
            _format_markdown_value(value)
            for value in row
        ]

        lines.append(
            "| "
            + " | ".join(formatted_values)
            + " |"
        )

    return "\n".join(lines)


def write_market_data_report(
    result: MarketDataQualityResult,
    path: Path,
    *,
    provider_name: str,
    requested_start_date: str,
    requested_end_date: str,
) -> Path:
    """Write a complete Markdown market-data quality report."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = (
        "PASS"
        if result.is_valid
        else "FAIL"
    )

    summary_frame = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value
            in result.summary.items()
        ]
    )

    lines = [
        "# Market Data Quality Report",
        "",
        f"- Generated UTC: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Status: **{status}**",
        f"- Provider: `{provider_name}`",
        f"- Requested start: `{requested_start_date}`",
        f"- Exclusive end: `{requested_end_date}`",
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(
            summary_frame
        ),
        "",
        "## Validation issues",
        "",
    ]

    if result.issues:
        lines.extend(
            f"- {issue}"
            for issue in result.issues
        )
    else:
        lines.append(
            "- No blocking validation issues."
        )

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )

    if result.warnings:
        lines.extend(
            f"- {warning}"
            for warning in result.warnings
        )
    else:
        lines.append(
            "- No warnings."
        )

    lines.extend(
        [
            "",
            "## Coverage by ticker",
            "",
            _dataframe_to_markdown(
                result.ticker_summary
            ),
            "",
            "## Largest absolute adjusted returns",
            "",
            _dataframe_to_markdown(
                result.extreme_returns,
                maximum_rows=50,
            ),
            "",
            "## Adjustment-factor anomalies",
            "",
            _dataframe_to_markdown(
                result.adjustment_anomalies,
                maximum_rows=50,
            ),
            "",
            "## Interpretation",
            "",
            (
                "Extreme returns and adjustment-factor changes are "
                "reported for manual review. Their presence does not "
                "automatically imply an error because corporate events "
                "and genuine market movements may produce large changes."
            ),
            "",
            (
                "A passing report confirms the configured mechanical "
                "quality checks. It does not certify that the provider "
                "is institutionally complete or point-in-time perfect."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path