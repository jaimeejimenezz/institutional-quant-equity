"""Quality validation and reporting for monthly labels."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_equity.labels.monthly import (
    MONTHLY_LABEL_COLUMNS,
    REBALANCE_CALENDAR_COLUMNS,
)


class MonthlyLabelQualityError(ValueError):
    """Raised when monthly labels cannot be validated."""


@dataclass
class MonthlyLabelQualityResult:
    """Complete monthly-label validation result."""

    is_valid: bool
    summary: dict[str, Any]
    coverage_by_date: pd.DataFrame
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


def _require_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataset_name: str,
) -> None:
    """Require a dataset to contain all expected columns."""
    missing_columns = sorted(set(required_columns).difference(data.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)

        raise MonthlyLabelQualityError(f"{dataset_name} is missing required columns: {missing}")


def validate_monthly_labels(
    rebalance_calendar: pd.DataFrame,
    monthly_labels: pd.DataFrame,
    *,
    expected_tickers: Iterable[str],
    horizon_sessions: int,
    top_quantile_fraction: float,
) -> MonthlyLabelQualityResult:
    """Validate stored monthly labels and temporal alignment."""
    _require_columns(
        rebalance_calendar,
        REBALANCE_CALENDAR_COLUMNS,
        dataset_name=("Rebalance calendar"),
    )

    _require_columns(
        monthly_labels,
        MONTHLY_LABEL_COLUMNS,
        dataset_name="Monthly labels",
    )

    if rebalance_calendar.empty:
        raise MonthlyLabelQualityError("The rebalance calendar is empty.")

    if monthly_labels.empty:
        raise MonthlyLabelQualityError("The monthly labels dataset is empty.")

    if horizon_sessions < 1:
        raise MonthlyLabelQualityError("horizon_sessions must be at least 1.")

    if not 0 < top_quantile_fraction <= 1:
        raise MonthlyLabelQualityError(
            "top_quantile_fraction must be greater than 0 and at most 1."
        )

    calendar = rebalance_calendar.copy()
    labels = monthly_labels.copy()

    for column in (
        "as_of_date",
        "first_future_date",
        "target_end_date",
    ):
        calendar[column] = pd.to_datetime(
            calendar[column],
            errors="coerce",
        ).dt.normalize()

        labels[column] = pd.to_datetime(
            labels[column],
            errors="coerce",
        ).dt.normalize()

    labels["ticker"] = labels["ticker"].astype("string").str.strip().str.upper()

    expected_ticker_set = {str(ticker).strip().upper() for ticker in expected_tickers}

    observed_ticker_set = set(labels["ticker"].dropna())

    issues: list[str] = []
    warnings: list[str] = []

    calendar_invalid_dates = int(calendar["as_of_date"].isna().sum())

    label_invalid_dates = int(
        labels[
            [
                "as_of_date",
                "first_future_date",
                "target_end_date",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    if calendar_invalid_dates:
        issues.append(f"{calendar_invalid_dates} calendar rows contain invalid as-of dates.")

    if label_invalid_dates:
        issues.append(f"{label_invalid_dates} label rows contain invalid dates.")

    calendar_duplicate_rows = int(calendar["as_of_date"].duplicated(keep=False).sum())

    label_duplicate_rows = int(
        labels.duplicated(
            subset=[
                "as_of_date",
                "ticker",
            ],
            keep=False,
        ).sum()
    )

    if calendar_duplicate_rows:
        issues.append(f"{calendar_duplicate_rows} duplicated calendar dates were found.")

    if label_duplicate_rows:
        issues.append(f"{label_duplicate_rows} duplicated date-ticker label rows were found.")

    if not calendar["as_of_date"].is_monotonic_increasing:
        issues.append("The rebalance calendar is not sorted chronologically.")

    calendar_horizon_mismatch = int(
        pd.to_numeric(
            calendar["horizon_sessions"],
            errors="coerce",
        )
        .ne(horizon_sessions)
        .sum()
    )

    label_horizon_mismatch = int(
        pd.to_numeric(
            labels["horizon_sessions"],
            errors="coerce",
        )
        .ne(horizon_sessions)
        .sum()
    )

    if calendar_horizon_mismatch:
        issues.append(f"{calendar_horizon_mismatch} calendar rows use an unexpected horizon.")

    if label_horizon_mismatch:
        issues.append(f"{label_horizon_mismatch} label rows use an unexpected horizon.")

    calendar["has_full_horizon"] = calendar["has_full_horizon"].astype(bool)

    incorrect_horizon_flags = int(
        calendar["has_full_horizon"].ne(calendar["target_end_date"].notna()).sum()
    )

    if incorrect_horizon_flags:
        issues.append(
            f"{incorrect_horizon_flags} calendar rows contain inconsistent horizon flags."
        )

    invalid_temporal_rows = int(
        (
            labels["first_future_date"].le(labels["as_of_date"])
            | labels["target_end_date"].lt(labels["first_future_date"])
        )
        .fillna(True)
        .sum()
    )

    if invalid_temporal_rows:
        issues.append(f"{invalid_temporal_rows} label rows violate temporal ordering.")

    eligible_calendar = calendar.loc[
        calendar["has_full_horizon"],
        [
            "as_of_date",
            "first_future_date",
            "target_end_date",
            "horizon_sessions",
        ],
    ].copy()

    expected_calendar = eligible_calendar.rename(
        columns={
            "first_future_date": ("expected_first_future_date"),
            "target_end_date": ("expected_target_end_date"),
            "horizon_sessions": ("expected_horizon_sessions"),
        }
    )

    temporal_comparison = labels.merge(
        expected_calendar,
        on="as_of_date",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    labels_outside_calendar = int(temporal_comparison["_merge"].ne("both").sum())

    if labels_outside_calendar:
        issues.append(
            f"{labels_outside_calendar} labels do not belong to an eligible calendar date."
        )

    matched_rows = temporal_comparison.loc[temporal_comparison["_merge"].eq("both")]

    incorrect_calendar_alignment = int(
        (
            matched_rows["first_future_date"].ne(matched_rows["expected_first_future_date"])
            | matched_rows["target_end_date"].ne(matched_rows["expected_target_end_date"])
        ).sum()
    )

    if incorrect_calendar_alignment:
        issues.append(
            f"{incorrect_calendar_alignment} labels do not match the calendar future dates."
        )

    eligible_date_set = set(eligible_calendar["as_of_date"].dropna())

    observed_date_set = set(labels["as_of_date"].dropna())

    missing_label_dates = sorted(eligible_date_set.difference(observed_date_set))

    unexpected_label_dates = sorted(observed_date_set.difference(eligible_date_set))

    if missing_label_dates:
        issues.append(f"{len(missing_label_dates)} eligible calendar dates have no labels.")

    if unexpected_label_dates:
        issues.append(f"{len(unexpected_label_dates)} label dates are not eligible calendar dates.")

    missing_required_values = int(
        labels.loc[
            :,
            MONTHLY_LABEL_COLUMNS,
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_required_values:
        issues.append(
            f"{missing_required_values} missing values were found in required label columns."
        )

    invalid_price_rows = int(
        (labels["as_of_adjusted_close"].le(0) | labels["target_end_adjusted_close"].le(0))
        .fillna(True)
        .sum()
    )

    if invalid_price_rows:
        issues.append(f"{invalid_price_rows} label rows contain non-positive prices.")

    numeric_target_columns = [
        "target_21d",
        "cross_sectional_median_21d",
        "target_21d_excess",
        "target_percentile",
    ]

    non_finite_target_rows = int(
        (~np.isfinite(labels[numeric_target_columns].to_numpy(dtype=float))).any(axis=1).sum()
    )

    if non_finite_target_rows:
        issues.append(f"{non_finite_target_rows} label rows contain non-finite target values.")

    expected_target = labels["target_end_adjusted_close"] / labels["as_of_adjusted_close"] - 1.0

    incorrect_target_rows = int(
        (
            ~np.isclose(
                labels["target_21d"],
                expected_target,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ).sum()
    )

    if incorrect_target_rows:
        issues.append(f"{incorrect_target_rows} label rows contain incorrectly calculated returns.")

    expected_median = labels.groupby(
        "as_of_date",
        sort=False,
    )["target_21d"].transform("median")

    incorrect_median_rows = int(
        (
            ~np.isclose(
                labels["cross_sectional_median_21d"],
                expected_median,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ).sum()
    )

    if incorrect_median_rows:
        issues.append(f"{incorrect_median_rows} rows contain an incorrect cross-sectional median.")

    expected_excess = labels["target_21d"] - labels["cross_sectional_median_21d"]

    incorrect_excess_rows = int(
        (
            ~np.isclose(
                labels["target_21d_excess"],
                expected_excess,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ).sum()
    )

    if incorrect_excess_rows:
        issues.append(f"{incorrect_excess_rows} rows contain an incorrect relative target.")

    median_excess_by_date = (
        labels.groupby(
            "as_of_date",
            sort=False,
        )["target_21d_excess"]
        .median()
        .abs()
    )

    maximum_median_excess = float(median_excess_by_date.max())

    if maximum_median_excess > 1e-12:
        issues.append("The relative target is not centered on zero for every date.")

    observed_group_sizes = labels.groupby(
        "as_of_date",
        sort=False,
    )["ticker"].transform("size")

    incorrect_size_rows = int(labels["cross_section_size"].ne(observed_group_sizes).sum())

    if incorrect_size_rows:
        issues.append(f"{incorrect_size_rows} rows contain an incorrect cross-section size.")

    invalid_label_values = int((~labels["label_top_quintile"].isin([0, 1])).sum())

    if invalid_label_values:
        issues.append(f"{invalid_label_values} rows contain invalid top-quintile labels.")

    duplicated_rank_rows = int(
        labels.duplicated(
            subset=[
                "as_of_date",
                "target_rank",
            ],
            keep=False,
        ).sum()
    )

    if duplicated_rank_rows:
        issues.append(f"{duplicated_rank_rows} rows contain duplicated ranks within a date.")

    expected_percentile = (labels["cross_section_size"] - labels["target_rank"] + 1) / labels[
        "cross_section_size"
    ]

    incorrect_percentile_rows = int(
        (
            ~np.isclose(
                labels["target_percentile"],
                expected_percentile,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        ).sum()
    )

    if incorrect_percentile_rows:
        issues.append(f"{incorrect_percentile_rows} rows contain an incorrect target percentile.")

    coverage_by_date = labels.groupby(
        "as_of_date",
        as_index=False,
        sort=True,
    ).agg(
        observations=(
            "ticker",
            "size",
        ),
        unique_tickers=(
            "ticker",
            "nunique",
        ),
        reported_cross_section_size=(
            "cross_section_size",
            "first",
        ),
        selected_top_quintile=(
            "label_top_quintile",
            "sum",
        ),
        minimum_rank=(
            "target_rank",
            "min",
        ),
        maximum_rank=(
            "target_rank",
            "max",
        ),
        median_target=(
            "target_21d",
            "median",
        ),
        median_excess=(
            "target_21d_excess",
            "median",
        ),
    )

    coverage_by_date["expected_top_quintile"] = coverage_by_date["observations"].map(
        lambda size: max(
            1,
            math.ceil(size * top_quantile_fraction),
        )
    )

    coverage_by_date["complete_universe"] = coverage_by_date["unique_tickers"].eq(
        len(expected_ticker_set)
    )

    invalid_rank_dates = int(
        (
            coverage_by_date["minimum_rank"].ne(1)
            | coverage_by_date["maximum_rank"].ne(coverage_by_date["observations"])
        ).sum()
    )

    if invalid_rank_dates:
        issues.append(f"{invalid_rank_dates} dates contain an incomplete ranking sequence.")

    incorrect_top_count_dates = int(
        coverage_by_date["selected_top_quintile"]
        .ne(coverage_by_date["expected_top_quintile"])
        .sum()
    )

    if incorrect_top_count_dates:
        issues.append(f"{incorrect_top_count_dates} dates contain an incorrect top-quintile count.")

    incomplete_universe_dates = int((~coverage_by_date["complete_universe"]).sum())

    if incomplete_universe_dates:
        warnings.append(
            f"{incomplete_universe_dates} dates do not contain the complete expected universe."
        )

    missing_global_tickers = sorted(expected_ticker_set.difference(observed_ticker_set))

    unexpected_tickers = sorted(observed_ticker_set.difference(expected_ticker_set))

    if missing_global_tickers:
        issues.append(
            "Expected tickers absent from the entire "
            "label dataset: " + ", ".join(missing_global_tickers) + "."
        )

    if unexpected_tickers:
        issues.append("Unexpected tickers found: " + ", ".join(unexpected_tickers) + ".")

    summary: dict[str, Any] = {
        "calendar_rows": len(calendar),
        "eligible_calendar_rows": len(eligible_calendar),
        "label_rows": len(labels),
        "label_dates": labels["as_of_date"].nunique(),
        "tickers": labels["ticker"].nunique(),
        "first_as_of_date": labels["as_of_date"].min(),
        "last_as_of_date": labels["as_of_date"].max(),
        "minimum_cross_section_size": int(coverage_by_date["observations"].min()),
        "maximum_cross_section_size": int(coverage_by_date["observations"].max()),
        "duplicate_label_rows": (label_duplicate_rows),
        "invalid_temporal_rows": (invalid_temporal_rows),
        "incorrect_target_rows": (incorrect_target_rows),
        "incorrect_median_rows": (incorrect_median_rows),
        "incorrect_excess_rows": (incorrect_excess_rows),
        "maximum_absolute_median_excess": (maximum_median_excess),
        "incomplete_universe_dates": (incomplete_universe_dates),
    }

    return MonthlyLabelQualityResult(
        is_valid=not issues,
        summary=summary,
        coverage_by_date=coverage_by_date,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _format_markdown_value(
    value: Any,
) -> str:
    """Format a scalar for Markdown output."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, float):
        return f"{value:.12f}"

    return str(value).replace(
        "|",
        "\\|",
    )


def _dataframe_to_markdown(
    frame: pd.DataFrame,
) -> str:
    """Convert a DataFrame to a Markdown table."""
    if frame.empty:
        return "_No observations._"

    columns = [str(column) for column in frame.columns]

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        values = [_format_markdown_value(value) for value in row]

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_monthly_labels_report(
    result: MonthlyLabelQualityResult,
    path: Path,
    *,
    horizon_sessions: int,
    top_quantile_fraction: float,
) -> Path:
    """Write a Markdown quality report for monthly labels."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = "PASS" if result.is_valid else "FAIL"

    summary_frame = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
            }
            for metric, value in result.summary.items()
        ]
    )

    lines = [
        "# Monthly Labels Quality Report",
        "",
        ("- Generated UTC: " + datetime.now(UTC).isoformat(timespec="seconds")),
        f"- Status: **{status}**",
        (f"- Horizon sessions: `{horizon_sessions}`"),
        (f"- Top quantile fraction: `{top_quantile_fraction:.2%}`"),
        "",
        "## Summary",
        "",
        _dataframe_to_markdown(summary_frame),
        "",
        "## Blocking issues",
        "",
    ]

    if result.issues:
        lines.extend(f"- {issue}" for issue in result.issues)
    else:
        lines.append("- No blocking validation issues.")

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )

    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- No warnings.")

    lines.extend(
        [
            "",
            "## Coverage by rebalance date",
            "",
            _dataframe_to_markdown(result.coverage_by_date),
            "",
            "## Temporal interpretation",
            "",
            (
                "Each label uses the adjusted close "
                "observed on the rebalance date and "
                "the adjusted close exactly the "
                "configured number of market sessions "
                "later."
            ),
            "",
            (
                "The first realized return occurs on "
                "the first session strictly after "
                "the rebalance date. Therefore, the "
                "target does not overlap with features "
                "calculated at the rebalance close."
            ),
            "",
            (
                "A passing report confirms mechanical "
                "temporal and cross-sectional "
                "consistency. It does not remove the "
                "documented survivorship bias of the "
                "initial universe."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
